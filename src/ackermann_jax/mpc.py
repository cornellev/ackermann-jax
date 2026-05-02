"""
mpc.py — Linear Time-Varying MPC for the Ackermann-JAX model.

Architecture mirrors ekf.py:
  - compute_G   : input Jacobian  (symmetric to compute_F)
  - MPCParams   : static hyperparameters (horizon, weights, constraints)
  - MPCState    : mutable reference trajectory + warm-start across steps
  - MPCResult   : outputs from a single solve
  - mpc_step    : one receding-horizon iteration

The lifted prediction equation is:
    X = Φ dx₀ + Θ U
where X ∈ R^{N·nx} stacks predicted error states, U ∈ R^{N·nu} stacks
input deviations, and Φ/Θ are built from per-step Jacobians F_k / G_k.

The resulting QP (OSQP convention, min ½ U'PU + q'U s.t. l ≤ AU ≤ u)
is solved with OSQP when installed, falling back to an unconstrained
analytic solve (U* = −H⁻¹f) otherwise.

Assumptions (match those discussed in the analytic derivation):
  A2  Small rotation errors: linearization is accurate near the reference.
  A4  Fixed Δt across the horizon.
  A5  Same semi-implicit Euler integrator as car.py.
  A6  LTV: Fₖ and Gₖ recomputed at every step from x_ref.
  A7  Quadratic cost.
  A8  Q ≽ 0, R ≻ 0; Pf ideally from DARE (default: 5·Q).
  A9  QP feasibility assumed; soft constraints not yet implemented.
  A10 Certainty equivalence: EKF covariance P ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import Array
from jaxopt import OSQP

from .car import AckermannCarState, AckermannCarInput, AckermannCarModel
from .ekf import compute_F, ERROR_DIM
from .errorDyn import (
    pack_error_state,
    state_difference,
)

INPUT_DIM = 5  # [delta, tau_w[0..3]]


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _pack_control(u: AckermannCarInput) -> Array:
    """AckermannCarInput → (5,) float32 vector [delta, tau_w]."""
    return jnp.concatenate(
        [jnp.atleast_1d(u.delta),u.tau_w]
    ).astype(jnp.float32)

def _unpack_control(u_flat: Array) -> AckermannCarInput:
    """(5,) float32 vector [delta, tau_w] → AckermannCarInput."""
    return AckermannCarInput(
        delta = u_flat[0].astype(jnp.float32),
        tau_w=u_flat[1:5].astype(jnp.float32)
    )

# Input Jacobian 
def compute_G(
    model: AckermannCarModel,
    x_nom: AckermannCarState,
    u_nom: AckermannCarInput,
    dt: float
) -> Array:
    """
    Input Jacobian  G = ∂(δx_{k+1}) / ∂(δu_k)  evaluated at δu = 0.
 
    Symmetric to compute_F in ekf.py: instead of perturbing the state,
    we perturb the input and measure the first-order shift of the
    *next* error state relative to the nominal next state.
 
    Derivation:
        δx_{k+1} ≈ F_k δx_k + G_k δu_k
        G_k = ∂/∂(δu) [ state_diff( step(x_nom, ū+δu, dt),
                                     step(x_nom, ū,    dt) ) ] |_{δu=0}
 
    Since x_nom is used for both calls (δx = 0), x_next_nom can be
    precomputed outside the autodiff.
 
    Shape: (ERROR_DIM, INPUT_DIM) = (16, 5)
    """
    u_flat = _pack_control(u_nom)
    x_next_nom = model.step(x_nom, u_nom, dt)

    def _response(du: Array) -> Array:
        u_pert = _unpack_control(u_flat + du)
        x_next_pert = model.step(x_nom, u_pert, dt)
        # state_difference(ref, x) = x − ref; ref=nom gives δx_{k+1} = pert − nom
        return pack_error_state(state_difference(x_next_nom, x_next_pert))

    return jax.jacobian(_response)(jnp.zeros(INPUT_DIM,dtype=jnp.float32))


@jax.jit
def _compute_FG(
    model: AckermannCarModel,
    x_nom: AckermannCarState,
    u_nom: AckermannCarInput,
    dt: float,
) -> Tuple[Array, Array]:
    """
    JIT-compiled (F, G) pair.

    compute_F and compute_G call jax.jacobian internally. Without a JIT
    boundary, JAX re-traces the Jacobian function on every Python call because
    the closed-over (x_nom, u_nom) values change each step.  Wrapping both
    in a single @jax.jit compiles the full Jacobian computation once (for the
    abstract pytree shapes) and dispatches XLA on every subsequent call —
    same pattern as ekf_predict which JITs over compute_F.

    Shape: F (ERROR_DIM, ERROR_DIM), G (ERROR_DIM, INPUT_DIM)
    """
    return compute_F(model, x_nom, u_nom, dt), compute_G(model, x_nom, u_nom, dt)


def _build_prediction_matrices_np(
    Fs: jnp.ndarray,   # (N, nx, nx) float64
    Gs: jnp.ndarray,   # (N, nx, nu) float64
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build Φ and Θ entirely in float64 numpy.

    This is the workhorse used inside mpc_step.  The JAX/scan version
    (build_prediction_matrices) is kept for API compatibility and testing,
    but should not be used when stiff eigenvalues are present (|λ| >> 1
    at dt ≥ 20 ms for this model's contact spring and wheel dynamics).

    Recurrence:
        Φ[k]   = F_k · F_{k−1} · … · F_0
        Θ[j,j] = G_j
        Θ[i+1,j] = F_{i+1} · Θ[i,j]       (propagate forward one step)

    Returns:
        Phi:   (N·nx, nx)   float64
        Theta: (N·nx, N·nu) float64
    """
    N, nx = Fs.shape[:2]
    nu = Gs.shape[2]

    # Phi
    Phi = jnp.zeros((N * nx, nx), dtype=jnp.float64)
    F_prod = jnp.eye(nx, dtype=jnp.float64)
    for k in range(N):
        F_prod = Fs[k] @ F_prod
        Phi = Phi.at[k * nx:(k + 1) * nx].set(F_prod)

    # Theta  (block-lower-triangular)
    Theta = jnp.zeros((N * nx, N * nu), dtype=jnp.float64)
    for j in range(N):
        col = Gs[j].copy()                                  # (nx, nu)
        Theta = Theta.at[j * nx:(j + 1) * nx, j * nu:(j + 1) * nu].set(col)
        for i in range(j + 1, N):
            col = Fs[i] @ col
            Theta = Theta.at[i * nx:(i + 1) * nx, j * nu:(j + 1) * nu].set(col)

    return Phi, Theta


def build_prediction_matrices(
    Fs: List[Array] = [],
    Gs: List[Array] = [],
) -> tuple[Array, Array]:
    """
    Assemble the N-step lifted matrices Φ ∈ R^{N·nx × nx} and
    Θ ∈ R^{N·nx × N·nu} from per-step Jacobians.
 
    Prediction equation:
        ⎡δx_1⎤         ⎡δu_0⎤
        ⎢δx_2⎥ = Φ δx₀ + Θ ⎢δu_1⎥
        ⎣ ⋮  ⎦         ⎣ ⋮  ⎦
 
    Φ block structure (row k = δx_{k+1}):
        Φ[k] = F_k · F_{k-1} · … · F_0
 
    Θ block structure (block-lower-triangular):
        Θ[i,j] = F_i · F_{i-1} · … · F_{j+1} · G_j   (i ≥ j)
        Θ[i,j] = 0                                       (i < j)
 
    Recurrence used:
        Θ[j, j]   = G_j
        Θ[i+1, j] = F_{i+1} · Θ[i, j]      (propagate forward one step)
 
    Args:
        Fs: [F_0, …, F_{N-1}]  state-transition Jacobians  (N, nx, nx) (stack before calling)
        Gs: [G_0, …, G_{N-1}]  input Jacobians             (N, nx, nu) (stack before calling)
 
    Returns:
        Phi:   (N·nx, nx)
        Theta: (N·nx, N·nu)
    """
    N, nx, nu = Fs.shape[0], Fs.shape[1], Gs.shape[2]

    # Phi via scan - carry is running product, output is each block
    def phi_step(F_prod, Fk):
        F_prod_next = Fk @ F_prod
        return F_prod_next, F_prod_next

    _, Phi = jax.lax.scan(phi_step, jnp.eye(nx, dtype=jnp.float32),Fs)
    Phi = Phi.reshape(N * nx, nx)

    # ── Theta via scan: carry is (N, nx, nu) array of active column vectors
    #    At step k: write G_k into slot k, propagate all earlier slots by F_k
    def theta_step(cols, args):
        Fk, Gk, k = args
        cols_next = jax.vmap(lambda c: Fk @ c)(cols)
        cols_next = cols_next.at[k].set(Gk)  # Insert G_k into slot k
        return cols_next, cols_next

    cols0 = jnp.zeros((N, nx, nu), dtype=jnp.float32)
    ks = jnp.arange(N)
    _, col_history = jax.lax.scan(theta_step, cols0, (Fs, Gs, ks))
    # col_history[k, j] = Theta column j at row k  → need lower-triangular mask
    # col_history shape: (N, N, nx, nu); entry [i, j] is valid only when j <= i

    # Reshape into the (N*nx, N*nu) matrix
    # Transpose axes: we want [step i, column j, nx, nu] -> [i*nx, j*nu]
    Theta = col_history.transpose(0,2,1,3) # (N, nx, N, nu)
    Theta = Theta.reshape(N* nx, N * nu)

    return Phi, Theta

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MPCParams:
    """
    Static hyperparameters for the MPC problem.
 
    State error layout (ERROR_DIM = 16):
        [0:3]   dp_W      position error              [m]
        [3:6]   dθ_B      rotation error (so3 vec)    [rad]
        [6:9]   dv_W      velocity error              [m/s]
        [9:12]  dw_B      angular velocity error      [rad/s]
        [12:16] dω_W      wheel speed error           [rad/s]
 
    Input layout (INPUT_DIM = 5):
        [0]     δ         steering angle              [rad]
        [1:5]   τ_w       wheel torques               [N·m]
    """
    N:      int     # prediction horizon (number of steps)
    dt:     float   # time step [s]
    Q:      Array   # (16, 16)  running state-error weight     Q  ≽ 0
    R:      Array   # (5,  5)   input-deviation weight         R  ≻ 0
    Pf:     Array   # (16, 16)  terminal state weight          Pf ≽ 0
    u_min:  Optional[Array] = None   # (5,) absolute input lower bound
    u_max:  Optional[Array] = None   # (5,) absolute input upper bound
    du_max: Optional[Array] = None   # (5,) per-step slew-rate limit

    def tree_flatten(self):
        children = [jnp.asarray(self.dt), self.Q, self.R, self.Pf]
        has_u_min = self.u_min is not None
        has_u_max = self.u_max is not None
        has_du_max = self.du_max is not None
        if has_u_min:
            children.append(self.u_min)
        if has_u_max:
            children.append(self.u_max)
        if has_du_max:
            children.append(self.du_max)
        aux = (self.N, has_u_min, has_u_max, has_du_max)
        return tuple(children), aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        N, has_u_min, has_u_max, has_du_max = aux
        idx = 0
        dt = children[idx]
        idx += 1
        Q = children[idx]
        idx += 1
        R = children[idx]
        idx += 1
        Pf = children[idx]
        idx += 1
        u_min = children[idx] if has_u_min else None
        idx += int(has_u_min)
        u_max = children[idx] if has_u_max else None
        idx += int(has_u_max)
        du_max = children[idx] if has_du_max else None
        return cls(N=N, dt=dt, Q=Q, R=R, Pf=Pf, u_min=u_min, u_max=u_max, du_max=du_max)

# Mutable MPC state for receeding-horizon iterations
@dataclass
class MPCState:
    """
    Carries the reference trajectory and the previous QP solution.
 
    x_ref / u_ref define the nominal trajectory the MPC tracks.
    The caller must update them between steps (e.g. from a path planner
    or by rolling the previous solution forward by one step).
 
    u_warm is shifted automatically from the previous DU* after each solve.
    """
    x_ref:  List[AckermannCarState]   # length N+1  — nominal states
    u_ref:  List[AckermannCarInput]   # length N    — nominal inputs
    u_warm: Optional[Array] = None    # (N·INPUT_DIM,) QP warm-start
    solver: Optional[OSQP] = None

def init_mpc_state(x_ref, u_ref, params: MPCParams) -> MPCState:
    N = params.N
    nu = INPUT_DIM
    n_constr = 0
    if params.u_min is not None or params.u_max is not None:
        n_constr += N * nu
    if params.du_max is not None:
        n_constr += N * nu
    solver = create_osqp_solver()
    return MPCState(
        x_ref=x_ref,
        u_ref=u_ref,
        u_warm=None,
        solver=solver,
    )      

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MPCResult:
    u_opt:   Array   # (N, 5)    optimal absolute inputs  ū_k + δu*_k
    du_opt:  Array   # (N, 5)    optimal input deviations δu*_k
    x_pred:  Array   # (N+1, 16) predicted error-state trajectory (incl. dx₀)
    cost:    Array   # evaluated QP objective
    solved:  Array   # False if OSQP failed/unavailable and fallback was used

    def tree_flatten(self):
        return (self.u_opt, self.du_opt, self.x_pred, self.cost, self.solved), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

# Assemble the QP
def _build_qp(
    Phi: Array,
    Theta: Array,
    dx0: Array,
    Q_bar: Array,
    R_bar: Array,
    u_nom_flat: Array,
    params: MPCParams
) -> Tuple[Array, Array, Optional[Array], Optional[Array], Optional[Array]]:
    """
    Assemble QP matrices in OSQP convention:
        min (1/2) U' P U + q' U s.t. l <= A U <= u_c

    Derivation:
        J  = (Φ dx₀ + Θ U)' Q̄ (Φ dx₀ + Θ U) + U' R̄ U
           = U' (Θ'Q̄Θ + R̄) U + 2(Θ'Q̄Φ dx₀)' U + const
           =: U' H U + 2 f' U

    OSQP maps: P = H + H' = 2H (symmetric), q = 2f

    Constraint blocks (each obtional, stacked if active)
        Box: A = I, l = u_min - u_bar, u = u_max - u_bar
        Rate: A = D (difference matrix), l/u = +- du_max

    Returns (P, q, A, l, u_c) - last three are None if no constraints are active.
    """
    N = params.N
    nu = INPUT_DIM

    TtQ = Theta.T @ Q_bar
    H = TtQ @ Theta + R_bar
    f = TtQ @ (Phi @ dx0)

    P_qp = H + H.T
    q_qp = 2.0 * f

    A_list, l_list, u_list = [], [], []

    # Box constraints on absolute input: u_min <= u_bar(k) + delta u_k <= u_max
    if params.u_min is not None or params.u_max is not None:
        A_list.append(jnp.eye(N * nu, dtype=jnp.float32))
        lo = jnp.tile(
            params.u_min if params.u_min is not None
            else jnp.full((nu,), -jnp.inf, dtype=jnp.float32),
            N,
        )
        hi = jnp.tile(
            params.u_max if params.u_max is not None
            else jnp.full((nu,),jnp.inf, dtype=jnp.float32),
            N,
        )
        l_list.append(lo - u_nom_flat)
        u_list.append(hi - u_nom_flat)

    # Slew-rate constraints:  |δu_k − δu_{k−1}| ≤ du_max,  δu_{−1} = 0
    # Difference matrix D ∈ R^{N·nu × N·nu}:
    #   D[k·nu:(k+1)·nu, :] extracts δu_k − δu_{k-1}  (with δu_{-1} = 0)
    if params.du_max is not None:
        D = (jnp.eye(N * nu, dtype=jnp.float32) - jnp.eye(N * nu, k=-nu, dtype=jnp.float32))
        A_list.append(D)
        du_tiled = jnp.tile(params.du_max, N)
        l_list.append(-du_tiled)
        u_list.append(du_tiled)

    if A_list:
        return(
            P_qp, q_qp,
            jnp.concatenate(A_list, axis=0),
            jnp.concatenate(l_list, axis=0),
            jnp.concatenate(u_list, axis=0)
        )
    return P_qp, q_qp, None, None, None

# Solvers
def _solve_unconstrained(
    P_np: jnp.ndarray,
    q_np: jnp.ndarray
) -> jnp.ndarray:
    """
    Analytic unconstrained minimum.
 
    From  J = ½ U'P U + q'U  →  ∂J/∂U = PU + q = 0  →  U* = −P⁻¹q
    where P = 2H (so ½ P = H, and the solve is equivalent to U* = −H⁻¹f).
    """
    return jnp.linalg.solve(P_np, -q_np)

# def _solve_osqp(
#     P_np: jnp.ndarray,
#     q_np: jnp.ndarray,
#     A_np: jnp.ndarray,
#     l_np: jnp.ndarray,
#     u_np: jnp.ndarray,
#     warm: Optional[jnp.ndarray]
# ) -> Tuple[jnp.ndarray, bool]:
#     """
#     Solve the constrained QP with OSQP.
#     """
#     try:
#         import osqp
#         import scipy.sparse as sp
#     except ImportError:
#         return _solve_unconstrained(P_np, q_np), False
    
#     P_np = np.asarray(P_np, dtype=np.float64)
#     q_np = np.asarray(q_np, dtype=np.float64)
#     A_np = np.asarray(A_np, dtype=np.float64)
#     l_np = np.asarray(l_np, dtype=np.float64)
#     u_np = np.asarray(u_np, dtype=np.float64)

#     prob = osqp.OSQP()
#     prob.setup(
#         sp.csc_matrix(P_np),
#         q_np,
#         sp.csc_matrix(A_np),
#         l_np, u_np,
#         warm_starting=True,
#         verbose=False,
#         eps_abs=1e-4,
#         eps_rel=1e-4,
#         max_iter=2000,
#     )
#     if warm is not None:
#         prob.warm_start(x=warm)
#     res = prob.solve()
#     ok = res.info.status in ("solved","solved_inaccurate")
#     if not ok:
#         return _solve_unconstrained(P_np, q_np), False
#     return res.x, True

def create_osqp_solver():
    return OSQP(
        maxiter=2000,
        tol=1e-4,
        verbose=False,
    )

def block_diag_jax(mats):
    sizes = [m.shape[0] for m in mats]
    total = sum(sizes)
    out = jnp.zeros((total, total), dtype=mats[0].dtype)

    i = 0
    for m in mats:
        s = m.shape[0]
        out = out.at[i:i+s, i:i+s].set(m)
        i += s
    return out


def _pack_control_batch(u_ref: AckermannCarInput) -> Array:
    """Batched AckermannCarInput -> (..., 5)."""
    return jnp.concatenate([u_ref.delta[..., None], u_ref.tau_w], axis=-1)


@jax.jit
def mpc_step_batched(
    model: AckermannCarModel,
    x_ref: AckermannCarState,
    u_ref: AckermannCarInput,
    params: MPCParams,
    x_current: AckermannCarState,
    u_warm: Optional[Array] = None,
) -> Tuple[MPCResult, Array]:
    """
    JIT-compatible MPC step for pre-stacked reference windows.

    x_ref is a pytree with leading dimension N+1, and u_ref is a pytree
    with leading dimension N.  This avoids Python list handling in the
    performance-critical path and can be called from lax.scan.
    """
    del u_warm
    N = params.N
    dt = params.dt
    nx = ERROR_DIM
    nu = INPUT_DIM

    dx0 = pack_error_state(state_difference(jax.tree.map(lambda a: a[0], x_ref), x_current))
    x_ref_linearization = jax.tree.map(lambda a: a[:N], x_ref)

    Fs_jax, Gs_jax = jax.vmap(lambda x, u: _compute_FG(model, x, u, dt))(
        x_ref_linearization, u_ref
    )

    Fs = Fs_jax.astype(jnp.float64)
    Gs = Gs_jax.astype(jnp.float64)
    dx0_64 = dx0.astype(jnp.float64)

    Phi, Theta = _build_prediction_matrices_np(Fs, Gs)

    Q = params.Q.astype(jnp.float64)
    Pf = params.Pf.astype(jnp.float64)
    R = params.R.astype(jnp.float64)
    Q_bar = block_diag_jax([Q] * (N - 1) + [Pf])
    R_bar = block_diag_jax([R] * N)

    TtQ = Theta.T @ Q_bar
    H = TtQ @ Theta + R_bar
    f = TtQ @ (Phi @ dx0_64)
    P = H + H.T + 1e-8 * jnp.eye(N * nu, dtype=jnp.float64)
    q = 2.0 * f

    u_nom_flat = _pack_control_batch(u_ref).reshape(N * nu).astype(jnp.float64)
    A_list, l_list, u_list = [], [], []

    if params.u_min is not None or params.u_max is not None:
        A_list.append(jnp.eye(N * nu, dtype=jnp.float64))
        lo = jnp.tile(
            params.u_min.astype(jnp.float64)
            if params.u_min is not None
            else jnp.full((nu,), -jnp.inf, dtype=jnp.float64),
            N,
        )
        hi = jnp.tile(
            params.u_max.astype(jnp.float64)
            if params.u_max is not None
            else jnp.full((nu,), jnp.inf, dtype=jnp.float64),
            N,
        )
        l_list.append(lo - u_nom_flat)
        u_list.append(hi - u_nom_flat)

    if params.du_max is not None:
        D = jnp.eye(N * nu, dtype=jnp.float64) - jnp.eye(N * nu, k=-nu, dtype=jnp.float64)
        du = jnp.tile(params.du_max.astype(jnp.float64), N)
        A_list.append(D)
        l_list.append(-du)
        u_list.append(du)

    if A_list:
        A = jnp.concatenate(A_list, axis=0)
        lower = jnp.concatenate(l_list, axis=0)
        upper = jnp.concatenate(u_list, axis=0)
        G = jnp.concatenate([A, -A], axis=0)
        h = jnp.concatenate([upper, -lower], axis=0)
        params_ineq = (G, h)
    else:
        params_ineq = None

    solver = create_osqp_solver()
    sol = solver.run(
        params_obj=(P, q),
        params_ineq=params_ineq,
        init_params=None,
    )

    DU_64 = sol.params.primal
    DU = DU_64.astype(jnp.float32)
    du_opt = DU.reshape(N, nu)
    u_opt = du_opt + u_nom_flat.astype(jnp.float32).reshape(N, nu)

    X_pred = (Phi @ dx0_64 + Theta @ DU_64).reshape(N, nx).astype(jnp.float32)
    x_pred = jnp.concatenate([dx0.astype(jnp.float32)[None, :], X_pred], axis=0)
    cost = (0.5 * DU_64 @ P @ DU_64 + q @ DU_64).astype(jnp.float32)
    solved = sol.state.status == 1
    warm_next = jnp.concatenate([DU[nu:], jnp.zeros(nu, dtype=jnp.float32)])

    return MPCResult(u_opt=u_opt, du_opt=du_opt, x_pred=x_pred, cost=cost, solved=solved), warm_next

def mpc_step(
    model: AckermannCarModel,
    mpc_state: MPCState,
    params: MPCParams,
    x_current: AckermannCarState
) -> Tuple[MPCResult, MPCState]:
    """
    One receding-horizon MPC iteration.
 
    Algorithm:
        1.  dx₀  = x_current ⊖ x_ref[0]        (error state from EKF)
        2.  Fₖ, Gₖ  for k = 0…N-1              (LTV linearization)
        3.  Build Φ, Θ                           (lifted prediction)
        4.  Build Q̄, R̄                          (block-diagonal cost matrices)
        5.  Assemble and solve QP
        6.  Extract u*₀ = ū₀ + δu*₀             (first input only applied)
        7.  Shift warm-start for next call
 
    Args:
        model:      vehicle model (same instance shared with the EKF)
        mpc_state:  current reference trajectory and warm-start
        params:     static MPC hyperparameters
        x_current:  current state estimate (pass ekf_state.x_nom directly)
 
    Returns:
        result:           MPCResult — apply result.u_opt[0] to the car
        mpc_state_next:   MPCState  — pass back in on the next call
                          (update x_ref / u_ref from your planner first)
    """
    N  = params.N
    nu = INPUT_DIM

    x_ref_batch = jax.tree.map(lambda *xs: jnp.stack(xs), *mpc_state.x_ref[: N + 1])
    u_ref_batch = jax.tree.map(lambda *us: jnp.stack(us), *mpc_state.u_ref[:N])
    warm = (
        jnp.asarray(mpc_state.u_warm, dtype=jnp.float32)
        if mpc_state.u_warm is not None
        else jnp.zeros(N * nu, dtype=jnp.float32)
    )
    result, warm_next = mpc_step_batched(model, x_ref_batch, u_ref_batch, params, x_current, warm)

    return (
        result,
        MPCState(x_ref=mpc_state.x_ref, u_ref=mpc_state.u_ref,
                 u_warm=warm_next, solver=mpc_state.solver),
    )
 
 
# ---------------------------------------------------------------------------
# Default parameters (matches default_params() in car.py)
# ---------------------------------------------------------------------------
 
def default_mpc_params(N: int = 20, dt: float = 0.05) -> MPCParams:
    """
    Reasonable starting weights for the 1.5 kg Ackermann car.
 
    Tuning philosophy:
      - Position (x, y) and heading are the primary tracking objectives.
      - Velocity and angular rate errors are penalised lightly.
      - Wheel speed deviations are essentially free (inner loop handles them).
      - Steering rate (du_max[0]) is tight to prevent rapid oscillation.
      - Torque rate is looser since the PI velocity controller is slow.
      - Pf = 5·Q as a conservative (not DARE-optimal) terminal cost.
        Replace with the DARE solution for a formal stability guarantee.
    """
    # NOTE: dp_z (index 2), dv_z (index 8), and dω_W (indices 12-15) are
    # zeroed out.  These modes have |F eigenvalue| >> 1 at dt=50 ms (contact
    # spring ≈12.3, wheel speeds ≈11.2), so their rows of Phi/Theta grow
    # unboundedly with horizon length.  Zeroing Q for these DOFs removes them
    # from the cost while keeping the float64 products numerically bounded.
    # The contact spring is uncontrollable on this time-scale anyway, and wheel
    # speeds are regulated by the torque inner loop.
    q_diag = jnp.array([
        25.0, 35.0,  0.0,   # dp_W       x, y tracked;  z uncontrollable (spring)
         1.0,  1.0,  20.0,   # dθ_B       heading matters
         8.0,  8.0,  8.0,   # dv_W       forward + lateral speed; z excluded
         0.1,  0.1,  5.0,   # dw_B       yaw rate penalised, pitch/roll light
         0.0,  0.0,  0.0,   # dω_W  wheel speeds: stiff + handled by inner loop
         0.0,               # dω_W[3]
    ], dtype=jnp.float32)
    Q  = jnp.diag(q_diag)
    Pf = 10.0 * Q            # terminal: heavier than running; swap for DARE
 
    r_diag = jnp.array(
        [1.5,  0.03, 0.03, 0.03, 0.03],   # delta, tau_w x4
        dtype=jnp.float32,
    )
    R = jnp.diag(r_diag)
 
    # Physical limits from car.py:
    #   delta_max = 0.35 rad  (map_heading_to_steering)
    #   tau_max context-dependent; 0.5 N·m is conservative for this scale
    u_min = jnp.array([-0.35, -0.5, -0.5, -0.5, -0.5], dtype=jnp.float32)
    u_max = jnp.array([ 0.35,  0.5,  0.5,  0.5,  0.5], dtype=jnp.float32)
 
    # Per-step slew limits:
    #   steering: 0.05 rad/step = 1 rad/s at 20 Hz
    #   torque:   0.1 N·m/step  = 2 N·m/s at 20 Hz
    du_max = jnp.array([0.04, 0.2, 0.2, 0.2, 0.2], dtype=jnp.float32)
 
    return MPCParams(
        N=N,
        dt=dt,
        Q=Q,
        R=R,
        Pf=Pf,
        u_min=u_min,
        u_max=u_max,
        du_max=du_max,
    )
 
