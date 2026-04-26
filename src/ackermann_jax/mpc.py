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
import jax.numpy as jnp
from jax import Array
import numpy as np
import scipy.linalg as sla

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
    Fs: np.ndarray,   # (N, nx, nx) float64
    Gs: np.ndarray,   # (N, nx, nu) float64
) -> Tuple[np.ndarray, np.ndarray]:
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
    Phi = np.zeros((N * nx, nx), dtype=np.float64)
    F_prod = np.eye(nx, dtype=np.float64)
    for k in range(N):
        F_prod = Fs[k] @ F_prod
        Phi[k * nx:(k + 1) * nx] = F_prod

    # Theta  (block-lower-triangular)
    Theta = np.zeros((N * nx, N * nu), dtype=np.float64)
    for j in range(N):
        col = Gs[j].copy()                                  # (nx, nu)
        Theta[j * nx:(j + 1) * nx, j * nu:(j + 1) * nu] = col
        for i in range(j + 1, N):
            col = Fs[i] @ col
            Theta[i * nx:(i + 1) * nx, j * nu:(j + 1) * nu] = col

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

@dataclass
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

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

@dataclass
class MPCResult:
    u_opt:   Array   # (N, 5)    optimal absolute inputs  ū_k + δu*_k
    du_opt:  Array   # (N, 5)    optimal input deviations δu*_k
    x_pred:  Array   # (N+1, 16) predicted error-state trajectory (incl. dx₀)
    cost:    float   # evaluated QP objective
    solved:  bool    # False if OSQP failed/unavailable and fallback was used

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
            else jnp.full((nu),jnp.inf, dtype=jnp.float32),
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
    P_np: np.ndarray,
    q_np: np.ndarray
) -> np.ndarray:
    """
    Analytic unconstrained minimum.
 
    From  J = ½ U'P U + q'U  →  ∂J/∂U = PU + q = 0  →  U* = −P⁻¹q
    where P = 2H (so ½ P = H, and the solve is equivalent to U* = −H⁻¹f).
    """
    return np.linalg.solve(P_np, -q_np)

def _solve_osqp(
    P_np: np.ndarray,
    q_np: np.ndarray,
    A_np: np.ndarray,
    l_np: np.ndarray,
    u_np: np.ndarray,
    warm: Optional[np.ndarray]
) -> Tuple[np.ndarray, bool]:
    """
    Solve the constrained QP with OSQP.
    """
    try:
        import osqp
        import scipy.sparse as sp
    except ImportError:
        return _solve_unconstrained(P_np, q_np), False

    prob = osqp.OSQP()
    prob.setup(
        sp.csc_matrix(P_np),
        q_np,
        sp.csc_matrix(A_np),
        l_np, u_np,
        warm_starting=True,
        verbose=False,
        eps_abs=1e-4,
        eps_rel=1e-4,
        max_iter=2000,
    )
    if warm is not None:
        prob.warm_start(x=warm)
    res = prob.solve()
    ok = res.info.status in ("solved","solved_inaccurate")
    if not ok:
        return _solve_unconstrained(P_np, q_np), False
    return res.x, True

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
    dt = params.dt
    nx = ERROR_DIM
    nu = INPUT_DIM

    # ── 1. Initial error state ──────────────────────────────────────────────
    dx0 = pack_error_state(state_difference(mpc_state.x_ref[0], x_current))

    # ── 2. LTV Jacobians F_k, G_k (JIT-compiled per step) ──────────────────
    x_ref_batch = jax.tree.map(lambda *xs: jnp.stack(xs), *mpc_state.x_ref[:N])
    u_ref_batch = jax.tree.map(lambda *us: jnp.stack(us), *mpc_state.u_ref[:N])

    FG_batched = jax.vmap(lambda x, u: _compute_FG(model, x, u, dt))(x_ref_batch, u_ref_batch)
    Fs_jax, Gs_jax = FG_batched

    # ── 3. Prediction matrices in float64 numpy ─────────────────────────────
    # Using float64 because the stiff modes of this model have |λ| >> 1
    # at dt = 50 ms (contact spring |λ|≈12.3, wheel speeds |λ|≈11.2, yaw
    # rate |λ|≈5.1).  In float32 Phi = F^N overflows completely for N ≥ 10.
    Fs_np  = np.stack([np.array(F, dtype=np.float64) for F in Fs_jax])
    Gs_np  = np.stack([np.array(G, dtype=np.float64) for G in Gs_jax])
    dx0_np = np.array(dx0, dtype=np.float64)

    Phi_np, Theta_np = _build_prediction_matrices_np(Fs_np, Gs_np)

    # ── 4. Cost matrices in float64 ─────────────────────────────────────────
    Q_np  = np.array(params.Q,  dtype=np.float64)
    Pf_np = np.array(params.Pf, dtype=np.float64)
    R_np  = np.array(params.R,  dtype=np.float64)
    Q_bar_np = sla.block_diag(*([Q_np] * (N - 1) + [Pf_np]))  # (N·nx, N·nx)
    R_bar_np = sla.block_diag(*([R_np] * N))                    # (N·nu, N·nu)

    # ── 5. Assemble QP in float64 (OSQP: min ½ U'PU + q'U) ─────────────────
    TtQ   = Theta_np.T @ Q_bar_np                               # (N·nu, N·nx)
    H_np  = TtQ @ Theta_np + R_bar_np                           # (N·nu, N·nu)
    f_np  = TtQ @ (Phi_np @ dx0_np)                             # (N·nu,)
    # Symmetrise + small Tikhonov regularisation for ill-conditioned H
    P_np  = H_np + H_np.T + 1e-8 * np.eye(N * nu, dtype=np.float64)
    q_np  = 2.0 * f_np

    # ── 6. Constraint matrices ───────────────────────────────────────────────
    u_nom_flat_np = np.concatenate(
        [np.array(_pack_control(u), dtype=np.float64) for u in mpc_state.u_ref]
    )                                                            # (N·nu,)
    A_list, l_list, u_list = [], [], []

    if params.u_min is not None or params.u_max is not None:
        A_list.append(np.eye(N * nu, dtype=np.float64))
        lo = np.tile(
            np.array(params.u_min, dtype=np.float64) if params.u_min is not None
            else np.full(nu, -np.inf),
            N,
        )
        hi = np.tile(
            np.array(params.u_max, dtype=np.float64) if params.u_max is not None
            else np.full(nu, np.inf),
            N,
        )
        l_list.append(lo - u_nom_flat_np)
        u_list.append(hi - u_nom_flat_np)

    if params.du_max is not None:
        D = (np.eye(N * nu, dtype=np.float64)
             - np.eye(N * nu, k=-nu, dtype=np.float64))
        A_list.append(D)
        du_np = np.tile(np.array(params.du_max, dtype=np.float64), N)
        l_list.append(-du_np)
        u_list.append(du_np)

    # ── 7. Solve ─────────────────────────────────────────────────────────────
    warm = (np.array(mpc_state.u_warm, dtype=np.float64)
            if mpc_state.u_warm is not None else None)

    if A_list:
        A_np = np.concatenate(A_list, axis=0)
        l_np = np.concatenate(l_list, axis=0)
        u_np = np.concatenate(u_list, axis=0)
        DU_np, solved = _solve_osqp(P_np, q_np, A_np, l_np, u_np, warm)
    else:
        DU_np  = _solve_unconstrained(P_np, q_np)
        solved = True

    # ── 8. Pack outputs ──────────────────────────────────────────────────────
    DU     = jnp.array(DU_np, dtype=jnp.float32)               # (N·nu,)
    du_opt = DU.reshape(N, nu)                                  # (N, 5)
    u_opt  = du_opt + jnp.array(u_nom_flat_np,
                                 dtype=jnp.float32).reshape(N, nu)  # (N, 5)

    X_pred_np = (Phi_np @ dx0_np + Theta_np @ DU_np).reshape(N, nx)
    X_pred    = jnp.array(X_pred_np, dtype=jnp.float32)
    x_pred    = jnp.concatenate([dx0[None], X_pred], axis=0)   # (N+1, nx)

    cost = float(0.5 * DU_np @ P_np @ DU_np + q_np @ DU_np)

    # ── 9. Warm-start: shift left by one block, pad zeros ───────────────────
    warm_next = jnp.concatenate([DU[nu:], jnp.zeros(nu, dtype=jnp.float32)])

    return (
        MPCResult(u_opt=u_opt, du_opt=du_opt, x_pred=x_pred,
                  cost=cost, solved=solved),
        MPCState(x_ref=mpc_state.x_ref, u_ref=mpc_state.u_ref,
                 u_warm=warm_next),
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
        10.0, 10.0,  0.0,   # dp_W       x, y tracked;  z uncontrollable (spring)
         5.0,  5.0,  5.0,   # dθ_B       heading matters
         2.0,  2.0,  0.0,   # dv_W       forward + lateral speed; z excluded
         0.1,  0.1,  0.5,   # dw_B       yaw rate penalised, pitch/roll light
         0.0,  0.0,  0.0,   # dω_W  wheel speeds: stiff + handled by inner loop
         0.0,               # dω_W[3]
    ], dtype=jnp.float32)
    Q  = jnp.diag(q_diag)
    Pf = 5.0 * Q            # terminal: heavier than running; swap for DARE
 
    r_diag = jnp.array(
        [0.5,  0.1, 0.1, 0.1, 0.1],   # delta, tau_w x4
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
    du_max = jnp.array([0.05, 0.1, 0.1, 0.1, 0.1], dtype=jnp.float32)
 
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
 
