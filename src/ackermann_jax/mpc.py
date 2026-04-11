"""Linear MPC for the Ackermann car using JAX autodiff Jacobians.

This module provides a **linear time-varying MPC** (LTV-MPC) controller that
reuses the same autodiff infrastructure already used by the error-state EKF.
At each control step the nonlinear dynamics are linearised around the current
nominal trajectory, a constrained QP is assembled, and OSQP is called via
``jaxopt.BoxOSQP`` (pure-JAX, JIT-compilable).

Typical usage::

    from ackermann_jax.mpc import (
        MPCConfig, mpc_init, mpc_step, default_mpc_config,
    )

    cfg   = default_mpc_config()
    state = mpc_init(model, x0, u0, cfg)
    u_opt, state = mpc_step(model, x_current, state, cfg)

Design decisions
----------------
* **Error-state formulation** – exactly the same 16-dim error state and
  SO(3) right-perturbation convention as the EKF, so both modules share
  :mod:`ackermann_jax.errorDyn` without modification.
* **Input vector** – 5-dim: ``[delta, tau_FL, tau_FR, tau_RL, tau_RR]``.
* **QP solver** – ``jaxopt.BoxOSQP`` for a pure-JAX, JIT-able solve.
  Swap for OSQP-python or HPIPM if you need hard real-time guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array

from .car import AckermannCarState, AckermannCarInput, AckermannCarModel
from .errorDyn import (
    AckermannCarErrorState,
    error_dynamics,
    inject_error,
    state_difference,
    pack_error_state,
    unpack_error_state,
    zero_error_state,
)
from .ekf import compute_F, ERROR_DIM

# ── Dimensions ───────────────────────────────────────────────────────────────

INPUT_DIM = 5  # [delta, tau_FL, tau_FR, tau_RL, tau_RR]


# ── Helpers: pack / unpack inputs ────────────────────────────────────────────

def pack_input(u: AckermannCarInput) -> Array:
    """Flatten an AckermannCarInput to a (5,) vector."""
    return jnp.concatenate([jnp.atleast_1d(u.delta), u.tau_w])


def unpack_input(u_vec: Array) -> AckermannCarInput:
    """Reconstruct an AckermannCarInput from a (5,) vector."""
    return AckermannCarInput(delta=u_vec[0], tau_w=u_vec[1:])


# ── Control Jacobian B ───────────────────────────────────────────────────────


def compute_B(
    model: AckermannCarModel,
    x_nom: AckermannCarState,
    u: AckermannCarInput,
    dt: float,
) -> Array:
    r"""Compute the discrete control Jacobian **B**.

    Linearises the error dynamics with respect to the input vector
    :math:`u = [\delta,\, \tau_1,\, \tau_2,\, \tau_3,\, \tau_4]` at
    :math:`\delta x = 0` via :func:`jax.jacobian`:

    .. math::

        B = \left.\frac{\partial f(\mathbf{0},\, u)}
                       {\partial u}\right|_{u = u_{\mathrm{nom}}}

    Args:
        model: Car physics model.
        x_nom: Nominal state (linearisation point).
        u: Nominal control input (linearisation point).
        dt: Timestep [s].

    Returns:
        Control Jacobian of shape :math:`16 \times 5`.
    """
    u_nom_vec = pack_input(u)

    def f_u(u_flat: Array) -> Array:
        u_pert = unpack_input(u_flat)
        dx0 = jnp.zeros(ERROR_DIM)
        return error_dynamics(model, dx0, x_nom, u_pert, dt)

    return jax.jacobian(f_u)(u_nom_vec)


# ── MPC configuration ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class MPCConfig:
    """Tuning knobs for the MPC controller.

    Attributes:
        N: Prediction horizon (number of steps).
        dt: Control timestep [s].
        Q: State cost weight matrix ``(16, 16)``.
        R: Input cost weight matrix ``(5, 5)``.
        Q_terminal: Terminal state cost ``(16, 16)``. If ``None``, uses ``Q``.
        u_min: Element-wise lower bound on inputs ``(5,)``.
        u_max: Element-wise upper bound on inputs ``(5,)``.
        du_max: Maximum input change per step ``(5,)`` (rate constraint).
            ``None`` disables rate constraints.
        max_qp_iters: OSQP iteration budget.
    """

    N: int
    dt: float
    Q: Array
    R: Array
    Q_terminal: Optional[Array]
    u_min: Array
    u_max: Array
    du_max: Optional[Array]
    max_qp_iters: int


def default_mpc_config(
    N: int = 15,
    dt: float = 0.01,
    max_qp_iters: int = 4000,
) -> MPCConfig:
    """Sensible defaults for the RC-car scale model.

    Cost weights:
        * Position (0:3) — moderate weight (1.0)
        * Heading (3:6) — high weight (10.0)
        * Velocity (6:9) — moderate (1.0)
        * Angular vel (9:12) — light (0.1)
        * Wheel spins (12:16) — very light (0.01)

    Input weights:
        * Steering — light (0.1), allow active steering
        * Torques — moderate (1.0), penalise aggressive torque

    Bounds (from car.py defaults):
        * delta ∈ [-0.35, 0.35] rad
        * tau_w ∈ [-0.5, 0.5] N·m per wheel
    """

    q_diag = jnp.array([
        # dp_W (3)        dtheta (3)       dv_W (3)
        1.0, 1.0, 0.1,    10.0, 10.0, 0.1,  1.0, 1.0, 0.1,
        # dw_B (3)         domega_W (4)
        0.1, 0.1, 0.1,    0.01, 0.01, 0.01, 0.01,
    ])
    Q = jnp.diag(q_diag)

    r_diag = jnp.array([
        0.1,                   # delta  — allow active steering
        1.0, 1.0, 1.0, 1.0,   # tau_w  — penalise aggressive torque
    ])
    R = jnp.diag(r_diag)

    # Terminal cost: 10x state cost to encourage convergence
    Q_terminal = 10.0 * Q

    u_min = jnp.array([-0.35, -0.5, -0.5, -0.5, -0.5])
    u_max = jnp.array([ 0.35,  0.5,  0.5,  0.5,  0.5])

    # Rate constraint: limit how fast inputs can change per step
    du_max = jnp.array([0.05, 0.2, 0.2, 0.2, 0.2])

    return MPCConfig(
        N=N, dt=dt, Q=Q, R=R, Q_terminal=Q_terminal,
        u_min=u_min, u_max=u_max, du_max=du_max,
        max_qp_iters=max_qp_iters,
    )


# ── MPC state (warm-start data) ─────────────────────────────────────────────


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MPCState:
    """Carry state between MPC calls (for warm-starting).

    Attributes:
        x_ref: Reference trajectory ``(N+1,)`` list of nominal states used
            for linearisation.
        u_ref: Reference input sequence ``(N, 5)`` used for linearisation.
        u_prev: Previous applied input ``(5,)`` for rate constraints.
    """

    x_ref: list  # length N+1, list of AckermannCarState (pytree-able)
    u_ref: Array  # (N, 5)
    u_prev: Array  # (5,)

    def tree_flatten(self):
        return (self.x_ref, self.u_ref, self.u_prev), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


# ── Trajectory linearisation ─────────────────────────────────────────────────


def linearise_trajectory(
    model: AckermannCarModel,
    x_traj: list,
    u_seq: Array,
    dt: float,
):
    """Linearise dynamics along a nominal trajectory.

    Args:
        model: Car physics model.
        x_traj: List of N+1 nominal states.
        u_seq: Input sequence ``(N, 5)``.
        dt: Timestep [s].

    Returns:
        F_seq: Stacked state Jacobians ``(N, 16, 16)``.
        B_seq: Stacked control Jacobians ``(N, 16, 5)``.
    """
    N = u_seq.shape[0]
    F_list = []
    B_list = []

    for k in range(N):
        u_k = unpack_input(u_seq[k])
        F_k = compute_F(model, x_traj[k], u_k, dt)
        B_k = compute_B(model, x_traj[k], u_k, dt)
        F_list.append(F_k)
        B_list.append(B_k)

    return jnp.stack(F_list), jnp.stack(B_list)


def rollout_nominal(
    model: AckermannCarModel,
    x0: AckermannCarState,
    u_seq: Array,
    dt: float,
) -> list:
    """Forward-simulate the nonlinear model to get a nominal trajectory.

    Args:
        model: Car physics model.
        x0: Initial state.
        u_seq: Input sequence ``(N, 5)``.
        dt: Timestep [s].

    Returns:
        List of N+1 states ``[x0, x1, ..., xN]``.
    """
    traj = [x0]
    x = x0
    for k in range(u_seq.shape[0]):
        u_k = unpack_input(u_seq[k])
        x = model.step(x, u_k, dt)
        traj.append(x)
    return traj


# ── Dense QP assembly ────────────────────────────────────────────────────────


def build_qp_matrices(
    F_seq: Array,
    B_seq: Array,
    Q: Array,
    R: Array,
    Q_terminal: Array,
    dx0: Array,
    N: int,
):
    """Assemble the dense QP matrices for the MPC problem.

    The optimisation variable is the stacked input deviation sequence:
    ``dU = [du_0, du_1, ..., du_{N-1}]`` of shape ``(N * INPUT_DIM,)``.

    The predicted state deviations are expressed as an affine function of dU:
        ``dX_k = M_k @ dU + c_k``
    where M_k and c_k come from recursively applying the linearised dynamics.

    The QP is:

    .. math::

        \\min_{dU}\\; \\frac{1}{2} dU^T H\\, dU + g^T dU

    subject to box constraints on ``u_ref + dU``.

    Args:
        F_seq: ``(N, 16, 16)`` state Jacobians.
        B_seq: ``(N, 16, 5)`` control Jacobians.
        Q: ``(16, 16)`` stage state cost.
        R: ``(5, 5)`` stage input cost.
        Q_terminal: ``(16, 16)`` terminal state cost.
        dx0: ``(16,)`` initial error state (current state minus reference).
        N: Horizon length.

    Returns:
        H: ``(N*5, N*5)`` QP Hessian (positive semi-definite).
        g: ``(N*5,)`` QP gradient vector.
    """
    n_x = ERROR_DIM
    n_u = INPUT_DIM
    n_opt = N * n_u

    # Build prediction matrices: dx_k = S_x @ dx0 + S_u @ dU
    # S_x: (N, 16)  — how initial error propagates
    # S_u: (N, 16, N*5) — how input deviations affect future states

    # Propagation matrices for each step
    # Phi[k] = F_{k-1} @ F_{k-2} @ ... @ F_0
    Phi = jnp.zeros((N + 1, n_x, n_x))
    Phi = Phi.at[0].set(jnp.eye(n_x))
    for k in range(N):
        Phi = Phi.at[k + 1].set(F_seq[k] @ Phi[k])

    # S_u[i, j] = how du_j affects dx_{i+1}
    # S_u[i, j] = Phi[i+1] @ Phi[j+1]^{-1} @ B[j]  for j <= i
    #           = F[i] @ F[i-1] @ ... @ F[j+1] @ B[j]
    S_u = jnp.zeros((N, n_x, n_opt))
    for i in range(N):
        for j in range(i + 1):
            # Compute F[i] @ F[i-1] @ ... @ F[j+1] @ B[j]
            block = B_seq[j]
            for m in range(j + 1, i + 1):
                block = F_seq[m] @ block
            S_u = S_u.at[i, :, j * n_u:(j + 1) * n_u].set(block)

    # S_x[k] = Phi[k+1] @ dx0 — free response (no input deviation)
    # dx_predicted[k] = S_x[k] + S_u[k] @ dU

    # Build cost: J = sum_{k=0}^{N-1} (dx_k' Q dx_k + du_k' R du_k) + dx_N' Qf dx_N
    # Substituting dx_k = Phi[k+1] @ dx0 + S_u[k] @ dU:
    # J = 0.5 * dU' H dU + g' dU + const

    H = jnp.zeros((n_opt, n_opt))
    g = jnp.zeros((n_opt,))

    # Stage costs k = 0 .. N-1
    for k in range(N):
        Q_k = Q_terminal if k == N - 1 else Q
        # Note: we use k for dx_{k+1} (the state AFTER applying du_k)
        # Actually, standard formulation: cost on x_1 through x_N
        S_k = S_u[k]  # (16, N*5)
        c_k = Phi[k + 1] @ dx0  # (16,) free response

        # State cost contribution
        H += S_k.T @ Q_k @ S_k
        g += S_k.T @ Q_k @ c_k

    # Terminal cost (already included as k = N-1 uses Q_terminal above)
    # Actually, let's be more precise. The state at step k+1 is indexed by k
    # in S_u. S_u has N rows for states x_1 through x_N. The terminal state
    # x_N is S_u[N-1]. So Q_terminal is applied at k = N-1. ✓

    # Input cost: R on each du_k
    R_block = jnp.kron(jnp.eye(N), R)
    H += R_block

    # Symmetrise (numerical safety)
    H = 0.5 * (H + H.T)

    return H, g


# ── QP solve (multiple backend options) ─────────────────────────────────────


def solve_qp_boxed(
    H: Array,
    g: Array,
    du_min: Array,
    du_max: Array,
    max_iters: int = 4000,
) -> Array:
    """Solve a box-constrained QP using jaxopt.BoxOSQP.

    .. math::

        \\min_{dU}\\; \\frac{1}{2} dU^T H\\, dU + g^T dU
        \\quad\\text{s.t.}\\quad du_{\\min} \\le dU \\le du_{\\max}

    Args:
        H: QP Hessian ``(n, n)``.
        g: QP gradient ``(n,)``.
        du_min: Lower bounds ``(n,)``.
        du_max: Upper bounds ``(n,)``.
        max_iters: OSQP iteration budget.

    Returns:
        Optimal ``dU`` of shape ``(n,)``.
    """
    try:
        from jaxopt import BoxOSQP

        # BoxOSQP expects: min 0.5 x'Qx + c'x  s.t.  Ax in [l, u]
        # For pure box constraints: A = I, l = du_min, u = du_max
        n = g.shape[0]
        A = jnp.eye(n)

        solver = BoxOSQP(
            maxit=max_iters,
            tol=1e-4,
            verbose=False,
            jit=True,
        )

        # BoxOSQP.run() returns (params, state)
        # params = (primal, dual_eq, dual_ineq)
        sol = solver.run(
            params_obj=(H, g),
            params_eq=A,
            params_ineq=(du_min, du_max),
        )
        return sol.params.primal

    except ImportError:
        # Fallback: projected gradient descent (simple, no dependencies)
        return _solve_qp_pgd(H, g, du_min, du_max, max_iters=500)


def _solve_qp_pgd(
    H: Array, g: Array,
    lb: Array, ub: Array,
    max_iters: int = 500,
    lr: float = 1e-3,
) -> Array:
    """Fallback projected gradient descent for box-constrained QP.

    Not recommended for production — use jaxopt.BoxOSQP instead.
    Provided so the module works without optional dependencies.
    """
    x = jnp.zeros_like(g)

    def body(carry, _):
        x = carry
        grad = H @ x + g
        x = x - lr * grad
        x = jnp.clip(x, lb, ub)
        return x, None

    x, _ = jax.lax.scan(body, x, None, length=max_iters)
    return x


# ── Top-level MPC interface ─────────────────────────────────────────────────


def mpc_init(
    model: AckermannCarModel,
    x0: AckermannCarState,
    u0: AckermannCarInput,
    cfg: MPCConfig,
) -> MPCState:
    """Initialise MPC state with a constant-input trajectory guess.

    Args:
        model: Car physics model.
        x0: Initial state.
        u0: Initial guess for the input (held constant over the horizon).
        cfg: MPC configuration.

    Returns:
        Initial :class:`MPCState`.
    """
    u0_vec = pack_input(u0)
    u_ref = jnp.tile(u0_vec, (cfg.N, 1))  # (N, 5)
    x_ref = rollout_nominal(model, x0, u_ref, cfg.dt)
    return MPCState(x_ref=x_ref, u_ref=u_ref, u_prev=u0_vec)


def mpc_step(
    model: AckermannCarModel,
    x_current: AckermannCarState,
    x_goal: AckermannCarState,
    mpc_state: MPCState,
    cfg: MPCConfig,
) -> tuple[AckermannCarInput, MPCState]:
    """Execute one MPC step: linearise, solve QP, return optimal input.

    This implements a **Real-Time Iteration (RTI)** style approach:

    1. Linearise the dynamics along the previous nominal trajectory.
    2. Compute the initial error (current state vs. reference).
    3. Assemble and solve the QP for input deviations.
    4. Apply the first optimal input.
    5. Shift the trajectory for warm-starting the next call.

    Args:
        model: Car physics model.
        x_current: Current measured/estimated state.
        x_goal: Desired goal state (the reference the MPC tracks towards).
        mpc_state: Previous MPC state (trajectory + warm-start).
        cfg: MPC configuration.

    Returns:
        Tuple ``(u_opt, mpc_state_next)``:
            - ``u_opt``: Optimal control input to apply now.
            - ``mpc_state_next``: Updated MPC state for warm-starting.
    """
    N = cfg.N
    Q_terminal = cfg.Q_terminal if cfg.Q_terminal is not None else cfg.Q

    # ── 1. Linearise along reference trajectory ──────────────────────────
    F_seq, B_seq = linearise_trajectory(
        model, mpc_state.x_ref, mpc_state.u_ref, cfg.dt,
    )

    # ── 2. Initial error: how far is current state from reference? ───────
    dx0 = pack_error_state(state_difference(mpc_state.x_ref[0], x_current))

    # ── 3. Rewrite reference relative to goal ────────────────────────────
    # The cost penalises deviation from goal, so we compute the error of
    # each reference state w.r.t. the goal and fold it into the QP.
    # Replace dx0 with the error relative to goal for the cost:
    dx0_goal = pack_error_state(state_difference(x_goal, x_current))

    # ── 4. Build and solve QP ────────────────────────────────────────────
    H, g = build_qp_matrices(F_seq, B_seq, cfg.Q, cfg.R, Q_terminal, dx0_goal, N)

    # Box constraints on absolute input: u_min <= u_ref + du <= u_max
    # => du_min = u_min - u_ref,  du_max = u_max - u_ref
    du_min_seq = jnp.tile(cfg.u_min, (N,)) - mpc_state.u_ref.ravel()
    du_max_seq = jnp.tile(cfg.u_max, (N,)) - mpc_state.u_ref.ravel()

    # Optionally add rate constraints: |u_k - u_{k-1}| <= du_max
    if cfg.du_max is not None:
        # For the first step: |u_ref[0] + du[0] - u_prev| <= du_max
        for k in range(N):
            if k == 0:
                u_prev = mpc_state.u_prev
            else:
                u_prev = mpc_state.u_ref[k - 1]
            u_ref_k = mpc_state.u_ref[k]
            # du such that u_ref_k + du - u_prev ∈ [-du_max, du_max]
            rate_lo = -cfg.du_max - (u_ref_k - u_prev)
            rate_hi = cfg.du_max - (u_ref_k - u_prev)
            sl = k * INPUT_DIM
            du_min_seq = du_min_seq.at[sl:sl + INPUT_DIM].set(
                jnp.maximum(du_min_seq[sl:sl + INPUT_DIM], rate_lo)
            )
            du_max_seq = du_max_seq.at[sl:sl + INPUT_DIM].set(
                jnp.minimum(du_max_seq[sl:sl + INPUT_DIM], rate_hi)
            )

    dU_opt = solve_qp_boxed(H, g, du_min_seq, du_max_seq, cfg.max_qp_iters)

    # ── 5. Extract first input ───────────────────────────────────────────
    du0 = dU_opt[:INPUT_DIM]
    u0_opt_vec = mpc_state.u_ref[0] + du0
    u0_opt_vec = jnp.clip(u0_opt_vec, cfg.u_min, cfg.u_max)
    u_opt = unpack_input(u0_opt_vec)

    # ── 6. Shift trajectory for warm-start ───────────────────────────────
    u_ref_new = jnp.concatenate([
        mpc_state.u_ref[1:],
        mpc_state.u_ref[-1:],  # repeat last input
    ])
    # Apply the solved deviations to get a better reference
    u_ref_shifted = jnp.zeros_like(u_ref_new)
    for k in range(N):
        if k < N - 1:
            du_k = dU_opt[(k + 1) * INPUT_DIM:(k + 2) * INPUT_DIM]
            u_ref_shifted = u_ref_shifted.at[k].set(
                jnp.clip(mpc_state.u_ref[k + 1] + du_k, cfg.u_min, cfg.u_max)
            )
        else:
            u_ref_shifted = u_ref_shifted.at[k].set(u_ref_new[k])

    # Re-rollout to get new reference trajectory
    x_next = model.step(x_current, u_opt, cfg.dt)
    x_ref_new = rollout_nominal(model, x_next, u_ref_shifted, cfg.dt)

    mpc_state_next = MPCState(
        x_ref=x_ref_new,
        u_ref=u_ref_shifted,
        u_prev=u0_opt_vec,
    )

    return u_opt, mpc_state_next
