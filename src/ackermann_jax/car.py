"""
car.py — Ackermann kinematic model, body-frame formulation.

═══════════════════════════════════════════════════════════════════════════════
State  (5-D, ordered pose-first then body-velocity)
    x = [px, py, θ, v, ω]     shape (5,)

    px, py  [m]      world-frame position
    θ       [rad]    heading (yaw, CCW from +x)
    v       [m/s]    body-frame forward speed   ← lateral speed ≡ 0 by construction
    ω       [rad/s]  yaw rate

Why body-frame velocity?
────────────────────────
  • No-slip structural:   v_lateral ≡ 0 is built into the state definition —
                          no pseudo-measurement needed, no covariance leakage.
  • Low-speed safe:       v → 0 is a scalar limit; H_speed = [0,0,0,1,0] is
                          a constant row — zero singularity at standstill.
  • Direct IMU coupling:  ω is measured directly by the z-axis rate gyro.
  • SO3 upgrade path:     replace (θ, ω) with (q∈ℝ⁴, ω_B∈ℝ³); p_W and v
                          keep the same role.

Rotation convention
───────────────────
  R_BW ∈ SO(2)  maps body → world:

      R_BW(θ) = [[cos θ,  −sin θ],
                  [sin θ,   cos θ]]

      v_world = R_BW @ [v, 0]ᵀ = [v·cos θ,  v·sin θ]

Physics  (continuous time)
──────────────────────────
  ṗ  =  R_BW(θ) @ [v, 0]ᵀ   →  [v·cos θ,  v·sin θ]
  θ̇  =  ω
  v̇  =  a                          (longitudinal acceleration, control input)
  ω̇  =  a · tan(δ) / L             d/dt of (v·tan δ / L) holding δ̇ ≈ 0
                                    → changes in δ reach ω via the IMU gyro update

Control:  u = (a, δ)  —  [m/s²] longitudinal acc,  [rad] front-wheel steer angle

EKF interface
─────────────
  state_to_vec / vec_to_state  →  flat (5,)  [px, py, θ, v, ω]
  linearise()                  →  (x_next_vec, F)  via jacfwd through full RK4
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────


@jax.tree_util.register_dataclass
@dataclass
class AckermannParams:
    """
    Physical parameters.  All fields are *static* (Python scalars, not JAX
    arrays) — they become part of the JIT cache key, not traced through AD.

    wheelbase : L  [m]   front-to-rear axle distance
    mass      : m  [kg]  body mass (reserved for future force / dynamics layers)
    """
    wheelbase : float = field(default=0.257, metadata=dict(static=True))
    mass      : float = field(default=1.5,   metadata=dict(static=True))


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────


@jax.tree_util.register_dataclass
@dataclass
class CarState:
    """
    Body-frame kinematic state.

    p_W   : (2,)  world-frame position         [m]
    theta :  ()   heading (yaw, CCW from +x)   [rad]
    v     :  ()   body-frame forward speed      [m/s]   (v_lateral ≡ 0)
    omega :  ()   yaw rate                      [rad/s]

    All fields are data leaves (non-static): traced by jit / grad / vmap.

    SO3 upgrade: replace
        theta : jax.Array   shape ()
        omega : jax.Array   shape ()
    with
        q     : jax.Array   shape (4,)  unit quaternion
        omega : jax.Array   shape (3,)  body-frame angular velocity
    and update body_axes() / xdot() accordingly.
    """
    p_W   : jax.Array   # shape (2,)
    theta : jax.Array   # shape ()
    v     : jax.Array   # shape ()
    omega : jax.Array   # shape ()


# ─────────────────────────────────────────────────────────────────────────────
# Control
# ─────────────────────────────────────────────────────────────────────────────


@jax.tree_util.register_dataclass
@dataclass
class CarControl:
    """
    a     :  ()  longitudinal acceleration  [m/s²]  (positive = forward)
    delta :  ()  front-wheel steering angle [rad]   (positive = left turn)
    """
    a     : jax.Array   # shape ()
    delta : jax.Array   # shape ()


# ─────────────────────────────────────────────────────────────────────────────
# Rotation helpers  (SO2 now; swap columns for SO3 later)
# ─────────────────────────────────────────────────────────────────────────────


def body_axes(theta: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Return (e_long, e_lat): forward and left unit vectors in world frame.

        e_long = R_BW @ [1, 0]ᵀ  =  [ cos θ,  sin θ]
        e_lat  = R_BW @ [0, 1]ᵀ  =  [−sin θ,  cos θ]

    SO3 analogue: first two columns of the body→world rotation matrix R_BW.
    """
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([c, s]), jnp.array([-s, c])


def rotation_BW(theta: jax.Array) -> jax.Array:
    """
    Body-to-world rotation matrix  R_BW ∈ SO(2).

        R_BW(θ) = [[cos θ,  −sin θ],
                   [sin θ,   cos θ]]

    Usage:  v_world = R_BW(θ) @ v_body
    """
    c, s = jnp.cos(theta), jnp.sin(theta)
    return jnp.array([[c, -s],
                      [s,  c]])


# ─────────────────────────────────────────────────────────────────────────────
# Continuous-time dynamics
# ─────────────────────────────────────────────────────────────────────────────


def xdot(state   : CarState,
         control : CarControl,
         params  : AckermannParams) -> CarState:
    """
    Continuous-time body-frame Ackermann kinematic dynamics.

    Returns a CarState whose fields are the time-derivatives of the
    corresponding state fields (a tangent vector in the same pytree shape).

        ṗ  =  R_BW(θ) @ [v, 0]ᵀ   →  [v·cos θ,  v·sin θ]
        θ̇  =  ω
        v̇  =  a
        ω̇  =  a · tan(δ) / L      (d/dt of v·tan δ / L with δ̇ ≈ 0)

    The body-frame body velocity [v, 0] makes the no-slip condition structural:
    there is no lateral speed component in the state, so no pseudo-measurement
    is needed to enforce the non-holonomic constraint.
    """
    v     = state.v
    theta = state.theta
    omega = state.omega
    a     = control.a
    delta = control.delta
    L     = params.wheelbase

    # Body-to-world rotation  R_BW ∈ SO(2)
    R_BW = rotation_BW(theta)

    # World-frame position rate: project body velocity [v, 0] into world frame
    p_dot = R_BW @ jnp.array([v, 0.0])   # = [v·cos θ,  v·sin θ]

    # Yaw-rate dynamics: ω̇ = a·tan(δ)/L
    # Derived from d/dt(v·tan δ / L) holding δ̇ = 0.
    # When δ changes (between timesteps), the discrepancy is absorbed by
    # the IMU gyro measurement update; Q_ω provides the process slack.
    omega_dot = a * jnp.tan(delta) / L

    return CarState(
        p_W   = p_dot,
        theta = omega,       # θ̇ = ω
        v     = a,           # v̇ = a
        omega = omega_dot,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Integrators
# ─────────────────────────────────────────────────────────────────────────────


def step_euler(state   : CarState,
               control : CarControl,
               params  : AckermannParams,
               dt      : float) -> CarState:
    """First-order Euler step."""
    d = xdot(state, control, params)
    return jax.tree.map(lambda s, dd: s + dt * dd, state, d)


def step_rk4(state   : CarState,
             control : CarControl,
             params  : AckermannParams,
             dt      : float) -> CarState:
    """
    Classical RK4 step, control held constant over [t, t+dt].

    Uses jax.tree.map throughout — valid for any pytree CarState,
    including future SO3 versions with quaternion + body-ω fields.
    """
    def f(s: CarState) -> CarState:
        return xdot(s, control, params)

    def axpy(alpha: float, tangent: CarState, base: CarState) -> CarState:
        return jax.tree.map(lambda b, t: b + alpha * t, base, tangent)

    k1 = f(state)
    k2 = f(axpy(0.5 * dt, k1, state))
    k3 = f(axpy(0.5 * dt, k2, state))
    k4 = f(axpy(      dt, k3, state))

    return jax.tree.map(
        lambda s, a, b, c, d: s + (dt / 6.0) * (a + 2.0*b + 2.0*c + d),
        state, k1, k2, k3, k4,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Flatten / unflatten  (EKF linear-algebra interface)
# ─────────────────────────────────────────────────────────────────────────────

STATE_DIM = 5  # [px, py, θ, v, ω]


def state_to_vec(state: CarState) -> jax.Array:
    """Pack CarState into a flat (5,) vector: [px, py, θ, v, ω]."""
    return jnp.array([state.p_W[0], state.p_W[1],
                      state.theta, state.v, state.omega])


def vec_to_state(vec: jax.Array) -> CarState:
    """Unpack a flat (5,) vector back to CarState."""
    return CarState(
        p_W   = vec[0:2],
        theta = vec[2],
        v     = vec[3],
        omega = vec[4],
    )


# ─────────────────────────────────────────────────────────────────────────────
# EKF linearisation
# ─────────────────────────────────────────────────────────────────────────────


def linearise(state   : CarState,
              control : CarControl,
              params  : AckermannParams,
              dt      : float) -> tuple[jax.Array, jax.Array]:
    """
    Propagate the state and compute the discrete-time Jacobian F.

    Uses jacfwd through the full RK4 step — no analytic Jacobian needed.

    The Jacobian has the approximate structure (at small dt):

        ∂x'/∂x ≈ I + dt · J_f   where J_f = ∂ẋ/∂x

        J_f =        px  py   θ         v         ω
              px  [  0   0  −v·sinθ   cosθ       0  ]
              py  [  0   0   v·cosθ   sinθ       0  ]
              θ   [  0   0    0         0         1  ]
              v   [  0   0    0         0         0  ]
              ω   [  0   0    0         0         0  ]

    The position-heading and position-speed cross-terms are non-trivial
    during turns — exactly the terms that jacfwd captures correctly.

    Returns
    -------
    x_next : (STATE_DIM,)            propagated state as flat vector
    F      : (STATE_DIM, STATE_DIM)  Jacobian  d(x_next) / d(x_k)
    """
    def propagate(vec: jax.Array) -> jax.Array:
        return state_to_vec(step_rk4(vec_to_state(vec), control, params, dt))

    x_vec  = state_to_vec(state)
    x_next = propagate(x_vec)
    F      = jax.jacfwd(propagate)(x_vec)

    return x_next, F
