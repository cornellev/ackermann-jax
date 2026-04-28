"""
car.py — Ackermann kinematic model, Layer 1.

Pytree convention
-----------------
All structs are @dataclass + @jax.tree_util.register_dataclass.

Fields are split into two categories:

  data fields   — JAX arrays; traced through jit/grad/vmap/jacfwd.
                  Default: every field that is NOT marked static.

  meta fields   — Python scalars / strings / tuples; treated as static
                  (part of the JIT cache key, not traced).
                  Marked with:  field(metadata=dict(static=True))

This is the idiomatic JAX >= 0.4.36 pattern.  When upgrading to SO3 later,
only CarState changes (replace theta: Array with q: Array of shape (4,));
all integrator and linearise code is unchanged.

State
-----
CarState:
    p_W   (2,)  world-frame position [m]
    v_W   (2,)  world-frame velocity [m/s]
    theta  ()   heading angle (yaw, CCW from +x) [rad]

Heading is explicit state -- not re-derived from arctan2(v_W) -- so the
Jacobian is smooth at all speeds and a parked-with-heading initial condition
is representable.  In the SO3 upgrade this field becomes a quaternion.

Physics
-------
Ackermann geometry gives heading rate:

    theta_dot = v * tan(delta) / L        (v regularised by eps_v)

Velocity dynamics from d/dt [v * cos th, v * sin th]:

    v_dot_W = a * e_long + v * theta_dot * e_lat

where e_long = [cos th, sin th]  (forward)
      e_lat  = [-sin th, cos th]  (left)

EKF interface
-------------
state_to_vec / vec_to_state  ->  flat (STATE_DIM,) = 5 vector
linearise()  ->  (x_next_vec, F)  via jacfwd through the full RK4 step
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@jax.tree_util.register_dataclass
@dataclass
class AckermannParams:
    """
    Physical / numerical parameters.

    All fields are static (meta) -- Python floats, not JAX arrays.
    Changing them triggers a JIT recompile, which is intentional: params are
    fixed at model-construction time and are not differentiated through.
    If you need to sweep params, pass them as separate JAX scalars into your
    dynamics closure instead.

    wheelbase : L    [m]    front-to-rear axle distance
    mass      : m    [kg]   body mass (reserved for force layers)
    eps_v     :      [m/s]  speed regularisation; gates theta_dot to zero
                             at standstill without a hard divide-by-zero
    """
    wheelbase : float = field(default=0.257, metadata=dict(static=True))
    mass      : float = field(default=1.5,   metadata=dict(static=True))
    eps_v     : float = field(default=1e-3,  metadata=dict(static=True))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@jax.tree_util.register_dataclass
@dataclass
class CarState:
    """
    Kinematic state of the car.

    p_W   : (2,)  world-frame position [m]
    v_W   : (2,)  world-frame velocity [m/s]
    theta :  ()   heading angle [rad] -- explicit, not derived from v_W

    All fields are data (non-static): JAX arrays traced by jit/grad/vmap.

    SO3 upgrade: replace
        theta : jax.Array  # shape ()
    with
        q     : jax.Array  # shape (4,)  quaternion
    and update body_axes() / xdot() accordingly.  Nothing else changes.
    """
    p_W   : jax.Array  # shape (2,)
    v_W   : jax.Array  # shape (2,)
    theta : jax.Array  # shape ()


# ---------------------------------------------------------------------------
# Control
# ---------------------------------------------------------------------------

@jax.tree_util.register_dataclass
@dataclass
class CarControl:
    """
    a     :  ()  longitudinal acceleration [m/s^2]  (positive = forward)
    delta :  ()  front-wheel steering angle [rad]   (positive = left turn)
    """
    a     : jax.Array  # shape ()
    delta : jax.Array  # shape ()


# ---------------------------------------------------------------------------
# Rotation helpers  (SO2 now; swap for SO3 equivalents later)
# ---------------------------------------------------------------------------

def body_axes(theta: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Return (e_long, e_lat): the forward and left unit vectors in world frame.

        e_long = R(theta) @ [1, 0]  =  [ cos th,  sin th]
        e_lat  = R(theta) @ [0, 1]  =  [-sin th,  cos th]

    SO3 analogue: extract the first two columns of the rotation matrix R_WB.
    """
    c, s = jnp.cos(theta), jnp.sin(theta)
    e_long = jnp.array([ c,  s])
    e_lat  = jnp.array([-s,  c])
    return e_long, e_lat


# ---------------------------------------------------------------------------
# Continuous-time dynamics
# ---------------------------------------------------------------------------

def xdot(state   : CarState,
         control : CarControl,
         params  : AckermannParams) -> CarState:
    """
    Continuous-time Ackermann kinematic dynamics.

    Returns a CarState whose fields are the time-derivatives of the
    corresponding state fields.  This is a tangent vector with the same
    pytree structure, which lets jax.tree.map work cleanly in the
    integrators below.

        d/dt p_W   = v_W
        d/dt v_W   = a * e_long + v * theta_dot * e_lat
        d/dt theta = theta_dot  =  v * tan(delta) / L
    """
    v_W   = state.v_W
    theta = state.theta
    a     = control.a
    delta = control.delta
    L     = params.wheelbase
    eps   = params.eps_v

    # Regularised forward speed -- always >= eps, gates theta_dot to 0 smoothly
    speed = jnp.sqrt(jnp.dot(v_W, v_W) + eps ** 2)

    e_long, e_lat = body_axes(theta)

    theta_dot = speed * jnp.tan(delta) / L

    # Longitudinal acceleration + centripetal rotation of the velocity vector
    v_dot = a * e_long + speed * theta_dot * e_lat

    return CarState(
        p_W   = v_W,
        v_W   = v_dot,
        theta = theta_dot,
    )


# ---------------------------------------------------------------------------
# Integrators
# ---------------------------------------------------------------------------

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

    Uses jax.tree.map throughout -- works for any pytree CarState,
    including future SO3 versions with quaternion fields.
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


# ---------------------------------------------------------------------------
# Flatten / unflatten  (EKF linear-algebra interface)
# ---------------------------------------------------------------------------

STATE_DIM = 5  # [px, py, vx, vy, theta]


def state_to_vec(state: CarState) -> jax.Array:
    """Pack CarState into a flat (5,) vector: [px, py, vx, vy, theta]."""
    return jnp.concatenate([state.p_W, state.v_W, state.theta[jnp.newaxis]])


def vec_to_state(vec: jax.Array) -> CarState:
    """Unpack a flat (5,) vector back to CarState."""
    return CarState(
        p_W   = vec[0:2],
        v_W   = vec[2:4],
        theta = vec[4],
    )


# ---------------------------------------------------------------------------
# EKF linearisation
# ---------------------------------------------------------------------------

def linearise(state   : CarState,
              control : CarControl,
              params  : AckermannParams,
              dt      : float) -> tuple[jax.Array, jax.Array]:
    """
    Propagate the state and compute the discrete-time Jacobian F.

    Uses forward-mode autodiff (jacfwd) through the full RK4 step.
    No analytic Jacobian required -- the pytree dynamics define everything.

    Returns
    -------
    x_next : (STATE_DIM,)            propagated state as a flat vector
    F      : (STATE_DIM, STATE_DIM)  Jacobian  d(x_next) / d(x_k)
    """
    def propagate(vec: jax.Array) -> jax.Array:
        return state_to_vec(step_rk4(vec_to_state(vec), control, params, dt))

    x_vec  = state_to_vec(state)
    x_next = propagate(x_vec)
    F      = jax.jacfwd(propagate)(x_vec)

    return x_next, F
