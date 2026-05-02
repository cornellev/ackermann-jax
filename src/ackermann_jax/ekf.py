"""
ekf.py  —  Extended Kalman Filter for the Ackermann bicycle model.

═══════════════════════════════════════════════════════════════════════════════
State vector  (body-frame formulation, matches car.py)
    x = [px, py, θ, v, ω]    shape (5,)

    px, py  [m]      world-frame position
    θ       [rad]    heading (yaw, CCW from +x)
    v       [m/s]    body-frame forward speed   (v_lateral ≡ 0 by construction)
    ω       [rad/s]  yaw rate

    Why this is better than world-frame [px, py, vx, vy, θ]
    ─────────────────────────────────────────────────────────
    • No-slip structural:  lateral speed ≡ 0 — no pseudo-measurement needed.
    • Low-speed safe:      H_speed = [0,0,0,1,0] is a constant row vector;
                           no sqrt(vx²+vy²) → no singularity at standstill.
    • Direct IMU coupling: ω measured directly by z-axis rate gyro.
    • Better consistency:  state components map 1-to-1 to sensor observables.

Process model  (RK4, from car.py; Jacobian via jacfwd — no hand-coding)
    x_{k+1}   = f(x_k, u_k, dt)
    F_k       = ∂f/∂x_k           computed by jax.jacfwd through full RK4
    P_{k+1|k} = F_k P_k F_kᵀ + Q

Measurement models  (all H Jacobians via jacfwd)
    GPS       z ∈ ℝ²    h(x) = [px, py]             H = [[1,0,0,0,0],[0,1,0,0,0]]
    Speed     z ∈ ℝ¹    h(x) = v = x[3]             H = [[0,0,0,1,0]]  (constant!)
    Heading   z ∈ ℝ¹    h(x) = θ = x[2]             H = [[0,0,1,0,0]]  (constant!)
    Yaw-rate  z ∈ ℝ¹    h(x) = ω = x[4]             H = [[0,0,0,0,1]]  (constant!)

All measurement Jacobians are constant — jacfwd will verify this automatically.
The only nontrivial Jacobian is F (from the position-heading-speed coupling).

Covariance update uses the numerically robust Joseph form:
    P⁺ = (I − KH) P (I − KH)ᵀ + K R Kᵀ

JIT usage
──────────
    _predict_jit  = jax.jit(ekf_predict,       static_argnums=(4,))  # dt static
    _upd_gps_jit  = jax.jit(ekf_update_gps)
    _upd_spd_jit  = jax.jit(ekf_update_speed)
    _upd_hdg_jit  = jax.jit(ekf_update_heading)
    _upd_gyr_jit  = jax.jit(ekf_update_yaw_rate)

ROS integration sketch  (100 Hz predict, 5 Hz GPS)
    def imu_callback(msg):
        self.est = _predict_jit(self.est, ctrl, CAR_P, EKF_P, DT)
        self.est = _upd_spd_jit(self.est, z_spd, EKF_P.R_speed)      # odometry
        self.est = _upd_hdg_jit(self.est, z_hdg, EKF_P.R_heading)    # IMU yaw
        self.est = _upd_gyr_jit(self.est, z_gyr, EKF_P.R_yaw_rate)   # IMU ω_z

    def gps_callback(msg):
        self.est = _upd_gps_jit(self.est, z_gps, EKF_P.R_gps)

    For offline / batch use: wrap the loop in jax.lax.scan for ~100× speed-up.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .car import (
    AckermannParams,
    CarControl,
    linearise,
    vec_to_state,
    STATE_DIM,
)

jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@jax.tree_util.register_dataclass
@dataclass
class EKFState:
    """
    Gaussian belief over the car state.

    mean : (STATE_DIM,)            map-estimate  [px, py, θ, v, ω]
    cov  : (STATE_DIM, STATE_DIM)  covariance matrix P
    """

    mean: jax.Array   # shape (5,)
    cov:  jax.Array   # shape (5, 5)


@jax.tree_util.register_dataclass
@dataclass
class EKFParams:
    """
    Tuning parameters for the EKF.  All fields are JAX arrays (data leaves).

    State ordering: x = [px, py, θ, v, ω]

    Q            (5, 5)   Discrete-time process noise covariance.
                          Safe diagonal start:
                          Q = diag([σ_p², σ_p², σ_θ², σ_v², σ_ω²])
    R_gps        (2, 2)   GPS position noise                [m²]
    R_speed      (1, 1)   Odometry / encoder speed noise   [(m/s)²]
    R_heading    (1, 1)   Compass / IMU yaw noise          [rad²]
    R_yaw_rate   (1, 1)   IMU z-axis rate gyro noise       [(rad/s)²]
    """

    Q:           jax.Array   # (5, 5)
    R_gps:       jax.Array   # (2, 2)
    R_speed:     jax.Array   # (1, 1)
    R_heading:   jax.Array   # (1, 1)
    R_yaw_rate:  jax.Array   # (1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────


def wrap_angle(a: jax.Array) -> jax.Array:
    """Map any angle (or array of angles) into (−π, π]."""
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def _sym(P: jax.Array) -> jax.Array:
    """Enforce exact symmetry: P ← (P + Pᵀ) / 2."""
    return 0.5 * (P + P.T)


# ─────────────────────────────────────────────────────────────────────────────
# Measurement functions  h : ℝ⁵ → ℝᵐ
#
# State layout: x = [px(0), py(1), θ(2), v(3), ω(4)]
# ─────────────────────────────────────────────────────────────────────────────


def h_gps(x: jax.Array) -> jax.Array:
    """GPS position fix: predicts [px, py].  Output shape (2,).
    H = [[1,0,0,0,0], [0,1,0,0,0]]  (constant)."""
    return x[:2]


def h_speed(x: jax.Array) -> jax.Array:
    """
    Odometry / wheel encoder: predicts body-frame forward speed v.
    Output shape (1,).

    h(x) = x[3]   →   H = [[0, 0, 0, 1, 0]]  (constant, no singularity).

    Compare with the old world-frame h(x)=‖[vx,vy]‖ which had H∝[vx/v, vy/v]
    and became ill-conditioned at standstill.  With the body-frame state, the
    Jacobian is a constant identity row — no special handling needed.
    """
    return x[3:4]


def h_heading(x: jax.Array) -> jax.Array:
    """Compass / IMU-integrated yaw: predicts θ.  Output shape (1,).
    H = [[0, 0, 1, 0, 0]]  (constant)."""
    return x[2:3]


def h_yaw_rate(x: jax.Array) -> jax.Array:
    """IMU z-axis rate gyro: predicts ω.  Output shape (1,).
    H = [[0, 0, 0, 0, 1]]  (constant)."""
    return x[4:5]


# ─────────────────────────────────────────────────────────────────────────────
# EKF predict (time-update)
# ─────────────────────────────────────────────────────────────────────────────


def ekf_predict(
    est:        EKFState,
    control:    CarControl,
    car_params: AckermannParams,
    ekf_params: EKFParams,
    dt:         float,
) -> EKFState:
    """
    EKF time-update (predict) step.

        μ_{k+1|k} = f(μ_k, u_k)         via RK4 (car.py)
        F_k       = ∂f/∂x |_{μ_k}       via jax.jacfwd through RK4
        P_{k+1|k} = F_k P_k F_kᵀ + Q

    dt is a Python float — mark it static_arg when JIT-compiling.
    """
    x_next, F = linearise(vec_to_state(est.mean), control, car_params, dt)

    # θ is at index 2 in the new state ordering [px, py, θ, v, ω]
    x_next = x_next.at[2].set(wrap_angle(x_next[2]))

    P_pred = _sym(F @ est.cov @ F.T + ekf_params.Q)
    return EKFState(mean=x_next, cov=P_pred)


# ─────────────────────────────────────────────────────────────────────────────
# Generic EKF measurement update (internal)
# ─────────────────────────────────────────────────────────────────────────────


def _ekf_update(
    est:         EKFState,
    z:           jax.Array,   # (m,)
    h_fn,                      # ℝ⁵ → ℝᵐ  — baked in by concrete wrappers
    R:           jax.Array,   # (m, m)
    angle_innov: bool,         # True → wrap the (scalar) innovation as an angle
) -> EKFState:
    """
    Generic EKF measurement-update.

    H = ∂h/∂x computed via jax.jacfwd.
    Covariance updated with the Joseph form for numerical robustness:
        P⁺ = (I − KH) P (I − KH)ᵀ + K R Kᵀ
    """
    z_hat = h_fn(est.mean)
    innov = z - z_hat

    if angle_innov:
        innov = wrap_angle(innov)

    H   = jax.jacfwd(h_fn)(est.mean)           # (m, 5)
    S   = H @ est.cov @ H.T + R                # (m, m)
    K   = est.cov @ H.T @ jnp.linalg.inv(S)   # (5, m)

    x_upd = est.mean + K @ innov
    x_upd = x_upd.at[2].set(wrap_angle(x_upd[2]))   # wrap θ at index 2

    I_n   = jnp.eye(STATE_DIM)
    IKH   = I_n - K @ H
    P_upd = _sym(IKH @ est.cov @ IKH.T + K @ R @ K.T)

    return EKFState(mean=x_upd, cov=P_upd)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sensor update functions  (directly JIT-able)
# ─────────────────────────────────────────────────────────────────────────────


def ekf_update_gps(
    est:   EKFState,
    z_gps: jax.Array,   # (2,)  [px_meas, py_meas]  [m]
    R_gps: jax.Array,   # (2, 2)
) -> EKFState:
    """GPS position update.  z_gps = [px, py] [m]."""
    return _ekf_update(est, z_gps, h_gps, R_gps, angle_innov=False)


def ekf_update_speed(
    est:     EKFState,
    z_speed: jax.Array,   # (1,)  [measured forward speed]  [m/s]
    R_speed: jax.Array,   # (1, 1)
) -> EKFState:
    """
    Wheel-encoder / odometry speed update.  z_speed = [v] [m/s].

    h(x) = x[3] — H is a constant row [0,0,0,1,0], never singular.
    """
    return _ekf_update(est, z_speed, h_speed, R_speed, angle_innov=False)


def ekf_update_heading(
    est:       EKFState,
    z_heading: jax.Array,   # (1,)  [measured heading]  [rad]
    R_heading: jax.Array,   # (1, 1)
) -> EKFState:
    """
    Compass / IMU-integrated yaw update.  z_heading = [θ] [rad].

    Innovation is wrapped to (−π, π] to handle the ±π discontinuity.
    """
    return _ekf_update(est, z_heading, h_heading, R_heading, angle_innov=True)


def ekf_update_yaw_rate(
    est:        EKFState,
    z_yaw_rate: jax.Array,   # (1,)  [measured ω_z]  [rad/s]
    R_yaw_rate: jax.Array,   # (1, 1)
) -> EKFState:
    """
    IMU z-axis rate-gyro update.  z_yaw_rate = [ω] [rad/s].

    h(x) = x[4] — H is a constant row [0,0,0,0,1].
    This update tightly couples the yaw-rate state to the IMU, automatically
    tracking changes in ω caused by steering-angle variations between steps
    (the effect that ω̇ = a·tan(δ)/L alone cannot capture when dδ/dt ≠ 0).
    """
    return _ekf_update(est, z_yaw_rate, h_yaw_rate, R_yaw_rate, angle_innov=False)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience constructor
# ─────────────────────────────────────────────────────────────────────────────


def make_ekf_params(
    sig_pos:   float = 0.05,   # process σ: position         [m]
    sig_hdg:   float = 0.02,   # process σ: heading          [rad]
    sig_vel:   float = 0.05,   # process σ: forward speed    [m/s]
    sig_omega: float = 0.05,   # process σ: yaw rate         [rad/s]
    sig_gps:   float = 0.50,   # GPS measurement σ           [m]
    sig_spd:   float = 0.05,   # odometry speed σ            [m/s]
    sig_yaw:   float = 0.02,   # compass / IMU heading σ     [rad]
    sig_gyro:  float = 0.02,   # IMU rate-gyro σ             [rad/s]
) -> EKFParams:
    """
    Build EKFParams from per-axis standard deviations.

    Q diagonal ordering follows the state: [px, py, θ, v, ω].

    Tuning guidance:
        • Increase Q entries if the filter lags behind fast manoeuvres.
        • Increase R entries if measurements are noisy / intermittent.
        • For a kinematic sim where the model is exact, very small Q works;
          for real hardware, inflate Q to absorb tyre slip and model error.
    """
    Q = jnp.diag(jnp.array([
        sig_pos**2,    # px
        sig_pos**2,    # py
        sig_hdg**2,    # θ
        sig_vel**2,    # v
        sig_omega**2,  # ω
    ]))
    return EKFParams(
        Q          = Q,
        R_gps      = sig_gps**2  * jnp.eye(2),
        R_speed    = jnp.array([[sig_spd**2]]),
        R_heading  = jnp.array([[sig_yaw**2]]),
        R_yaw_rate = jnp.array([[sig_gyro**2]]),
    )
