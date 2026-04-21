"""
test_ekf2.py — same as test_ekf.py but with no front-wheel RPM sensing.

Only rear wheel encoders (RL, RR) are fused; front wheel angular velocities
are treated as unobserved.
"""
import time

import jax
import jax.numpy as jnp
from jax import Array
from matplotlib import pyplot as plt
import jaxlie
import addcopyfighandler

from ackermann_jax import (
    default_params,
    default_state,
    AckermannCarModel,
    AckermannCarInput,
    AckermannCarState,
)

from ackermann_jax.ekf import EKFState, ekf_predict, ekf_update, ERROR_DIM

jax.config.update("jax_enable_x64", False)
# jax.config.update("jax_log_compiles", True)  # Print out compiled HLO for debugging


# Error-state index slices (must match pack/unpack_error_state ordering)
_P_IDX = {
    "p_W": slice(0, 3),
    "theta": slice(3, 6),
    "v_W": slice(6, 9),
    "w_B": slice(9, 12),
    "omega_W": slice(12, 16),
}


def yaw_from_R(R_WB: jaxlie.SO3) -> jnp.ndarray:
    return R_WB.compute_yaw_radians()


def run_model(
    model,
    x0,
    dt=0.01,
    T_settle=1.5,
    T_straight=2.0,
    T_turn=2.0,
    v_cmd=1.0,
    psi_rate_cmd=1.0,
    Kp_v=10.0,
    Ki_v=4.0,
    tau_max=2.0,
    integ_max_v=0.5,
    Kp_s=1.5,
    Ki_s=0.2,
    Kd_s=0.5,
    delta_max=0.35,
    integ_max_s=0.5,
    method="semi_implicit_euler",
):
    N_settle = int(T_settle / dt)
    N_straight = int(T_straight / dt)
    N_turn = int(T_turn / dt)
    N = N_settle + N_straight + N_turn

    t_turn_start = T_settle + T_straight

    def v_cmd_schedule(k):
        return jnp.where(k < N_settle, 0.0, v_cmd)

    def psi_cmd_schedule(k):
        t = k * dt
        turn_elapsed = jnp.maximum(0.0, t - t_turn_start)
        return jnp.where(k < (N_settle + N_straight), 0.0, psi_rate_cmd * turn_elapsed)

    def step_once(carry, k):
        x, integ_v, integ_s = carry

        v_ref = v_cmd_schedule(k)
        psi_cmd = psi_cmd_schedule(k)

        tau_w, integ_v_next = model.map_velocity_to_wheel_torques(
            x=x,
            v_cmd=v_ref,
            integral_state=integ_v,
            dt=dt,
            Kp=Kp_v,
            Ki=Ki_v,
            tau_max=tau_max,
            integ_max=integ_max_v,
            use_traction_limit=True,
        )

        delta, integ_s_next = model.map_heading_to_steering(
            x=x,
            psi_cmd=psi_cmd,
            integral_state=integ_s,
            dt=dt,
            Kp=Kp_s,
            Ki=Ki_s,
            Kd=Kd_s,
            delta_max=delta_max,
            integ_max=integ_max_s,
        )

        u = AckermannCarInput(delta=delta, tau_w=tau_w)
        x_next = model.step(x=x, u=u, dt=dt, method=method)

        yaw = yaw_from_R(x.R_WB)
        R = x.R_WB.as_matrix()
        v_B = R.T @ x.v_W

        log = {
            "p_W": x_next.p_W,
            "v_W": x_next.v_W,
            "v_B": v_B,
            "v_X": v_B[0],
            "v_Y": v_B[1],
            "yaw": yaw,
            "tau_w": tau_w,
            "delta": delta,
            "psi_cmd": psi_cmd,
            "v_cmd": v_ref,
        }
        return (x_next, integ_v_next, integ_s_next), (log, x_next)

    ks = jnp.arange(N, dtype=jnp.int32)
    (_, _, _), (logs, stateHist) = jax.lax.scan(
        step_once,
        (
            x0,
            jnp.array(0.0, dtype=jnp.float32),
            jnp.array(0.0, dtype=jnp.float32),
        ),
        ks,
    )

    logs["t"] = jnp.arange(N) * dt
    logs["N_settle"] = N_settle
    logs["N_straight"] = N_straight
    logs["N_turn"] = N_turn
    return logs, stateHist


def run_case(model, x0, dt, T_settle, T_straight, T_turn, v_cmd, delta_turn):
    L = float(model.params.geom.L)
    psi_rate_cmd = v_cmd / L * float(jnp.tan(jnp.array(delta_turn)))

    return run_model(
        model=model,
        x0=x0,
        dt=dt,
        T_settle=T_settle,
        T_straight=T_straight,
        T_turn=T_turn,
        v_cmd=v_cmd,
        psi_rate_cmd=psi_rate_cmd,
        Kp_v=40.0,
        Ki_v=2.0,
        tau_max=0.35,
        integ_max_v=0.5,
        Kp_s=3.0,
        Ki_s=2.0,
        Kd_s=0.4,
        delta_max=0.35,
        integ_max_s=0.5,
    )


# ── Measurement functions ─────────────────────────────────────────────
#
# Each h(x) maps AckermannCarState -> Array.
# The EKF computes H = d(h ∘ inject_error) / d(dx) automatically via AD.


def h_gps(x: AckermannCarState) -> Array:
    """GPS: measures world-frame position."""
    return x.p_W  # (3,)


def h_gyro(x: AckermannCarState) -> Array:
    """Gyroscope: measures body angular velocity."""
    return x.w_B  # (3,)


def h_rear_wheels(x: AckermannCarState) -> Array:
    """Rear wheel encoders (RL, RR): omega_W[2:4]."""
    return x.omega_W[2:]  # (2,)


def h_front_eq_rear(x: AckermannCarState) -> Array:
    """Pseudo-measurement: FL - RL and FR - RR, expected to be ~0."""
    return x.omega_W[:2] - x.omega_W[2:]  # (2,)


def make_h_accel(model: AckermannCarModel, u: AckermannCarInput):
    """
    IMU accelerometer: specific force in body frame.

    The accelerometer reads:  z = R_BW @ (dv_W - g_W) = R_BW @ F_W / m

    model.xdot returns dv_W = F_W/m + g_W  (car.py line 127), so:
        h(x) = R_BW @ (model.xdot(x, u).v_W - g_W)

    u is closed over as a JAX-traced value; make_h_accel must be called
    inside the scan body so that u is the current-step input.
    """
    g_W = jnp.array([0.0, 0.0, -model.params.chassis.g], dtype=jnp.float32)

    def h_accel(x: AckermannCarState) -> Array:
        dv_W = model.xdot(x, u).v_W
        R_BW = x.R_WB.as_matrix().T
        return R_BW @ (dv_W - g_W)  # (3,) specific force in body frame

    return h_accel


def generate_measurements(
    stateHist,
    g: float,
    dt: float,
    R_gps: float,
    R_gyro: float,
    R_accel: float,
    R_rear_wheels: float,
    key,
):
    """
    Generate noisy sensor measurements from a ground-truth state history.

    Pattern:  z = h(x_true) + sqrt(R) * randn

    The accelerometer signal includes true vehicle linear acceleration
    (finite-differenced from v_W) so that z_accel matches what a real IMU
    would report.  The EKF model only predicts the gravity component, so
    R_accel must be large enough to absorb the unmodeled vehicle dynamics.
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)

    # GPS — position
    z_gps = stateHist.p_W + jnp.sqrt(R_gps) * jax.random.normal(
        k1, stateHist.p_W.shape
    )

    # Gyroscope — body angular velocity
    z_gyro = stateHist.w_B + jnp.sqrt(R_gyro) * jax.random.normal(
        k2, stateHist.w_B.shape
    )

    # IMU accelerometer — R_BW @ (a_W_true + [0,0,g])
    # a_W_true approximated by finite difference of v_W; last step padded with 0.
    g_up_W = jnp.array([0.0, 0.0, g], dtype=jnp.float32)
    a_W_approx = jnp.concatenate(
        [jnp.diff(stateHist.v_W, axis=0) / dt, jnp.zeros((1, 3), dtype=jnp.float32)],
        axis=0,
    )  # (N, 3)
    z_accel_true = jax.vmap(
        lambda wxyz, a_W: jaxlie.SO3(wxyz).as_matrix().T @ (a_W + g_up_W)
    )(stateHist.R_WB.wxyz, a_W_approx)
    z_accel = z_accel_true + jnp.sqrt(R_accel) * jax.random.normal(
        k3, z_accel_true.shape
    )

    # Rear wheel encoders only (RL=index 2, RR=index 3)
    omega_rear = stateHist.omega_W[:, 2:]  # (N, 2)
    z_rear_wheels = omega_rear + jnp.sqrt(R_rear_wheels) * jax.random.normal(
        k4, omega_rear.shape
    )

    return z_gps, z_gyro, z_accel, z_rear_wheels


def run_ekf(
    model: AckermannCarModel,
    stateHist,
    logs,
    z_gps,
    z_gyro,
    z_accel,
    z_rear_wheels,
    R_gps_val: float,
    R_gyro_val: float,
    R_accel_val: float,
    R_rear_wheels_val: float,
    R_front_eq_rear_val: float,
    Q: Array,
    P0: Array,
    dt: float,
    start_idx: int = 0,
):
    """
    Run the EKF with sequential measurement updates (no front wheel RPM).

    Per step:  predict → GPS → gyro → accel → rear encoders
                       → FL=RL / FR=RR pseudo-measurement

    Args:
        R_front_eq_rear_val: Variance for the FL≈RL, FR≈RR soft constraint.
                             Smaller → tighter constraint.
        start_idx: Index into stateHist / measurements to initialise from.
    """
    delta_hist = logs["delta"][start_idx:]
    tau_w_hist = logs["tau_w"][start_idx:]

    z_gps_run = z_gps[start_idx:]
    z_gyro_run = z_gyro[start_idx:]
    z_accel_run = z_accel[start_idx:]
    z_rear_wheels_run = z_rear_wheels[start_idx:]

    R_gps_mat = R_gps_val * jnp.eye(3)
    R_gyro_mat = R_gyro_val * jnp.eye(3)
    R_accel_mat = R_accel_val * jnp.eye(3)
    R_rear_wheels_mat = R_rear_wheels_val * jnp.eye(2)
    R_front_eq_rear_mat = R_front_eq_rear_val * jnp.eye(2)

    x0 = jax.tree.map(lambda a: a[start_idx], stateHist)
    ekf0 = EKFState(x_nom=x0, P=P0)

    z_constraint = jnp.zeros(2)

    def _scan_body(ekf, k):
        u = AckermannCarInput(delta=delta_hist[k], tau_w=tau_w_hist[k])

        # Build h_accel here so it closes over the current u as a traced value.
        # _scan_body is traced once by jax.lax.scan, so h_accel is a single
        # Python function object for the whole scan — no recompilation per step.
        h_accel = make_h_accel(model, u)

        ekf = ekf_predict(model, ekf, u, Q, dt)
        ekf = ekf_update(ekf, z_gps_run[k], h_gps, R_gps_mat)
        ekf = ekf_update(ekf, z_gyro_run[k], h_gyro, R_gyro_mat)
        ekf = ekf_update(ekf, z_accel_run[k], h_accel, R_accel_mat)
        ekf = ekf_update(ekf, z_rear_wheels_run[k], h_rear_wheels, R_rear_wheels_mat)
        ekf = ekf_update(ekf, z_constraint, h_front_eq_rear, R_front_eq_rear_mat)

        return ekf, ekf

    N_run = z_gps_run.shape[0]
    _, ekf_hist = jax.lax.scan(_scan_body, ekf0, jnp.arange(N_run))
    return ekf_hist


# ── Plotting ─────────────────────────────────────────────────────────


def _sigma_band(P_hist, idx_slice):
    """Extract ±2σ from the covariance diagonal for a given state slice."""
    # P_hist: (N, ERROR_DIM, ERROR_DIM)
    diag = jax.vmap(lambda P: jnp.diag(P)[idx_slice])(P_hist)  # (N, dim)
    return 2.0 * jnp.sqrt(jnp.maximum(diag, 0.0))


def plot_ekf_vs_truth(
    logs,
    stateHist,
    ekf_hist,
    start_idx=0,
    title_prefix="EKF (no front RPM)",
    measurements=None,
    meas_stride=5,
):
    """
    Compare EKF estimates against ground truth.
    Plots both the overlay (truth vs EKF) and the error with ±2σ bounds.

    If *measurements* is provided it should be a dict with keys
    ``"gps"``, ``"gyro"``, ``"gravity"``, ``"rear_wheels"`` holding the
    full-length noisy measurement arrays.  They are sliced from
    *start_idx* and down-sampled by *meas_stride* before plotting.
    """
    t = logs["t"][start_idx:]
    truth_p = stateHist.p_W[start_idx:]
    truth_v = stateHist.v_W[start_idx:]
    truth_w = stateHist.w_B[start_idx:]
    truth_omega = stateHist.omega_W[start_idx:]

    ekf_p = ekf_hist.x_nom.p_W
    ekf_v = ekf_hist.x_nom.v_W
    ekf_w = ekf_hist.x_nom.w_B
    ekf_omega = ekf_hist.x_nom.omega_W

    # Prepare down-sampled measurement scatter data
    s = meas_stride
    t_s = t[::s]
    m_gps = measurements["gps"][start_idx:][::s] if measurements else None
    m_gyro = measurements["gyro"][start_idx:][::s] if measurements else None
    m_rear = measurements["rear_wheels"][start_idx:][::s] if measurements else None

    meas_kw = dict(
        s=8, alpha=0.35, zorder=1, color="C0", label="meas", edgecolors="none"
    )

    # ── Position: overlay + error with ±2σ ──
    sigma_p = _sigma_band(ekf_hist.P, _P_IDX["p_W"])
    fig, axes = plt.subplots(3, 2, sharex="col", figsize=(14, 8))
    for i, lbl in enumerate(["x", "y", "z"]):
        axes[i, 0].plot(t, truth_p[:, i], "k-", label="truth")
        axes[i, 0].plot(t, ekf_p[:, i], "r--", label="EKF")
        if m_gps is not None:
            axes[i, 0].scatter(t_s, m_gps[:, i], **meas_kw)
        axes[i, 0].set_ylabel(f"p_{lbl} [m]")
        axes[i, 0].legend(fontsize=8)
        err = ekf_p[:, i] - truth_p[:, i]
        axes[i, 1].plot(t, err, "b-", linewidth=0.8)
        axes[i, 1].fill_between(
            t, -sigma_p[:, i], sigma_p[:, i], alpha=0.25, color="r"
        )
        axes[i, 1].set_ylabel(f"Δp_{lbl} [m]")
        axes[i, 1].axhline(0, color="k", linewidth=0.3)
    axes[-1, 0].set_xlabel("t [s]")
    axes[-1, 1].set_xlabel("t [s]")
    axes[0, 0].set_title("Overlay")
    axes[0, 1].set_title("Error  ± 2σ")
    fig.suptitle(f"{title_prefix} — Position")
    fig.tight_layout()

    # ── Velocity: overlay + error with ±2σ  (no direct measurement) ──
    sigma_v = _sigma_band(ekf_hist.P, _P_IDX["v_W"])
    fig, axes = plt.subplots(3, 2, sharex="col", figsize=(14, 8))
    for i, lbl in enumerate(["vx", "vy", "vz"]):
        axes[i, 0].plot(t, truth_v[:, i], "k-", label="truth")
        axes[i, 0].plot(t, ekf_v[:, i], "r--", label="EKF")
        axes[i, 0].set_ylabel(f"{lbl} [m/s]")
        axes[i, 0].legend(fontsize=8)
        err = ekf_v[:, i] - truth_v[:, i]
        axes[i, 1].plot(t, err, "b-", linewidth=0.8)
        axes[i, 1].fill_between(
            t, -sigma_v[:, i], sigma_v[:, i], alpha=0.25, color="r"
        )
        axes[i, 1].set_ylabel(f"Δ{lbl} [m/s]")
        axes[i, 1].axhline(0, color="k", linewidth=0.3)
    axes[-1, 0].set_xlabel("t [s]")
    axes[-1, 1].set_xlabel("t [s]")
    axes[0, 0].set_title("Overlay")
    axes[0, 1].set_title("Error  ± 2σ")
    fig.suptitle(f"{title_prefix} — Velocity (world)")
    fig.tight_layout()

    # ── Angular velocity: overlay + error with ±2σ ──
    sigma_w = _sigma_band(ekf_hist.P, _P_IDX["w_B"])
    fig, axes = plt.subplots(3, 2, sharex="col", figsize=(14, 8))
    for i, lbl in enumerate(["wx", "wy", "wz"]):
        axes[i, 0].plot(t, truth_w[:, i], "k-", label="truth")
        axes[i, 0].plot(t, ekf_w[:, i], "r--", label="EKF")
        if m_gyro is not None:
            axes[i, 0].scatter(t_s, m_gyro[:, i], **meas_kw)
        axes[i, 0].set_ylabel(f"{lbl} [rad/s]")
        axes[i, 0].legend(fontsize=8)
        err = ekf_w[:, i] - truth_w[:, i]
        axes[i, 1].plot(t, err, "b-", linewidth=0.8)
        axes[i, 1].fill_between(
            t, -sigma_w[:, i], sigma_w[:, i], alpha=0.25, color="r"
        )
        axes[i, 1].set_ylabel(f"Δ{lbl} [rad/s]")
        axes[i, 1].axhline(0, color="k", linewidth=0.3)
    axes[-1, 0].set_xlabel("t [s]")
    axes[-1, 1].set_xlabel("t [s]")
    axes[0, 0].set_title("Overlay")
    axes[0, 1].set_title("Error  ± 2σ")
    fig.suptitle(f"{title_prefix} — Angular velocity (body)")
    fig.tight_layout()

    # ── Wheel speeds: overlay + error with ±2σ ──
    # All four wheels are shown, but only RL/RR were fused.
    sigma_om = _sigma_band(ekf_hist.P, _P_IDX["omega_W"])
    fig, axes = plt.subplots(4, 2, sharex="col", figsize=(14, 10))
    wheel_names = ["FL", "FR", "RL", "RR"]
    for i, name in enumerate(wheel_names):
        axes[i, 0].plot(t, truth_omega[:, i], "k-", label="truth")
        axes[i, 0].plot(t, ekf_omega[:, i], "r--", label="EKF")
        # Scatter rear-wheel measurements (indices 0,1 in z_rear_wheels)
        if m_rear is not None and i >= 2:
            axes[i, 0].scatter(t_s, m_rear[:, i - 2], **meas_kw)
        elif i < 2:
            axes[i, 0].text(
                0.02, 0.95, "no sensor", transform=axes[i, 0].transAxes,
                fontsize=8, color="gray", va="top"
            )
        axes[i, 0].set_ylabel(f"ω_{name} [rad/s]")
        axes[i, 0].legend(fontsize=8)
        err = ekf_omega[:, i] - truth_omega[:, i]
        axes[i, 1].plot(t, err, "b-", linewidth=0.8)
        axes[i, 1].fill_between(
            t, -sigma_om[:, i], sigma_om[:, i], alpha=0.25, color="r"
        )
        axes[i, 1].set_ylabel(f"Δω_{name} [rad/s]")
        axes[i, 1].axhline(0, color="k", linewidth=0.3)
    axes[-1, 0].set_xlabel("t [s]")
    axes[-1, 1].set_xlabel("t [s]")
    axes[0, 0].set_title("Overlay  (FL/FR: no sensor)")
    axes[0, 1].set_title("Error  ± 2σ")
    fig.suptitle(f"{title_prefix} — Wheel speeds")
    fig.tight_layout()

    # ── Roll, Pitch, Yaw: overlay + error with ±2σ ──
    def _roll_pitch(wxyz):
        R = jaxlie.SO3(wxyz).as_matrix()
        pitch = jnp.arcsin(-R[2, 0])
        roll = jnp.arctan2(R[2, 1], R[2, 2])
        return roll, pitch

    roll_true, pitch_true = jax.vmap(_roll_pitch)(stateHist.R_WB.wxyz[start_idx:])
    roll_ekf, pitch_ekf = jax.vmap(_roll_pitch)(ekf_hist.x_nom.R_WB.wxyz)
    yaw_true = jax.vmap(lambda q: jaxlie.SO3(q).compute_yaw_radians())(
        stateHist.R_WB.wxyz[start_idx:]
    )
    yaw_ekf = jax.vmap(lambda q: jaxlie.SO3(q).compute_yaw_radians())(
        ekf_hist.x_nom.R_WB.wxyz
    )
    sigma_theta = _sigma_band(ekf_hist.P, _P_IDX["theta"])  # (N, 3): roll=0, pitch=1, yaw=2

    fig, axes = plt.subplots(3, 2, sharex="col", figsize=(14, 8))
    for i, (name, true_ang, ekf_ang, sigma_col) in enumerate([
        ("roll",  roll_true,  roll_ekf,  0),
        ("pitch", pitch_true, pitch_ekf, 1),
        ("yaw",   yaw_true,   yaw_ekf,   2),
    ]):
        axes[i, 0].plot(t, jnp.rad2deg(true_ang), "k-", label="truth")
        axes[i, 0].plot(t, jnp.rad2deg(ekf_ang), "r--", label="EKF")
        axes[i, 0].set_ylabel(f"{name} [deg]")
        axes[i, 0].legend(fontsize=8)
        err = jnp.rad2deg(ekf_ang - true_ang)
        sigma_deg = jnp.rad2deg(sigma_theta[:, sigma_col])
        axes[i, 1].plot(t, err, "b-", linewidth=0.8)
        axes[i, 1].fill_between(t, -sigma_deg, sigma_deg, alpha=0.25, color="r")
        axes[i, 1].set_ylabel(f"Δ{name} [deg]")
        axes[i, 1].axhline(0, color="k", linewidth=0.3)
    axes[0, 0].set_title("Overlay")
    axes[0, 1].set_title("Error  ± 2σ")
    axes[-1, 0].set_xlabel("t [s]")
    axes[-1, 1].set_xlabel("t [s]")
    fig.suptitle(f"{title_prefix} — Roll, Pitch & Yaw")
    fig.tight_layout()

    plt.show()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    params = default_params()
    model = AckermannCarModel(params)

    dt = 0.01
    T_settle = 1.5
    T_straight = 2.0
    T_turn = 2.0
    v_cmd = 1.0
    delta_turn = 0.25

    x0 = default_state(z0=0.10)

    # Run the forward simulation to get ground-truth states
    out_L, states_out = run_case(
        model, x0, dt, T_settle, T_straight, T_turn, v_cmd, +delta_turn
    )

    # Skip the settle phase: contact-model relu makes the F Jacobian
    # unreliable during the ground-bounce transient.
    N_settle = int(T_settle / dt)

    # ── Sensor noise variances ──
    R_gps = 2.25           # GPS position    σ ≈ 1.5   [m]
    R_gyro = 1e-4          # gyroscope       σ ≈ 0.01  [rad/s]
    R_accel = 1e-2         # IMU accel       σ ≈ 0.1   [m/s²]  (vehicle dynamics now in model)
    R_rear_wheels = 1e-4   # rear encoders   σ ≈ 0.01  [rad/s]
    R_front_eq_rear = 1e-4 # FL≈RL/FR≈RR    σ ≈ 0.01  [rad/s]

    key = jax.random.PRNGKey(42)

    z_gps, z_gyro, z_accel, z_rear_wheels = generate_measurements(
        states_out, model.params.chassis.g, dt,
        R_gps, R_gyro, R_accel, R_rear_wheels, key,
    )

    N_run = z_gps.shape[0] - N_settle
    print(f"Skipping first {N_settle} steps (settle phase)")
    print(f"Running EKF on {N_run} steps (no front wheel RPM):")
    print(f"  GPS              σ = {jnp.sqrt(R_gps):.4f} m")
    print(f"  Gyro             σ = {jnp.sqrt(R_gyro):.4f} rad/s")
    print(f"  Accel            σ = {jnp.sqrt(R_accel):.4f} m/s²")
    print(f"  Rear wheels      σ = {jnp.sqrt(R_rear_wheels):.4f} rad/s")
    print(f"  FL≈RL / FR≈RR    σ = {jnp.sqrt(R_front_eq_rear):.4f} rad/s")

    # ── Run the EKF ──
    Q = 1e-6 * jnp.eye(ERROR_DIM)
    P0 = 1e-4 * jnp.eye(ERROR_DIM)

    ekf_kwargs = dict(
        model=model,
        stateHist=states_out,
        logs=out_L,
        z_gps=z_gps,
        z_gyro=z_gyro,
        z_accel=z_accel,
        z_rear_wheels=z_rear_wheels,
        R_gps_val=R_gps,
        R_gyro_val=R_gyro,
        R_accel_val=R_accel,
        R_rear_wheels_val=R_rear_wheels,
        R_front_eq_rear_val=R_front_eq_rear,
        Q=Q,
        P0=P0,
        dt=dt,
        start_idx=N_settle,
    )

    # First call: JIT compilation + execution
    t0 = time.perf_counter()
    ekf_hist = run_ekf(**ekf_kwargs)
    jax.block_until_ready(ekf_hist)
    t_compile = time.perf_counter() - t0

    # Second call: execution only (JIT cache warm)
    t0 = time.perf_counter()
    ekf_hist = run_ekf(**ekf_kwargs)
    jax.block_until_ready(ekf_hist)
    t_run = time.perf_counter() - t0

    print(f"\nTiming over {N_run} EKF steps:")
    print(f"  1st call (compile + run): {t_compile * 1e3:.1f} ms")
    print(f"  2nd call (run only):      {t_run * 1e3:.1f} ms  ({t_run / N_run * 1e6:.2f} µs/step)")

    # ── Metrics (post-settle only) ──
    truth_p = states_out.p_W[N_settle:]
    truth_v = states_out.v_W[N_settle:]
    pos_err = jnp.linalg.norm(ekf_hist.x_nom.p_W - truth_p, axis=-1)
    vel_err = jnp.linalg.norm(ekf_hist.x_nom.v_W - truth_v, axis=-1)
    print(f"\nEKF position RMSE:  {jnp.sqrt(jnp.mean(pos_err**2)):.6f} m")
    print(f"EKF position max:   {jnp.max(pos_err):.6f} m")
    print(f"EKF velocity RMSE:  {jnp.sqrt(jnp.mean(vel_err**2)):.6f} m/s")

    # ── Plot ──
    meas = {
        "gps": z_gps,
        "gyro": z_gyro,
        "rear_wheels": z_rear_wheels,
    }
    plot_ekf_vs_truth(
        out_L, states_out, ekf_hist,
        start_idx=N_settle, measurements=meas,
    )


if __name__ == "__main__":
    main()
