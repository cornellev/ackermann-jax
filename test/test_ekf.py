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


def h_wheels(x: AckermannCarState) -> Array:
    """Wheel encoders: measures wheel angular velocities of back wheels."""
    return x.omega_W[2:]  # (2,)


def make_h_gravity(g: float):
    """
    Gravity-in-body-frame measurement (simplified accelerometer).

    Models the accelerometer as reading only the gravity vector
    projected into the body frame:  h(x) = R_BW @ [0, 0, g]

    This gives roll/pitch (tilt) observability without touching
    the contact model, which is too nonlinear for the EKF to
    linearize reliably.
    """
    g_up_W = jnp.array([0.0, 0.0, g], dtype=jnp.float32)

    def h_gravity(x: AckermannCarState) -> Array:
        R_BW = x.R_WB.as_matrix().T
        return R_BW @ g_up_W  # (3,)

    return h_gravity


def generate_measurements(
    stateHist,
    g: float,
    R_gps: float,
    R_gyro: float,
    R_gravity: float,
    R_wheels: float,
    key,
):
    """
    Generate noisy sensor measurements from a ground-truth state history.

    Pattern:  z = h(x_true) + sqrt(R) * randn
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

    # Gravity (simplified accel) — R_BW @ [0,0,g]
    g_up_W = jnp.array([0.0, 0.0, g], dtype=jnp.float32)
    z_gravity_true = jax.vmap(
        lambda wxyz: jaxlie.SO3(wxyz).as_matrix().T @ g_up_W
    )(stateHist.R_WB.wxyz)
    z_gravity = z_gravity_true + jnp.sqrt(R_gravity) * jax.random.normal(
        k3, z_gravity_true.shape
    )

    # Wheel encoders — wheel angular velocities
    z_wheels = stateHist.omega_W + jnp.sqrt(R_wheels) * jax.random.normal(
        k4, stateHist.omega_W.shape
    )

    return z_gps, z_gyro, z_gravity, z_wheels


def run_ekf(
    model: AckermannCarModel,
    stateHist,
    logs,
    z_gps,
    z_gyro,
    z_gravity,
    z_wheels,
    R_gps_val: float,
    R_gyro_val: float,
    R_gravity_val: float,
    R_wheels_val: float,
    Q: Array,
    P0: Array,
    dt: float,
    start_idx: int = 0,
):
    """
    Run the EKF with sequential measurement updates.

    Per step:  predict → GPS → gyro → gravity → wheel encoders

    Args:
        start_idx: Index into stateHist / measurements to initialise from.
                   Use this to skip transient phases (e.g. ground-contact
                   settle) where the contact model's relu discontinuity
                   makes the F Jacobian unreliable.
    """
    delta_hist = logs["delta"][start_idx:]
    tau_w_hist = logs["tau_w"][start_idx:]

    z_gps_run = z_gps[start_idx:]
    z_gyro_run = z_gyro[start_idx:]
    z_gravity_run = z_gravity[start_idx:]
    z_wheels_run = z_wheels[start_idx:]

    R_gps_mat = R_gps_val * jnp.eye(3)
    R_gyro_mat = R_gyro_val * jnp.eye(3)
    R_gravity_mat = R_gravity_val * jnp.eye(3)
    R_wheels_mat = R_wheels_val * jnp.eye(2)

    h_gravity = make_h_gravity(model.params.chassis.g)

    x0 = jax.tree.map(lambda a: a[start_idx], stateHist)
    ekf0 = EKFState(x_nom=x0, P=P0)

    def _scan_body(ekf, k):
        u = AckermannCarInput(delta=delta_hist[k], tau_w=tau_w_hist[k])

        ekf = ekf_predict(model, ekf, u, Q, dt)
        ekf = ekf_update(ekf, z_gps_run[k], h_gps, R_gps_mat)
        ekf = ekf_update(ekf, z_gyro_run[k], h_gyro, R_gyro_mat)
        ekf = ekf_update(ekf, z_gravity_run[k], h_gravity, R_gravity_mat)
        ekf = ekf_update(ekf, z_wheels_run[k,2:4], h_wheels, R_wheels_mat)

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
    title_prefix="EKF",
    measurements=None,
    meas_stride=5,
):
    """
    Compare EKF estimates against ground truth.
    Plots both the overlay (truth vs EKF) and the error with ±2σ bounds.

    If *measurements* is provided it should be a dict with keys
    ``"gps"``, ``"gyro"``, ``"gravity"``, ``"wheels"`` holding the
    full-length noisy measurement arrays.  They are sliced from
    *start_idx* and down-sampled by *meas_stride* before plotting as
    scatter points on the overlay panels.
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
    m_wheels = measurements["wheels"][start_idx:][::s] if measurements else None

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
    sigma_om = _sigma_band(ekf_hist.P, _P_IDX["omega_W"])
    fig, axes = plt.subplots(4, 2, sharex="col", figsize=(14, 10))
    wheel_names = ["FL", "FR", "RL", "RR"]
    for i, name in enumerate(wheel_names):
        axes[i, 0].plot(t, truth_omega[:, i], "k-", label="truth")
        axes[i, 0].plot(t, ekf_omega[:, i], "r--", label="EKF")
        if m_wheels is not None:
            axes[i, 0].scatter(t_s, m_wheels[:, i], **meas_kw)
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
    axes[0, 0].set_title("Overlay")
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
    R_gps = 2.25       # GPS position  σ ≈ 1.5    [m]
    R_gyro = 1e-4      # gyroscope     σ ≈ 0.01   [rad/s]
    R_gravity = 1e-2   # accel/gravity σ ≈ 0.1    [m/s²]
    R_wheels = 1e-4    # wheel encoder σ ≈ 0.01   [rad/s]

    key = jax.random.PRNGKey(42)

    z_gps, z_gyro, z_gravity, z_wheels = generate_measurements(
        states_out, model.params.chassis.g,
        R_gps, R_gyro, R_gravity, R_wheels, key,
    )

    N_run = z_gps.shape[0] - N_settle
    print(f"Skipping first {N_settle} steps (settle phase)")
    print(f"Running EKF on {N_run} steps:")
    print(f"  GPS       σ = {jnp.sqrt(R_gps):.4f} m")
    print(f"  Gyro      σ = {jnp.sqrt(R_gyro):.4f} rad/s")
    print(f"  Gravity   σ = {jnp.sqrt(R_gravity):.4f} m/s²")
    print(f"  Wheels    σ = {jnp.sqrt(R_wheels):.4f} rad/s")

    # ── Run the EKF ──
    Q = 1e-6 * jnp.eye(ERROR_DIM)
    P0 = 1e-4 * jnp.eye(ERROR_DIM)

    ekf_kwargs = dict(
        model=model,
        stateHist=states_out,
        logs=out_L,
        z_gps=z_gps,
        z_gyro=z_gyro,
        z_gravity=z_gravity,
        z_wheels=z_wheels,
        R_gps_val=R_gps,
        R_gyro_val=R_gyro,
        R_gravity_val=R_gravity,
        R_wheels_val=R_wheels,
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
        "gravity": z_gravity,
        "wheels": z_wheels,
    }
    plot_ekf_vs_truth(
        out_L, states_out, ekf_hist,
        start_idx=N_settle, measurements=meas,
    )


if __name__ == "__main__":
    main()
