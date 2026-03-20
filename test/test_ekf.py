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


# ── IMU measurement functions ────────────────────────────────────────

def h_gyro(x: AckermannCarState) -> Array:
    """Gyroscope: directly measures body angular velocity w_B."""
    return x.w_B  # (3,)


def generate_imu_measurements(
    model: AckermannCarModel,
    stateHist,
    logs,
    R_gyro: float,
    R_accel: float,
    key,
):
    """
    Generate noisy IMU measurements from a state history.

    Pattern:
        z = h(x_true) + sqrt(R) * randn
    """
    key_g, key_a = jax.random.split(key)

    # ── Gyroscope: z_gyro = w_B + noise ──
    z_gyro_true = stateHist.w_B  # (N, 3)
    noise_gyro = jnp.sqrt(R_gyro) * jax.random.normal(key_g, z_gyro_true.shape)
    z_gyro = z_gyro_true + noise_gyro

    # ── Accelerometer: z_accel = R_BW @ (a_W - g_W) + noise ──
    delta_hist = logs["delta"]  # (N,)
    tau_w_hist = logs["tau_w"]  # (N, 4)

    g_W = jnp.array([0.0, 0.0, -model.params.chassis.g], dtype=jnp.float32)

    def _single_accel(p_W, wxyz, v_W, w_B, omega_W, delta, tau_w):
        x = AckermannCarState(
            p_W=p_W,
            R_WB=jaxlie.SO3(wxyz),
            v_W=v_W,
            w_B=w_B,
            omega_W=omega_W,
        )
        u = AckermannCarInput(delta=delta, tau_w=tau_w)
        xdot = model.xdot(x, u)
        sf_W = xdot.v_W - g_W
        R_BW = x.R_WB.as_matrix().T
        return R_BW @ sf_W

    z_accel_true = jax.vmap(_single_accel)(
        stateHist.p_W,
        stateHist.R_WB.wxyz,
        stateHist.v_W,
        stateHist.w_B,
        stateHist.omega_W,
        delta_hist,
        tau_w_hist,
    )  # (N, 3)

    noise_accel = jnp.sqrt(R_accel) * jax.random.normal(key_a, z_accel_true.shape)
    z_accel = z_accel_true + noise_accel

    return z_gyro, z_accel


def run_ekf_imu(
    model: AckermannCarModel,
    stateHist,
    logs,
    z_gyro,
    z_accel,
    R_gyro_val: float,
    R_accel_val: float,
    Q: Array,
    P0: Array,
    dt: float,
):
    """
    Run the EKF over the full trajectory using IMU (gyro + accel) measurements.

    At each step:
      1. Predict with current input u
      2. Update with gyroscope measurement
      3. Update with accelerometer measurement (sequential update)
    """
    N = z_gyro.shape[0]
    delta_hist = logs["delta"]
    tau_w_hist = logs["tau_w"]

    R_gyro_mat = R_gyro_val * jnp.eye(3)
    R_accel_mat = R_accel_val * jnp.eye(3)

    x0 = jax.tree.map(lambda a: a[0], stateHist)
    ekf0 = EKFState(x_nom=x0, P=P0)

    g_W = jnp.array([0.0, 0.0, -model.params.chassis.g], dtype=jnp.float32)

    def _scan_body(ekf, k):
        u = AckermannCarInput(delta=delta_hist[k], tau_w=tau_w_hist[k])

        # 1) Predict
        ekf_pred = ekf_predict(model, ekf, u, Q, dt)

        # 2) Gyro update
        ekf_g = ekf_update(ekf_pred, z_gyro[k], h_gyro, R_gyro_mat)

        # 3) Accel update (h depends on u at this step, so build a closure)
        def _h_accel(x):
            xdot = model.xdot(x, u)
            sf_W = xdot.v_W - g_W
            R_BW = x.R_WB.as_matrix().T
            return R_BW @ sf_W

        ekf_ga = ekf_update(ekf_g, z_accel[k], _h_accel, R_accel_mat)

        return ekf_ga, ekf_ga

    _, ekf_hist = jax.lax.scan(_scan_body, ekf0, jnp.arange(N))
    return ekf_hist


def plot_ekf_vs_truth(logs, stateHist, ekf_hist, title_prefix="EKF IMU"):
    """Compare EKF estimates against ground truth."""
    t = logs["t"]

    # ── Position ──
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    for i, (ax, lbl) in enumerate(zip(axes, ["x", "y", "z"])):
        ax.plot(t, stateHist.p_W[:, i], "k-", label=f"truth {lbl}")
        ax.plot(t, ekf_hist.x_nom.p_W[:, i], "r--", label=f"EKF {lbl}")
        ax.set_ylabel(f"p_{lbl} [m]")
        ax.legend()
    axes[-1].set_xlabel("t [s]")
    fig.suptitle(f"{title_prefix} — Position")

    # ── Velocity ──
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    for i, (ax, lbl) in enumerate(zip(axes, ["vx", "vy", "vz"])):
        ax.plot(t, stateHist.v_W[:, i], "k-", label=f"truth {lbl}")
        ax.plot(t, ekf_hist.x_nom.v_W[:, i], "r--", label=f"EKF {lbl}")
        ax.set_ylabel(f"{lbl} [m/s]")
        ax.legend()
    axes[-1].set_xlabel("t [s]")
    fig.suptitle(f"{title_prefix} — Velocity (world)")

    # ── Angular velocity (body) ──
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    for i, (ax, lbl) in enumerate(zip(axes, ["wx", "wy", "wz"])):
        ax.plot(t, stateHist.w_B[:, i], "k-", label=f"truth {lbl}")
        ax.plot(t, ekf_hist.x_nom.w_B[:, i], "r--", label=f"EKF {lbl}")
        ax.set_ylabel(f"{lbl} [rad/s]")
        ax.legend()
    axes[-1].set_xlabel("t [s]")
    fig.suptitle(f"{title_prefix} — Angular velocity (body)")

    # ── Yaw ──
    yaw_true = jax.vmap(lambda q: jaxlie.SO3(q).compute_yaw_radians())(
        stateHist.R_WB.wxyz
    )
    yaw_ekf = jax.vmap(lambda q: jaxlie.SO3(q).compute_yaw_radians())(
        ekf_hist.x_nom.R_WB.wxyz
    )
    plt.figure(figsize=(10, 4))
    plt.plot(t, jnp.rad2deg(yaw_true), "k-", label="truth yaw")
    plt.plot(t, jnp.rad2deg(yaw_ekf), "r--", label="EKF yaw")
    plt.xlabel("t [s]")
    plt.ylabel("yaw [deg]")
    plt.title(f"{title_prefix} — Yaw")
    plt.legend()

    # ── Position error norm ──
    pos_err = jnp.linalg.norm(ekf_hist.x_nom.p_W - stateHist.p_W, axis=-1)
    plt.figure(figsize=(10, 4))
    plt.plot(t, pos_err, "b-")
    plt.xlabel("t [s]")
    plt.ylabel("||p_ekf - p_true|| [m]")
    plt.title(f"{title_prefix} — Position error norm")

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

    # ── Generate noisy IMU measurements ──
    R_gyro = 0.001   # gyroscope noise variance
    R_accel = 0.01   # accelerometer noise variance
    key = jax.random.PRNGKey(42)

    z_gyro, z_accel = generate_imu_measurements(
        model, states_out, out_L, R_gyro, R_accel, key
    )

    print(f"Generated {z_gyro.shape[0]} IMU measurements")
    print(f"  z_gyro  shape: {z_gyro.shape}  (R = {R_gyro})")
    print(f"  z_accel shape: {z_accel.shape}  (R = {R_accel})")

    # ── Run the EKF ──
    Q = 1e-4 * jnp.eye(ERROR_DIM)
    P0 = 1e-3 * jnp.eye(ERROR_DIM)

    ekf_hist = run_ekf_imu(
        model=model,
        stateHist=states_out,
        logs=out_L,
        z_gyro=z_gyro,
        z_accel=z_accel,
        R_gyro_val=R_gyro,
        R_accel_val=R_accel,
        Q=Q,
        P0=P0,
        dt=dt,
    )

    # ── Position RMSE ──
    pos_err = jnp.linalg.norm(ekf_hist.x_nom.p_W - states_out.p_W, axis=-1)
    print(f"\nEKF position RMSE: {jnp.sqrt(jnp.mean(pos_err**2)):.6f} m")
    print(f"EKF position max error: {jnp.max(pos_err):.6f} m")

    # ── Plot ──
    plot_ekf_vs_truth(out_L, states_out, ekf_hist)


if __name__ == "__main__":
    main()
