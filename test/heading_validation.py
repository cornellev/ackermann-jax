import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import jaxlie
import addcopyfighandler

from ackermann_jax import (
    default_params,
    default_state,
    AckermannCarModel,
    AckermannCarInput,
)


def yaw_from_R(R_WB: jaxlie.SO3) -> jnp.ndarray:
    return R_WB.compute_yaw_radians()


def wrap_pi(a: jnp.ndarray) -> jnp.ndarray:
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def finite_diff(x, dt):
    dx = jnp.zeros_like(x)
    dx = dx.at[1:-1].set((x[2:] - x[:-2]) / (2.0 * dt))
    dx = dx.at[0].set((x[1] - x[0]) / dt)
    dx = dx.at[-1].set((x[-1] - x[-2]) / dt)
    return dx


def run_model(
    model,
    x0,
    dt=0.01,
    T_settle=1.5,
    T_straight=2.0,
    T_turn=2.0,
    v_cmd=1.0,
    delta_turn=0.25,
    Kp_v=10.0,
    Ki_v=4.0,
    tau_max=2.0,
    integ_max=0.5,
    method="semi_implicit_euler",
):
    N_settle = int(T_settle / dt)
    N_straight = int(T_straight / dt)
    N_turn = int(T_turn / dt)
    N = N_settle + N_straight + N_turn

    def v_cmd_schedule(k):
        return jnp.where(k < N_settle, 0.0, v_cmd)

    def delta_schedule(k):
        return jnp.where(k < (N_settle + N_straight), 0.0, delta_turn)

    def step_once(carry, k):
        x, integ = carry

        v_ref = v_cmd_schedule(k)
        delta = delta_schedule(k)

        tau_w, integ_next = model.map_velocity_to_wheel_torques(
            x=x,
            v_cmd=v_ref,
            integral_state=integ,
            dt=dt,
            Kp=Kp_v,
            Ki=Ki_v,
            tau_max=tau_max,
            integ_max=integ_max,
            use_traction_limit=True,
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
            "v_cmd": v_ref,
        }
        return (x_next, integ_next), log

    ks = jnp.arange(N, dtype=jnp.int32)
    (_, _), logs = jax.lax.scan(
        step_once,
        (x0, jnp.array(0.0, dtype=jnp.float32)),
        ks,
    )

    logs["t"] = jnp.arange(N) * dt
    logs["N_settle"] = N_settle
    logs["N_straight"] = N_straight
    logs["N_turn"] = N_turn
    return logs


def build_expected(out, model, dt, T_settle, T_straight, v_cmd, delta_turn):
    t = out["t"]
    p = out["p_W"]
    yaw = out["yaw"]

    N_settle = int(out["N_settle"])
    N_turn = int(out["N_turn"])

    p0 = p[N_settle - 1]
    yaw0 = yaw[N_settle - 1]

    tau = jnp.clip(t - T_settle, a_min=0.0)

    t1 = jnp.clip(tau, a_min=0.0, a_max=T_straight)
    x1 = p0[0] + v_cmd * t1 * jnp.cos(yaw0)
    y1 = p0[1] + v_cmd * t1 * jnp.sin(yaw0)
    yaw1 = yaw0 + 0.0 * t1

    L = model.params.geom.L
    yaw_rate_exp = v_cmd / L * jnp.tan(delta_turn)

    xS = p0[0] + v_cmd * T_straight * jnp.cos(yaw0)
    yS = p0[1] + v_cmd * T_straight * jnp.sin(yaw0)
    yawS = yaw0

    yaw_exp = jnp.where(
        tau <= T_straight,
        yaw1,
        yawS + yaw_rate_exp * (tau - T_straight),
    )

    eps = 1e-9
    t_turn = jnp.clip(tau - T_straight, 0.0, N_turn * dt)
    psi = yawS + yaw_rate_exp * t_turn

    x_turn = xS + (v_cmd / (yaw_rate_exp + eps)) * (jnp.sin(psi) - jnp.sin(yawS))
    y_turn = yS - (v_cmd / (yaw_rate_exp + eps)) * (jnp.cos(psi) - jnp.cos(yawS))

    x_exp = jnp.where(tau <= T_straight, x1, x_turn)
    y_exp = jnp.where(tau <= T_straight, y1, y_turn)

    return x_exp, y_exp, yaw_exp, yaw_rate_exp


def fit_circle_kasa(x, y):
    """
    Simple algebraic circle fit:
        x^2 + y^2 + A x + B y + C = 0
    """
    A = jnp.stack([x, y, jnp.ones_like(x)], axis=1)
    b = -(x**2 + y**2)
    sol, *_ = jnp.linalg.lstsq(A, b, rcond=None)

    Acoef, Bcoef, Ccoef = sol
    xc = -Acoef / 2.0
    yc = -Bcoef / 2.0
    R = jnp.sqrt(jnp.maximum((xc**2 + yc**2 - Ccoef), 1e-12))
    return xc, yc, R


def compute_validation_metrics(out, model, dt, T_settle, T_straight, v_cmd, delta_turn):
    # t = out["t"]
    p = out["p_W"]
    # vX = out["v_X"]
    # vY = out["v_Y"]
    yaw = out["yaw"]

    x_exp, y_exp, yaw_exp, yaw_rate_exp = build_expected(
        out, model, dt, T_settle, T_straight, v_cmd, delta_turn
    )

    N_settle = int(out["N_settle"])
    N_straight = int(out["N_straight"])
    N_turn = int(out["N_turn"])

    i0 = N_settle
    i1 = N_settle + N_straight
    i2 = i1 + N_turn

    straight_idx = slice(i0, i1)
    # turn_idx = slice(i1, i2)

    # Straight segment metrics
    y_straight = p[straight_idx, 1]
    yaw_straight = wrap_pi(yaw[straight_idx] - yaw[i0])
    straight_y_drift_max = jnp.max(jnp.abs(y_straight - y_straight[0]))
    straight_yaw_drift_deg_max = jnp.rad2deg(jnp.max(jnp.abs(yaw_straight)))

    # Turn metrics
    yaw_unwrapped = jnp.unwrap(yaw)
    yaw_rate = finite_diff(yaw_unwrapped, dt)

    # Use second half of turn for "steady-state-ish" comparison
    turn_mid = i1 + N_turn // 2
    steady_idx = slice(turn_mid, i2)

    yaw_rate_mean = jnp.mean(yaw_rate[steady_idx])
    yaw_rate_err_pct = 100.0 * (yaw_rate_mean - yaw_rate_exp) / (jnp.abs(yaw_rate_exp) + 1e-9)

    speed_mag = jnp.linalg.norm(out["v_W"][steady_idx, :2], axis=1)
    speed_mean = jnp.mean(speed_mag)

    R_kin = model.params.geom.L / (jnp.tan(delta_turn) + 1e-9)
    R_from_v_over_r = speed_mean / (jnp.abs(yaw_rate_mean) + 1e-9)

    # Geometric fit on second half of turn
    x_turn = p[steady_idx, 0]
    y_turn = p[steady_idx, 1]
    _, _, R_fit = fit_circle_kasa(x_turn, y_turn)

    radius_err_fit_pct = 100.0 * (R_fit - jnp.abs(R_kin)) / (jnp.abs(R_kin) + 1e-9)
    radius_err_vr_pct = 100.0 * (R_from_v_over_r - jnp.abs(R_kin)) / (jnp.abs(R_kin) + 1e-9)

    # Path/yaw tracking errors over the full motion after settle
    x_err = p[:, 0] - x_exp
    y_err = p[:, 1] - y_exp
    pos_err = jnp.sqrt(x_err**2 + y_err**2)
    yaw_err = wrap_pi(yaw - yaw_exp)

    pos_rmse = jnp.sqrt(jnp.mean(pos_err[i0:] ** 2))
    pos_max = jnp.max(pos_err[i0:])
    yaw_rmse_deg = jnp.rad2deg(jnp.sqrt(jnp.mean(yaw_err[i0:] ** 2)))
    yaw_final_err_deg = jnp.rad2deg(wrap_pi(yaw[i2 - 1] - yaw_exp[i2 - 1]))

    metrics = {
        "straight_y_drift_max_m": float(straight_y_drift_max),
        "straight_yaw_drift_deg_max": float(straight_yaw_drift_deg_max),
        "yaw_rate_expected_rad_s": float(yaw_rate_exp),
        "yaw_rate_actual_mean_rad_s": float(yaw_rate_mean),
        "yaw_rate_error_pct": float(yaw_rate_err_pct),
        "R_kinematic_m": float(jnp.abs(R_kin)),
        "R_from_v_over_r_m": float(R_from_v_over_r),
        "R_fit_m": float(R_fit),
        "R_error_fit_pct": float(radius_err_fit_pct),
        "R_error_v_over_r_pct": float(radius_err_vr_pct),
        "path_rmse_m": float(pos_rmse),
        "path_max_err_m": float(pos_max),
        "yaw_rmse_deg": float(yaw_rmse_deg),
        "yaw_final_err_deg": float(yaw_final_err_deg),
    }

    aux = {
        "x_exp": x_exp,
        "y_exp": y_exp,
        "yaw_exp": yaw_exp,
        "yaw_rate": yaw_rate,
        "pos_err": pos_err,
    }
    return metrics, aux


def print_validation_report(name, metrics, thresholds=None):
    if thresholds is None:
        thresholds = {
            "straight_y_drift_max_m": 0.03,
            "straight_yaw_drift_deg_max": 1.0,
            "yaw_rate_error_pct": 15.0,
            "R_error_fit_pct": 15.0,
            "path_rmse_m": 0.10,
            "yaw_final_err_deg": 10.0,
        }

    print("\n" + "=" * 72)
    print(f"VALIDATION REPORT: {name}")
    print("=" * 72)

    for k, v in metrics.items():
        print(f"{k:30s}: {v: .6f}")

    print("\nPass/fail:")
    for k, lim in thresholds.items():
        val = abs(metrics[k])
        ok = val <= lim
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {k:30s} | value={val:.6f}, limit={lim:.6f}")


def plot_validation(out, aux, T_settle, T_straight, title_prefix=""):
    t = out["t"]
    p = out["p_W"]
    yaw = out["yaw"]
    tau_w = out["tau_w"]
    vX = out["v_X"]

    x_exp = aux["x_exp"]
    y_exp = aux["y_exp"]
    yaw_exp = aux["yaw_exp"]
    yaw_rate = aux["yaw_rate"]
    pos_err = aux["pos_err"]

    # XY
    plt.figure()
    plt.plot(p[:, 0], p[:, 1], label="actual")
    plt.plot(x_exp, y_exp, "--", label="expected (straight+arc)")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(f"{title_prefix} XY path")
    plt.legend()

    # Yaw
    plt.figure()
    plt.plot(t, jnp.rad2deg(wrap_pi(yaw)), label="yaw actual [deg]")
    plt.plot(t, jnp.rad2deg(wrap_pi(yaw_exp)), "--", label="yaw expected [deg]")
    plt.axvline(T_settle, linestyle="--")
    plt.axvline(T_settle + T_straight, linestyle="--")
    plt.xlabel("t [s]")
    plt.ylabel("yaw [deg]")
    plt.title(f"{title_prefix} yaw")
    plt.legend()

    # Yaw rate
    plt.figure()
    plt.plot(t, jnp.rad2deg(yaw_rate), label="yaw rate actual [deg/s]")
    plt.axvline(T_settle, linestyle="--")
    plt.axvline(T_settle + T_straight, linestyle="--")
    plt.xlabel("t [s]")
    plt.ylabel("yaw rate [deg/s]")
    plt.title(f"{title_prefix} yaw rate")
    plt.legend()

    # Position error
    plt.figure()
    plt.plot(t, pos_err, label="position error magnitude [m]")
    plt.axvline(T_settle, linestyle="--")
    plt.axvline(T_settle + T_straight, linestyle="--")
    plt.xlabel("t [s]")
    plt.ylabel("position error [m]")
    plt.title(f"{title_prefix} path error")
    plt.legend()

    # Wheel torques
    plt.figure()
    plt.plot(t, tau_w[:, 0], label="tau_FL")
    plt.plot(t, tau_w[:, 1], label="tau_FR")
    plt.plot(t, tau_w[:, 2], label="tau_RL")
    plt.plot(t, tau_w[:, 3], label="tau_RR")
    plt.axvline(T_settle, linestyle="--")
    plt.axvline(T_settle + T_straight, linestyle="--")
    plt.xlabel("t [s]")
    plt.ylabel("tau [N*m]")
    plt.title(f"{title_prefix} wheel torques")
    plt.legend()

    # Longitudinal speed
    plt.figure()
    plt.plot(t, vX, label="v_X actual")
    plt.plot(t, out["v_cmd"], "--", label="v_cmd")
    plt.axvline(T_settle, linestyle="--")
    plt.axvline(T_settle + T_straight, linestyle="--")
    plt.xlabel("t [s]")
    plt.ylabel("v_X [m/s]")
    plt.title(f"{title_prefix} longitudinal speed")
    plt.legend()

    plt.show()


def run_case(model, x0, dt, T_settle, T_straight, T_turn, v_cmd, delta_turn):
    out = run_model(
        model=model,
        x0=x0,
        dt=dt,
        T_settle=T_settle,
        T_straight=T_straight,
        T_turn=T_turn,
        v_cmd=v_cmd,
        delta_turn=delta_turn,
        Kp_v=40.0,
        Ki_v=2.0,
        tau_max=0.35,
        integ_max=0.5,
    )
    metrics, aux = compute_validation_metrics(
        out=out,
        model=model,
        dt=dt,
        T_settle=T_settle,
        T_straight=T_straight,
        v_cmd=v_cmd,
        delta_turn=delta_turn,
    )
    return out, metrics, aux


def compute_symmetry_metrics(out_left, out_right, T_settle):
    i0 = int(out_left["N_settle"])
    pL = out_left["p_W"][i0:, :]
    pR = out_right["p_W"][i0:, :]
    yawL = out_left["yaw"][i0:]
    yawR = out_right["yaw"][i0:]

    # For a symmetric system, right turn should be mirror of left turn:
    # x should match, y should be opposite, yaw should be opposite.
    x_rmse = jnp.sqrt(jnp.mean((pL[:, 0] - pR[:, 0]) ** 2))
    y_mirror_rmse = jnp.sqrt(jnp.mean((pL[:, 1] + pR[:, 1]) ** 2))
    yaw_mirror_rmse_deg = jnp.rad2deg(
        jnp.sqrt(jnp.mean((wrap_pi(yawL + yawR)) ** 2))
    )

    return {
        "sym_x_rmse_m": float(x_rmse),
        "sym_y_mirror_rmse_m": float(y_mirror_rmse),
        "sym_yaw_mirror_rmse_deg": float(yaw_mirror_rmse_deg),
    }


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

    # Left turn
    out_L, metrics_L, aux_L = run_case(
        model, x0, dt, T_settle, T_straight, T_turn, v_cmd, +delta_turn
    )
    print_validation_report("left turn", metrics_L)
    plot_validation(out_L, aux_L, T_settle, T_straight, title_prefix="Left turn")

    # Right turn symmetry check
    out_R, metrics_R, aux_R = run_case(
        model, x0, dt, T_settle, T_straight, T_turn, v_cmd, -delta_turn
    )
    print_validation_report("right turn", metrics_R)
    plot_validation(out_R, aux_R, T_settle, T_straight, title_prefix="Right turn")

    sym = compute_symmetry_metrics(out_L, out_R, T_settle)
    print("\n" + "=" * 72)
    print("SYMMETRY CHECK")
    print("=" * 72)
    for k, v in sym.items():
        print(f"{k:30s}: {v: .6f}")


if __name__ == "__main__":
    main()
