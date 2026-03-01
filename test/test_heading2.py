import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import jaxlie
import addcopyfighandler

from ackermann_jax import (default_params, default_state,
AckermannCarModel, AckermannCarInput)
# Helper function
def yaw_from_R(R_WB: jaxlie.SO3) -> jnp.ndarray:
    return R_WB.compute_yaw_radians()

def wrap_pi(a: jnp.float32) -> jnp.float32:
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi


def run_model(
    model,
    x0,
    dt=0.01,
    T_settle=1.5,
    T_straight=2.0,
    T_turn=2.0,
    v_cmd=1.0,
    delta_turn=0.25, # rad, left turn
    Kp_v=40.0,
    Ki_v=2.0,
    tau_max=2.0,
    integ_max=0.5,
    method="semi_implicit_euler"
):
    p = model.params
    N_settle = int(T_settle/dt)
    N_straight = int(T_straight/dt)
    N_turn = int(T_turn/dt)
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
            use_traction_limit=True
        )

        u = AckermannCarInput(delta=delta, tau_w=tau_w)
        x_next = model.step(x=x, u=u, dt=dt, method=method)

        yaw = yaw_from_R(x.R_WB)

        R = x.R_WB.as_matrix()
        v_B = R.T @ x.v_W
        v_X = v_B[0]

        log = {
            "p_W": x_next.p_W,
            "v_W": x_next.v_W,
            "v_X": v_X,
            "yaw": yaw,
            "tau_w": tau_w,
            "delta": delta,
            "v_cmd": v_ref
        }
        return (x_next, integ_next), log

    ks = jnp.arange(N,dtype=jnp.int32)
    (xN, integN), logs = jax.lax.scan(step_once, (x0, jnp.array(0.0,dtype=jnp.float32)), ks)

    # stack logs
    logs["t"] = jnp.arange(N) * dt
    logs["N_straight"] = N_straight
    logs["N_settle"] = N_settle
    logs["N_turn"] = N_turn
    return logs 

def build_expected(
    out,
    model,
    dt,
    T_settle,
    T_straight,
    v_cmd,
    delta_turn
):
    t = out["t"]
    p = out["p_W"]
    yaw = out["yaw"]

    N_settle = int(out["N_settle"])
    N_straight = int(out["N_straight"])
    N_turn = int(out["N_turn"])

    # anchor at end of settling time
    p0 = p[N_settle - 1]
    yaw0 = yaw[N_settle - 1]

    # time since settle
    tau = jnp.clip(t - T_settle, a_min=0.0)

    # straight segment
    t1 = jnp.clip(tau, a_min=0.0, a_max=T_straight)
    x1 = p0[0] + v_cmd * t1 * jnp.cos(yaw0)
    y1 = p0[1] + v_cmd * t1 * jnp.sin(yaw0)
    yaw1 = yaw0 + 0.0 * t1

    # Turn part (bicycle approximation)
    # yaw_rate = v/L * tan(delta)
    L = model.params.geom.L
    print(f"Length: {L}")
    print(f"Turn angle: {delta_turn}")
    w = v_cmd / L * jnp.tan(delta_turn)
    print(f"Turn rate: {w}")

    t2 = jnp.clip(tau - T_straight, 0.0, T_straight + out["N_turn"] * dt)

    # starting point of turn = end of straight
    xS = p0[0] + v_cmd * T_straight * jnp.cos(yaw0)
    yS = p0[1] + v_cmd * T_straight * jnp.sin(yaw0)
    yawS = yaw0

    # integrate constant yaw rate and speed
    yaw_exp = jnp.where(
        tau <= T_straight,
        yaw1,
        yawS + w * (tau - T_straight)
    )

    # radius
    Rturn = v_cmd / (w + 1e-9)

    # For constant-speed constant-curvature motion, parametric circle from (xS,yS)
    # Using heading based integration
    def turn_xy(t_turn):
        psi = yawS + w * t_turn
        # integrate v*cos(psi), v*sin(psi)
        # closed form
        x = xS + (v_cmd / w) * (jnp.sin(psi) - jnp.sin(yawS))
        y = yS - (v_cmd / w) * (jnp.cos(psi) - jnp.cos(yawS))
        return x, y

    x_turn, y_turn = turn_xy(jnp.clip(tau - T_straight, 0.0))

    x_exp = jnp.where(tau <= T_straight, x1, x_turn)
    y_exp = jnp.where(tau <= T_straight, y1, y_turn)

    return x_exp, y_exp, yaw_exp

def plot_results(out, model, dt, T_settle, T_straight, v_cmd, delta_turn):
    t = out["t"]
    p = out["p_W"]
    vX = out["v_X"]
    yaw = out["yaw"]
    tau_w = out["tau_w"]

    x_exp, y_exp, yaw_exp = build_expected(
        out, model, dt, T_settle, T_straight, v_cmd, delta_turn
    )

    # XY
    plt.figure()
    plt.plot(p[:,0], p[:,1], label="actual")
    plt.plot(x_exp, y_exp, "--", label="expected (straight+arc)")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Drop + straight + left turn: XY")
    plt.legend()

    # yaw
    plt.figure()
    plt.plot(t, jnp.rad2deg(wrap_pi(yaw)), label="yaw actual [deg]")
    plt.plot(t, jnp.rad2deg(wrap_pi(yaw_exp)), "--", label="yaw expected [deg]")
    plt.axvline(T_settle, linestyle="--")
    plt.axvline(T_settle + T_straight, linestyle="--")
    plt.xlabel("t [s]")
    plt.ylabel("yaw [deg]")
    plt.title("Yaw: straight then left turn")
    plt.legend()

    # torques
    plt.figure()
    plt.plot(t, tau_w[:,0], label="tau_FL")
    plt.plot(t, tau_w[:,1], label="tau_FR")
    plt.plot(t, tau_w[:,2], label="tau_RL")
    plt.plot(t, tau_w[:,3], label="tau_RR")
    plt.axvline(T_settle, linestyle="--")
    plt.axvline(T_settle + T_straight, linestyle="--")
    plt.xlabel("t [s]")
    plt.ylabel("tau [N*m]")
    plt.title("Wheel torques")
    plt.legend()

    plt.figure()
    plt.plot(t, vX, label="v_X actual")
    plt.plot(t, out["v_cmd"], "--", label="v_cmd")
    plt.axvline(T_settle, linestyle="--")
    plt.axvline(T_settle + T_straight, linestyle="--")
    plt.xlabel("t [s]")
    plt.ylabel("v_X [m/s]")
    plt.title("Longitudal velocity")
    plt.legend()
    plt.show()



def main():
    params = default_params()
    model = AckermannCarModel(params)

    dt = 0.01
    T_settle = 1.5
    T_straight = 2.0
    T_turn = 2.0

    v_cmd = 1.0
    delta_turn = 0.25 # rad (~14 deg), gentle left turn

    x0 = default_state(z0 = 0.10)

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
        integ_max=0.5
    )

    plot_results(out, model, dt, T_settle, T_straight, v_cmd, delta_turn)

if __name__ == '__main__':
    main()
