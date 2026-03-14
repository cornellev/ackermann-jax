from __future__ import annotations
import addcopyfighandler
import jax
import jax.numpy as jnp
# from jax import Array
import jaxlie
from matplotlib import pyplot as plt

from ackermann_jax import (
    default_params,
    default_state,
    AckermannCarModel,
    AckermannCarInput
)


def yaw_from_R(R_WB: jaxlie.SO3) -> jnp.ndarray:
    return R_WB.compute_yaw_radians()

def wrap_pi(a: jnp.float) -> jnp.float32:
    return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi

def main():
    # Timings
    dt = 0.01
    T_settle = 1.5 # for bouncing behavior
    T_drive = 2.0
    N_settle = int(T_settle/dt)
    N_drive = int(T_drive/dt)
    N = N_settle + N_drive

    # Commands
    v_cmd_drive = 1.0 # m/s (straight)
    delta_cmd = 0.0


    # Velocity torque PI
    Kp_v = 40.0
    Ki_v = 2.0
    tau_max = 0.35

    x0 = default_state(z0=0.15)

    params = default_params()
    model = AckermannCarModel(params)

    def v_cmd_schedule(k: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(k < N_settle, 0.0, v_cmd_drive)

    # JAX LAX
    def step_fn(carry, k):
        x, integ = carry

        v_cmd = v_cmd_schedule(k)

        # Compute wheel torques
        tau_w, integ_next = model.map_velocity_to_wheel_torques(
            x=x,
            v_cmd=v_cmd,
            integral_state=integ,
            Kp=Kp_v,
            Ki=Ki_v,
            dt=dt,
            tau_max=tau_max,
            use_traction_limit=True
        )


        u = AckermannCarInput(delta=delta_cmd, tau_w=tau_w)

        x_next = model.step(x=x, u=u, dt=dt, method="semi_implicit_euler")
        debug = model.diagnostics(x,u)

        # Log state
        yaw = yaw_from_R(x_next.R_WB)
        p = x_next.p_W
        v = x_next.v_W
        Fx = debug.Fx
        Fz = debug.Fz
        omega = x.omega_W
        kappa = debug.kappa

        # compute velocity in two different ways
        R = x.R_WB.as_matrix()
        r_B = params.geom.wheel_contact_points_body()
        r_W = (R @ r_B.T).T

        vA = x.v_W[None,:] + (R @ jnp.cross(x.w_B[None,:], r_B).T).T
        w_W = (R @ x.w_B)

        dv = vA[2] - vA[3]   # RL - RR (3,)
        dr = r_W[2] - r_W[3] # RL - RR (3,)

        rhs = jnp.cross(w_W, dr)
        kin_err = dv - rhs
        err_norm = jnp.linalg.norm(kin_err)


        return (x_next, integ_next), (p, v, tau_w, yaw, Fx, Fz, omega, kappa, err_norm)

    k_list = jnp.arange(N,dtype=jnp.int32)
    (xN, integN), hists = jax.lax.scan(step_fn, (x0, jnp.array(0.0,dtype=jnp.float32)), k_list)

    (p_hist, v_hist, tau_hist, yaw_hist, Fx_hist, Fz_hist, omega_hist, kappa_hist, errHist) = hists

    t = jnp.arange(N) * dt

    p_settle = p_hist[N_settle-1]
    t_drive = jnp.clip(t - T_settle, a_min=0.0)

    x_exp = p_settle[0] + v_cmd_drive * t_drive
    y_exp = p_settle[1] + 0.0 * t_drive
    yaw_exp = jnp.zeros_like(t)

    # --------- Plotting ---------

    plt.figure()
    plt.plot(p_hist[:,0], p_hist[:,1], label="actual")
    plt.plot(x_exp, y_exp, "--", label="expected (straight)")
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.title("Drop + straight: XY")
    plt.show()

    print(f"Final Position: {p_hist[-1,:]}")

    # yaw vs expected
    plt.figure()
    plt.plot(t, jnp.rad2deg(wrap_pi(yaw_hist)), label="yaw actual [deg]")
    plt.plot(t, jnp.rad2deg(wrap_pi(yaw_exp)), "--", label="yaw expected [deg]")
    plt.axvline(T_settle, linestyle="--")
    plt.xlabel("t [s]")
    plt.ylabel("yaw [deg]")
    plt.legend()
    plt.title("Yaw (should stay ~0 after settle)")
    plt.show()

    # wheel torques
    plt.figure()
    plt.plot(tau_hist[:,0], label="tau_FL")
    plt.plot(tau_hist[:,1], label="tau_FR")
    plt.plot(tau_hist[:,2], label="tau_RL")
    plt.plot(tau_hist[:,3], label="tau_RR")
    plt.axvline(N_settle, linestyle="--")
    plt.legend()
    plt.title("Wheel torques")
    plt.show()

    # wheel normal forces 
    plt.figure()
    plt.plot(Fx_hist[:,0], label="F_x(FL)")
    plt.plot(Fx_hist[:,1], label="F_x(FR)")
    plt.plot(Fx_hist[:,2], label="F_x(RL)")
    plt.plot(Fx_hist[:,3], label="F_x(RR)")
    plt.axvline(N_settle, linestyle="--")
    plt.legend()
    plt.title("Wheel longitudal forces")
    plt.show()

    plt.figure()
    plt.plot(Fz_hist[:,0], label="F_z(FL)")
    plt.plot(Fz_hist[:,1], label="F_z(FR)")
    plt.plot(Fz_hist[:,2], label="F_z(RL)")
    plt.plot(Fz_hist[:,3], label="F_z(RR)")
    plt.axvline(N_settle, linestyle="--")
    plt.legend()
    plt.title("Wheel normal forces")
    plt.show()

   

    # Difference of front normal forces
    plt.figure()
    plt.plot(Fx_hist[:,2], label="F_x(RL)")
    plt.plot(Fx_hist[:,3], label="F_x(RR)")
    plt.plot(Fx_hist[:,2] - Fx_hist[:,3], label="F_x(RL) - F_x(RR)")
    plt.axvline(N_settle, linestyle="--")
    plt.legend()
    plt.title("Wheel longitudal force differences (rear)")
    plt.show()

    # Kappa plot per wheel
    # plt.figure()
    # plt.plot(Fx_hist[:,0], label="F_x(FL)")
    # plt.plot(Fx_hist[:,1], label="F_x(FR)")
    # plt.plot(Fx_hist[:,2], label="F_x(RL)")
    # plt.plot(Fx_hist[:,3], label="F_x(RR)")
    # plt.axvline(N_settle, linestyle="--")
    # plt.legend()
    # plt.title("Wheel longitudal forces")
    # plt.show()

if __name__ == "__main__":
    main()
