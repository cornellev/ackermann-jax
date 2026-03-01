# test_contact_settle.py
import jax.numpy as jnp
from ackermann_jax import default_params, default_state, AckermannCarModel, AckermannCarInput
from matplotlib import pyplot as plt
import jax

def main():
    params = default_params()
    # kill tire forces
    params = params.__class__(
        geom=params.geom,
        chassis=params.chassis,
        wheels=params.wheels,
        tires=params.tires.__class__(mu=0.0, C_kappa=0.0, C_alpha=0.0, eps_v=params.tires.eps_v, eps_force=params.tires.eps_force),
        contact=params.contact,
        motor=params.motor,
    )

    model = AckermannCarModel(params)
    x = default_state(z0=0.15)  # start above ground
    u = AckermannCarInput(delta=jnp.array(0.0, jnp.float32), tau_w=jnp.zeros((4,), jnp.float32))

    dt = 0.002
    T = 3.0 # seconds
    N = T/dt
    z_hist = []
    Fz_hist = []
    def step_fn(carry, k):
        x = carry 
        x = model.step(x, u, dt, method="semi_implicit_euler")
        D = model.diagnostics(x, u)
        z = x.p_W[2]
        Fz = D.Fz
        # if k % 500 == 0:
        #     print(k, "z=", float(x.p_W[2]), "sumFz=", float(jnp.sum(D.Fz)))

        return (x), (z, Fz)

    k_list = jnp.arange(N, dtype=jnp.int32)
    xN, hists = jax.lax.scan(step_fn, x, k_list)
    z_hist, Fz_hist = hists
    # At rest, sum normal ~ mg (ballpark)
    # plt.plot(z_hist)
    # plt.show()
    mg = params.chassis.mass * params.chassis.g
    print("final z:", float(x.p_W[2]), "sumFz:", float(jnp.sum(Fz_hist[-1])), "mg:", mg)
    assert jnp.sum(Fz_hist[-1]) > 0.5 * mg, "Contact did not support weight (sumFz too small)."
    assert jnp.sum(Fz_hist[-1]) < 2.0 * mg, "Contact supports way too much weight (likely stiff bounce)."

    # Plot tire forces in rear
    plt.figure()
    plt.plot(Fz_hist[:,2], label="Fz (RL)")
    plt.plot(Fz_hist[:,3], label="Fz (RR)")
    plt.plot(Fz_hist[:,2] - Fz_hist[:,3], label="Fz diff (RL-RR)")
    plt.legend()
    plt.title("Normal forces")
    plt.show()

if __name__ == "__main__":
    main()
