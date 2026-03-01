from __future__ import annotations
import math
from typing import Any, Dict, Tuple, Optional
from matplotlib import pyplot as plt

import jax
import jax.numpy as jnp
import jaxlie

from ackermann_jax import (
    default_params,
    default_state,
    AckermannCarModel,
    AckermannCarInput,
)

"""
Quick sanity test for `ackermann_car.py`

Goal:
- Drive straight for T_straight seconds (delta = 0)
- Then turn left for T_turn seconds (delta = + delta_turn)
- Validate:
    (1) Yaw stays approximately zero during straight segment
    (2) Yaw becomes *positive* during the left turn
    (3) Final yaw is close to the bicycle model expectation:
        yaw_exp = (v_cmd / L) * tan(delta_turn) * T_turn

"""

# Helper function
def yaw_from_R(R_WB: jaxlie.SO3) -> jnp.ndarray:
    """
    Extract yaw from a rotation matrix assuming ZYX (yaw-pitch-roll) convention.
    With body-> world rotation:
        yaw = atan2(R[1,0],R[0,0])
    """
    R = R_WB.as_matrix()
    return jnp.arctan2(R[1,0],R[0,0])


def main():
    dt = 0.01
    T_straight = 2.0
    T_turn = 2.0 # seconds

    N_straight = int(T_straight / dt) # Number of steps to take
    N_turn = int(T_turn / dt)
    N = N_straight + N_turn

    v_cmd = 1.0 # m/s

    delta_turn = jnp.deg2rad(15.0) # left turn

    # PI gains
    Kp_v = 40.0
    Ki_v = 2.0
    tau_max = 0.35 # N.m per wheel (clip)

    # Tolerances
    yaw_straight_tol = jnp.deg2rad(5.0)
    yaw_final_tol = jnp.deg2rad(20.0)
    # this allows a bit of tire slip / other real modeling stuff
    
    # Model initialize
    params = default_params()
    model = AckermannCarModel(params)
    params = params.__class__(
        geom=params.geom,
        chassis=params.chassis,
        wheels=params.wheels,
        tires=params.tires.__class__(mu=params.tires.mu, C_kappa=0.0, C_alpha=0.0, eps_v=params.tires.eps_v, eps_force=params.tires.eps_force),
        contact=params.contact.__class__(k_n=0.0,c_n=0.0,z0=params.contact.z0),
        motor=params.motor.__class__(has_motor=jnp.zeros(4),alpha=None),
    )

    x0 = default_state(z0=0.10)

    # Small helper: choose delta based on time index
    def delta_schedule(k: jnp.ndarray) -> jnp.ndarray:
        return jnp.where(k < N_straight, 0.0, delta_turn)

    # Do a lax.scan over steps to compute efficiently
    def step_fn(carry,k):
        x, integ = carry
        delta = delta_schedule(k)

        # Map chassis speed command to wheel torques via command
        tau_w, integ_next = model.map_velocity_to_wheel_torques(
            x=x,
            v_cmd=v_cmd,
            integral_state=integ,
            Kp=Kp_v,
            Ki=Ki_v,
            dt = dt,
            tau_max=tau_max,
            use_traction_limit=True
        )

        u = AckermannCarInput(delta=delta,tau_w=tau_w)
        x_next = model.step(x=x,u=u, dt=dt, method="euler")

        # Log yaw and position
        yaw_next = yaw_from_R(x_next.R_WB)
        return (x_next, integ_next), (yaw_next, x_next.p_W, x_next.v_W, tau_w)

    ks = jnp.arange(N, dtype=jnp.int32)
    (xN, _integN), (yaw_hist, p_hist, v_hist, tau_hist) = jax.lax.scan(
        step_fn, (x0,jnp.array(0.0,dtype=jnp.float32)), ks
    )
    print(jnp.shape(yaw_hist))
    yaw_final = yaw_hist[-1]
    yaw_straight_end = yaw_hist[N_straight-1]

    # Expected yaw
    L = params.geom.L
    yaw_expected = (v_cmd / L) * jnp.tan(delta_turn) * T_turn

    # Wrap to pi
    def wrap_pi(a):
        return (a + jnp.pi) % (2 * jnp.pi) - jnp.pi

    yaw_final_wrapped = wrap_pi(yaw_final)
    yaw_expected_wrapped = wrap_pi(yaw_expected)

    # -----------------------
    # Assertions (sanity)
    plt.plot(tau_hist[:,0], label="tau_FL")
    plt.plot(tau_hist[:,1], label="tau_FR")
    plt.plot(tau_hist[:,2], label="tau_RL")
    plt.plot(tau_hist[:,3], label="tau_RR")
    plt.legend()
    plt.show()

    plt.plot(p_hist[:,0], p_hist[:,1])
    plt.show()
    assert jnp.abs(p_hist[N_straight-1,1]) > 0, f"Distance not traveled, {p_hist[N_straight-1,1]}"
    #
    # -----------------------
    # 1) Should be roughly straight at end of straight segment
    assert jnp.abs(yaw_straight_end) < yaw_straight_tol, (
        f"Yaw drifted too much while going straight.\n"
        f"  yaw_end_straight = {float(yaw_straight_end):.3f} rad "
        f"({float(jnp.rad2deg(yaw_straight_end)):.1f} deg)\n"
        f"  tol = {float(yaw_straight_tol):.3f} rad"
    )

    # 2) Left turn should yield positive yaw
    assert yaw_final > 0.0, (
        f"Expected positive yaw after left turn, got yaw_final={float(yaw_final):.3f} rad"
    )

    # 3) Final yaw should be in the ballpark of bicycle model expectation
    yaw_err = wrap_pi(yaw_final_wrapped - yaw_expected_wrapped)
    assert jnp.abs(yaw_err) < yaw_final_tol, (
        f"Final yaw not close to expectation.\n"
        f"  yaw_final     = {float(yaw_final_wrapped):.3f} rad "
        f"({float(jnp.rad2deg(yaw_final_wrapped)):.1f} deg)\n"
        f"  yaw_expected  = {float(yaw_expected_wrapped):.3f} rad "
        f"({float(jnp.rad2deg(yaw_expected_wrapped)):.1f} deg)\n"
        f"  yaw_err       = {float(yaw_err):.3f} rad "
        f"({float(jnp.rad2deg(yaw_err)):.1f} deg)\n"
        f"  tol           = {float(yaw_final_tol):.3f} rad"
    )

    # Optional: also sanity-check that we moved forward and left (y should increase in a left turn)
    p_final = p_hist[-1]
    assert p_final[0] > 0.5, f"Did not move forward enough: x_final={float(p_final[0]):.3f} m"
    assert p_final[1] > 0.0, f"Expected positive lateral displacement after left turn: y_final={float(p_final[1]):.3f} m"

    print("✅ Trajectory sanity test passed.")
    print(f"  yaw_end_straight = {float(yaw_straight_end):.3f} rad ({float(jnp.rad2deg(yaw_straight_end)):.1f} deg)")
    print(f"  yaw_final        = {float(yaw_final_wrapped):.3f} rad ({float(jnp.rad2deg(yaw_final_wrapped)):.1f} deg)")
    print(f"  yaw_expected     = {float(yaw_expected_wrapped):.3f} rad ({float(jnp.rad2deg(yaw_expected_wrapped)):.1f} deg)")
    print(f"  p_final          = [{float(p_final[0]):.3f}, {float(p_final[1]):.3f}, {float(p_final[2]):.3f}] m")
    
if __name__ == "__main__":
    main()
