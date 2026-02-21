from __future__ import annotations
import math
from typing import Any, Dict, Tuple, Optional

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
            tau_max=tau_max,
            use_traction_limit=True
        )

        u = AckermannCarInput(delta=delta,tau_w=tau_w)
        x_next = model.step(x=x,u=u, dt=dt, method="semi_implicit_euler")

        # Log yaw and position
        yaw_next = yaw_from_R(x_next.R_WB)
        return (x_next, integ_next), (yaw_next, x_next.p_W, x_next.v_W, tau_w)

    ks = jnp.arange(N, dtype=jnp.int32)
    (xN, _integN), (yaw_hist, p_hist, v_hist, tau_hist) = jax.lax.scan(
        step_fn, (x0,jnp.array(0.0,dtype=jnp.float32)), ks
    )
    print(jnp.shape(xN))
    yaw_final = yaw_hist[-1]
    yaw_straight_end = yaw_hist[N_straight-1]

    # Expected yaw
    
if __name__ == "__main__":
    main()
