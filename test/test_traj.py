from __future__ import annotations

import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
from ackermann_jax import (
    AckermannParams, CarState, CarControl,
    step_rk4, linearise,
    state_to_vec, vec_to_state, STATE_DIM,
)

jax.config.update("jax_enable_x64", True)

PARAMS = AckermannParams(wheelbase=0.257, mass=1.5, eps_v=1e-3)
DT     = 0.01  # [s]

# JIT Compile functions to run in real time
_step_jit      = jax.jit(step_rk4,   static_argnums=(3,))
_linearise_jit = jax.jit(linearise,  static_argnums=(3,))

def _simulate(state0: CarState, control: CarControl, n_steps: int) -> CarState:
    s = state0
    for _ in range(n_steps):
        s = _step_jit(s, control, PARAMS, DT)
    return s

def _state(px=0.0, py=0.0, vx=1.0, vy=0.0, theta=0.0) -> CarState:
    return CarState(
        p_W   = jnp.array([px, py]),
        v_W   = jnp.array([vx, vy]),
        theta = jnp.array(theta),
    )

def _ctrl(a=0.0, delta=0.0) -> CarControl:
    return CarControl(a=jnp.array(a), delta=jnp.array(delta))

def main():
    # Test sinusoidal trajectory and plot it
    v0 = 1.0
    T_sim = 10.0 # seconds
    # Generate sinusoidal control input for delta steering angle
    delta_traj = 0.2 * jnp.sin(jnp.linspace(0, 4 * jnp.pi, int(T_sim / DT)))
    # Generate JAX friendly step to pass into lax.scan()
    def step_main(x, carry):
        prev_state = x

if __name__ == '__main__':
    main()

