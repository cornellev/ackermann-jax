import jax
import jax.numpy as jnp
import jaxlie

from ackermann_jax import (
    default_params,
    default_state,
    AckermannCarModel,
    AckermannCarInput,
    rotation_error,
    inject_rotation_error,
    inject_error,
    zero_error_state,
    AckermannCarErrorState,
    state_difference
)

# random rotation
key = jax.random.PRNGKey(0)
axis = jax.random.normal(key, (3,))
axis /= jnp.linalg.norm(axis)

angle = 0.3
R = jaxlie.SO3.exp(axis * angle) #rodrigues equation

# small perturbation
dtheta = jnp.array([0.02, -0.01, 0.03])

R2 = inject_rotation_error(R, dtheta)

# recover error
dtheta_est = rotation_error(R, R2)

print("Testing rotation perturbation \n")
print(f"True: {dtheta}")
print(f"Estimated: {dtheta_est}")
print(f"Error: {dtheta - dtheta_est}")

# state differnece -> inject_error -> identity
x = default_state()

dx = AckermannCarErrorState(
    dp_W    = jnp.array([0.1, -0.2, 0.05]),
    dtheta_B = jnp.array([0.01, -0.02, 0.005]),
    dv_W    = jnp.array([0.2, 0.0, -0.1]),
    dw_B    = jnp.array([0.01, 0.02, -0.01]),
    # domega_W removed: kinematic rolling assumption (omega_w = v_t / r_wheel)
)

# inject and recover
x2 = inject_error(x, dx)
dx_est = state_difference(x, x2)

print("Test 2: Small Perturbation Reocvery \n")
print(f"original: {dx}")
print(f"recovered: {dx_est}")
