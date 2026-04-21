import jax
import jax.numpy as jnp
from ackermann_jax.car import AckermannCarModel, AckermannCarState, AckermannCarInput
from ackermann_jax.ekf import EKFState, ERROR_DIM

n = 100 # steps to look ahead
dt = 0.01 # should match ekf dt
n_iter = 100 # num of gradient descent iterations
lr = 0.001 # learning rate for gradient descent
u_dim = 5  # [delta, tau_w_fl, tau_w_fr, tau_w_rl, tau_w_rr] - based on AckermannCarInput

u_min = jnp.array([-180, -1.0, -1.0, -1.0, -1.0]) # min steering, throttle - incorrect values for now
u_max = jnp.array([180, 1.0, 1.0, 1.0, 1.0]) # max steering, throttle - incorrect values for now

Q = 0.01 * jnp.eye(ERROR_DIM) # penalise state tracking error
R = 0.01 * jnp.eye(u_dim) # penalise control effort
P = 0.1 * jnp.eye(ERROR_DIM) # terminal cost weight

def rollout(model, x0, u): # n future ackermann states
   def scan_func(carry, u_flat):
       x_next = model.step(carry, AckermannCarInput(delta=u_flat[0], tau_w=u_flat[1:]), dt)
       return x_next, x_next
   _, x_traj = jax.lax.scan(scan_func, x0, u)
   return x_traj

def state_error(x_pred, x_ref): # need this to calculate error because of different subcomponents in state
    dp = x_pred.p_W - x_ref.p_W
    dR = (x_ref.R_WB.inverse() @ x_pred.R_WB).log()
    dv = x_pred.v_W - x_ref.v_W
    dw = x_pred.w_B - x_ref.w_B
    do = x_pred.omega_W - x_ref.omega_W
    return jnp.concatenate([dp, dR, dv, dw, do])

def cost(u, model, x0, x_ref): # cost function to optimize over
    # using ackermann state constrains us so that the waypoints also have to be in same format
    # might change later once we know how waypoints are generated
    x_pred = rollout(model, x0, u)
    e_k = jax.vmap(state_error)(x_pred, x_ref)
    stage_cost = jnp.sum(jnp.einsum('ni,ij,nj->n', e_k, Q, e_k)) + jnp.sum(jnp.einsum('ni,ij,nj->n', u, R, u))
    terminal_cost = e_k[-1] @ P @ e_k[-1]
    total_cost = stage_cost + terminal_cost
    return total_cost

def warm_start(u_prev): # so that we don't have to recalculate future steps all over again
    u_prev = jnp.roll(u_prev, shift=-1, axis=0)
    u_prev = u_prev.at[-1].set(jnp.zeros(u_prev.shape[1]))
    return u_prev
    
def gradient_descent(model, x0, x_ref, u): # gradient descent
    grad = jax.grad(cost, argnums=0)(u, model, x0, x_ref)
    u = u - lr * grad
    u = jnp.clip(u, u_min, u_max)
    return u

def mpc_step(model, ekf_state, x_ref, U_prev): # actual step function
    x0 = ekf_state.x_nom
    u = warm_start(U_prev)
    u = jax.lax.fori_loop(0, n_iter, lambda i, u: gradient_descent(model, x0, x_ref, u), u)
    u0_flat = u[0]
    u0 = AckermannCarInput(delta = u0_flat[0], tau_w = u0_flat[1:])
    # u0 is input to car
    # u is warm start for next iteration
    return u0, u