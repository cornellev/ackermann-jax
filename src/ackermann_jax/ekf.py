from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array

from .car import AckermannCarState, AckermannCarInput, AckermannCarModel
from .errorDyn import (
    AckermannCarErrorState,
    error_dynamics,
    inject_error,
    unpack_error_state,
    zero_error_state
)

ERROR_DIM = 16

# Create state container to properly x and P
@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EKFState:
    x_nom: AckermannCarState
    P: Array

    def tree_flatten(self):
        return (self.x_nom, self.P), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

# Jacobian helpers
def compute_F(
    model: AckermannCarModel,
    x_nom: AckermannCarState,
    u: AckermannCarInput,
    dt: float,
) -> Array:
    """
    State-transition Jacobian F = d(error_dynamics) / d(dx) at dx=0.

    Linearizes the discrete error dynamics around the current nominal state and input.
    Shape: (ERROR_DIM, ERROR_DIM)
    """
    dx0 = jnp.zeros(ERROR_DIM)
    return jax.jacobian(
        lambda dx: error_dynamics(model, dx, x_nom, u, dt)
    )(dx0)

def compute_H(
    h: Callable[[AckermannCarState], Array],
    x_nom: AckermannCarState
) -> Array:
    """
    Measurement Jacobian H in error-state coordinates.

    Differentiates h through `inject_error` so H correctly accounts
    for SO(3) perturbation convention.

    Shape: (m, ERROR_DIM)

    Args:
        h: measurement function (AckermannCarState) -> (m,)
        x_nom: current nominal state (linearization point)
    """
    def h_error(dx_vex: Array) -> Array:
        x_pert = inject_error(x_nom, unpack_error_state(dx_vex))
        return h(x_pert)

    dx0 = jnp.zeros(ERROR_DIM)
    return jax.jacobian(h_error)(dx0)

# Prediction and stuff
@jax.jit
def ekf_predict(
    model: AckermannCarModel,
    ekf: EKFState,
    u: AckermannCarInput,
    Q: Array, # Process noise
    dt: float
) -> EKFState:
    """
    Predict step: propagate nominal state and
    covariance forward by dt.
    """
    x_next = model.step(ekf.x_nom, u, dt)
    F = compute_F(model, ekf.x_nom, u, dt)
    P_next = F @ ekf.P @ F.T + Q
    return EKFState(x_next, P_next)

@partial(jax.jit, static_argnames=('h',))
def ekf_update(
    ekf: EKFState,
    z: Array,
    h: Callable[[AckermannCarState], Array],
    R: Array
) -> EKFState:
    """
    Update step: fuse measurement z using measurement function h.

    Uses the Joseph form for P to preserve symmetry and numerical stabillity:
        P = (I - KH) P (I- KH)^T + K R K^T

    Args:
        ekf: predicted EKF state (EKFState)
        z: (m,) measurement vector (Array)
        h: measurement function (AckermannCarState) -> (m,)
        R: (m, m) measurement noise covariance (Array)
    """
    H = compute_H(h, ekf.x_nom)

    # compute innovation of EKF
    S = H @ ekf.P @ H.T + R
    # get inverse of S via cholesky factorization for speed
    c, low = jax.scipy.linalg.cho_factor(S)
    I = jnp.eye(jnp.shape(S)[0]) #noqa
    s_inv = jax.scipy.linalg.cho_solve((c, low), I)
    
    # K = ekf.P @ H.T @ jnp.linalg.inv(S)
    K = ekf.P @ H.T @ s_inv

    innovation = z - h(ekf.x_nom)
    dx_vex = K @ innovation

    x_cor = inject_error(ekf.x_nom, unpack_error_state(dx_vex))
 
    # Joseph form of covariance update
    I_KH = jnp.eye(ERROR_DIM) - K @ H
    P_cor = I_KH @ ekf.P @ I_KH.T + K @ R @ K.T
    return EKFState(x_cor, P_cor)

def ekf_step(
    model: AckermannCarModel,
    ekf: EKFState,
    u: AckermannCarInput,
    z: Array,
    h: Callable[[AckermannCarState], Array],
    Q: Array,
    R: Array,
    dt: float
) -> EKFState:
    """
    Full predict & update cycle.
    """
    ekf_pred = ekf_predict(model, ekf, u, Q, dt)
    ekf_upd = ekf_update(ekf_pred, z, h, R)
    return ekf_upd
