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
    """State container for the error-state EKF.

    Holds the nominal trajectory state and the associated error-state covariance.
    Registered as a JAX pytree so it can be passed through :func:`jax.jit`
    and :func:`jax.lax.scan`.

    Attributes:
        x_nom (AckermannCarState): Nominal car state (the EKF's current best estimate).
        P (Array): Error-state covariance matrix of shape ``(ERROR_DIM, ERROR_DIM)``.
    """

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
    """Compute the discrete state-transition Jacobian F.

    Linearises the error dynamics around the current nominal state and input
    by computing ``d(error_dynamics) / d(dx)`` at ``dx = 0`` via
    :func:`jax.jacobian`.

    Args:
        model: Car physics model.
        x_nom: Nominal state (linearisation point).
        u: Control input.
        dt: Timestep [s].

    Returns:
        State-transition Jacobian of shape ``(ERROR_DIM, ERROR_DIM)``.
    """
    dx0 = jnp.zeros(ERROR_DIM)
    return jax.jacobian(
        lambda dx: error_dynamics(model, dx, x_nom, u, dt)
    )(dx0)

def compute_H(
    h: Callable[[AckermannCarState], Array],
    x_nom: AckermannCarState
) -> Array:
    """Compute the measurement Jacobian H in error-state coordinates.

    Differentiates ``h`` through :func:`~ackermann_jax.errorDyn.inject_error`
    so that **H** correctly accounts for the SO(3) perturbation convention.

    Args:
        h: Measurement function mapping a state to a measurement vector ``(m,)``.
        x_nom: Nominal state (linearisation point).

    Returns:
        Measurement Jacobian of shape ``(m, ERROR_DIM)``.
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
    """EKF predict step: propagate nominal state and covariance forward by dt.

    Computes the state-transition Jacobian **F** via automatic differentiation
    and updates the covariance as ``P = F P Fᵀ + Q``.

    Args:
        model: Car physics model.
        ekf: Current EKF state.
        u: Control input applied over the interval.
        Q: Process noise covariance of shape ``(ERROR_DIM, ERROR_DIM)``.
        dt: Timestep [s].

    Returns:
        Predicted :class:`EKFState` after propagation.
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
    """EKF update step: fuse a measurement into the current estimate.

    Computes the measurement Jacobian **H** via automatic differentiation and
    applies the Kalman correction. Uses the Joseph form for the covariance
    update to preserve symmetry and numerical stability:
    ``P = (I - KH) P (I - KH)ᵀ + K R Kᵀ``

    Args:
        ekf: Predicted EKF state (output of :func:`ekf_predict`).
        z: Measurement vector of shape ``(m,)``.
        h: Measurement function mapping a state to a vector of shape ``(m,)``.
            Must be a static (compile-time constant) Python callable.
        R: Measurement noise covariance of shape ``(m, m)``.

    Returns:
        Corrected :class:`EKFState` after fusing the measurement.
    """
    H = compute_H(h, ekf.x_nom)

    # compute innovation of EKF
    S = H @ ekf.P @ H.T + R
    K = ekf.P @ H.T @ jnp.linalg.inv(S)

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
    """Full EKF predict-then-update cycle for a single timestep.

    Convenience wrapper that calls :func:`ekf_predict` followed by
    :func:`ekf_update`.

    Args:
        model: Car physics model.
        ekf: Current EKF state.
        u: Control input for the predict step.
        z: Measurement vector of shape ``(m,)``.
        h: Measurement function (static callable). See :func:`ekf_update`.
        Q: Process noise covariance ``(ERROR_DIM, ERROR_DIM)``.
        R: Measurement noise covariance ``(m, m)``.
        dt: Timestep [s].

    Returns:
        Updated :class:`EKFState` after predict and measurement fusion.
    """
    ekf_pred = ekf_predict(model, ekf, u, Q, dt)
    ekf_upd = ekf_update(ekf_pred, z, h, R)
    return ekf_upd
