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
    zero_error_state,
)

ERROR_DIM = 16
"""Dimension of the error-state vector used throughout the EKF.

The 16 components are laid out as:

.. list-table::
   :header-rows: 1
   :widths: 12 18 52 8

   * - Slice
     - Symbol
     - Quantity
     - Dim
   * - ``0:3``
     - :math:`\\delta p_W`
     - World-frame position error
     - 3
   * - ``3:6``
     - :math:`\\delta\\theta`
     - SO(3) rotation error (axis-angle)
     - 3
   * - ``6:9``
     - :math:`\\delta v_W`
     - World-frame velocity error
     - 3
   * - ``9:12``
     - :math:`\\delta\\omega_B`
     - Body-frame angular velocity error
     - 3
   * - ``12:16``
     - :math:`\\delta\\Omega`
     - Wheel angular velocity errors
     - 4
"""


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class EKFState:
    """State container for the error-state EKF.

    Bases: ``@dataclass(frozen=True)``,
    :func:`jax.tree_util.register_pytree_node_class`

    **Frozen** — fields cannot be mutated after construction; all EKF functions
    return a new :class:`EKFState` rather than modifying in place.
    **Pytree-registered** — instances are transparently handled by
    :func:`jax.jit` and :func:`jax.lax.scan` without wrapping in a dict or
    tuple.

    Holds the nominal trajectory state and the associated error-state
    covariance.

    Attributes:
        x_nom (AckermannCarState): Nominal car state (the EKF's current best
            estimate).
        P (Array): Error-state covariance matrix of shape
            :math:`16 \\times 16` (i.e. ``(ERROR_DIM, ERROR_DIM)``).
    """

    x_nom: AckermannCarState
    P: Array

    def tree_flatten(self):
        return (self.x_nom, self.P), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


# ── Jacobian helpers ──────────────────────────────────────────────────────────


def compute_F(
    model: AckermannCarModel,
    x_nom: AckermannCarState,
    u: AckermannCarInput,
    dt: float,
) -> Array:
    """Compute the discrete state-transition Jacobian **F**.

    Linearises the error dynamics around the current nominal state and input
    by evaluating :math:`\\partial f / \\partial \\delta x` at
    :math:`\\delta x = 0` via :func:`jax.jacobian`.

    Args:
        model: Car physics model.
        x_nom: Nominal state (linearisation point).
        u: Control input.
        dt: Timestep [s].

    Returns:
        State-transition Jacobian of shape :math:`16 \\times 16`.
    """
    dx0 = jnp.zeros(ERROR_DIM)
    return jax.jacobian(lambda dx: error_dynamics(model, dx, x_nom, u, dt))(dx0)


def compute_H(
    h: Callable[[AckermannCarState], Array],
    x_nom: AckermannCarState,
) -> Array:
    """Compute the measurement Jacobian **H** in error-state coordinates.

    Differentiates ``h`` through :func:`~ackermann_jax.errorDyn.inject_error`
    so that **H** correctly accounts for the SO(3) perturbation convention:

    .. math::

        H = \\left.
              \\frac{\\partial\\, h\\!\\left(
                \\mathrm{inject\\_error}(x_{\\mathrm{nom}},\\, \\delta x)
              \\right)}{\\partial\\, \\delta x}
            \\right|_{\\delta x = 0}

    Args:
        h: Measurement function mapping a state to a measurement vector
            of shape :math:`(m,)`.
        x_nom: Nominal state (linearisation point).

    Returns:
        Measurement Jacobian of shape :math:`m \\times 16`.
    """

    def h_error(dx_vex: Array) -> Array:
        x_pert = inject_error(x_nom, unpack_error_state(dx_vex))
        return h(x_pert)

    dx0 = jnp.zeros(ERROR_DIM)
    return jax.jacobian(h_error)(dx0)


# ── Core EKF steps ────────────────────────────────────────────────────────────


@jax.jit
def ekf_predict(
    model: AckermannCarModel,
    ekf: EKFState,
    u: AckermannCarInput,
    Q: Array,
    dt: float,
) -> EKFState:
    """EKF predict step: propagate nominal state and covariance forward by *dt*.

    Integrates the car model forward one step, computes **F** via automatic
    differentiation, and propagates the covariance:

    .. math::

        P^- = F \\, P \\, F^\\top + Q

    Args:
        model: Car physics model.
        ekf: Current EKF state.
        u: Control input applied over the interval.
        Q: Process noise covariance of shape :math:`16 \\times 16`.
        dt: Timestep [s].

    Returns:
        Predicted :class:`EKFState` after propagation.
    """
    x_next = model.step(ekf.x_nom, u, dt)
    F = compute_F(model, ekf.x_nom, u, dt)
    P_next = F @ ekf.P @ F.T + Q
    return EKFState(x_next, P_next)


@partial(jax.jit, static_argnames=("h",))
def ekf_update(
    ekf: EKFState,
    z: Array,
    h: Callable[[AckermannCarState], Array],
    R: Array,
) -> EKFState:
    """EKF update step: fuse a single measurement into the current estimate.

    Computes the measurement Jacobian :math:`H` via automatic differentiation
    and applies the Kalman correction.

    **Innovation covariance and Kalman gain:**

    .. math::

        S = H P H^\\top + R

    .. math::

        K = P H^\\top S^{-1}

    .. warning::
        The gain is computed via ``jnp.linalg.inv(S)``.  On Jetson Orin
        (JetPack 6.x) this can trigger a cuSolver internal error.  Replace
        that line with::

            K = jnp.linalg.solve(S, H @ ekf.P).T

        or set ``XLA_FLAGS=--xla_gpu_cusolver_disable=true`` as a temporary
        workaround.

    **Covariance update** — Joseph form, which preserves symmetry and positive
    semi-definiteness regardless of numerical gain accuracy:

    .. math::

        P^+ = (I - KH) \\, P \\, (I - KH)^\\top + K R K^\\top

    Args:
        ekf: Predicted EKF state (output of :func:`ekf_predict`).
        z: Measurement vector of shape :math:`(m,)`.
        h: Measurement function mapping a state to a vector of shape
            :math:`(m,)`. Must be a **static** (compile-time constant) Python
            callable — it is differentiated through, not passed as traced
            array data.
        R: Measurement noise covariance of shape :math:`m \\times m`.

    Returns:
        Corrected :class:`EKFState` after fusing the measurement.
    """
    H = compute_H(h, ekf.x_nom)

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
    dt: float,
) -> EKFState:
    """**Single-sensor** predict-then-update convenience wrapper.

    Calls :func:`ekf_predict` followed by exactly **one** call to
    :func:`ekf_update` for a **single** measurement source.  This is **not**
    a complete sensor-fusion cycle.  When fusing multiple sensors per
    timestep, call :func:`ekf_predict` once and then chain
    :func:`ekf_update` for each source::

        ekf = ekf_predict(model, ekf, u, Q, dt)
        ekf = ekf_update(ekf, z_gps,     h_gps,     R_gps)
        ekf = ekf_update(ekf, z_gyro,    h_gyro,    R_gyro)
        ekf = ekf_update(ekf, z_gravity, h_gravity, R_gravity)
        ekf = ekf_update(ekf, z_wheels,  h_wheels,  R_wheels)

    Args:
        model: Car physics model.
        ekf: Current EKF state.
        u: Control input for the predict step.
        z: Measurement vector of shape :math:`(m,)`.
        h: Measurement function (static callable). See :func:`ekf_update`.
        Q: Process noise covariance :math:`16 \\times 16`.
        R: Measurement noise covariance :math:`m \\times m`.
        dt: Timestep [s].

    Returns:
        Updated :class:`EKFState` after predict and single-sensor fusion.
    """
    ekf_pred = ekf_predict(model, ekf, u, Q, dt)
    ekf_upd = ekf_update(ekf_pred, z, h, R)
    return ekf_upd
