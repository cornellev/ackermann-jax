from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Literal, Optional # noqa
from .car import AckermannCarState, AckermannCarInput, AckermannCarModel

import jax
import jax.numpy as jnp
from jax import Array

import jaxlie
# -- EKF Functions for Jacobians and things --

def rotation_error(R_ref: jaxlie.SO3, R: jaxlie.SO3):
    """Compute the SO(3) rotation error between two orientations.

    Returns ``log(R_ref^T @ R)``, the Lie-algebra vector representing the
    rotation needed to go from ``R_ref`` to ``R``.

    Args:
        R_ref: Reference rotation.
        R: Query rotation.

    Returns:
        Error vector in the SO(3) Lie algebra ``(3,)`` [rad].
    """
    Rref = R_ref.as_matrix()
    Rdef = R.as_matrix()
    return jaxlie.SO3.from_matrix(Rref.T @ Rdef).log()

def inject_rotation_error(R_nom, dtheta):
    """Perturb a nominal rotation by a Lie-algebra vector.

    Computes ``R_nom @ Exp(dtheta)`` (right-perturbation convention).

    Args:
        R_nom (jaxlie.SO3): Nominal rotation.
        dtheta (Array): Perturbation in the Lie algebra ``(3,)`` [rad].

    Returns:
        jaxlie.SO3: Perturbed rotation.
    """
    return R_nom @ jaxlie.SO3.exp(dtheta)

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AckermannCarErrorState:
    """Additive error state for the error-state EKF (ES-EKF).

    Represents small perturbations around a nominal
    :class:`~ackermann_jax.car.AckermannCarState`. The rotation error
    ``dtheta_B`` lives in the SO(3) Lie algebra and is injected via
    :func:`inject_error` / recovered via :func:`rotation_error`.

    Attributes:
        dp_W (Array): Position error in the world frame ``(3,)`` [m].
        dtheta_B (Array): Rotation error as a Lie-algebra vector ``(3,)`` [rad].
        dv_W (Array): Linear velocity error in the world frame ``(3,)`` [m/s].
        dw_B (Array): Angular velocity error in the body frame ``(3,)`` [rad/s].
        domega_W (Array): Wheel spin-rate error ``(4,)`` [rad/s].
    """

    dp_W: Array
    dtheta_B: Array
    dv_W: Array
    dw_B: Array
    domega_W: Array

    def tree_flatten(self):
        children = (
            self.dp_W,
            self.dtheta_B,
            self.dv_W,
            self.dw_B,
            self.domega_W
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        dp_W, dtheta_B, dv_W, dw_B, domega_W = children
        return cls(dp_W, dtheta_B, dv_W, dw_B, domega_W)

def zero_error_state() -> AckermannCarErrorState:
    """Return a zero-initialised error state.

    Returns:
        :class:`AckermannCarErrorState` with all fields set to float32 zeros.
    """
    return AckermannCarErrorState(
        dp_W = jnp.zeros((3,), dtype=jnp.float32),
        dtheta_B = jnp.zeros((3,), dtype=jnp.float32),
        dv_W = jnp.zeros((3,), dtype=jnp.float32),
        dw_B = jnp.zeros((3,), dtype=jnp.float32),
        domega_W = jnp.zeros((4,), dtype=jnp.float32)
    )

def inject_error(
    x: AckermannCarState,
    dx: AckermannCarErrorState
) -> AckermannCarState:
    """Apply an error state to a nominal state to obtain a perturbed state.

    Position, velocity, angular velocity, and wheel speeds are added directly.
    Rotation uses the right-perturbation convention:
    ``R_perturbed = R_nom @ Exp(dtheta)``.

    Args:
        x: Nominal car state.
        dx: Error-state perturbation.

    Returns:
        Perturbed :class:`~ackermann_jax.car.AckermannCarState`.
    """
    return AckermannCarState(
        p_W = x.p_W + dx.dp_W,
        R_WB = inject_rotation_error(x.R_WB, dx.dtheta_B),
        v_W = x.v_W + dx.dv_W,
        w_B = x.w_B + dx.dw_B,
        omega_W = x.omega_W + dx.domega_W
    )

def state_difference(
    x_ref: AckermannCarState,
    x: AckermannCarState
) -> AckermannCarErrorState:
    """Compute the error state between two car states.

    The approximate inverse of :func:`inject_error`. Rotation difference
    is computed via :func:`rotation_error`.

    Args:
        x_ref: Reference (nominal) state.
        x: Query state.

    Returns:
        :class:`AckermannCarErrorState` representing ``x - x_ref``.
    """
    return AckermannCarErrorState(
        dp_W = x.p_W - x_ref.p_W,
        dtheta_B = rotation_error(x_ref.R_WB, x.R_WB),
        dv_W = x.v_W - x_ref.v_W,
        dw_B = x.w_B - x_ref.w_B,
        domega_W = x.omega_W - x_ref.omega_W
    )

def pack_error_state(dx: AckermannCarErrorState) -> Array:
    """Flatten an error state to a 1-D vector of length 16.

    Field order: ``dp_W`` (3), ``dtheta_B`` (3), ``dv_W`` (3),
    ``dw_B`` (3), ``domega_W`` (4).

    Args:
        dx: Error state to pack.

    Returns:
        Flat array of shape ``(16,)``.
    """
    return jnp.concatenate([
        dx.dp_W,
        dx.dtheta_B,
        dx.dv_W,
        dx.dw_B,
        dx.domega_W
    ])

def unpack_error_state(vec: Array) -> AckermannCarErrorState:
    """Reconstruct an error state from a 1-D vector of length 16.

    Inverse of :func:`pack_error_state`.

    Args:
        vec: Flat array of shape ``(16,)``.

    Returns:
        :class:`AckermannCarErrorState`.
    """
    return AckermannCarErrorState(
        dp_W = vec[0:3],
        dtheta_B = vec[3:6],
        dv_W = vec[6:9],
        dw_B = vec[9:12],
        domega_W = vec[12:16],
    )


def error_dynamics(
    model: AckermannCarModel,
    dx_vex: Array,
    x_nom: AckermannCarState,
    u: AckermannCarInput,
    dt: float,
    method: Literal["semi_implicit_euler","euler"] = "semi_implicit_euler"
):
    """Discrete error dynamics for the ES-EKF state-transition Jacobian.

    Propagates a small perturbation ``dx`` forward by one step and returns
    the resulting error ``dx_next`` in packed vector form. This function is
    differentiated by :func:`~ackermann_jax.ekf.compute_F` to obtain **F**.

    Args:
        model: Car physics model.
        dx_vex: Packed current error state vector of shape ``(16,)``.
        x_nom: Nominal car state at the current timestep.
        u: Control input.
        dt: Timestep [s].
        method: Integration method forwarded to
            :meth:`~ackermann_jax.car.AckermannCarModel.step`.

    Returns:
        Packed next error state vector of shape ``(16,)``.
    """
    dx = unpack_error_state(dx_vex)

    x_perturbed = inject_error(x_nom, dx)

    x_nom_next = model.step(x_nom, u, dt, method=method)
    x_perturbed_next = model.step(x_perturbed, u, dt, method=method)

    dx_next = state_difference(x_perturbed_next, x_nom_next)
    return pack_error_state(dx_next)
