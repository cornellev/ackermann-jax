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
    Rref = R_ref.as_matrix()
    Rdef = R.as_matrix()
    return jaxlie.SO3.from_matrix(Rref.T @ Rdef).log()

def inject_rotation_error(R_nom, dtheta):
    return R_nom @ jaxlie.SO3.exp(dtheta)

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AckermannCarErrorState:
    dp_W: Array
    dtheta_B: Array
    dv_W: Array
    dw_B: Array

    def tree_flatten(self):
        children = (
            self.dp_W,
            self.dtheta_B,
            self.dv_W,
            self.dw_B,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        dp_W, dtheta_B, dv_W, dw_B = children
        return cls(dp_W, dtheta_B, dv_W, dw_B)

def zero_error_state() -> AckermannCarErrorState:
    return AckermannCarErrorState(
        dp_W = jnp.zeros((3,), dtype=jnp.float32),
        dtheta_B = jnp.zeros((3,), dtype=jnp.float32),
        dv_W = jnp.zeros((3,), dtype=jnp.float32),
        dw_B = jnp.zeros((3,), dtype=jnp.float32),
    )

def inject_error(
    x: AckermannCarState,
    dx: AckermannCarErrorState
) -> AckermannCarState:
    return AckermannCarState(
        p_W = x.p_W + dx.dp_W,
        R_WB = inject_rotation_error(x.R_WB, dx.dtheta_B),
        v_W = x.v_W + dx.dv_W,
        w_B = x.w_B + dx.dw_B,
    )

def state_difference(
    x_ref: AckermannCarState,
    x: AckermannCarState
) -> AckermannCarErrorState:
    return AckermannCarErrorState(
        dp_W = x.p_W - x_ref.p_W,
        dtheta_B = rotation_error(x_ref.R_WB, x.R_WB),
        dv_W = x.v_W - x_ref.v_W,
        dw_B = x.w_B - x_ref.w_B,
    )

def pack_error_state(dx: AckermannCarErrorState) -> Array:
    return jnp.concatenate([
        dx.dp_W,
        dx.dtheta_B,
        dx.dv_W,
        dx.dw_B,
    ])

def unpack_error_state(vec: Array) -> AckermannCarErrorState:
    return AckermannCarErrorState(
        dp_W = vec[0:3],
        dtheta_B = vec[3:6],
        dv_W = vec[6:9],
        dw_B = vec[9:12],
    )


# dynamics for EKF
def error_dynamics(
    model: AckermannCarModel,
    dx_vex: Array,
    x_nom: AckermannCarState,
    u: AckermannCarInput,
    dt: float,
    method: Literal["semi_implicit_euler","euler"] = "semi_implicit_euler"
):
    dx = unpack_error_state(dx_vex)

    x_perturbed = inject_error(x_nom, dx)

    x_nom_next = model.step(x_nom, u, dt, method=method)
    x_perturbed_next = model.step(x_perturbed, u, dt, method=method)

    # state_difference(ref, x) = x − ref, so ref=nom, x=perturbed gives δx_{k+1}
    dx_next = state_difference(x_nom_next, x_perturbed_next)
    return pack_error_state(dx_next)
