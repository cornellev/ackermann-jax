from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Literal, Optional

import jax
import jax.numpy as jnp
from jax import Array

import jaxlie


WheelName = Literal["FL","FR","RL","RR"]
WHEEL_ORDER: Tuple[WheelName, ...] = ("FL","FR","RL","RR")

def _clip(x: Array, lo: Array, hi: Array) -> Array:
    return jnp.minimum(jnp.maximum(x, lo), hi)

## Pytree friendly dataclasses

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AckermannGeometry:
    L: float
    W: float
    a: float
    b: float
    h: float
    wheel_radius: float

    def wheel_contact_points_body(self) -> Array:
        a, b, W, h = self.a, self.b, self.W, self.h
        return jnp.array([
            [ a , +W / 2.0, -h], #FL
            [ a , -W / 2.0, -h], #FR
            [-b , +W / 2.0, -h], #RL
            [-b , -W / 2.0, -h], #RR
        ], dtype=jnp.float32)

def ackermann_front_angles(self, delta: Array, eps: float = 1e6) -> Array:
    tan_delta = jnp.tan(delta)
    tan_delta = jnp.where(jnp.abs(tan_delta) < eps, jnp.sign(tan_delta) * eps + eps, tan_delta)
    R_turn = self.L / tan_delta

    denom_L = (R_turn - self.W / 2.0)
    denom_R = (R_turn + self.W / 2.0)
    denom_L = jnp.where(jnp.abs(denom_L) < eps, jnp.sign(denom_L) * eps + eps, denom_L)
    denom_R = jnp.where(jnp.abs(denom_R) < eps, jnp.sign(denom_R) * eps + eps, denom_R)

    delta_FL = jnp.arctan(self.L / denom_L)
    delta_FR = jnp.arctan(self.L / denom_R)

    return jnp.array([delta_FL, delta_FR, 0.0, 0.0],dtype=jnp.float32)

    def tree_flatten(self):
        children = (
            jnp.array(self.L, dtype=jnp.float32),
            jnp.array(self.W, dtype=jnp.float32),
            jnp.array(self.a, dtype=jnp.float32),
            jnp.array(self.b, dtype=jnp.float32),
            jnp.array(self.h, dtype=jnp.float32),
            jnp.array(self.wheel_radius, dtype=jnp.float32),
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        L, W, a, b, h, wheel_radius = children
        return cls(float(L), float(W), float(a), float(b), float(h), float(wheel_radius))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ContactParams:
    k_n: float
    c_n: float
    z0: float = 0.0

    def tree_flatten(self):
        children = (
            jnp.array(self.k_n, dtype=jnp.float32),
            jnp.array(self.c_n, dtype=jnp.float32),
            jnp.array(self.z0, dtype=jnp.float32),
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        k_n, c_n, z0 = children
        return cls(float(k_n), float(c_n), float(z0))



