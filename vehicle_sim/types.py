from __future__ import annotations
from dataclasses import dataclass
import jax
from jax import tree_util
import jax.numpy as jnp

@tree_util.register_pytree_node_class
@dataclass
class BicycleParams:
    """Physical + actuation parameters for a bicycle model."""

    wheelbase: float # [m] L

    # Limits
    delta_max: float = jnp.deg2rad(35.0)
    a_min:  float = -6.0
    a_max: float = 3.0
    v_min: float = 0.0
    v_max: float = 50.0


    delta_rate_max: float = jnp.deg2rad(100.0)
    a_rate_max: float = 20.0

    def tree_flatten(self):
        children = (
            self.wheelbase,
            self.delta_max, self.a_min, self.a_max,
            self.v_min, self.v_max,
            self.delta_rate_max, self.a_rate_max,
        )
        aux = None
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)
