from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Literal, Optional

import jax
import jax.numpy as jnp
from jax import Array

import jaxlie


WheelName = Literal["FL","FR","RL","RR"]
WHEEL_ORDER: Tuple[WheelName, ...] = ("FL","FR","RL","RR")

def smooth_relu(x, eps=1e-4):
    # eps has units of x; smaller = sharper
    return eps * jnp.log1p(jnp.exp(x / eps))

## Pytree friendly dataclasses
#TODO: add print methods for dataclasses for debugging/interactivity

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Diagnostics:
    delta_i: Array      # (4,)
    r_B: Array          # (4,3)
    p_i_W: Array        # (4,3)
    v_i_W: Array        # (4,3)
    Fz: Array           # (4,)
    v_t: Array          # (4,)
    v_n: Array          # (4,)
    kappa: Array        # (4,)
    alpha: Array        # (4,)
    Fx: Array           # (4,)
    Fy: Array           # (4,)
    f_i_W: Array        # (4,3)
    F_W: Array          # (3,)
    tau_W: Array        # (3,)
    tau_B: Array        # (3,)

    def tree_flatten(self):
        return (self.delta_i, self.r_B, self.p_i_W, self.v_i_W, self.Fz,
                self.v_t, self.v_n, self.kappa, self.alpha, self.Fx, self.Fy,
                self.f_i_W, self.F_W, self.tau_W, self.tau_B), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

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
        return cls(
            jnp.asarray(L, dtype=jnp.float32),
            jnp.asarray(W, dtype=jnp.float32),
            jnp.asarray(a, dtype=jnp.float32),
            jnp.asarray(b, dtype=jnp.float32),
            jnp.asarray(h, dtype=jnp.float32),
            jnp.asarray(wheel_radius, dtype=jnp.float32),
        )

    def ackermann_front_angles(self, delta: Array, eps: float = 1e-6) -> Array:
        """
        delta: steering angle of the front wheels (positive = left turn)
        Returns [delta_FL, delta_FR, 0, 0]

        Uses lax.cond instead of jnp.where so that only the selected branch's
        gradient is evaluated.  jnp.where evaluates both branches' gradients,
        which causes NaN in d/dL via the L/tan(0)=inf path even when the
        near-zero branch is selected.
        """
        def _near_zero(_):
            return jnp.array([delta, delta, 0.0, 0.0], dtype=jnp.float32)

        def _ackermann(_):
            tan_delta = jnp.tan(delta)
            R = self.L / tan_delta
            denom_FL = R - self.W / 2.0
            denom_FR = R + self.W / 2.0
            delta_FL = jnp.arctan2(self.L, denom_FL)
            delta_FR = jnp.arctan2(self.L, denom_FR)
            return jnp.array([delta_FL, delta_FR, 0.0, 0.0], dtype=jnp.float32)

        return jax.lax.cond(jnp.abs(delta) < eps, _near_zero, _ackermann, None)


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
        return cls(
            jnp.asarray(k_n, dtype=jnp.float32),
            jnp.asarray(c_n, dtype=jnp.float32),
            jnp.asarray(z0, dtype=jnp.float32),
        )

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TireParams:
    mu: float
    C_kappa: float
    C_alpha: float
    eps_v: float = 1e-4
    eps_force: float = 1e-8

    def tree_flatten(self):
        children = (
            jnp.array(self.mu, dtype=jnp.float32),
            jnp.array(self.C_kappa, dtype=jnp.float32),
            jnp.array(self.C_alpha, dtype=jnp.float32),
            jnp.array(self.eps_v, dtype=jnp.float32),
            jnp.array(self.eps_force, dtype=jnp.float32),
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        mu, Ck, Ca, eps_v, eps_f = children
        return cls(
            jnp.asarray(mu, dtype=jnp.float32),
            jnp.asarray(Ck, dtype=jnp.float32),
            jnp.asarray(Ca, dtype=jnp.float32),
            jnp.asarray(eps_v, dtype=jnp.float32),
            jnp.asarray(eps_f, dtype=jnp.float32),
        )

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class WheelParams:
    I_w: float # moment of inertia about the wheel
    b_w: float = 0.0

    def tree_flatten(self):
        children = (
            jnp.array(self.I_w,dtype=jnp.float32),
            jnp.array(self.b_w,dtype=jnp.float32)
        )
        return children, None


    @classmethod
    def tree_unflatten(cls, aux, children):
        I_w, b_w = children
        return cls(
            jnp.asarray(I_w, dtype=jnp.float32),
            jnp.asarray(b_w, dtype=jnp.float32),
        )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ChassisParams:
    mass: float
    I_body: Array # moment of inertia of the chassis in body frame
    g: float = 9.81 #m/s^2

    def tree_flatten(self):
        children = (
            jnp.array(self.mass,dtype=jnp.float32),
            self.I_body,
            jnp.array(self.g,dtype=jnp.float32)
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        mass, I_body, g = children
        return cls(
            jnp.asarray(mass, dtype=jnp.float32),
            I_body,
            jnp.asarray(g, dtype=jnp.float32),
        )

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MotorConfig:
    has_motor: Array # (4,)
    alpha: Optional[Array] = None

    def mask(self) -> Array:
        return self.has_motor.astype(jnp.float32)

    def tree_flatten(self):
        alpha = self.alpha if self.alpha is not None else jnp.array([],dtype=jnp.float32)
        aux = {"has_alpha": self.alpha is not None}
        children = (self.has_motor, alpha)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        has_motor, alpha = children
        if not aux["has_alpha"]:
            return cls(has_motor=has_motor, alpha=alpha)
        return cls(has_motor=has_motor, alpha=None)

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AckermannCarParams:
    geom: AckermannGeometry
    chassis: ChassisParams
    wheels: WheelParams
    tires: TireParams
    contact: ContactParams
    motor: MotorConfig


    def tree_flatten(self):
        children = (
            self.geom,
            self.chassis,
            self.wheels,
            self.tires,
            self.contact,
            self.motor
        )
        return children, None


    @classmethod
    def tree_unflatten(cls, aux, children):
        geom, chassis, wheels, tires, contact, motor = children
        return cls(geom,chassis,wheels,tires,contact,motor)

