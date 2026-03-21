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
    """Smooth approximation of ReLU using softplus.

    Computes ``eps * log(1 + exp(x / eps))``. As ``eps → 0`` this
    converges to ``max(0, x)``.

    Args:
        x: Input array.
        eps: Sharpness parameter; smaller values give a sharper corner.

    Returns:
        Smooth approximation of ``max(0, x)``.
    """
    return eps * jnp.log1p(jnp.exp(x / eps))

## Pytree friendly dataclasses
#TODO: add print methods for dataclasses for debugging/interactivity

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Diagnostics:
    """Per-step diagnostic quantities computed by the car model.

    All arrays have a leading wheel axis of length 4 ordered [FL, FR, RL, RR],
    except ``F_W``, ``tau_W``, and ``tau_B`` which are 3-vectors.

    Attributes:
        delta_i (Array): Ackermann-corrected individual wheel steering angles ``(4,)`` [rad].
        r_B (Array): Wheel contact positions in the body frame ``(4, 3)`` [m].
        p_i_W (Array): Wheel contact positions in the world frame ``(4, 3)`` [m].
        v_i_W (Array): Wheel contact velocities in the world frame ``(4, 3)`` [m/s].
        Fz (Array): Normal forces at each wheel ``(4,)`` [N].
        v_t (Array): Tangential contact-patch speeds ``(4,)`` [m/s].
        v_n (Array): Lateral contact-patch speeds ``(4,)`` [m/s].
        kappa (Array): Longitudinal slip ratios ``(4,)``.
        alpha (Array): Slip angles ``(4,)`` [rad].
        Fx (Array): Longitudinal tire forces ``(4,)`` [N].
        Fy (Array): Lateral tire forces ``(4,)`` [N].
        f_i_W (Array): Total force at each wheel in the world frame ``(4, 3)`` [N].
        F_W (Array): Net force on the chassis in the world frame ``(3,)`` [N].
        tau_W (Array): Net torque on the chassis in the world frame ``(3,)`` [N·m].
        tau_B (Array): Net torque on the chassis in the body frame ``(3,)`` [N·m].
    """

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
    """Geometric parameters of the Ackermann car chassis and wheels.

    Attributes:
        L (float): Wheelbase — front-to-rear axle distance [m].
        W (float): Track width — left-to-right wheel distance [m].
        a (float): Longitudinal distance from CoM to front axle [m].
        b (float): Longitudinal distance from CoM to rear axle [m].
        h (float): Height of wheel contact points below CoM [m].
        wheel_radius (float): Wheel rolling radius [m].
    """

    L: float
    W: float
    a: float
    b: float
    h: float
    wheel_radius: float

    def wheel_contact_points_body(self) -> Array:
        """Return wheel contact-point positions in the body frame.

        Returns:
            Array of shape ``(4, 3)`` ordered [FL, FR, RL, RR].
        """
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
        """Compute individual front wheel steering angles from a commanded steer angle.

        Applies Ackermann geometry so each front wheel tracks a different arc radius.
        Rear wheel angles are always zero. Uses ``jax.lax.cond`` rather than
        ``jnp.where`` to avoid NaN gradients through the ``L / tan(0)`` singularity
        in the near-zero branch.

        Args:
            delta: Commanded steering angle [rad]. Positive is a left turn.
            eps: Near-zero threshold below which both front wheels receive ``delta``
                directly, avoiding the tangent singularity.

        Returns:
            Array of shape ``(4,)``: ``[delta_FL, delta_FR, 0, 0]``.
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
    """Spring-damper ground contact model parameters.

    The normal force at each wheel is:
    ``Fz = k_n * relu(d) + c_n * relu(ḋ) * (d > 0)``
    where ``d = z0 - z`` is the penetration depth.

    Attributes:
        k_n (float): Normal stiffness [N/m].
        c_n (float): Normal damping coefficient [N·s/m].
        z0 (float): Ground-plane height [m]. Contact activates when ``z < z0``.
    """

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
    """Pacejka-inspired combined-slip tire model parameters.

    Attributes:
        mu (float): Peak friction coefficient (dimensionless).
        C_kappa (float): Longitudinal slip stiffness [N].
        C_alpha (float): Lateral slip stiffness [N/rad].
        eps_v (float): Velocity regularisation to avoid division by zero [m/s].
        eps_force (float): Force-magnitude regularisation term [N²].
    """

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
    """Rotational inertia parameters for the wheels.

    Attributes:
        I_w (float): Moment of inertia about the spin axis [kg·m²].
        b_w (float): Viscous spin damping coefficient [N·m·s].
    """

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
    """Rigid-body inertial parameters for the car chassis.

    Attributes:
        mass (float): Total vehicle mass [kg].
        I_body (Array): 3×3 inertia tensor in the body frame [kg·m²].
        g (float): Gravitational acceleration magnitude [m/s²].
    """

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
    """Motor assignment and torque-distribution configuration.

    Attributes:
        has_motor (Array): Binary mask ``(4,)`` indicating which wheels are driven,
            ordered [FL, FR, RL, RR].
        alpha (Array, optional): Explicit torque-distribution weights ``(4,)``.
            If ``None``, torque is shared equally among driven wheels.
    """

    has_motor: Array # (4,)
    alpha: Optional[Array] = None

    def mask(self) -> Array:
        """Return the float32 motor-enable mask.

        Returns:
            ``has_motor`` cast to float32, shape ``(4,)``.
        """
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
    """Aggregated parameter set for the full Ackermann car model.

    Attributes:
        geom (AckermannGeometry): Geometric parameters.
        chassis (ChassisParams): Chassis inertial parameters.
        wheels (WheelParams): Wheel inertial parameters.
        tires (TireParams): Tire force model parameters.
        contact (ContactParams): Ground contact spring-damper parameters.
        motor (MotorConfig): Motor assignment and distribution.
    """

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

