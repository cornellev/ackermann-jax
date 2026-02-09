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
        return cls(float(mu), float(Ck), float(Ca), float(eps_v), float(eps_f))

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
        return cls(float(I_w), float(b_w))


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
        return cls(float(mass), I_body, float(g))

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class MotorConfig:
    has_motor: Array # (4,)
    alpha: Optional[Array] = None

    def mask(self) -> Array:
        return self.has_motor.astype(jnp.float32)

    def tree_flatten(self):
        alpha = self.alpha if self.alpha is not None else jnp.array([],dtype=jnp.float32)
        aux = {"hax_alpha": self.alpha is not None}
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


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AckermannCarState:
    p_W: Array # (3,) position of the car in world frame
    R_WB: jaxlie.SO3 # rotation from body to world frame
    v_W: Array # linear velocity of the car in world frame
    w_B: Array # angular velocity of the car in body frame
    omega_W: Array # (4,) angular velocity of the wheels in world frame

    def tree_flatten(self):
        children = (
            self.p_W,
            self.R_WB,
            self.v_W,
            self.w_B,
            self.omega_W
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        p_W,R_WB,v_W,w_B,omega_W = children
        return cls(p_W,R_WB,v_W,w_B,omega_W)

@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class AckermannCarInput:
    delta: Array
    tau_w: Array

    def tree_flatten(self):
        children = (self.delta, self.tau_w)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        delta, tau_w = children
        return cls(delta, tau_w)

# Model

class AckermannCarModel:
    def __init__(self, params: AckermannCarParams):
        self.params = params

    def xdot(self, x: AckermannCarState, u: AckermannCarInput) -> AckermannCarState:
        p = self.params

        p_W = x.p_W
        R_WB = x.R_WB
        v_W = x.v_W
        w_B = x.w_B
        omega_w = x.omega_W

        delta_i = p.geom.ackermann_front_angles(u.delta)
        r_B = p.geom.wheel_contact_points_body()

        c = jnp.cos(delta_i)
        s = jnp.sin(delta_i)
        t_B = jnp.stack([c,s, jnp.zeros_like(c)],axis=-1)
        n_B = jnp.stack([-s,c, jnp.zeros_like(c)],axis=-1)

        R = R_WB.as_matrix()
        t_W = (R @ t_B.T).T
        n_W = (R @ n_B.T).T
        z_W = jnp.array([0.0,0.0,1.0],dtype=jnp.float32)

        p_i_W = p_W[None,:] + (R @ r_B.T).T

        w_cross_r_B = jnp.cross(w_B[None,:],r_B)
        v_i_W = v_W[None,:] + (R @ w_cross_r_B.T).T # transposrt equation

        Fz = self._normal_forces(p_i_W,v_i_W)

        v_t = jnp.sum(t_W * v_i_W, axis=-1)
        v_n = jnp.sum(n_W * v_i_W, axis=-1)
        kappa, alpha = self._slip(omega_w, v_t, v_n)

        Fx, Fy = self._tire_forces(kappa, alpha, Fz)

        f_i_W = Fx[:,None] * t_W + Fy[:,None] * n_W + Fz[:,None] * z_W[None,:]
        F_W = jnp.sum(f_i_W, axis=0)

        r_i_W = p_i_W - p_W[None,:]
        tau_W = jnp.sum(jnp.cross(r_i_W, f_i_W), axis=0)
        tau_B = R.T @ tau_W

        tau_cmd = p.motor_mask() * u.tau_w
        domega_w = (tau_cmd - p.geom.wheel_radius * Fx - p.wheels.b_w * omega_w) / p.wheels.I_w

        g_W = jnp.array([0.0, 0.0 -p.chassis.g],dtype=jnp.float32)
        dv_W = F_W / p.chassis.mass + g_W


        I = p.chassis.I_body
        Iw = I @ w_B
        dW_B = jnp.linalg.solve(I,(tau_B-jnp.cross(w_B,Iw)))

        dp_W = v_W

        # Tangent for SO3 is handled in integrator via w_B; derivative R is placeholder.
        return AckermannCarState(
            p_W=dp_W,
            r_WB=jaxlie.SO3.identity(),
            v_W=dv_W,
            w_B=dW_B,
            omega_w=domega_w
        )

    def step(
        self,
        x: AckermannCarState,
        u: AckermannCarInput,
        dt: float,
        method: Literal["semi_implicit_euler","euler"] = "semi_implicit_euler"
    ) -> AckermannCarState:
        if method == "euler":
            xdot = self.xdot(x,u)
            return self._integrate_euler(x,xdot,dt)
        if method == "semi_implicit_euler":
            xdot = self.xdot(x,u)
            return self._integrate_semi_implicit(x,xdot,dt)
        raise ValueError(f"Unknown integration method {method}")

    def map_velocity_to_wheel_torques(
        self,
        x: AckermannCarState,
        v_cmd: Array,
        integral_state: Array,
        Kp: float,
        Ki: float,
        tau_max: float,
        use_traction_limit: bool = True
    ):
        raise NotImplementedError


    def _normal_forces(self, p_i_W: Array, v_i_W: Array) -> Array:
        k_n = self.p.contact.k_n
        c_n = self.p.contact.c_n
        z0 = self.p.contact.z0

        z = p_i_W[:,2]
        d = jnp.maximum(0.0,z0-z)
        vz = v_i_W[:,2]
        d_dot = jnp.where(d > 0.0, -vz, 0.0)

        Fz = k_n * d + c_n * d_dot
        return jnp.maximim(0.0, Fz)

    def _slip(self, omega_w: Array, v_t: Array, v_n: Array) -> Tuple[Array, Array]:
        rw = self.p.geom.wheel_radius
        eps_v = self.p.tires.eps_v
        denom = jnp.maximum(eps_v,jnp.abs(v_t))
        kappa = (rw * omega_w - v_t) / denom
        alpha = jnp.arctan2(v_n, denom)
        return kappa, alpha

    def _tire_forces(self, kappa: Array, alpha: Array, Fz: Array) -> Tuple[Array,Array]:
        tp = self.p.tires
        Fx_star = tp.C_kappa * kappa
        Fy_star = -tp.C_alpha * alpha

        Fmax = tp.mu * Fz
        mag = jnp.sqrt(Fx_star * Fx_star + Fy_star * Fy_star + tp.eps_force)
        scale = jnp.minimum(1.0, Fmax / mag)

        Fx = scale * Fx_star
        Fy = scale * Fy_star
        return Fx, Fy

    def _integrate_euler(self, x: AckermannCarState, xdot: AckermannCarState, dt: float) -> AckermannCarState:
        p_W_next = x.p_W + dt * xdot.p_W
        v_W_next = x.v_W + dt * xdot.v_W
        w_B_next = x.w_B + dt * xdot.w_B
        omega_w_next = x.omega_W + dt * xdot.omega_W

        # SO3 integration
        R_next = x.R_WB @ jaxlie.SO3.exp(w_B_next * dt)

        return AckermannCarState(
            p_W=p_W_next,
            R_WB=R_next,
            v_W = v_W_next,
            w_B = w_B_next,
            omega_W = omega_w_next
        )

    def _integrate_semi_implicit(self, x: AckermannCarState, u: AckermannCarInput, dt: float) -> AckermannCarState:
        xdot = self.xdot(x,u)

        v_W_next = x.v_W + dt * xdot.v_W
        w_B_next = x.w_B + dt * xdot.w_B
        omega_w_next = x.omega_W + dt * xdot.omega_W

        p_W_next = x.p_W + dt * v_W_next
        R_next = x.R_WB @ jaxlie.SO3.exp(w_B_next * dt)

        return AckermannCarState(
            p_W=p_W_next,
            R_WB=R_next,
            v_W=v_W_next,
            w_B=w_B_next,
            omega_W=omega_w_next
        )

