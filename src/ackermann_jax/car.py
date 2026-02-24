from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Literal, Optional

import jax
import jax.numpy as jnp
from jax import Array

import jaxlie


WheelName = Literal["FL","FR","RL","RR"]
WHEEL_ORDER: Tuple[WheelName, ...] = ("FL","FR","RL","RR")

## Pytree friendly dataclasses

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

    def tree_unflatten(cls, aux, children):
        return cls(*children)

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

        tau_cmd = p.motor.mask() * u.tau_w
        domega_w = (tau_cmd - p.geom.wheel_radius * Fx - p.wheels.b_w * omega_w) / p.wheels.I_w

        g_W = jnp.array([0.0, 0.0, -p.chassis.g],dtype=jnp.float32)
        dv_W = (F_W / p.chassis.mass) + g_W


        I_b = p.chassis.I_body
        Iw = I_b @ w_B
        dW_B = jnp.linalg.solve(I_b,(tau_B-jnp.cross(w_B,Iw)))

        dp_W = v_W

        # Tangent for SO3 is handled in integrator via w_B; derivative R is placeholder.
        return AckermannCarState(
            p_W=dp_W,
            R_WB=jaxlie.SO3.identity(),
            v_W=dv_W,
            w_B=dW_B,
            omega_W=domega_w
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
        p = self.params
        R = x.R_WB.as_matrix()

        v_B = R.T @ x.v_W
        v_X = v_B[0]
        err = v_cmd - v_X
        integ_next = integral_state + err

        Fx_cmd = Kp * err + Ki * integ_next

        mask = p.motor.mask()
        if p.motor.alpha is None:
            denom = jnp.maximum(1.0, jnp.sum(mask))
            alpha = mask / denom
        else:
            alpha = p.motor.alpha

        Fx_i_cmd = mask * alpha * Fx_cmd

        if use_traction_limit:
            r_B = p.geom.wheel_contact_points_body()
            p_i_W = x.p_W[None,:] + (R @ r_B.T).T # transform points to world frame
            w_cross_r_B = jnp.cross(x.w_B[None,:],r_B)
            v_i_W = x.v_W[None,:] + (R @ w_cross_r_B.T).T # transposrt equation
            Fz = self._normal_forces(p_i_W,v_i_W)
            Fx_lim = p.tires.mu * Fz
            Fx_i_cmd = jnp.clip(Fx_i_cmd,-Fx_lim,Fx_lim)

        tau_w = p.geom.wheel_radius * Fx_i_cmd
        tau_w = jnp.clip(tau_w, -tau_max, tau_max)
        return tau_w, integ_next



    def _normal_forces(self, p_i_W: Array, v_i_W: Array) -> Array:
        k_n = self.params.contact.k_n
        c_n = self.params.contact.c_n
        z0 = self.params.contact.z0

        z = p_i_W[:,2]
        d = jnp.maximum(0.0,z0-z)
        vz = v_i_W[:,2]
        d_dot = jnp.where(d > 0.0, -vz, 0.0)

        Fz = k_n * d + c_n * d_dot
        return jnp.maximum(0.0, Fz)

    def _slip(self, omega_w: Array, v_t: Array, v_n: Array) -> Tuple[Array, Array]:
        rw = self.params.geom.wheel_radius
        eps_v = self.params.tires.eps_v
        denom = jnp.maximum(eps_v,jnp.abs(v_t))
        kappa = (rw * omega_w - v_t) / denom
        alpha = jnp.arctan2(v_n, denom)
        return kappa, alpha

    def _tire_forces(self, kappa: Array, alpha: Array, Fz: Array) -> Tuple[Array,Array]:
        tp = self.params.tires
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

    def _integrate_semi_implicit(self, x: AckermannCarState, xdot: AckermannCarState, dt: float) -> AckermannCarState:
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

    def diagnostics(
        self,
        x: AckermannCarState,
        u: AckermannCarInput
    ) -> Diagnostics:
        p = self.params
        p_W, R_WB, v_W, w_B, omega_w = x.p_W, x.R_WB, x.v_W, x.w_B, x.omega_W
        R = R_WB.as_matrix()

        delta_i = p.geom.ackermann_front_angles(u.delta)
        r_B = p.geom.wheel_contact_points_body()

        c = jnp.cos(delta_i); s = jnp.sin(delta_i)
        # body frame vectors of steering angle
        n_B = jnp.stack([-s,c, jnp.zeros_like(c)],axis=-1)
        t_B = jnp.stack([c,s, jnp.zeros_like(c)],axis=-1)

        t_W = (R @ t_B.T).T
        n_W = (R @ n_B.T).T
        z_W = jnp.array([0., 0., 1.],dtype=jnp.float32)

        p_i_W = p_W[None, :] + (R @ r_B.T).T
        w_cross_r_B = jnp.cross(w_B[None,:],r_B)
        v_i_W = v_W[None,:] + (R @ w_cross_r_B.T).T

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

        return Diagnostics(
            delta_i=delta_i,
            r_B=r_B,
            p_i_W=p_i_W,
            v_i_W=v_i_W,
            Fz=Fz,
            v_t=v_t,
            v_n=v_n,
            kappa=kappa,
            alpha=alpha,
            Fx=Fx,
            Fy=Fy,
            f_i_W=f_i_W,
            F_W=F_W,
            tau_W=tau_W,
            tau_B=tau_B
        )


# ---
# Factory Helpers
# ---

def default_params() -> AckermannCarParams:
    geom = AckermannGeometry(
        L=0.26,
        W=0.16,
        a=0.13,
        b=0.13,
        h=0.06,
        wheel_radius=0.03
    )


    mass = 1.5 # kg
    I_body = jnp.diag(jnp.array([0.02, 0.02, 0.04], dtype=jnp.float32))
    #TODO: I_body should be a function from jaxsim/URDF file
    chassis = ChassisParams(mass=mass,I_body=I_body,g=9.81)

    wheels = WheelParams(I_w=0.001, b_w=0.01) # these need to be dynamically determined as well
    tires = TireParams(mu=0.9, C_kappa=30.0, C_alpha=25.0,eps_v=1e-3)
    contact = ContactParams(k_n=2e3,c_n=50,z0=0.0)

    has_motor = jnp.array([0.0,0.0,1.0,1.0],dtype=jnp.float32) # RWD car
    motor = MotorConfig(has_motor=has_motor, alpha=None)

    return AckermannCarParams(
        geom=geom,
        chassis=chassis,
        wheels=wheels,
        tires=tires,
        contact=contact,
        motor=motor
    )

def default_state(z0: float = 0.08) -> AckermannCarState:
    p_W = jnp.array([0.0, 0.0, z0], dtype=jnp.float32)
    R_WB = jaxlie.SO3.identity()
    v_W = jnp.zeros((3,), dtype=jnp.float32)
    w_B = jnp.zeros((3,), dtype=jnp.float32)
    omega_W = jnp.zeros((4,), dtype=jnp.float32)

    return AckermannCarState(
        p_W=p_W,
        R_WB=R_WB,
        v_W=v_W,
        w_B=w_B,
        omega_W=omega_W
    )

def pack_input(delta: float, tau_w: Array) -> AckermannCarInput:
    return AckermannCarInput(
        delta = jnp.array(delta, dtype=jnp.float32),
        tau_w=tau_w.astype(jnp.float32)
    )


