from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Literal, Optional
from .parameters import (
    Diagnostics,
    AckermannGeometry,
    ContactParams,
    TireParams,
    WheelParams,
    ChassisParams,
    MotorConfig,
    AckermannCarParams,
    WheelName,
    WHEEL_ORDER,
    smooth_relu,
)

import jax
import jax.numpy as jnp
from jax import Array

import jaxlie



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

@jax.tree_util.register_pytree_node_class
class AckermannCarModel:
    def __init__(self, params: AckermannCarParams):
        self.params = params

    def tree_flatten(self):
        return (self.params,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0])

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
        dt: float,
        Kp: float,
        Ki: float,
        tau_max: float,
        integ_max: float = 10.0,
        use_traction_limit: bool = True
    ):
        p = self.params
        R = x.R_WB.as_matrix()

        # v_B = R.T @ x.v_W
        # v_X = v_B[0]
        v_X = jnp.linalg.norm(x.v_W[:2])
        err = v_cmd - v_X
        integ_cand = jnp.clip(integral_state + err * dt,-integ_max, integ_max)

        Fx_cmd = Kp * err + Ki * integ_cand 

        mask = p.motor.mask()
        if p.motor.alpha is None:
            denom = jnp.maximum(1.0, jnp.sum(mask))
            alpha = mask / denom
        else:
            alpha = p.motor.alpha

        Fx_i_cmd = alpha * Fx_cmd

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

        #anti wind up logic
        sat = jnp.any(jnp.abs(tau_w) >= tau_max + 1e-6)

        integ_next = jnp.where(sat, integral_state, integ_cand)

        return tau_w, integ_next



    def map_heading_to_steering(
        self,
        x: AckermannCarState,
        psi_cmd: Array,
        integral_state: Array,
        dt: float,
        Kp: float = 3.0,
        Ki: float = 2.0,
        Kd: float = 0.4,
        delta_max: float = 0.35,
        integ_max: float = 0.5,
    ) -> Tuple[Array, Array]:
        """
        PID controller that maps a desired heading to a steering angle delta.

        Analogous to map_velocity_to_wheel_torques for the motor.

        Parameters:
            psi_cmd:        desired heading angle [rad], world frame
            integral_state: scalar integral of heading error (carry between steps)
            Kp, Ki, Kd:     PID gains
            delta_max:      saturation limit on steering angle [rad]
            integ_max:      anti-windup clamp on integral [rad·s]

        Returns:
            delta:      steering angle command [rad]
            integ_next: updated integral state

        Pole placement (linearised about v, L):
            Plant:  ψ̇ = (v/L)·δ  →  G(s) = a/s,  a = v/L
            Closed-loop char. poly:
                (1 + a·Kd)·s² + a·Kp·s + a·Ki = 0
                ωn  = sqrt(a·Ki / (1 + a·Kd))
                ζ   = a·Kp / (2·sqrt(a·Ki·(1 + a·Kd)))
            Default gains target ζ≈1.3, poles at s≈{-0.81, -3.74} at v=1 m/s, L=0.26 m.
            Both poles are real (overdamped) → no oscillation.
        """
        R = x.R_WB.as_matrix()
        yaw = jnp.arctan2(R[1, 0], R[0, 0])

        e_psi = psi_cmd - yaw
        # wrap to [-pi, pi]
        e_psi = jnp.mod(e_psi + jnp.pi, 2.0 * jnp.pi) - jnp.pi

        integ_cand = jnp.clip(integral_state + e_psi * dt, -integ_max, integ_max)

        r = x.w_B[2]  # yaw rate (z-component of body angular velocity)

        delta = Kp * e_psi + Ki * integ_cand - Kd * r
        delta = jnp.clip(delta, -delta_max, delta_max)

        # anti-windup: freeze integrator when saturated
        sat = jnp.abs(delta) >= delta_max - 1e-6
        integ_next = jnp.where(sat, integral_state, integ_cand)

        return delta, integ_next

    def _normal_forces(self, p_i_W: Array, v_i_W: Array) -> Array:
        k_n = self.params.contact.k_n
        c_n = self.params.contact.c_n
        z0 = self.params.contact.z0
        z = p_i_W[:,2]
        vz = v_i_W[:,2]

        d = z0-z
        ddot = -vz

        Fz = k_n * jax.nn.relu(d) + c_n * jax.nn.relu(ddot) * (d > 0)
        return jnp.maximum(0.0, Fz)

    def _slip(self, omega_w: Array, v_t: Array, v_n: Array) -> Tuple[Array, Array]:
        """
        Wheel slip function

        Parameters:
        omega_w: (4,) angular velocity of the wheels in wheel frame
        v_t: (4,) tangential velocity of the wheel contact patch in the direction of the wheel plane
        v_n: (4,) normal velocity of the wheel contact patch

        Returns:
        kappa: (4,) longitudinal slip ratio
        alpha: (4,) slip angle
        """
        kappa_max = 2.0
        rw = self.params.geom.wheel_radius
        eps_v = self.params.tires.eps_v
        # denom = jnp.maximum(eps_v,jnp.abs(v_t))
        denom = jnp.sqrt(v_t * v_t + eps_v * eps_v)
        kappa_prior = (rw * omega_w - v_t) / denom
        kappa = kappa_max * jnp.tanh(kappa_prior / kappa_max)
        alpha = jnp.arctan2(v_n, jnp.abs(v_t) + eps_v)
        alpha = 0.5 * jnp.tanh(alpha / 0.5)
        return kappa, alpha

    def _tire_forces(self, kappa: Array, alpha: Array, Fz: Array) -> Tuple[Array,Array]:
        tp = self.params.tires
        Fx_star = tp.C_kappa * kappa
        Fy_star = -tp.C_alpha * alpha
        # Fx_star = 0

        Fmax = tp.mu * Fz
        mag = jnp.sqrt(Fx_star * Fx_star + Fy_star * Fy_star + tp.eps_force)
        scale = jnp.minimum(1.0, Fmax / jnp.sqrt(mag**2 + Fmax**2))

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

        c = jnp.cos(delta_i)
        s = jnp.sin(delta_i)
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

    # choose wheel mass to be about 0.05 kg
    m_wheel = 0.05 # kg
    fac = 10 # inertia scale factor
    I_w = fac * 0.5 * m_wheel * geom.wheel_radius**2
    # print("Wheel inertia:", I_w)
    # we want wheels to settle within about 0.1s, so:
    tau_spin = 0.3 # seconds #NOTE: wheel damping was too high here causing issues
    b_w = jnp.array([
        I_w / 0.5,
        I_w / 0.5,
        I_w / 0.1,
        I_w / 0.1,
    ], dtype=jnp.float32)
    # b_w = I_w / tau_spin
    wheels = WheelParams(I_w=I_w, b_w=b_w) # these need to be dynamically determined as well
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
