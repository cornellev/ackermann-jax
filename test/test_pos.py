"""
test_pos.py — Test suite for car.py, body-frame formulation.

State: x = [px, py, θ, v, ω]

Tests
─────
1. Static equilibrium        zero control; car must not move
2. Straight-line kinematics  δ=0, constant a; check p, v vs closed form
3. Circle closure            a=0, constant δ; lap closes in position and heading
4. Circle radius             arc stays on Ackermann-predicted circle
5. Jacobian vs FD            linearise() agrees with central finite differences
6. Omega / heading-rate      θ̇ = ω to within O(dt²) along any trajectory
7. Pytree smoke tests        register_dataclass works with jit/vmap/tree ops

Each test prints PASS / FAIL with the key residual.
"""

from __future__ import annotations

import sys
from dataclasses import replace

import jax
import jax.numpy as jnp

from ackermann_jax import (
    AckermannParams, CarState, CarControl,
    step_rk4, linearise,
    state_to_vec, vec_to_state, STATE_DIM,
)
from ackermann_jax.ekf import wrap_angle

jax.config.update("jax_enable_x64", True)


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

PARAMS = AckermannParams(wheelbase=0.257, mass=1.5)
DT     = 0.01   # [s]

_step_jit      = jax.jit(step_rk4,  static_argnums=(3,))
_linearise_jit = jax.jit(linearise, static_argnums=(3,))


def _simulate(state0: CarState, control: CarControl, n_steps: int) -> CarState:
    s = state0
    for _ in range(n_steps):
        s = _step_jit(s, control, PARAMS, DT)
    return s


def _state(px=0.0, py=0.0, theta=0.0, v=1.0, omega=0.0) -> CarState:
    return CarState(
        p_W   = jnp.array([px, py]),
        theta = jnp.array(theta),
        v     = jnp.array(v),
        omega = jnp.array(omega),
    )


def _ctrl(a=0.0, delta=0.0) -> CarControl:
    return CarControl(a=jnp.array(a), delta=jnp.array(delta))


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

_results: list[tuple[str, bool, str]] = []


def _check(name: str, passed: bool, detail: str) -> None:
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}]  {name}  --  {detail}")
    _results.append((name, passed, detail))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Static equilibrium
# ─────────────────────────────────────────────────────────────────────────────

def test_static() -> None:
    """Zero velocity + zero control ⇒ nothing moves."""
    s0   = _state(px=1.0, py=2.0, theta=0.5, v=0.0, omega=0.0)
    ctrl = _ctrl(a=0.0, delta=0.0)
    sf   = _simulate(s0, ctrl, n_steps=100)

    dp    = float(jnp.linalg.norm(sf.p_W - s0.p_W))
    dth   = float(jnp.abs(sf.theta - s0.theta))
    dv    = float(jnp.abs(sf.v - s0.v))
    dom   = float(jnp.abs(sf.omega - s0.omega))

    tol = 1e-10
    _check("static / position",  dp  < tol, f"|dp|     = {dp:.2e}")
    _check("static / heading",   dth < tol, f"|dθ|     = {dth:.2e}")
    _check("static / speed",     dv  < tol, f"|dv|     = {dv:.2e}")
    _check("static / yaw rate",  dom < tol, f"|dω|     = {dom:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Straight-line kinematics
# ─────────────────────────────────────────────────────────────────────────────

def test_straight_line() -> None:
    """
    δ=0, constant a=a0, initial speed v0 heading +x.

    Closed form:
        v(T) = v0 + a0·T
        px(T) = v0·T + ½·a0·T²
        py(T) = 0,   θ(T) = 0,   ω(T) = 0
    """
    v0, a0, T = 1.0, 0.5, 1.0
    s0   = _state(v=v0)
    ctrl = _ctrl(a=a0, delta=0.0)
    sf   = _simulate(s0, ctrl, n_steps=int(T / DT))

    v_exp = v0 + a0 * T
    p_exp = v0 * T + 0.5 * a0 * T ** 2

    tol_v, tol_p = 1e-6, 1e-6

    _check("straight / v",
           float(jnp.abs(sf.v - v_exp)) < tol_v,
           f"|dv|  = {float(jnp.abs(sf.v - v_exp)):.2e}  (want {v_exp:.4f} m/s)")
    _check("straight / px",
           float(jnp.abs(sf.p_W[0] - p_exp)) < tol_p,
           f"|dpx| = {float(jnp.abs(sf.p_W[0] - p_exp)):.2e}  (want {p_exp:.4f} m)")
    _check("straight / py = 0",
           float(jnp.abs(sf.p_W[1])) < tol_p,
           f"|py|  = {float(jnp.abs(sf.p_W[1])):.2e}")
    _check("straight / θ = 0",
           float(jnp.abs(sf.theta)) < 1e-10,
           f"|θ|   = {float(jnp.abs(sf.theta)):.2e}")
    _check("straight / ω = 0",
           float(jnp.abs(sf.omega)) < 1e-10,
           f"|ω|   = {float(jnp.abs(sf.omega)):.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Constant-radius circle — closure
# ─────────────────────────────────────────────────────────────────────────────

def test_circle_closure() -> None:
    """
    a=0, constant δ.  After exactly one Ackermann period both position and
    heading return to their initial values; speed and yaw rate are conserved.
    """
    v0    = 1.0
    delta = 0.3
    L     = PARAMS.wheelbase

    R      = float(L / jnp.tan(delta))
    period = 2.0 * jnp.pi * R / v0
    n      = int(period / DT)

    omega0 = float(v0 * jnp.tan(jnp.array(delta)) / L)
    s0     = _state(v=v0, omega=omega0)
    ctrl   = _ctrl(a=0.0, delta=float(delta))
    sf     = _simulate(s0, ctrl, n_steps=n)

    dp  = float(jnp.linalg.norm(sf.p_W - s0.p_W))
    dth = float(sf.theta % (2.0 * jnp.pi))
    dth = min(dth, 2.0 * jnp.pi - dth)
    dv  = float(jnp.abs(sf.v - s0.v))
    dom = float(jnp.abs(sf.omega - s0.omega))

    _check("circle / position closure",
           dp  < 1e-2,
           f"|dp|  = {dp:.4f} m  (R={R:.3f} m, T={period:.3f} s)")
    _check("circle / heading closure",
           dth < 1e-2,
           f"|dθ|  = {dth:.4f} rad")
    _check("circle / speed conserved",
           dv  < 1e-10,
           f"|dv|  = {dv:.2e} m/s")
    _check("circle / omega conserved  (a=0 ⟹ ω̇=0)",
           dom < 1e-10,
           f"|dω|  = {dom:.2e} rad/s")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Circle radius accuracy
# ─────────────────────────────────────────────────────────────────────────────

def test_circle_radius() -> None:
    """
    Sample the arc and confirm every point is within tolerance of the
    Ackermann-predicted radius from the turn centre.

    Starting at origin heading +x, turning left with constant δ,
    the centre is at (0, R).
    """
    v0    = 1.0
    delta = 0.3
    L     = PARAMS.wheelbase

    R      = float(L / jnp.tan(delta))
    centre = jnp.array([0.0, R])

    omega0 = float(v0 * jnp.tan(jnp.array(delta)) / L)
    s      = _state(v=v0, omega=omega0)
    ctrl   = _ctrl(a=0.0, delta=delta)

    errs = []
    for _ in range(200):
        s = _step_jit(s, ctrl, PARAMS, DT)
        r = float(jnp.linalg.norm(s.p_W - centre))
        errs.append(abs(r - R))

    max_err = max(errs)
    _check("circle / radius accuracy",
           max_err < 1e-4,
           f"max |r − R| = {max_err:.2e} m  (R={R:.3f} m)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Jacobian vs central finite differences
# ─────────────────────────────────────────────────────────────────────────────

def test_jacobian_fd() -> None:
    """
    linearise() (jacfwd through RK4) must agree with a central FD Jacobian
    to well within the FD truncation error O(ε²).
    """
    omega0 = float(1.2 * jnp.tan(jnp.array(0.15)) / PARAMS.wheelbase)
    s0     = _state(px=0.1, py=-0.2, theta=0.4, v=1.2, omega=omega0)
    ctrl   = _ctrl(a=0.3, delta=0.15)

    _, F_ad = _linearise_jit(s0, ctrl, PARAMS, DT)

    eps_fd = 1e-5
    x0     = state_to_vec(s0)

    def prop(vec):
        return state_to_vec(step_rk4(vec_to_state(vec), ctrl, PARAMS, DT))

    F_fd = jnp.zeros((STATE_DIM, STATE_DIM))
    for j in range(STATE_DIM):
        e    = jnp.zeros(STATE_DIM).at[j].set(1.0)
        F_fd = F_fd.at[:, j].set((prop(x0 + eps_fd*e) - prop(x0 - eps_fd*e))
                                  / (2.0 * eps_fd))

    max_err  = float(jnp.max(jnp.abs(F_ad - F_fd)))
    norm_FmI = float(jnp.linalg.norm(F_ad - jnp.eye(STATE_DIM)))

    _check("jacobian / max |F_ad − F_fd|",
           max_err < 1e-7,
           f"max entry error = {max_err:.2e}  (ε_fd={eps_fd:.0e})")
    _check("jacobian / ‖F − I‖_F is O(dt)",
           norm_FmI < 0.5,
           f"‖F − I‖_F = {norm_FmI:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Omega / heading-rate consistency
# ─────────────────────────────────────────────────────────────────────────────

def test_omega_heading_consistency() -> None:
    """
    Since θ̇ = ω in the dynamics, the finite-difference heading rate
    dθ/dt ≈ (θ_{k+1} − θ_k) / dt should equal ω_k to within O(dt²).

    Test A — Constant-speed circle (a=0, δ=const):
        ω̇ = a·tan(δ)/L = 0  ⟹  ω is constant exactly.
        θ(t) = θ₀ + ω·t  (linear).

    Test B — Accelerating turn (a≠0, δ≠0):
        ω changes, FD heading rate still tracks ω to within RK4 accuracy.
    """
    delta = 0.25
    L     = PARAMS.wheelbase

    # ── Test A: constant-speed circle ────────────────────────────────────────
    v0     = 1.0
    omega0 = float(v0 * jnp.tan(jnp.array(delta)) / L)
    s0     = _state(v=v0, omega=omega0)
    ctrl_A = _ctrl(a=0.0, delta=delta)

    max_omega_drift = 0.0
    max_theta_lin   = 0.0
    s = s0

    for k in range(200):
        s_prev = s
        s      = _step_jit(s_prev, ctrl_A, PARAMS, DT)

        # ω should be exactly constant (a=0 ⟹ ω̇=0)
        dom = float(jnp.abs(s.omega - s0.omega))
        max_omega_drift = max(max_omega_drift, dom)

        # θ should grow linearly: θ(k) = θ₀ + ω·k·dt
        theta_exp = float(s0.theta) + float(omega0) * (k + 1) * DT
        dth = float(jnp.abs(s.theta - theta_exp))
        max_theta_lin = max(max_theta_lin, dth)

    _check("omega / constant when a=0",
           max_omega_drift < 1e-10,
           f"max |Δω|   = {max_omega_drift:.2e} rad/s")
    _check("omega / θ grows linearly at rate ω",
           max_theta_lin < 1e-8,
           f"max |θ−θ_exp| = {max_theta_lin:.2e} rad")

    # ── Test B: FD heading rate ≈ ω along arbitrary trajectory ───────────────
    omega_init = float(0.5 * jnp.tan(jnp.array(0.15)) / L)
    s      = _state(v=0.5, omega=omega_init)
    ctrl_B = _ctrl(a=0.2, delta=0.15)

    max_fd_err = 0.0
    for _ in range(200):
        s_prev = s
        s      = _step_jit(s_prev, ctrl_B, PARAMS, DT)
        # FD heading rate (wrap to handle ±π crossing)
        dtheta_fd = float(wrap_angle(s.theta - s_prev.theta)) / DT
        # Should match ω at the start of the step (Euler approximation)
        err = abs(dtheta_fd - float(s_prev.omega))
        max_fd_err = max(max_fd_err, err)

    _check("omega / FD θ̇ ≈ ω  (arbitrary traj, O(dt) tolerance)",
           max_fd_err < 5e-2,
           f"max |dθ/dt − ω| = {max_fd_err:.2e} rad/s")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Pytree / register_dataclass smoke tests
# ─────────────────────────────────────────────────────────────────────────────

def test_pytree_registration() -> None:
    """
    Verify that register_dataclass is correctly wired:
      (a) jax.tree.leaves returns 4 arrays: [p_W, theta, v, omega]
      (b) jax.tree.map round-trips cleanly
      (c) dataclasses.replace works on data fields
      (d) AckermannParams has zero leaves (all static)
      (e) jax.vmap batches over a stacked CarState
    """
    omega0 = float(1.0 * jnp.tan(jnp.array(0.3)) / PARAMS.wheelbase)
    s = _state(px=1.0, py=2.0, theta=0.5, v=3.0, omega=omega0)

    # (a) leaves — should be [p_W, theta, v, omega] → 4 arrays
    leaves = jax.tree.leaves(s)
    _check("pytree / leaf count",
           len(leaves) == 4,
           f"got {len(leaves)} leaves (want 4: p_W, theta, v, omega)")

    # (b) tree.map round-trip: multiply all leaves by 1.0
    s2  = jax.tree.map(lambda x: x * 1.0, s)
    err = float(jnp.max(jnp.abs(state_to_vec(s2) - state_to_vec(s))))
    _check("pytree / tree.map round-trip",
           err == 0.0,
           f"max deviation = {err:.2e}")

    # (c) dataclasses.replace on a data field
    s3 = replace(s, theta=jnp.array(0.0))
    _check("pytree / dataclass replace",
           float(s3.theta) == 0.0 and float(s3.p_W[0]) == 1.0,
           f"theta={float(s3.theta)}, px={float(s3.p_W[0])}")

    # (d) AckermannParams: all static → zero JAX leaves
    p_leaves = jax.tree.leaves(PARAMS)
    _check("pytree / params has no leaves (all static)",
           len(p_leaves) == 0,
           f"got {len(p_leaves)} leaves (want 0)")

    # (e) vmap: batch step_rk4 over a stack of initial states
    batch_size = 4
    s_batch    = jax.tree.map(lambda x: jnp.stack([x] * batch_size), s)
    ctrl       = _ctrl(a=0.5, delta=0.1)

    step_vmap  = jax.vmap(lambda si: step_rk4(si, ctrl, PARAMS, DT))
    sf_batch   = step_vmap(s_batch)

    _check("pytree / vmap step_rk4",
           sf_batch.p_W.shape == (batch_size, 2),
           f"p_W.shape = {sf_batch.p_W.shape}  (want ({batch_size}, 2))")

    row_spread = float(jnp.max(jnp.std(sf_batch.p_W, axis=0)))
    _check("pytree / vmap rows identical",
           row_spread == 0.0,
           f"std across batch rows = {row_spread:.2e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n-- car.py body-frame test suite ---------------------------------------")

    print("\n[1] Static equilibrium")
    test_static()

    print("\n[2] Straight-line kinematics")
    test_straight_line()

    print("\n[3] Constant-radius circle -- closure")
    test_circle_closure()

    print("\n[4] Constant-radius circle -- radius accuracy")
    test_circle_radius()

    print("\n[5] Jacobian vs finite differences")
    test_jacobian_fd()

    print("\n[6] Omega / heading-rate consistency")
    test_omega_heading_consistency()

    print("\n[7] Pytree / register_dataclass")
    test_pytree_registration()

    n_pass = sum(1 for _, p, _ in _results if p)
    n_fail = sum(1 for _, p, _ in _results if not p)
    print(f"\n-- {n_pass} passed, {n_fail} failed "
          + "-" * max(0, 55 - len(str(n_pass)) - len(str(n_fail))))

    if n_fail:
        print("\nFailed:")
        for name, passed, detail in _results:
            if not passed:
                print(f"  {name}: {detail}")
        sys.exit(1)


if __name__ == "__main__":
    main()
