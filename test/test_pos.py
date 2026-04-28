"""
test_pos.py — Test suite for car.py, Layer 1 (position / velocity / heading).

Tests
-----
1. Static equilibrium        zero control; car must not move
2. Straight-line kinematics  delta=0, constant a; check p and v vs closed form
3. Circle closure            a=0, constant delta; lap closes in p and theta
4. Circle radius             arc stays on Ackermann-predicted circle
5. Jacobian vs FD            linearise() agrees with central finite differences
6. Heading-velocity drift    theta tracks atan2(v_W) to within eps_v error
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

jax.config.update("jax_enable_x64", True)



# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PARAMS = AckermannParams(wheelbase=0.257, mass=1.5, eps_v=1e-3)
DT     = 0.01  # [s]

_step_jit      = jax.jit(step_rk4,   static_argnums=(3,))
_linearise_jit = jax.jit(linearise,  static_argnums=(3,))


def _simulate(state0: CarState, control: CarControl, n_steps: int) -> CarState:
    s = state0
    for _ in range(n_steps):
        s = _step_jit(s, control, PARAMS, DT)
    return s


def _state(px=0.0, py=0.0, vx=1.0, vy=0.0, theta=0.0) -> CarState:
    return CarState(
        p_W   = jnp.array([px, py]),
        v_W   = jnp.array([vx, vy]),
        theta = jnp.array(theta),
    )


def _ctrl(a=0.0, delta=0.0) -> CarControl:
    return CarControl(a=jnp.array(a), delta=jnp.array(delta))


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def _check(name: str, passed: bool, detail: str) -> None:
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}]  {name}  --  {detail}")
    _results.append((name, passed, detail))


# ---------------------------------------------------------------------------
# Test 1: Static equilibrium
# ---------------------------------------------------------------------------

def test_static() -> None:
    """Zero velocity + zero control => nothing moves."""
    s0   = _state(px=1.0, py=2.0, vx=0.0, vy=0.0, theta=0.5)
    ctrl = _ctrl(a=0.0, delta=0.0)
    sf   = _simulate(s0, ctrl, n_steps=100)

    dp  = jnp.linalg.norm(sf.p_W - s0.p_W)
    dv  = jnp.linalg.norm(sf.v_W - s0.v_W)
    dth = jnp.abs(sf.theta - s0.theta)

    tol = 1e-10
    _check("static / position", float(dp)  < tol, f"|dp| = {float(dp):.2e}")
    _check("static / velocity", float(dv)  < tol, f"|dv| = {float(dv):.2e}")
    _check("static / heading",  float(dth) < tol, f"|dth| = {float(dth):.2e}")


# ---------------------------------------------------------------------------
# Test 2: Straight-line kinematics
# ---------------------------------------------------------------------------

def test_straight_line() -> None:
    """
    delta=0, constant a=a0, initial speed v0 heading +x.

    Closed-form:
        v(T) = v0 + a0 * T
        p(T) = v0*T + 0.5*a0*T^2
        theta constant
    All lateral components must remain exactly zero.
    """
    v0, a0, T = 1.0, 0.5, 1.0
    s0   = _state(vx=v0)
    ctrl = _ctrl(a=a0, delta=0.0)
    sf   = _simulate(s0, ctrl, n_steps=int(T / DT))

    v_exp = v0 + a0 * T
    p_exp = v0 * T + 0.5 * a0 * T ** 2

    tol_v, tol_p = 1e-6, 1e-6

    _check("straight / vx",       float(jnp.abs(sf.v_W[0] - v_exp)) < tol_v,
           f"|dvx| = {float(jnp.abs(sf.v_W[0]-v_exp)):.2e}  (want {v_exp:.4f})")
    _check("straight / vy = 0",   float(jnp.abs(sf.v_W[1])) < tol_v,
           f"|vy| = {float(jnp.abs(sf.v_W[1])):.2e}")
    _check("straight / px",       float(jnp.abs(sf.p_W[0] - p_exp)) < tol_p,
           f"|dpx| = {float(jnp.abs(sf.p_W[0]-p_exp)):.2e}  (want {p_exp:.4f})")
    _check("straight / py = 0",   float(jnp.abs(sf.p_W[1])) < tol_p,
           f"|py| = {float(jnp.abs(sf.p_W[1])):.2e}")
    _check("straight / theta = 0",float(jnp.abs(sf.theta)) < 1e-10,
           f"|theta| = {float(jnp.abs(sf.theta)):.2e}")


# ---------------------------------------------------------------------------
# Test 3: Constant-radius circle — closure
# ---------------------------------------------------------------------------

def test_circle_closure() -> None:
    """
    a=0, constant delta.  After exactly one Ackermann period, both position
    and heading should return to their initial values.
    """
    v0    = 1.0
    delta = 0.3   # rad
    L     = PARAMS.wheelbase

    R      = L / jnp.tan(delta)
    period = 2.0 * jnp.pi * R / v0
    n      = int(period / DT)

    s0   = _state(vx=v0)
    ctrl = _ctrl(a=0.0, delta=float(delta))
    sf   = _simulate(s0, ctrl, n_steps=n)

    dp  = float(jnp.linalg.norm(sf.p_W - s0.p_W))
    dth = float(sf.theta % (2.0 * jnp.pi))
    dth = min(dth, 2.0 * jnp.pi - dth)   # distance to 0 mod 2pi
    dv  = float(jnp.abs(jnp.linalg.norm(sf.v_W) - v0))

    _check("circle / position closure",
           dp  < 1e-2,
           f"|dp| = {dp:.4f} m  (R={float(R):.3f} m, T={float(period):.3f} s)")
    _check("circle / heading closure",
           dth < 1e-2,
           f"|dtheta| = {dth:.4f} rad")
    _check("circle / speed conserved",
           dv  < 1e-4,
           f"|dv| = {dv:.2e} m/s")


# ---------------------------------------------------------------------------
# Test 4: Circle radius accuracy
# ---------------------------------------------------------------------------

def test_circle_radius() -> None:
    """
    Sample the arc and confirm every point is within tolerance of the
    Ackermann-predicted radius from the turn centre.

    Starting at origin heading +x with left turn, the centre is at (0, R).
    """
    v0    = 1.0
    delta = 0.3
    L     = PARAMS.wheelbase

    R      = float(L / jnp.tan(delta))
    centre = jnp.array([0.0, R])

    s    = _state(vx=v0)
    ctrl = _ctrl(a=0.0, delta=delta)

    errs = []
    for _ in range(200):
        s = _step_jit(s, ctrl, PARAMS, DT)
        r = float(jnp.linalg.norm(s.p_W - centre))
        errs.append(abs(r - R))

    max_err = max(errs)
    _check("circle / radius accuracy",
           max_err < 1e-4,
           f"max |r - R| = {max_err:.2e} m  (R={R:.3f} m)")


# ---------------------------------------------------------------------------
# Test 5: Jacobian vs central finite differences
# ---------------------------------------------------------------------------

def test_jacobian_fd() -> None:
    """
    linearise() (jacfwd through RK4) must agree with a central FD Jacobian
    to well within the FD truncation error O(eps_fd^2).
    """
    s0   = _state(px=0.1, py=-0.2, vx=1.2, vy=0.3, theta=0.4)
    ctrl = _ctrl(a=0.3, delta=0.15)

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

    _check("jacobian / max |F_ad - F_fd|",
           max_err < 1e-7,
           f"max entry error = {max_err:.2e}  (eps_fd={eps_fd:.0e})")
    _check("jacobian / ||F - I||_F is O(dt)",
           norm_FmI < 0.5,
           f"||F - I||_F = {norm_FmI:.4f}")


# ---------------------------------------------------------------------------
# Test 6: Heading-velocity drift
# ---------------------------------------------------------------------------

def test_heading_velocity_consistency() -> None:
    """
    Along any trajectory with nonzero speed, theta should track
    atan2(v_W) to within O(eps_v / v) ~ 1e-3 rad.
    """
    s    = _state(vx=1.0)
    ctrl = _ctrl(a=0.2, delta=0.25)

    max_err = 0.0
    for _ in range(200):
        s = _step_jit(s, ctrl, PARAMS, DT)
        theta_v = float(jnp.arctan2(s.v_W[1], s.v_W[0]))
        theta_s = float(s.theta)
        diff = (theta_s - theta_v + jnp.pi) % (2*jnp.pi) - jnp.pi
        max_err = max(max_err, abs(float(diff)))

    _check("heading / consistency with v_W",
           max_err < 5e-3,
           f"max |theta - atan2(v_W)| = {max_err:.2e} rad")


# ---------------------------------------------------------------------------
# Test 7: Pytree / register_dataclass smoke tests
# ---------------------------------------------------------------------------

def test_pytree_registration() -> None:
    """
    Verify that register_dataclass wired everything up correctly:
      (a) jax.tree.leaves returns only the JAX array fields
      (b) jax.tree.map round-trips cleanly
      (c) dataclasses.replace works (meta fields are static, not leaves)
      (d) jax.jit traces through CarState without issues
      (e) jax.vmap batches over a stacked CarState
    """
    s = _state(px=1.0, py=2.0, vx=3.0, vy=4.0, theta=0.5)

    # (a) leaves — should be [p_W, v_W, theta], i.e. 3 arrays
    leaves = jax.tree.leaves(s)
    _check("pytree / leaf count",
           len(leaves) == 3,
           f"got {len(leaves)} leaves (want 3: p_W, v_W, theta)")

    # (b) tree.map round-trip: multiply all leaves by 1.0
    s2 = jax.tree.map(lambda x: x * 1.0, s)
    err = float(jnp.max(jnp.abs(state_to_vec(s2) - state_to_vec(s))))
    _check("pytree / tree.map round-trip",
           err == 0.0,
           f"max deviation = {err:.2e}")

    # (c) dataclasses.replace on a data field (should produce a new CarState)
    s3 = replace(s, theta=jnp.array(0.0))
    _check("pytree / dataclass replace",
           float(s3.theta) == 0.0 and float(s3.p_W[0]) == 1.0,
           f"theta={float(s3.theta)}, px={float(s3.p_W[0])}")

    # (d) AckermannParams: static fields must NOT appear as leaves
    p_leaves = jax.tree.leaves(PARAMS)
    _check("pytree / params has no leaves (all static)",
           len(p_leaves) == 0,
           f"got {len(p_leaves)} leaves (want 0)")

    # (e) vmap: batch step_rk4 over a stack of initial states
    batch_size = 4
    s_batch = jax.tree.map(
        lambda x: jnp.stack([x] * batch_size),
        s,
    )
    ctrl = _ctrl(a=0.5, delta=0.1)

    # vmap over axis 0 of all state leaves
    step_vmap = jax.vmap(lambda si: step_rk4(si, ctrl, PARAMS, DT))
    sf_batch  = step_vmap(s_batch)

    _check("pytree / vmap step_rk4",
           sf_batch.p_W.shape == (batch_size, 2),
           f"p_W.shape = {sf_batch.p_W.shape}  (want ({batch_size}, 2))")

    # All rows identical (same initial state)
    row_spread = float(jnp.max(jnp.std(sf_batch.p_W, axis=0)))
    _check("pytree / vmap rows identical",
           row_spread == 0.0,
           f"std across batch rows = {row_spread:.2e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n-- car.py Layer 1 test suite ------------------------------------------")

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

    print("\n[6] Heading-velocity consistency")
    test_heading_velocity_consistency()

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
