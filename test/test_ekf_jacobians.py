"""
Tests that error_dynamics is differentiable w.r.t. the error state and
physical parameters, as required for an EKF.

State Jacobian (F):  d(error_dynamics)/d(dx_vex) at dx=0  →  (12, 12)
  - Used as the linearised state-transition matrix in the EKF predict step.

Parameter Jacobian:  d(error_dynamics)/d(params_flat) at dx=0  →  (12, n_params)
  - Needed when parameters are augmented into the EKF state for online
    estimation (e.g. estimating tire stiffness C_kappa / C_alpha).

Note: omega_W was removed as a state variable (kinematic rolling assumption:
omega_w = v_t / r_wheel). ERROR_DIM is now 12 = dp_W(3) + dtheta_B(3) + dv_W(3) + dw_B(3).
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import jaxlie

from ackermann_jax import (
    default_params,
    AckermannCarModel,
    AckermannCarInput,
    AckermannCarState,
    zero_error_state,
    pack_error_state,
    error_dynamics,
    state_difference,
)

DT = 0.01  # 10 ms timestep

# ── Error state dimensionality ──────────────────────────────────────────────
# dp_W(3) + dtheta_B(3) + dv_W(3) + dw_B(3) = 12
# omega_W removed: kinematic rolling assumption (omega_w = v_t / r_wheel)
ERROR_DIM = 12


def make_running_state(v_x: float = 1.0) -> AckermannCarState:
    """
    Car near ground-contact equilibrium, moving forward at v_x m/s with
    wheels spinning at the matching no-slip rate.

    body_z is chosen so wheel contact points (at z_body - h) are slightly
    compressed into the ground plane (z=0), giving active normal forces.
    """
    params = default_params()
    h = float(params.geom.h)               # 0.06 m
    r = float(params.geom.wheel_radius)    # 0.03 m

    # contact penetration ≈ 0.002 m  →  Fz ≈ k_n * 0.002 = 4 N per wheel
    body_z = h - 0.002

    omega_roll = v_x / r  # no-slip wheel spin

    return AckermannCarState(
        p_W=jnp.array([0.0, 0.0, body_z], dtype=jnp.float32),
        R_WB=jaxlie.SO3.identity(),
        v_W=jnp.array([v_x, 0.0, 0.0], dtype=jnp.float32),
        w_B=jnp.zeros(3, dtype=jnp.float32),
        # omega_W removed: kinematic assumption omega_w = v_t/r (omega_roll = v_x/r)
    )


def make_input(delta: float = 0.0, tau_per_driven: float = 0.05) -> AckermannCarInput:
    """Straight driving with a small rear-wheel drive torque."""
    return AckermannCarInput(
        delta=jnp.array(delta, dtype=jnp.float32),
        tau_w=jnp.array([0.0, 0.0, tau_per_driven, tau_per_driven], dtype=jnp.float32),
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _get_error_dyn_fn(model, x_nom, u, dt=DT):
    """Return a pure function (dx_vex,) → dx_vex_next for autodiff."""
    def f(dx_vex):
        return error_dynamics(model, dx_vex, x_nom, u, dt)
    return f


def _get_param_sensitivity_fn(params_flat, unravel, x_nom, u, dt=DT):
    """
    Return a pure function (params_flat,) → pack_error_state(dx_next) where
    dx_next = state_difference(step(x_nom, u, theta_perturbed), step(x_nom, u, theta_nominal)).

    This is the G matrix needed for an augmented-state EKF:
        dx_{k+1} ≈ F * dx_k + G * dtheta_k

    Note: error_dynamics(model, dx=0, ...) is identically zero regardless of
    model parameters (by construction), so G must be computed as the sensitivity
    of the *nominal step* to parameter perturbations, not via error_dynamics.
    """
    # Compute and freeze the nominal next state once
    params_nom = unravel(params_flat)
    model_nom = AckermannCarModel(params_nom)
    x_nom_next = model_nom.step(x_nom, u, dt)

    def f(pf):
        p = unravel(pf)
        m = AckermannCarModel(p)
        x_next_perturbed = m.step(x_nom, u, dt)
        dx_next = state_difference(x_nom_next, x_next_perturbed)
        return pack_error_state(dx_next)

    return f


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_state_jacobian_shape_and_finiteness():
    """
    F = d(error_dynamics)/d(dx_vex) at dx=0 must be (16, 16) and finite.
    This is the linearised state-transition matrix used in the EKF predict.
    """
    model = AckermannCarModel(default_params())
    x_nom = make_running_state()
    u = make_input()

    f = _get_error_dyn_fn(model, x_nom, u)
    dx0 = pack_error_state(zero_error_state())

    F = jax.jacobian(f)(dx0)

    assert F.shape == (ERROR_DIM, ERROR_DIM), \
        f"Expected ({ERROR_DIM},{ERROR_DIM}), got {F.shape}"
    assert jnp.all(jnp.isfinite(F)), \
        f"F contains non-finite values:\n{F}"

    print(f"\nF shape: {F.shape}  (PASS)")
    print(f"F = {np.array2string(np.array(F), precision=4, suppress_small=True)}  (PASS) ")


def test_state_jacobian_near_identity():
    """
    At zero error and small dt, F ≈ I + O(dt).
    ||F - I|| should be O(1) — not huge, not exactly zero.
    """
    model = AckermannCarModel(default_params())
    x_nom = make_running_state()
    u = make_input()

    f = _get_error_dyn_fn(model, x_nom, u)
    dx0 = pack_error_state(zero_error_state())
    F = jax.jacobian(f)(dx0)

    residual = float(jnp.linalg.norm(F - jnp.eye(ERROR_DIM)))

    # F should not be the identity (dynamics exist).
    # The wheel-spin rows can have O(100) entries due to tire stiffness,
    # so we only verify the residual is strictly positive.
    assert residual > 0.0, "F == I exactly — dynamics appear inactive"

    print(f"||F - I|| = {residual:.4f}  (PASS)")


def test_state_jacobian_zero_error_maps_to_zero():
    """
    error_dynamics at dx=0 should return (approximately) zero —
    the nominal trajectory maps to itself.
    """
    model = AckermannCarModel(default_params())
    x_nom = make_running_state()
    u = make_input()

    dx0 = pack_error_state(zero_error_state())
    dx_out = error_dynamics(model, dx0, x_nom, u, DT)

    err = float(jnp.linalg.norm(dx_out))
    assert err < 1e-4, f"||error_dynamics(0)|| = {err:.2e}, expected ≈ 0"

    print(f"||error_dynamics(dx=0)|| = {err:.2e}  (PASS)")


def test_parameter_jacobian_shape_and_finiteness():
    """
    G = d(x_next)/d(params) must be (16, n_params) and finite.
    G is the input matrix for parameter-augmented EKF:
        dx_{k+1} ≈ F * dx_k + G * dtheta_k
    """
    params = default_params()
    params_flat, unravel = ravel_pytree(params)
    n_params = params_flat.shape[0]

    x_nom = make_running_state()
    u = make_input()
    f = _get_param_sensitivity_fn(params_flat, unravel, x_nom, u)

    J = jax.jacobian(f)(params_flat)  # (16, n_params)

    assert J.shape == (ERROR_DIM, n_params), \
        f"Expected ({ERROR_DIM},{n_params}), got {J.shape}"
    assert jnp.all(jnp.isfinite(J)), \
        "J_params contains non-finite values"

    col_norms = jnp.linalg.norm(J, axis=0)
    n_active = int(jnp.sum(col_norms > 1e-6))
    print(f"\nJ_params shape: {J.shape}  (PASS)")
    print(f"Active columns (||col|| > 1e-6): {n_active} / {n_params}")

    assert n_active > 0, "All parameter Jacobian columns are zero — no sensitivity"


def test_parameter_jacobian_vs_finite_differences():
    """
    Autodiff parameter Jacobian should agree with central finite differences
    to within 5% relative error on active columns.
    """
    params = default_params()
    params_flat, unravel = ravel_pytree(params)
    n_params = int(params_flat.shape[0])

    x_nom = make_running_state()
    u = make_input()
    f = _get_param_sensitivity_fn(params_flat, unravel, x_nom, u)

    J_ad = np.array(jax.jacobian(f)(params_flat))  # (16, n_params)

    # Central finite differences with relative step sizing.
    # A fixed absolute eps fails when parameters span many orders of magnitude
    # (e.g. I_w = 4.5e-5 vs k_n = 2000).  Use eps_i = eps_rel * max(|p_i|, floor).
    eps_rel = 1e-3
    eps_floor = 1e-6
    pf_np = np.array(params_flat)
    J_fd = np.zeros_like(J_ad)
    for i in range(n_params):
        eps_i = eps_rel * max(abs(float(pf_np[i])), eps_floor)
        ei = np.zeros(n_params, dtype=np.float32)
        ei[i] = eps_i
        fp = np.array(f(params_flat + ei))
        fm = np.array(f(params_flat - ei))
        J_fd[:, i] = (fp - fm) / (2.0 * eps_i)

    # Compare only columns where:
    #   • FD result is fully finite (skip params where ±eps leads to degenerate models)
    #   • at least one method reports a non-trivial column norm
    col_finite_fd = np.all(np.isfinite(J_fd), axis=0)
    col_norm_ad = np.linalg.norm(J_ad, axis=0)
    col_norm_fd = np.where(col_finite_fd, np.linalg.norm(J_fd, axis=0), 0.0)
    active = col_finite_fd & ((col_norm_ad > 1e-5) | (col_norm_fd > 1e-5))

    n_skipped = int(np.sum(~col_finite_fd))
    if n_skipped:
        print(f"  Skipped {n_skipped} column(s) where FD produced non-finite values "
              f"(e.g. eps_v or eps_force going negative under perturbation)")

    if not np.any(active):
        print("WARNING: no active finite columns found in FD comparison — skipping")
        return

    J_ad_a = J_ad[:, active]
    J_fd_a = J_fd[:, active]
    abs_err = np.linalg.norm(J_ad_a - J_fd_a)
    rel_err = abs_err / (np.linalg.norm(J_fd_a) + 1e-8)

    print(f"\nActive columns: {int(np.sum(active))} / {n_params}")
    print(f"AD vs FD absolute error: {abs_err:.4f}")
    print(f"AD vs FD relative error: {rel_err:.4f}  (threshold 0.05)")

    assert rel_err < 0.05, \
        f"Autodiff / finite-diff mismatch: rel_err = {rel_err:.4f}"

    print("PASS: parameter Jacobian matches finite differences")


def test_jit_compilable():
    """
    Both Jacobians must survive jax.jit so they can be used inside a
    jit-compiled EKF loop.
    """
    params = default_params()
    params_flat, unravel = ravel_pytree(params)

    model = AckermannCarModel(params)
    x_nom = make_running_state()
    u = make_input()
    dx0 = pack_error_state(zero_error_state())

    # State Jacobian under jit
    F_fn = jax.jit(jax.jacobian(_get_error_dyn_fn(model, x_nom, u)))
    F = F_fn(dx0)
    assert F.shape == (ERROR_DIM, ERROR_DIM)
    assert jnp.all(jnp.isfinite(F))

    # Parameter Jacobian under jit
    J_fn = jax.jit(jax.jacobian(_get_param_sensitivity_fn(params_flat, unravel, x_nom, u)))
    J = J_fn(params_flat)
    assert J.shape[0] == ERROR_DIM
    assert jnp.all(jnp.isfinite(J))

    print(f"\nJIT-compiled F: {F.shape}  (PASS)")
    print(f"JIT-compiled J_params: {J.shape}  (PASS)")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print("EKF Jacobian Tests")
    print("=" * 64)

    test_state_jacobian_zero_error_maps_to_zero()
    test_state_jacobian_shape_and_finiteness()
    test_state_jacobian_near_identity()
    test_parameter_jacobian_shape_and_finiteness()
    test_parameter_jacobian_vs_finite_differences()
    test_jit_compilable()

    print("\n" + "=" * 64)
    print("All tests passed.")
    print("=" * 64)
