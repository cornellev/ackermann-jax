"""
test_ekf.py  —  EKF validation for the body-frame Ackermann bicycle model.

State vector: x = [px, py, θ, v, ω]

Tests
─────
1.  Jacobian sanity      : H matrices are constant, F is structured
2.  Identity update      : perfect measurement → state unchanged
3.  Covariance shrinkage : update reduces uncertainty
4.  Predict drift        : covariance grows over prediction steps
5.  Monte-Carlo NEES     : normalized NEES in χ²(5) 95% CI
6.  2-sigma bounds       : 95% of errors inside ±2σ bands
7.  RMSE & NEES metrics  : all maneuvers meet thresholds
8.  Angle wrapping       : θ stays in (−π, π] through predict/update
9.  Covariance PD        : P positive definite over full trajectory
10. JIT compilation      : all functions JIT-compile and are fast

Run with:
    PYTHONPATH=src python3 test/test_ekf.py
"""

from __future__ import annotations

import sys
import os
import math
import time
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import chi2

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ackermann_jax import (
    CarState, CarControl, AckermannParams,
    step_rk4, state_to_vec, vec_to_state, linearise,
    EKFState, EKFParams,
    ekf_predict, ekf_update_gps, ekf_update_speed,
    ekf_update_heading, ekf_update_yaw_rate,
    make_ekf_params, wrap_angle,
    h_gps, h_speed, h_heading, h_yaw_rate,
    STATE_DIM,
)

# ─────────────────────────────────────────────────────────────────────────────
# Global constants
# ─────────────────────────────────────────────────────────────────────────────

DT      = 0.05          # seconds — 20 Hz
CAR_P   = AckermannParams(wheelbase=0.257, mass=1.5)

# Measurement noise std devs (match EKF_P below)
SIG_GPS  = 0.50   # m
SIG_SPD  = 0.05   # m/s
SIG_HDG  = 0.02   # rad
SIG_GYRO = 0.02   # rad/s

# Build EKF params (slightly inflated Q for real-hardware slack)
EKF_P = make_ekf_params(
    sig_pos   = 0.10,
    sig_hdg   = 0.05,
    sig_vel   = 0.10,
    sig_omega = 0.10,
    sig_gps   = SIG_GPS,
    sig_spd   = SIG_SPD,
    sig_yaw   = SIG_HDG,
    sig_gyro  = SIG_GYRO,
)

# Initial covariance: generous uncertainty at filter start
P0_DIAG = jnp.array([0.5**2, 0.5**2, 0.1**2, 0.2**2, 0.1**2])

# JIT-compiled update / predict functions
_predict_jit = jax.jit(ekf_predict, static_argnums=(4,))
_upd_gps_jit = jax.jit(ekf_update_gps)
_upd_spd_jit = jax.jit(ekf_update_speed)
_upd_hdg_jit = jax.jit(ekf_update_heading)
_upd_gyr_jit = jax.jit(ekf_update_yaw_rate)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _state(px: float, py: float, theta: float,
           v: float, omega: float) -> CarState:
    return CarState(
        p_W   = jnp.array([px, py]),
        theta = jnp.array(theta),
        v     = jnp.array(v),
        omega = jnp.array(omega),
    )


def _ctrl(a: float, delta: float) -> CarControl:
    return CarControl(a=jnp.array(a), delta=jnp.array(delta))


def _init_est(true_vec: jax.Array, rng: np.random.Generator) -> EKFState:
    """Perturb true state to get an initial filter estimate."""
    noise = rng.standard_normal(STATE_DIM) * np.sqrt(np.array(P0_DIAG))
    mean  = true_vec + jnp.array(noise)
    mean  = mean.at[2].set(wrap_angle(mean[2]))   # wrap θ at index 2
    return EKFState(mean=mean, cov=jnp.diag(P0_DIAG))


# ─────────────────────────────────────────────────────────────────────────────
# Simulation + EKF loop
# ─────────────────────────────────────────────────────────────────────────────

class RunResult(NamedTuple):
    true_states : np.ndarray   # (T, 5)
    est_means   : np.ndarray   # (T, 5)
    est_covs    : np.ndarray   # (T, 5, 5)


def _run_ekf(
    controls  : list[CarControl],
    init_state: CarState,
    rng       : np.random.Generator,
    gps_rate  : int = 4,
) -> RunResult:
    """
    Simulate ground truth and run the EKF.

    Each step: predict → speed + heading + gyro updates → GPS every gps_rate steps.
    """
    T      = len(controls)
    true_s = init_state
    est    = _init_est(state_to_vec(init_state), rng)

    trues = np.zeros((T, STATE_DIM))
    means = np.zeros((T, STATE_DIM))
    covs  = np.zeros((T, STATE_DIM, STATE_DIM))

    for k, ctrl in enumerate(controls):
        # Ground truth step
        true_s = step_rk4(true_s, ctrl, CAR_P, DT)
        x_true = state_to_vec(true_s)

        # EKF predict
        est = _predict_jit(est, ctrl, CAR_P, EKF_P, DT)

        # Speed measurement (body-frame v — index 3)
        z_spd = jnp.array([float(x_true[3]) + SIG_SPD * rng.standard_normal()])
        est   = _upd_spd_jit(est, z_spd, EKF_P.R_speed)

        # Heading measurement (θ — index 2)
        z_hdg = jnp.array([float(x_true[2]) + SIG_HDG * rng.standard_normal()])
        est   = _upd_hdg_jit(est, z_hdg, EKF_P.R_heading)

        # Yaw-rate measurement (ω — index 4)
        z_gyr = jnp.array([float(x_true[4]) + SIG_GYRO * rng.standard_normal()])
        est   = _upd_gyr_jit(est, z_gyr, EKF_P.R_yaw_rate)

        # GPS every gps_rate steps
        if (k + 1) % gps_rate == 0:
            z_gps = np.array(x_true[:2]) + SIG_GPS * rng.standard_normal(2)
            est   = _upd_gps_jit(est, jnp.array(z_gps), EKF_P.R_gps)

        trues[k] = np.array(x_true)
        means[k] = np.array(est.mean)
        covs [k] = np.array(est.cov)

    return RunResult(trues, means, covs)


# ─────────────────────────────────────────────────────────────────────────────
# Maneuver generators
# ─────────────────────────────────────────────────────────────────────────────

def _make_straight(T: int = 200, v0: float = 3.0, a: float = 0.3) -> tuple:
    s0 = _state(0, 0, 0, v0, 0)
    return s0, [_ctrl(a, 0.0)] * T


def _make_circle(T: int = 400, v0: float = 3.0, delta: float = 0.20) -> tuple:
    omega0 = v0 * math.tan(delta) / CAR_P.wheelbase
    s0 = _state(0, 0, 0, v0, omega0)
    return s0, [_ctrl(0.0, delta)] * T


def _make_lane_change(T: int = 300, v0: float = 4.0) -> tuple:
    s0  = _state(0, 0, 0, v0, 0)
    T3  = T // 3
    ctrls = (
        [_ctrl(0.0,  0.15)] * T3 +
        [_ctrl(0.0, -0.15)] * T3 +
        [_ctrl(0.0,  0.00)] * (T - 2 * T3)
    )
    return s0, ctrls


def _make_combined(T: int = 500) -> tuple:
    s0  = _state(0, 0, 0, 1.0, 0)
    T4  = T // 4
    ctrls = (
        [_ctrl( 1.0,  0.00)] * T4 +
        [_ctrl( 0.0,  0.20)] * T4 +
        [_ctrl( 0.0,  0.00)] * T4 +
        [_ctrl(-0.5,  0.00)] * (T - 3 * T4)
    )
    return s0, ctrls


def _make_low_speed(T: int = 200, v0: float = 0.3) -> tuple:
    s0  = _state(0, 0, 0, v0, 0)
    T3  = T // 3
    ctrls = (
        [_ctrl(0.0,  0.25)] * T3 +
        [_ctrl(0.0, -0.25)] * T3 +
        [_ctrl(0.0,  0.00)] * (T - 2 * T3)
    )
    return s0, ctrls


MANEUVERS = {
    "straight"   : _make_straight,
    "circle"     : _make_circle,
    "lane_change": _make_lane_change,
    "combined"   : _make_combined,
    "low_speed"  : _make_low_speed,
}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    rmse_pos   : float
    rmse_hdg   : float
    rmse_spd   : float
    rmse_omega : float
    mean_nees  : float
    frac_2sig  : float


def _compute_metrics(res: RunResult) -> Metrics:
    errors = res.est_means - res.true_states   # (T, 5)
    errors[:, 2] = np.array(wrap_angle(jnp.array(errors[:, 2])))

    rmse_pos   = float(np.sqrt(np.mean(errors[:, 0]**2 + errors[:, 1]**2)))
    rmse_hdg   = float(np.sqrt(np.mean(errors[:, 2]**2)))
    rmse_spd   = float(np.sqrt(np.mean(errors[:, 3]**2)))
    rmse_omega = float(np.sqrt(np.mean(errors[:, 4]**2)))

    T = len(errors)
    nees_vals = np.zeros(T)
    for k in range(T):
        e = errors[k]
        P = res.est_covs[k]
        try:
            nees_vals[k] = float(e @ np.linalg.solve(P, e))
        except np.linalg.LinAlgError:
            nees_vals[k] = float("nan")
    mean_nees = float(np.nanmean(nees_vals))

    T_in = T_total = 0
    for k in range(T):
        for i in range(STATE_DIM):
            sigma2 = math.sqrt(max(res.est_covs[k, i, i], 0.0))
            T_total += 1
            if abs(errors[k, i]) <= 2.0 * sigma2:
                T_in += 1
    frac_2sig = T_in / max(T_total, 1)

    return Metrics(rmse_pos, rmse_hdg, rmse_spd, rmse_omega, mean_nees, frac_2sig)


def _check_metrics(name: str, m: Metrics) -> list[str]:
    fails = []
    if m.rmse_pos   > 2.0:               fails.append(f"{name}: rmse_pos={m.rmse_pos:.3f} > 2.0 m")
    if m.rmse_hdg   > 0.20:              fails.append(f"{name}: rmse_hdg={m.rmse_hdg:.4f} > 0.20 rad")
    if m.rmse_spd   > 0.30:              fails.append(f"{name}: rmse_spd={m.rmse_spd:.3f} > 0.30 m/s")
    if m.rmse_omega > 0.30:              fails.append(f"{name}: rmse_omega={m.rmse_omega:.4f} > 0.30 rad/s")
    if m.mean_nees  > 3.0 * STATE_DIM:  fails.append(f"{name}: mean_nees={m.mean_nees:.2f} > {3.0*STATE_DIM:.1f}")
    if m.frac_2sig  < 0.85:             fails.append(f"{name}: 2σ coverage={m.frac_2sig:.3f} < 0.85")
    return fails


# ─────────────────────────────────────────────────────────────────────────────
# Monte-Carlo NEES
# ─────────────────────────────────────────────────────────────────────────────

def _mc_nees(
    init_state: CarState,
    controls  : list[CarControl],
    n_runs    : int = 30,
    seed      : int = 0,
) -> tuple[float, tuple[float, float]]:
    rng_master = np.random.default_rng(seed)
    nees_all   = []
    for _ in range(n_runs):
        rng = np.random.default_rng(rng_master.integers(0, 2**31))
        res = _run_ekf(controls, init_state, rng)
        errors = res.est_means - res.true_states
        errors[:, 2] = np.array(wrap_angle(jnp.array(errors[:, 2])))
        for k in range(len(errors)):
            e = errors[k]
            P = res.est_covs[k]
            try:
                nees_all.append(float(e @ np.linalg.solve(P, e)))
            except np.linalg.LinAlgError:
                pass

    mean_nees = float(np.mean(nees_all))
    norm_nees = mean_nees / STATE_DIM
    ci_lo = chi2.ppf(0.025, STATE_DIM) / STATE_DIM
    ci_hi = chi2.ppf(0.975, STATE_DIM) / STATE_DIM
    return norm_nees, (ci_lo, ci_hi)


# ─────────────────────────────────────────────────────────────────────────────
# Test runner
# ─────────────────────────────────────────────────────────────────────────────

_pass_count = 0
_fail_count = 0
_failures   = []


def _assert(cond: bool, msg: str):
    global _pass_count, _fail_count
    if cond:
        _pass_count += 1
        print(f"  ✓  {msg}")
    else:
        _fail_count += 1
        _failures.append(msg)
        print(f"  ✗  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Jacobian sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_jacobians():
    print("\n── Test 1: Jacobian sanity ──")
    x = jnp.array([1.0, 2.0, 0.3, 4.0, 0.5])

    H_gps = jax.jacfwd(h_gps)(x)
    _assert(H_gps.shape == (2, 5), f"H_gps shape {H_gps.shape}")
    _assert(jnp.allclose(H_gps, jnp.array([[1,0,0,0,0],[0,1,0,0,0]], dtype=float)),
            "H_gps == [[1,0,0,0,0],[0,1,0,0,0]]")

    H_spd = jax.jacfwd(h_speed)(x)
    _assert(H_spd.shape == (1, 5), f"H_speed shape {H_spd.shape}")
    _assert(jnp.allclose(H_spd, jnp.array([[0,0,0,1,0]], dtype=float)),
            "H_speed == [[0,0,0,1,0]] (constant, no singularity at low speed)")

    H_hdg = jax.jacfwd(h_heading)(x)
    _assert(H_hdg.shape == (1, 5), f"H_heading shape {H_hdg.shape}")
    _assert(jnp.allclose(H_hdg, jnp.array([[0,0,1,0,0]], dtype=float)),
            "H_heading == [[0,0,1,0,0]]")

    H_gyr = jax.jacfwd(h_yaw_rate)(x)
    _assert(H_gyr.shape == (1, 5), f"H_yaw_rate shape {H_gyr.shape}")
    _assert(jnp.allclose(H_gyr, jnp.array([[0,0,0,0,1]], dtype=float)),
            "H_yaw_rate == [[0,0,0,0,1]]")

    # F — structured non-trivial Jacobian
    state0 = _state(0, 0, 0.3, 4.0, 0.1)
    ctrl0  = _ctrl(0.5, 0.1)
    _, F   = linearise(state0, ctrl0, CAR_P, DT)
    _assert(F.shape == (5, 5), f"F shape {F.shape}")
    _assert(abs(float(F[0, 2])) > 1e-6, f"|F[px,θ]|={abs(float(F[0,2])):.4f} > 0")
    _assert(abs(float(F[0, 3])) > 1e-6, f"|F[px,v]|={abs(float(F[0,3])):.4f} > 0")
    _assert(abs(float(F[2, 4]) - DT) < 0.01, f"F[θ,ω]≈dt={DT:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Identity update
# ─────────────────────────────────────────────────────────────────────────────

def test_identity_update():
    print("\n── Test 2: Identity update ──")
    x0  = jnp.array([1.0, 2.0, 0.3, 4.0, 0.5])
    P0  = jnp.diag(jnp.array([0.01, 0.01, 0.001, 0.01, 0.01]))
    est = EKFState(mean=x0, cov=P0)
    tiny = jnp.array([[1e-8]])

    est2 = ekf_update_gps(est, x0[:2], 1e-8 * jnp.eye(2))
    _assert(jnp.allclose(est2.mean[:2], x0[:2], atol=1e-6), "GPS identity: mean[:2] unchanged")
    _assert(jnp.all(jnp.diag(est2.cov)[:2] <= jnp.diag(P0)[:2] + 1e-10),
            "GPS identity: position variance non-increasing")

    est3 = ekf_update_speed(est, x0[3:4], tiny)
    _assert(abs(float(est3.mean[3] - x0[3])) < 1e-6, "Speed identity: v unchanged")

    est4 = ekf_update_heading(est, x0[2:3], tiny)
    _assert(abs(float(est4.mean[2] - x0[2])) < 1e-6, "Heading identity: θ unchanged")

    est5 = ekf_update_yaw_rate(est, x0[4:5], tiny)
    _assert(abs(float(est5.mean[4] - x0[4])) < 1e-6, "Yaw-rate identity: ω unchanged")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Covariance shrinkage
# ─────────────────────────────────────────────────────────────────────────────

def test_cov_shrinkage():
    print("\n── Test 3: Covariance shrinkage ──")
    x0  = jnp.array([0.0, 0.0, 0.0, 3.0, 0.0])
    P0  = jnp.diag(jnp.array([1.0, 1.0, 0.1, 0.5, 0.1]))
    est = EKFState(mean=x0, cov=P0)

    for upd, R, idx, label in [
        (ekf_update_gps,      EKF_P.R_gps,      [0, 1],  "GPS→px,py"),
        (ekf_update_speed,    EKF_P.R_speed,     [3],     "Speed→v"),
        (ekf_update_heading,  EKF_P.R_heading,   [2],     "Heading→θ"),
        (ekf_update_yaw_rate, EKF_P.R_yaw_rate,  [4],     "YawRate→ω"),
    ]:
        z = x0[:2] if idx == [0, 1] else x0[idx[0]:idx[0]+1]
        e = upd(est, z, R)
        for i in idx:
            _assert(float(e.cov[i, i]) < float(P0[i, i]),
                    f"{label}: P[{i},{i}] {float(P0[i,i]):.4f} → {float(e.cov[i,i]):.4f} (↓)")
        _assert(jnp.allclose(e.cov, e.cov.T, atol=1e-10), f"{label}: cov symmetric")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Predict drift
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_drift():
    print("\n── Test 4: Predict drift ──")
    x0   = jnp.array([0.0, 0.0, 0.0, 3.0, 0.0])
    P0   = jnp.diag(jnp.array([0.01, 0.01, 0.001, 0.01, 0.001]))
    est  = EKFState(mean=x0, cov=P0)
    ctrl = _ctrl(0.5, 0.1)

    for _ in range(20):
        est = _predict_jit(est, ctrl, CAR_P, EKF_P, DT)

    _assert(float(jnp.trace(est.cov)) > float(jnp.trace(P0)),
            f"trace(P) grew after 20 predict steps ({float(jnp.trace(P0)):.4f} → {float(jnp.trace(est.cov)):.4f})")
    _assert(abs(float(est.mean[2])) <= math.pi + 1e-6,
            f"θ wrapped after predict: {float(est.mean[2]):.4f}")
    eigvals = jnp.linalg.eigvalsh(est.cov)
    _assert(bool(jnp.all(eigvals > 0)),
            f"P positive definite (min eig={float(eigvals.min()):.2e})")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Monte-Carlo NEES consistency
# ─────────────────────────────────────────────────────────────────────────────

def test_mc_nees():
    print("\n── Test 5: Monte-Carlo NEES consistency ──")
    for name, maker in [("straight", _make_straight), ("circle", _make_circle)]:
        s0, ctrls = maker()
        t0 = time.time()
        norm_nees, (ci_lo, ci_hi) = _mc_nees(s0, ctrls)
        elapsed = time.time() - t0
        _assert(ci_lo <= norm_nees <= ci_hi,
                f"{name}: norm_NEES={norm_nees:.3f} in 95% CI [{ci_lo:.3f}, {ci_hi:.3f}] ({elapsed:.1f}s)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — 2-sigma bound coverage
# ─────────────────────────────────────────────────────────────────────────────

def test_2sigma_bounds():
    print("\n── Test 6: 2-sigma bound coverage ──")
    rng = np.random.default_rng(42)
    for name, maker in MANEUVERS.items():
        s0, ctrls = maker()
        res = _run_ekf(ctrls, s0, rng)
        m   = _compute_metrics(res)
        _assert(m.frac_2sig >= 0.85, f"{name}: 2σ coverage={m.frac_2sig:.3f} ≥ 0.85")


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — RMSE & NEES for all maneuvers
# ─────────────────────────────────────────────────────────────────────────────

def test_rmse_all_maneuvers():
    print("\n── Test 7: RMSE & NEES for all maneuvers ──")
    rng = np.random.default_rng(99)
    all_fails = []
    for name, maker in MANEUVERS.items():
        s0, ctrls = maker()
        res   = _run_ekf(ctrls, s0, rng)
        m     = _compute_metrics(res)
        fails = _check_metrics(name, m)
        all_fails.extend(fails)
        status = "✓" if not fails else "✗"
        print(f"  {status}  {name:12s}  pos={m.rmse_pos:.3f}m  hdg={m.rmse_hdg:.4f}rad"
              f"  spd={m.rmse_spd:.3f}m/s  ω={m.rmse_omega:.4f}rad/s"
              f"  NEES={m.mean_nees:.1f}  2σ={m.frac_2sig:.3f}")
        for f in fails:
            print(f"       ↳ {f}")

    global _pass_count, _fail_count
    if not all_fails:
        _pass_count += 1
    else:
        _fail_count += len(all_fails)
        _failures.extend(all_fails)


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — Angle wrapping
# ─────────────────────────────────────────────────────────────────────────────

def test_angle_wrapping():
    print("\n── Test 8: Angle wrapping ──")
    for theta_init in [3.1, -3.1, 0.0, math.pi - 0.01]:
        x0  = jnp.array([0.0, 0.0, theta_init, 2.0, 0.5])
        P0  = jnp.eye(5) * 0.01
        est = EKFState(mean=x0, cov=P0)
        for _ in range(40):
            est = _predict_jit(est, _ctrl(0.0, 0.15), CAR_P, EKF_P, DT)
        theta_out = float(est.mean[2])
        _assert(abs(theta_out) <= math.pi + 1e-6,
                f"predict wrap (θ₀={theta_init:.3f}): θ={theta_out:.4f} ∈ (−π, π]")

    # Update with heading near ±π boundary
    x0  = jnp.array([0.0, 0.0, 3.10, 2.0, 0.5])
    P0  = jnp.diag(jnp.array([0.1, 0.1, 0.5, 0.1, 0.1]))
    est = EKFState(mean=x0, cov=P0)
    z   = jnp.array([-3.10])
    est2 = ekf_update_heading(est, z, jnp.array([[0.01]]))
    _assert(abs(float(est2.mean[2])) <= math.pi + 1e-6,
            f"update wrap at ±π: θ={float(est2.mean[2]):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 9 — Covariance PD over full trajectory
# ─────────────────────────────────────────────────────────────────────────────

def test_cov_pd_trajectory():
    print("\n── Test 9: Covariance PD over full trajectory ──")
    rng = np.random.default_rng(7)
    s0, ctrls = _make_combined()
    res = _run_ekf(ctrls, s0, rng)

    min_eig = np.inf
    n_violations = 0
    for k in range(len(res.est_covs)):
        eigvals = np.linalg.eigvalsh(res.est_covs[k])
        min_eig = min(min_eig, float(eigvals.min()))
        if eigvals.min() <= 0:
            n_violations += 1

    _assert(n_violations == 0,
            f"P PD at all {len(res.est_covs)} steps (min eig={min_eig:.2e})")


# ─────────────────────────────────────────────────────────────────────────────
# Test 10 — JIT compilation
# ─────────────────────────────────────────────────────────────────────────────

def test_jit_compile():
    print("\n── Test 10: JIT compilation ──")
    x0   = jnp.array([0.0, 0.0, 0.0, 3.0, 0.0])
    P0   = jnp.eye(STATE_DIM) * 0.1
    est  = EKFState(mean=x0, cov=P0)
    ctrl = _ctrl(0.5, 0.1)

    for fn, args, label in [
        (_predict_jit, (est, ctrl, CAR_P, EKF_P, DT), "ekf_predict"),
        (_upd_gps_jit, (est, x0[:2], EKF_P.R_gps),    "ekf_update_gps"),
        (_upd_spd_jit, (est, x0[3:4], EKF_P.R_speed), "ekf_update_speed"),
        (_upd_hdg_jit, (est, x0[2:3], EKF_P.R_heading),"ekf_update_heading"),
        (_upd_gyr_jit, (est, x0[4:5], EKF_P.R_yaw_rate),"ekf_update_yaw_rate"),
    ]:
        try:
            fn(*args)
            _assert(True, f"{label} JIT compiled")
        except Exception as e:
            _assert(False, f"{label} JIT failed: {e}")

    # Throughput test
    t0 = time.time()
    for _ in range(200):
        est = _predict_jit(est, ctrl, CAR_P, EKF_P, DT)
    elapsed = time.time() - t0
    _assert(elapsed < 3.0, f"200 JIT predict steps in {elapsed:.3f}s < 3.0s")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.time()

    test_jacobians()
    test_identity_update()
    test_cov_shrinkage()
    test_predict_drift()
    test_mc_nees()
    test_2sigma_bounds()
    test_rmse_all_maneuvers()
    test_angle_wrapping()
    test_cov_pd_trajectory()
    test_jit_compile()

    elapsed = time.time() - t_start
    total   = _pass_count + _fail_count
    print(f"\n{'='*60}")
    print(f"Results: {_pass_count}/{total} passed  ({elapsed:.1f}s)")
    if _failures:
        print("\nFAILURES:")
        for f in _failures:
            print(f"  ✗  {f}")
        sys.exit(1)
    else:
        print("All tests passed ✓")
        sys.exit(0)
