"""
diagnose_mpc.py — Systematic MPC pipeline diagnostics.

Runs every stage of the MPC pipeline in isolation and prints a PASS/WARN/FAIL
verdict for each.  Run this before test_mpc_sinusoidal.py to find exactly
which component is broken.

Stages tested (in pipeline order):
  D1  Reference trajectory sanity
  D2  state_difference / pack_error_state convention
  D3  Jacobian F — structure and eigenvalue spectrum
  D4  Jacobian G — structure and input→state coupling by DOF
  D5  Prediction matrices Phi and Theta — growth, rank, sparsity
  D6  Theta column structure — does position respond to inputs at all?
  D7  Cost matrices Q_bar / R_bar — weights, definiteness
  D8  QP gradient q = 2 Theta^T Q_bar Phi dx0 — is it nonzero?
  D9  Unconstrained solve — U* with known nonzero dx0
  D10 Constraint feasibility — do box/slew constraints admit U* = 0?
  D11 Full mpc_step end-to-end with synthetic perturbation
  D12 Closed-loop plant response — does applying u_opt[0] move the car?
"""

from __future__ import annotations

import sys
import textwrap
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla

jax.config.update("jax_enable_x64", False)

from ackermann_jax import (
    AckermannCarInput,
    AckermannCarModel,
    AckermannCarState,
    default_params,
    default_state,
)
from ackermann_jax.mpc import (
    DECISION_DIM,
    INPUT_DIM,
    MPCParams,
    MPCState,
    _build_prediction_matrices_np,
    _compute_FG,
    _solve_unconstrained,
    default_mpc_params,
    mpc_step,
)
from ackermann_jax.ekf import ERROR_DIM
from ackermann_jax.errorDyn import pack_error_state, state_difference

# ── Colour helpers ────────────────────────────────────────────────────────────
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

def _pass(msg):  print(f"  {_GREEN}PASS{_RESET}  {msg}")
def _warn(msg):  print(f"  {_YELLOW}WARN{_RESET}  {msg}")
def _fail(msg):  print(f"  {_RED}FAIL{_RESET}  {msg}")

def _header(tag, title):
    print(f"\n{_BOLD}{'─'*70}{_RESET}")
    print(f"{_BOLD}[{tag}] {title}{_RESET}")
    print(f"{_BOLD}{'─'*70}{_RESET}")

def _array_summary(name, arr, indent=4):
    pad = " " * indent
    print(f"{pad}{name}:  shape={arr.shape}  dtype={arr.dtype}")
    print(f"{pad}  min={arr.min():.4g}  max={arr.max():.4g}  "
          f"norm={np.linalg.norm(arr):.4g}  "
          f"finite={np.all(np.isfinite(arr))}")

NX = ERROR_DIM      # 16
FULL_NU = INPUT_DIM # 5
NU = DECISION_DIM   # 3

# ─────────────────────────────────────────────────────────────────────────────
# Setup — build model + reference trajectory (minimal, 5 s weave at 20 Hz)
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n{_BOLD}Setting up model and reference trajectory…{_RESET}")

params = default_params()
model  = AckermannCarModel(params)
dt     = 0.01
T_settle = 1.5
T_weave  = 5.0
v_cmd    = 1.0
psi_amp  = 0.30
psi_freq = 0.4

N_settle = int(T_settle / dt)
N_weave  = int(T_weave  / dt)
N_total  = N_settle + N_weave

x0 = default_state(z0=0.10)

ref_states: List[AckermannCarState] = [x0]
ref_inputs: List[AckermannCarInput] = []

x = x0
integ_v = integ_s = 0.0
for k in range(N_total):
    t_w   = max(0.0, k * dt - T_settle)
    v_ref = 0.0 if k < N_settle else v_cmd
    psi   = (0.0 if k < N_settle
             else psi_amp * float(jnp.sin(2 * jnp.pi * psi_freq * t_w)))

    tau_w, integ_v = model.map_velocity_to_wheel_torques(
        x=x, v_cmd=v_ref, integral_state=jnp.float32(integ_v),
        dt=dt, Kp=40., Ki=2., tau_max=0.35, integ_max=0.5,
        use_traction_limit=True)
    delta, integ_s = model.map_heading_to_steering(
        x=x, psi_cmd=psi, integral_state=jnp.float32(integ_s),
        dt=dt, Kp=3., Ki=2., Kd=0.4, delta_max=0.35, integ_max=0.5)

    u = AckermannCarInput(delta=delta, tau_w=tau_w)
    ref_inputs.append(u)
    ref_states.append(model.step(x=x, u=u, dt=dt))
    x = ref_states[-1]
    integ_v, integ_s = float(integ_v), float(integ_s)

mpc_params = default_mpc_params(N=20, dt=dt)
N = mpc_params.N

# Pick two linearisation points: boundary of settle, and mid-weave
k_settle = N_settle          # just after settle ends
k_weave  = N_settle + 50     # well into weave (v > 0, steering active)

print(f"  N_total={N_total}  N_settle={N_settle}  N_weave={N_weave}")
print(f"  Linearisation points: k_settle={k_settle}, k_weave={k_weave}")

# ─────────────────────────────────────────────────────────────────────────────
# D1 — Reference trajectory sanity
# ─────────────────────────────────────────────────────────────────────────────
_header("D1", "Reference trajectory sanity")

p_traj = np.array([s.p_W for s in ref_states])
v_traj = np.array([s.v_W for s in ref_states])
yaw    = np.array([float(s.R_WB.compute_yaw_radians()) for s in ref_states])

_array_summary("p_W (all steps)", p_traj)
_array_summary("v_W (all steps)", v_traj)

p_range = p_traj.max(axis=0) - p_traj.min(axis=0)
v_max   = np.linalg.norm(v_traj, axis=1).max()

if p_range[0] < 0.01 and p_range[1] < 0.01:
    _fail(f"Car never moved: p range = {p_range}")
else:
    _pass(f"Position range: Δx={p_range[0]:.3f} m  Δy={p_range[1]:.3f} m")

if v_max < 0.05:
    _fail(f"Velocity never exceeded 0.05 m/s (max={v_max:.4f})")
else:
    _pass(f"Max speed during trajectory: {v_max:.3f} m/s")

print(f"    p_W at k_settle : {np.array(ref_states[k_settle].p_W)}")
print(f"    v_W at k_settle : {np.array(ref_states[k_settle].v_W)}")
print(f"    p_W at k_weave  : {np.array(ref_states[k_weave].p_W)}")
print(f"    v_W at k_weave  : {np.array(ref_states[k_weave].v_W)}")

# ─────────────────────────────────────────────────────────────────────────────
# D2 — state_difference / pack_error_state convention
# ─────────────────────────────────────────────────────────────────────────────
_header("D2", "state_difference / pack_error_state convention")

x_ref  = ref_states[k_weave]

# Synthesise known perturbations
PERTURB_P = jnp.array([0.1, 0.0, 0.0], dtype=jnp.float32)  # +0.1 m in x
PERTURB_V = jnp.array([0.1, 0.0, 0.0], dtype=jnp.float32)  # +0.1 m/s in x

x_pert_p = AckermannCarState(
    p_W=x_ref.p_W + PERTURB_P, R_WB=x_ref.R_WB,
    v_W=x_ref.v_W, w_B=x_ref.w_B, omega_W=x_ref.omega_W)
x_pert_v = AckermannCarState(
    p_W=x_ref.p_W, R_WB=x_ref.R_WB,
    v_W=x_ref.v_W + PERTURB_V, w_B=x_ref.w_B, omega_W=x_ref.omega_W)

dx_p = np.array(pack_error_state(state_difference(x_pert_p, x_ref)))
dx_v = np.array(pack_error_state(state_difference(x_pert_v, x_ref)))

print(f"\n  Position perturbation +0.1 m in x  →  dx[0:3] = {dx_p[0:3]}")
print(f"  Velocity perturbation +0.1 m/s in x →  dx[6:9] = {dx_v[6:9]}")
print(f"  Full dx (position pert): {dx_p}")
print(f"  Full dx (velocity pert): {dx_v}")

# Convention check: state_difference(pert, ref) should give positive error
# when pert > ref
if abs(dx_p[0] - 0.1) < 0.01:
    _pass("state_difference(pert, ref): position error has correct sign (+)")
elif abs(dx_p[0] + 0.1) < 0.01:
    _warn("state_difference(pert, ref): position sign is FLIPPED (negative). "
          "Check argument order — should be (current, reference).")
else:
    _fail(f"position perturbation not in dx[0:3] at all — "
          f"check pack_error_state index layout. dx[0:6]={dx_p[0:6]}")

if abs(dx_v[6] - 0.1) < 0.01:
    _pass("Velocity error correctly at index 6.")
else:
    _warn(f"Velocity perturbation not clearly at index 6. dx[6:9]={dx_v[6:9]}")

# Zero test
dx_zero = np.array(pack_error_state(state_difference(x_ref, x_ref)))
if np.max(np.abs(dx_zero)) < 1e-5:
    _pass("state_difference(x, x) = 0  ✓")
else:
    _fail(f"state_difference(x, x) ≠ 0: max={np.max(np.abs(dx_zero)):.2e}")

# ─────────────────────────────────────────────────────────────────────────────
# D3 — Jacobian F
# ─────────────────────────────────────────────────────────────────────────────
_header("D3", "Jacobian F — structure and spectrum")

for label, kidx in [("settle boundary", k_settle), ("mid-weave", k_weave)]:
    F, _ = _compute_FG(model, ref_states[kidx], ref_inputs[kidx], dt)
    F_np = np.array(F, dtype=np.float64)
    eigs = np.sort(np.abs(np.linalg.eigvals(F_np)))[::-1]
    print(f"\n  [{label}]  k={kidx}")
    print(f"    F norm     : {np.linalg.norm(F_np):.4f}")
    print(f"    F finite   : {np.all(np.isfinite(F_np))}")
    print(f"    |eig| top8 : {eigs[:8].round(4)}")
    print(f"    |eig| bot4 : {eigs[-4:].round(4)}")
    if not np.all(np.isfinite(F_np)):
        _fail("F contains NaN/inf!")
    elif eigs[0] > 50:
        _warn(f"Very stiff eigenvalue |λ|={eigs[0]:.1f} — "
              "Phi will grow rapidly with horizon N")
    else:
        _pass(f"F looks healthy (max |λ|={eigs[0]:.3f})")

# ─────────────────────────────────────────────────────────────────────────────
# D4 — Jacobian G — per-DOF input coupling
# ─────────────────────────────────────────────────────────────────────────────
_header("D4", "Jacobian G — per-DOF input→state coupling")

DOF_LABELS = [
    "dp_x", "dp_y", "dp_z",       # 0-2
    "dθ_x", "dθ_y", "dθ_z",       # 3-5
    "dv_x", "dv_y", "dv_z",       # 6-8
    "dω_x", "dω_y", "dω_z",       # 9-11
    "dΩ_0", "dΩ_1", "dΩ_2","dΩ_3",# 12-15
]
INPUT_LABELS = ["δ", "τ_RL", "τ_RR"]

for label, kidx in [("settle boundary", k_settle), ("mid-weave", k_weave)]:
    _, G_full = _compute_FG(model, ref_states[kidx], ref_inputs[kidx], dt)
    G_np = np.array(G_full[:, [0, 3, 4]], dtype=np.float64)
    print(f"\n  [{label}]  k={kidx}   G shape={G_np.shape}")
    print(f"    G norm : {np.linalg.norm(G_np):.4f}")
    print(f"    G finite: {np.all(np.isfinite(G_np))}")
    print()
    # Print row norms with DOF labels
    row_norms = np.linalg.norm(G_np, axis=1)
    for i, (lbl, rn) in enumerate(zip(DOF_LABELS, row_norms)):
        bar   = "█" * int(min(rn * 20, 40))
        warn  = " ← ZERO" if rn < 1e-8 else ""
        print(f"    G[{i:2d}] {lbl:6s}  |row|={rn:.4e}  {bar}{warn}")
    # Flag which DOFs have zero coupling
    zero_rows = [DOF_LABELS[i] for i in range(NX) if row_norms[i] < 1e-8]
    nonzero   = [DOF_LABELS[i] for i in range(NX) if row_norms[i] >= 1e-8]
    if zero_rows:
        _warn(f"Zero G rows (no direct input coupling): {zero_rows}")
    _pass(f"Nonzero G rows: {nonzero}")

# ─────────────────────────────────────────────────────────────────────────────
# D5 — Prediction matrices Phi and Theta
# ─────────────────────────────────────────────────────────────────────────────
_header("D5", "Prediction matrices Phi and Theta")

# Build at mid-weave (healthiest linearisation point)
Fs_np = np.stack([
    np.array(_compute_FG(model, ref_states[k_weave+k], ref_inputs[k_weave+k], dt)[0],
             dtype=np.float64)
    for k in range(N)
])
Gs_np = np.stack([
    np.array(_compute_FG(model, ref_states[k_weave+k], ref_inputs[k_weave+k], dt)[1][:, [0, 3, 4]],
             dtype=np.float64)
    for k in range(N)
])

Phi, Theta = _build_prediction_matrices_np(Fs_np, Gs_np)
Phi = np.array(Phi, dtype=np.float64)
Theta = np.array(Theta, dtype=np.float64)

_array_summary("Phi  ", Phi)
_array_summary("Theta", Theta)

if not np.all(np.isfinite(Phi)):
    _fail("Phi contains NaN/inf — stiff eigenvalues are overflowing even in float64")
elif np.max(np.abs(Phi)) > 1e6:
    _warn(f"Phi has very large entries (max={np.max(np.abs(Phi)):.2e}) — "
          "reduce N or zero out stiff Q rows")
else:
    _pass(f"Phi finite and bounded (max={np.max(np.abs(Phi)):.2e})")

if not np.all(np.isfinite(Theta)):
    _fail("Theta contains NaN/inf")
elif np.linalg.norm(Theta) < 1e-10:
    _fail("Theta is essentially zero — inputs have NO effect on predicted states")
else:
    _pass(f"Theta nonzero (norm={np.linalg.norm(Theta):.4f})")

# Rank
rank_Theta = np.linalg.matrix_rank(Theta, tol=1e-8)
print(f"    Theta rank : {rank_Theta} / {min(Theta.shape)}")
if rank_Theta < NU:
    _warn(f"Theta rank ({rank_Theta}) < nu ({NU}) — system may be uncontrollable "
          "in the lifted sense")

# ─────────────────────────────────────────────────────────────────────────────
# D6 — Theta column structure: which DOFs respond to inputs?
# ─────────────────────────────────────────────────────────────────────────────
_header("D6", "Theta column structure — per-DOF response to inputs")

print("  Row-block norms of Theta  (each block = one prediction step × one DOF group)")
print("  Format:  step | DOF group | norm-of-that-block-row")
print()

# For each prediction step k, look at each DOF
for k in [0, 1, 2, 5, 10, 19]:
    row_start = k * NX
    block     = Theta[row_start:row_start + NX, :]   # (NX, N*NU)
    row_norms = np.linalg.norm(block, axis=1)
    nonzero_dofs = [(DOF_LABELS[i], f"{row_norms[i]:.3e}")
                    for i in range(NX) if row_norms[i] > 1e-8]
    zero_dofs    = [DOF_LABELS[i] for i in range(NX) if row_norms[i] <= 1e-8]
    print(f"  Step k={k:2d}:")
    print(f"    Nonzero DOFs : {nonzero_dofs}")
    print(f"    Zero DOFs    : {zero_dofs}")

# Critical check: position rows
pos_rows_of_Theta = Theta[[i for i in range(N * NX) if i % NX in (0, 1, 2)], :]
pos_norm = np.linalg.norm(pos_rows_of_Theta)
print(f"\n  Norm of ALL position rows in Theta: {pos_norm:.4e}")
if pos_norm < 1e-8:
    _fail("Position rows of Theta are ALL zero — "
          "inputs can NEVER affect predicted position. "
          "This means Q_bar @ Theta = 0 for position weights → q = 0 always.")
    print(textwrap.dedent("""
      Root cause candidates:
        (a) Integrator uses explicit Euler for position (p_{k+1} = p_k + dt*v_k),
            so ∂p/∂u = 0 at step k=0. Position only couples via F^j @ G for j>=1.
            Check if Q is penalising the *right* step — it should see those later rows.
        (b) state_difference returns error in body frame not world frame,
            placing position error in a slot that Q zeroes out.
        (c) pack_error_state reorders components differently than Q_diag assumes.
    """).strip())
else:
    _pass(f"Position rows of Theta have nonzero norm ({pos_norm:.4e})")

# Velocity rows
vel_rows_of_Theta = Theta[[i for i in range(N * NX) if i % NX in (6, 7, 8)], :]
vel_norm = np.linalg.norm(vel_rows_of_Theta)
print(f"  Norm of ALL velocity rows in Theta: {vel_norm:.4e}")
if vel_norm < 1e-8:
    _fail("Velocity rows of Theta are ALL zero")
else:
    _pass(f"Velocity rows of Theta nonzero ({vel_norm:.4e})")

# ─────────────────────────────────────────────────────────────────────────────
# D7 — Cost matrices
# ─────────────────────────────────────────────────────────────────────────────
_header("D7", "Cost matrices Q_bar and R_bar")

Q_np  = np.array(mpc_params.Q,  dtype=np.float64)
Pf_np = np.array(mpc_params.Pf, dtype=np.float64)
R_np  = np.array(mpc_params.R,  dtype=np.float64)
Q_bar = sla.block_diag(*([Q_np] * (N - 1) + [Pf_np]))
R_bar = sla.block_diag(*([R_np] * N))

q_diag = np.diag(Q_np)
print(f"  Q diagonal  : {q_diag.round(3)}")
print(f"  Pf diagonal : {np.diag(Pf_np).round(3)}")
print(f"  R diagonal  : {np.diag(R_np).round(3)}")

# Check positive definiteness
R_eigs = np.linalg.eigvalsh(R_np)
Q_eigs = np.linalg.eigvalsh(Q_np)
if R_eigs.min() <= 0:
    _fail(f"R is NOT positive definite (min eig={R_eigs.min():.4e})")
else:
    _pass(f"R ≻ 0  (min eig={R_eigs.min():.4e})")
if Q_eigs.min() < -1e-8:
    _fail(f"Q has negative eigenvalue ({Q_eigs.min():.4e})")
else:
    _pass(f"Q ≽ 0  (min eig={Q_eigs.min():.4e}, "
          f"{(q_diag == 0).sum()} zero diagonal entries)")

# Show which DOFs Q actually penalises
penalised = [DOF_LABELS[i] for i in range(NX) if q_diag[i] > 0]
zeroed    = [DOF_LABELS[i] for i in range(NX) if q_diag[i] == 0]
print(f"  Q-penalised DOFs : {penalised}")
print(f"  Q-zeroed DOFs    : {zeroed}")

# ─────────────────────────────────────────────────────────────────────────────
# D8 — QP gradient q with a known nonzero dx0
# ─────────────────────────────────────────────────────────────────────────────
_header("D8", "QP gradient q = 2·Θᵀ·Q̄·Φ·dx0  (with synthetic dx0)")

# Use mid-weave reference, perturb position by 0.1 m in x
x_ref_w = ref_states[k_weave]
x_pert  = AckermannCarState(
    p_W=x_ref_w.p_W + jnp.array([0.1, 0.0, 0.0], dtype=jnp.float32),
    R_WB=x_ref_w.R_WB, v_W=x_ref_w.v_W,
    w_B=x_ref_w.w_B, omega_W=x_ref_w.omega_W)

dx0_np = np.array(pack_error_state(state_difference(x_pert, x_ref_w)), dtype=np.float64)

Phi_dx0     = Phi @ dx0_np
QbarPhi_dx0 = Q_bar @ Phi_dx0
TtQPhi_dx0  = Theta.T @ QbarPhi_dx0
q_vec       = 2.0 * TtQPhi_dx0

print(f"  dx0          : {dx0_np.round(4)}")
print(f"  ||dx0||      : {np.linalg.norm(dx0_np):.4e}")
print(f"  ||Phi@dx0||  : {np.linalg.norm(Phi_dx0):.4e}")
print(f"  ||(Q_bar@Phi@dx0)[pos]||  : "
      f"{np.linalg.norm(QbarPhi_dx0[[i for i in range(N*NX) if i%NX in (0,1,2)]]):.4e}")
print(f"  ||Theta^T @ Q_bar @ Phi @ dx0|| : {np.linalg.norm(TtQPhi_dx0):.4e}")
print(f"  ||q||        : {np.linalg.norm(q_vec):.4e}")
print(f"  q (reshaped per horizon step):")
print(f"  {q_vec.reshape(N, NU).round(6)}")

if np.linalg.norm(q_vec) < 1e-10:
    _fail("q is zero — QP has no gradient, U* = 0 is trivially optimal")
    # Drill down to find where it dies
    if np.linalg.norm(Phi_dx0) < 1e-10:
        _fail("  ↳ Phi @ dx0 = 0  (Phi nulls the error direction)")
    elif np.linalg.norm(QbarPhi_dx0) < 1e-10:
        _fail("  ↳ Q_bar zeros out Phi@dx0  "
              "(error lives in a zero-weight DOF of Q)")
    else:
        _fail("  ↳ Theta^T kills Q_bar@Phi@dx0  "
              "(inputs have no controllability over the penalised states)")
else:
    _pass(f"q is nonzero (norm={np.linalg.norm(q_vec):.4e})")

# ─────────────────────────────────────────────────────────────────────────────
# D9 — Unconstrained solve
# ─────────────────────────────────────────────────────────────────────────────
_header("D9", "Unconstrained solve U* = -H⁻¹q")

H   = Theta.T @ Q_bar @ Theta + R_bar
P   = H + H.T + 1e-8 * np.eye(N * NU)

H_eigs = np.linalg.eigvalsh(H)
print(f"  H eig range : [{H_eigs.min():.4e}, {H_eigs.max():.4e}]")
print(f"  H cond      : {np.linalg.cond(H):.4e}")

if H_eigs.min() <= 0:
    _fail(f"H is NOT positive definite (min eig={H_eigs.min():.4e})")
else:
    _pass(f"H ≻ 0")

U_star = _solve_unconstrained(P, q_vec)
print(f"  U* (reshaped per horizon step):")
print(f"  {U_star.reshape(N, NU).round(6)}")

if np.linalg.norm(U_star) < 1e-8:
    _fail("U* ≈ 0  (either q=0 or H is singular)")
else:
    _pass(f"U* nonzero (norm={np.linalg.norm(U_star):.4e})")
    print(f"  U*[0] (first optimal input deviation): {U_star[:NU].round(6)}")

# ─────────────────────────────────────────────────────────────────────────────
# D10 — Constraint feasibility
# ─────────────────────────────────────────────────────────────────────────────
_header("D10", "Constraint feasibility — do box/slew limits admit U*?")

if mpc_params.u_min is None and mpc_params.u_max is None and mpc_params.du_max is None:
    _warn("No constraints configured — solver uses unconstrained path only")
else:
    u_nom_flat = np.zeros(N * NU, dtype=np.float64)  # ref input deviations = 0

    violations = []

    if mpc_params.u_min is not None or mpc_params.u_max is not None:
        u_min_np = np.array(mpc_params.u_min, dtype=np.float64) if mpc_params.u_min is not None else np.full(NU, -np.inf)
        u_max_np = np.array(mpc_params.u_max, dtype=np.float64) if mpc_params.u_max is not None else np.full(NU,  np.inf)
        # Box constraint on delta-u: u_min - u_nom <= du <= u_max - u_nom
        lo = np.tile(u_min_np, N) - u_nom_flat
        hi = np.tile(u_max_np, N) - u_nom_flat
        viol = np.sum((U_star < lo - 1e-6) | (U_star > hi + 1e-6))
        print(f"  Box constraint violations in U*: {viol}/{N*NU}")
        if viol:
            violations.append(f"box ({viol} entries)")
        else:
            _pass("U* satisfies box constraints")

    if mpc_params.du_max is not None:
        du_max_np = np.array(mpc_params.du_max, dtype=np.float64)
        D = (np.eye(N * NU, dtype=np.float64)
             - np.eye(N * NU, k=-NU, dtype=np.float64))
        DU = D @ U_star
        du_tiled = np.tile(du_max_np, N)
        viol = np.sum(np.abs(DU) > du_tiled + 1e-6)
        # Check if du_max is so tight that it forces U*≈0
        max_du = np.max(np.abs(DU))
        print(f"  Slew violation in U*: {viol}/{N*NU}  "
              f"(max |ΔU|={max_du:.4e}, du_max={du_max_np})")
        if viol:
            violations.append(f"slew ({viol} entries)")
            _warn(f"Slew constraint active — OSQP will clip U*. "
                  f"If du_max << |U*[0]|, the first step gets clamped near zero.")
        else:
            _pass("U* satisfies slew constraints")

        # Would clamping force u_opt[0] ≈ 0?
        if np.all(du_max_np < np.abs(U_star[:NU])):
            _warn(f"du_max={du_max_np} < |U*[0]|={np.abs(U_star[:NU]).round(5)} — "
                  f"first step is slew-rate limited to near zero on first call "
                  f"(warm_start=None → du_prev=0)")

    if violations:
        _warn(f"Active constraint violations: {violations}")

# ─────────────────────────────────────────────────────────────────────────────
# D11 — Full mpc_step end-to-end
# ─────────────────────────────────────────────────────────────────────────────
_header("D11", "Full mpc_step end-to-end with synthetic perturbation")

mpc_state_test = MPCState(
    x_ref=ref_states[k_weave : k_weave + N + 1],
    u_ref=ref_inputs[k_weave : k_weave + N],
    u_warm=None,
)

for label, x_in in [("zero error (x_current = x_ref)", x_ref_w),
                     ("position perturbed +0.1 m x",   x_pert)]:
    result, _ = mpc_step(model, mpc_state_test, mpc_params, x_in)
    du0 = np.array(result.du_opt[0])
    u0  = np.array(result.u_opt[0])
    print(f"\n  [{label}]")
    print(f"    solved   : {result.solved}")
    print(f"    QP cost  : {result.cost:.6e}")
    print(f"    du_opt[0]: {du0.round(6)}")
    print(f"    u_opt[0] : {u0.round(6)}")
    if np.linalg.norm(du0) < 1e-8 and label != "zero error (x_current = x_ref)":
        _fail("du_opt[0] = 0 despite nonzero error — MPC is not correcting")
    elif label == "zero error (x_current = x_ref)":
        if np.linalg.norm(du0) < 1e-6:
            _pass("du_opt[0] ≈ 0 when error = 0  ✓")
        else:
            _warn(f"Nonzero du_opt[0] even with zero error: {du0}")
    else:
        _pass(f"du_opt[0] nonzero for perturbed state  ✓")

# ─────────────────────────────────────────────────────────────────────────────
# D12 — Closed-loop plant response
# ─────────────────────────────────────────────────────────────────────────────
_header("D12", "Closed-loop plant response — does u_opt[0] move the car?")

result_pert, _ = mpc_step(model, mpc_state_test, mpc_params, x_pert)
u_apply = AckermannCarInput(
    delta=result_pert.u_opt[0, 0],
    tau_w=result_pert.u_opt[0, 1:5],
)
u_zero = AckermannCarInput(
    delta=jnp.float32(0.0),
    tau_w=jnp.zeros(4, dtype=jnp.float32),
)

x_next_mpc  = model.step(x=x_pert, u=u_apply, dt=dt)
x_next_zero = model.step(x=x_pert, u=u_zero,  dt=dt)
x_next_ref  = model.step(x=x_ref_w, u=ref_inputs[k_weave], dt=dt)

print(f"  Applied u  : delta={float(u_apply.delta):.5f}  tau_w={np.array(u_apply.tau_w).round(5)}")
print(f"  p_W after MPC  input : {np.array(x_next_mpc.p_W).round(5)}")
print(f"  p_W after zero input : {np.array(x_next_zero.p_W).round(5)}")
print(f"  p_W after ref  input : {np.array(x_next_ref.p_W).round(5)}")
print(f"  Δp (MPC vs zero)     : {np.array(x_next_mpc.p_W - x_next_zero.p_W).round(6)}")

if np.linalg.norm(np.array(x_next_mpc.p_W - x_next_zero.p_W)) < 1e-8:
    _warn("MPC input produces same next state as zero input — "
          "either du_opt=0 or the plant is insensitive to inputs at this point")
else:
    _pass("MPC input moves the car differently from zero input  ✓")

# Error reduction check
err_before = np.linalg.norm(np.array(x_pert.p_W)      - np.array(x_ref_w.p_W))
err_after  = np.linalg.norm(np.array(x_next_mpc.p_W)  - np.array(x_next_ref.p_W))
print(f"  Position error before: {err_before:.5f} m")
print(f"  Position error after : {err_after:.5f} m")
if err_after < err_before:
    _pass(f"Error reduced by MPC: {err_before:.5f} → {err_after:.5f} m  ✓")
else:
    _warn(f"Error NOT reduced (before={err_before:.5f}, after={err_after:.5f}). "
          "MPC may be steering in the wrong direction or doing nothing.")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{_BOLD}{'═'*70}{_RESET}")
print(f"{_BOLD}DIAGNOSTIC COMPLETE — review FAIL/WARN lines above{_RESET}")
print(f"{_BOLD}{'═'*70}{_RESET}\n")
