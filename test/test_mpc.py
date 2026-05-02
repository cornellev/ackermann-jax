"""
MPC sinusoidal-weave test.

Mirrors test_ekf_sinusoidal.py in structure:
  1. Settle phase  — car drops to the ground with zero velocity command.
  2. Weave phase   — MPC tracks a pre-computed reference trajectory that
                     oscillates heading sinusoidally at *psi_frequency* Hz.

Reference trajectory generation
---------------------------------
Because mpc_step needs an (N+1)-length x_ref and N-length u_ref window
at every receding-horizon step, we first simulate the *ideal* open-loop
trajectory (same PI controllers as the EKF test), then at each MPC step
we slice the next N+1 states and N inputs from that pre-computed buffer as
the nominal reference.  This gives the MPC a meaningful linearisation
point without requiring a separate path planner.

MPC loop
---------
The loop runs in plain Python (not jax.lax.scan) because mpc_step is not
JIT-compatible end-to-end (OSQP is a Python-level call).  Each iteration:
  1.  Slice reference window from the pre-computed trajectory.
  2.  Call mpc_step → get u_opt[0], the optimal first input.
  3.  Step the *true* plant with that input (adding optional process noise).
  4.  Log everything.

Plots
------
  • XY trajectory  : reference vs. MPC-controlled path.
  • Heading        : psi_cmd / reference yaw / actual yaw over time.
  • Position error : |p_actual − p_ref| per axis, with ±2σ from MPC cost.
  • Control inputs : steering angle δ and mean wheel torque τ̄.
  • QP diagnostics : solve status and QP cost per step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from matplotlib import pyplot as plt

import addcopyfighandler  # noqa: F401  (copy-figure-on-click)

from ackermann_jax import (
    AckermannCarInput,
    AckermannCarModel,
    AckermannCarState,
    default_params,
    default_state,
    pack_error_state,
    state_difference,
)
from ackermann_jax.mpc import (
    MPCParams,
    MPCResult,
    MPCState,
    default_mpc_params,
    init_mpc_state,
    mpc_step,
    _compute_FG,
    _build_prediction_matrices_np
)
import scipy.linalg as sla

jax.config.update("jax_enable_x64", False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def yaw_from_state(x: AckermannCarState) -> float:
    return float(x.R_WB.compute_yaw_radians())


# ---------------------------------------------------------------------------
# Step 1 — Open-loop reference trajectory (same PI controllers as EKF test)
# ---------------------------------------------------------------------------
def build_reference_trajectory(
    model: AckermannCarModel,
    x0: AckermannCarState,
    *,
    dt: float = 0.05,
    T_settle: float = 1.5,
    T_weave: float = 15.0,
    v_cmd: float = 1.0,
    sine_amplitude: float = 0.50,
    sine_frequency: float = 0.2,
    tau_max: float = 0.35,
    method: str ="semi_implicit_euler"
):
    """
    Build a pure geometric sine wave reference:
        x(t) = v_cmd * t
        y(t) + A * sin(2πf t)
    """
    N_settle = int(T_settle / dt)
    N_weave  = int(T_weave  / dt)
    N_total  = N_settle + N_weave

    # 1. Settle phase
    ref_states: List[AckermannCarState] = []
    ref_inputs: List[AckermannCarInput] = []

    zero_u = AckermannCarInput(delta=jnp.array(0.0,dtype=jnp.float32), tau_w=jnp.zeros(4,dtype=jnp.float32))

    x = x0
    for _ in range(N_settle):
        x_next = model.step(x=x, u=zero_u, dt=dt, method=method)
        ref_inputs.append(zero_u)
        ref_states.append(x_next)
        x = x_next

    z_ref = float(x.p_W[2])

    x_origin = float(x.p_W[0])
    y_origin = float(x.p_W[1])

    # Helper: compute rolling wheel speeds from the reference kinematics
    def rolling_omega_for_state(x_ref: AckermannCarState, delta_ref: float):
        u_tmp = AckermannCarInput(delta=jnp.array(delta_ref,dtype=jnp.float32), tau_w=jnp.zeros(4,dtype=jnp.float32))
        diag = model.diagnostics(x_ref, u_tmp)
        return diag.v_t / model.params.geom.wheel_radius

    # Pure swine wave phase
    logs = {
        "p": [],
        "yaw": [],
        "psi_cmd": [],
        "delta": [],
        "tau_w": [],
        "t": [],
    }
    for k in range(N_settle):
        xs = ref_states[k]
        logs["p"].append(xs.p_W)
        logs["yaw"].append(yaw_from_state(xs))
        logs["psi_cmd"].append(0.0)
        logs["delta"].append(0.0)
        logs["tau_w"].append(np.zeros(4))
        logs["t"].append(k * dt)

    omega_sine = 2.0 * np.pi * sine_frequency

    for i in range(N_weave):
        t = i * dt

        x_pos = x_origin + v_cmd * t
        y_pos = y_origin + sine_amplitude * np.sin(omega_sine * t)

        x_dot = v_cmd
        y_dot = sine_amplitude * omega_sine * np.cos(omega_sine * t)
        y_ddot = -sine_amplitude * omega_sine**2 * np.sin(omega_sine * t)

        speed = np.hypot(x_dot, y_dot)

        yaw = np.arctan2(y_dot, x_dot)

        yaw_rate = (x_dot * y_ddot) / (x_dot**2 + y_dot**2)

        delta_ref = np.arctan2(model.params.geom.L * yaw_rate, speed)
        delta_ref = np.clip(delta_ref, -0.35, 0.35)

        R_WB = jaxlie.SO3.from_z_radians(jnp.array(yaw, dtype=jnp.float32))

        v_W = jnp.array([x_dot, y_dot, 0.0], dtype=jnp.float32)
        w_B = jnp.array([0.0, 0.0, yaw_rate], dtype=jnp.float32)

        # Temporary omega: overwritten by rolling_omega_for_state
        x_ref_tmp = AckermannCarState(
            p_W=jnp.array([x_pos, y_pos, z_ref], dtype=jnp.float32),
            R_WB=R_WB,
            v_W=v_W,
            w_B=w_B,
            omega_W=jnp.zeros(4,dtype=jnp.float32)
        )

        omega_roll = rolling_omega_for_state(x_ref_tmp, delta_ref)

        x_ref = AckermannCarState(
            p_W = x_ref_tmp.p_W,
            R_WB = x_ref_tmp.R_WB,
            v_W = x_ref_tmp.v_W,
            w_B = x_ref_tmp.w_B,
            omega_W = omega_roll
        ) 

        # Refernece torque: enough to balance wheel damping on driven wheels
        motor_mask = model.params.motor.mask()
        tau_w = model.params.wheels.b_w * omega_roll * motor_mask
        tau_w = jnp.clip(tau_w, -tau_max, tau_max)

        u_ref = AckermannCarInput(
            delta=jnp.array(delta_ref,dtype=jnp.float32), 
            tau_w=tau_w.astype(jnp.float32)
        )

        ref_states.append(x_ref)
        ref_inputs.append(u_ref)

        logs["p"].append(x_ref.p_W)
        logs["yaw"].append(yaw_from_state(x_ref))
        logs["psi_cmd"].append(delta_ref)
        logs["delta"].append(delta_ref)
        logs["tau_w"].append(tau_w)
        logs["t"].append((N_settle + i) * dt)

    ref_states = ref_states[:N_total]
    ref_inputs = ref_inputs[:N_total]

    return ref_states, ref_inputs, logs, N_settle


# ---------------------------------------------------------------------------
# Step 2 — MPC closed-loop simulation
# ---------------------------------------------------------------------------

def run_mpc(
    model: AckermannCarModel,
    ref_states: List[AckermannCarState],
    ref_inputs: List[AckermannCarInput],
    params: MPCParams,
    *,
    start_idx: int = 0,
    method: str = "semi_implicit_euler",
    process_noise_std: float = 0.0,
    rng_key=None,
):
    """
    Closed-loop MPC simulation starting from ref_states[start_idx].

    At each step k the reference window is:
        x_ref = ref_states[k : k + N + 1]
        u_ref = ref_inputs[k : k + N]

    The first optimal input u_opt[0] is applied to the true plant.

    Returns
    -------
    results : list[MPCResult]   — one per step
    true_states : list[AckermannCarState]
    """
    N      = params.N
    dt     = params.dt
    n_steps = len(ref_inputs) - start_idx - N   # steps where full window exists

    x_true = ref_states[start_idx]
    mpc_state = init_mpc_state(
        ref_states[start_idx : start_idx + N + 1],
        ref_inputs[start_idx : start_idx + N],
        params,
    )

    results: List[MPCResult]          = []
    true_states: List[AckermannCarState] = [x_true]

    key = rng_key if rng_key is not None else jax.random.PRNGKey(0)

    for step in range(n_steps):
        k = start_idx + step

        # Update reference window (receding horizon)
        mpc_state = MPCState(
            x_ref=ref_states[k : k + N + 1],
            u_ref=ref_inputs[k : k + N],
            u_warm=mpc_state.u_warm,
            solver=mpc_state.solver,
        )

        result, mpc_state = mpc_step(model, mpc_state, params, x_true)

        # Apply first optimal input
        u_apply = AckermannCarInput(
            delta=result.u_opt[0, 0],
            tau_w=result.u_opt[0, 1:5],
        )
        x_next = model.step(x=x_true, u=u_apply, dt=dt, method=method)

        # Optional Gaussian process noise on position + velocity
        if process_noise_std > 0.0:
            key, subkey = jax.random.split(key)
            noise = process_noise_std * jax.random.normal(subkey, (3,))
            x_next = AckermannCarState(
                p_W=x_next.p_W + noise,
                R_WB=x_next.R_WB,
                v_W=x_next.v_W,
                w_B=x_next.w_B,
                omega_W=x_next.omega_W,
            )

        results.append(result)
        true_states.append(x_next)
        x_true = x_next

    return results, true_states


# ---------------------------------------------------------------------------
# Step 3 — Plotting
# ---------------------------------------------------------------------------

def plot_mpc_results(
    ref_states: List[AckermannCarState],
    true_states: List[AckermannCarState],
    results: List[MPCResult],
    ref_logs: dict,
    start_idx: int,
    title_prefix: str = "MPC",
):
    n_steps = len(results)
    t = np.array([start_idx * ref_logs["t"][1] + i * ref_logs["t"][1]
                  for i in range(n_steps + 1)])
    # More robust: use the time step from logs
    dt = float(ref_logs["t"][1] - ref_logs["t"][0]) if len(ref_logs["t"]) > 1 else 0.05
    t  = np.arange(n_steps + 1) * dt

    # Unpack trajectories
    ref_p   = np.array([ref_states[start_idx + i].p_W for i in range(n_steps + 1)])
    true_p  = np.array([s.p_W for s in true_states])

    ref_yaw  = np.array([float(ref_states[start_idx + i].R_WB.compute_yaw_radians())
                         for i in range(n_steps + 1)])
    true_yaw = np.array([float(s.R_WB.compute_yaw_radians()) for s in true_states])

    u_opt_all  = np.array([r.u_opt[0] for r in results])   # (n_steps, 5)
    qp_cost    = np.array([r.cost     for r in results])
    qp_solved  = np.array([r.solved   for r in results])

    psi_cmd_seg = ref_logs["psi_cmd"][start_idx : start_idx + n_steps]

    # ── 1. XY trajectory ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(ref_p[:, 0],  ref_p[:, 1],  "k--",  linewidth=1.0, label="reference")
    ax.plot(true_p[:, 0], true_p[:, 1], "b-",   linewidth=1.5, label="MPC (true)")
    ax.scatter(ref_p[0, 0],  ref_p[0, 1],  c="green",  s=60, zorder=5, label="start")
    ax.scatter(ref_p[-1, 0], ref_p[-1, 1], c="red",    s=60, zorder=5, label="end (ref)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend()
    ax.set_title(f"{title_prefix} — XY trajectory")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # ── 2. Heading ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t[:-1], np.rad2deg(psi_cmd_seg), "g-",  linewidth=1.0, label="psi_cmd")
    ax.plot(t,      np.rad2deg(ref_yaw),     "k--", linewidth=1.0, label="ref yaw")
    ax.plot(t,      np.rad2deg(true_yaw),    "b-",  linewidth=1.5, label="MPC yaw")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("[deg]")
    ax.legend()
    ax.set_title(f"{title_prefix} — Heading command vs actual yaw")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # ── 3. Position error per axis ────────────────────────────────────────────
    pos_err = true_p - ref_p                   # (n_steps+1, 3)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    for i, lbl in enumerate(["x", "y", "z"]):
        axes[i].plot(t, pos_err[:, i], "b-", linewidth=1.0)
        axes[i].axhline(0, color="k", linewidth=0.4)
        axes[i].set_ylabel(f"Δ{lbl} [m]")
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("t [s]")
    axes[0].set_title(f"{title_prefix} — Position tracking error (true − ref)")
    fig.tight_layout()

    # ── 4. Control inputs ─────────────────────────────────────────────────────
    t_u = t[:-1]   # inputs have one fewer sample than states
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    axes[0].plot(t_u, np.rad2deg(u_opt_all[:, 0]), "b-", linewidth=1.0)
    axes[0].axhline(0, color="k", linewidth=0.4)
    axes[0].set_ylabel("δ [deg]")
    axes[0].set_title(f"{title_prefix} — Steering angle (MPC output)")
    axes[0].grid(True, alpha=0.3)

    tau_mean = u_opt_all[:, 1:5].mean(axis=1)
    axes[1].plot(t_u, tau_mean, "r-", linewidth=1.0, label="mean τ_w")
    for i in range(4):
        axes[1].plot(t_u, u_opt_all[:, 1 + i], alpha=0.3, linewidth=0.6)
    axes[1].axhline(0, color="k", linewidth=0.4)
    axes[1].set_ylabel("τ_w [N·m]")
    axes[1].set_xlabel("t [s]")
    axes[1].set_title(f"{title_prefix} — Wheel torques (MPC output)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()

    # ── 5. QP diagnostics ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 5))
    axes[0].semilogy(t_u, np.abs(qp_cost), "m-", linewidth=1.0)
    axes[0].set_ylabel("|QP cost|")
    axes[0].set_title(f"{title_prefix} — QP diagnostics")
    axes[0].grid(True, alpha=0.3)
    axes[1].step(t_u, qp_solved.astype(int), "g-", linewidth=1.5)
    axes[1].set_ylim(-0.1, 1.3)
    axes[1].set_yticks([0, 1], ["fallback", "solved"])
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("QP status")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()

    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    params = default_params()
    model  = AckermannCarModel(params)

    # Simulation parameters — MPC runs at 20 Hz (dt=0.05 s)
    dt            = 0.01
    T_settle      = 1.5
    T_weave       = 10.0   # shorter than EKF test; MPC loop is slower
    v_cmd         = 1.0
    sine_amplitude = 0.50   # ±0.30 rad ≈ ±17° heading oscillation
    sine_frequency = 0.20    # Hz

    x0 = default_state(z0=0.10)

    # ── Step 1: reference trajectory ─────────────────────────────────────────
    print("Building reference trajectory …", flush=True)
    t0 = time.perf_counter()
    ref_states, ref_inputs, ref_logs, N_settle = build_reference_trajectory(
        model, x0,
        dt=dt,
        T_settle=T_settle,
        T_weave=T_weave,
        v_cmd=v_cmd,
        sine_amplitude=sine_amplitude,
        sine_frequency=sine_frequency,
    )
    print(f"  done in {(time.perf_counter() - t0)*1e3:.1f} ms  "
          f"({len(ref_inputs)} steps, settle={N_settle})")

    print("Check that F and G are reasonable to begin with")
    F,G_settle = _compute_FG(model, ref_states[N_settle], ref_inputs[N_settle], dt)
    print(f"F Norm: {jnp.linalg.norm(F)}")
    print(f"G Norm: {jnp.linalg.norm(G_settle)}")
    print(f"F Eigenvalues, by magnitude: {jnp.sort(jnp.abs(jnp.linalg.eigvals(F)))[::-1]}")

    # Do the same thing, but for the weaving part
    F_weave, G_weave= _compute_FG(model, ref_states[N_settle+50],ref_inputs[N_settle+50], dt)
    print("G_settle[0:3, :] (position rows):\n", np.array(G_settle[0:3, :]))
    print("G_weave [0:3, :] (position rows):\n", np.array(G_weave [0:3, :]))
    print("G_weave [6:9, :] (velocity rows):\n", np.array(G_weave [6:9, :]))

    # ── Step 2: MPC hyperparameters ───────────────────────────────────────────
    N_horizon = 5
    mpc_params = default_mpc_params(N=N_horizon, dt=dt)

    n_avail = len(ref_inputs) - N_settle - N_horizon
    print(f"\nMPC horizon N={N_horizon}, dt={dt} s")
    print(f"Closed-loop steps available: {n_avail}  ({n_avail * dt:.1f} s)")

    # ── Step 3: Warm-up / JIT compilation (first call) ────────────────────────
    print("\nWarm-up MPC step (JIT compile) …", flush=True)
    t0 = time.perf_counter()
    _warmup_state = init_mpc_state(
        ref_states[N_settle : N_settle + N_horizon + 1],
        ref_inputs[N_settle : N_settle + N_horizon],
        mpc_params,
    )
    _, _ = mpc_step(model, _warmup_state, mpc_params, ref_states[N_settle])
    jax.block_until_ready(None)
    print(f"  done in {(time.perf_counter() - t0)*1e3:.1f} ms")

    x_pert = AckermannCarState(
        p_W=ref_states[N_settle].p_W + jnp.array([0.0, 0.1, 0.0]),
        R_WB = ref_states[N_settle].R_WB,
        v_W = ref_states[N_settle].v_W,
        w_B = ref_states[N_settle].w_B,
        omega_W = ref_states[N_settle].omega_W,
    )
    result_test, _ = mpc_step(model, _warmup_state, mpc_params, x_pert)
    print("du_opt[0] with perturbation:", result_test.du_opt[0])
    print("u_opt[0]  with perturbation:", result_test.u_opt[0])

    print("Check that position error will actually work")
    dx0 = pack_error_state(state_difference(x_pert,ref_states[N_settle]))
    print(f"dx0: {dx0}")
    print(f"dx0[0:3] (position error, should be ~[0, -0.1, 0]): {dx0[0:3]}")

    print("Check example MPC solve and see if position error lands in zeroed rows")
    N, nx, nu = mpc_params.N, 16, 5
    Fs_np = np.stack([np.array(_compute_FG(model, ref_states[N_settle+k], ref_inputs[N_settle+k], dt)[0], dtype=np.float64) for k in range(N)])
    Gs_np = np.stack([np.array(_compute_FG(model, ref_states[N_settle+k], ref_inputs[N_settle+k], dt)[1], dtype=np.float64) for k in range(N)])
    Phi, Theta = _build_prediction_matrices_np(Fs_np, Gs_np)

    Q_np  = np.array(mpc_params.Q,  dtype=np.float64)
    Pf_np = np.array(mpc_params.Pf, dtype=np.float64)
    R_np  = np.array(mpc_params.R,  dtype=np.float64)
    Q_bar = sla.block_diag(*([Q_np] * (N-1) + [Pf_np]))

    dx0_np = np.array(pack_error_state(state_difference(ref_states[N_settle],x_pert)), dtype=np.float64)

    Phi_dx0 = Phi @ dx0_np
    TtQ_Phi_dx0 = Theta.T @ Q_bar @ Phi_dx0
    q = 2.0 * TtQ_Phi_dx0
    print("dx0_np:", dx0_np)
    print("Phi @ dx0 (first 6):", Phi_dx0[:6])          # should show position propagating
    print("Q_bar @ Phi @ dx0 (first 6):", (Q_bar @ Phi_dx0)[:6])  # zeroed by Q?
    print("q norm:", np.linalg.norm(q))
    print("q:", q)
    # ── Step 4: Closed-loop MPC run ───────────────────────────────────────────
    print(f"\nRunning MPC closed-loop for {n_avail} steps …", flush=True)
    t0 = time.perf_counter()
    results, true_states = run_mpc(
        model, ref_states, ref_inputs, mpc_params,
        start_idx=N_settle,
        process_noise_std=0.0,
    )
    t_run = time.perf_counter() - t0
    n_steps = len(results)
    print(f"  done in {t_run*1e3:.1f} ms  ({t_run / n_steps * 1e3:.2f} ms/step)")

    # ── Step 5: Summary statistics ────────────────────────────────────────────
    true_p = np.array([s.p_W for s in true_states])
    ref_p  = np.array([ref_states[N_settle + i].p_W for i in range(n_steps + 1)])
    pos_err = np.linalg.norm(true_p - ref_p, axis=-1)

    n_solved  = sum(r.solved for r in results)
    n_fallback = n_steps - n_solved

    print(f"\nTracking statistics over {n_steps} steps:")
    print(f"  Position RMSE : {np.sqrt(np.mean(pos_err**2)):.4f} m")
    print(f"  Position max  : {np.max(pos_err):.4f} m")
    print(f"  QP solved     : {n_solved}/{n_steps}")
    if n_fallback:
        print(f"  QP fallbacks  : {n_fallback}")

    # ── Step 6: Plots ─────────────────────────────────────────────────────────
    plot_mpc_results(
        ref_states, true_states, results, ref_logs,
        start_idx=N_settle,
        title_prefix="MPC (sinusoidal weave)",
    )


if __name__ == "__main__":
    main()
