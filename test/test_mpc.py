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
)
from ackermann_jax.mpc import (
    MPCParams,
    MPCResult,
    MPCState,
    default_mpc_params,
    mpc_step,
)

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
    psi_amplitude: float = 0.30,
    psi_frequency: float = 0.4,
    # PI / PID controller gains (same defaults as EKF test)
    Kp_v: float = 40.0,
    Ki_v: float = 2.0,
    tau_max: float = 0.35,
    integ_max_v: float = 0.5,
    Kp_s: float = 3.0,
    Ki_s: float = 2.0,
    Kd_s: float = 0.4,
    delta_max: float = 0.35,
    integ_max_s: float = 0.5,
    method: str = "semi_implicit_euler",
):
    """
    Simulate the car with PI/PID controllers to produce the ideal reference.

    Returns
    -------
    ref_states : list[AckermannCarState]   length N_total + 1
    ref_inputs : list[AckermannCarInput]   length N_total
    logs       : dict of numpy arrays
    N_settle   : int  — number of settle steps
    """
    N_settle = int(T_settle / dt)
    N_weave  = int(T_weave  / dt)
    N_total  = N_settle + N_weave

    ref_states: List[AckermannCarState] = [x0]
    ref_inputs: List[AckermannCarInput] = []
    log_p, log_yaw, log_psi_cmd, log_delta, log_tau_w, log_t = (
        [], [], [], [], [], []
    )

    integ_v = 0.0
    integ_s = 0.0
    x = x0

    for k in range(N_total):
        t_weave = max(0.0, k * dt - T_settle)
        v_ref   = 0.0 if k < N_settle else v_cmd
        psi_cmd = (
            0.0 if k < N_settle
            else psi_amplitude * float(jnp.sin(2.0 * jnp.pi * psi_frequency * t_weave))
        )

        tau_w, integ_v = model.map_velocity_to_wheel_torques(
            x=x,
            v_cmd=v_ref,
            integral_state=jnp.float32(integ_v),
            dt=dt,
            Kp=Kp_v, Ki=Ki_v,
            tau_max=tau_max,
            integ_max=integ_max_v,
            use_traction_limit=True,
        )

        delta, integ_s = model.map_heading_to_steering(
            x=x,
            psi_cmd=psi_cmd,
            integral_state=jnp.float32(integ_s),
            dt=dt,
            Kp=Kp_s, Ki=Ki_s, Kd=Kd_s,
            delta_max=delta_max,
            integ_max=integ_max_s,
        )

        u      = AckermannCarInput(delta=delta, tau_w=tau_w)
        x_next = model.step(x=x, u=u, dt=dt, method=method)

        ref_inputs.append(u)
        ref_states.append(x_next)

        log_p.append(np.array(x.p_W))
        log_yaw.append(float(x.R_WB.compute_yaw_radians()))
        log_psi_cmd.append(psi_cmd)
        log_delta.append(float(delta))
        log_tau_w.append(np.array(tau_w))
        log_t.append(k * dt)

        x = x_next
        integ_v = float(integ_v)
        integ_s = float(integ_s)

    # Append last state quantities
    log_p.append(np.array(x.p_W))
    log_yaw.append(float(x.R_WB.compute_yaw_radians()))

    logs = {
        "p_W":     np.array(log_p),           # (N_total+1, 3)
        "yaw":     np.array(log_yaw),          # (N_total+1,)
        "psi_cmd": np.array(log_psi_cmd),      # (N_total,)
        "delta":   np.array(log_delta),        # (N_total,)
        "tau_w":   np.array(log_tau_w),        # (N_total, 4)
        "t":       np.array(log_t),            # (N_total,)
    }
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
    mpc_state = MPCState(
        x_ref=ref_states[start_idx : start_idx + N + 1],
        u_ref=ref_inputs[start_idx : start_idx + N],
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
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(ref_p[:, 0],  ref_p[:, 1],  "k--",  linewidth=1.0, label="reference")
    ax.plot(true_p[:, 0], true_p[:, 1], "b-",   linewidth=1.5, label="MPC (true)")
    ax.scatter(ref_p[0, 0],  ref_p[0, 1],  c="green",  s=60, zorder=5, label="start")
    ax.scatter(ref_p[-1, 0], ref_p[-1, 1], c="red",    s=60, zorder=5, label="end (ref)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
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
    dt            = 0.05
    T_settle      = 1.5
    T_weave       = 10.0   # shorter than EKF test; MPC loop is slower
    v_cmd         = 1.0
    psi_amplitude = 0.30   # ±0.30 rad ≈ ±17° heading oscillation
    psi_frequency = 0.4    # Hz

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
        psi_amplitude=psi_amplitude,
        psi_frequency=psi_frequency,
    )
    print(f"  done in {(time.perf_counter() - t0)*1e3:.1f} ms  "
          f"({len(ref_inputs)} steps, settle={N_settle})")

    # ── Step 2: MPC hyperparameters ───────────────────────────────────────────
    N_horizon = 20
    mpc_params = default_mpc_params(N=N_horizon, dt=dt)

    n_avail = len(ref_inputs) - N_settle - N_horizon
    print(f"\nMPC horizon N={N_horizon}, dt={dt} s")
    print(f"Closed-loop steps available: {n_avail}  ({n_avail * dt:.1f} s)")

    # ── Step 3: Warm-up / JIT compilation (first call) ────────────────────────
    print("\nWarm-up MPC step (JIT compile) …", flush=True)
    t0 = time.perf_counter()
    _warmup_state = MPCState(
        x_ref=ref_states[N_settle : N_settle + N_horizon + 1],
        u_ref=ref_inputs[N_settle : N_settle + N_horizon],
    )
    _, _ = mpc_step(model, _warmup_state, mpc_params, ref_states[N_settle])
    jax.block_until_ready(None)
    print(f"  done in {(time.perf_counter() - t0)*1e3:.1f} ms")

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
