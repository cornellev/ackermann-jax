"""
Evaluate MPC path drift against a sinusoidal reference trajectory.

This script is intentionally standalone instead of importing test_mpc.py so it
can run in headless environments with MPLBACKEND=Agg.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from matplotlib import pyplot as plt

from ackermann_jax import (
    AckermannCarInput,
    AckermannCarModel,
    AckermannCarState,
    default_params,
    default_state,
)
from ackermann_jax.mpc import (
    DECISION_DIM,
    MPCParams,
    MPCResult,
    default_mpc_params,
    mpc_step_batched,
)

jax.config.update("jax_enable_x64", True)


def yaw_from_state(x: AckermannCarState) -> float:
    return float(x.R_WB.compute_yaw_radians())


def build_reference_trajectory(
    model: AckermannCarModel,
    x0: AckermannCarState,
    *,
    dt: float,
    T_settle: float,
    T_weave: float,
    v_cmd: float,
    sine_amplitude: float,
    sine_frequency: float,
    tau_max: float,
    method: str = "semi_implicit_euler",
):
    n_settle = int(T_settle / dt)
    n_weave = int(T_weave / dt)
    n_total = n_settle + n_weave

    ref_states: List[AckermannCarState] = []
    ref_inputs: List[AckermannCarInput] = []
    zero_u = AckermannCarInput(
        delta=jnp.array(0.0, dtype=jnp.float32),
        tau_w=jnp.zeros(4, dtype=jnp.float32),
    )

    x = x0
    for _ in range(n_settle):
        x_next = model.step(x=x, u=zero_u, dt=dt, method=method)
        ref_inputs.append(zero_u)
        ref_states.append(x_next)
        x = x_next

    z_ref = float(x.p_W[2])
    x_origin = float(x.p_W[0])
    y_origin = float(x.p_W[1])
    omega_sine = 2.0 * np.pi * sine_frequency

    logs = {"p": [], "yaw": [], "delta": [], "tau_w": [], "t": []}
    for k in range(n_settle):
        xs = ref_states[k]
        logs["p"].append(xs.p_W)
        logs["yaw"].append(yaw_from_state(xs))
        logs["delta"].append(0.0)
        logs["tau_w"].append(np.zeros(4))
        logs["t"].append(k * dt)

    def rolling_omega_for_state(x_ref: AckermannCarState, delta_ref: float):
        u_tmp = AckermannCarInput(
            delta=jnp.array(delta_ref, dtype=jnp.float32),
            tau_w=jnp.zeros(4, dtype=jnp.float32),
        )
        diag = model.diagnostics(x_ref, u_tmp)
        return diag.v_t / model.params.geom.wheel_radius

    for i in range(n_weave):
        t = i * dt
        x_pos = x_origin + v_cmd * t
        y_pos = y_origin + sine_amplitude * np.sin(omega_sine * t)
        x_dot = v_cmd
        y_dot = sine_amplitude * omega_sine * np.cos(omega_sine * t)
        y_ddot = -sine_amplitude * omega_sine**2 * np.sin(omega_sine * t)
        speed = np.hypot(x_dot, y_dot)
        yaw = np.arctan2(y_dot, x_dot)
        yaw_rate = (x_dot * y_ddot) / (x_dot**2 + y_dot**2)
        delta_ref = np.clip(np.arctan2(model.params.geom.L * yaw_rate, speed), -0.35, 0.35)

        x_ref_tmp = AckermannCarState(
            p_W=jnp.array([x_pos, y_pos, z_ref], dtype=jnp.float32),
            R_WB=jaxlie.SO3.from_z_radians(jnp.array(yaw, dtype=jnp.float32)),
            v_W=jnp.array([x_dot, y_dot, 0.0], dtype=jnp.float32),
            w_B=jnp.array([0.0, 0.0, yaw_rate], dtype=jnp.float32),
            omega_W=jnp.zeros(4, dtype=jnp.float32),
        )
        omega_roll = rolling_omega_for_state(x_ref_tmp, delta_ref)
        x_ref = AckermannCarState(
            p_W=x_ref_tmp.p_W,
            R_WB=x_ref_tmp.R_WB,
            v_W=x_ref_tmp.v_W,
            w_B=x_ref_tmp.w_B,
            omega_W=omega_roll,
        )
        tau_w = model.params.wheels.b_w * omega_roll * model.params.motor.mask()
        tau_w = jnp.clip(tau_w, -tau_max, tau_max).astype(jnp.float32)
        u_ref = AckermannCarInput(
            delta=jnp.array(delta_ref, dtype=jnp.float32),
            tau_w=tau_w,
        )

        ref_states.append(x_ref)
        ref_inputs.append(u_ref)
        logs["p"].append(x_ref.p_W)
        logs["yaw"].append(yaw_from_state(x_ref))
        logs["delta"].append(delta_ref)
        logs["tau_w"].append(tau_w)
        logs["t"].append((n_settle + i) * dt)

    return ref_states[:n_total], ref_inputs[:n_total], logs, n_settle


def run_mpc(
    model: AckermannCarModel,
    ref_states: List[AckermannCarState],
    ref_inputs: List[AckermannCarInput],
    params: MPCParams,
    *,
    start_idx: int,
    mpc_period_steps: int,
    process_noise_std: float,
    method: str = "semi_implicit_euler",
):
    n_horizon = params.N
    n_steps = len(ref_inputs) - start_idx - n_horizon
    if n_steps <= 0:
        raise ValueError("Not enough reference samples for the requested horizon")
    if mpc_period_steps < 1:
        raise ValueError("mpc_period_steps must be >= 1")

    ref_state_batch = jax.tree.map(lambda *xs: jnp.stack(xs), *ref_states)
    ref_input_batch = jax.tree.map(lambda *us: jnp.stack(us), *ref_inputs)
    x0 = jax.tree.map(lambda a: a[start_idx], ref_state_batch)
    u0 = jnp.concatenate(
        [jnp.atleast_1d(ref_input_batch.delta[start_idx]), ref_input_batch.tau_w[start_idx]]
    ).astype(jnp.float32)
    warm0 = jnp.zeros((n_horizon * DECISION_DIM,), dtype=jnp.float32)
    key0 = jax.random.PRNGKey(0)

    def _window(tree, start, length):
        return jax.tree.map(
            lambda a: jax.lax.dynamic_slice_in_dim(a, start, length, axis=0),
            tree,
        )

    def _zero_result(u_apply):
        return MPCResult(
            u_opt=jnp.tile(u_apply[None, :], (n_horizon, 1)),
            du_opt=jnp.zeros((n_horizon, 5), dtype=jnp.float32),
            x_pred=jnp.zeros((n_horizon + 1, 16), dtype=jnp.float32),
            cost=jnp.array(jnp.nan, dtype=jnp.float32),
            solved=jnp.array(False),
        )

    def _scan_body(carry, step):
        x_true, u_hold, warm, key = carry
        k = start_idx + step
        x_ref_window = _window(ref_state_batch, k, n_horizon + 1)
        u_ref_window = _window(ref_input_batch, k, n_horizon)

        def _solve(_):
            result, warm_next = mpc_step_batched(
                model, x_ref_window, u_ref_window, params, x_true, warm
            )
            return result, result.u_opt[0], warm_next

        def _hold(_):
            return _zero_result(u_hold), u_hold, warm

        result, u_apply_flat, warm_next = jax.lax.cond(
            step % mpc_period_steps == 0,
            _solve,
            _hold,
            operand=None,
        )
        u_apply = AckermannCarInput(delta=u_apply_flat[0], tau_w=u_apply_flat[1:5])
        x_next = model.step(x=x_true, u=u_apply, dt=params.dt, method=method)

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

        return (x_next, u_apply_flat, warm_next, key), (result, x_next)

    @jax.jit
    def _run_scan():
        carry0 = (x0, u0, warm0, key0)
        return jax.lax.scan(_scan_body, carry0, jnp.arange(n_steps))

    _, (result_hist, x_next_hist) = _run_scan()
    jax.block_until_ready(x_next_hist)
    jax.block_until_ready(result_hist.u_opt)

    true_state_hist = jax.tree.map(
        lambda first, rest: jnp.concatenate([first[None, ...], rest], axis=0),
        x0,
        x_next_hist,
    )
    results = [jax.tree.map(lambda a, i=i: a[i], result_hist) for i in range(n_steps)]
    true_states = [jax.tree.map(lambda a, i=i: a[i], true_state_hist) for i in range(n_steps + 1)]
    return results, true_states


def compute_metrics(ref_states, true_states, results, start_idx, dt, mpc_period_steps):
    true_p = np.array([s.p_W for s in true_states])
    ref_p = np.array([ref_states[start_idx + i].p_W for i in range(len(true_states))])
    err = true_p - ref_p
    planar = np.linalg.norm(err[:, :2], axis=1)
    worst_idx = int(np.argmax(planar))
    solve_indices = np.arange(0, len(results), mpc_period_steps)
    solved = np.array([bool(results[i].solved) for i in solve_indices], dtype=bool)

    metrics = {
        "planar_rmse_m": float(np.sqrt(np.mean(planar**2))),
        "planar_max_m": float(np.max(planar)),
        "planar_p95_m": float(np.percentile(planar, 95)),
        "planar_final_m": float(planar[-1]),
        "worst_time_s": float(worst_idx * dt),
        "worst_planar_m": float(planar[worst_idx]),
        "x_rmse_m": float(np.sqrt(np.mean(err[:, 0] ** 2))),
        "y_rmse_m": float(np.sqrt(np.mean(err[:, 1] ** 2))),
        "z_rmse_m": float(np.sqrt(np.mean(err[:, 2] ** 2))),
        "x_max_abs_m": float(np.max(np.abs(err[:, 0]))),
        "y_max_abs_m": float(np.max(np.abs(err[:, 1]))),
        "z_max_abs_m": float(np.max(np.abs(err[:, 2]))),
        "qp_solved": int(np.sum(solved)),
        "qp_requested": int(len(solve_indices)),
        "closed_loop_steps": int(len(results)),
    }
    return metrics, ref_p, true_p, err, planar


def print_report(metrics, elapsed_s):
    print("\nMPC drift evaluation")
    print("=" * 72)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:24s}: {value: .6f}")
        else:
            print(f"{key:24s}: {value}")
    print(f"{'elapsed_s':24s}: {elapsed_s: .6f}")


def plot_results(ref_p, true_p, err, planar, results, dt, save_prefix=None, show=True):
    t = np.arange(len(true_p)) * dt
    t_u = t[:-1]
    u_opt = np.array([r.u_opt[0] for r in results])

    fig_xy, ax = plt.subplots(figsize=(8, 8))
    ax.plot(ref_p[:, 0], ref_p[:, 1], "k--", linewidth=1.0, label="reference")
    ax.plot(true_p[:, 0], true_p[:, 1], "b-", linewidth=1.3, label="MPC")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("MPC path drift")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig_xy.tight_layout()

    fig_drift, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, planar, "b-", linewidth=1.0)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("planar error [m]")
    ax.set_title("Planar position drift")
    ax.grid(True, alpha=0.3)
    fig_drift.tight_layout()

    fig_axis, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    for i, label in enumerate(["x", "y", "z"]):
        axes[i].plot(t, err[:, i], linewidth=1.0)
        axes[i].axhline(0.0, color="k", linewidth=0.4)
        axes[i].set_ylabel(f"{label} error [m]")
        axes[i].grid(True, alpha=0.3)
    axes[-1].set_xlabel("t [s]")
    axes[0].set_title("Position error by axis")
    fig_axis.tight_layout()

    fig_u, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axes[0].plot(t_u, np.rad2deg(u_opt[:, 0]), "b-", linewidth=1.0)
    axes[0].set_ylabel("delta [deg]")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t_u, u_opt[:, 3], label="tau_RL", linewidth=1.0)
    axes[1].plot(t_u, u_opt[:, 4], label="tau_RR", linewidth=1.0)
    axes[1].set_xlabel("t [s]")
    axes[1].set_ylabel("torque [N*m]")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[0].set_title("MPC commands")
    fig_u.tight_layout()

    figures = {
        "xy": fig_xy,
        "drift": fig_drift,
        "axis_error": fig_axis,
        "commands": fig_u,
    }
    if save_prefix is not None:
        prefix = Path(save_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        for suffix, fig in figures.items():
            fig.savefig(f"{prefix}_{suffix}.png", dpi=150)
    if show:
        plt.show()
    else:
        plt.close("all")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plots", action="store_true", help="Skip plot creation")
    parser.add_argument("--save-prefix", type=str, default=None, help="Save plots as PREFIX_*.png")
    parser.add_argument("--T-weave", type=float, default=10.0)
    parser.add_argument("--N-horizon", type=int, default=20)
    parser.add_argument("--mpc-period-steps", type=int, default=5)
    parser.add_argument("--process-noise-std", type=float, default=0.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--T-settle", type=float, default=1.5)
    parser.add_argument("--v-cmd", type=float, default=1.0)
    parser.add_argument("--sine-amplitude", type=float, default=0.50)
    parser.add_argument("--sine-frequency", type=float, default=0.20)
    return parser.parse_args()


def main():
    args = parse_args()
    model = AckermannCarModel(default_params())
    ref_states, ref_inputs, _, n_settle = build_reference_trajectory(
        model,
        default_state(z0=0.10),
        dt=args.dt,
        T_settle=args.T_settle,
        T_weave=args.T_weave,
        v_cmd=args.v_cmd,
        sine_amplitude=args.sine_amplitude,
        sine_frequency=args.sine_frequency,
        tau_max=0.35,
    )
    mpc_params = default_mpc_params(N=args.N_horizon, dt=args.dt)

    t0 = time.perf_counter()
    results, true_states = run_mpc(
        model,
        ref_states,
        ref_inputs,
        mpc_params,
        start_idx=n_settle,
        mpc_period_steps=args.mpc_period_steps,
        process_noise_std=args.process_noise_std,
    )
    elapsed_s = time.perf_counter() - t0

    metrics, ref_p, true_p, err, planar = compute_metrics(
        ref_states, true_states, results, n_settle, args.dt, args.mpc_period_steps
    )
    print_report(metrics, elapsed_s)

    if not args.no_plots:
        plot_results(
            ref_p,
            true_p,
            err,
            planar,
            results,
            args.dt,
            save_prefix=args.save_prefix,
            show=args.save_prefix is None,
        )


if __name__ == "__main__":
    main()
