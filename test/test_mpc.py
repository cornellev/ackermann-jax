"""MPC integration test: track a straight-then-turn path.

Mirrors the manoeuvre in ``test_ekf.py`` — settle, straight, left turn — but
uses MPC instead of PID+EKF.  Plots the resulting trajectory, inputs, and
tracking error so you can visually verify the controller.

Usage::

    python test/test_mpc.py
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from ackermann_jax import (
    AckermannCarState,
    AckermannCarInput,
    AckermannCarModel,
)
from ackermann_jax.car import default_params, default_state
from ackermann_jax.errorDyn import state_difference, pack_error_state
from ackermann_jax.mpc import (
    MPCConfig,
    MPCState,
    default_mpc_config,
    mpc_init,
    mpc_step,
    pack_input,
)

import jaxlie


# ── Build a goal trajectory: straight then left turn ─────────────────────────

def make_goal_state(
    x: float, y: float, yaw: float, v: float,
) -> AckermannCarState:
    """Convenience: create a goal state at a given pose and speed."""
    R = jaxlie.SO3.from_z_radians(jnp.float32(yaw))
    R_mat = R.as_matrix()
    v_W = R_mat @ jnp.array([v, 0.0, 0.0], dtype=jnp.float32)
    return AckermannCarState(
        p_W=jnp.array([x, y, 0.03], dtype=jnp.float32),
        R_WB=R,
        v_W=v_W,
        w_B=jnp.zeros(3, dtype=jnp.float32),
        omega_W=jnp.full(4, v / 0.03, dtype=jnp.float32),  # v/r_wheel
    )


def main():
    # ── Setup ────────────────────────────────────────────────────────────
    params = default_params()
    model = AckermannCarModel(params)
    x0 = default_state()
    dt = 0.01

    # Let the car settle onto the ground first (100 steps, no input)
    print("Settling...")
    x = x0
    u_zero = AckermannCarInput(
        delta=jnp.float32(0.0),
        tau_w=jnp.zeros(4, dtype=jnp.float32),
    )
    for _ in range(100):
        x = model.step(x, u_zero, dt)
    x_settled = x
    print(f"  Settled at z = {x_settled.p_W[2]:.4f} m")

    # ── MPC config ───────────────────────────────────────────────────────
    cfg = default_mpc_config(N=15, dt=dt)

    # ── Define phases ────────────────────────────────────────────────────
    v_target = 0.5  # m/s
    n_straight = 300   # 3 seconds straight
    n_turn = 500       # 5 seconds turning left

    # ── Phase 1: Straight ────────────────────────────────────────────────
    print("Phase 1: Straight ahead...")
    goal_straight = make_goal_state(x=5.0, y=0.0, yaw=0.0, v=v_target)

    mpc_state = mpc_init(model, x_settled, u_zero, cfg)

    traj_x, traj_y, traj_yaw = [], [], []
    inputs_delta, inputs_torque = [], []
    tracking_err = []

    x = x_settled
    for step in range(n_straight):
        u_opt, mpc_state = mpc_step(model, x, goal_straight, mpc_state, cfg)
        x = model.step(x, u_opt, dt)

        # Log
        traj_x.append(float(x.p_W[0]))
        traj_y.append(float(x.p_W[1]))
        R = x.R_WB.as_matrix()
        traj_yaw.append(float(jnp.arctan2(R[1, 0], R[0, 0])))
        inputs_delta.append(float(u_opt.delta))
        inputs_torque.append(float(jnp.mean(u_opt.tau_w)))
        dx = pack_error_state(state_difference(goal_straight, x))
        tracking_err.append(float(jnp.linalg.norm(dx[:3])))  # position error

        if step % 100 == 0:
            print(f"  Step {step}: pos=({x.p_W[0]:.3f}, {x.p_W[1]:.3f}), "
                  f"v={jnp.linalg.norm(x.v_W):.3f} m/s, "
                  f"delta={u_opt.delta:.4f} rad")

    # ── Phase 2: Left turn ───────────────────────────────────────────────
    print("Phase 2: Left turn...")
    goal_turn = make_goal_state(
        x=float(x.p_W[0]) + 2.0,
        y=float(x.p_W[1]) + 2.0,
        yaw=jnp.pi / 2,
        v=v_target,
    )

    for step in range(n_turn):
        u_opt, mpc_state = mpc_step(model, x, goal_turn, mpc_state, cfg)
        x = model.step(x, u_opt, dt)

        traj_x.append(float(x.p_W[0]))
        traj_y.append(float(x.p_W[1]))
        R = x.R_WB.as_matrix()
        traj_yaw.append(float(jnp.arctan2(R[1, 0], R[0, 0])))
        inputs_delta.append(float(u_opt.delta))
        inputs_torque.append(float(jnp.mean(u_opt.tau_w)))
        dx = pack_error_state(state_difference(goal_turn, x))
        tracking_err.append(float(jnp.linalg.norm(dx[:3])))

        if step % 100 == 0:
            print(f"  Step {n_straight + step}: "
                  f"pos=({x.p_W[0]:.3f}, {x.p_W[1]:.3f}), "
                  f"yaw={traj_yaw[-1]:.3f} rad, "
                  f"delta={u_opt.delta:.4f} rad")

    # ── Plot ─────────────────────────────────────────────────────────────
    t = jnp.arange(len(traj_x)) * dt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MPC Path Tracking — Ackermann Car", fontsize=14)

    # Top-left: XY trajectory
    ax = axes[0, 0]
    ax.plot(traj_x, traj_y, "b-", linewidth=1.5, label="MPC trajectory")
    ax.plot(traj_x[0], traj_y[0], "go", markersize=8, label="Start")
    ax.plot(traj_x[-1], traj_y[-1], "rs", markersize=8, label="End")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("XY Trajectory")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Top-right: heading
    ax = axes[0, 1]
    ax.plot(t, jnp.rad2deg(jnp.array(traj_yaw)), "b-")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(90, color="gray", linestyle="--", alpha=0.5, label="Goal (turn)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Heading [deg]")
    ax.set_title("Heading over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-left: control inputs
    ax = axes[1, 0]
    ax.plot(t, inputs_delta, "r-", label="Steering δ [rad]")
    ax.plot(t, inputs_torque, "b-", alpha=0.7, label="Mean torque [N·m]")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Input")
    ax.set_title("Control Inputs")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom-right: tracking error
    ax = axes[1, 1]
    ax.plot(t, tracking_err, "k-")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position error [m]")
    ax.set_title("Tracking Error (‖Δp‖)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("mpc_tracking_result.png", dpi=150)
    print("\nPlot saved to mpc_tracking_result.png")
    plt.show()


if __name__ == "__main__":
    main()
