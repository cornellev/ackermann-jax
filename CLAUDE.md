# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`ackermann-jax` is a JAX-based 3D rigid-body Ackermann vehicle model with a contact/tire physics engine, an Error-State Extended Kalman Filter (ES-EKF), and a Linear Time-Varying MPC controller. It targets small RC-scale cars (~1.5 kg) and ships ROS 2 Humble nodes for real-time deployment.

## Environment & Commands

This project uses [pixi](https://prefix.dev/) for environment management. Three environments are defined: `default` (CPU JAX), `ros` (adds ROS 2 Humble), and `gpu` (CUDA 12.6).

```bash
# Install dependencies and activate default env
pixi install

# Run tests (no ROS required)
pixi run pytest test/

# Run a single test file
pixi run pytest test/test_ekf.py

# Lint / format
pixi run ruff check src/
pixi run black src/

# Launch EKF-only node (requires ros environment)
pixi run ros2 launch launch/ekf.launch.py

# Launch combined MPC + EKF node
pixi run ros2 launch launch/mpc_ekf.launch.py
```

The package is installed in editable mode (`pip install -e .`) via pixi, so changes to `src/` are immediately reflected without reinstalling.

## Architecture

### Core physics (`src/ackermann_jax/`)

| File | Role |
|------|------|
| `parameters.py` | All parameter dataclasses (`AckermannGeometry`, `TireParams`, `ContactParams`, `ChassisParams`, `MotorConfig`, `AckermannCarParams`). All are JAX pytree nodes. |
| `car.py` | `AckermannCarState`, `AckermannCarInput`, `AckermannCarModel`. `xdot` computes continuous dynamics; `step` integrates (Euler or semi-implicit Euler). Factory helpers: `default_params()`, `default_state()`. |
| `errorDyn.py` | Error-state representation for EKF/MPC linearization. `inject_error` / `state_difference` handle the SO(3) perturbation convention. `pack_error_state` / `unpack_error_state` flatten to/from a 16-dim vector. |
| `ekf.py` | ES-EKF: `EKFState(x_nom, P)`. `ekf_predict` (JIT-compiled), `ekf_update` (JIT, static `h`), `ekf_step`. Jacobians `compute_F` and `compute_H` use `jax.jacobian` through the pytree dynamics. |
| `mpc.py` | LTV-MPC: `MPCParams`, `MPCState`, `MPCResult`. `mpc_step_batched` is the JIT-compiled receding-horizon solve (OSQP via `jaxopt`). Decision variables are `[delta, tau_RL, tau_RR]` (3-dim subset of the 5-dim full input). |

### State vector (16-dim error state)

```
[0:3]   dp_W      position error (world frame)
[3:6]   dθ_B      rotation error (so3 tangent vector)
[6:9]   dv_W      velocity error (world frame)
[9:12]  dw_B      angular velocity error (body frame)
[12:16] dω_W      wheel speed errors (FL, FR, RL, RR)
```

### ROS 2 nodes

| File | Entry point | Description |
|------|------------|-------------|
| `ekf_ros_node.py` | `ekf_ros_node` | EKF-only node. Subscribes to `/gps/fix`, `/imu/data`, `/wheel_speeds`, `/ackermann_cmd`. Publishes to `/ekf/odom`, `/ekf/ackermann`, `/ekf/wheel_speeds`. |
| `mpc_ekf_ros_node.py` | `mpc_ekf_ros_node` | Combined EKF + MPC node. Reads sensors from shared memory, publishes `/kalman/odom`, `/kalman/wheel_speeds`, `/mpc/control`. |
| `ekf_sensors.py` | `ekf_sensors` | Standalone sensor bridge that reads shared memory and runs the EKF loop without MPC. |

Shared-memory helpers: `read_sensor_shm.py`, `write_kalman_shm.py`, `read_kalman_shm.py`, `publish_kalman_ros.py`.

### Wheel ordering convention

`WHEEL_ORDER = ("FL", "FR", "RL", "RR")` — indices 0–3 throughout. The default car is RWD (`has_motor = [0, 0, 1, 1]`). Wheel speeds on the `/wheel_speeds` topic can be 2-element `[RL, RR]` or 4-element `[FL, FR, RL, RR]`.

### Key design decisions

- **All parameter dataclasses are frozen pytrees** — required so JAX can trace through them in JIT/`vmap`/`jacobian` calls without recompilation on value changes.
- **Error-state EKF, not full-state** — rotation is on SO(3); the 16-dim error state uses a tangent-space (so3) perturbation for the orientation block. `compute_H` differentiates through `inject_error` to correctly account for this.
- **`mpc.py` enables `jax_enable_x64`** at import time — the prediction matrix build and QP solve require float64 for numerical stability (contact spring eigenvalues ~12 at dt=50 ms). `ekf_ros_node.py` explicitly disables x64.
- **MPC stiff-mode DOFs** — `dp_z` (index 2), `dv_z` (index 8), and `dω_W` (indices 12–15) have zero weight in `Q` because their `Phi/Theta` rows diverge with horizon length at 50 ms dt. These DOFs are handled by the inner PI torque loop.
