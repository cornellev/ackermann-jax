"""
mpc_ekf_ros_node.py — Combined EKF + MPC ROS 2 node.

Reads sensor data from shared memory, runs an EKF state estimator and an
LTV-MPC controller in the same timer callback, then publishes:
  - kalman/odom            (nav_msgs/Odometry)   — EKF position + twist
  - kalman/wheel_speeds    (std_msgs/Float32MultiArray) — EKF wheel speeds
  - mpc/control            (std_msgs/Float32MultiArray) — [delta, 0, 0, tau_RL, tau_RR, cost, solved]

EKF state is also written to kalman_shm so that other tools reading the
shared-memory bus continue to work unchanged.

Reference trajectory: on start-up the MPC is seeded with N+1 copies of
the initial state (stabilize-at-origin reference).  Update mpc_state.x_ref
and mpc_state.u_ref from an external planner to track a path.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import jaxlie
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from ackermann_jax.car import (
    AckermannCarInput,
    AckermannCarState,
    AckermannCarModel,
    default_params,
)
from ackermann_jax.ekf import EKFState, ekf_predict, ekf_update, ERROR_DIM
from ackermann_jax.mpc import (
    MPCState,
    default_mpc_params,
    init_mpc_state,
    mpc_step,
)
from ackermann_jax.publish_kalman_ros import KalmanRosPublisher
from ackermann_jax.read_sensor_shm import SensorShmReader
from ackermann_jax.write_kalman_shm import KalmanShmWriter

# ── sensor index constants (mirror ekf_sensors.py) ──────────────────────────
_IDX_GLOBAL_TS = 0
_IDX_TURN_ANGLE = 6
_IDX_RPM_FL = 8
_IDX_RPM_FR = 9
_IDX_RPM_RL = 11
_IDX_RPM_RR = 12
_IDX_GPS_LAT = 14
_IDX_GPS_LON = 15
_IDX_THROTTLE = 18

_RPM_TO_RADS = 2.0 * math.pi / 60.0
_R_EARTH = 6_371_000.0

# ── noise / tuning ───────────────────────────────────────────────────────────
_R_GPS_VAL = 1e-4
_R_WHEELS_VAL = 1e-4
_Q_SCALE = 1e-6
_P0_SCALE = 1e-4
_TAU_MAX = 0.35
_DT_DEFAULT = 0.05   # 20 Hz timer period [s]
_DT_MAX = 1.0

TOPIC_MPC_CONTROL = "mpc/control"


def _h_gps_2d(x: AckermannCarState):
    return x.p_W[:2]


def _h_wheels(x: AckermannCarState):
    return x.omega_W


def _latlon_to_local_xy(lat, lon, lat0, lon0):
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    x = dlon * math.cos(math.radians(lat0)) * _R_EARTH
    y = dlat * _R_EARTH
    return x, y


def _zero_input() -> AckermannCarInput:
    return AckermannCarInput(
        delta=jnp.zeros((), dtype=jnp.float32),
        tau_w=jnp.zeros(4, dtype=jnp.float32),
    )


class MpcEkfNode(Node):
    def __init__(self) -> None:
        super().__init__("mpc_ekf_node")

        car_params = default_params()
        self._model = AckermannCarModel(car_params)
        self._wheel_radius = float(car_params.geom.wheel_radius)

        self._reader = SensorShmReader()
        self._writer = KalmanShmWriter()
        self._ros_pub = KalmanRosPublisher(self)
        self._pub_ctrl = self.create_publisher(Float32MultiArray, TOPIC_MPC_CONTROL, 10)

        self._mpc_params = default_mpc_params()

        # EKF noise matrices
        self._Q = _Q_SCALE * jnp.eye(ERROR_DIM)
        self._P0 = _P0_SCALE * jnp.eye(ERROR_DIM)
        self._R_gps = _R_GPS_VAL * jnp.eye(2)
        self._R_wheels = _R_WHEELS_VAL * jnp.eye(4)
        self._motor_mask = jnp.array([0.0, 0.0, 1.0, 1.0], dtype=jnp.float32)

        # Runtime state
        self._ekf: EKFState | None = None
        self._mpc_state: MPCState | None = None
        self._lat0: float | None = None
        self._lon0: float | None = None
        self._prev_ts_s: float | None = None

        self._timer = self.create_timer(_DT_DEFAULT, self._step)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _init_mpc(self, x0: AckermannCarState) -> None:
        N = self._mpc_params.N
        x_ref = [x0] * (N + 1)
        u_ref = [_zero_input()] * N
        self._mpc_state = init_mpc_state(x_ref, u_ref, self._mpc_params)

    # ── main callback ─────────────────────────────────────────────────────────

    def _step(self) -> None:
        snap = self._reader.read_snapshot()
        if snap is None:
            return
        _, d = snap

        ts_us = int(d[_IDX_GLOBAL_TS])
        ts_s = ts_us * 1e-6
        lat = float(d[_IDX_GPS_LAT])
        lon = float(d[_IDX_GPS_LON])
        delta_sensor = float(d[_IDX_TURN_ANGLE])
        throttle = float(d[_IDX_THROTTLE])
        omega_meas = jnp.array(
            [
                float(d[_IDX_RPM_FL]) * _RPM_TO_RADS,
                float(d[_IDX_RPM_FR]) * _RPM_TO_RADS,
                float(d[_IDX_RPM_RL]) * _RPM_TO_RADS,
                float(d[_IDX_RPM_RR]) * _RPM_TO_RADS,
            ],
            dtype=jnp.float32,
        )

        # Set GPS origin on first valid fix
        if self._lat0 is None and lat != 0.0 and lon != 0.0:
            self._lat0, self._lon0 = lat, lon
            self.get_logger().info(
                f"GPS origin set: ({self._lat0:.7f}, {self._lon0:.7f})"
            )

        # Wait for GPS origin before initialising EKF
        if self._ekf is None:
            if self._lat0 is None:
                return
            gx, gy = _latlon_to_local_xy(lat, lon, self._lat0, self._lon0)
            p0 = jnp.array([gx, gy, self._wheel_radius], dtype=jnp.float32)
            x0 = AckermannCarState(
                p_W=p0,
                R_WB=jaxlie.SO3.identity(),
                v_W=jnp.zeros(3, dtype=jnp.float32),
                w_B=jnp.zeros(3, dtype=jnp.float32),
                omega_W=omega_meas,
            )
            self._ekf = EKFState(x_nom=x0, P=self._P0)
            self._prev_ts_s = ts_s
            self._init_mpc(x0)
            self.get_logger().info("EKF + MPC initialised.")
            return

        # ── EKF predict ───────────────────────────────────────────────────────
        dt = ts_s - self._prev_ts_s if self._prev_ts_s is not None else _DT_DEFAULT
        if dt <= 0.0 or dt > _DT_MAX:
            dt = _DT_DEFAULT
        self._prev_ts_s = ts_s

        tau_w = self._motor_mask * (throttle * _TAU_MAX)
        u_sensor = AckermannCarInput(
            delta=jnp.array(delta_sensor, dtype=jnp.float32),
            tau_w=tau_w.astype(jnp.float32),
        )
        self._ekf = ekf_predict(self._model, self._ekf, u_sensor, self._Q, dt)

        # ── EKF update ────────────────────────────────────────────────────────
        if lat != 0.0 and lon != 0.0 and self._lat0 is not None:
            gx, gy = _latlon_to_local_xy(lat, lon, self._lat0, self._lon0)
            z_gps = jnp.array([gx, gy], dtype=jnp.float32)
            self._ekf = ekf_update(self._ekf, z_gps, _h_gps_2d, self._R_gps)
        self._ekf = ekf_update(self._ekf, omega_meas, _h_wheels, self._R_wheels)

        # ── Publish EKF state ─────────────────────────────────────────────────
        self._writer.write_state(self._ekf, ts_us)
        self._ros_pub.publish_state(self._ekf, ts_us)

        # ── MPC step ──────────────────────────────────────────────────────────
        if self._mpc_state is not None:
            result, self._mpc_state = mpc_step(
                self._model, self._mpc_state, self._mpc_params, self._ekf.x_nom
            )
            # u_opt[0]: [delta, 0, 0, tau_RL, tau_RR] (full 5-vector, first horizon step)
            u0 = result.u_opt[0]
            ctrl_msg = Float32MultiArray()
            ctrl_msg.data = [
                float(u0[0]),   # steering angle [rad]
                float(u0[1]),   # tau_FL (always 0 — front wheels undriven)
                float(u0[2]),   # tau_FR (always 0)
                float(u0[3]),   # tau_RL [N·m]
                float(u0[4]),   # tau_RR [N·m]
                float(result.cost),
                float(result.solved),
            ]
            self._pub_ctrl.publish(ctrl_msg)

    # ── cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        self._reader.close()
        self._writer.close()
        self._ros_pub.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MpcEkfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
