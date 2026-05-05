"""
EKF ROS 2 node for the Ackermann car model.

Subscriptions (measurements):
    /gps/fix          sensor_msgs/NavSatFix       -- GPS position
    /imu/data         sensor_msgs/Imu             -- gyroscope + accelerometer
    /wheel_speeds     std_msgs/Float32MultiArray   -- rear wheel angular velocities [RL,RR] rad/s;
                                                       also accepts [FL,FR,RL,RR]
    /ackermann_cmd    ackermann_msgs/AckermannDriveStamped -- steering + throttle command

Publications (state estimates):
    /ekf/odom         nav_msgs/Odometry           -- pose + twist with covariance
    /ekf/ackermann    ackermann_msgs/AckermannDriveStamped -- forward speed + steering estimate
    /ekf/wheel_speeds std_msgs/Float32MultiArray   -- estimated wheel angular velocities [FL,FR,RL,RR]

Usage:
    ros2 run ackermann_jax ekf_ros_node
"""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Vector3
from sensor_msgs.msg import Imu, NavSatFix
from std_msgs.msg import Float32MultiArray

from ackermann_jax.car import (
    AckermannCarInput,
    AckermannCarModel,
    AckermannCarState,
    default_params,
)
from ackermann_jax.ekf import ERROR_DIM, EKFState, ekf_predict, ekf_update

jax.config.update("jax_enable_x64", False)

# ── Sensor noise variances (can be overridden via ROS params) ──────────────
_R_GPS_DEFAULT = 1e-4  # [m²]         σ ≈ 0.01 m
_R_GYRO_DEFAULT = 1e-4  # [(rad/s)²]   σ ≈ 0.01 rad/s
_R_GRAVITY_DEFAULT = 1e-2  # [(m/s²)²]    σ ≈ 0.10 m/s²
_R_WHEELS_DEFAULT = 1e-4  # [(rad/s)²]   σ ≈ 0.01 rad/s
_Q_SCALE_DEFAULT = 1e-6  # process noise scale
_P0_SCALE_DEFAULT = 1e-4  # initial covariance scale

_R_EARTH = 6_371_000.0  # [m]
_TAU_MAX = 0.35  # peak wheel torque [N·m]
_DT_DEFAULT = 0.01  # nominal step [s]
_DT_MAX = 1.0  # cap: larger gaps are treated as a re-init guard

FRAME_ID = "world"
CHILD_FRAME_ID = "base_link"

# Error-state index slices (must match errorDyn.py pack/unpack ordering)
_P_IDX = {
    "p_W": slice(0, 3),
    "theta": slice(3, 6),
    "v_W": slice(6, 9),
    "w_B": slice(9, 12),
    "omega_W": slice(12, 16),
}


# ── Measurement functions ──────────────────────────────────────────────────


def h_gps_2d(x: AckermannCarState) -> jnp.ndarray:
    """GPS: measures world-frame XY position."""
    return x.p_W[:2]


def h_gyro(x: AckermannCarState) -> jnp.ndarray:
    """Gyroscope: measures body angular velocity."""
    return x.w_B


def make_h_gravity(g: float):
    """Accelerometer gravity direction in body frame: R_BW @ [0, 0, g]."""
    g_up_W = jnp.array([0.0, 0.0, g], dtype=jnp.float32)

    def h_gravity(x: AckermannCarState) -> jnp.ndarray:
        R_BW = x.R_WB.as_matrix().T
        return R_BW @ g_up_W

    return h_gravity


def h_wheels(x: AckermannCarState) -> jnp.ndarray:
    """Wheel encoders: measures rear wheel angular velocities [RL, RR]."""
    return x.omega_W[2:]


# ── Coordinate helpers ─────────────────────────────────────────────────────


def _latlon_to_local_xy(lat: float, lon: float, lat0: float, lon0: float):
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    x = dlon * math.cos(math.radians(lat0)) * _R_EARTH
    y = dlat * _R_EARTH
    return x, y


def _ros_stamp_to_ns(stamp: Time) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


# ── Covariance extraction ─────────────────────────────────────────────────


def _build_pose_covariance(P: np.ndarray) -> list[float]:
    """
    Extract a 6×6 pose covariance (x,y,z,roll,pitch,yaw) from the EKF
    error-state covariance P (16×16), flattened row-major for ROS.
    """
    cov6 = np.zeros((6, 6), dtype=np.float64)
    # Position block (p_W → rows/cols 0:3)
    cov6[0:3, 0:3] = P[0:3, 0:3]
    # Orientation block (theta → rows/cols 3:6)
    cov6[3:6, 3:6] = P[3:6, 3:6]
    # Cross terms
    cov6[0:3, 3:6] = P[0:3, 3:6]
    cov6[3:6, 0:3] = P[3:6, 0:3]
    return cov6.flatten().tolist()


def _build_twist_covariance(P: np.ndarray) -> list[float]:
    """
    Extract a 6×6 twist covariance (vx,vy,vz,wx,wy,wz) from P (16×16),
    flattened row-major for ROS.
    """
    cov6 = np.zeros((6, 6), dtype=np.float64)
    # Linear velocity block (v_W → rows/cols 0:3)
    cov6[0:3, 0:3] = P[6:9, 6:9]
    # Angular velocity block (w_B → rows/cols 3:6)
    cov6[3:6, 3:6] = P[9:12, 9:12]
    # Cross terms
    cov6[0:3, 3:6] = P[6:9, 9:12]
    cov6[3:6, 0:3] = P[9:12, 6:9]
    return cov6.flatten().tolist()


# ── ROS 2 node ─────────────────────────────────────────────────────────────


class EKFNode(Node):
    def __init__(self):
        super().__init__("ackermann_ekf")

        # ── Parameters ──
        self.declare_parameter("dt", _DT_DEFAULT)
        self.declare_parameter("r_gps", _R_GPS_DEFAULT)
        self.declare_parameter("r_gyro", _R_GYRO_DEFAULT)
        self.declare_parameter("r_gravity", _R_GRAVITY_DEFAULT)
        self.declare_parameter("r_wheels", _R_WHEELS_DEFAULT)
        self.declare_parameter("q_scale", _Q_SCALE_DEFAULT)
        self.declare_parameter("p0_scale", _P0_SCALE_DEFAULT)
        self.declare_parameter("tau_max", _TAU_MAX)

        dt = self.get_parameter("dt").value
        r_gps = self.get_parameter("r_gps").value
        r_gyro = self.get_parameter("r_gyro").value
        r_gravity = self.get_parameter("r_gravity").value
        r_wheels = self.get_parameter("r_wheels").value
        q_scale = self.get_parameter("q_scale").value
        p0_scale = self.get_parameter("p0_scale").value
        self._tau_max = self.get_parameter("tau_max").value
        self._dt = dt

        # ── Model + noise matrices ──
        params = default_params()
        self._model = AckermannCarModel(params)
        self._g = float(params.chassis.g)

        self._Q = q_scale * jnp.eye(ERROR_DIM)
        self._P0 = p0_scale * jnp.eye(ERROR_DIM)
        self._R_gps = r_gps * jnp.eye(2)
        self._R_gyro = r_gyro * jnp.eye(3)
        self._R_gravity = r_gravity * jnp.eye(3)
        self._R_wheels = r_wheels * jnp.eye(2)
        self._h_gravity = make_h_gravity(self._g)
        self._motor_mask = jnp.array([0.0, 0.0, 1.0, 1.0], dtype=jnp.float32)

        # ── EKF state ──
        self._ekf: Optional[EKFState] = None
        self._lat0: Optional[float] = None
        self._lon0: Optional[float] = None
        self._prev_stamp_ns: Optional[int] = None

        # Cached latest measurements (None = not yet received)
        self._z_gps: Optional[jnp.ndarray] = None
        self._z_gyro: Optional[jnp.ndarray] = None
        self._z_gravity: Optional[jnp.ndarray] = None
        self._z_wheels: Optional[jnp.ndarray] = None

        # Cached latest control input
        self._delta: float = 0.0
        self._throttle: float = 0.0  # [0, 1]

        # ── Subscribers ──
        self.create_subscription(NavSatFix, "/gps/fix", self._cb_gps, 10)
        self.create_subscription(Imu, "/imu/data", self._cb_imu, 10)
        self.create_subscription(
            Float32MultiArray, "/wheel_speeds", self._cb_wheels, 10
        )
        self.create_subscription(
            AckermannDriveStamped, "/ackermann_cmd", self._cb_cmd, 10
        )

        # ── Publishers ──
        self._pub_odom = self.create_publisher(Odometry, "/ekf/odom", 10)
        self._pub_ackermann = self.create_publisher(
            AckermannDriveStamped, "/ekf/ackermann", 10
        )
        self._pub_wheels = self.create_publisher(
            Float32MultiArray, "/ekf/wheel_speeds", 10
        )

        # ── Predict timer ──
        self.create_timer(dt, self._timer_cb)

        self.get_logger().info("ackermann_ekf node started")

    # ── Subscriber callbacks ───────────────────────────────────────────────

    def _cb_gps(self, msg: NavSatFix) -> None:
        lat = msg.latitude
        lon = msg.longitude
        if lat == 0.0 and lon == 0.0:
            return

        if self._lat0 is None:
            self._lat0, self._lon0 = lat, lon
            self.get_logger().info(
                f"GPS origin set: ({self._lat0:.7f}, {self._lon0:.7f})"
            )

        gx, gy = _latlon_to_local_xy(lat, lon, self._lat0, self._lon0)
        self._z_gps = jnp.array([gx, gy], dtype=jnp.float32)

        if self._ekf is None:
            self._try_init(gx, gy)

    def _cb_imu(self, msg: Imu) -> None:
        av = msg.angular_velocity
        self._z_gyro = jnp.array([av.x, av.y, av.z], dtype=jnp.float32)

        la = msg.linear_acceleration
        self._z_gravity = jnp.array([la.x, la.y, la.z], dtype=jnp.float32)

    def _cb_wheels(self, msg: Float32MultiArray) -> None:
        if len(msg.data) >= 4:
            rear_wheels = msg.data[2:4]
        elif len(msg.data) >= 2:
            rear_wheels = msg.data[:2]
        else:
            self.get_logger().warn(
                f"Expected 2 rear wheel speeds or 4 full wheel speeds, got {len(msg.data)}"
            )
            return
        self._z_wheels = jnp.array(rear_wheels, dtype=jnp.float32)

        if self._ekf is None and self._lat0 is not None:
            self._try_init(0.0, 0.0)

    def _cb_cmd(self, msg: AckermannDriveStamped) -> None:
        self._delta = float(msg.drive.steering_angle)
        # Map speed command to throttle in [0, 1].
        # AckermannDriveStamped.drive.speed is in m/s; store raw throttle
        # approximation as speed / some reference speed.  Users can replace
        # this with a proper torque subscriber if available.
        self._throttle = float(msg.drive.speed)

    # ── Initialisation ─────────────────────────────────────────────────────

    def _try_init(self, gx: float, gy: float) -> None:
        """Initialise the EKF from the first valid GPS fix."""
        z = float(self._model.params.geom.wheel_radius)
        p0 = jnp.array([gx, gy, z], dtype=jnp.float32)
        omega0 = jnp.zeros(4, dtype=jnp.float32)
        if self._z_wheels is not None:
            omega0 = omega0.at[2:4].set(self._z_wheels)
        x0 = AckermannCarState(
            p_W=p0,
            R_WB=jaxlie.SO3.identity(),
            v_W=jnp.zeros(3, dtype=jnp.float32),
            w_B=jnp.zeros(3, dtype=jnp.float32),
            omega_W=omega0,
        )
        self._ekf = EKFState(x_nom=x0, P=self._P0)
        self._prev_stamp_ns = self.get_clock().now().nanoseconds
        self.get_logger().info("EKF initialised")

    # ── Predict + update timer ─────────────────────────────────────────────

    def _timer_cb(self) -> None:
        if self._ekf is None:
            return

        now_ns = self.get_clock().now().nanoseconds
        if self._prev_stamp_ns is not None:
            dt = (now_ns - self._prev_stamp_ns) * 1e-9
            if dt <= 0.0 or dt > _DT_MAX:
                dt = self._dt
        else:
            dt = self._dt
        self._prev_stamp_ns = now_ns

        # ── Predict ──
        tau_w = self._motor_mask * (self._throttle * self._tau_max)
        u = AckermannCarInput(
            delta=jnp.array(self._delta, dtype=jnp.float32),
            tau_w=tau_w.astype(jnp.float32),
        )
        self._ekf = ekf_predict(self._model, self._ekf, u, self._Q, dt)

        # ── Update: GPS ──
        if self._z_gps is not None:
            self._ekf = ekf_update(self._ekf, self._z_gps, h_gps_2d, self._R_gps)

        # ── Update: gyroscope ──
        if self._z_gyro is not None:
            self._ekf = ekf_update(self._ekf, self._z_gyro, h_gyro, self._R_gyro)

        # ── Update: gravity/accelerometer ──
        if self._z_gravity is not None:
            self._ekf = ekf_update(
                self._ekf, self._z_gravity, self._h_gravity, self._R_gravity
            )

        # ── Update: wheel encoders ──
        if self._z_wheels is not None:
            self._ekf = ekf_update(self._ekf, self._z_wheels, h_wheels, self._R_wheels)

        self._publish(now_ns)

    # ── Publishing ─────────────────────────────────────────────────────────

    def _publish(self, stamp_ns: int) -> None:
        ekf = self._ekf
        x = ekf.x_nom

        stamp = Time()
        stamp.sec = stamp_ns // 1_000_000_000
        stamp.nanosec = stamp_ns % 1_000_000_000

        P = np.array(ekf.P, dtype=np.float64)

        # ── Odometry ──
        p = x.p_W
        wxyz = x.R_WB.wxyz  # jaxlie: (w, x, y, z)
        v = x.v_W
        w = x.w_B

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = FRAME_ID
        odom.child_frame_id = CHILD_FRAME_ID

        odom.pose.pose.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
        # ROS quaternion convention: (x, y, z, w)
        odom.pose.pose.orientation = Quaternion(
            x=float(wxyz[1]),
            y=float(wxyz[2]),
            z=float(wxyz[3]),
            w=float(wxyz[0]),
        )
        odom.pose.covariance = _build_pose_covariance(P)

        odom.twist.twist.linear = Vector3(x=float(v[0]), y=float(v[1]), z=float(v[2]))
        odom.twist.twist.angular = Vector3(x=float(w[0]), y=float(w[1]), z=float(w[2]))
        odom.twist.covariance = _build_twist_covariance(P)

        self._pub_odom.publish(odom)

        # ── AckermannDriveStamped ──
        # Publish the estimated forward body speed and the last commanded
        # steering angle (the EKF does not directly estimate delta).
        R = x.R_WB.as_matrix()
        v_B = R.T @ np.array(v)
        speed_est = float(v_B[0])

        ack_msg = AckermannDriveStamped()
        ack_msg.header.stamp = stamp
        ack_msg.header.frame_id = CHILD_FRAME_ID
        ack_msg.drive.speed = speed_est
        ack_msg.drive.steering_angle = self._delta

        self._pub_ackermann.publish(ack_msg)

        # ── Wheel speeds ──
        wheel_msg = Float32MultiArray()
        wheel_msg.data = [float(x.omega_W[i]) for i in range(4)]
        self._pub_wheels.publish(wheel_msg)


# ── Entry point ───────────────────────────────────────────────────────────


def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
