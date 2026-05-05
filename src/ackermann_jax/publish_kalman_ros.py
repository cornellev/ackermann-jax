import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Point, Quaternion, Vector3
from builtin_interfaces.msg import Time

from ackermann_jax.ekf import EKFState

TOPIC_ODOM = "kalman/odom"
TOPIC_WHEEL_SPEEDS = "kalman/wheel_speeds"
FRAME_ID = "world"
CHILD_FRAME_ID = "base_link"


class KalmanRosPublisher:
    def __init__(
        self,
        node: Node,
        topic_odom: str = TOPIC_ODOM,
        topic_wheel_speeds: str = TOPIC_WHEEL_SPEEDS,
    ):
        self._node = node
        self._pub_odom = node.create_publisher(Odometry, topic_odom, 10)
        self._pub_wheels = node.create_publisher(
            Float32MultiArray, topic_wheel_speeds, 10
        )

    def publish_state(self, ekf: EKFState, timestamp_us: int) -> None:
        """Publish EKF state to ROS topics.

        Args:
            ekf: current EKF state
            timestamp_us: microsecond timestamp from sensor_shm / kalman_shm.
        """
        x = ekf.x_nom
        p = x.p_W
        wxyz = x.R_WB.wxyz  # jaxlie convention: (w, x, y, z)
        v = x.v_W
        w = x.w_B
        om = x.omega_W

        stamp = Time()
        timestamp_ns = int(timestamp_us) * 1_000
        stamp.sec = timestamp_ns // 1_000_000_000
        stamp.nanosec = timestamp_ns % 1_000_000_000

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

        odom.twist.twist.linear = Vector3(x=float(v[0]), y=float(v[1]), z=float(v[2]))
        odom.twist.twist.angular = Vector3(x=float(w[0]), y=float(w[1]), z=float(w[2]))

        self._pub_odom.publish(odom)

        wheel_msg = Float32MultiArray()
        wheel_msg.data = [float(om[i]) for i in range(4)]
        self._pub_wheels.publish(wheel_msg)

    def close(self) -> None:
        self._pub_odom.destroy()
        self._pub_wheels.destroy()
