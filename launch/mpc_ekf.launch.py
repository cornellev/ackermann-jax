"""
ROS 2 launch file for the combined MPC + EKF node.

Usage (from repo root, inside the pixi environment):
    pixi run ros2 launch launch/mpc_ekf.launch.py

Or if roslaunch is already on PATH:
    ros2 launch launch/mpc_ekf.launch.py

Optional arguments (override at launch time):
    log_level   — rclpy log level: DEBUG | INFO | WARN | ERROR  (default: INFO)

Published topics:
    /kalman/odom            nav_msgs/Odometry
    /kalman/wheel_speeds    std_msgs/Float32MultiArray
    /mpc/control            std_msgs/Float32MultiArray
        layout: [delta_rad, tau_FL, tau_FR, tau_RL, tau_RR, qp_cost, qp_solved]
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="INFO",
        description="rclpy log level: DEBUG | INFO | WARN | ERROR",
    )

    mpc_ekf_node = ExecuteProcess(
        cmd=[
            "python3", "-m", "ackermann_jax.mpc_ekf_ros_node",
            "--ros-args", "--log-level", LaunchConfiguration("log_level"),
        ],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            log_level_arg,
            LogInfo(msg="Launching MPC + EKF node …"),
            mpc_ekf_node,
        ]
    )
