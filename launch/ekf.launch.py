from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration

def generate_launch_description() -> LaunchDescription:
    log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="INFO",
        description="rclpy log level: DEBUG | INFO | WARN | ERROR",
    )

    ekf_node = ExecuteProcess(
        cmd=[
            "python3", "-m", "ackermann_jax.ekf_ros_node",
            "--ros-args", "--log-level", LaunchConfiguration("log_level"),
        ],
        output="screen",
        emulate_tty=True,
    )

    return LaunchDescription(
        [
            log_level_arg,
            LogInfo(msg="Launching EKF node..."),
            ekf_node,
        ]
    )
