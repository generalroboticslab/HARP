from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mod_explore',
            executable='explore_zigzag',
            output='screen',
            parameters=[
                {
                    'ns': '/explore/mavros',
                    'rate_hz': 20.0,
                    'takeoff_alt': 4.5,
                    'xy_speed': 1.5,
                    'warmup_sec': 2.0,
                    'alt_tol': 0.3,
                    'lock_yaw': True,
                    'yaw_offset_deg': 0.0,
                    'reverse_x': False,
                    'reverse_y': False,
                    'zigzag.length_m': 20.0,       # forward coverage distance along initial body x
                    'zigzag.width_m': 10.0,        # total lateral sweep centered on body x-axis
                    'zigzag.resolution_m': 2.0,    # forward spacing between lateral sweeps
                },
            ],
        ),
    ])
