from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Launch file for test_follower node.
    Allows you to set parameters such as takeoff_alt and rate_hz.
    """
    return LaunchDescription([
        Node(
            package='scout',                 # your ROS2 package name
            executable='test_follower',     # must match your Python entrypoint name
            name='test_follower',
            output='screen',
            parameters=[{
                'ns': 'mavros',
                'rate_hz': 20.0,             # control loop rate
                'takeoff_alt': 6.0,          # <-- easily change this for takeoff altitude
            }],
        ),
    ])
