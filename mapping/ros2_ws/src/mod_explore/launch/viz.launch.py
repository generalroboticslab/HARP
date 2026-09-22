from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """
    two nodes here:
    - leader (w/ sine wave)
    - follower (w/ velocity cmd)
    """
    rviz_config_path = os.path.join(
        get_package_share_directory('mod_explore'),
        'launch',      
        'track.rviz'
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            'reset_world',
            default_value='false',
            description='Reset Gazebo world on launch (also resets spawned vehicles).'
        ),
        DeclareLaunchArgument(
            'reset_delay',
            default_value='1.5',
            description='Seconds to wait before calling world reset.'
        ),
        DeclareLaunchArgument(
            'world_name',
            default_value='default',
            description='Gazebo world name for reset service.'
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config_path]
        ),
        Node(
            package='mod_explore',
            executable='explore_viz',
            name='explore_viz',
            output='screen',
            parameters=[{
                # Gazebo dual-UAV: often True + sim_offset [0,0,0]; field experiment: False.
                'is_gazebo': True,
                'sim_offset': [-4.0, 0.0, 0.0],
                'enable_acoustic': False,
                'enable_d2_csv_log': False,
                'd2_csv_path': '',
                'reset_world_on_startup': LaunchConfiguration('reset_world'),
                'reset_world_name': LaunchConfiguration('world_name'),
                'reset_delay_sec': LaunchConfiguration('reset_delay'),
            }],
        ),
    ])
