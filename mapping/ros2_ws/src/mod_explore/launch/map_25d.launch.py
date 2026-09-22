"""Build and export a 2.5D map from an already-running FAST-LIO instance."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory('mod_explore'), 'launch', 'map_25d.rviz')
    arguments = [
        ('cloud_topic', '/cloud_registered', 'Registered PointCloud2 input'),
        ('odom_topic', '/Odometry', 'Odometry corresponding to the registered cloud'),
        ('use_mavros_reference', 'false', 'Align the map to the first MAVROS pose'),
        ('mavros_pose_topic', '/follower/mavros/local_position/pose',
         'PoseStamped reference; used only when use_mavros_reference is true'),
        ('map_resolution', '0.2', '2.5D cell size in meters'),
        ('update_range_xy', '6.0', 'Map update radius around odometry in meters'),
        ('vegetation_range_max', '0.5', 'Vertical range for maximum roughness score'),
        ('export_map', 'true', 'Enable PCD and standalone HTML export'),
        ('map_output_dir', os.path.join(os.getcwd(), 'maps'), 'Export parent directory'),
        ('map_export_idle_seconds', '3.0', 'Export after idle seconds; 0 disables idle export'),
        ('use_sim_time', 'false', 'Use /clock, for example during bag playback'),
        ('rviz', 'true', 'Show the 2.5D map in RViz'),
        ('rviz_cfg', rviz_config, 'RViz configuration file'),
    ]

    def parameter(name, value_type):
        return ParameterValue(LaunchConfiguration(name), value_type=value_type)

    return LaunchDescription([
        DeclareLaunchArgument(name, default_value=default, description=description)
        for name, default, description in arguments
    ] + [
        Node(
            package='mod_explore',
            executable='map_recorder',
            parameters=[{
                'use_sim_time': parameter('use_sim_time', bool),
                'cloud_topic': '/explore_map_color',
                'cell_size': parameter('map_resolution', float),
                'output_dir': parameter('map_output_dir', str),
                'idle_seconds': parameter('map_export_idle_seconds', float),
            }],
            condition=IfCondition(LaunchConfiguration('export_map')),
            sigterm_timeout='120',
            output='screen',
        ),
        Node(
            package='mod_explore_cpp',
            executable='lio_post',
            parameters=[{
                'use_sim_time': parameter('use_sim_time', bool),
                'map_source_topic': parameter('cloud_topic', str),
                'use_mavros_reference': parameter('use_mavros_reference', bool),
                'mavros_pose_topic': parameter('mavros_pose_topic', str),
                'map_resolution': parameter('map_resolution', float),
                'update_range_xy': parameter('update_range_xy', float),
                'vegetation_range_max': parameter('vegetation_range_max', float),
            }],
            remappings=[('/Odometry', LaunchConfiguration('odom_topic'))],
            output='screen',
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', LaunchConfiguration('rviz_cfg')],
            parameters=[{'use_sim_time': parameter('use_sim_time', bool)}],
            condition=IfCondition(LaunchConfiguration('rviz')),
            output='screen',
        ),
    ])
