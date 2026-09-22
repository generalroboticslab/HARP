"""Launch leader CSV replay and C++ Duke follower together."""

import os

from launch import LaunchDescription
from launch_ros.actions import Node

# 0 0505_st2
# 1 0505_rh2
# 2 0505_lu3
# 3 0505_ru0_
# 4 0505_st45
# 5 0505_lh45
# 6 0508_c0
# 7 0505_st6

CSV_DIR = "/home/grl-deploy-0/SonicScout_dev_clean/process/single_traj_plots"
RUN_ID = "0505_lh45"

LEADER_CSV_PATH = os.path.join(
    CSV_DIR,
    f"{RUN_ID}_leader_relative_to_home.csv",
)
FOLLOWER_CSV_PATH = os.path.join(
    CSV_DIR,
    f"{RUN_ID}_follower_relative_to_leader.csv",
)

# FOLLOWER_CSV_PATH = os.path.join(
#     CSV_DIR,
#     f"{RUN_ID}_follower_relative_to_home.csv",
# )

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='scout_cpp',
            executable='x500_csv_replay',
            output='screen',
            parameters=[{
                'ns': '/leader/mavros',
                'rate_hz': 20.0,
                'takeoff_alt': 6.0,
                'warmup_sec': 2.0,
                'alt_tol': 0.3,
                'lock_yaw': True,
                'yaw_offset_deg': 0.0,
                'csv_path': LEADER_CSV_PATH,
                'trajectory_use_csv_time': True,
                'csv_yaw_mode': 'relative',
                'rotate_xy_to_initial_yaw': True,
            }],
        ),


        Node(
            package='scout_cpp',
            executable='x500_follower_duke_cpp',
            output='screen',
            parameters=[{
                'ns': '/follower/mavros',
                'rate_hz': 20.0,
                'takeoff_alt': 6.5,
                'warmup_sec': 2.0,
                'alt_tol': 0.3,
                'yaw_offset_deg': 0.0,
                'leader_home_global_alt': -1.0,
                'rel_csv_path': FOLLOWER_CSV_PATH,
                'trajectory_use_csv_time': True,
                'reverse_x': False,
                'sim_offset': [0.0, 0.0, 0.0],
                'leader_pose_topic': '/leader/mavros/local_position/pose',
                'rotate_rel_xy_to_initial_yaw': True,
            }],
        ),

        # Node(
        #     package='scout_cpp',
        #     executable='x500_csv_replay',
        #     output='screen',
        #     parameters=[{
        #         'ns': '/follower/mavros',
        #         'rate_hz': 20.0,
        #         'takeoff_alt': 5.5,
        #         'warmup_sec': 2.0,
        #         'alt_tol': 0.3,
        #         'lock_yaw': True,
        #         'yaw_offset_deg': 0.0,
        #         'csv_path': FOLLOWER_CSV_PATH,
        #         # Use t_relative_s column for playback speed; set false for one-row-per-tick.
        #         'trajectory_use_csv_time': True,
        #         # 'relative' = keep yaw delta vs CSV start; 'absolute' = use CSV yaw directly.
        #         'csv_yaw_mode': 'relative',
        #     }],
        # ),

    ])
