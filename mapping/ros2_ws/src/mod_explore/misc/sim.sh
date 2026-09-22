#!/bin/bash
SESSION=multi_sim
PX4_DIR="$HOME/PX4-Autopilot"
ROS_WS="$HOME/mod_explore/ros2_ws"

tmux kill-session -t "$SESSION" 2>/dev/null || true

# launch gazebo and leader PX4
tmux new-session -d -s "$SESSION" -n gazebo
tmux send-keys -t "$SESSION":0 "cd $PX4_DIR && PX4_GZ_MODEL_POSE=\"0,0,0.3,0,0,1.57\" make px4_sitl gz_x500" C-m

# multi-UAV pane
tmux split-window -h -t "$SESSION":0
tmux send-keys -t "$SESSION":0.1 "sleep 8 && cd $PX4_DIR && PX4_GZ_STANDALONE=1 PX4_SYS_AUTOSTART=4001 PX4_GZ_MODEL_POSE=\"-4,0,0.3,0,0,1.57\" PX4_SIM_MODEL=gz_x500 ./build/px4_sitl_default/bin/px4 -i 1" C-m

# launch mavros
tmux new-window -t "$SESSION":1 -n mavros
tmux send-keys -t "$SESSION":1 "sleep 12 && cd $ROS_WS && source install/setup.bash && ros2 launch mod_explore uav_leader_gazebo.launch" C-m
tmux split-window -h -t "$SESSION":1
tmux send-keys -t "$SESSION":1.1 "sleep 12 && cd $ROS_WS && source install/setup.bash && ros2 launch mod_explore uav_follower_gazebo.launch tgt_system:=2" C-m

# launch rviz & acoustic inference
# tmux new-window -t "$SESSION":2 -n rviz
# tmux send-keys -t "$SESSION":2 "cd $ROS_WS && source install/setup.bash && ros2 launch scout rviz.launch.py" C-m
# tmux split-window -h -t "$SESSION":2
# tmux send-keys -t "$SESSION":2.1 "cd $ROS_WS && source install/setup.bash && ros2 run scout z_acoustic_pub" C-m

# attach to it
tmux attach -t "$SESSION":0.1
