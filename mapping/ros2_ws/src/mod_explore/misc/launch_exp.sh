#!/bin/bash

# Script to launch three tmux panes with commands ready to run
# Left pane: Run the acoustic model
# Top-right pane: Launch the scout follower
# Bottom-right pane: General terminal in ros2_ws

# Get the workspace directory
WS_DIR="$HOME/SonicScout_dev_clean"

# Create a new tmux session
SESSION_NAME="scout_experiment"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new session with first pane (left)
tmux new-session -d -s $SESSION_NAME -c "$WS_DIR/ros2_ws/src/scout/scout"

# Split the window horizontally to create right side
tmux split-window -h -t $SESSION_NAME:0.0 -c "$WS_DIR/data_logs"

# Split the window horizontally again (creates a third column)
tmux split-window -h -t $SESSION_NAME:0.0 -c "$WS_DIR/ros2_ws"

# Split the middle pane vertically to create bottom pane
tmux split-window -v -t $SESSION_NAME:0.1 -c "$WS_DIR/ros2_ws"

# Type commands in each pane
tmux send-keys -t $SESSION_NAME:0.0 "python acoustic_model_inference_gpu_filter.py"
tmux send-keys -t $SESSION_NAME:0.1 "ros2 launch scout x500_follower_v2_cpp.launch.py"

echo "Tmux session created: $SESSION_NAME"
echo "  Left pane (AcousticModel): Running python acoustic_model_inference_gpu_filter.py"
echo "  Top-right pane (ScoutLauncher): Running ros2 launch scout x500_follower_v2_cpp.launch.py"
echo "  Bottom-right pane: Ready in $WS_DIR"
echo ""

tmux set -g mouse on
# Attach to the session unless disabled
if [[ "${NO_ATTACH:-0}" != "1" ]]; then
  tmux attach -t $SESSION_NAME
fi
