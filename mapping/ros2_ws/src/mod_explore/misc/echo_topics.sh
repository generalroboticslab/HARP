#!/bin/bash

SESSION=topics_echo

# kill any existing session with the same name
tmux kill-session -t "$SESSION" 2>/dev/null || true

# echo audio
tmux new-session -d -s "$SESSION" -n topics
tmux send-keys -t "$SESSION":0 'ros2 topic echo /audio/multichannel' C-m
tmux select-pane -t "$SESSION":0.0 -T "Audio"

# echo leader
tmux split-window -h -t "$SESSION":0
tmux send-keys -t "$SESSION":0.1 'ros2 topic echo /leader/mavros/local_position/pose' C-m
tmux select-pane -t "$SESSION":0.1 -T "Leader"

# echo follower
tmux split-window -v -t "$SESSION":0.1
tmux send-keys -t "$SESSION":0.2 'ros2 topic echo /follower/mavros/local_position/pose' C-m
tmux select-pane -t "$SESSION":0.2 -T "Follower"

# focus back to the first pane and attach
tmux select-pane -t "$SESSION":0.0
if [[ "${NO_ATTACH:-0}" != "1" ]]; then
  tmux attach -t "$SESSION"
fi


