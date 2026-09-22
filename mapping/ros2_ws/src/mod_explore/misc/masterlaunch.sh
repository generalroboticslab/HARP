#!/bin/bash

SESSION=scout

# kill any existing session
tmux kill-session -t "$SESSION" 2>/dev/null || true

# create session and launch
tmux new-session -d -s "$SESSION" -n follower
tmux send-keys -t "$SESSION":0 'followermin' C-m
tmux split-window -h -t "$SESSION":0
tmux send-keys -t "$SESSION":0.1 'audio' C-m

# enable mouse support for clickable panels
tmux set -g mouse on
# attach to the session (will show both panes) unless disabled
if [[ "${NO_ATTACH:-0}" != "1" ]]; then
  tmux attach -t "$SESSION"
fi
