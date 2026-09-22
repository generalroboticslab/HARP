tmux kill-session -t multi_sim 2>/dev/null || true
tmux kill-session -t scout 2>/dev/null || true
tmux kill-session -t topics_echo 2>/dev/null || true

# cleanup stray Gazebo / PX4 if they remain
# Try to kill as user first, then with sudo if needed (for root processes)
pkill -f "gz sim" 2>/dev/null || true
pkill -f px4 2>/dev/null || true
# If processes are running as root, use sudo (will prompt for password if needed)
if pgrep -f "gz sim" > /dev/null 2>&1 || pgrep -f px4 > /dev/null 2>&1; then
    echo "Some processes are running as root, attempting to kill with sudo..."
    sudo pkill -f "gz sim" 2>/dev/null || true
    sudo pkill -f px4 2>/dev/null || true
fi
