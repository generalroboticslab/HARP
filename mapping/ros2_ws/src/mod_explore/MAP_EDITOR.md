# Build, export, and edit a 2.5D map

Build the mapper and exporter, then launch them alongside your running FAST-LIO:

```bash
cd mod_explore/ros2_ws
colcon build --symlink-install --packages-select mod_explore_cpp mod_explore
source install/setup.bash
ros2 launch mod_explore map_25d.launch.py
```

This standalone launch starts `lio_post`, `map_recorder`, and RViz. It does not
start FAST-LIO or load its package assets. Required inputs are:

| Argument | Default topic | Message type |
| --- | --- | --- |
| `cloud_topic` | `/cloud_registered` | `sensor_msgs/msg/PointCloud2` |
| `odom_topic` | `/Odometry` | `nav_msgs/msg/Odometry` |

Use registered clouds and their corresponding odometry. The mapper retains the
existing upside-down sensor correction (180° pitch); this is not a raw-scan
registration pipeline. Output frame: `camera_init`. By default, MAVROS is not
required. To align the corrected map with the first MAVROS pose, set
`use_mavros_reference:=true`; mapping then also waits for a
`geometry_msgs/msg/PoseStamped` on `mavros_pose_topic` (default
`/follower/mavros/local_position/pose`).

`map_recorder` saves the latest complete `/explore_map_color` message, the colored
2.5D cell map shown in RViz. Each point retains its cell-center XY, relative height
Z, and packed RGB. The mapper applies range filtering and per-cell maximum
relative heights; export and editing do not recompute the colors. Each new message
replaces the snapshot, even if only heights or colors change.

After three seconds without a new map, or on normal Ctrl-C shutdown, it writes:

```text
maps/<run timestamp>/map.pcd
maps/<run timestamp>/map_editor.html
```

The default `maps` directory is relative to the directory where you launch ROS.
Each run has a separate directory. The console prints the absolute output paths.
You can also export a snapshot explicitly:

```bash
ros2 service call /save_map_editor std_srvs/srv/Trigger '{}'
```

For bag playback, start mapping before the bag. If using simulated time, launch
with `use_sim_time:=true` and play the bag with `--clock`. Idle export still works
when the bag clock is paused. Pause playback before a manual snapshot of a large
map so file writing does not compete with incoming scans.

Launch options:

```bash
ros2 launch mod_explore map_25d.launch.py cloud_topic:=/my/cloud odom_topic:=/my/odom
ros2 launch mod_explore map_25d.launch.py map_output_dir:=/path/to/maps map_export_idle_seconds:=5.0
ros2 launch mod_explore map_25d.launch.py map_resolution:=0.2 update_range_xy:=6.0
ros2 launch mod_explore map_25d.launch.py export_map:=false rviz:=false
```

Default cell size is 0.2 m, update radius is 6 m, and vegetation height-range
threshold is 0.5 m (`vegetation_range_max`). RViz displays the colored map and
trajectory; enable **Occupancy map** for `/explore_map`. Use `rviz_cfg` to supply
another RViz configuration.

Set `map_export_idle_seconds:=0.0` for manual/shutdown export only. Normal shutdown
allows up to 120 seconds for writing. A force-killed process cannot finish an
export. The latest map is held in memory until exported.

Open `map_editor.html` directly in a browser. It contains the map and works
offline, without a web server:

1. Use the 3D view to inspect the map: drag to rotate, Shift/right-drag to pan, and
   scroll to zoom. Switch to Top view for cropping; drag the four numbered corners
   to enclose the cells to keep. Cells are selected by their center XY, including
   centers on the boundary. Their published Z values and colors are retained.
2. Pick a map point for the new origin. Its XYZ coordinates become zero in the
   saved cloud. You can adjust XYZ numerically, especially where surfaces overlap
   in the top-down view.
3. Pick the direction that should become positive X (yaw zero), or enter its angle
   in degrees. Positive angles are counterclockwise from the original positive X.
4. Save the edited PCD. The browser downloads a separate file and leaves the
   original map intact. The preview may show fewer points for speed; export uses
   every cell in the saved snapshot.

The saved coordinates are `Rz(-yaw) * (point - origin)`. Only yaw and translation
are changed; roll and pitch remain unchanged. Published RGB values are retained
even after moving the origin, changing yaw, or cropping. PCD output is binary PCD
0.7 with `x y z rgb` fields; `rgb` is a packed 32-bit color stored in a FLOAT32
field, following the
[PCL file format](https://pointclouds.org/documentation/tutorials/pcd_file_format.html).

Open the generated HTML in a browser tab, not an editor's static HTML preview.
You can also open `mod_explore/map_editor.html` directly and choose **Open PCD** to
load a saved file. The picker accepts this exporter's binary `x y z rgb` or
`x y z intensity` format. PCD comments retain the cell size for later display.
Existing exports made from raw scans remain raw scans; export a new map to obtain
the colored 2.5D version. Loading an old PCD does not reconstruct the 2.5D map or
its published colors.
