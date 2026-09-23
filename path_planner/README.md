# HARP air/ground DP planner

Plans routes for four rover-drones and one drill using a shared value function.
Requires Python 3.10+. From the HARP repository root:

```bash
cd path_planner
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

python3 -m script.dynamic_programming_air_ground \
  --map-25d maps/slope7p5_crop/map_25d.npz \
  --initial 6,16 --initial 4,35 --initial 6,50 \
  --target 55,30 --max-ground-slope-deg 7.5 \
  --output results/dp_slope7p5.json \
  --visualize-cost-to-go --overlay-policies
```

Writes `results/dp_slope7p5.json`, two PNGs (`_cost_to_go`, `_path_planning_3d`),
and interactive `_cost_to_go_layers_3d.html`. Use a new output name for each run,
or omit `--output` for timestamped names. On headless machines, prefix the
command with `MPLBACKEND=Agg`. Use the virtual environment above to avoid mixed
Matplotlib installations (`Unknown projection '3d'`).

Coordinates are raster-relative meters; the sample grid spacing is 1 m.
See [map metadata](maps/slope7p5_crop/metadata.json) and
[vehicle/energy settings](config/default.yaml).

Planning constraints:

- Takeoff locations are optimized among feasible grid positions. The optional
  Python-only `takeoff_y_range_m` restriction is disabled in this command.
- Both driving and flight are required by default. Add `--allow-single-mode`
  for optional flight, or `--ground-only` for driving only.
- Routes start and finish on the ground. Landing at the target is forbidden,
  so the final approach must drive, including with `--allow-single-mode`.
- Each flight leg must cover at least 1 m on this map; no maximum is imposed.
- Takeoff and landing require a feasible platform footprint, slope/attitude
  limits, and the map's landing-safety checks.

```text
path_planner/
├── script/           # DP planner, PCD conversion, energy model, and visualization
├── config/           # vehicle, energy, and planner settings
├── maps/             # input maps, metadata, and previews
│   ├── hybrid_challenge.html  # default map, also used by regression tests
│   └── slope7p5_crop/
├── tests/            # planner regression tests
├── requirements.txt
└── README.md
```

`results/` is created when the planner runs and is ignored by Git.

Run the DP entry point as a module from `path_planner/`:

```bash
python3 -m script.dynamic_programming_air_ground --help
```

The DP planner uses `maps/hybrid_challenge.html` when `--map-25d` is omitted.
Pass an existing `map_25d.npz` with `--map-25d` to plan on a recorded raster map.
The `dp` section of `config/default.yaml` controls footprint sampling, attitude
limits, and docking energy; shared vehicle, battery, and platform settings are
in the same file.

[`script/pcd_25d.py`](script/pcd_25d.py) prepares LiDAR point clouds, such as
FAST-LIO2 PCD output, for the DP planner's 2.5D terrain grid. It estimates
elevation and roughness, derives slope, obstacle, and landing-safety layers,
and saves `map_25d.npz`, map metadata, and a preview image. This lets DP plan
on recorded terrain instead of the built-in example map.

Convert a point cloud from `path_planner/`:

```bash
python3 -m script.pcd_25d /path/to/cloud.pcd --output-dir maps/my_map
```

Use `--help` for crop, resolution, and terrain options, then pass
`--map-25d maps/my_map/map_25d.npz` to the DP planner.

Run tests from `path_planner/`:

```bash
python3 -m pip install pytest
python3 -m pytest -q
```
