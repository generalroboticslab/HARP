from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .terrain import TerrainMap
from .visualize_html3d import (
    _html,
    _state_point,
    _surface_display_range,
    _surface_layer,
)


def export_dp_html3d(
    result: dict[str, Any],
    output: str | Path,
    z_scale: float = 1.0,
    surface_layer: str = "combined_cost",
    *,
    terrain: TerrainMap,
) -> Path:
    """Write a self-contained interactive HTML view of the one-state team path."""
    layer = _surface_layer(terrain, surface_layer)
    display_min, display_max = _surface_display_range(layer, surface_layer)
    cruise_clearance_m = float(
        result.get("model_assumptions", {}).get("cruise_altitude_m", 5.0)
    )
    fixed_flight_altitude_m = float(np.max(terrain.elevation)) + cruise_clearance_m
    paths = []
    for event in result.get("mission_events", []):
        action = str(event["action"])
        is_flight = action in {"takeoff", "fly", "land"}
        paths.append(
            {
                "name": f"{event['event_index']}: {action}",
                "kind": "flight" if is_flight else "platform",
                "color": "#ff0000",
                "shadow": "#020617" if not is_flight else None,
                "points": _level_event_points(
                    terrain,
                    event.get("path", []),
                    action,
                    fixed_flight_altitude_m,
                    z_scale,
                ),
                "eventIndex": int(event["event_index"]),
                "phase": action,
                "startTime": float(event["start_time_s"]),
                "endTime": float(event["end_time_s"]),
            }
        )

    home = result.get("home", {})
    target = result.get("target", {})
    markers = []
    if home:
        markers.append(
            {
                "name": "Home: assembled team",
                "color": "#38bdf8",
                "shape": "circle",
                "point": _state_point(terrain, home, z_scale, 0.55),
            }
        )
    if target:
        markers.append(
            {
                "name": "Target: all five assembled",
                "color": "#facc15",
                "shape": "star",
                "point": _state_point(terrain, target, z_scale, 0.65),
            }
        )

    timeline_by_index = {
        int(item["event_index"]): item for item in result.get("mission_timeline", [])
    }
    timeline = []
    for event in result.get("mission_events", []):
        item = timeline_by_index.get(int(event["event_index"]), {})
        timeline.append(
            {
                **item,
                "label": item.get("label", str(event["action"])),
                "startTime": float(event["start_time_s"]),
                "endTime": float(event["end_time_s"]),
            }
        )

    scene = {
        "title": (
            f"{result.get('planner', 'Air-Ground Dynamic Programming')}: "
            f"{result['scenario']} | {result.get('objective_energy_wh', float('inf')):.3f} Wh"
        ),
        "success": bool(result.get("success")),
        "width": terrain.width,
        "height": terrain.height,
        "resolution": terrain.resolution,
        "zScale": z_scale,
        "fixedFlightAltitudeM": fixed_flight_altitude_m,
        "surfaceLayer": surface_layer,
        "elevation": (terrain.elevation * z_scale).round(4).tolist(),
        "layer": layer.round(4).tolist(),
        "layerMin": float(np.min(layer)),
        "layerMax": float(np.max(layer)),
        "layerDisplayMin": display_min,
        "layerDisplayMax": display_max,
        "obstacle": terrain.obstacle.astype(int).tolist(),
        "unsafe": terrain.unsafe_landing.astype(int).tolist(),
        "paths": paths,
        "markers": markers,
        "timeline": timeline,
        "totalTime": float(result.get("mission_duration_s", 0.0)),
    }
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html = _html(scene)
    # The shared renderer's legend describes the older blue/white visualizer.
    # Make both the legend and the canvas stroke unconditionally bright red so
    # stale per-path styling cannot change the requested trajectory color.
    html = html.replace('style="--c:#60a5fa"', 'style="--c:#ff0000"')
    html = html.replace('style="--c:#f8fafc"', 'style="--c:#ff0000"')
    html = html.replace(
        "    path.color,\n    path.kind === 'platform' ? 4 : 2.6,",
        "    '#ff0000',\n    path.kind === 'platform' ? 4 : 2.6,",
    )
    html = html.replace(
        "    const active = currentTime >= path.startTime && currentTime < path.endTime;\n    if (active) {",
        "    const active = currentTime >= path.startTime && currentTime < path.endTime;\n"
        "    if (currentTime >= path.endTime) {\n"
        "      paths.push(path);\n"
        "      continue;\n"
        "    }\n"
        "    if (active) {",
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _level_event_points(
    terrain: Any,
    points: list[dict[str, float]],
    action: str,
    fixed_flight_altitude_m: float,
    z_scale: float,
) -> list[list[float]]:
    rendered: list[list[float]] = []
    for index, point in enumerate(points):
        airborne = (
            action == "fly"
            or (action == "takeoff" and index > 0)
            or (action == "land" and index < len(points) - 1)
        )
        z = (
            fixed_flight_altitude_m * z_scale
            if airborne
            else terrain.value("elevation", point["x"], point["y"]) * z_scale + 0.22
        )
        rendered.append([float(point["x"]), float(point["y"]), float(z)])
    return rendered
