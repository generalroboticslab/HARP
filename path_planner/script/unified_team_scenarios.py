from __future__ import annotations

import numpy as np

from .terrain import Scenario, TerrainMap, make_scenario


def make_unified_team_scenario(name: str) -> Scenario:
    """Return standard terrain or a terrain dedicated to collective-mode demos."""
    normalized = name.lower().replace("-", "_")
    if normalized != "multi_ridge_hops":
        return make_scenario(normalized)

    width, height = 31, 13
    xs = np.arange(width, dtype=float)[None, :]
    ys = np.arange(height, dtype=float)[:, None]
    elevation = (
        0.12 * np.sin(xs / 4.0)
        + 0.08 * np.cos(ys / 3.0)
        + 0.30 * np.exp(-((xs - 10.0) ** 2) / 5.0)
        + 0.34 * np.exp(-((xs - 21.0) ** 2) / 5.0)
    )
    elevation = np.broadcast_to(elevation, (height, width)).copy()
    roughness = np.full((height, width), 0.06, dtype=float)
    roughness += 0.12 * np.exp(-((xs - 10.0) ** 2) / 12.0)
    roughness += 0.14 * np.exp(-((xs - 21.0) ** 2) / 12.0)
    roughness = np.broadcast_to(roughness, (height, width)).copy()

    # Two map-spanning ridges divide the ground into three regions.  With the
    # demo's 8 m maximum collective flight leg, one flight cannot cross both;
    # the team must land and assemble in the safe central region before the
    # second collective takeoff.
    obstacle = np.zeros((height, width), dtype=bool)
    obstacle[:, 9:11] = True
    obstacle[:, 20:22] = True
    unsafe_landing = obstacle.copy()
    unsafe_landing[:, 8:12] = True
    unsafe_landing[:, 19:23] = True

    return Scenario(
        terrain=TerrainMap(
            elevation=elevation,
            roughness=roughness,
            obstacle=obstacle,
            unsafe_landing=unsafe_landing,
            resolution=1.0,
        ),
        start=(3.0, 6.0),
        goal=(27.0, 6.0),
    )
