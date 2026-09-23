from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class TerrainMap:
    elevation: np.ndarray
    roughness: np.ndarray
    obstacle: np.ndarray
    unsafe_landing: np.ndarray
    resolution: float = 1.0

    def __post_init__(self) -> None:
        shapes = {a.shape for a in (self.elevation, self.roughness, self.obstacle, self.unsafe_landing)}
        if len(shapes) != 1:
            raise ValueError("all terrain layers must have the same shape")
        gy, gx = np.gradient(self.elevation, self.resolution)
        self.slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
        self.traversability = np.clip(1.0 - self.roughness - self.obstacle.astype(float), 0.0, 1.0)
        self.flight_cost = 0.05 + 0.02 * np.clip(self.elevation, 0.0, None)
        self.landing_cost = (
            0.5
            + 4.0 * self.roughness
            + 20.0 * self.obstacle.astype(float)
            + 20.0 * self.unsafe_landing.astype(float)
            + 0.04 * self.slope_deg
        )

    @property
    def height(self) -> int:
        return int(self.elevation.shape[0])

    @property
    def width(self) -> int:
        return int(self.elevation.shape[1])

    @property
    def extent_m(self) -> tuple[float, float]:
        return self.width * self.resolution, self.height * self.resolution

    def to_index(self, x: float, y: float) -> tuple[int, int]:
        ix = int(np.clip(round(x / self.resolution), 0, self.width - 1))
        iy = int(np.clip(round(y / self.resolution), 0, self.height - 1))
        return iy, ix

    def contains(self, x: float, y: float) -> bool:
        return 0.0 <= x <= (self.width - 1) * self.resolution and 0.0 <= y <= (self.height - 1) * self.resolution

    def value(self, layer: str, x: float, y: float) -> float:
        if not self.contains(x, y):
            return float("inf")
        iy, ix = self.to_index(x, y)
        return float(getattr(self, layer)[iy, ix])

    def continuous_value(self, layer: str, x: float, y: float) -> float:
        """Bilinearly interpolate a scalar terrain layer in world coordinates."""
        if not self.contains(x, y):
            return float("inf")
        values = np.asarray(getattr(self, layer), dtype=float)
        fx = x / self.resolution
        fy = y / self.resolution
        x0 = int(np.floor(fx))
        y0 = int(np.floor(fy))
        x1 = min(x0 + 1, self.width - 1)
        y1 = min(y0 + 1, self.height - 1)
        tx = fx - x0
        ty = fy - y0
        return float(
            (1.0 - tx) * (1.0 - ty) * values[y0, x0]
            + tx * (1.0 - ty) * values[y0, x1]
            + (1.0 - tx) * ty * values[y1, x0]
            + tx * ty * values[y1, x1]
        )

    def continuous_occupied(self, layer: str, x: float, y: float) -> bool:
        """Conservatively query a closed boolean grid cell region."""
        if not self.contains(x, y):
            return True
        values = np.asarray(getattr(self, layer), dtype=bool)
        fx = x / self.resolution
        fy = y / self.resolution
        eps = 1e-10
        x_indices = {
            int(np.clip(np.floor(fx + 0.5 - eps), 0, self.width - 1)),
            int(np.clip(np.floor(fx + 0.5 + eps), 0, self.width - 1)),
        }
        y_indices = {
            int(np.clip(np.floor(fy + 0.5 - eps), 0, self.height - 1)),
            int(np.clip(np.floor(fy + 0.5 + eps), 0, self.height - 1)),
        }
        return any(values[iy, ix] for iy in y_indices for ix in x_indices)

    def continuous_safe_landing(self, x: float, y: float, threshold: float) -> bool:
        return (
            self.contains(x, y)
            and not self.continuous_occupied("obstacle", x, y)
            and not self.continuous_occupied("unsafe_landing", x, y)
            and self.continuous_value("landing_cost", x, y) <= threshold
        )

    def continuous_ground_cost(self, x: float, y: float, max_slope_deg: float) -> float:
        if not self.contains(x, y) or self.continuous_occupied("obstacle", x, y):
            return float("inf")
        rough = self.continuous_value("roughness", x, y)
        slope = self.continuous_value("slope_deg", x, y)
        if slope > max_slope_deg:
            return float("inf")
        return float(rough * 8.0 + (slope / max(max_slope_deg, 1e-6)) * 3.0)

    def is_obstacle(self, x: float, y: float) -> bool:
        return bool(self.value("obstacle", x, y) >= 0.5)

    def is_safe_landing(self, x: float, y: float, threshold: float) -> bool:
        return self.contains(x, y) and self.value("landing_cost", x, y) <= threshold and not self.is_obstacle(x, y)

    def ground_cost(self, x: float, y: float, max_slope_deg: float) -> float:
        if not self.contains(x, y):
            return float("inf")
        rough = self.value("roughness", x, y)
        slope = self.value("slope_deg", x, y)
        obstacle = self.value("obstacle", x, y)
        if obstacle >= 0.5 or slope > max_slope_deg:
            return float("inf")
        return rough * 8.0 + (slope / max(max_slope_deg, 1e-6)) * 3.0

    def line_samples(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        step_m: float,
    ) -> Iterable[tuple[float, float]]:
        sx, sy = start
        ex, ey = end
        dist = float(np.hypot(ex - sx, ey - sy))
        count = max(2, int(np.ceil(dist / max(step_m, 1e-6))) + 1)
        for t in np.linspace(0.0, 1.0, count):
            yield sx + (ex - sx) * t, sy + (ey - sy) * t


def combined_ground_landing_cost(terrain: TerrainMap) -> np.ndarray:
    ground = np.nan_to_num(terrain.roughness * 8.0 + terrain.slope_deg / 5.0, nan=0.0, posinf=20.0)
    return ground + terrain.landing_cost


def make_default_terrain() -> TerrainMap:
    """Build the reference terrain used by the default DP example."""
    elevation, roughness, obstacle, unsafe_landing = _hybrid_challenge_layers(56, 31)
    return TerrainMap(
        elevation=_limit_elevation_slope(elevation, 1.0, 30.0),
        roughness=roughness,
        obstacle=obstacle,
        unsafe_landing=unsafe_landing,
        resolution=1.0,
    )


def _limit_elevation_slope(
    elevation: np.ndarray,
    resolution: float,
    max_slope_deg: float,
) -> np.ndarray:
    """Normalize non-flat generated relief so its maximum slope is the target."""
    gy, gx = np.gradient(elevation, resolution)
    maximum_grade = float(np.max(np.hypot(gx, gy)))
    allowed_grade = float(np.tan(np.deg2rad(max_slope_deg)))
    if maximum_grade <= 1e-12:
        return elevation
    center = float(np.mean(elevation))
    scale = allowed_grade / maximum_grade
    return center + (elevation - center) * scale


def _hybrid_challenge_layers(width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys = _grid(width, height)
    cy = (height - 1) * 0.5
    route_progress = np.clip((xs - 8.0) / (45.0 - 8.0), 0.0, 1.0)
    route_progress = route_progress**2 * (3.0 - 2.0 * route_progress)
    mission_route_y = cy + (28.0 - cy) * route_progress
    mission_corridor = np.exp(-((ys - mission_route_y) ** 2) / (2.0 * 2.4**2))
    mission_corridor_clear = np.abs(ys - mission_route_y) <= 3.2
    elevation = (
        0.18 * np.sin(xs / 2.4)
        + 0.16 * np.cos(ys / 2.1)
        + _gaussian(xs, ys, cx=18.0, cy=cy - 7.0, sx=4.8, sy=3.2, amp=1.5)
        + _gaussian(xs, ys, cx=29.0, cy=cy + 7.0, sx=6.2, sy=3.4, amp=1.9)
        + _gaussian(xs, ys, cx=43.0, cy=cy - 5.0, sx=5.6, sy=3.0, amp=1.4)
        - _gaussian(xs, ys, cx=36.0, cy=cy - 1.5, sx=4.5, sy=2.6, amp=0.8)
    )
    corridor = np.exp(-((ys - cy) ** 2) / (2.0 * 2.6**2))
    corridor_profile = (
        0.24 * np.sin(xs / 3.2)
        + 0.13 * np.sin(xs / 1.55)
        + 0.08 * np.cos((xs + 0.7 * ys) / 2.8)
        + _gaussian(xs, ys, cx=22.0, cy=cy + 1.2, sx=3.0, sy=1.7, amp=0.42)
        - _gaussian(xs, ys, cx=31.0, cy=cy - 1.0, sx=2.5, sy=1.4, amp=0.34)
        + _gaussian(xs, ys, cx=41.0, cy=cy + 0.8, sx=3.8, sy=1.8, amp=0.36)
    )
    elevation = elevation * (1.0 - 0.68 * corridor) + corridor_profile * (0.68 * corridor)
    approach_guard = 1.0 - np.exp(-((ys - cy) ** 2) / (2.0 * 1.3**2))
    approach_relief = (
        _gaussian(xs, ys, cx=7.0, cy=cy + 6.2, sx=2.2, sy=1.9, amp=1.15)
        + _gaussian(xs, ys, cx=7.0, cy=cy - 6.2, sx=2.2, sy=1.9, amp=1.15)
        + _gaussian(xs, ys, cx=12.0, cy=cy + 9.5, sx=2.0, sy=1.7, amp=0.88)
        + _gaussian(xs, ys, cx=12.0, cy=cy - 9.5, sx=2.0, sy=1.7, amp=0.88)
        - _gaussian(xs, ys, cx=10.0, cy=cy + 2.8, sx=1.8, sy=1.3, amp=0.46)
        - _gaussian(xs, ys, cx=10.0, cy=cy - 2.8, sx=1.8, sy=1.3, amp=0.46)
    )
    elevation += approach_relief * approach_guard
    mission_profile = 0.10 * np.sin(xs / 5.0) + 0.05 * np.cos(xs / 2.7)
    elevation = elevation * (1.0 - 0.78 * mission_corridor) + mission_profile * (0.78 * mission_corridor)

    roughness = 0.08 + 0.06 * (np.sin(xs / 2.7) * np.cos(ys / 2.2) + 1.0)
    roughness += _gaussian(xs, ys, cx=8.5, cy=cy + 4.5, sx=3.0, sy=2.5, amp=0.90)
    roughness += _gaussian(xs, ys, cx=8.5, cy=cy - 4.5, sx=3.0, sy=2.5, amp=0.90)
    roughness += _gaussian(xs, ys, cx=24.0, cy=cy, sx=4.5, sy=5.0, amp=0.55)
    roughness += _gaussian(xs, ys, cx=36.0, cy=cy + 4.0, sx=4.0, sy=2.6, amp=0.58)
    roughness += _gaussian(xs, ys, cx=44.0, cy=cy - 5.0, sx=3.8, sy=2.4, amp=0.50)
    roughness += 0.12 * corridor * (0.5 + 0.5 * np.sin(xs / 2.4 + ys / 3.1))
    roughness += 0.75 * corridor * _gaussian(xs, ys, cx=34.0, cy=cy, sx=5.5, sy=2.4, amp=1.0)
    roughness += 0.45 * corridor * _gaussian(xs, ys, cx=34.0, cy=cy, sx=2.8, sy=2.0, amp=1.0)
    roughness[(ys >= cy + 1.0) & (ys <= cy + 6.0) & (xs >= 4.0) & (xs <= 7.0)] = 0.96
    roughness[(ys >= cy - 6.0) & (ys <= cy - 1.0) & (xs >= 4.0) & (xs <= 7.0)] = 0.96
    roughness[(np.abs(ys - cy) <= 3.6) & (xs >= 16.0) & (xs <= 46.0)] *= 0.45
    roughness += 0.05 * corridor * (np.sin(xs / 1.7) ** 2)
    roughness[(np.abs(ys - cy) <= 3.0) & (xs >= 29.0) & (xs <= 39.0)] = np.maximum(
        roughness[(np.abs(ys - cy) <= 3.0) & (xs >= 29.0) & (xs <= 39.0)],
        0.82,
    )
    roughness = np.clip(roughness, 0.0, 0.96)
    route_roughness = 0.07 + 0.04 * np.sin(xs / 2.8) ** 2
    roughness[mission_corridor_clear] = np.minimum(
        roughness[mission_corridor_clear],
        np.broadcast_to(route_roughness, roughness.shape)[mission_corridor_clear],
    )
    rough_gate = mission_corridor_clear & (xs >= 29.0) & (xs <= 33.0)
    roughness[rough_gate] = np.maximum(roughness[rough_gate], 0.22)

    obstacle = np.zeros((height, width), dtype=bool)
    obstacle[(ys >= cy - 8.0) & (ys <= cy + 8.0) & (xs >= 25.0) & (xs <= 28.0)] = True
    obstacle[(ys >= cy + 6.0) & (ys <= cy + 10.0) & (xs >= 40.0) & (xs <= 47.0)] = True
    obstacle[(ys >= cy - 10.0) & (ys <= cy - 6.0) & (xs >= 38.0) & (xs <= 45.0)] = True
    _add_disk(obstacle, 17, int(round(cy + 7.0)), 2)
    _add_disk(obstacle, 17, int(round(cy - 7.0)), 2)
    _add_disk(obstacle, 45, int(round(cy + 1.0)), 2)
    corridor_clear = np.broadcast_to(np.abs(ys - cy) <= 3.5, obstacle.shape)
    obstacle[corridor_clear] = False
    obstacle[mission_corridor_clear] = False

    unsafe_landing = np.zeros((height, width), dtype=bool)
    unsafe_landing[(ys >= cy - 8.0) & (ys <= cy + 8.0) & (xs >= 20.0) & (xs <= 33.0)] = True
    unsafe_landing[corridor_clear] = False
    _add_disk(unsafe_landing, 42, int(round(cy + 7.0)), 3)
    _add_disk(unsafe_landing, 40, int(round(cy - 7.0)), 3)
    unsafe_landing[(ys >= cy - 5.0) & (ys <= cy + 5.0) & (xs >= 4.0) & (xs <= 14.0)] = False
    unsafe_landing[mission_corridor_clear] = False
    return elevation, roughness, obstacle, unsafe_landing


def _grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(width, dtype=float)[None, :]
    ys = np.arange(height, dtype=float)[:, None]
    return xs, ys


def _gaussian(
    xs: np.ndarray,
    ys: np.ndarray,
    cx: float,
    cy: float,
    sx: float,
    sy: float,
    amp: float,
) -> np.ndarray:
    return amp * np.exp(-(((xs - cx) ** 2) / (2.0 * sx**2) + ((ys - cy) ** 2) / (2.0 * sy**2)))


def _add_disk(mask: np.ndarray, cx: int, cy: int, radius: int) -> None:
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2] = True
