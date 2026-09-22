from __future__ import annotations

from dataclasses import dataclass
from math import atan, cos, degrees, hypot, pi, sin
from typing import Iterable

import numpy as np

from .terrain import TerrainMap

Pose2D = tuple[float, float, float]


@dataclass(frozen=True)
class RectFootprint:
    length_m: float
    width_m: float


@dataclass(frozen=True)
class FormationCost:
    total: float
    mean: float
    variance: float
    rover_costs: tuple[float, ...]
    roll_deg: float
    pitch_deg: float


def platform_to_rover_poses(platform_pose: Pose2D, rover_offsets: Iterable[Pose2D]) -> list[Pose2D]:
    """Transform rover offsets from the docked platform frame into world poses."""
    px, py, theta = platform_pose
    c = cos(theta)
    s = sin(theta)
    poses: list[Pose2D] = []
    for dx, dy, dtheta in rover_offsets:
        x = px + c * dx - s * dy
        y = py + s * dx + c * dy
        poses.append((x, y, wrap_angle(theta + dtheta)))
    return poses


def platform_terrain_cost(
    platform_pose: Pose2D,
    rover_offsets: Iterable[Pose2D],
    terrain: TerrainMap,
    max_slope_deg: float,
    lambda_var: float = 1.0,
) -> FormationCost:
    rover_poses = platform_to_rover_poses(platform_pose, rover_offsets)
    costs = []
    for x, y, _ in rover_poses:
        cost = terrain.ground_cost(x, y, max_slope_deg)
        if not np.isfinite(cost):
            return FormationCost(float("inf"), float("inf"), float("inf"), tuple(), float("inf"), float("inf"))
        costs.append(float(cost))
    mean = float(np.mean(costs)) if costs else 0.0
    variance = float(np.var(costs)) if costs else 0.0
    roll_deg, pitch_deg = estimate_platform_roll_pitch(platform_pose, rover_offsets, terrain)
    total = float(sum(costs) + lambda_var * variance)
    return FormationCost(total, mean, variance, tuple(costs), roll_deg, pitch_deg)


def estimate_platform_roll_pitch(
    platform_pose: Pose2D,
    rover_offsets: Iterable[Pose2D],
    terrain: TerrainMap,
) -> tuple[float, float]:
    points = []
    for x, y, _ in platform_to_rover_poses(platform_pose, rover_offsets):
        z = terrain.value("elevation", x, y)
        if not np.isfinite(z):
            return float("inf"), float("inf")
        points.append((x, y, z))
    if len(points) < 3:
        return 0.0, 0.0
    a = np.array([[x, y, 1.0] for x, y, _ in points], dtype=float)
    b = np.array([z for _, _, z in points], dtype=float)
    slope_x, slope_y, _ = np.linalg.lstsq(a, b, rcond=None)[0]
    pitch = degrees(atan(float(slope_x)))
    roll = degrees(atan(float(slope_y)))
    return roll, pitch


def footprint_collision(
    pose: Pose2D,
    footprint: RectFootprint,
    terrain: TerrainMap,
    sample_step_m: float,
    max_slope_deg: float | None = None,
) -> bool:
    for x, y in rectangle_sample_points(pose, footprint, sample_step_m):
        if not terrain.contains(x, y) or terrain.is_obstacle(x, y):
            return True
        if max_slope_deg is not None and not np.isfinite(terrain.ground_cost(x, y, max_slope_deg)):
            return True
    return False


def platform_collision(
    platform_pose: Pose2D,
    rover_offsets: Iterable[Pose2D],
    platform_footprint: RectFootprint,
    rover_footprint: RectFootprint,
    terrain: TerrainMap,
    sample_step_m: float,
    max_slope_deg: float,
) -> bool:
    if footprint_collision(platform_pose, platform_footprint, terrain, sample_step_m):
        return True
    for rover_pose in platform_to_rover_poses(platform_pose, rover_offsets):
        if footprint_collision(rover_pose, rover_footprint, terrain, sample_step_m, max_slope_deg=max_slope_deg):
            return True
    return False


def rectangle_sample_points(pose: Pose2D, footprint: RectFootprint, sample_step_m: float) -> list[tuple[float, float]]:
    half_l = footprint.length_m * 0.5
    half_w = footprint.width_m * 0.5
    step = max(sample_step_m, 1e-6)
    xs = np.arange(-half_l, half_l + step, step)
    ys = np.arange(-half_w, half_w + step, step)
    corners = [(-half_l, -half_w), (-half_l, half_w), (half_l, -half_w), (half_l, half_w)]
    local_points = [(float(x), float(y)) for x in xs for y in ys] + corners
    px, py, theta = pose
    c = cos(theta)
    s = sin(theta)
    return [(px + c * x - s * y, py + s * x + c * y) for x, y in local_points]


def rover_paths_from_platform_path(path: Iterable[Pose2D], rover_offsets: Iterable[Pose2D]) -> dict[str, list[Pose2D]]:
    offsets = list(rover_offsets)
    paths = {f"rover_{i + 1}": [] for i in range(len(offsets))}
    for platform_pose in path:
        for i, pose in enumerate(platform_to_rover_poses(platform_pose, offsets)):
            paths[f"rover_{i + 1}"].append(pose)
    return paths


def pose_distance(a: Pose2D, b: Pose2D) -> float:
    return float(hypot(b[0] - a[0], b[1] - a[1]))


def wrap_angle(angle: float) -> float:
    return (float(angle) + pi) % (2.0 * pi) - pi


def angle_error(a: float, b: float) -> float:
    return abs(wrap_angle(a - b))
