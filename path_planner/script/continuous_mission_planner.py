from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import atan2, ceil, cos, hypot, pi, sin
from typing import Any, Literal

import numpy as np

from .platform import (
    Pose2D,
    RectFootprint,
    platform_to_rover_poses,
    rectangle_sample_points,
    rover_paths_from_platform_path,
    wrap_angle,
)
from .rover import (
    ground_speed_and_power,
    mass_scaled_flight_power,
    minimum_ground_energy_per_m,
)
from .terrain import TerrainMap


EdgeKind = Literal["H", "G", "F"]


@dataclass(frozen=True)
class ContinuousConnectorConfig:
    trunk_connection_k: int = 12
    trunk_connection_radius_m: float = 11.0
    trunk_air_connection_radius_m: float = 12.0
    formation_radius_m: float = 3.0
    formation_radius_jitter_m: float = 1.0
    formation_separation_time_s: float = 0.75
    branch_refinement_enabled: bool = True
    branch_global_candidate_limit: int = 16
    min_formation_branch_standoff_m: float = 2.6
    connector_step_m: float = 0.45
    max_curvature_1pm: float = 0.65
    max_slope_deg: float = 15.0
    max_roll_deg: float = 12.0
    max_pitch_deg: float = 12.0
    footprint_sample_step_m: float = 0.4
    safe_landing_threshold: float = 5.0
    docking_energy_wh: float = 0.8
    undocking_energy_wh: float = 0.2
    drill_staging_enabled: bool = True
    drill_staging_radius_m: float = 4.0
    drill_staging_min_standoff_m: float = 1.5
    drill_staging_sample_step_m: float = 1.0
    drill_final_hop_altitude_m: float = 1.5


@dataclass(frozen=True)
class DockSample:
    index: int
    platform_pose: Pose2D
    branch_pose: Pose2D
    is_target: bool


@dataclass
class MissionEdge:
    kind: EdgeKind
    source: int | None
    target: int
    energy_wh: float
    duration_s: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class TrunkEdge:
    source: int
    target: int
    mode: Literal["drive", "flight"]
    energy_wh: float
    duration_s: float
    path: list[dict[str, float]]


class ContinuousMissionConnectorEvaluator:
    """Physical H/G/F connectors shared by deterministic mission search."""

    def __init__(
        self,
        terrain: TerrainMap,
        home: tuple[float, float],
        target: tuple[float, float],
        vehicle_cfg: dict[str, Any],
        platform_cfg: dict[str, Any],
        config: ContinuousConnectorConfig,
        docking_cfg: dict[str, Any] | None = None,
        battery_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.terrain = terrain
        self.home = (float(home[0]), float(home[1]))
        self.target = (float(target[0]), float(target[1]))
        self.vehicle_cfg = vehicle_cfg
        self.battery_cfg = battery_cfg or _default_battery_config(vehicle_cfg)
        self.platform_cfg = platform_cfg
        self.config = config
        self.rover_offsets = _pose_list(platform_cfg.get("rover_offsets", _default_rover_offsets()))
        self.platform_footprint = _rect(platform_cfg.get("platform_footprint", {"length_m": 2.6, "width_m": 2.6}))
        self.rover_footprint = _rect(platform_cfg.get("rover_footprint", {"length_m": 0.8, "width_m": 0.6}))
        self.payload_scale = _payload_scale(vehicle_cfg, len(self.rover_offsets))
        self.cruise_altitude_m = float(platform_cfg.get("cruise_altitude_m", 5.0))
        self.drill_offset = _drill_attachment_offset(platform_cfg)
        docking_cfg = docking_cfg or {}
        self.slot_names = list(docking_cfg.get("slots", {})) or [
            f"slot_{index + 1}" for index in range(len(self.rover_offsets))
        ]
        configured_assignment = docking_cfg.get("fixed_slot_assignment", {})
        self.fixed_slot_assignment = {
            f"rover_{index + 1}": str(
                configured_assignment.get(
                    f"rover_{index + 1}",
                    self.slot_names[index],
                )
            )
            for index in range(len(self.rover_offsets))
        }
        if (
            len(self.slot_names) != len(self.rover_offsets)
            or set(self.fixed_slot_assignment.values()) != set(self.slot_names)
        ):
            raise ValueError("fixed rover-to-slot assignment must be one-to-one and cover every formation slot")
        self.samples: list[DockSample] = []
        self._home_edges: dict[int, MissionEdge | None] = {}
        self._motion_edges: dict[tuple[int, int], MissionEdge | None] = {}
        self._trunk_adjacency: dict[int, list[TrunkEdge]] | None = None
        self._formation_cache: dict[tuple[int, bool], dict[str, Any]] = {}

    def _home_edge(self, target: DockSample) -> MissionEdge | None:
        if target.index in self._home_edges:
            return self._home_edges[target.index]
        trunk = self._shared_trunk(-1, target.index)
        formation = self._formation_connector(target, outbound=False)
        if trunk is None or formation is None:
            self._home_edges[target.index] = None
            return None
        rover_energy = trunk["energy_wh"] + formation["energy_wh"]
        rover_time = trunk["time_s"] + formation["time_s"]
        drill = self._drill_transfer(
            self.home,
            target.platform_pose[:2],
            rover_time,
            start_wait_power_w=0.0,
        )
        duration = max(rover_time, drill["time_s"])
        edge = MissionEdge(
            "H",
            None,
            target.index,
            float(rover_energy + drill["energy_wh"] + self.config.docking_energy_wh),
            float(duration),
            {
                "shared_trunk": trunk,
                "formation": formation,
                "drill_flight": drill,
                "rover_ready_time_s": float(rover_time),
                "drill_flight_time_s": float(drill["time_s"]),
                "docking_energy_wh": self.config.docking_energy_wh,
            },
        )
        self._home_edges[target.index] = edge
        return edge

    def _best_motion_edge(self, source: DockSample, target: DockSample) -> MissionEdge | None:
        key = (source.index, target.index)
        if key in self._motion_edges:
            return self._motion_edges[key]
        ground = self._ground_edge(source, target)
        reflight = self._reflight_edge(source, target)
        feasible = [edge for edge in (ground, reflight) if edge is not None]
        best = min(feasible, key=lambda edge: edge.energy_wh) if feasible else None
        self._motion_edges[key] = best
        return best

    def _ground_edge(self, source: DockSample, target: DockSample) -> MissionEdge | None:
        path = self._platform_connector(source.platform_pose, target.platform_pose)
        if path is None:
            return None
        rover_paths = rover_paths_from_platform_path(path, self.rover_offsets)
        energy = 0.0
        duration = 0.0
        rover_energy_wh: dict[str, float] = {}
        loaded_rover_mass_kg = (
            float(self.vehicle_cfg["mass_kg"])
            + float(self.vehicle_cfg.get("drill_mass_kg", 3.5)) / len(self.rover_offsets)
        )
        for rover_name, rover_path in rover_paths.items():
            local_energy, local_time = self._ground_path_energy(
                rover_path,
                mass_kg=loaded_rover_mass_kg,
            )
            rover_energy_wh[rover_name] = float(local_energy)
            energy += local_energy
            duration = max(duration, local_time)
        return MissionEdge(
            "G",
            source.index,
            target.index,
            float(energy),
            float(duration),
            {
                "platform_path": [_pose_json(pose) for pose in path],
                "rover_paths": {
                    name: [_pose_json(pose) for pose in poses] for name, poses in rover_paths.items()
                },
                "rover_energy_wh": rover_energy_wh,
                "payload_scale": self.payload_scale,
                "loaded_rover_mass_kg": float(loaded_rover_mass_kg),
            },
        )

    def _reflight_edge(self, source: DockSample, target: DockSample) -> MissionEdge | None:
        outbound = self._formation_connector(source, outbound=True)
        trunk = self._shared_trunk(source.index, target.index)
        inbound = self._formation_connector(target, outbound=False)
        if outbound is None or trunk is None or inbound is None:
            return None
        rover_energy = outbound["energy_wh"] + trunk["energy_wh"] + inbound["energy_wh"]
        rover_time = outbound["time_s"] + trunk["time_s"] + inbound["time_s"]
        drill = self._drill_transfer(
            source.platform_pose[:2],
            target.platform_pose[:2],
            rover_time,
            start_wait_power_w=float(
                self.vehicle_cfg.get("drill_ground_wait_power_w", 10.0)
            ),
        )
        duration = max(rover_time, drill["time_s"])
        energy = (
            rover_energy
            + drill["energy_wh"]
            + self.config.undocking_energy_wh
            + self.config.docking_energy_wh
        )
        return MissionEdge(
            "F",
            source.index,
            target.index,
            float(energy),
            float(duration),
            {
                "outbound_formation": outbound,
                "shared_trunk": trunk,
                "formation": inbound,
                "drill_flight": drill,
                "rover_ready_time_s": float(rover_time),
                "drill_flight_time_s": float(drill["time_s"]),
                "docking_energy_wh": self.config.docking_energy_wh,
                "undocking_energy_wh": self.config.undocking_energy_wh,
            },
        )

    def _refine_reflight_branch(self, edge: MissionEdge) -> MissionEdge:
        """Jointly optimize common B/C points over the whole branch roadmap.

        The mission lattice initially uses a small number of fixed-radius branch
        alternatives per docked platform.  Once an F edge is selected, this pass
        treats every spatially distinct, safe branch location in the constructed
        whole-map roadmap as a possible common split point B and regroup point C.
        A geometric energy proxy screens that global set, then the retained B/C
        Cartesian product is evaluated with the full formation, terrain, mode,
        timing, and energy models.
        """
        if not self.config.branch_refinement_enabled or edge.kind != "F" or edge.source is None:
            return edge
        source = self.samples[edge.source]
        target = self.samples[edge.target]
        spatial_candidates: dict[tuple[int, int], DockSample] = {}
        for sample in self.samples:
            key = (
                int(round(sample.branch_pose[0] * 1000.0)),
                int(round(sample.branch_pose[1] * 1000.0)),
            )
            spatial_candidates.setdefault(key, sample)

        candidate_limit = max(1, int(self.config.branch_global_candidate_limit))
        min_standoff = max(float(self.config.min_formation_branch_standoff_m), 0.5)
        minimum_team_energy_per_m = 4.0 * self._minimum_energy_per_m()

        def screening_score(sample: DockSample) -> tuple[float, int]:
            point = sample.branch_pose
            # Lower-cost candidates lie near the source-to-target ellipse, but
            # the exact ranking below still uses the full physical connectors.
            distance = hypot(
                point[0] - source.platform_pose[0],
                point[1] - source.platform_pose[1],
            ) + hypot(
                target.platform_pose[0] - point[0],
                target.platform_pose[1] - point[1],
            )
            return minimum_team_energy_per_m * distance, sample.index

        global_candidates = sorted(spatial_candidates.values(), key=screening_score)
        b_candidates = [
            sample
            for sample in global_candidates
            if hypot(
                sample.branch_pose[0] - source.platform_pose[0],
                sample.branch_pose[1] - source.platform_pose[1],
            ) >= min_standoff
        ][:candidate_limit]
        c_candidates = [
            sample
            for sample in global_candidates
            if hypot(
                sample.branch_pose[0] - target.platform_pose[0],
                sample.branch_pose[1] - target.platform_pose[1],
            ) >= min_standoff
        ][:candidate_limit]

        outbound_options: list[tuple[DockSample, dict[str, Any]]] = []
        for candidate_index, sample in enumerate(b_candidates):
            temporary = DockSample(
                index=-(1_000_000 + candidate_index),
                platform_pose=source.platform_pose,
                branch_pose=sample.branch_pose,
                is_target=False,
            )
            formation = self._formation_connector(temporary, outbound=True)
            self._formation_cache.pop((temporary.index, True), None)
            if formation is not None:
                outbound_options.append((sample, formation))

        inbound_options: list[tuple[DockSample, dict[str, Any]]] = []
        for candidate_index, sample in enumerate(c_candidates):
            temporary = DockSample(
                index=-(2_000_000 + candidate_index),
                platform_pose=target.platform_pose,
                branch_pose=sample.branch_pose,
                is_target=target.is_target,
            )
            formation = self._formation_connector(temporary, outbound=False)
            self._formation_cache.pop((temporary.index, False), None)
            if formation is not None:
                inbound_options.append((sample, formation))

        best = edge
        evaluated_pairs = 0
        for b_sample, outbound in outbound_options:
            for c_sample, inbound in inbound_options:
                trunk = self._shared_trunk(b_sample.index, c_sample.index)
                if trunk is None:
                    continue
                evaluated_pairs += 1
                candidate = self._compose_reflight_edge(
                    source,
                    target,
                    outbound,
                    trunk,
                    inbound,
                    edge_source=edge.source,
                    refinement={
                        "original_branch_b": _pose_json(source.branch_pose),
                        "original_branch_c": _pose_json(target.branch_pose),
                        "selected_branch_b": _pose_json(b_sample.branch_pose),
                        "selected_branch_c": _pose_json(c_sample.branch_pose),
                    },
                )
                if candidate.energy_wh + 1e-9 < best.energy_wh:
                    best = candidate

        if best is edge:
            return edge
        best.payload["branch_refinement"].update(
            {
                "whole_map_spatial_candidates": len(spatial_candidates),
                "retained_b_candidates": len(outbound_options),
                "retained_c_candidates": len(inbound_options),
                "evaluated_pairs": evaluated_pairs,
                "energy_improvement_wh": float(edge.energy_wh - best.energy_wh),
                "method": "whole-map screened joint B/C search",
            }
        )
        return best

    def _compose_reflight_edge(
        self,
        source: DockSample,
        target: DockSample,
        outbound: dict[str, Any],
        trunk: dict[str, Any],
        inbound: dict[str, Any],
        edge_source: int | None = None,
        refinement: dict[str, Any] | None = None,
    ) -> MissionEdge:
        rover_energy = outbound["energy_wh"] + trunk["energy_wh"] + inbound["energy_wh"]
        rover_time = outbound["time_s"] + trunk["time_s"] + inbound["time_s"]
        drill = self._drill_transfer(
            source.platform_pose[:2],
            target.platform_pose[:2],
            rover_time,
            start_wait_power_w=float(self.vehicle_cfg.get("drill_ground_wait_power_w", 10.0)),
        )
        duration = max(rover_time, drill["time_s"])
        energy = (
            rover_energy
            + drill["energy_wh"]
            + self.config.undocking_energy_wh
            + self.config.docking_energy_wh
        )
        payload = {
            "outbound_formation": outbound,
            "shared_trunk": trunk,
            "formation": inbound,
            "drill_flight": drill,
            "rover_ready_time_s": float(rover_time),
            "drill_flight_time_s": float(drill["time_s"]),
            "docking_energy_wh": self.config.docking_energy_wh,
            "undocking_energy_wh": self.config.undocking_energy_wh,
        }
        if refinement is not None:
            payload["branch_refinement"] = refinement
        return MissionEdge(
            "F",
            source.index if edge_source is None else edge_source,
            target.index,
            float(energy),
            float(duration),
            payload,
        )

    def _formation_connector(self, sample: DockSample, outbound: bool) -> dict[str, Any] | None:
        cache_key = (sample.index, outbound)
        if cache_key in self._formation_cache:
            return self._formation_cache[cache_key]
        slots = platform_to_rover_poses(sample.platform_pose, self.rover_offsets)
        slot_by_name = dict(zip(self.slot_names, slots))
        paths: dict[str, list[dict[str, float]]] = {}
        modes: dict[str, str] = {}
        rover_energy_wh: dict[str, float] = {}
        rover_duration_s: dict[str, float] = {}
        total_energy = 0.0
        total_time = 0.0
        branch = sample.branch_pose
        assigned_slots: dict[str, Pose2D] = {}
        for index in range(1, len(slots) + 1):
            rover_name = f"rover_{index}"
            slot_name = self.fixed_slot_assignment[rover_name]
            slot = slot_by_name[slot_name]
            assigned_slots[slot_name] = slot
            if outbound:
                branch_heading = atan2(branch[1] - slot[1], branch[0] - slot[0])
                start, end = slot, (branch[0], branch[1], branch_heading)
            else:
                branch_heading = atan2(slot[1] - branch[1], slot[0] - branch[0])
                start, end = (branch[0], branch[1], branch_heading), slot
            ground_path = self._formation_ground_connector(start, end)
            ground_option: tuple[float, float, list[Pose2D]] | None = None
            if ground_path is not None:
                energy, duration = self._ground_path_energy(ground_path)
                ground_option = (energy, duration, ground_path)
            flight = self._flight_connector(start[:2], end[:2])
            if ground_option is not None and ground_option[0] <= flight["energy_wh"]:
                energy, duration, selected_path = ground_option
                path_json = [_pose_json(pose) for pose in selected_path]
                mode = "drive"
            else:
                energy = float(flight["energy_wh"])
                duration = float(flight["time_s"])
                path_json = list(flight["path"])
                mode = "flight"
            paths[rover_name] = path_json
            modes[rover_name] = mode
            rover_energy_wh[rover_name] = float(energy)
            rover_duration_s[rover_name] = float(duration)
            total_energy += energy
        separation = max(0.0, float(self.config.formation_separation_time_s))
        rover_names = list(paths)
        if outbound:
            longest = max(rover_duration_s.values(), default=0.0)
            start_offsets = {
                rover_name: longest - rover_duration_s[rover_name] + index * separation
                for index, rover_name in enumerate(rover_names)
            }
        else:
            start_offsets = {
                rover_name: index * separation
                for index, rover_name in enumerate(rover_names)
            }
        rover_timing = {
            rover_name: {
                "start_offset_s": float(start_offsets[rover_name]),
                "duration_s": rover_duration_s[rover_name],
                "end_offset_s": float(start_offsets[rover_name] + rover_duration_s[rover_name]),
            }
            for rover_name in rover_names
        }
        total_time = max(
            (timing["end_offset_s"] for timing in rover_timing.values()),
            default=0.0,
        )
        result = {
            "branch_pose": _pose_json(branch),
            "slot_poses": {name: _pose_json(slot_by_name[name]) for name in self.slot_names},
            "fixed_slot_assignment": dict(self.fixed_slot_assignment),
            "rover_paths": paths,
            "rover_modes": modes,
            "rover_energy_wh": rover_energy_wh,
            "rover_timing": rover_timing,
            "energy_wh": float(total_energy),
            "time_s": float(total_time),
            "collision_schedule": "parallel traversal with small branch-entry time offsets",
            "direction": "outbound" if outbound else "inbound",
        }
        self._formation_cache[cache_key] = result
        return result

    def _shared_trunk(self, source_index: int, target_index: int) -> dict[str, Any] | None:
        """Plan a multi-segment shared rover route without forcing docking."""
        adjacency = self._shared_trunk_adjacency()
        best = {source_index: 0.0}
        parent: dict[int, tuple[int, TrunkEdge]] = {}
        queue: list[tuple[float, int]] = [(0.0, source_index)]
        while queue:
            cost, current = heappop(queue)
            if cost > best.get(current, float("inf")) + 1e-9:
                continue
            if current == target_index:
                break
            for edge in adjacency.get(current, []):
                next_cost = cost + edge.energy_wh
                if next_cost + 1e-9 >= best.get(edge.target, float("inf")):
                    continue
                best[edge.target] = next_cost
                parent[edge.target] = (current, edge)
                heappush(queue, (next_cost, edge.target))

        candidates: list[tuple[float, float, list[TrunkEdge]]] = []
        if target_index in best:
            edges: list[TrunkEdge] = []
            current = target_index
            while current != source_index:
                previous, edge = parent[current]
                edges.append(edge)
                current = previous
            edges.reverse()
            candidates.append(
                (
                    float(sum(edge.energy_wh for edge in edges)),
                    float(sum(edge.duration_s for edge in edges)),
                    edges,
                )
            )

        start_pose = self._trunk_pose(source_index)
        end_pose = self._trunk_pose(target_index)
        direct_ground_start = start_pose
        if source_index == -1:
            direct_ground_start = (
                start_pose[0],
                start_pose[1],
                atan2(end_pose[1] - start_pose[1], end_pose[0] - start_pose[0]),
            )
        direct_ground = self._rover_ground_connector(direct_ground_start, end_pose)
        if direct_ground is not None:
            ground_energy, ground_duration = self._ground_path_energy(direct_ground)
            candidates.append(
                (
                    float(ground_energy),
                    float(ground_duration),
                    [
                        TrunkEdge(
                            source_index,
                            target_index,
                            "drive",
                            float(ground_energy),
                            float(ground_duration),
                            [_pose_json(pose) for pose in direct_ground],
                        )
                    ],
                )
            )
        direct_air = self._flight_connector(start_pose[:2], end_pose[:2])
        candidates.append(
            (
                float(direct_air["energy_wh"]),
                float(direct_air["time_s"]),
                [
                    TrunkEdge(
                        source_index,
                        target_index,
                        "flight",
                        float(direct_air["energy_wh"]),
                        float(direct_air["time_s"]),
                        list(direct_air["path"]),
                    )
                ],
            )
        )
        single_energy, duration, selected = min(candidates, key=lambda item: item[0])
        modes = {edge.mode for edge in selected}
        mode = next(iter(modes)) if len(modes) == 1 else "mixed"
        flattened: list[dict[str, float]] = []
        segments = []
        for edge in selected:
            _extend_json_path(flattened, edge.path)
            segments.append(
                {
                    "mode": edge.mode,
                    "energy_wh": edge.energy_wh,
                    "time_s": edge.duration_s,
                    "path": edge.path,
                }
            )
        return {
            "mode": mode,
            "path": flattened,
            "segments": segments,
            "energy_wh": float(4.0 * single_energy),
            "rover_energy_wh": {
                f"rover_{index}": float(single_energy) for index in range(1, 5)
            },
            "time_s": float(duration),
            "roadmap_nodes": len(self.samples) + 1,
        }

    def _shared_trunk_adjacency(self) -> dict[int, list[TrunkEdge]]:
        if self._trunk_adjacency is not None:
            return self._trunk_adjacency
        node_indices = [-1, *range(len(self.samples))]
        positions = {index: self._trunk_pose(index) for index in node_indices}
        adjacency: dict[int, list[TrunkEdge]] = {index: [] for index in node_indices}
        max_radius = max(
            self.config.trunk_connection_radius_m,
            self.config.trunk_air_connection_radius_m,
        )
        for source in node_indices:
            ranked = sorted(
                (
                    (hypot(positions[target][0] - positions[source][0], positions[target][1] - positions[source][1]), target)
                    for target in node_indices
                    if target != source and target != -1
                ),
                key=lambda item: item[0],
            )
            for distance, target in ranked[: self.config.trunk_connection_k]:
                if distance > max_radius:
                    continue
                start = positions[source]
                end = positions[target]
                if source == -1:
                    start = (start[0], start[1], atan2(end[1] - start[1], end[0] - start[0]))
                if distance <= self.config.trunk_connection_radius_m:
                    ground_path = self._rover_ground_connector(start, end)
                    if ground_path is not None:
                        energy, duration = self._ground_path_energy(ground_path)
                        adjacency[source].append(
                            TrunkEdge(
                                source,
                                target,
                                "drive",
                                float(energy),
                                float(duration),
                                [_pose_json(pose) for pose in ground_path],
                            )
                        )
                if distance <= self.config.trunk_air_connection_radius_m:
                    flight = self._flight_connector(start[:2], end[:2])
                    adjacency[source].append(
                        TrunkEdge(
                            source,
                            target,
                            "flight",
                            float(flight["energy_wh"]),
                            float(flight["time_s"]),
                            list(flight["path"]),
                        )
                    )
        self._trunk_adjacency = adjacency
        return adjacency

    def _trunk_pose(self, index: int) -> Pose2D:
        if index == -1:
            return (self.home[0], self.home[1], 0.0)
        return self.samples[index].branch_pose

    def _platform_connector(self, start: Pose2D, end: Pose2D) -> list[Pose2D] | None:
        options = []
        for reverse in (False, True):
            path = self._hermite_path(start, end, reverse)
            if path is None or not all(self._dock_feasible(pose) for pose in path):
                continue
            options.append(path)
        return min(options, key=_path_length) if options else None

    def _rover_ground_connector(self, start: Pose2D, end: Pose2D) -> list[Pose2D] | None:
        options = []
        for reverse in (False, True):
            for tangent_ratio in (0.45, 0.85, 1.25, 1.8):
                path = self._hermite_path(start, end, reverse, tangent_ratio=tangent_ratio)
                if path is None:
                    continue
                if all(self._representative_pose_feasible(pose, require_safe_landing=False) for pose in path):
                    options.append(path)
        return min(options, key=_path_length) if options else None

    def _formation_ground_connector(self, start: Pose2D, end: Pose2D) -> list[Pose2D] | None:
        direct = self._rover_ground_connector(start, end)
        if direct is not None:
            return direct
        distance = hypot(end[0] - start[0], end[1] - start[1])
        if distance <= 1e-8:
            return None
        mx = 0.5 * (start[0] + end[0])
        my = 0.5 * (start[1] + end[1])
        nx = -(end[1] - start[1]) / distance
        ny = (end[0] - start[0]) / distance
        travel_heading = atan2(end[1] - start[1], end[0] - start[0])
        candidates: list[list[Pose2D]] = []
        for lateral_ratio in (-0.6, 0.0, 0.6):
            px = mx + nx * distance * lateral_ratio
            py = my + ny * distance * lateral_ratio
            for heading in (end[2], travel_heading, wrap_angle(travel_heading + pi)):
                middle = (float(px), float(py), float(heading))
                if not self._representative_pose_feasible(middle, require_safe_landing=False):
                    continue
                first = self._rover_ground_connector(start, middle)
                second = self._rover_ground_connector(middle, end)
                if first is None or second is None:
                    continue
                combined = [*first, *second[1:]]
                candidates.append(combined)
        return min(candidates, key=_path_length) if candidates else None

    def _hermite_path(
        self,
        start: Pose2D,
        end: Pose2D,
        reverse: bool,
        tangent_ratio: float = 0.85,
    ) -> list[Pose2D] | None:
        distance = hypot(end[0] - start[0], end[1] - start[1])
        if distance <= 1e-8:
            return [start, end] if abs(wrap_angle(end[2] - start[2])) < 1e-4 else None
        sign = -1.0 if reverse else 1.0
        tangent_scale = max(distance * tangent_ratio, 0.5)
        p0 = np.array(start[:2], dtype=float)
        p1 = np.array(end[:2], dtype=float)
        m0 = sign * tangent_scale * np.array([cos(start[2]), sin(start[2])])
        m1 = sign * tangent_scale * np.array([cos(end[2]), sin(end[2])])
        count = max(3, int(ceil(distance / self.config.connector_step_m)) + 1)
        path: list[Pose2D] = []
        for u in np.linspace(0.0, 1.0, count):
            h00 = 2 * u**3 - 3 * u**2 + 1
            h10 = u**3 - 2 * u**2 + u
            h01 = -2 * u**3 + 3 * u**2
            h11 = u**3 - u**2
            point = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
            dh00 = 6 * u**2 - 6 * u
            dh10 = 3 * u**2 - 4 * u + 1
            dh01 = -6 * u**2 + 6 * u
            dh11 = 3 * u**2 - 2 * u
            tangent = dh00 * p0 + dh10 * m0 + dh01 * p1 + dh11 * m1
            velocity_heading = atan2(float(tangent[1]), float(tangent[0]))
            body_heading = wrap_angle(velocity_heading + (pi if reverse else 0.0))
            path.append((float(point[0]), float(point[1]), body_heading))
        path[0] = start
        path[-1] = end
        for a, b in zip(path[:-1], path[1:]):
            ds = hypot(b[0] - a[0], b[1] - a[1])
            if ds <= 1e-8 or abs(wrap_angle(b[2] - a[2])) / ds > self.config.max_curvature_1pm + 1e-6:
                return None
        return path

    def _flight_connector(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        cruise_altitude_m: float | None = None,
        start_z_m: float = 0.0,
        end_z_m: float = 0.0,
        mass_kg: float | None = None,
    ) -> dict[str, Any]:
        distance = hypot(end[0] - start[0], end[1] - start[1])
        cruise_time = distance / float(self.vehicle_cfg["air_max_speed_mps"])
        takeoff_time = float(self.vehicle_cfg["takeoff_time_s"])
        landing_time = float(self.vehicle_cfg["landing_time_s"])
        flight_mass = float(
            self.vehicle_cfg["mass_kg"] if mass_kg is None else mass_kg
        )
        takeoff_power = mass_scaled_flight_power(
            float(self.vehicle_cfg["takeoff_power_w"]),
            flight_mass,
            self.vehicle_cfg,
        )
        cruise_power = mass_scaled_flight_power(
            float(self.vehicle_cfg["forward_flight_power_w"]),
            flight_mass,
            self.vehicle_cfg,
        )
        landing_power = mass_scaled_flight_power(
            float(self.vehicle_cfg["landing_power_w"]),
            flight_mass,
            self.vehicle_cfg,
        )
        cruise_altitude = (
            self.cruise_altitude_m
            if cruise_altitude_m is None
            else float(cruise_altitude_m)
        )
        energy = (
            takeoff_power * takeoff_time
            + cruise_power * cruise_time
            + landing_power * landing_time
        ) / 3600.0
        return {
            "path": [
                {"x": float(start[0]), "y": float(start[1]), "z": float(start_z_m)},
                {"x": float(start[0]), "y": float(start[1]), "z": cruise_altitude},
                {"x": float(end[0]), "y": float(end[1]), "z": cruise_altitude},
                {"x": float(end[0]), "y": float(end[1]), "z": float(end_z_m)},
            ],
            "energy_wh": float(energy),
            "time_s": float(takeoff_time + cruise_time + landing_time),
            "mass_kg": flight_mass,
            "phase_power_w": {
                "takeoff": float(takeoff_power),
                "cruise": float(cruise_power),
                "landing": float(landing_power),
            },
        }

    def _drill_transfer(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        rover_ready_time_s: float,
        start_wait_power_w: float = 0.0,
    ) -> dict[str, Any]:
        direct = self._flight_connector(
            start,
            end,
            end_z_m=self.drill_offset[2],
            mass_kg=float(self.vehicle_cfg.get("drill_mass_kg", 3.5)),
        )
        departure_offset_s = max(0.0, rover_ready_time_s - direct["time_s"])
        start_wait_energy_wh = start_wait_power_w * departure_offset_s / 3600.0
        direct["energy_wh"] = float(direct["energy_wh"] + start_wait_energy_wh)
        direct.update(
            {
                "strategy": "delayed_direct",
                "departure_offset_s": float(departure_offset_s),
                "staging_wait_time_s": 0.0,
                "ground_wait_time_s": float(departure_offset_s),
                "ground_wait_power_w": float(start_wait_power_w),
                "ground_wait_energy_wh": float(start_wait_energy_wh),
                "final_hop_altitude_m": None,
                "alternatives_evaluated": 1,
            }
        )
        candidates = [direct]
        if self.config.drill_staging_enabled:
            for staging in self._drill_staging_points(end):
                first_leg = self._flight_connector(
                    start,
                    staging,
                    mass_kg=float(self.vehicle_cfg.get("drill_mass_kg", 3.5)),
                )
                final_hop = self._flight_connector(
                    staging,
                    end,
                    cruise_altitude_m=self.config.drill_final_hop_altitude_m,
                    end_z_m=self.drill_offset[2],
                    mass_kg=float(self.vehicle_cfg.get("drill_mass_kg", 3.5)),
                )
                airborne_time = first_leg["time_s"] + final_hop["time_s"]
                wait_time = max(0.0, rover_ready_time_s - airborne_time)
                wait_power_w = float(
                    self.vehicle_cfg.get("drill_ground_wait_power_w", 10.0)
                )
                wait_energy = wait_power_w * wait_time / 3600.0
                path = list(first_leg["path"])
                _extend_json_path(path, final_hop["path"])
                candidates.append(
                    {
                        "path": path,
                        "energy_wh": float(first_leg["energy_wh"] + final_hop["energy_wh"] + wait_energy),
                        "time_s": float(airborne_time + wait_time),
                        "strategy": "staged_ground_wait",
                        "departure_offset_s": 0.0,
                        "staging_pose": {"x": staging[0], "y": staging[1], "z": 0.0},
                        "staging_arrival_time_s": float(first_leg["time_s"]),
                        "staging_wait_time_s": float(wait_time),
                        "ground_wait_time_s": float(wait_time),
                        "ground_wait_power_w": float(wait_power_w),
                        "ground_wait_energy_wh": float(wait_energy),
                        "final_hop_time_s": float(final_hop["time_s"]),
                        "final_hop_altitude_m": float(self.config.drill_final_hop_altitude_m),
                    }
                )
        selected = min(candidates, key=lambda item: (item["energy_wh"], item["time_s"]))
        selected["alternatives_evaluated"] = len(candidates)
        return selected

    def _drill_staging_points(self, target: tuple[float, float]) -> list[tuple[float, float]]:
        radius = self.config.drill_staging_radius_m
        standoff = self.config.drill_staging_min_standoff_m
        step = max(self.config.drill_staging_sample_step_m, 0.25)
        points: list[tuple[float, float]] = []
        for x in np.arange(target[0] - radius, target[0] + radius + 0.5 * step, step):
            for y in np.arange(target[1] - radius, target[1] + radius + 0.5 * step, step):
                distance = hypot(float(x) - target[0], float(y) - target[1])
                if distance < standoff or distance > radius:
                    continue
                if self.terrain.continuous_safe_landing(
                    float(x),
                    float(y),
                    self.config.safe_landing_threshold,
                ):
                    points.append((float(x), float(y)))
        return sorted(points, key=lambda point: hypot(point[0] - target[0], point[1] - target[1]))[:24]

    def _ground_path_energy(
        self,
        path: list[Pose2D],
        mass_kg: float | None = None,
    ) -> tuple[float, float]:
        energy = 0.0
        duration = 0.0
        for a, b in zip(path[:-1], path[1:]):
            distance = hypot(b[0] - a[0], b[1] - a[1])
            if distance <= 1e-10:
                continue
            mx = 0.5 * (a[0] + b[0])
            my = 0.5 * (a[1] + b[1])
            dz = self.terrain.continuous_value("elevation", b[0], b[1]) - self.terrain.continuous_value("elevation", a[0], a[1])
            speed, power = ground_speed_and_power(
                _ContinuousTerrainAdapter(self.terrain),
                mx,
                my,
                self.vehicle_cfg,
                dz / distance,
                mass_kg=mass_kg,
            )
            local_time = distance / speed
            duration += local_time
            energy += power * local_time / 3600.0
        return float(energy), float(duration)

    def _dock_feasible(self, pose: Pose2D) -> bool:
        if not self.terrain.continuous_safe_landing(pose[0], pose[1], self.config.safe_landing_threshold):
            return False
        if self._footprint_collision(pose, self.platform_footprint, check_slope=False):
            return False
        rover_poses = platform_to_rover_poses(pose, self.rover_offsets)
        if any(self._footprint_collision(rover_pose, self.rover_footprint, check_slope=True) for rover_pose in rover_poses):
            return False
        points = []
        for x, y, _ in rover_poses:
            z = self.terrain.continuous_value("elevation", x, y)
            if not np.isfinite(z):
                return False
            points.append((x, y, z))
        matrix = np.array([[x, y, 1.0] for x, y, _ in points], dtype=float)
        heights = np.array([z for _, _, z in points], dtype=float)
        slope_x, slope_y, _ = np.linalg.lstsq(matrix, heights, rcond=None)[0]
        pitch = np.degrees(np.arctan(float(slope_x)))
        roll = np.degrees(np.arctan(float(slope_y)))
        return abs(roll) <= self.config.max_roll_deg and abs(pitch) <= self.config.max_pitch_deg

    def _representative_pose_feasible(self, pose: Pose2D, require_safe_landing: bool) -> bool:
        if require_safe_landing and not self.terrain.continuous_safe_landing(
            pose[0], pose[1], self.config.safe_landing_threshold
        ):
            return False
        return not self._footprint_collision(pose, self.rover_footprint, check_slope=True)

    def _footprint_collision(self, pose: Pose2D, footprint: RectFootprint, check_slope: bool) -> bool:
        for x, y in rectangle_sample_points(pose, footprint, self.config.footprint_sample_step_m):
            if not self.terrain.contains(x, y) or self.terrain.continuous_occupied("obstacle", x, y):
                return True
            if check_slope and not np.isfinite(self.terrain.continuous_ground_cost(x, y, self.config.max_slope_deg)):
                return True
        return False

    def _platform_center_for_target(self, theta: float) -> tuple[float, float]:
        dx, dy = self.drill_offset[:2]
        return (
            self.target[0] - cos(theta) * dx + sin(theta) * dy,
            self.target[1] - sin(theta) * dx - cos(theta) * dy,
        )

    def _in_informed_set(self, x: float, y: float, incumbent: float) -> bool:
        optimistic = self._minimum_energy_per_m() * (
            hypot(x - self.home[0], y - self.home[1]) + hypot(self.target[0] - x, self.target[1] - y)
        ) + self.config.docking_energy_wh
        return optimistic < incumbent

    def _minimum_energy_per_m(self) -> float:
        air = mass_scaled_flight_power(
            float(self.vehicle_cfg["forward_flight_power_w"]),
            float(self.vehicle_cfg.get("mass_kg", 3.0)),
            self.vehicle_cfg,
        ) / (
            float(self.vehicle_cfg["air_max_speed_mps"]) * 3600.0
        )
        ground = minimum_ground_energy_per_m(self.vehicle_cfg, robot_count=4)
        return float(min(air, ground))

    def _global_lower_bound(self) -> float:
        return float(
            hypot(self.target[0] - self.home[0], self.target[1] - self.home[1])
            * self._minimum_energy_per_m()
            + self.config.docking_energy_wh
        )

    def _heuristic(self, sample: DockSample) -> float:
        return float(
            hypot(self.target[0] - sample.platform_pose[0], self.target[1] - sample.platform_pose[1])
            * self._minimum_energy_per_m()
        )

    def _result(
        self,
        edges: list[MissionEdge],
        objective: float,
        elapsed: float,
        history: list[dict[str, Any]],
        target_samples: int,
    ) -> dict[str, Any]:
        mission_events = []
        branch_points = []
        shared_trunks = []
        formations = []
        drill_flights = []
        platform_paths = []
        mission_timeline = []
        timing = []
        current_time = 0.0
        formation_stage_index = 0
        breakdown = {
            "shared_rover_wh": 0.0,
            "formation_wh": 0.0,
            "drill_flight_wh": 0.0,
            "drill_ground_wait_wh": 0.0,
            "platform_wh": 0.0,
            "switch_wh": 0.0,
        }
        agent_energy_wh = {
            **{rover_name: 0.0 for rover_name in self.fixed_slot_assignment},
            "drill": 0.0,
        }
        for event_index, edge in enumerate(edges, start=1):
            target_sample = self.samples[edge.target]
            event = {
                "index": event_index,
                "type": edge.kind,
                "from": "Home" if edge.source is None else _pose_json(self.samples[edge.source].platform_pose),
                "to": _pose_json(target_sample.platform_pose),
                "energy_wh": edge.energy_wh,
                "duration_s": edge.duration_s,
            }
            mission_events.append(event)
            if edge.kind in {"H", "F"}:
                payload = edge.payload
                if edge.kind == "H":
                    formation_stage_index += 1
                    stage = _alphabetic_stage_name(formation_stage_index)
                    inbound = {
                        **payload["formation"],
                        "event_index": event_index,
                        "phase": "assemble",
                        "stage": stage,
                        "label": f"Assemble {stage}",
                    }
                    branch_points.append(
                        {
                            **inbound["branch_pose"],
                            "event_index": event_index,
                            "phase": "assemble",
                            "stage": stage,
                            "label": f"Branch {stage}",
                        }
                    )
                    shared_trunks.append(
                        {
                            **payload["shared_trunk"],
                            "event_index": event_index,
                            "label": f"H{event_index} Home to Branch {stage}",
                        }
                    )
                    formations.append(inbound)
                    mission_timeline.extend(
                        [
                            {
                                "event_index": event_index,
                                "event_type": "H",
                                "phase": "shared_trunk",
                                "label": f"H{event_index}: Home to Branch {stage}",
                            },
                            {
                                "event_index": event_index,
                                "event_type": "H",
                                "phase": "assemble",
                                "stage": stage,
                                "label": f"Assemble {stage}: Branch {stage} to fixed slots",
                            },
                        ]
                    )
                else:
                    formation_stage_index += 1
                    outbound_stage = _alphabetic_stage_name(formation_stage_index)
                    outbound = {
                        **payload["outbound_formation"],
                        "event_index": event_index,
                        "phase": "disassemble",
                        "stage": outbound_stage,
                        "label": f"Disassemble {outbound_stage}",
                    }
                    branch_points.append(
                        {
                            **outbound["branch_pose"],
                            "event_index": event_index,
                            "phase": "disassemble",
                            "stage": outbound_stage,
                            "label": f"Branch {outbound_stage}",
                        }
                    )
                    formations.append(outbound)

                    formation_stage_index += 1
                    inbound_stage = _alphabetic_stage_name(formation_stage_index)
                    inbound = {
                        **payload["formation"],
                        "event_index": event_index,
                        "phase": "assemble",
                        "stage": inbound_stage,
                        "label": f"Assemble {inbound_stage}",
                    }
                    branch_points.append(
                        {
                            **inbound["branch_pose"],
                            "event_index": event_index,
                            "phase": "assemble",
                            "stage": inbound_stage,
                            "label": f"Branch {inbound_stage}",
                        }
                    )
                    shared_trunks.append(
                        {
                            **payload["shared_trunk"],
                            "event_index": event_index,
                            "label": f"F{event_index} Branch {outbound_stage} to Branch {inbound_stage}",
                        }
                    )
                    formations.append(inbound)
                    mission_timeline.extend(
                        [
                            {
                                "event_index": event_index,
                                "event_type": "F",
                                "phase": "disassemble",
                                "stage": outbound_stage,
                                "label": f"Disassemble {outbound_stage}: fixed slots to Branch {outbound_stage}",
                            },
                            {
                                "event_index": event_index,
                                "event_type": "F",
                                "phase": "shared_trunk",
                                "label": f"F{event_index}: Branch {outbound_stage} to Branch {inbound_stage}",
                            },
                            {
                                "event_index": event_index,
                                "event_type": "F",
                                "phase": "assemble",
                                "stage": inbound_stage,
                                "label": f"Assemble {inbound_stage}: Branch {inbound_stage} to fixed slots",
                            },
                        ]
                    )
                drill_flights.append(payload["drill_flight"])
                rover_ready = float(payload["rover_ready_time_s"])
                drill_duration = float(payload["drill_flight_time_s"])
                drill = payload["drill_flight"]
                drill_departure = current_time + float(drill.get("departure_offset_s", 0.0))
                touchdown_time = current_time + max(rover_ready, drill_duration)
                timing.append(
                    {
                        "event_index": event_index,
                        "start_time_s": current_time,
                        "rover_ready_time_s": current_time + rover_ready,
                        "drill_strategy": drill.get("strategy", "delayed_direct"),
                        "drill_departure_time_s": drill_departure,
                        "drill_arrival_time_s": touchdown_time,
                        "touchdown_time_s": touchdown_time,
                        "drill_staging_pose": drill.get("staging_pose"),
                        "drill_staging_wait_time_s": float(drill.get("staging_wait_time_s", 0.0)),
                        "drill_ground_wait_time_s": float(drill.get("ground_wait_time_s", 0.0)),
                        "drill_ground_wait_power_w": float(
                            drill.get(
                                "ground_wait_power_w",
                                self.vehicle_cfg.get("drill_ground_wait_power_w", 10.0),
                            )
                        ),
                        "drill_ground_wait_energy_wh": float(drill.get("ground_wait_energy_wh", 0.0)),
                        "drill_final_hop_altitude_m": drill.get("final_hop_altitude_m"),
                        "drill_hover_wait_time_s": 0.0,
                        "drill_hover_wait_energy_wh": 0.0,
                    }
                )
                breakdown["shared_rover_wh"] += float(payload["shared_trunk"]["energy_wh"])
                breakdown["formation_wh"] += float(payload["formation"]["energy_wh"])
                if edge.kind == "F":
                    breakdown["formation_wh"] += float(payload["outbound_formation"]["energy_wh"])
                ground_wait_energy = float(
                    payload["drill_flight"].get("ground_wait_energy_wh", 0.0)
                )
                breakdown["drill_flight_wh"] += float(
                    payload["drill_flight"]["energy_wh"]
                ) - ground_wait_energy
                breakdown["drill_ground_wait_wh"] += ground_wait_energy
                switch_energy = float(payload.get("docking_energy_wh", 0.0)) + float(
                    payload.get("undocking_energy_wh", 0.0)
                )
                breakdown["switch_wh"] += switch_energy
                for rover_name in self.fixed_slot_assignment:
                    agent_energy_wh[rover_name] += float(
                        payload["shared_trunk"]["rover_energy_wh"][rover_name]
                    )
                    agent_energy_wh[rover_name] += float(
                        payload["formation"]["rover_energy_wh"][rover_name]
                    )
                    if edge.kind == "F":
                        agent_energy_wh[rover_name] += float(
                            payload["outbound_formation"]["rover_energy_wh"][rover_name]
                        )
                    # Mechanical dock/undock actuation is assigned equally to
                    # the four rover batteries in the accounting model.
                    agent_energy_wh[rover_name] += switch_energy / len(
                        self.fixed_slot_assignment
                    )
                agent_energy_wh["drill"] += float(payload["drill_flight"]["energy_wh"])
            else:
                platform_paths.append(edge.payload["platform_path"])
                mission_timeline.append(
                    {
                        "event_index": event_index,
                        "event_type": "G",
                        "phase": "docked_drive",
                        "label": f"G{event_index}: docked platform drive",
                    }
                )
                timing.append(
                    {
                        "event_index": event_index,
                        "start_time_s": current_time,
                        "transport_end_time_s": current_time + edge.duration_s,
                    }
                )
                breakdown["platform_wh"] += edge.energy_wh
                for rover_name in self.fixed_slot_assignment:
                    agent_energy_wh[rover_name] += float(
                        edge.payload["rover_energy_wh"][rover_name]
                    )
            current_time += edge.duration_s

        goal_sample = self.samples[edges[-1].target]
        goal_slots = dict(
            zip(
                self.slot_names,
                platform_to_rover_poses(goal_sample.platform_pose, self.rover_offsets),
            )
        )
        rover_states = {
            rover_name: _pose_json(goal_slots[slot_name])
            for rover_name, slot_name in self.fixed_slot_assignment.items()
        }
        lower_bound = self._global_lower_bound()
        battery_report = _battery_report(self.battery_cfg, agent_energy_wh)
        return {
            "planner": "Mission Connector Evaluation",
            "scenario": "",
            "success": True,
            "reason": None,
            "objective_energy_wh": float(objective),
            "solve_time_s": float(elapsed),
            "mission_events": mission_events,
            "mission_timeline": mission_timeline,
            "continuous_paths": {
                "shared_rover_trunks": shared_trunks,
                "formations": formations,
                "drill_flights": drill_flights,
                "platform_paths": platform_paths,
            },
            "formation_branch_points": branch_points,
            "timing_schedule": timing,
            "terminal_docked_state": {
                "mode": "docked-ground",
                "platform_pose": _pose_json(goal_sample.platform_pose),
                "drill_state": {"x": self.target[0], "y": self.target[1], "mode": "on_platform"},
                "rover_states": rover_states,
                "all_rovers_in_distinct_slots": True,
                "fixed_slot_assignment": dict(self.fixed_slot_assignment),
                "drill_attached_to_platform": True,
            },
            "goal": {"x": self.target[0], "y": self.target[1], "theta": goal_sample.platform_pose[2], "heading_free": True},
            "samples": {
                "dock_configurations": len(self.samples),
                "target_manifold_samples": target_samples,
            },
            "convergence_history": history,
            "energy_breakdown": {key: float(value) for key, value in breakdown.items()},
            "agent_energy_wh": {
                key: float(value) for key, value in agent_energy_wh.items()
            },
            "battery_report": battery_report,
            "lower_bound_wh": lower_bound,
            "reported_gap": "the mission-search layer supplies the lattice certificate",
            "optimality_claim": {
                "finite_run": "connector-evaluated feasible route; the mission-search layer supplies the certificate",
                "scope": "shared rover trunk with continuous formation branches on the interpolated 2.5D map",
                "unrestricted_four_rover_problem": "feasible upper bound only",
            },
            "model_assumptions": {
                "target_heading": "free",
                "shared_trunk_collision_avoidance": "time offsets",
                "formation_collision_avoidance": "parallel paths with small branch-entry time offsets",
                "formation_separation_time_s": float(self.config.formation_separation_time_s),
                "rover_slot_policy": "fixed identity-to-slot assignment at every formation",
                "rover_mass_kg": float(self.vehicle_cfg.get("mass_kg", 3.0)),
                "drill_mass_kg": float(self.vehicle_cfg.get("drill_mass_kg", 3.5)),
                "docked_payload_per_rover_kg": float(
                    self.vehicle_cfg.get("drill_mass_kg", 3.5)
                )
                / len(self.rover_offsets),
                "docked_loaded_rover_mass_kg": float(
                    self.vehicle_cfg.get("mass_kg", 3.0)
                )
                + float(self.vehicle_cfg.get("drill_mass_kg", 3.5))
                / len(self.rover_offsets),
                "max_ground_slope_deg": float(self.config.max_slope_deg),
                "max_generated_terrain_slope_deg": 30.0,
                "ground_energy_model": "quasi-static rolling resistance, gravity, speed, and drivetrain efficiency",
                "flight_energy_model": (
                    "reference takeoff, cruise, and landing powers with the aerodynamic "
                    "share scaled as (mass/reference_mass)^flight_power_mass_exponent"
                ),
                "flight_reference_mass_kg": float(
                    self.vehicle_cfg.get(
                        "flight_reference_mass_kg",
                        self.vehicle_cfg.get("mass_kg", 3.0),
                    )
                ),
                "flight_power_mass_exponent": float(
                    self.vehicle_cfg.get("flight_power_mass_exponent", 1.5)
                ),
                "flight_fixed_power_fraction": float(
                    self.vehicle_cfg.get("flight_fixed_power_fraction", 0.0)
                ),
                "drill_staging_wait": "optional safe terrain landing followed by a 1.5 m final hop",
                "drill_ground_wait_power_w": float(
                    self.vehicle_cfg.get("drill_ground_wait_power_w", 10.0)
                ),
                "home_wait_power_w": 0.0,
                "deadline": None,
                "terrain_interpolation": "bilinear scalar layers; conservative closed occupancy cells",
            },
        }

    def _failure(self, elapsed: float, history: list[dict[str, Any]], target_samples: int) -> dict[str, Any]:
        return {
            "planner": "Mission Connector Evaluation",
            "scenario": "",
            "success": False,
            "reason": "no feasible target-docked connector route found",
            "objective_energy_wh": float("inf"),
            "solve_time_s": float(elapsed),
            "mission_events": [],
            "continuous_paths": {"shared_rover_trunks": [], "formations": [], "drill_flights": [], "platform_paths": []},
            "formation_branch_points": [],
            "timing_schedule": [],
            "terminal_docked_state": {},
            "samples": {"dock_configurations": len(self.samples), "target_manifold_samples": target_samples},
            "convergence_history": history,
            "optimality_claim": {"finite_run": "no feasible upper bound found"},
        }


class _ContinuousTerrainAdapter:
    """Expose continuous roughness to the existing power model."""

    def __init__(self, terrain: TerrainMap) -> None:
        self.terrain = terrain

    def value(self, layer: str, x: float, y: float) -> float:
        return self.terrain.continuous_value(layer, x, y)


def _default_battery_config(vehicle_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build a conservative fallback when no explicit battery block is supplied."""
    nominal_energy_wh = float(vehicle_cfg.get("battery_energy_wh", 48.84))
    reserve_ratio = float(vehicle_cfg.get("battery_reserve_ratio", 0.20))
    spec = {
        "nominal_energy_wh": nominal_energy_wh,
        "nominal_voltage_v": 22.2,
        "capacity_ah": nominal_energy_wh / 22.2,
    }
    return {"reserve_ratio": reserve_ratio, "rover": dict(spec), "drill": dict(spec)}


def _battery_report(
    battery_cfg: dict[str, Any],
    agent_energy_wh: dict[str, float],
) -> dict[str, Any]:
    reserve_ratio = float(battery_cfg.get("reserve_ratio", 0.20))
    if not 0.0 <= reserve_ratio < 1.0:
        raise ValueError("battery reserve_ratio must be in [0, 1)")
    agents: dict[str, dict[str, Any]] = {}
    for agent_name, used_energy_wh in agent_energy_wh.items():
        kind = "drill" if agent_name == "drill" else "rover"
        spec = battery_cfg.get(kind, {})
        voltage_v = float(spec.get("nominal_voltage_v", 22.2))
        capacity_ah = float(spec.get("capacity_ah", 0.0))
        nominal_energy_wh = float(
            spec.get("nominal_energy_wh", voltage_v * capacity_ah)
        )
        usable_energy_wh = nominal_energy_wh * (1.0 - reserve_ratio)
        usable_margin_wh = usable_energy_wh - float(used_energy_wh)
        agents[agent_name] = {
            "battery_type": kind,
            "series_cells": int(spec.get("series_cells", 6)),
            "nominal_voltage_v": voltage_v,
            "capacity_ah": capacity_ah,
            "nominal_energy_wh": nominal_energy_wh,
            "reserve_ratio": reserve_ratio,
            "usable_energy_wh": usable_energy_wh,
            "used_energy_wh": float(used_energy_wh),
            "usable_margin_wh": usable_margin_wh,
            "remaining_nominal_energy_wh": nominal_energy_wh - float(used_energy_wh),
            "reserve_satisfied": usable_margin_wh >= -1e-9,
        }
    return {
        "enforced_during_search": False,
        "postcheck_feasible": all(
            item["reserve_satisfied"] for item in agents.values()
        ),
        "switch_energy_policy": "dock/undock energy split equally across four rover batteries",
        "agents": agents,
    }


def _planner_config(raw: dict[str, Any], config: dict[str, Any]) -> ContinuousConnectorConfig:
    aerial_cfg = config.get("aerial", {})
    values = {
        "max_slope_deg": config.get("vehicle", {}).get("max_slope_deg", ContinuousConnectorConfig.max_slope_deg),
        "safe_landing_threshold": aerial_cfg.get("safe_landing_threshold", ContinuousConnectorConfig.safe_landing_threshold),
        "drill_staging_enabled": aerial_cfg.get("drill_staging_enabled", ContinuousConnectorConfig.drill_staging_enabled),
        "drill_staging_radius_m": aerial_cfg.get("drill_staging_radius_m", ContinuousConnectorConfig.drill_staging_radius_m),
        "drill_staging_min_standoff_m": aerial_cfg.get(
            "drill_staging_min_standoff_m", ContinuousConnectorConfig.drill_staging_min_standoff_m
        ),
        "drill_staging_sample_step_m": aerial_cfg.get(
            "drill_staging_sample_step_m", ContinuousConnectorConfig.drill_staging_sample_step_m
        ),
        "drill_final_hop_altitude_m": aerial_cfg.get(
            "drill_final_hop_altitude_m", ContinuousConnectorConfig.drill_final_hop_altitude_m
        ),
        **raw,
    }
    allowed = ContinuousConnectorConfig.__dataclass_fields__.keys()
    return ContinuousConnectorConfig(**{key: value for key, value in values.items() if key in allowed})


def _payload_scale(vehicle_cfg: dict[str, Any], rover_count: int) -> float:
    rover_mass = float(vehicle_cfg.get("mass_kg", 3.0))
    drill_mass = float(vehicle_cfg.get("drill_mass_kg", 3.5))
    return float((rover_mass + drill_mass / max(rover_count, 1)) / max(rover_mass, 1e-9))


def _alphabetic_stage_name(index: int) -> str:
    """Return spreadsheet-style stage names: A..Z, AA..AZ, ..."""
    if index < 1:
        raise ValueError("stage index must be positive")
    result = ""
    value = index
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


def _drill_attachment_offset(platform_cfg: dict[str, Any]) -> tuple[float, float, float, float]:
    raw = platform_cfg.get("drill_attachment_offset", {})
    return (
        float(raw.get("x", 0.0)),
        float(raw.get("y", 0.0)),
        float(raw.get("z", platform_cfg.get("drill_touchdown_height_m", 0.45))),
        float(raw.get("theta", 0.0)),
    )


def _path_length(path: list[Pose2D]) -> float:
    return float(sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:])))


def _extend_json_path(target: list[dict[str, float]], source: list[dict[str, float]]) -> None:
    for point in source:
        if target and all(abs(float(target[-1].get(key, 0.0)) - float(point.get(key, 0.0))) <= 1e-9 for key in ("x", "y", "z")):
            continue
        target.append(dict(point))


def _pose_json(pose: Pose2D) -> dict[str, float]:
    return {"x": float(pose[0]), "y": float(pose[1]), "theta": float(pose[2])}


def _pose_list(raw: list[dict[str, float]] | list[Pose2D]) -> list[Pose2D]:
    return [
        (float(item["x"]), float(item["y"]), float(item.get("theta", 0.0)))
        if isinstance(item, dict)
        else (float(item[0]), float(item[1]), float(item[2]))
        for item in raw
    ]


def _rect(raw: dict[str, float]) -> RectFootprint:
    return RectFootprint(float(raw["length_m"]), float(raw["width_m"]))


def _default_rover_offsets() -> list[dict[str, float]]:
    return [
        {"x": 0.9, "y": 0.9, "theta": 0.0},
        {"x": 0.9, "y": -0.9, "theta": 0.0},
        {"x": -0.9, "y": 0.9, "theta": pi},
        {"x": -0.9, "y": -0.9, "theta": pi},
    ]
