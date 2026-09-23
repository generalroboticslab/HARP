from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import atan2, hypot
from pathlib import Path
from time import perf_counter, strftime
from typing import Any, Iterable, Literal

import numpy as np

from .config import load_config
from .pcd_25d import load_planner_terrain
from .terrain import TerrainMap, make_default_terrain
from .team_model import (
    TeamEdge,
    TeamState,
    TeamEnergyModel,
    TeamConfig,
)
from .visualize_dp import export_dp_html3d


Mode = Literal["assembled_drive", "disassembled_flight"]
DEFAULT_CASE_1_MAP = Path(__file__).resolve().parents[1] / "maps" / "hybrid_challenge.html"


@dataclass(frozen=True)
class InitialCondition:
    x: float
    y: float
    heading_rad: float | None = None


@dataclass(frozen=True)
class DPState:
    node: int
    heading_bin: int
    mode: Mode
    used_ground: bool
    used_air: bool
    air_distance_ticks: int = 0


@dataclass(frozen=True)
class TwoLayerState:
    node: int
    mode: Mode


class TwoLayerAirGroundDP:
    """Exact shortest-path DP on the requested ``(x, y, ground/air)`` lattice."""

    _MOVES = TeamEnergyModel._MOVES

    def __init__(
        self,
        terrain: TerrainMap,
        target: tuple[float, float],
        initial_conditions: list[InitialCondition],
        config: dict[str, Any],
    ) -> None:
        if not initial_conditions:
            raise ValueError("at least one ground initial condition is required")
        self.terrain = terrain
        self.target = (float(target[0]), float(target[1]))
        self.initial_conditions = initial_conditions
        self.config = config
        self.team_config = TeamConfig(
            grid_resolution_m=terrain.resolution,
            heading_bins=8,
            connector_step_m=0.35,
            footprint_sample_step_m=0.4,
            max_slope_deg=float(config["vehicle"].get("max_slope_deg", 15.0)),
            max_roll_deg=12.0,
            max_pitch_deg=12.0,
            safe_landing_threshold=float(config["aerial"].get("safe_landing_threshold", 5.0)),
            cruise_altitude_m=float(config["aerial"].get("cruise_altitude_m", 5.0)),
            minimum_flight_leg_m=1.0,
            maximum_flight_leg_m=None,
            require_ground_and_air=False,
        )
        first = initial_conditions[0]
        self.energy_model = TeamEnergyModel(
            terrain=terrain,
            home=(first.x, first.y),
            target=self.target,
            vehicle_cfg=config["vehicle"],
            platform_cfg=config["platform"],
            battery_cfg=config["batteries"],
            docking_energy_wh=float(config["dp"].get("docking_energy_wh", 0.8)),
            undocking_energy_wh=float(config["dp"].get("undocking_energy_wh", 0.2)),
            config=self.team_config,
        )
        self.positions = [
            (float(ix * terrain.resolution), float(iy * terrain.resolution))
            for iy in range(terrain.height)
            for ix in range(terrain.width)
        ]
        self.target_node = self._grid_node(self.target)
        self.start_nodes = [self._grid_node((item.x, item.y)) for item in initial_conditions]
        self.ground_out = {node: [] for node in range(len(self.positions))}
        self.air_out = {node: [] for node in range(len(self.positions))}
        self.ground_edges: dict[tuple[int, int], TeamEdge] = {}
        self.air_edges: dict[tuple[int, int], TeamEdge] = {}
        self.switch_safe: dict[int, bool] = {}
        self.switch_heading: dict[int, int] = {}
        self._build_graph()
        self.ground_in = MultiStartAirGroundDP._reverse_adjacency(self.ground_out)
        self.air_in = MultiStartAirGroundDP._reverse_adjacency(self.air_out)

    def solve(self) -> dict[str, Any]:
        started = perf_counter()
        goal = TwoLayerState(self.target_node, "assembled_drive")
        value: dict[TwoLayerState, float] = {goal: 0.0}
        policy: dict[TwoLayerState, tuple[TwoLayerState, TeamEdge]] = {}
        queue: list[tuple[float, int, TwoLayerState]] = [(0.0, 0, goal)]
        tie = count(1)
        expanded = 0
        peak_open = 1
        while queue:
            cost, _, state = heappop(queue)
            if cost > value.get(state, float("inf")) + 1e-10:
                continue
            expanded += 1
            for predecessor, edge in self._predecessors(state):
                candidate = cost + edge.energy_wh
                if candidate + 1e-10 >= value.get(predecessor, float("inf")):
                    continue
                value[predecessor] = candidate
                policy[predecessor] = (state, edge)
                heappush(queue, (candidate, next(tie), predecessor))
                peak_open = max(peak_open, len(queue))
        elapsed = perf_counter() - started
        results = [self._make_result(index, value, policy, elapsed) for index in range(len(self.initial_conditions))]
        ground = self._layer_grid(value, "assembled_drive")
        air = self._layer_grid(value, "disassembled_flight")
        return {
            "planner": "Two-Layer Air-Ground Dynamic Programming",
            "algorithm": "backward Bellman shortest-path recursion solved by reverse Dijkstra",
            "scenario": "hybrid_challenge",
            "map_source": str(DEFAULT_CASE_1_MAP),
            "state_space": "V(x, y, z), z in {ground, air}",
            "boundary_conditions": {
                "initial_mode": "ground only",
                "goal_mode": "ground only",
                "air_role": "intermediate states only",
            },
            "target": {"x": self.target[0], "y": self.target[1], "mode": "assembled_drive"},
            "initial_condition_count": len(results),
            "all_succeeded": all(item["success"] for item in results),
            "shared_dp_statistics": {
                "solve_time_s": elapsed,
                "expanded_states": expanded,
                "stored_value_states": len(value),
                "peak_open_size": peak_open,
                "full_reachable_value_function": True,
                "single_value_function_reused": True,
            },
            "hybrid_cost_to_go": {
                "state": "V(x, y, mode)",
                "ground": ground,
                "air": air,
                "continuous_visualization": "piecewise interpolation of the 1 m DP lattice; blocked ground states remain infeasible",
            },
            "assembled_initial_cost_to_go": ground,
            "results": results,
        }

    def _predecessors(self, state: TwoLayerState) -> Iterable[tuple[TwoLayerState, TeamEdge]]:
        if state.mode == "assembled_drive":
            for source in self.ground_in[state.node]:
                yield TwoLayerState(source, "assembled_drive"), self.ground_edges[(source, state.node)]
            if self._switch_safe(state.node):
                yield TwoLayerState(state.node, "disassembled_flight"), self.energy_model._landing_edge()
        else:
            for source in self.air_in[state.node]:
                yield TwoLayerState(source, "disassembled_flight"), self.air_edges[(source, state.node)]
            if self._switch_safe(state.node):
                yield TwoLayerState(state.node, "assembled_drive"), self.energy_model._takeoff_edge()

    def _build_graph(self) -> None:
        for iy in range(self.terrain.height):
            for ix in range(self.terrain.width):
                source = iy * self.terrain.width + ix
                for dx, dy in self._MOVES:
                    nx, ny = ix + dx, iy + dy
                    if not (0 <= nx < self.terrain.width and 0 <= ny < self.terrain.height):
                        continue
                    target = ny * self.terrain.width + nx
                    distance = hypot(dx, dy) * self.terrain.resolution
                    self.air_out[source].append(target)
                    self.air_edges[(source, target)] = self.energy_model._flight_edge(distance)
                    ground_edge = self._ground_edge(source, target)
                    if ground_edge is not None:
                        self.ground_out[source].append(target)
                        self.ground_edges[(source, target)] = ground_edge

    def _ground_edge(self, source: int, target: int) -> TeamEdge | None:
        start, end = self.positions[source], self.positions[target]
        heading = atan2(end[1] - start[1], end[0] - start[0])
        for x, y in self.terrain.line_samples(start, end, self.team_config.connector_step_m):
            if not self.energy_model._ground_pose_feasible_world(x, y, heading):
                return None
        each_energy, duration = self.energy_model._ground_segment_energy(start, end)
        agents = (each_energy, each_energy, each_energy, each_energy, 0.0)
        return TeamEdge("drive", float(sum(agents)), duration, agents)

    def _switch_safe(self, node: int) -> bool:
        if node not in self.switch_safe:
            x, y = self.positions[node]
            self.switch_safe[node] = False
            for heading_bin in range(self.team_config.heading_bins):
                heading = self.energy_model._heading(heading_bin)
                if (
                    self.energy_model._ground_pose_feasible_world(x, y, heading)
                    and self.terrain.continuous_safe_landing(
                        x, y, self.team_config.safe_landing_threshold
                    )
                ):
                    self.switch_safe[node] = True
                    self.switch_heading[node] = heading_bin
                    break
        return self.switch_safe[node]

    def _layer_grid(self, value: dict[TwoLayerState, float], mode: Mode) -> dict[str, Any]:
        costs: list[list[float | None]] = []
        for iy in range(self.terrain.height):
            row = []
            for ix in range(self.terrain.width):
                cost = value.get(TwoLayerState(iy * self.terrain.width + ix, mode), float("inf"))
                row.append(float(cost) if np.isfinite(cost) else None)
            costs.append(row)
        return {
            "mode": mode,
            "meaning": "optimal energy to the ground-only goal",
            "units": "Wh",
            "x_resolution_m": self.terrain.resolution,
            "y_resolution_m": self.terrain.resolution,
            "cost_wh": costs,
        }

    def _make_result(
        self,
        index: int,
        value: dict[TwoLayerState, float],
        policy: dict[TwoLayerState, tuple[TwoLayerState, TeamEdge]],
        elapsed: float,
    ) -> dict[str, Any]:
        initial = self.initial_conditions[index]
        start = TwoLayerState(self.start_nodes[index], "assembled_drive")
        if start not in value:
            return {
                "planner": "Two-Layer Air-Ground Dynamic Programming",
                "success": False,
                "reason": "ground initial state has no feasible policy",
                "initial_condition_index": index + 1,
                "home": {"x": initial.x, "y": initial.y, "mode": "assembled_drive"},
            }
        states = [start]
        edges: list[TeamEdge] = []
        cursor = start
        while cursor != TwoLayerState(self.target_node, "assembled_drive"):
            next_state, edge = policy[cursor]
            states.append(next_state)
            edges.append(edge)
            cursor = next_state
        helper = TeamEnergyModel(
            self.terrain,
            (initial.x, initial.y),
            self.target,
            self.config["vehicle"],
            self.config["platform"],
            self.config["batteries"],
            float(self.config["dp"].get("docking_energy_wh", 0.8)),
            float(self.config["dp"].get("undocking_energy_wh", 0.2)),
            self.team_config,
        )
        heading_bin = helper._heading_bin(
            initial.heading_rad
            if initial.heading_rad is not None
            else atan2(self.target[1] - initial.y, self.target[0] - initial.x)
        )
        used_ground = False
        used_air = False
        team_states = []
        for state_index, state in enumerate(states):
            if state_index:
                previous_edge = edges[state_index - 1]
                if previous_edge.action in {"drive", "fly"}:
                    previous_point = self.positions[states[state_index - 1].node]
                    point = self.positions[state.node]
                    heading_bin = helper._heading_bin(
                        atan2(point[1] - previous_point[1], point[0] - previous_point[0])
                    )
                used_ground = used_ground or previous_edge.action == "drive"
                used_air = used_air or previous_edge.action == "fly"
            ix = state.node % self.terrain.width
            iy = state.node // self.terrain.width
            team_states.append(
                TeamState(ix, iy, heading_bin, state.mode, used_ground, used_air, 0)
            )
        result = helper._result(team_states, edges, elapsed, 0, len(value), 0, "two_layer_dp")
        result.update(
            {
                "planner": "Two-Layer Air-Ground Dynamic Programming",
                "scenario": "hybrid_challenge",
                "initial_condition_index": index + 1,
                "home": {"x": initial.x, "y": initial.y, "mode": "assembled_drive"},
                "target": {"x": self.target[0], "y": self.target[1], "mode": "assembled_drive"},
                "dp_cost_to_go_wh": value[start],
            }
        )
        return result

    def _grid_node(self, point: tuple[float, float]) -> int:
        ix = int(round(point[0] / self.terrain.resolution))
        iy = int(round(point[1] / self.terrain.resolution))
        snapped = (ix * self.terrain.resolution, iy * self.terrain.resolution)
        if hypot(snapped[0] - point[0], snapped[1] - point[1]) > 1e-8:
            raise ValueError("two-layer DP initial and target positions must lie on the map grid")
        if not (0 <= ix < self.terrain.width and 0 <= iy < self.terrain.height):
            raise ValueError("initial or target position lies outside the map")
        return iy * self.terrain.width + ix


class MultiStartAirGroundDP:
    """Backward shortest-path dynamic programming for many initial poses.

    One reverse Dijkstra solve evaluates the Bellman cost-to-go function from a
    fixed assembled target. The team energy model supplies physical edge costs;
    special start nodes and the target connect to the 2-D lattice through the
    model's start/target sentinels.
    """

    _MOVES = TeamEnergyModel._MOVES

    def __init__(
        self,
        terrain: TerrainMap,
        target: tuple[float, float],
        initial_conditions: list[InitialCondition],
        config: dict[str, Any],
        team_config: TeamConfig,
        *,
        ground_only: bool = False,
        takeoff_y_range_m: tuple[float, float] | None = None,
    ) -> None:
        if not initial_conditions:
            raise ValueError("at least one initial condition is required")
        if ground_only and team_config.require_ground_and_air:
            raise ValueError("ground-only DP cannot require both ground and air")
        self.terrain = terrain
        self.target = (float(target[0]), float(target[1]))
        self.initial_conditions = initial_conditions
        self.config = config
        self.team_config = team_config
        self.ground_only = bool(ground_only)
        if (
            takeoff_y_range_m is not None
            and takeoff_y_range_m[0] > takeoff_y_range_m[1]
        ):
            raise ValueError("takeoff y range must satisfy minimum <= maximum")
        self.takeoff_y_range_m = takeoff_y_range_m
        self.resolution = team_config.grid_resolution_m
        self.minimum_air_ticks = int(np.ceil(team_config.minimum_flight_leg_m * 10.0))
        if team_config.maximum_flight_leg_m is not None:
            raise ValueError("the multi-start DP currently requires an unlimited maximum flight leg")

        first = initial_conditions[0]
        self.energy_model = TeamEnergyModel(
            terrain=terrain,
            home=(first.x, first.y),
            target=self.target,
            vehicle_cfg=config["vehicle"],
            platform_cfg=config["platform"],
            battery_cfg=config["batteries"],
            docking_energy_wh=float(config["dp"].get("docking_energy_wh", 0.8)),
            undocking_energy_wh=float(config["dp"].get("undocking_energy_wh", 0.2)),
            config=team_config,
        )
        self.positions: list[tuple[float, float]] = []
        self.regular_by_index: dict[tuple[int, int], int] = {}
        self.regular_index_by_node: dict[int, tuple[int, int]] = {}
        self._make_regular_nodes()
        self.target_node = self._add_special_position(self.target)
        self.start_nodes = [self._add_special_position((item.x, item.y)) for item in initial_conditions]
        self.ground_out: dict[int, list[int]] = {node: [] for node in range(len(self.positions))}
        self.air_out: dict[int, list[int]] = {node: [] for node in range(len(self.positions))}
        self._build_spatial_graph()
        self.ground_in = self._reverse_adjacency(self.ground_out)
        self.air_in = self._reverse_adjacency(self.air_out)
        self.ground_edges: dict[tuple[int, int], tuple[int, TeamEdge]] = {}
        self.air_edges: dict[tuple[int, int], tuple[int, int, TeamEdge]] = {}
        self.safe_switch: dict[tuple[int, int], bool] = {}
        self._precompute_edges()

    def solve(self) -> dict[str, Any]:
        started = perf_counter()
        initial_states = [self._initial_state(i) for i in range(len(self.initial_conditions))]
        unresolved = set(initial_states)
        value: dict[DPState, float] = {}
        policy: dict[DPState, tuple[DPState, TeamEdge]] = {}
        queue: list[tuple[float, int, DPState]] = []
        tie = count()
        if self.ground_only:
            required_goal_flags = ((False, False), (True, False))
        elif self.team_config.require_ground_and_air:
            required_goal_flags = ((True, True),)
        else:
            required_goal_flags = ((False, False), (True, False), (False, True), (True, True))
        for heading_bin in range(self.team_config.heading_bins):
            for used_ground, used_air in required_goal_flags:
                goal = DPState(
                    self.target_node,
                    heading_bin,
                    "assembled_drive",
                    used_ground,
                    used_air,
                    0,
                )
                value[goal] = 0.0
                heappush(queue, (0.0, next(tie), goal))

        expanded = 0
        peak_open = len(queue)
        while queue:
            cost, _, state = heappop(queue)
            if cost > value.get(state, float("inf")) + 1e-10:
                continue
            expanded += 1
            unresolved.discard(state)
            for predecessor, edge in self._predecessors(state):
                next_cost = cost + edge.energy_wh
                if next_cost + 1e-10 >= value.get(predecessor, float("inf")):
                    continue
                value[predecessor] = next_cost
                policy[predecessor] = (state, edge)
                heappush(queue, (next_cost, next(tie), predecessor))
                peak_open = max(peak_open, len(queue))

        elapsed = perf_counter() - started
        ground_cost_to_go = self._cost_to_go_grid(
            value,
            mode="assembled_drive",
            air_distance_ticks=0,
            meaning=(
                "assembled ground state that has not yet used ground or air; "
                "minimum over heading"
            ),
        )
        air_cost_to_go = self._cost_to_go_grid(
            value,
            mode="disassembled_flight",
            air_distance_ticks=0,
            meaning=(
                "airborne state at the start of a flight leg that has not yet "
                "used ground or air; minimum over heading"
            ),
        )
        results = []
        for index, initial_state in enumerate(initial_states):
            results.append(
                self._make_result(
                    index,
                    initial_state,
                    value,
                    policy,
                    elapsed,
                    expanded,
                    peak_open,
                )
            )
        return {
            "planner": "Multi-Start Air-Ground Dynamic Programming",
            "algorithm": "backward Bellman shortest-path recursion solved by reverse Dijkstra",
            "scenario": "hybrid_challenge",
            "map_source": str(DEFAULT_CASE_1_MAP),
            "target": {"x": self.target[0], "y": self.target[1], "mode": "assembled_drive"},
            "initial_condition_count": len(results),
            "all_succeeded": all(item["success"] for item in results),
            "shared_dp_statistics": {
                "solve_time_s": elapsed,
                "expanded_states": expanded,
                "stored_value_states": len(value),
                "peak_open_size": peak_open,
                "single_value_function_reused": True,
                "full_reachable_value_function": True,
                "unresolved_initial_conditions": len(unresolved),
            },
            "lattice": {
                "grid_resolution_m": self.resolution,
                "heading_bins": self.team_config.heading_bins,
                "minimum_flight_leg_m": self.team_config.minimum_flight_leg_m,
                "require_ground_and_air": self.team_config.require_ground_and_air,
                "ground_only": self.ground_only,
                "takeoff_y_range_m": (
                    list(self.takeoff_y_range_m)
                    if self.takeoff_y_range_m is not None
                    else None
                ),
                "max_ground_slope_deg": self.team_config.max_slope_deg,
            },
            "hybrid_cost_to_go": {
                "state": "V(x, y, mode) visualization slices",
                "ground": ground_cost_to_go,
                "air": air_cost_to_go,
                "internal_state_note": (
                    "exact planning also indexes heading, used-mode flags, and "
                    "flight-leg distance; each displayed layer fixes those values "
                    "as stated and minimizes over heading"
                ),
            },
            "assembled_initial_cost_to_go": ground_cost_to_go,
            "results": results,
        }

    def _cost_to_go_grid(
        self,
        value: dict[DPState, float],
        *,
        mode: Mode,
        air_distance_ticks: int,
        meaning: str,
    ) -> dict[str, Any]:
        costs: list[list[float | None]] = [
            [None for _ in range(self.energy_model.grid_width)]
            for _ in range(self.energy_model.grid_height)
        ]
        best_headings: list[list[int | None]] = [
            [None for _ in range(self.energy_model.grid_width)]
            for _ in range(self.energy_model.grid_height)
        ]
        for node, (ix, iy) in self.regular_index_by_node.items():
            options = [
                (
                    value.get(
                        DPState(
                            node,
                            heading,
                            mode,
                            False,
                            False,
                            air_distance_ticks,
                        ),
                        float("inf"),
                    ),
                    heading,
                )
                for heading in range(self.team_config.heading_bins)
            ]
            cost, heading = min(options)
            if np.isfinite(cost):
                costs[iy][ix] = float(cost)
                best_headings[iy][ix] = int(heading)
        return {
            "mode": mode,
            "meaning": meaning,
            "units": "Wh",
            "x_resolution_m": self.resolution,
            "y_resolution_m": self.resolution,
            "cost_wh": costs,
            "best_heading_bin": best_headings,
        }

    def _predecessors(self, state: DPState) -> Iterable[tuple[DPState, TeamEdge]]:
        if state.mode == "assembled_drive":
            if (
                not self.ground_only
                and self._switch_safe(state.node, state.heading_bin)
                and state.node != self.target_node
            ):
                yield (
                    DPState(
                        state.node,
                        state.heading_bin,
                        "disassembled_flight",
                        state.used_ground,
                        state.used_air,
                        self.minimum_air_ticks,
                    ),
                    self.energy_model._landing_edge(),
                )
            if state.used_ground:
                for source in self.ground_in.get(state.node, []):
                    edge_heading, edge = self.ground_edges.get((source, state.node), (-1, None))
                    if edge is None or edge_heading != state.heading_bin:
                        continue
                    for previous_heading in range(self.team_config.heading_bins):
                        for previous_used_ground in (False, True):
                            yield (
                                DPState(
                                    source,
                                    previous_heading,
                                    "assembled_drive",
                                    previous_used_ground,
                                    state.used_air,
                                    0,
                                ),
                                edge,
                            )
            return

        if self.ground_only:
            return
        if (
            state.air_distance_ticks == 0
            and self._takeoff_allowed(state.node)
            and self._switch_safe(state.node, state.heading_bin)
        ):
            yield (
                DPState(
                    state.node,
                    state.heading_bin,
                    "assembled_drive",
                    state.used_ground,
                    state.used_air,
                    0,
                ),
                self.energy_model._takeoff_edge(),
            )
        if not state.used_air:
            return
        for source in self.air_in.get(state.node, []):
            edge_heading, increment, edge = self.air_edges[(source, state.node)]
            if edge_heading != state.heading_bin:
                continue
            if state.air_distance_ticks < self.minimum_air_ticks:
                previous_ticks = [state.air_distance_ticks - increment]
            else:
                previous_ticks = range(
                    max(0, self.minimum_air_ticks - increment),
                    self.minimum_air_ticks + 1,
                )
            for ticks in previous_ticks:
                if ticks < 0 or min(self.minimum_air_ticks, ticks + increment) != state.air_distance_ticks:
                    continue
                for previous_heading in range(self.team_config.heading_bins):
                    for previous_used_air in (False, True):
                        yield (
                            DPState(
                                source,
                                previous_heading,
                                "disassembled_flight",
                                state.used_ground,
                                previous_used_air,
                                ticks,
                            ),
                            edge,
                        )

    def _takeoff_allowed(self, node: int) -> bool:
        if self.takeoff_y_range_m is None:
            return True
        y = self.positions[node][1]
        return self.takeoff_y_range_m[0] <= y <= self.takeoff_y_range_m[1]

    def _make_result(
        self,
        index: int,
        initial_state: DPState,
        value: dict[DPState, float],
        policy: dict[DPState, tuple[DPState, TeamEdge]],
        elapsed: float,
        expanded: int,
        peak_open: int,
    ) -> dict[str, Any]:
        initial = self.initial_conditions[index]
        helper = TeamEnergyModel(
            terrain=self.terrain,
            home=(initial.x, initial.y),
            target=self.target,
            vehicle_cfg=self.config["vehicle"],
            platform_cfg=self.config["platform"],
            battery_cfg=self.config["batteries"],
            docking_energy_wh=float(self.config["dp"].get("docking_energy_wh", 0.8)),
            undocking_energy_wh=float(self.config["dp"].get("undocking_energy_wh", 0.2)),
            config=self.team_config,
        )
        if initial_state not in value:
            return {
                "planner": "Multi-Start Air-Ground Dynamic Programming",
                "scenario": "hybrid_challenge",
                "success": False,
                "reason": "initial condition has no feasible hybrid policy",
                "initial_condition_index": index + 1,
                "home": {"x": initial.x, "y": initial.y, "mode": "assembled_drive"},
                "target": {"x": self.target[0], "y": self.target[1], "mode": "assembled_drive"},
            }

        dp_states = [initial_state]
        edges: list[TeamEdge] = []
        cursor = initial_state
        seen = {cursor}
        while cursor.node != self.target_node or cursor.mode != "assembled_drive":
            if cursor not in policy:
                raise RuntimeError("incomplete DP policy during reconstruction")
            next_state, edge = policy[cursor]
            if next_state in seen:
                raise RuntimeError("cycle in DP policy")
            edges.append(edge)
            dp_states.append(next_state)
            seen.add(next_state)
            cursor = next_state

        team_states = [self._team_state(state, helper, index) for state in dp_states]
        result = helper._result(
            team_states,
            edges,
            elapsed,
            expanded,
            len(value),
            peak_open,
            "dp_initial_state_settled",
        )
        result.update(
            {
                "planner": "Multi-Start Air-Ground Dynamic Programming",
                "planner_role": "shared cost-to-go policy for multiple initial conditions",
                "scenario": "hybrid_challenge",
                "initial_condition_index": index + 1,
                "home": {"x": initial.x, "y": initial.y, "mode": "assembled_drive"},
                "target": {"x": self.target[0], "y": self.target[1], "mode": "assembled_drive"},
                "dp_cost_to_go_wh": value[initial_state],
                "optimality_claim": {
                    "finite_run": "minimum summed five-agent energy on the configured finite hybrid DP lattice",
                    "continuous_space": "no claim outside the configured grid, headings, and synchronized-team abstraction",
                },
            }
        )
        result["search_statistics"] = {
            "algorithm": "reverse Dijkstra dynamic programming",
            "shared_solve_across_initial_conditions": True,
            "expanded_states": expanded,
            "stored_value_states": len(value),
            "peak_open_size": peak_open,
            "grid_resolution_m": self.resolution,
            "heading_bins": self.team_config.heading_bins,
            "minimum_flight_leg_m": self.team_config.minimum_flight_leg_m,
            "optimal_on_finite_hybrid_lattice": True,
        }
        return result

    def _team_state(
        self,
        state: DPState,
        helper: TeamEnergyModel,
        initial_index: int,
    ) -> TeamState:
        if state.node == self.start_nodes[initial_index] and state.node not in self.regular_index_by_node:
            ix, iy = helper._start_sentinel
        elif state.node == self.target_node and state.node not in self.regular_index_by_node:
            ix, iy = helper._target_sentinel
        else:
            ix, iy = self.regular_index_by_node[state.node]
        return TeamState(
            ix,
            iy,
            state.heading_bin,
            state.mode,
            state.used_ground,
            state.used_air,
            state.air_distance_ticks,
        )

    def _initial_state(self, index: int) -> DPState:
        item = self.initial_conditions[index]
        heading = item.heading_rad
        if heading is None:
            heading = atan2(self.target[1] - item.y, self.target[0] - item.x)
        heading_bin = self.energy_model._heading_bin(heading)
        if not self.energy_model._ground_pose_feasible_world(item.x, item.y, self.energy_model._heading(heading_bin)):
            raise ValueError(f"initial condition {index + 1} is not feasible for the assembled footprint")
        return DPState(self.start_nodes[index], heading_bin, "assembled_drive", False, False, 0)

    def _make_regular_nodes(self) -> None:
        for iy in range(self.energy_model.grid_height):
            for ix in range(self.energy_model.grid_width):
                point = (ix * self.resolution, iy * self.resolution)
                if not self.terrain.contains(*point):
                    continue
                node = len(self.positions)
                self.positions.append(point)
                self.regular_by_index[(ix, iy)] = node
                self.regular_index_by_node[node] = (ix, iy)

    def _add_special_position(self, point: tuple[float, float]) -> int:
        for node, existing in enumerate(self.positions):
            if hypot(existing[0] - point[0], existing[1] - point[1]) <= 1e-9:
                return node
        self.positions.append((float(point[0]), float(point[1])))
        return len(self.positions) - 1

    def _build_spatial_graph(self) -> None:
        for (ix, iy), source in self.regular_by_index.items():
            for dx, dy in self._MOVES:
                target = self.regular_by_index.get((ix + dx, iy + dy))
                if target is None:
                    continue
                self.ground_out[source].append(target)
                self.air_out[source].append(target)
            if source != self.target_node and hypot(
                self.positions[source][0] - self.target[0],
                self.positions[source][1] - self.target[1],
            ) <= 1.75 * self.resolution:
                self.ground_out[source].append(self.target_node)

        for start_node in self.start_nodes:
            if start_node in self.regular_index_by_node:
                continue
            start = self.positions[start_node]
            for target in self.regular_index_by_node:
                distance = hypot(
                    self.positions[target][0] - start[0],
                    self.positions[target][1] - start[1],
                )
                if 1e-9 < distance <= 1.75 * self.resolution:
                    self.ground_out[start_node].append(target)
                    self.air_out[start_node].append(target)

    @staticmethod
    def _reverse_adjacency(outgoing: dict[int, list[int]]) -> dict[int, list[int]]:
        incoming = {node: [] for node in outgoing}
        for source, targets in outgoing.items():
            for target in targets:
                incoming[target].append(source)
        return incoming

    def _precompute_edges(self) -> None:
        for source, targets in self.ground_out.items():
            for target in targets:
                edge = self._ground_edge(source, target)
                if edge is not None:
                    self.ground_edges[(source, target)] = edge
        if not self.ground_only:
            for source, targets in self.air_out.items():
                for target in targets:
                    start = self.positions[source]
                    end = self.positions[target]
                    if not self._air_edge_feasible(start, end):
                        continue
                    distance = hypot(end[0] - start[0], end[1] - start[1])
                    heading = self.energy_model._heading_bin(atan2(end[1] - start[1], end[0] - start[0]))
                    increment = int(round(distance * 10.0))
                    self.air_edges[(source, target)] = (
                        heading,
                        increment,
                        self.energy_model._flight_edge(distance),
                    )
        # Remove infeasible ground arcs from both adjacency directions.
        for source, targets in self.ground_out.items():
            self.ground_out[source] = [target for target in targets if (source, target) in self.ground_edges]
        self.ground_in = self._reverse_adjacency(self.ground_out)

        # Air adjacency must be filtered too; otherwise reverse DP can use a
        # spatial arc for which no collision-checked air edge was created.
        for source, targets in self.air_out.items():
            self.air_out[source] = [target for target in targets if (source, target) in self.air_edges]
        self.air_in = self._reverse_adjacency(self.air_out)

    def _air_edge_feasible(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> bool:
        """Reject fixed-AGL flight where mapped objects consume the clearance."""
        if not hasattr(self.terrain, "surface_height_m"):
            return True
        available_height = (
            self.team_config.cruise_altitude_m
            - self.team_config.air_obstacle_clearance_m
        )
        for x, y in self.terrain.line_samples(
            start,
            end,
            self.team_config.connector_step_m,
        ):
            if self.terrain.continuous_value("surface_height_m", x, y) >= available_height:
                return False
        return True

    def _ground_edge(self, source: int, target: int) -> tuple[int, TeamEdge] | None:
        start = self.positions[source]
        end = self.positions[target]
        heading_bin = self.energy_model._heading_bin(atan2(end[1] - start[1], end[0] - start[0]))
        heading = self.energy_model._heading(heading_bin)
        for x, y in self.terrain.line_samples(start, end, self.team_config.connector_step_m):
            if not self.energy_model._ground_pose_feasible_world(x, y, heading):
                return None
        each_energy, duration = self.energy_model._ground_segment_energy(start, end)
        agents = (each_energy, each_energy, each_energy, each_energy, 0.0)
        return heading_bin, TeamEdge("drive", float(sum(agents)), duration, agents)

    def _switch_safe(self, node: int, heading_bin: int) -> bool:
        key = (node, heading_bin)
        if key not in self.safe_switch:
            x, y = self.positions[node]
            self.safe_switch[key] = (
                self.energy_model._ground_pose_feasible_world(
                    x,
                    y,
                    self.energy_model._heading(heading_bin),
                )
                and self.terrain.continuous_safe_landing(
                    x,
                    y,
                    self.team_config.safe_landing_threshold,
                )
            )
        return self.safe_switch[key]


def load_case_1_terrain(html_path: str | Path) -> TerrainMap:
    """Load the case-1 source terrain and verify it against the HTML scene."""
    html_path = Path(html_path)
    html = html_path.read_text(encoding="utf-8")
    scene_text = html.split("const scene = ", 1)[1].split(";\n", 1)[0]
    scene = json.loads(scene_text)
    terrain = make_default_terrain()
    if scene["width"] != terrain.width or scene["height"] != terrain.height:
        raise ValueError("HTML dimensions do not match the hybrid_challenge source map")
    if not np.allclose(np.asarray(scene["elevation"]), terrain.elevation, atol=5.1e-5):
        raise ValueError("HTML elevation does not match the hybrid_challenge source map")
    if not np.array_equal(np.asarray(scene["obstacle"], dtype=bool), terrain.obstacle):
        raise ValueError("HTML obstacles do not match the hybrid_challenge source map")
    if not np.array_equal(np.asarray(scene["unsafe"], dtype=bool), terrain.unsafe_landing):
        raise ValueError("HTML unsafe-landing layer does not match the hybrid_challenge source map")
    return terrain


def _parse_initial_conditions(values: list[str], target: tuple[float, float]) -> list[InitialCondition]:
    if not values:
        values = ["5,3", "8,15", "5,27"]
    result = []
    for value in values:
        parts = [float(item.strip()) for item in value.split(",")]
        if len(parts) not in (2, 3):
            raise ValueError("each --initial must be x,y or x,y,heading_radians")
        heading = parts[2] if len(parts) == 3 else atan2(target[1] - parts[1], target[0] - parts[0])
        result.append(InitialCondition(parts[0], parts[1], heading))
    return result


def _parse_xy(value: str) -> tuple[float, float]:
    parts = [float(item.strip()) for item in value.split(",")]
    if len(parts) != 2:
        raise ValueError("expected x,y")
    return parts[0], parts[1]


def _full_dp_team_config(
    terrain: TerrainMap,
    config: dict[str, Any],
    *,
    require_ground_and_air: bool = True,
) -> TeamConfig:
    dp = config.get("dp", {})
    aerial = config.get("aerial", {})
    return TeamConfig(
        grid_resolution_m=terrain.resolution,
        heading_bins=8,
        connector_step_m=float(dp.get("connector_step_m", 0.35)),
        footprint_sample_step_m=float(dp.get("footprint_sample_step_m", 0.4)),
        max_slope_deg=float(config["vehicle"].get("max_slope_deg", 15.0)),
        max_roll_deg=float(dp.get("max_roll_deg", 12.0)),
        max_pitch_deg=float(dp.get("max_pitch_deg", 12.0)),
        safe_landing_threshold=float(aerial.get("safe_landing_threshold", 5.0)),
        cruise_altitude_m=float(aerial.get("cruise_altitude_m", 5.0)),
        air_obstacle_clearance_m=0.75,
        # Do not impose the previous artificial 6 m flight-leg minimum.
        # Keep one grid edge as the lower bound so takeoff and landing at the
        # same lattice point cannot be counted as a flight segment.
        minimum_flight_leg_m=terrain.resolution,
        maximum_flight_leg_m=None,
        require_ground_and_air=require_ground_and_air,
    )


def export_cost_to_go_png(
    result: dict[str, Any],
    terrain: TerrainMap,
    output: str | Path,
    *,
    overlay_policies: bool = False,
) -> Path:
    """Render the assembled-state DP value function, optionally with policies."""
    import os

    matplotlib_cache = Path("/tmp/mod_algo_matplotlib")
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    field = result["assembled_initial_cost_to_go"]
    costs = np.asarray(
        [[np.nan if value is None else value for value in row] for row in field["cost_wh"]],
        dtype=float,
    )
    resolution = float(field["x_resolution_m"])
    xs = np.arange(costs.shape[1], dtype=float) * resolution
    ys = np.arange(costs.shape[0], dtype=float) * resolution
    metadata = getattr(terrain, "map_25d_metadata", {})
    origin = metadata.get("local_enu_grid_origin_m", (0.0, 0.0))
    origin_x, origin_y = float(origin[0]), float(origin[1])
    display_xs = xs + origin_x
    display_ys = ys + origin_y

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    mesh = ax.pcolormesh(display_xs, display_ys, costs, shading="nearest", cmap="viridis_r")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("Optimal cost-to-go (Wh)")

    extent = (
        origin_x - 0.5 * terrain.resolution,
        origin_x + (terrain.width - 0.5) * terrain.resolution,
        origin_y - 0.5 * terrain.resolution,
        origin_y + (terrain.height - 0.5) * terrain.resolution,
    )
    obstacle_overlay = np.ma.masked_where(~terrain.obstacle, terrain.obstacle)
    unsafe_only = terrain.unsafe_landing & ~terrain.obstacle
    unsafe_overlay = np.ma.masked_where(~unsafe_only, unsafe_only)
    ax.imshow(
        unsafe_overlay,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=ListedColormap(["#f97316"]),
        alpha=0.28,
        zorder=2,
    )
    ax.imshow(
        obstacle_overlay,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap=ListedColormap(["#111827"]),
        alpha=0.88,
        zorder=3,
    )

    if overlay_policies:
        colors = ("#ef4444", "#38bdf8", "#facc15", "#a78bfa", "#34d399")
        for index, item in enumerate(result["results"]):
            if not item.get("success"):
                continue
            color = colors[index % len(colors)]
            points: list[tuple[float, float]] = []
            for event in item.get("mission_events", []):
                event_points = [
                    (float(p["x"]), float(p["y"])) for p in event.get("path", [])
                ]
                if points and event_points and points[-1] == event_points[0]:
                    event_points = event_points[1:]
                points.extend(event_points)
            if points:
                ax.plot(
                    [point[0] + origin_x for point in points],
                    [point[1] + origin_y for point in points],
                    color=color,
                    linewidth=2.2,
                    label=f"initial condition {item['initial_condition_index']}",
                    zorder=5,
                )
            home = item["home"]
            ax.scatter(
                home["x"] + origin_x,
                home["y"] + origin_y,
                s=55,
                color=color,
                edgecolor="white",
                zorder=6,
            )

    target = result["target"]
    ax.scatter(
        target["x"] + origin_x,
        target["y"] + origin_y,
        marker="*",
        s=180,
        color="#ffffff",
        edgecolor="#111827",
        label="assembled target",
        zorder=7,
    )
    region = metadata.get("requested_bounds_xy_m")
    if region and len(region) == 4:
        x_limits = (float(region[0]), float(region[1]))
        y_limits = (float(region[2]), float(region[3]))
        region_text = (
            f"selected region x=[{x_limits[0]:.2f}, {x_limits[1]:.2f}], "
            f"y=[{y_limits[0]:.2f}, {y_limits[1]:.2f}]"
        )
    else:
        x_limits = (origin_x, origin_x + (terrain.width - 1) * terrain.resolution)
        y_limits = (origin_y, origin_y + (terrain.height - 1) * terrain.resolution)
        region_text = "full map"
    ax.set(
        title=f"Air-Ground DP Cost-to-Go — {region_text}",
        xlabel="PCD/world x (m)",
        ylabel="PCD/world y (m)",
        xlim=x_limits,
        ylim=y_limits,
        aspect="equal",
    )
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(color="white", alpha=0.12, linewidth=0.5)
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def export_dp_path_planning_3d_png(
    result: dict[str, Any],
    terrain: TerrainMap,
    output: str | Path,
) -> Path:
    """Render the DP value layers and trajectories in the selected-region view."""
    import os

    matplotlib_cache = Path("/tmp/mod_algo_matplotlib")
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from scipy.interpolate import griddata

    layers = result["hybrid_cost_to_go"]
    ground = np.asarray(
        [
            [np.nan if value is None else value for value in row]
            for row in layers["ground"]["cost_wh"]
        ],
        dtype=float,
    )
    air = np.asarray(
        [
            [np.nan if value is None else value for value in row]
            for row in layers["air"]["cost_wh"]
        ],
        dtype=float,
    )
    resolution = float(layers["ground"]["x_resolution_m"])
    source_xs = np.arange(ground.shape[1], dtype=float) * resolution
    source_ys = np.arange(ground.shape[0], dtype=float) * resolution
    source_x_grid, source_y_grid = np.meshgrid(source_xs, source_ys)
    metadata = getattr(terrain, "map_25d_metadata", {})
    origin = metadata.get("local_enu_grid_origin_m", (0.0, 0.0))
    origin_x, origin_y = float(origin[0]), float(origin[1])
    display_resolution = min(0.5, resolution)
    xs = np.arange(0.0, source_xs[-1] + 0.5 * display_resolution, display_resolution)
    ys = np.arange(0.0, source_ys[-1] + 0.5 * display_resolution, display_resolution)
    x_grid, y_grid = np.meshgrid(xs, ys)
    display_x_grid = x_grid + origin_x
    display_y_grid = y_grid + origin_y

    def continuous_field(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        if not np.any(finite):
            return np.full(x_grid.shape, np.nan, dtype=float)
        points = np.column_stack((source_x_grid[finite], source_y_grid[finite]))
        samples = values[finite]
        linear = griddata(points, samples, (x_grid, y_grid), method="linear")
        nearest = griddata(points, samples, (x_grid, y_grid), method="nearest")
        interpolated = np.where(np.isfinite(linear), linear, nearest)
        return np.clip(interpolated, float(np.min(samples)), float(np.max(samples)))

    ground_continuous = continuous_field(ground)
    air_continuous = continuous_field(air)
    ground_feasible = griddata(
        np.column_stack((source_x_grid.ravel(), source_y_grid.ravel())),
        np.isfinite(ground).astype(float).ravel(),
        (x_grid, y_grid),
        method="nearest",
    ) >= 0.5
    ground_z = np.asarray(
        [
            [terrain.continuous_value("elevation", x, y) for x in xs]
            for y in ys
        ],
        dtype=float,
    )
    ground_base_layer = (
        "ground_elevation_m"
        if hasattr(terrain, "ground_elevation_m")
        else "elevation"
    )
    ground_base_grid = np.asarray(
        [
            [terrain.continuous_value(ground_base_layer, x, y) for x in xs]
            for y in ys
        ],
        dtype=float,
    )
    surface_height_layer = (
        "measured_surface_height_m"
        if hasattr(terrain, "measured_surface_height_m")
        else "surface_height_m"
    )
    surface_height_grid = np.asarray(
        [
            [terrain.continuous_value(surface_height_layer, x, y) for x in xs]
            for y in ys
        ],
        dtype=float,
    )
    effective_surface_height_grid = np.asarray(
        [
            [terrain.continuous_value("surface_height_m", x, y) for x in xs]
            for y in ys
        ],
        dtype=float,
    )
    obstacle_grid = griddata(
        np.column_stack((source_x_grid.ravel(), source_y_grid.ravel())),
        np.asarray(terrain.obstacle, dtype=float).ravel(),
        (x_grid, y_grid),
        method="nearest",
    ) >= 0.5
    surface_mask = obstacle_grid & np.isfinite(surface_height_grid) & (surface_height_grid > 0.05)
    surface_tree_z = np.where(surface_mask, ground_base_grid + surface_height_grid, np.nan)
    assumptions = result["results"][0].get("model_assumptions", {})
    cruise_altitude_m = float(assumptions.get("cruise_altitude_m", 5.0))
    air_clearance_m = float(assumptions.get("air_obstacle_clearance_m", 0.75))
    air_altitude = float(np.max(terrain.elevation)) + cruise_altitude_m
    air_z = np.full_like(air_continuous, air_altitude)
    air_blocked = (
        np.isfinite(effective_surface_height_grid)
        & (effective_surface_height_grid + air_clearance_m >= cruise_altitude_m)
    )
    blocked_any = obstacle_grid | air_blocked
    blocked_z = np.where(
        blocked_any,
        ground_base_grid + np.maximum(effective_surface_height_grid, 0.0) + 0.12,
        np.nan,
    )
    color_parts = [ground_continuous[ground_feasible]]
    if np.isfinite(air_continuous).any():
        color_parts.append(air_continuous[np.isfinite(air_continuous)])
    finite_values = np.concatenate(color_parts)
    if finite_values.size:
        value_min = float(np.min(finite_values))
        value_max = float(np.max(finite_values))
    else:
        value_min, value_max = 0.0, 1.0
    value_norm = Normalize(vmin=value_min, vmax=value_max)
    no_go_color = "#6b7280"

    figure = plt.figure(figsize=(12, 8), dpi=160)
    axis = figure.add_subplot(111, projection="3d")
    axis.plot_surface(
        display_x_grid,
        display_y_grid,
        ground_z,
        facecolors=plt.get_cmap("viridis_r")(value_norm(ground_continuous)),
        linewidth=0,
        antialiased=True,
        alpha=0.95,
        rcount=min(120, len(ys)),
        ccount=min(120, len(xs)),
    )
    if np.isfinite(air_continuous).any():
        axis.plot_surface(
            display_x_grid,
            display_y_grid,
            air_z,
            facecolors=plt.get_cmap("viridis_r")(value_norm(air_continuous)),
            linewidth=0,
            antialiased=True,
            alpha=0.72,
            rcount=min(120, len(ys)),
            ccount=min(120, len(xs)),
        )
    if np.isfinite(surface_tree_z).any():
        axis.plot_surface(
            display_x_grid,
            display_y_grid,
            surface_tree_z,
            color=no_go_color,
            linewidth=0,
            antialiased=True,
            alpha=0.92,
            rcount=min(120, len(ys)),
            ccount=min(120, len(xs)),
        )
    if np.isfinite(blocked_z).any():
        axis.plot_surface(
            display_x_grid,
            display_y_grid,
            blocked_z,
            color=no_go_color,
            linewidth=0,
            antialiased=True,
            alpha=0.96,
            rcount=min(120, len(ys)),
            ccount=min(120, len(xs)),
        )
    value_mappable = ScalarMappable(norm=value_norm, cmap="viridis_r")
    value_mappable.set_array([])
    figure.colorbar(value_mappable, ax=axis, shrink=0.75, pad=0.10, label="V (Wh)")

    colors = ("#ef4444", "#38bdf8", "#facc15", "#a78bfa", "#34d399")
    path_z_values: list[float] = []
    for index, item in enumerate(result["results"]):
        if not item.get("success"):
            continue
        path_x: list[float] = []
        path_y: list[float] = []
        path_z: list[float] = []
        for event in item.get("mission_events", []):
            action = event["action"]
            event_points = event.get("path", [])
            for point_index, point in enumerate(event_points):
                x, y = float(point["x"]), float(point["y"])
                airborne = (
                    action == "fly"
                    or (action == "takeoff" and point_index > 0)
                    or (action == "land" and point_index < len(event_points) - 1)
                )
                local_ground_z = terrain.continuous_value("elevation", x, y)
                z = air_altitude if airborne else local_ground_z + 0.16
                display_x, display_y = x + origin_x, y + origin_y
                if path_x and (display_x, display_y, z) == (
                    path_x[-1],
                    path_y[-1],
                    path_z[-1],
                ):
                    continue
                path_x.append(display_x)
                path_y.append(display_y)
                path_z.append(z)
                path_z_values.append(z)
        if path_x:
            axis.plot(
                path_x,
                path_y,
                path_z,
                color=colors[index % len(colors)],
                linewidth=3.0,
                label=f"trajectory from ground initial {item['initial_condition_index']}",
                zorder=10,
            )

    for item in result["results"]:
        home = item["home"]
        home_z = terrain.continuous_value("elevation", home["x"], home["y"])
        success = bool(item.get("success"))
        marker = "o" if success else "x"
        status = "" if success else " (no feasible policy)"
        axis.scatter(
            [home["x"] + origin_x],
            [home["y"] + origin_y],
            [home_z],
            s=18,
            c=colors[(item["initial_condition_index"] - 1) % len(colors)],
            marker=marker,
            edgecolors="white" if success else colors[(item["initial_condition_index"] - 1) % len(colors)],
            depthshade=False,
            label=f"ground initial {item['initial_condition_index']}{status}",
        )
    target = result["target"]
    target_z = terrain.continuous_value("elevation", target["x"], target["y"])
    axis.scatter(
        [target["x"] + origin_x],
        [target["y"] + origin_y],
        [target_z],
        marker="D",
        s=32,
        c="white",
        edgecolors="#111827",
        depthshade=False,
        label="ground goal",
    )

    region = metadata.get("requested_bounds_xy_m")
    if region and len(region) == 4:
        x_limits = (float(region[0]), float(region[1]))
        y_limits = (float(region[2]), float(region[3]))
        region_text = (
            f"selected region x=[{x_limits[0]:.2f}, {x_limits[1]:.2f}], "
            f"y=[{y_limits[0]:.2f}, {y_limits[1]:.2f}]"
        )
    else:
        x_limits = (origin_x, origin_x + (terrain.width - 1) * resolution)
        y_limits = (origin_y, origin_y + (terrain.height - 1) * resolution)
        region_text = "full map"
    z_values = np.concatenate(
        (
            ground_z[np.isfinite(ground_z)],
            blocked_z[np.isfinite(blocked_z)],
            np.asarray(path_z_values, dtype=float),
            np.asarray([air_altitude], dtype=float),
        )
    )
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    axis.set(
        title=f"DP Path Planning 3D — {region_text}",
        xlabel="PCD/world x (m)",
        ylabel="PCD/world y (m)",
        zlabel="terrain / discrete mode layer",
        xlim=x_limits,
        ylim=y_limits,
        zlim=(z_min - 0.5, z_max + max(0.5, 0.08 * (z_max - z_min))),
    )
    # Keep this identical to lidar_point_cloud_selected_region_3d.png.
    axis.view_init(elev=30.0, azim=-120.0)
    axis.legend(loc="upper left", framealpha=0.90)
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def export_cost_to_go_layers_html(
    result: dict[str, Any],
    terrain: TerrainMap,
    output: str | Path,
    *,
    overlay_policies: bool = True,
) -> Path:
    """Render ground and intermediate-air value slices as two spatial layers."""
    import plotly.graph_objects as go
    from scipy.interpolate import griddata

    layers = result["hybrid_cost_to_go"]
    ground = np.asarray(
        [
            [np.nan if value is None else value for value in row]
            for row in layers["ground"]["cost_wh"]
        ],
        dtype=float,
    )
    air = np.asarray(
        [
            [np.nan if value is None else value for value in row]
            for row in layers["air"]["cost_wh"]
        ],
        dtype=float,
    )
    resolution = float(layers["ground"]["x_resolution_m"])
    source_xs = np.arange(ground.shape[1], dtype=float) * resolution
    source_ys = np.arange(ground.shape[0], dtype=float) * resolution
    source_x_grid, source_y_grid = np.meshgrid(source_xs, source_ys)
    display_resolution = min(0.5, resolution)
    xs = np.arange(0.0, source_xs[-1] + 0.5 * display_resolution, display_resolution)
    ys = np.arange(0.0, source_ys[-1] + 0.5 * display_resolution, display_resolution)
    x_grid, y_grid = np.meshgrid(xs, ys)
    metadata = getattr(terrain, "map_25d_metadata", {})
    origin = metadata.get("local_enu_grid_origin_m", (0.0, 0.0))
    origin_x, origin_y = float(origin[0]), float(origin[1])
    display_x_grid = x_grid + origin_x
    display_y_grid = y_grid + origin_y

    def continuous_field(values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        if not np.any(finite):
            return np.full(x_grid.shape, np.nan, dtype=float)
        points = np.column_stack((source_x_grid[finite], source_y_grid[finite]))
        samples = values[finite]
        linear = griddata(points, samples, (x_grid, y_grid), method="linear")
        nearest = griddata(points, samples, (x_grid, y_grid), method="nearest")
        interpolated = np.where(np.isfinite(linear), linear, nearest)
        return np.clip(interpolated, float(np.min(samples)), float(np.max(samples)))

    ground_continuous = continuous_field(ground)
    air_continuous = continuous_field(air)
    ground_feasible = griddata(
        np.column_stack((source_x_grid.ravel(), source_y_grid.ravel())),
        np.isfinite(ground).astype(float).ravel(),
        (x_grid, y_grid),
        method="nearest",
    ) >= 0.5
    ground_z = np.asarray(
        [
            [terrain.continuous_value("elevation", x, y) for x in xs]
            for y in ys
        ],
        dtype=float,
    )
    surface_height_layer = (
        "measured_surface_height_m"
        if hasattr(terrain, "measured_surface_height_m")
        else "surface_height_m"
    )
    surface_height_grid = np.asarray(
        [
            [terrain.continuous_value(surface_height_layer, x, y) for x in xs]
            for y in ys
        ],
        dtype=float,
    )
    ground_base_layer = (
        "ground_elevation_m"
        if hasattr(terrain, "ground_elevation_m")
        else "elevation"
    )
    ground_base_grid = np.asarray(
        [
            [terrain.continuous_value(ground_base_layer, x, y) for x in xs]
            for y in ys
        ],
        dtype=float,
    )
    effective_surface_height_grid = np.asarray(
        [
            [terrain.continuous_value("surface_height_m", x, y) for x in xs]
            for y in ys
        ],
        dtype=float,
    )
    obstacle_grid = griddata(
        np.column_stack((source_x_grid.ravel(), source_y_grid.ravel())),
        np.asarray(terrain.obstacle, dtype=float).ravel(),
        (x_grid, y_grid),
        method="nearest",
    ) >= 0.5
    surface_mask = obstacle_grid & np.isfinite(surface_height_grid) & (surface_height_grid > 0.05)
    surface_tree_z = np.where(surface_mask, ground_base_grid + surface_height_grid, np.nan)
    assumptions = result["results"][0].get("model_assumptions", {})
    cruise_altitude_m = float(assumptions.get("cruise_altitude_m", 5.0))
    air_clearance_m = float(assumptions.get("air_obstacle_clearance_m", 0.75))
    air_altitude = float(np.max(terrain.elevation)) + cruise_altitude_m
    air_z = np.full_like(air_continuous, air_altitude)
    # This is a physical mode-availability overlay, not DP reachability:
    # ground obstacles and terrain-limit cells are not driveable; sufficiently
    # tall effective surface returns are not flyable at the fixed cruise AGL.
    air_blocked = (
        np.isfinite(effective_surface_height_grid)
        & (effective_surface_height_grid + air_clearance_m >= cruise_altitude_m)
    )
    blocked_any = obstacle_grid | air_blocked
    blocked_z = np.where(
        blocked_any,
        ground_base_grid + np.maximum(effective_surface_height_grid, 0.0) + 0.12,
        np.nan,
    )
    no_go_color = "#6b7280"
    color_parts = [ground_continuous[ground_feasible]]
    if np.isfinite(air_continuous).any():
        color_parts.append(air_continuous[np.isfinite(air_continuous)])
    finite_values = np.concatenate(color_parts)
    if finite_values.size:
        color_min = float(np.min(finite_values))
        color_max = float(np.max(finite_values))
    else:
        color_min, color_max = 0.0, 1.0

    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=display_x_grid,
            y=display_y_grid,
            z=ground_z,
            surfacecolor=ground_continuous,
            coloraxis="coloraxis",
            name="Ground cost-to-go",
            showscale=False,
            connectgaps=False,
            hovertemplate=(
                "ground<br>x=%{x:.1f} m<br>y=%{y:.1f} m"
                "<br>terrain z=%{z:.2f} m<br>V=%{surfacecolor:.3f} Wh<extra></extra>"
            ),
        )
    )
    if np.isfinite(air_continuous).any():
        figure.add_trace(
            go.Surface(
                x=display_x_grid,
                y=display_y_grid,
                z=air_z,
                surfacecolor=air_continuous,
                coloraxis="coloraxis",
                name="Intermediate air cost-to-go",
                showscale=False,
                opacity=0.72,
                connectgaps=False,
                hovertemplate=(
                    "air (intermediate only)<br>x=%{x:.1f} m<br>y=%{y:.1f} m"
                    "<br>flight z=%{z:.2f} m<br>V=%{surfacecolor:.3f} Wh<extra></extra>"
                ),
            )
        )
    if np.isfinite(surface_tree_z).any():
        figure.add_trace(
            go.Surface(
                x=display_x_grid,
                y=display_y_grid,
                z=surface_tree_z,
                surfacecolor=np.zeros_like(surface_tree_z),
                colorscale=[[0.0, no_go_color], [1.0, no_go_color]],
                cmin=0.0,
                cmax=1.0,
                name="No-go surface / trees",
                showscale=False,
                opacity=0.92,
                connectgaps=False,
                hovertemplate=(
                    "LiDAR surface / tree returns<br>x=%{x:.1f} m<br>y=%{y:.1f} m"
                    "<br>surface z=%{z:.2f} m<br>no-go surface-height region"
                    "<extra></extra>"
                ),
            )
        )
    figure.add_trace(
        go.Surface(
            x=display_x_grid,
            y=display_y_grid,
            z=blocked_z,
            surfacecolor=np.zeros_like(blocked_z),
            colorscale=[[0.0, no_go_color], [1.0, no_go_color]],
            cmin=0.0,
            cmax=1.0,
            name="Blocked for ground or air",
            showscale=False,
            opacity=0.96,
            connectgaps=False,
            hovertemplate=(
                "blocked for ground drive or air flight"
                "<br>x=%{x:.1f} m<br>y=%{y:.1f} m<extra></extra>"
            ),
        )
    )

    if overlay_policies:
        colors = ("#ef4444", "#38bdf8", "#facc15", "#a78bfa", "#34d399")
        for index, item in enumerate(result["results"]):
            if not item.get("success"):
                continue
            path_x: list[float] = []
            path_y: list[float] = []
            path_z: list[float] = []
            for event in item.get("mission_events", []):
                action = event["action"]
                event_points = event.get("path", [])
                for point_index, point in enumerate(event_points):
                    x, y = float(point["x"]), float(point["y"])
                    airborne = (
                        action == "fly"
                        or (action == "takeoff" and point_index > 0)
                        or (action == "land" and point_index < len(event_points) - 1)
                    )
                    z = (
                        air_altitude
                        if airborne
                        else terrain.continuous_value("elevation", x, y) + 0.16
                    )
                    display_x, display_y = x + origin_x, y + origin_y
                    if path_x and (display_x, display_y, z) == (
                        path_x[-1],
                        path_y[-1],
                        path_z[-1],
                    ):
                        continue
                    path_x.append(display_x)
                    path_y.append(display_y)
                    path_z.append(z)
            figure.add_trace(
                go.Scatter3d(
                    x=path_x,
                    y=path_y,
                    z=path_z,
                    mode="lines",
                    line={"width": 7, "color": colors[index % len(colors)]},
                    name=f"trajectory from ground initial {item['initial_condition_index']}",
                    hovertemplate=f"trajectory {item['initial_condition_index']}<extra></extra>",
                )
            )

    initial_colors = ("#ef4444", "#38bdf8", "#facc15", "#a78bfa", "#34d399")
    for item in result["results"]:
        home = item["home"]
        ground_height = terrain.continuous_value("elevation", home["x"], home["y"])
        success = bool(item.get("success"))
        marker_symbol = "circle" if success else "x"
        status = "" if success else " (no feasible policy)"
        figure.add_trace(
            go.Scatter3d(
                x=[home["x"] + origin_x],
                y=[home["y"] + origin_y],
                z=[ground_height],
                mode="markers",
                marker={
                    "size": 7,
                    "symbol": marker_symbol,
                    "color": initial_colors[
                        (item["initial_condition_index"] - 1) % len(initial_colors)
                    ],
                    "line": {"color": "white", "width": 1},
                },
                name=f"ground initial {item['initial_condition_index']}{status}",
                hovertemplate=(
                    f"ground initial {item['initial_condition_index']}"
                    "<br>x=%{x:.2f} m<br>y=%{y:.2f} m<extra></extra>"
                ),
            )
        )
    target = result["target"]
    target_height = terrain.continuous_value("elevation", target["x"], target["y"])
    figure.add_trace(
        go.Scatter3d(
            x=[target["x"] + origin_x],
            y=[target["y"] + origin_y],
            z=[target_height],
            mode="markers",
            marker={"size": 8, "color": "white", "symbol": "diamond", "line": {"color": "black", "width": 2}},
            name="ground goal",
            hovertemplate="ground goal<extra></extra>",
        )
    )
    region = metadata.get("requested_bounds_xy_m")
    if region and len(region) == 4:
        x_range = [float(region[0]), float(region[1])]
        y_range = [float(region[2]), float(region[3])]
        region_text = (
            f"selected region x=[{x_range[0]:.2f}, {x_range[1]:.2f}], "
            f"y=[{y_range[0]:.2f}, {y_range[1]:.2f}]"
        )
    else:
        x_range = None
        y_range = None
        region_text = "full map"
    scene = {
        "xaxis_title": "PCD/world x (m)",
        "yaxis_title": "PCD/world y (m)",
        "zaxis_title": "terrain / discrete mode layer",
        "aspectmode": "manual",
        "aspectratio": {"x": 1.8, "y": 1.0, "z": 0.55},
        # Plotly camera elevation is atan(z / sqrt(x^2 + y^2)); this is 30°.
        "camera": {"eye": {"x": 1.5, "y": -1.7, "z": 1.31}},
    }
    if x_range is not None and y_range is not None:
        scene["xaxis"] = {"range": x_range}
        scene["yaxis"] = {"range": y_range}
    figure.update_layout(
        title=f"Continuous Two-Layer Air-Ground Cost-to-Go — {region_text}",
        coloraxis={
            "colorscale": "Viridis_r",
            "cmin": color_min,
            "cmax": color_max,
            "colorbar": {"title": "V (Wh)"},
        },
        scene=scene,
        legend={"x": 0.01, "y": 0.99},
        margin={"l": 0, "r": 0, "b": 0, "t": 55},
    )
    output_path = Path(output)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(output_path, include_plotlyjs=True, full_html=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Solve one case-1 air/ground DP policy for multiple initial conditions."
    )
    parser.add_argument(
        "--map-html",
        type=Path,
        default=DEFAULT_CASE_1_MAP,
    )
    parser.add_argument(
        "--map-25d",
        type=Path,
        help="load a generated map_25d.npz instead of the built-in case-1 HTML map",
    )
    parser.add_argument(
        "--target",
        default="45,28",
        help="target x,y in TerrainMap coordinates; for map_25d.npz this is relative to the raster origin",
    )
    parser.add_argument(
        "--initial",
        action="append",
        default=[],
        help="repeatable x,y or x,y,heading_radians initial condition",
    )
    parser.add_argument(
        "--max-ground-slope-deg",
        type=float,
        default=None,
        help="force ground-to-air routing when terrain exceeds this driving slope limit",
    )
    parser.add_argument(
        "--allow-single-mode",
        action="store_true",
        help="do not force an arbitrary flight; fly only when it lowers cost or ground is infeasible",
    )
    parser.add_argument(
        "--ground-only",
        action="store_true",
        help="forbid takeoff and flight; every trajectory must drive on the ground",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="JSON output path; omitted uses a fresh timestamped filename",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--visualize-cost-to-go",
        action="store_true",
        help="write a PNG of the complete assembled-state DP value function",
    )
    parser.add_argument(
        "--overlay-policies",
        action="store_true",
        help="overlay paths for the supplied initial conditions on the cost-to-go PNG",
    )
    args = parser.parse_args()

    target = _parse_xy(args.target)
    initial_conditions = _parse_initial_conditions(args.initial, target)
    map_source = str(args.map_25d or args.map_html)
    scenario = "pcd_25d" if args.map_25d else "hybrid_challenge"
    if args.map_25d:
        terrain, _ = load_planner_terrain(args.map_25d)
    else:
        terrain = load_case_1_terrain(args.map_html)
    config = load_config(
        overrides=(
            {"vehicle": {"max_slope_deg": args.max_ground_slope_deg}}
            if args.max_ground_slope_deg is not None
            else None
        )
    )
    team_config = _full_dp_team_config(
        terrain,
        config,
        require_ground_and_air=not args.allow_single_mode and not args.ground_only,
    )
    planner = MultiStartAirGroundDP(
        terrain,
        target,
        initial_conditions,
        config,
        team_config,
        ground_only=args.ground_only,
    )
    result = planner.solve()
    result["scenario"] = scenario
    result["map_source"] = map_source
    for item in result["results"]:
        item["scenario"] = scenario
        item["map_source"] = map_source
    output = (
        Path(args.output)
        if args.output
        else Path("results")
        / f"case_1_dp_multi_start_{strftime('%Y%m%d_%H%M%S')}.json"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output}")

    if args.render:
        for item in result["results"]:
            if not item["success"]:
                continue
            html_output = output.with_name(
                f"{output.stem}_initial_{item['initial_condition_index']:02d}_3d.html"
            )
            export_dp_html3d(item, html_output, terrain=terrain)
            print(f"wrote {html_output}")
    if args.visualize_cost_to_go:
        layer_output = output.with_name(f"{output.stem}_cost_to_go_layers_3d.html")
        png_output = output.with_name(f"{output.stem}_cost_to_go.png")
        path_png_output = output.with_name(f"{output.stem}_path_planning_3d.png")
        export_cost_to_go_png(
            result,
            terrain,
            png_output,
            overlay_policies=args.overlay_policies,
        )
        print(f"wrote {png_output}")
        export_cost_to_go_layers_html(
            result,
            terrain,
            layer_output,
            overlay_policies=args.overlay_policies,
        )
        print(f"wrote {layer_output}")
        export_dp_path_planning_3d_png(result, terrain, path_png_output)
        print(f"wrote {path_png_output}")


if __name__ == "__main__":
    main()
