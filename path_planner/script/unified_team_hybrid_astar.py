from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import atan2, ceil, hypot, pi
from pathlib import Path
from time import perf_counter, strftime
from typing import Any, Literal

from .config import load_config
from .continuous_mission_planner import _battery_report
from .platform import (
    Pose2D,
    RectFootprint,
    estimate_platform_roll_pitch,
    platform_collision,
    wrap_angle,
)
from .rover import ground_speed_and_power, mass_scaled_flight_power
from .terrain import TerrainMap
from .unified_team_scenarios import make_unified_team_scenario


TeamMode = Literal["assembled_drive", "disassembled_flight"]


@dataclass(frozen=True)
class UnifiedTeamConfig:
    """Finite hybrid lattice for one five-module team state."""

    grid_resolution_m: float = 1.0
    heading_bins: int = 8
    connector_step_m: float = 0.35
    footprint_sample_step_m: float = 0.4
    max_slope_deg: float = 15.0
    max_roll_deg: float = 12.0
    max_pitch_deg: float = 12.0
    safe_landing_threshold: float = 5.0
    cruise_altitude_m: float = 5.0
    air_obstacle_clearance_m: float = 0.75
    minimum_flight_leg_m: float = 6.0
    maximum_flight_leg_m: float | None = None
    require_ground_and_air: bool = True
    heuristic_weight: float = 5.0
    max_expansions: int = 250_000


@dataclass(frozen=True)
class TeamState:
    ix: int
    iy: int
    heading_bin: int
    mode: TeamMode
    used_ground: bool
    used_air: bool
    air_distance_ticks: int = 0


@dataclass(frozen=True)
class TeamEdge:
    action: Literal["drive", "takeoff", "fly", "land"]
    energy_wh: float
    duration_s: float
    agent_energy_wh: tuple[float, float, float, float, float]


class UnifiedFiveAgentHybridAStar:
    """A* for a five-module team represented by exactly one planning state.

    There are no per-rover spatial states or independently selected paths.
    The team is either assembled and driving along one platform centerline, or
    disassembled and flying along one shared team centerline. Edge energy is
    always the sum of the four rover-drone batteries and the drill battery.
    """

    _MOVES = (
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    )
    _AGENT_NAMES = ("rover_1", "rover_2", "rover_3", "rover_4", "drill")

    def __init__(
        self,
        terrain: TerrainMap,
        home: tuple[float, float],
        target: tuple[float, float],
        vehicle_cfg: dict[str, Any],
        platform_cfg: dict[str, Any],
        battery_cfg: dict[str, Any],
        docking_energy_wh: float,
        undocking_energy_wh: float,
        config: UnifiedTeamConfig,
    ) -> None:
        self.terrain = terrain
        self.home = (float(home[0]), float(home[1]))
        self.target = (float(target[0]), float(target[1]))
        self.vehicle_cfg = vehicle_cfg
        self.battery_cfg = battery_cfg
        self.docking_energy_wh = float(docking_energy_wh)
        self.undocking_energy_wh = float(undocking_energy_wh)
        self.config = config
        self.rover_offsets = [
            (float(item["x"]), float(item["y"]), float(item.get("theta", 0.0)))
            for item in platform_cfg["rover_offsets"]
        ]
        if len(self.rover_offsets) != 4:
            raise ValueError("the unified team model requires exactly four rover-drone modules")
        self.platform_footprint = RectFootprint(**platform_cfg["platform_footprint"])
        self.rover_footprint = RectFootprint(**platform_cfg["rover_footprint"])
        self.rover_mass_kg = float(vehicle_cfg["mass_kg"])
        self.drill_mass_kg = float(vehicle_cfg.get("drill_mass_kg", 3.5))
        self.loaded_rover_mass_kg = self.rover_mass_kg + self.drill_mass_kg / 4.0
        self.grid_width = int(round((terrain.width - 1) * terrain.resolution / config.grid_resolution_m)) + 1
        self.grid_height = int(round((terrain.height - 1) * terrain.resolution / config.grid_resolution_m)) + 1
        self._start_sentinel = (-1, -1)
        self._target_sentinel = (self.grid_width, self.grid_height)
        self.minimum_air_ticks = int(ceil(config.minimum_flight_leg_m * 10.0))
        self.maximum_air_ticks = (
            None
            if config.maximum_flight_leg_m is None
            else int(config.maximum_flight_leg_m * 10.0 + 1e-9)
        )
        self._ground_pose_cache: dict[tuple[int, int, int], bool] = {}
        self._validate()
        self._cached_takeoff_edge = self._make_takeoff_edge()
        self._cached_landing_edge = self._make_landing_edge()
        self._ground_lower_bound_wh_per_m = (
            4.0
            * float(self.vehicle_cfg["ground_electronics_power_w"])
            / float(self.vehicle_cfg["ground_max_speed_mps"])
            / 3600.0
        )
        rover_flight_power = mass_scaled_flight_power(
            float(self.vehicle_cfg["forward_flight_power_w"]),
            self.rover_mass_kg,
            self.vehicle_cfg,
        )
        drill_flight_power = mass_scaled_flight_power(
            float(self.vehicle_cfg["forward_flight_power_w"]),
            self.drill_mass_kg,
            self.vehicle_cfg,
        )
        self._air_energy_wh_per_m = (
            4.0 * rover_flight_power + drill_flight_power
        ) / float(self.vehicle_cfg["air_max_speed_mps"]) / 3600.0

    def plan(self, time_limit_s: float) -> dict[str, Any]:
        started = perf_counter()
        start_ix, start_iy = self._start_sentinel
        goal_ix, goal_iy = self._target_sentinel
        direct_heading = atan2(self.target[1] - self.home[1], self.target[0] - self.home[0])
        start_heading = self._heading_bin(direct_heading)
        start = TeamState(
            start_ix,
            start_iy,
            start_heading,
            "assembled_drive",
            False,
            False,
            0,
        )
        if not self._ground_pose_feasible(start_ix, start_iy, start_heading):
            return self._failure(started, "Home is not feasible for the assembled team footprint")

        best: dict[TeamState, float] = {start: 0.0}
        parent: dict[TeamState, tuple[TeamState, TeamEdge]] = {}
        queue: list[tuple[float, int, float, TeamState]] = []
        tie = count()
        heappush(
            queue,
            (self.config.heuristic_weight * self._heuristic(start), next(tie), 0.0, start),
        )
        expanded = 0
        generated = 1
        peak_open = 1
        goal_state: TeamState | None = None
        termination = "open_exhausted"

        while queue:
            if perf_counter() - started >= time_limit_s:
                termination = "time_limit"
                break
            _, _, cost, state = heappop(queue)
            if cost > best.get(state, float("inf")) + 1e-10:
                continue
            if self._is_goal(state, goal_ix, goal_iy):
                goal_state = state
                termination = "astar_goal_popped"
                break
            if expanded >= self.config.max_expansions:
                termination = "max_expansions"
                break
            expanded += 1

            for next_state, edge in self._successors(state):
                next_cost = cost + edge.energy_wh
                if next_cost + 1e-10 >= best.get(next_state, float("inf")):
                    continue
                best[next_state] = next_cost
                parent[next_state] = (state, edge)
                heappush(
                    queue,
                    (
                        next_cost + self.config.heuristic_weight * self._heuristic(next_state),
                        next(tie),
                        next_cost,
                        next_state,
                    ),
                )
                generated += 1
                peak_open = max(peak_open, len(queue))

        elapsed = perf_counter() - started
        if goal_state is None:
            return self._failure(
                started,
                f"no unified team route ({termination})",
                elapsed=elapsed,
                expanded=expanded,
                generated=generated,
                peak_open=peak_open,
            )

        states = [goal_state]
        edges: list[TeamEdge] = []
        cursor = goal_state
        while cursor != start:
            previous, edge = parent[cursor]
            states.append(previous)
            edges.append(edge)
            cursor = previous
        states.reverse()
        edges.reverse()
        return self._result(
            states,
            edges,
            elapsed,
            expanded,
            generated,
            peak_open,
            termination,
        )

    def _successors(self, state: TeamState) -> list[tuple[TeamState, TeamEdge]]:
        successors: list[tuple[TeamState, TeamEdge]] = []
        if state.mode == "assembled_drive":
            for next_ix, next_iy, heading_bin, _ in self._neighbor_grid_nodes(state):
                if not self._ground_edge_feasible(state, next_ix, next_iy, heading_bin):
                    continue
                edge = self._drive_edge(state, next_ix, next_iy)
                successors.append(
                    (
                        TeamState(
                            next_ix,
                            next_iy,
                            heading_bin,
                            "assembled_drive",
                            True,
                            state.used_air,
                            0,
                        ),
                        edge,
                    )
                )
            x, y = self._world(state.ix, state.iy)
            target_distance = hypot(self.target[0] - x, self.target[1] - y)
            goal_flags_ready = (
                not self.config.require_ground_and_air
                or (state.used_ground and state.used_air)
            )
            if (
                goal_flags_ready
                and 1e-9 < target_distance <= 1.75 * self.config.grid_resolution_m
            ):
                target_heading = self._heading_bin(
                    atan2(self.target[1] - y, self.target[0] - x)
                )
                goal_ix, goal_iy = self._target_sentinel
                if self._ground_edge_feasible(
                    state,
                    goal_ix,
                    goal_iy,
                    target_heading,
                ):
                    edge = self._drive_edge(state, goal_ix, goal_iy)
                    successors.append(
                        (
                            TeamState(
                                goal_ix,
                                goal_iy,
                                target_heading,
                                "assembled_drive",
                                True,
                                state.used_air,
                                0,
                            ),
                            edge,
                        )
                    )
            if self._safe_switch_pose(state):
                edge = self._takeoff_edge()
                successors.append(
                    (
                        TeamState(
                            state.ix,
                            state.iy,
                            state.heading_bin,
                            "disassembled_flight",
                            state.used_ground,
                            state.used_air,
                            0,
                        ),
                        edge,
                    )
                )
        else:
            for next_ix, next_iy, heading_bin, distance in self._neighbor_grid_nodes(state):
                if not self._air_edge_feasible(state, next_ix, next_iy):
                    continue
                raw_ticks = state.air_distance_ticks + int(round(distance * 10.0))
                if self.maximum_air_ticks is not None and raw_ticks > self.maximum_air_ticks:
                    continue
                ticks = (
                    raw_ticks
                    if self.maximum_air_ticks is not None
                    else min(self.minimum_air_ticks, raw_ticks)
                )
                successors.append(
                    (
                        TeamState(
                            next_ix,
                            next_iy,
                            heading_bin,
                            "disassembled_flight",
                            state.used_ground,
                            True,
                            ticks,
                        ),
                        self._flight_edge(distance),
                    )
                )
            if (
                state.air_distance_ticks >= self.minimum_air_ticks
                and self._safe_switch_pose(state)
            ):
                successors.append(
                    (
                        TeamState(
                            state.ix,
                            state.iy,
                            state.heading_bin,
                            "assembled_drive",
                            state.used_ground,
                            state.used_air,
                            0,
                        ),
                        self._landing_edge(),
                    )
                )
        return successors

    def _drive_edge(self, state: TeamState, next_ix: int, next_iy: int) -> TeamEdge:
        start = self._world(state.ix, state.iy)
        end = self._world(next_ix, next_iy)
        energy_each, duration = self._ground_segment_energy(start, end)
        agents = (energy_each, energy_each, energy_each, energy_each, 0.0)
        return TeamEdge("drive", float(sum(agents)), duration, agents)

    def _takeoff_edge(self) -> TeamEdge:
        return self._cached_takeoff_edge

    def _make_takeoff_edge(self) -> TeamEdge:
        duration = float(self.vehicle_cfg["takeoff_time_s"])
        rover_energy = (
            mass_scaled_flight_power(
                float(self.vehicle_cfg["takeoff_power_w"]),
                self.rover_mass_kg,
                self.vehicle_cfg,
            )
            * duration
            / 3600.0
        )
        drill_energy = (
            mass_scaled_flight_power(
                float(self.vehicle_cfg["takeoff_power_w"]),
                self.drill_mass_kg,
                self.vehicle_cfg,
            )
            * duration
            / 3600.0
        )
        switch_share = self.undocking_energy_wh / 4.0
        agents = (
            rover_energy + switch_share,
            rover_energy + switch_share,
            rover_energy + switch_share,
            rover_energy + switch_share,
            drill_energy,
        )
        return TeamEdge("takeoff", float(sum(agents)), duration, agents)

    def _flight_edge(self, distance_m: float) -> TeamEdge:
        duration = distance_m / float(self.vehicle_cfg["air_max_speed_mps"])
        rover_energy = (
            mass_scaled_flight_power(
                float(self.vehicle_cfg["forward_flight_power_w"]),
                self.rover_mass_kg,
                self.vehicle_cfg,
            )
            * duration
            / 3600.0
        )
        drill_energy = (
            mass_scaled_flight_power(
                float(self.vehicle_cfg["forward_flight_power_w"]),
                self.drill_mass_kg,
                self.vehicle_cfg,
            )
            * duration
            / 3600.0
        )
        agents = (rover_energy, rover_energy, rover_energy, rover_energy, drill_energy)
        return TeamEdge("fly", float(sum(agents)), duration, agents)

    def _landing_edge(self) -> TeamEdge:
        return self._cached_landing_edge

    def _make_landing_edge(self) -> TeamEdge:
        duration = float(self.vehicle_cfg["landing_time_s"])
        rover_energy = (
            mass_scaled_flight_power(
                float(self.vehicle_cfg["landing_power_w"]),
                self.rover_mass_kg,
                self.vehicle_cfg,
            )
            * duration
            / 3600.0
        )
        drill_energy = (
            mass_scaled_flight_power(
                float(self.vehicle_cfg["landing_power_w"]),
                self.drill_mass_kg,
                self.vehicle_cfg,
            )
            * duration
            / 3600.0
        )
        switch_share = self.docking_energy_wh / 4.0
        agents = (
            rover_energy + switch_share,
            rover_energy + switch_share,
            rover_energy + switch_share,
            rover_energy + switch_share,
            drill_energy,
        )
        return TeamEdge("land", float(sum(agents)), duration, agents)

    def _ground_segment_energy(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> tuple[float, float]:
        distance = hypot(end[0] - start[0], end[1] - start[1])
        samples = list(self.terrain.line_samples(start, end, self.config.connector_step_m))
        energy_wh = 0.0
        duration_s = 0.0
        for first, second in zip(samples[:-1], samples[1:]):
            local_distance = hypot(second[0] - first[0], second[1] - first[1])
            if local_distance <= 1e-12:
                continue
            midpoint = ((first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5)
            grade = (
                self.terrain.continuous_value("elevation", second[0], second[1])
                - self.terrain.continuous_value("elevation", first[0], first[1])
            ) / local_distance
            speed, power = ground_speed_and_power(
                _ContinuousTerrainAdapter(self.terrain),
                midpoint[0],
                midpoint[1],
                self.vehicle_cfg,
                directional_grade=grade,
                mass_kg=self.loaded_rover_mass_kg,
            )
            local_duration = local_distance / speed
            duration_s += local_duration
            energy_wh += power * local_duration / 3600.0
        if distance <= 1e-12:
            return 0.0, 0.0
        return float(energy_wh), float(duration_s)

    def _ground_edge_feasible(
        self,
        state: TeamState,
        next_ix: int,
        next_iy: int,
        heading_bin: int,
    ) -> bool:
        if not self._ground_pose_feasible(next_ix, next_iy, heading_bin):
            return False
        start = self._world(state.ix, state.iy)
        end = self._world(next_ix, next_iy)
        heading = self._heading(heading_bin)
        for x, y in self.terrain.line_samples(start, end, self.config.connector_step_m):
            if not self._ground_pose_feasible_world(x, y, heading):
                return False
        return True

    def _ground_pose_feasible(self, ix: int, iy: int, heading_bin: int) -> bool:
        key = (ix, iy, heading_bin)
        if key not in self._ground_pose_cache:
            x, y = self._world(ix, iy)
            self._ground_pose_cache[key] = self._ground_pose_feasible_world(
                x,
                y,
                self._heading(heading_bin),
            )
        return self._ground_pose_cache[key]

    def _ground_pose_feasible_world(self, x: float, y: float, heading: float) -> bool:
        pose: Pose2D = (x, y, heading)
        if platform_collision(
            pose,
            self.rover_offsets,
            self.platform_footprint,
            self.rover_footprint,
            self.terrain,
            self.config.footprint_sample_step_m,
            self.config.max_slope_deg,
        ):
            return False
        roll, pitch = estimate_platform_roll_pitch(pose, self.rover_offsets, self.terrain)
        return abs(roll) <= self.config.max_roll_deg and abs(pitch) <= self.config.max_pitch_deg

    def _safe_switch_pose(self, state: TeamState) -> bool:
        x, y = self._world(state.ix, state.iy)
        return (
            self._ground_pose_feasible(state.ix, state.iy, state.heading_bin)
            and self.terrain.continuous_safe_landing(
                x,
                y,
                self.config.safe_landing_threshold,
            )
        )

    def _air_edge_feasible(
        self,
        state: TeamState,
        next_ix: int,
        next_iy: int,
    ) -> bool:
        """Check fixed-AGL flight against optional above-ground surface height."""
        if not hasattr(self.terrain, "surface_height_m"):
            return True
        start = self._world(state.ix, state.iy)
        end = self._world(next_ix, next_iy)
        available_height = (
            self.config.cruise_altitude_m
            - self.config.air_obstacle_clearance_m
        )
        for x_m, y_m in self.terrain.line_samples(
            start,
            end,
            self.config.connector_step_m,
        ):
            if self.terrain.continuous_value("surface_height_m", x_m, y_m) >= available_height:
                return False
        return True

    def _heuristic(self, state: TeamState) -> float:
        x, y = self._world(state.ix, state.iy)
        distance = hypot(self.target[0] - x, self.target[1] - y)
        minimum_rate = min(
            self._ground_lower_bound_wh_per_m,
            self._air_energy_wh_per_m,
        )
        lower_bound = distance * minimum_rate
        # The requested hybrid route makes some costs unavoidable.  Adding
        # them prevents A* from exhaustively preferring superficially cheap
        # ground-only prefixes while preserving admissibility.
        if self.config.require_ground_and_air:
            if state.mode == "assembled_drive" and not state.used_air:
                lower_bound += self._cached_takeoff_edge.energy_wh
                lower_bound += self._cached_landing_edge.energy_wh
                lower_bound += self.config.minimum_flight_leg_m * max(
                    0.0,
                    self._air_energy_wh_per_m - minimum_rate,
                )
            elif state.mode == "disassembled_flight":
                lower_bound += self._cached_landing_edge.energy_wh
                remaining_air_m = max(
                    0.0,
                    (self.minimum_air_ticks - state.air_distance_ticks) / 10.0,
                )
                lower_bound += remaining_air_m * max(
                    0.0,
                    self._air_energy_wh_per_m - minimum_rate,
                )
        return float(lower_bound)

    def _is_goal(self, state: TeamState, goal_ix: int, goal_iy: int) -> bool:
        if (state.ix, state.iy) != (goal_ix, goal_iy) or state.mode != "assembled_drive":
            return False
        if self.config.require_ground_and_air:
            return state.used_ground and state.used_air
        return True

    def _result(
        self,
        states: list[TeamState],
        edges: list[TeamEdge],
        elapsed: float,
        expanded: int,
        generated: int,
        peak_open: int,
        termination: str,
    ) -> dict[str, Any]:
        agent_energy = {name: 0.0 for name in self._AGENT_NAMES}
        breakdown = {
            "assembled_drive_wh": 0.0,
            "disassembled_flight_wh": 0.0,
            "takeoff_and_undock_wh": 0.0,
            "landing_and_dock_wh": 0.0,
        }
        for edge in edges:
            for name, energy in zip(self._AGENT_NAMES, edge.agent_energy_wh):
                agent_energy[name] += float(energy)
            key = {
                "drive": "assembled_drive_wh",
                "fly": "disassembled_flight_wh",
                "takeoff": "takeoff_and_undock_wh",
                "land": "landing_and_dock_wh",
            }[edge.action]
            breakdown[key] += edge.energy_wh

        mission_events, timeline, total_time = self._consolidate_events(states, edges)
        objective = float(sum(edge.energy_wh for edge in edges))
        battery_report = _battery_report(self.battery_cfg, agent_energy)
        goal_state = states[-1]
        return {
            "planner": "Unified Five-Agent Hybrid A*",
            "planner_role": "single-state collective air-ground planner",
            "scenario": "",
            "success": True,
            "reason": None,
            "objective_energy_wh": objective,
            "solve_time_s": float(elapsed),
            "mission_duration_s": float(total_time),
            "team_contract": {
                "module_count": 5,
                "planning_states_per_time_step": 1,
                "independent_agent_paths": False,
                "drive_policy": "all five modules move as one assembled platform; the drill rides and its mass is shared by four rover drivetrains",
                "flight_policy": "all four rover-drones and the drill fly together on one shared team centerline",
                "objective": "sum of energy used by rover_1, rover_2, rover_3, rover_4, and drill",
            },
            "required_modes": (
                ["assembled_drive", "disassembled_flight"]
                if self.config.require_ground_and_air
                else ["assembled_drive or disassembled_flight"]
            ),
            "mode_sequence": [event["mode"] for event in mission_events],
            "mission_events": mission_events,
            "mission_timeline": timeline,
            "team_path": [self._state_json(state) for state in states],
            "energy_breakdown": {key: float(value) for key, value in breakdown.items()},
            "agent_energy_wh": {key: float(value) for key, value in agent_energy.items()},
            "energy_sum_check_wh": float(sum(agent_energy.values())),
            "battery_report": battery_report,
            "terminal_team_state": {
                **self._state_json(goal_state),
                "all_five_modules_present": True,
                "drill_attached": True,
            },
            "search_statistics": {
                "expanded_states": expanded,
                "generated_states": generated,
                "peak_open_size": peak_open,
                "termination": termination,
                "optimal_on_finite_hybrid_lattice": self.config.heuristic_weight <= 1.0,
                "grid_resolution_m": self.config.grid_resolution_m,
                "heading_bins": self.config.heading_bins,
                "minimum_flight_leg_m": self.config.minimum_flight_leg_m,
                "maximum_flight_leg_m": self.config.maximum_flight_leg_m,
                "heuristic_weight": self.config.heuristic_weight,
            },
            "optimality_claim": {
                "finite_run": (
                    "minimum summed five-agent energy on the configured finite hybrid lattice"
                    if self.config.heuristic_weight <= 1.0
                    else "feasible summed-five-agent-energy route from weighted A*; no optimality certificate"
                ),
                "continuous_space": "no global-optimality claim outside the configured grid, headings, and synchronized team abstraction",
                "heuristic": "admissible minimum straight-line team energy lower bound, multiplied by the reported search weight",
            },
            "model_assumptions": {
                "team_abstraction": "one synchronized team pose; no individually planned rover or drill trajectory",
                "assembled_ground_mass_per_rover_kg": self.loaded_rover_mass_kg,
                "drill_ground_propulsion_energy_wh": 0.0,
                "drill_ground_role": "passive payload whose mass is included in rover drivetrain energy",
                "flight_masses_kg": {
                    "each_rover": self.rover_mass_kg,
                    "drill": self.drill_mass_kg,
                },
                "switching": "takeoff only after collective undocking; landing immediately followed by collective docking",
                "minimum_flight_leg_m": self.config.minimum_flight_leg_m,
                "maximum_flight_leg_m": self.config.maximum_flight_leg_m,
                "terrain": "assembled motion checks the full platform/rover footprint, slope, roll, and pitch; mode switches require safe terrain",
                "air_clearance": "cruise altitude is modeled relative to local terrain and clears map obstacles",
                "cruise_altitude_m": self.config.cruise_altitude_m,
                "air_obstacle_clearance_m": self.config.air_obstacle_clearance_m,
                "surface_height_constraint": (
                    "flight edges are rejected where above-ground canopy/building height plus clearance reaches cruise altitude"
                    if hasattr(self.terrain, "surface_height_m")
                    else "no above-ground surface-height layer supplied"
                ),
            },
        }

    def _consolidate_events(
        self,
        states: list[TeamState],
        edges: list[TeamEdge],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
        events: list[dict[str, Any]] = []
        current_time = 0.0
        for edge_index, edge in enumerate(edges):
            source = states[edge_index]
            target = states[edge_index + 1]
            mode: TeamMode = (
                "assembled_drive"
                if edge.action in {"drive", "land"}
                else "disassembled_flight"
            )
            if events and events[-1]["action"] == edge.action and edge.action in {"drive", "fly"}:
                event = events[-1]
                event["energy_wh"] += edge.energy_wh
                event["duration_s"] += edge.duration_s
                event["end_time_s"] += edge.duration_s
                event["path"].append(self._path_point(target, edge.action))
                for name, energy in zip(self._AGENT_NAMES, edge.agent_energy_wh):
                    event["agent_energy_wh"][name] += float(energy)
            else:
                event = {
                    "event_index": len(events) + 1,
                    "action": edge.action,
                    "mode": mode,
                    "energy_wh": float(edge.energy_wh),
                    "duration_s": float(edge.duration_s),
                    "start_time_s": float(current_time),
                    "end_time_s": float(current_time + edge.duration_s),
                    "agent_energy_wh": {
                        name: float(energy)
                        for name, energy in zip(self._AGENT_NAMES, edge.agent_energy_wh)
                    },
                    "path": [
                        self._path_point(source, edge.action),
                        self._path_point(target, edge.action),
                    ],
                }
                if edge.action == "takeoff":
                    event["path"][0]["z"] = 0.0
                    event["path"][1]["z"] = self.config.cruise_altitude_m
                elif edge.action == "land":
                    event["path"][0]["z"] = self.config.cruise_altitude_m
                    event["path"][1]["z"] = 0.0
                events.append(event)
            current_time += edge.duration_s

        labels = {
            "drive": "Assembled five-module platform drive",
            "takeoff": "Collective disassembly and five-agent takeoff",
            "fly": "Disassembled five-agent synchronized flight",
            "land": "Five-agent landing and collective assembly",
        }
        timeline = [
            {
                "event_index": event["event_index"],
                "event_type": event["action"],
                "phase": event["action"],
                "label": labels[event["action"]],
                "start_time_s": event["start_time_s"],
                "end_time_s": event["end_time_s"],
            }
            for event in events
        ]
        return events, timeline, float(current_time)

    def _path_point(self, state: TeamState, action: str) -> dict[str, float]:
        x, y = self._world(state.ix, state.iy)
        point = {"x": x, "y": y, "theta": self._heading(state.heading_bin)}
        if action in {"takeoff", "fly", "land"}:
            point["z"] = self.config.cruise_altitude_m
        return point

    def _state_json(self, state: TeamState) -> dict[str, Any]:
        x, y = self._world(state.ix, state.iy)
        return {
            "x": x,
            "y": y,
            "theta": self._heading(state.heading_bin),
            "z": self.config.cruise_altitude_m if state.mode == "disassembled_flight" else 0.0,
            "mode": state.mode,
            "used_ground": state.used_ground,
            "used_air": state.used_air,
        }

    def _failure(
        self,
        started: float,
        reason: str,
        *,
        elapsed: float | None = None,
        expanded: int = 0,
        generated: int = 0,
        peak_open: int = 0,
    ) -> dict[str, Any]:
        return {
            "planner": "Unified Five-Agent Hybrid A*",
            "scenario": "",
            "success": False,
            "reason": reason,
            "objective_energy_wh": float("inf"),
            "solve_time_s": float(perf_counter() - started if elapsed is None else elapsed),
            "mission_events": [],
            "mission_timeline": [],
            "team_path": [],
            "search_statistics": {
                "expanded_states": expanded,
                "generated_states": generated,
                "peak_open_size": peak_open,
            },
        }

    def _world(self, ix: int, iy: int) -> tuple[float, float]:
        if (ix, iy) == self._start_sentinel:
            return self.home
        if (ix, iy) == self._target_sentinel:
            return self.target
        return (
            float(ix * self.config.grid_resolution_m),
            float(iy * self.config.grid_resolution_m),
        )

    def _grid_index(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(round(x / self.config.grid_resolution_m)),
            int(round(y / self.config.grid_resolution_m)),
        )

    def _inside_grid(self, ix: int, iy: int) -> bool:
        if not (0 <= ix < self.grid_width and 0 <= iy < self.grid_height):
            return False
        return self.terrain.contains(*self._world(ix, iy))

    def _neighbor_grid_nodes(
        self,
        state: TeamState,
    ) -> list[tuple[int, int, int, float]]:
        if (state.ix, state.iy) != self._start_sentinel:
            result = []
            x, y = self._world(state.ix, state.iy)
            for dx, dy in self._MOVES:
                next_ix = state.ix + dx
                next_iy = state.iy + dy
                if not self._inside_grid(next_ix, next_iy):
                    continue
                next_x, next_y = self._world(next_ix, next_iy)
                heading = self._heading_bin(atan2(next_y - y, next_x - x))
                result.append(
                    (next_ix, next_iy, heading, hypot(next_x - x, next_y - y))
                )
            return result

        result = []
        radius = 1.75 * self.config.grid_resolution_m
        for iy in range(self.grid_height):
            for ix in range(self.grid_width):
                x, y = self._world(ix, iy)
                dx = x - self.home[0]
                dy = y - self.home[1]
                distance = hypot(dx, dy)
                if not 1e-9 < distance <= radius:
                    continue
                result.append((ix, iy, self._heading_bin(atan2(dy, dx)), distance))
        return result

    def _heading(self, heading_bin: int) -> float:
        return wrap_angle(2.0 * pi * heading_bin / self.config.heading_bins)

    def _heading_bin(self, heading: float) -> int:
        return int(round(wrap_angle(heading) * self.config.heading_bins / (2.0 * pi))) % self.config.heading_bins

    def _validate(self) -> None:
        if self.config.grid_resolution_m <= 0.0 or self.config.connector_step_m <= 0.0:
            raise ValueError("team grid and connector resolutions must be positive")
        if self.config.heading_bins != len(self._MOVES):
            raise ValueError("the current unified lattice requires eight heading bins")
        if self.config.minimum_flight_leg_m <= 0.0:
            raise ValueError("minimum_flight_leg_m must be positive")
        if (
            self.config.maximum_flight_leg_m is not None
            and self.config.maximum_flight_leg_m < self.config.minimum_flight_leg_m
        ):
            raise ValueError("maximum_flight_leg_m must be at least minimum_flight_leg_m")
        if self.config.heuristic_weight <= 0.0:
            raise ValueError("heuristic_weight must be positive")
        if self.config.air_obstacle_clearance_m < 0.0:
            raise ValueError("air obstacle clearance must be nonnegative")
        if self.config.max_expansions <= 0:
            raise ValueError("max_expansions must be positive")


class _ContinuousTerrainAdapter:
    def __init__(self, terrain: TerrainMap) -> None:
        self.terrain = terrain

    def value(self, layer: str, x: float, y: float) -> float:
        return self.terrain.continuous_value(layer, x, y)


def _team_config(raw: dict[str, Any], config: dict[str, Any]) -> UnifiedTeamConfig:
    defaults = {
        "max_slope_deg": config.get("vehicle", {}).get("max_slope_deg", 15.0),
        "safe_landing_threshold": config.get("aerial", {}).get("safe_landing_threshold", 5.0),
        "cruise_altitude_m": config.get("aerial", {}).get("cruise_altitude_m", 5.0),
        "footprint_sample_step_m": config.get("continuous_planner", {}).get(
            "footprint_sample_step_m", 0.4
        ),
        **raw,
    }
    allowed = UnifiedTeamConfig.__dataclass_fields__.keys()
    return UnifiedTeamConfig(**{key: value for key, value in defaults.items() if key in allowed})


def run_unified_team_hybrid_experiment(
    scenario_name: str = "hybrid_challenge",
    time_limit_s: float = 300.0,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or load_config()
    scenario = make_unified_team_scenario(scenario_name)
    planner = UnifiedFiveAgentHybridAStar(
        terrain=scenario.terrain,
        home=scenario.start,
        target=scenario.goal,
        vehicle_cfg=config["vehicle"],
        platform_cfg=config["platform"],
        battery_cfg=config["batteries"],
        docking_energy_wh=float(config["continuous_planner"].get("docking_energy_wh", 0.8)),
        undocking_energy_wh=float(config["continuous_planner"].get("undocking_energy_wh", 0.2)),
        config=_team_config(config.get("unified_team_astar", {}), config),
    )
    result = planner.plan(float(time_limit_s))
    result["scenario"] = scenario_name
    result["home"] = {"x": scenario.start[0], "y": scenario.start[1], "mode": "assembled_drive"}
    result["target"] = {"x": scenario.goal[0], "y": scenario.goal[1], "mode": "assembled_drive"}
    return result


def _unique_stem(output_dir: Path, base: str) -> str:
    timestamped = f"{base}_{strftime('%Y%m%d_%H%M%S')}"
    stem = timestamped
    suffix = 1
    while any((output_dir / f"{stem}{extension}").exists() for extension in (".json", "_3d.html")):
        stem = f"{timestamped}_{suffix:02d}"
        suffix += 1
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plan one synchronized five-module team through assembled drive and disassembled flight modes.",
    )
    parser.add_argument("--scenario", default="hybrid_challenge")
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output", default=None, help="new JSON path; existing files are rejected")
    parser.add_argument(
        "--output-stem",
        default=None,
        help="stable filename stem for JSON and HTML; existing files are rejected",
    )
    parser.add_argument("--render", action="store_true", help="also write an interactive 3D HTML result")
    parser.add_argument("--allow-single-mode", action="store_true")
    parser.add_argument(
        "--grid-resolution",
        type=float,
        default=None,
        help="override the unified team lattice resolution in meters",
    )
    parser.add_argument(
        "--maximum-flight-leg",
        type=float,
        default=None,
        help="maximum distance in meters between collective takeoff and landing",
    )
    parser.add_argument(
        "--heuristic-weight",
        type=float,
        default=None,
        help="weighted-A* heuristic multiplier",
    )
    args = parser.parse_args()

    config = load_config()
    if args.allow_single_mode:
        config.setdefault("unified_team_astar", {})["require_ground_and_air"] = False
    if args.grid_resolution is not None:
        config.setdefault("unified_team_astar", {})["grid_resolution_m"] = args.grid_resolution
    if args.maximum_flight_leg is not None:
        config.setdefault("unified_team_astar", {})["maximum_flight_leg_m"] = args.maximum_flight_leg
    if args.heuristic_weight is not None:
        config.setdefault("unified_team_astar", {})["heuristic_weight"] = args.heuristic_weight
    result = run_unified_team_hybrid_experiment(args.scenario, args.time_limit, config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_stem is not None and Path(args.output_stem).name != args.output_stem:
        raise ValueError("output-stem must be a filename stem, not a path")
    stem = args.output_stem or _unique_stem(
        output_dir,
        f"mission_{args.scenario}_unified_five_agent",
    )
    result_path = Path(args.output) if args.output else output_dir / f"{stem}.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {result_path}")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {result_path}")

    if args.render:
        from .visualize_unified_team import export_unified_team_html3d

        html_path = output_dir / f"{stem}_3d.html"
        export_unified_team_html3d(result, html_path)
        print(f"wrote {html_path}")


if __name__ == "__main__":
    main()
