"""Team energy, terrain feasibility, and route reporting for the DP planners."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, pi
from typing import Any, Literal

from .platform import (
    Pose2D,
    RectFootprint,
    estimate_platform_roll_pitch,
    platform_collision,
    wrap_angle,
)
from .rover import ground_speed_and_power, mass_scaled_flight_power
from .terrain import TerrainMap


TeamMode = Literal["assembled_drive", "disassembled_flight"]


@dataclass(frozen=True)
class TeamConfig:
    """Physical and lattice parameters shared by the team DP planners."""

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


class TeamEnergyModel:
    """Costs and feasibility for a synchronized team of five modules.

    The assembled platform follows one ground centerline. During flight, the
    four rover-drone modules and drill share one team centerline. Edge energy
    sums all five batteries; the DP planners choose the route.
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
        config: TeamConfig,
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
            raise ValueError("the team energy model requires exactly four rover-drone modules")
        self.platform_footprint = RectFootprint(**platform_cfg["platform_footprint"])
        self.rover_footprint = RectFootprint(**platform_cfg["rover_footprint"])
        self.rover_mass_kg = float(vehicle_cfg["mass_kg"])
        self.drill_mass_kg = float(vehicle_cfg.get("drill_mass_kg", 3.5))
        self.loaded_rover_mass_kg = self.rover_mass_kg + self.drill_mass_kg / 4.0
        self.grid_width = int(round((terrain.width - 1) * terrain.resolution / config.grid_resolution_m)) + 1
        self.grid_height = int(round((terrain.height - 1) * terrain.resolution / config.grid_resolution_m)) + 1
        self._start_sentinel = (-1, -1)
        self._target_sentinel = (self.grid_width, self.grid_height)
        self._validate()
        self._cached_takeoff_edge = self._make_takeoff_edge()
        self._cached_landing_edge = self._make_landing_edge()

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
            "planner": "Air-Ground Dynamic Programming",
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
                "optimal_on_finite_hybrid_lattice": True,
                "grid_resolution_m": self.config.grid_resolution_m,
                "heading_bins": self.config.heading_bins,
                "minimum_flight_leg_m": self.config.minimum_flight_leg_m,
                "maximum_flight_leg_m": self.config.maximum_flight_leg_m,
            },
            "optimality_claim": {
                "finite_run": "minimum summed five-agent energy on the configured finite hybrid DP lattice",
                "continuous_space": "no global-optimality claim outside the configured grid, headings, and synchronized team abstraction",
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

    def _world(self, ix: int, iy: int) -> tuple[float, float]:
        if (ix, iy) == self._start_sentinel:
            return self.home
        if (ix, iy) == self._target_sentinel:
            return self.target
        return (
            float(ix * self.config.grid_resolution_m),
            float(iy * self.config.grid_resolution_m),
        )

    def _heading(self, heading_bin: int) -> float:
        return wrap_angle(2.0 * pi * heading_bin / self.config.heading_bins)

    def _heading_bin(self, heading: float) -> int:
        return int(round(wrap_angle(heading) * self.config.heading_bins / (2.0 * pi))) % self.config.heading_bins

    def _validate(self) -> None:
        if self.config.grid_resolution_m <= 0.0 or self.config.connector_step_m <= 0.0:
            raise ValueError("team grid and connector resolutions must be positive")
        if self.config.heading_bins != len(self._MOVES):
            raise ValueError("the team DP lattice requires eight heading bins")
        if self.config.minimum_flight_leg_m <= 0.0:
            raise ValueError("minimum_flight_leg_m must be positive")
        if (
            self.config.maximum_flight_leg_m is not None
            and self.config.maximum_flight_leg_m < self.config.minimum_flight_leg_m
        ):
            raise ValueError("maximum_flight_leg_m must be at least minimum_flight_leg_m")
        if self.config.air_obstacle_clearance_m < 0.0:
            raise ValueError("air obstacle clearance must be nonnegative")


class _ContinuousTerrainAdapter:
    def __init__(self, terrain: TerrainMap) -> None:
        self.terrain = terrain

    def value(self, layer: str, x: float, y: float) -> float:
        return self.terrain.continuous_value(layer, x, y)


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
