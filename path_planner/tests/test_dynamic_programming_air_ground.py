from __future__ import annotations

from pathlib import Path

import numpy as np

from script.config import load_config
from script.dynamic_programming_air_ground import (
    InitialCondition,
    MultiStartAirGroundDP,
    TwoLayerAirGroundDP,
    _full_dp_team_config,
    export_cost_to_go_layers_html,
    load_case_1_terrain,
)
from script.terrain import TerrainMap


def test_case_1_html_map_matches_source_terrain() -> None:
    terrain = load_case_1_terrain(Path(__file__).resolve().parents[1] / "maps" / "hybrid_challenge.html")

    assert terrain.elevation.shape == (31, 56)
    assert terrain.obstacle.any()
    assert terrain.unsafe_landing.any()


def test_two_layer_dp_supports_multiple_ground_initial_conditions() -> None:
    shape = (11, 15)
    obstacle = np.zeros(shape, dtype=bool)
    obstacle[:, 7] = True
    terrain = TerrainMap(
        elevation=np.zeros(shape),
        roughness=np.zeros(shape),
        obstacle=obstacle,
        unsafe_landing=np.zeros(shape, dtype=bool),
    )
    planner = TwoLayerAirGroundDP(
        terrain=terrain,
        target=(12.0, 5.0),
        initial_conditions=[InitialCondition(2.0, 5.0), InitialCondition(4.0, 5.0)],
        config=load_config(),
    )

    result = planner.solve()

    assert result["all_succeeded"]
    assert result["initial_condition_count"] == 2
    assert result["boundary_conditions"]["initial_mode"] == "ground only"
    assert result["boundary_conditions"]["goal_mode"] == "ground only"
    assert result["shared_dp_statistics"]["single_value_function_reused"] is True
    assert result["shared_dp_statistics"]["full_reachable_value_function"] is True
    assert set(result["hybrid_cost_to_go"]) >= {"ground", "air"}
    for item in result["results"]:
        assert item["success"]
        assert np.isclose(item["objective_energy_wh"], item["dp_cost_to_go_wh"])
        assert item["team_path"][0]["mode"] == "assembled_drive"
        assert item["team_path"][-1]["mode"] == "assembled_drive"
        assert {event["action"] for event in item["mission_events"]} >= {"takeoff", "fly", "land"}
        assert item["terminal_team_state"]["drill_attached"] is True


def test_full_dp_requires_ground_and_allows_one_grid_edge_flight_leg() -> None:
    shape = (11, 15)
    obstacle = np.zeros(shape, dtype=bool)
    obstacle[:, 7] = True
    terrain = TerrainMap(
        elevation=np.zeros(shape),
        roughness=np.zeros(shape),
        obstacle=obstacle,
        unsafe_landing=np.zeros(shape, dtype=bool),
    )
    config = load_config()
    team_config = _full_dp_team_config(terrain, config)
    planner = MultiStartAirGroundDP(
        terrain=terrain,
        target=(12.0, 5.0),
        initial_conditions=[InitialCondition(2.0, 5.0)],
        config=config,
        team_config=team_config,
    )

    result = planner.solve()

    assert result["all_succeeded"]
    assert result["lattice"]["require_ground_and_air"] is True
    assert result["lattice"]["minimum_flight_leg_m"] == terrain.resolution
    assert "used-mode flags" in result["hybrid_cost_to_go"]["internal_state_note"]
    actions = [event["action"] for event in result["results"][0]["mission_events"]]
    assert "drive" in actions
    assert {"takeoff", "fly", "land"} <= set(actions)
    path = result["results"][0]["team_path"]
    flight_distance = sum(
        float(np.hypot(end["x"] - start["x"], end["y"] - start["y"]))
        for start, end in zip(path, path[1:])
        if start["mode"] == end["mode"] == "disassembled_flight"
    )
    assert flight_distance >= team_config.minimum_flight_leg_m


def test_full_dp_ground_only_never_takes_off() -> None:
    shape = (11, 15)
    terrain = TerrainMap(
        elevation=np.zeros(shape),
        roughness=np.zeros(shape),
        obstacle=np.zeros(shape, dtype=bool),
        unsafe_landing=np.zeros(shape, dtype=bool),
    )
    config = load_config()
    team_config = _full_dp_team_config(
        terrain,
        config,
        require_ground_and_air=False,
    )
    planner = MultiStartAirGroundDP(
        terrain=terrain,
        target=(12.0, 5.0),
        initial_conditions=[InitialCondition(2.0, 5.0)],
        config=config,
        team_config=team_config,
        ground_only=True,
    )

    result = planner.solve()

    assert result["all_succeeded"]
    assert result["lattice"]["ground_only"] is True
    actions = [event["action"] for event in result["results"][0]["mission_events"]]
    assert actions == ["drive"]
    assert "takeoff" not in actions


def test_full_dp_single_mode_does_not_force_flight_on_flat_ground() -> None:
    shape = (11, 15)
    terrain = TerrainMap(
        elevation=np.zeros(shape),
        roughness=np.zeros(shape),
        obstacle=np.zeros(shape, dtype=bool),
        unsafe_landing=np.zeros(shape, dtype=bool),
    )
    config = load_config(overrides={"vehicle": {"max_slope_deg": 8.0}})
    team_config = _full_dp_team_config(
        terrain,
        config,
        require_ground_and_air=False,
    )
    planner = MultiStartAirGroundDP(
        terrain=terrain,
        target=(12.0, 5.0),
        initial_conditions=[InitialCondition(2.0, 5.0)],
        config=config,
        team_config=team_config,
    )

    result = planner.solve()

    assert result["all_succeeded"]
    assert result["lattice"]["require_ground_and_air"] is False
    assert result["lattice"]["max_ground_slope_deg"] == 8.0
    actions = [event["action"] for event in result["results"][0]["mission_events"]]
    assert actions == ["drive"]


def test_full_dp_rejects_air_edges_blocked_by_tall_surface_objects() -> None:
    shape = (11, 15)
    terrain = TerrainMap(
        elevation=np.zeros(shape),
        roughness=np.zeros(shape),
        obstacle=np.zeros(shape, dtype=bool),
        unsafe_landing=np.zeros(shape, dtype=bool),
    )
    terrain.surface_height_m = np.zeros(shape)
    terrain.surface_height_m[:, 7] = 4.25
    config = load_config()
    team_config = _full_dp_team_config(terrain, config)
    planner = MultiStartAirGroundDP(
        terrain=terrain,
        target=(12.0, 5.0),
        initial_conditions=[InitialCondition(2.0, 5.0)],
        config=config,
        team_config=team_config,
    )

    for source, target in planner.air_edges:
        source_x = planner.positions[source][0]
        target_x = planner.positions[target][0]
        assert not (min(source_x, target_x) <= 7.0 <= max(source_x, target_x))

    assert np.isclose(
        team_config.cruise_altitude_m - team_config.air_obstacle_clearance_m,
        4.25,
    )


def test_cost_to_go_html_handles_empty_display_layers(tmp_path) -> None:
    shape = (11, 15)
    terrain = TerrainMap(
        elevation=np.zeros(shape),
        roughness=np.zeros(shape),
        obstacle=np.zeros(shape, dtype=bool),
        unsafe_landing=np.ones(shape, dtype=bool),
    )
    terrain.surface_height_m = np.zeros(shape)
    config = load_config()
    team_config = _full_dp_team_config(terrain, config)
    planner = MultiStartAirGroundDP(
        terrain=terrain,
        target=(12.0, 5.0),
        initial_conditions=[InitialCondition(2.0, 5.0)],
        config=config,
        team_config=team_config,
    )
    result = planner.solve()
    output = tmp_path / "empty_layers.html"

    export_cost_to_go_layers_html(result, terrain, output)

    assert output.is_file()
    assert output.stat().st_size > 0
