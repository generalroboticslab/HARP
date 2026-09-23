from __future__ import annotations

import numpy as np

from .terrain import TerrainMap


def mass_scaled_flight_power(
    nominal_power_w: float,
    mass_kg: float,
    vehicle_cfg: dict,
) -> float:
    """Scale a measured reference power with a rotorcraft mass model.

    The aerodynamic share follows induced-power scaling ``P ~ m^(3/2)``.
    A configurable fixed share represents avionics and other approximately
    mass-independent loads. Nominal powers are calibrated at
    ``flight_reference_mass_kg``.
    """
    reference_mass = max(
        float(vehicle_cfg.get("flight_reference_mass_kg", vehicle_cfg["mass_kg"])),
        1e-9,
    )
    exponent = float(vehicle_cfg.get("flight_power_mass_exponent", 1.5))
    fixed_fraction = float(vehicle_cfg.get("flight_fixed_power_fraction", 0.0))
    if exponent <= 0.0:
        raise ValueError("flight_power_mass_exponent must be positive")
    if not 0.0 <= fixed_fraction <= 1.0:
        raise ValueError("flight_fixed_power_fraction must be in [0, 1]")
    ratio = max(float(mass_kg), 1e-9) / reference_mass
    scale = fixed_fraction + (1.0 - fixed_fraction) * ratio**exponent
    return float(nominal_power_w) * float(scale)


def ground_speed_and_power(
    terrain: TerrainMap,
    x: float,
    y: float,
    vehicle_cfg: dict,
    directional_grade: float = 0.0,
    mass_kg: float | None = None,
) -> tuple[float, float]:
    """Return ground speed and battery power from a quasi-static force model.

    The required wheel force is rolling resistance plus the signed gravity
    component along the travel direction. Negative force is treated as passive
    coasting/braking; regenerative braking is not modeled.
    """
    rough = terrain.value("roughness", x, y)
    mass = float(vehicle_cfg["mass_kg"] if mass_kg is None else mass_kg)
    gravity = 9.80665
    theta = float(np.arctan(directional_grade))
    c_rr = (
        float(vehicle_cfg["rolling_resistance_coefficient"])
        + float(vehicle_cfg["roughness_rolling_resistance_gain"]) * rough
    )
    rolling_force_n = mass * gravity * c_rr * np.cos(theta)
    grade_force_n = mass * gravity * np.sin(theta)
    required_traction_n = max(0.0, rolling_force_n + grade_force_n)

    speed = float(vehicle_cfg["ground_max_speed_mps"]) * (
        1.0 - float(vehicle_cfg["roughness_speed_loss"]) * rough
    )
    speed = max(float(vehicle_cfg["min_ground_speed_mps"]), speed)
    mechanical_limit_w = float(vehicle_cfg["max_drive_mechanical_power_w"])
    if required_traction_n > 1e-9:
        speed = min(speed, mechanical_limit_w / required_traction_n)
    speed = max(float(vehicle_cfg["min_ground_speed_mps"]), speed)

    drivetrain_efficiency = max(float(vehicle_cfg["drivetrain_efficiency"]), 1e-6)
    wheel_power_w = required_traction_n * speed
    battery_power_w = (
        float(vehicle_cfg["ground_electronics_power_w"])
        + wheel_power_w / drivetrain_efficiency
    )
    return float(speed), float(battery_power_w)
