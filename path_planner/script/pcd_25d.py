from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, grey_opening, median_filter

from .terrain import TerrainMap


PCD_TYPE_MAP = {
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    ("I", 1): "<i1",
    ("I", 2): "<i2",
    ("I", 4): "<i4",
    ("I", 8): "<i8",
    ("U", 1): "<u1",
    ("U", 2): "<u2",
    ("U", 4): "<u4",
    ("U", 8): "<u8",
}


def load_planner_terrain(
    path: str | Path,
    *,
    roughness_scale_m: float = 0.30,
) -> tuple[TerrainMap, dict[str, Any]]:
    """Load a generated ``map_25d.npz`` into the planner's south-up convention."""
    source = Path(path)
    if roughness_scale_m <= 0.0:
        raise ValueError("roughness_scale_m must be positive")
    required = {
        "elevation_local_m",
        "surface_height_m",
        "roughness_m",
        "observed_mask",
        "obstacle",
        "unsafe_landing",
        "metadata_json",
    }
    with np.load(source, allow_pickle=False) as payload:
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"2.5D map is missing arrays: {sorted(missing)}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        arrays = {
            name: np.flipud(np.asarray(payload[name])).copy()
            for name in required.difference({"metadata_json"})
        }
        # Keep the measured top envelope available for visualization even when
        # the planner was configured to ignore surface height for obstacles.
        # Older maps do not contain this optional array, so fall back to the
        # planner-effective surface height in that case.
        measured_surface_height = (
            np.flipud(np.asarray(payload["measured_surface_height_m"])).copy()
            if "measured_surface_height_m" in payload.files
            else arrays["surface_height_m"].copy()
        )
        ground_elevation = (
            np.flipud(np.asarray(payload["ground_elevation_local_m"])).copy()
            if "ground_elevation_local_m" in payload.files
            else arrays["elevation_local_m"].copy()
        )
    shape = arrays["elevation_local_m"].shape
    if len(shape) != 2 or any(array.shape != shape for array in arrays.values()):
        raise ValueError("2.5D map layers must be identically shaped 2-D arrays")
    resolution = float(metadata["resolution_m"])
    if resolution <= 0.0:
        raise ValueError("2.5D map resolution must be positive")

    observed = np.asarray(arrays["observed_mask"], dtype=bool)
    terrain = TerrainMap(
        elevation=np.asarray(arrays["elevation_local_m"], dtype=float),
        roughness=np.clip(np.asarray(arrays["roughness_m"], dtype=float) / roughness_scale_m, 0.0, 1.0),
        # A filled elevation in an unobserved cell is only an interpolation aid.
        # It must not become traversable ground in the planner.
        obstacle=np.asarray(arrays["obstacle"], dtype=bool) | ~observed,
        unsafe_landing=np.asarray(arrays["unsafe_landing"], dtype=bool),
        resolution=resolution,
    )
    terrain.surface_height_m = np.asarray(arrays["surface_height_m"], dtype=float)
    terrain.measured_surface_height_m = np.asarray(measured_surface_height, dtype=float)
    terrain.ground_elevation_m = np.asarray(ground_elevation, dtype=float)
    terrain.surface_observed_mask = observed
    terrain.surface_unknown = ~terrain.surface_observed_mask
    terrain.map_25d_metadata = metadata
    return terrain, metadata


def convert_pcd(
    pcd: str | Path,
    output_dir: str | Path,
    *,
    invert_z: bool = True,
    bounds_xy: tuple[float, float, float, float] | None = None,
    ignore_surface_height: bool = False,
    resolution_m: float = 0.5,
    padding_m: float = 2.0,
    crop_quantile: float = 0.001,
    minimum_points_per_cell: int = 3,
    z_bin_m: float = 0.10,
    ground_quantile: float = 0.10,
    surface_quantile: float = 0.95,
    obstacle_height_m: float = 1.5,
    obstacle_slope_deg: float = 28.0,
    landing_height_m: float = 0.5,
    landing_roughness_m: float = 0.30,
    landing_slope_deg: float = 8.0,
    planner_elevation: str = "ground",
) -> dict[str, Any]:
    """Convert a FAST-LIO/PCD point cloud into DP-ready 2.5D raster layers."""
    source = Path(pcd)
    if not source.is_file():
        raise FileNotFoundError(source)
    if resolution_m <= 0.0 or padding_m < 0.0:
        raise ValueError("resolution_m must be positive and padding_m non-negative")
    if not 0.0 <= crop_quantile < 0.5:
        raise ValueError("crop_quantile must be in [0.0, 0.5)")
    if bounds_xy is not None:
        x_min_bound, x_max_bound, y_min_bound, y_max_bound = bounds_xy
        if x_min_bound >= x_max_bound or y_min_bound >= y_max_bound:
            raise ValueError("bounds_xy must satisfy x_min < x_max and y_min < y_max")
    if minimum_points_per_cell < 1:
        raise ValueError("minimum_points_per_cell must be at least 1")
    if planner_elevation not in {"ground", "surface"}:
        raise ValueError("planner_elevation must be 'ground' or 'surface'")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cloud, header = read_pcd_xyz(source)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    if len(cloud) == 0:
        raise RuntimeError("PCD contains no finite xyz points")
    if invert_z:
        cloud[:, 2] *= -1.0

    if bounds_xy is None:
        low = np.quantile(cloud[:, :2], crop_quantile, axis=0)
        high = np.quantile(cloud[:, :2], 1.0 - crop_quantile, axis=0)
        x_min = float(np.floor((float(low[0]) - padding_m) / resolution_m) * resolution_m)
        x_max = float(np.ceil((float(high[0]) + padding_m) / resolution_m) * resolution_m)
        y_min = float(np.floor((float(low[1]) - padding_m) / resolution_m) * resolution_m)
        y_max = float(np.ceil((float(high[1]) + padding_m) / resolution_m) * resolution_m)
    else:
        x_min, x_max, y_min, y_max = (float(value) for value in bounds_xy)
        low = np.asarray([x_min, y_min])
        high = np.asarray([x_max, y_max])
    inside_crop = (
        (cloud[:, 0] >= low[0])
        & (cloud[:, 0] <= high[0])
        & (cloud[:, 1] >= low[1])
        & (cloud[:, 1] <= high[1])
    )
    cloud = cloud[inside_crop]
    columns = int(round((x_max - x_min) / resolution_m)) + 1
    rows = int(round((y_max - y_min) / resolution_m)) + 1

    ix = np.floor((cloud[:, 0] - x_min) / resolution_m + 0.5).astype(np.int64)
    iy = np.floor((cloud[:, 1] - y_min) / resolution_m + 0.5).astype(np.int64)
    inside_grid = (ix >= 0) & (ix < columns) & (iy >= 0) & (iy < rows)
    ix = ix[inside_grid]
    iy = iy[inside_grid]
    z = cloud[inside_grid, 2].astype(np.float32, copy=False)
    flat = iy * columns + ix

    point_count, q10, q25, q75, q95 = _cell_quantiles(
        flat,
        z,
        rows * columns,
        (ground_quantile, 0.25, 0.75, surface_quantile),
        z_bin_m=z_bin_m,
    )
    point_count = point_count.reshape(rows, columns)
    observed = point_count >= minimum_points_per_cell
    q10 = q10.reshape(rows, columns)
    q25 = q25.reshape(rows, columns)
    q75 = q75.reshape(rows, columns)
    q95 = q95.reshape(rows, columns)

    preliminary_ground = _fill_nearest(np.where(observed, q10, np.nan))
    opening_cells = max(3, int(round(5.0 / resolution_m)) | 1)
    ground = gaussian_filter(
        median_filter(grey_opening(preliminary_ground, size=(opening_cells, opening_cells)), size=3),
        sigma=max(0.6, 0.6 / resolution_m),
    ).astype(np.float32)
    surface = _fill_nearest(np.where(observed, q95, np.nan)).astype(np.float32)
    surface_height = np.maximum(0.0, surface - ground).astype(np.float32)
    planner_surface_height = (
        np.zeros_like(surface_height) if ignore_surface_height else surface_height
    )
    roughness = np.maximum(0.0, q75 - q25)
    roughness = np.where(np.isfinite(roughness), roughness, 0.0).astype(np.float32)
    gy, gx = np.gradient(ground, resolution_m)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    obstacle = (
        (planner_surface_height >= obstacle_height_m) | (slope_deg > obstacle_slope_deg)
    ) & observed
    unsafe_landing = (
        (~observed)
        | (planner_surface_height >= landing_height_m)
        | (roughness >= landing_roughness_m)
        | (slope_deg >= landing_slope_deg)
    )
    planner_height = ground if planner_elevation == "ground" else surface

    metadata = {
        "source": "FAST-LIO2 accumulated PCD point cloud",
        "pcd_path": str(source),
        "pcd_header": header,
        "coordinate_transform": {
            "invert_z": bool(invert_z),
            "reason": "compensate for an upside-down LiDAR mount" if invert_z else "none",
        },
        "requested_bounds_xy_m": list(bounds_xy) if bounds_xy is not None else None,
        "ignore_surface_height": bool(ignore_surface_height),
        "resolution_m": float(resolution_m),
        "rows": int(rows),
        "columns": int(columns),
        "size_x_m": float((columns - 1) * resolution_m),
        "size_y_m": float((rows - 1) * resolution_m),
        "local_enu_grid_origin_m": [x_min, y_min],
        "array_orientation": "north-up: row 0 is maximum y/northing; load_planner_terrain flips for TerrainMap",
        "input_point_count": int(header.get("points", len(cloud))),
        "accepted_point_count": int(len(z)),
        "observed_cell_count": int(observed.sum()),
        "unknown_cells_are_unsafe": True,
        "planner_elevation": planner_elevation,
        "estimator": {
            "ground": f"{ground_quantile:.2f} z quantile, 5 m grey opening, median, Gaussian smoothing",
            "surface": f"{surface_quantile:.2f} z quantile",
            "roughness": "interquartile z range q75 - q25",
            "z_bin_m": float(z_bin_m),
            "minimum_points_per_cell": int(minimum_points_per_cell),
            "crop_quantile": float(crop_quantile),
            "padding_m": float(padding_m),
            "note": (
                "surface mode saves the top-envelope 2.5D surface as elevation_local_m; "
                "ground mode saves the smoothed low-return ground estimate"
            ),
        },
        "thresholds": {
            "obstacle_height_m": float(obstacle_height_m),
            "obstacle_slope_deg": float(obstacle_slope_deg),
            "landing_height_m": float(landing_height_m),
            "landing_roughness_m": float(landing_roughness_m),
            "landing_slope_deg": float(landing_slope_deg),
        },
    }

    north_up = lambda array: np.flipud(array)
    np.savez_compressed(
        destination / "map_25d.npz",
        elevation_m=north_up(planner_height),
        elevation_local_m=north_up(planner_height),
        ground_elevation_local_m=north_up(ground),
        surface_elevation_local_m=north_up(surface),
        surface_height_m=north_up(planner_surface_height),
        measured_surface_height_m=north_up(surface_height),
        roughness_m=north_up(roughness),
        slope_deg=north_up(slope_deg),
        point_count=north_up(point_count.astype(np.uint32)),
        observed_mask=north_up(observed),
        obstacle=north_up(obstacle),
        unsafe_landing=north_up(unsafe_landing),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    np.savez_compressed(
        destination / "terrain.npz",
        elevation_m=north_up(planner_height),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_preview(
        destination / "map_25d_preview.png",
        ground,
        surface_height,
        point_count,
        unsafe_landing,
        (x_min, x_max, y_min, y_max),
    )
    return metadata


def read_pcd_xyz(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read xyz from an ASCII or binary PCD file without requiring PCL/Open3D."""
    source = Path(path)
    header, data_offset = _read_pcd_header(source)
    fields = header["fields"]
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError("PCD must contain x, y, and z fields")
    if header["data"] == "ascii":
        rows = np.loadtxt(source, comments="#", skiprows=header["line_count"], dtype=np.float32)
        rows = np.atleast_2d(rows)
        return rows[:, [fields.index("x"), fields.index("y"), fields.index("z")]], header
    if header["data"] != "binary":
        raise ValueError(f"unsupported PCD DATA mode: {header['data']!r}")

    dtype = _pcd_dtype(header)
    with source.open("rb") as stream:
        stream.seek(data_offset)
        payload = np.fromfile(stream, dtype=dtype, count=int(header["points"]))
    xyz = np.column_stack((payload["x"], payload["y"], payload["z"])).astype(np.float32, copy=False)
    return xyz, header


def _read_pcd_header(path: Path) -> tuple[dict[str, Any], int]:
    header: dict[str, Any] = {}
    line_count = 0
    with path.open("rb") as stream:
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("PCD header ended before DATA line")
            line_count += 1
            text = raw.decode("ascii", errors="strict").strip()
            if not text or text.startswith("#"):
                continue
            key, *values = text.split()
            key = key.lower()
            if key == "data":
                header[key] = values[0].lower()
                break
            header[key] = values
        data_offset = stream.tell()
    fields = header.get("fields")
    if not fields:
        raise ValueError("PCD header is missing FIELDS")
    field_count = len(fields)
    header["fields"] = fields
    header["size"] = _header_ints(header, "size", field_count, 4)
    header["type"] = header.get("type", ["F"] * field_count)
    header["count"] = _header_ints(header, "count", field_count, 1)
    header["width"] = int(header.get("width", [0])[0])
    header["height"] = int(header.get("height", [1])[0])
    header["points"] = int(header.get("points", [header["width"] * header["height"]])[0])
    header["line_count"] = line_count
    return header, data_offset


def _header_ints(header: dict[str, Any], key: str, count: int, default: int) -> list[int]:
    values = header.get(key)
    if values is None:
        return [default] * count
    parsed = [int(value) for value in values]
    if len(parsed) != count:
        raise ValueError(f"PCD {key.upper()} length does not match FIELDS")
    return parsed


def _pcd_dtype(header: dict[str, Any]) -> np.dtype:
    items = []
    for field, size, type_code, count in zip(
        header["fields"],
        header["size"],
        header["type"],
        header["count"],
    ):
        scalar = PCD_TYPE_MAP.get((type_code.upper(), int(size)))
        if scalar is None:
            raise ValueError(f"unsupported PCD field type: {field} {type_code} {size}")
        shape = (int(count),) if int(count) > 1 else ()
        items.append((field, scalar, shape))
    return np.dtype(items)


def _cell_quantiles(
    flat: np.ndarray,
    z: np.ndarray,
    cell_count: int,
    quantiles: tuple[float, ...],
    *,
    z_bin_m: float,
) -> tuple[np.ndarray, ...]:
    if z_bin_m <= 0.0:
        raise ValueError("z_bin_m must be positive")
    order = np.lexsort((z, flat))
    sorted_flat = flat[order]
    sorted_z = z[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_flat)) + 1]
    ends = np.r_[starts[1:], len(sorted_flat)]
    cells = sorted_flat[starts]
    counts = np.zeros(cell_count, dtype=np.uint32)
    outputs = [np.full(cell_count, np.nan, dtype=np.float32) for _ in quantiles]
    for start, end, cell in zip(starts, ends, cells):
        count = int(end - start)
        counts[cell] = count
        values = sorted_z[start:end]
        for output, quantile in zip(outputs, quantiles):
            index = int(np.clip(round(quantile * (count - 1)), 0, count - 1))
            output[cell] = float(np.round(float(values[index]) / z_bin_m) * z_bin_m)
    return (counts, *outputs)


def _fill_nearest(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if not valid.any():
        raise RuntimeError("no populated cells remain in the 2.5D raster")
    indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return values[tuple(indices)]


def _write_preview(
    path: Path,
    elevation: np.ndarray,
    surface_height: np.ndarray,
    point_count: np.ndarray,
    unsafe: np.ndarray,
    extent: tuple[float, float, float, float],
) -> None:
    import os

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mod_algo_matplotlib")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    panels = (
        (elevation, "Ground elevation (local m)", "terrain"),
        (surface_height, "Surface height above ground (m)", "viridis"),
        (np.log1p(point_count), "log(1 + point count)", "magma"),
        (unsafe.astype(float), "Unsafe / unknown landing cells", "gray_r"),
    )
    for axis, (values, title, cmap) in zip(axes.ravel(), panels):
        image = axis.imshow(values, origin="lower", extent=extent, cmap=cmap, aspect="equal")
        axis.set(title=title, xlabel="local map x (m)", ylabel="local map y (m)")
        figure.colorbar(image, ax=axis, shrink=0.82)
    figure.suptitle("FAST-LIO2 PCD 2.5D map diagnostic")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a FAST-LIO2 PCD map to a DP-ready 2.5D map")
    parser.add_argument("pcd", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--invert-z",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="negate PCD z coordinates (default; use --no-invert-z to disable)",
    )
    parser.add_argument("--x-min", type=float)
    parser.add_argument("--x-max", type=float)
    parser.add_argument("--y-min", type=float)
    parser.add_argument("--y-max", type=float)
    parser.add_argument(
        "--ignore-surface-height",
        action="store_true",
        help="exclude trees and other above-ground returns from planner obstacles and air clearance",
    )
    parser.add_argument("--resolution-m", type=float, default=0.5)
    parser.add_argument("--padding-m", type=float, default=2.0)
    parser.add_argument("--crop-quantile", type=float, default=0.001)
    parser.add_argument("--minimum-points-per-cell", type=int, default=3)
    parser.add_argument("--obstacle-height-m", type=float, default=1.5)
    parser.add_argument("--obstacle-slope-deg", type=float, default=28.0)
    parser.add_argument("--landing-height-m", type=float, default=0.5)
    parser.add_argument("--landing-roughness-m", type=float, default=0.30)
    parser.add_argument("--landing-slope-deg", type=float, default=8.0)
    parser.add_argument(
        "--planner-elevation",
        choices=("ground", "surface"),
        default="ground",
        help="choose which 2.5D layer is exposed to TerrainMap as elevation_local_m",
    )
    args = parser.parse_args()
    bounds_values = (args.x_min, args.x_max, args.y_min, args.y_max)
    if any(value is not None for value in bounds_values) and not all(
        value is not None for value in bounds_values
    ):
        parser.error("--x-min, --x-max, --y-min, and --y-max must be supplied together")
    bounds_xy = (
        tuple(float(value) for value in bounds_values)
        if all(value is not None for value in bounds_values)
        else None
    )
    result = convert_pcd(
        args.pcd,
        args.output_dir,
        invert_z=args.invert_z,
        bounds_xy=bounds_xy,
        ignore_surface_height=args.ignore_surface_height,
        resolution_m=args.resolution_m,
        padding_m=args.padding_m,
        crop_quantile=args.crop_quantile,
        minimum_points_per_cell=args.minimum_points_per_cell,
        obstacle_height_m=args.obstacle_height_m,
        obstacle_slope_deg=args.obstacle_slope_deg,
        landing_height_m=args.landing_height_m,
        landing_roughness_m=args.landing_roughness_m,
        landing_slope_deg=args.landing_slope_deg,
        planner_elevation=args.planner_elevation,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
