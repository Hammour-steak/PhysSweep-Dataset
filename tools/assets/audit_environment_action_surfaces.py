"""Audit clear ground-action regions in reviewed visual environments."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage

from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES = PROJECT_ROOT / "configs/scene_mesh_profiles.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts/environment_action_surface_audit_v1"


def gltf_to_blender(values: np.ndarray) -> np.ndarray:
    """Match Blender's glTF import transform: (x, y, z) -> (x, -z, y)."""

    return np.column_stack((values[:, 0], -values[:, 2], values[:, 1]))


def normalized_source_mesh(root: Path, asset: dict[str, Any]) -> trimesh.Trimesh:
    source = trimesh.load(root / str(asset["path"]), force="scene", process=False)
    mesh = source.to_mesh() if isinstance(source, trimesh.Scene) else source
    vertices = gltf_to_blender(np.asarray(mesh.vertices, dtype=np.float64))
    expected = np.asarray(asset["source_bbox_size"], dtype=np.float64)
    actual = np.ptp(vertices, axis=0)
    relative_error = np.abs(actual - expected) / np.maximum(expected, 1.0e-8)
    if float(relative_error.max()) > 0.005:
        raise ValueError(
            f"source bounds changed for {asset['asset_id']}: "
            f"expected={expected.tolist()} actual={actual.tolist()}"
        )

    contract = asset["collision_proxy"]["transform_contract"]
    bottom_center = np.asarray(contract["source_bottom_center"], dtype=np.float64)
    scale = float(contract["scale"])
    normalized = (vertices - bottom_center) * scale
    return trimesh.Trimesh(
        vertices=normalized,
        faces=np.asarray(mesh.faces, dtype=np.int64),
        process=False,
    )


def selector_matches(
    triangles: np.ndarray, selector: dict[str, Any]
) -> np.ndarray:
    axis = {"x": 0, "y": 1, "z": 2}[str(selector["axis"])]
    value = float(selector["value"])
    if str(selector["comparison"]) == "at_or_above":
        return triangles[:, :, axis].min(axis=1) >= value
    if str(selector["comparison"]) == "at_or_below":
        return triangles[:, :, axis].max(axis=1) <= value
    raise ValueError(f"unsupported face selector: {selector}")


def retained_face_mask(mesh: trimesh.Trimesh, asset: dict[str, Any]) -> np.ndarray:
    retained = np.ones(len(mesh.faces), dtype=bool)
    source_triangles = np.asarray(mesh.triangles, dtype=np.float64)
    contract = asset["collision_proxy"]["transform_contract"]
    scale = float(contract["scale"])
    bottom_center = np.asarray(contract["source_bottom_center"], dtype=np.float64)
    for selector in asset.get("source_space_face_exclusions", []):
        normalized_selector = dict(selector)
        axis = {"x": 0, "y": 1, "z": 2}[str(selector["axis"])]
        normalized_selector["value"] = (
            float(selector["value"]) - float(bottom_center[axis])
        ) * scale
        retained &= ~selector_matches(source_triangles, normalized_selector)
    return retained


def grid_geometry(
    triangles_xy: np.ndarray, resolution_m: float, padding_m: float
) -> tuple[np.ndarray, np.ndarray, int, int]:
    low = triangles_xy.reshape(-1, 2).min(axis=0) - padding_m
    high = triangles_xy.reshape(-1, 2).max(axis=0) + padding_m
    width = max(2, int(math.ceil((high[0] - low[0]) / resolution_m)) + 1)
    height = max(2, int(math.ceil((high[1] - low[1]) / resolution_m)) + 1)
    return low, high, width, height


def pixel_polygon(
    triangle_xy: np.ndarray,
    low: np.ndarray,
    resolution_m: float,
    height: int,
) -> list[tuple[float, float]]:
    result = []
    for x, y in triangle_xy:
        px = (float(x) - float(low[0])) / resolution_m
        py = height - 1 - (float(y) - float(low[1])) / resolution_m
        result.append((px, py))
    return result


def rasterize_triangles(
    triangles_xy: np.ndarray,
    low: np.ndarray,
    resolution_m: float,
    width: int,
    height: int,
) -> np.ndarray:
    image = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for triangle in triangles_xy:
        draw.polygon(
            pixel_polygon(triangle, low, resolution_m, height),
            fill=1,
        )
    return np.asarray(image, dtype=bool)


def local_xy_from_pixel(
    row: int,
    column: int,
    low: np.ndarray,
    resolution_m: float,
    height: int,
) -> list[float]:
    return [
        float(low[0]) + float(column) * resolution_m,
        float(low[1]) + float(height - 1 - row) * resolution_m,
    ]


def candidate_peaks(
    distance_m: np.ndarray,
    low: np.ndarray,
    resolution_m: float,
    count: int,
    suppression_radius_m: float,
) -> list[dict[str, Any]]:
    working = distance_m.copy()
    peaks: list[dict[str, Any]] = []
    radius_px = max(1, int(math.ceil(suppression_radius_m / resolution_m)))
    height = int(distance_m.shape[0])
    for _ in range(count):
        flat_index = int(np.argmax(working))
        clearance = float(working.flat[flat_index])
        if clearance <= 0.0:
            break
        row, column = np.unravel_index(flat_index, working.shape)
        peaks.append(
            {
                "anchor_xy_local_m": [
                    round(value, 6)
                    for value in local_xy_from_pixel(
                        int(row),
                        int(column),
                        low,
                        resolution_m,
                        height,
                    )
                ],
                "clear_radius_m": round(clearance, 6),
            }
        )
        yy, xx = np.ogrid[: working.shape[0], : working.shape[1]]
        suppressed = (yy - row) ** 2 + (xx - column) ** 2 <= radius_px**2
        working[suppressed] = 0.0
    return peaks


def diagnostic_image(
    path: Path,
    floor_mask: np.ndarray,
    visual_obstacle_mask: np.ndarray,
    proxy_obstacle_mask: np.ndarray,
    distance_m: np.ndarray,
    peaks: list[dict[str, Any]],
    low: np.ndarray,
    resolution_m: float,
    profile_id: str,
) -> None:
    height, width = floor_mask.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[floor_mask] = [70, 80, 90]
    obstacle_mask = visual_obstacle_mask | proxy_obstacle_mask
    free = floor_mask & ~obstacle_mask
    if np.any(free):
        normalized = np.clip(distance_m / max(float(distance_m.max()), 1.0e-8), 0.0, 1.0)
        rgb[free, 0] = (25 + normalized[free] * 30).astype(np.uint8)
        rgb[free, 1] = (85 + normalized[free] * 145).astype(np.uint8)
        rgb[free, 2] = (75 + normalized[free] * 70).astype(np.uint8)
    rgb[proxy_obstacle_mask & floor_mask] = [210, 65, 55]
    rgb[visual_obstacle_mask & ~proxy_obstacle_mask & floor_mask] = [235, 145, 35]
    image = Image.fromarray(rgb, mode="RGB")
    scale = max(1, int(math.ceil(900 / max(width, height))))
    image = image.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    for index, peak in enumerate(peaks):
        x, y = peak["anchor_xy_local_m"]
        column = (float(x) - float(low[0])) / resolution_m
        row = height - 1 - (float(y) - float(low[1])) / resolution_m
        radius = float(peak["clear_radius_m"]) / resolution_m
        bounds = [
            (column - radius) * scale,
            (row - radius) * scale,
            (column + radius) * scale,
            (row + radius) * scale,
        ]
        color = (30, 220, 255) if index == 0 else (245, 210, 45)
        draw.ellipse(bounds, outline=color, width=max(2, scale))
        draw.text((column * scale + 4, row * scale + 4), str(index + 1), fill=color)
    draw.rectangle((0, 0, image.width, 22), fill=(15, 18, 22))
    draw.text((6, 5), profile_id, fill=(240, 240, 240))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def audit_profile(
    root: Path,
    profile: dict[str, Any],
    output_root: Path,
    resolution_m: float,
    floor_half_band_m: float,
    clearance_height_m: float,
) -> dict[str, Any]:
    asset = profile["asset"]
    source = normalized_source_mesh(root, asset)
    floor_z = float(
        asset["collision_proxy"]["transform_contract"]["authoritative_floor_z_m"]
    )
    triangles = np.asarray(source.triangles, dtype=np.float64)
    normals = np.asarray(source.face_normals, dtype=np.float64)
    retained = retained_face_mask(source, asset)
    floor_faces = (
        retained
        & (np.abs(normals[:, 2]) >= 0.94)
        & (np.abs(triangles[:, :, 2].mean(axis=1) - floor_z) <= floor_half_band_m)
        & (np.ptp(triangles[:, :, 2], axis=1) <= floor_half_band_m)
    )
    if not np.any(floor_faces):
        raise ValueError(f"no reviewed floor faces found for {profile['id']}")
    floor_triangles_xy = triangles[floor_faces, :, :2]
    low, high, width, height = grid_geometry(
        floor_triangles_xy, resolution_m, padding_m=resolution_m * 2.0
    )
    floor_mask = rasterize_triangles(
        floor_triangles_xy, low, resolution_m, width, height
    )

    source_above_floor = triangles[:, :, 2].max(axis=1) > floor_z + 0.006
    source_below_clearance = (
        triangles[:, :, 2].min(axis=1) < floor_z + clearance_height_m
    )
    source_obstacles = (
        retained & ~floor_faces & source_above_floor & source_below_clearance
    )
    visual_obstacle_triangles = triangles[source_obstacles, :, :2]
    visual_obstacle_mask = rasterize_triangles(
        visual_obstacle_triangles, low, resolution_m, width, height
    )

    proxy = trimesh.load(
        root / str(asset["collision_proxy"]["path"]),
        force="mesh",
        process=False,
    )
    proxy_triangles = np.asarray(proxy.triangles, dtype=np.float64)
    above_floor = proxy_triangles[:, :, 2].max(axis=1) > floor_z + 0.02
    below_clearance = (
        proxy_triangles[:, :, 2].min(axis=1) < floor_z + clearance_height_m
    )
    proxy_obstacle_triangles = proxy_triangles[
        above_floor & below_clearance, :, :2
    ]
    proxy_obstacle_mask = rasterize_triangles(
        proxy_obstacle_triangles, low, resolution_m, width, height
    )
    obstacle_mask = visual_obstacle_mask | proxy_obstacle_mask
    valid = floor_mask & ~obstacle_mask
    distance_m = ndimage.distance_transform_edt(valid) * resolution_m
    peaks = candidate_peaks(
        distance_m,
        low,
        resolution_m,
        count=5,
        suppression_radius_m=0.6,
    )
    image_path = output_root / "diagnostics" / f"{profile['id']}.png"
    diagnostic_image(
        image_path,
        floor_mask,
        visual_obstacle_mask,
        proxy_obstacle_mask,
        distance_m,
        peaks,
        low,
        resolution_m,
        str(profile["id"]),
    )
    best_radius = float(peaks[0]["clear_radius_m"]) if peaks else 0.0
    return {
        "profile_id": str(profile["id"]),
        "asset_id": str(asset["asset_id"]),
        "status": "candidate" if best_radius >= 0.45 else "insufficient_clearance",
        "floor_z_local_m": round(floor_z, 6),
        "floor_face_count": int(np.count_nonzero(floor_faces)),
        "floor_raster_area_m2": round(
            float(np.count_nonzero(floor_mask)) * resolution_m**2, 6
        ),
        "visual_obstacle_face_count": int(len(visual_obstacle_triangles)),
        "proxy_obstacle_face_count": int(len(proxy_obstacle_triangles)),
        "obstacle_face_count": int(
            len(visual_obstacle_triangles) + len(proxy_obstacle_triangles)
        ),
        "grid": {
            "resolution_m": resolution_m,
            "bounds_min_xy_local_m": [round(float(value), 6) for value in low],
            "bounds_max_xy_local_m": [round(float(value), 6) for value in high],
            "shape": [height, width],
        },
        "candidates": peaks,
        "diagnostic_image": str(image_path.relative_to(root)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--profile-id", action="append", default=[])
    parser.add_argument("--resolution-m", type=float, default=0.04)
    parser.add_argument("--floor-half-band-m", type=float, default=0.012)
    parser.add_argument("--clearance-height-m", type=float, default=1.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = PROJECT_ROOT
    profiles_path = args.profiles.resolve()
    output_root = args.output.resolve()
    payload = load_json(profiles_path)
    requested = {str(value) for value in args.profile_id}
    profiles = [
        profile
        for profile in payload["profiles"]
        if not requested or str(profile["id"]) in requested
    ]
    if requested - {str(profile["id"]) for profile in profiles}:
        missing = sorted(requested - {str(profile["id"]) for profile in profiles})
        raise ValueError(f"unknown profile ids: {missing}")
    records = []
    for profile in profiles:
        try:
            record = audit_profile(
                root,
                profile,
                output_root,
                float(args.resolution_m),
                float(args.floor_half_band_m),
                float(args.clearance_height_m),
            )
        except Exception as error:  # Keep the full batch auditable.
            record = {
                "profile_id": str(profile["id"]),
                "asset_id": str(profile["asset"]["asset_id"]),
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
        records.append(record)
        candidates = record.get("candidates", [])
        radius = candidates[0]["clear_radius_m"] if candidates else 0.0
        print(
            f"{record['profile_id']}: {record['status']} "
            f"radius={radius}"
        )
    manifest = {
        "schema_version": "physweep_environment_action_surface_audit_v1",
        "source_profiles": str(profiles_path.relative_to(root)),
        "policy": {
            "surface": "reviewed_source_floor_faces",
            "floor_half_band_m": float(args.floor_half_band_m),
            "minimum_visual_obstacle_height_m": 0.006,
            "obstacles": (
                "union_of_projected_visible_source_geometry_and_static_environment_"
                "proxy_below_clearance_height"
            ),
            "selection": "largest_clear_circle_with_four_alternatives",
            "human_visual_review_required": True,
        },
        "counts": {
            "profiles": len(records),
            "candidates": sum(record["status"] == "candidate" for record in records),
            "insufficient_clearance": sum(
                record["status"] == "insufficient_clearance" for record in records
            ),
            "errors": sum(record["status"] == "error" for record in records),
        },
        "records": records,
    }
    write_json(output_root / "audit_manifest.json", manifest)
    print(output_root / "audit_manifest.json")


if __name__ == "__main__":
    main()
