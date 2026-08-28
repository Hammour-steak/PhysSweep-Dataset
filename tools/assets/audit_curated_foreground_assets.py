#!/usr/bin/env python3
"""Audit curated GLB orientation and visual/collision dimension consistency."""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json
from tools.core.blender_runtime import (
    blender_argv,
    clear_blender_scene,
    patch_numpy_for_blender_gltf,
)


DEFAULT_SOURCE_MANIFEST = PROJECT_ROOT / "assets/manifests/sketchfab_foreground_source_v1.json"


def imported_meshes(path: Path) -> list[Any]:
    import bpy  # pylint: disable=import-outside-toplevel

    clear_blender_scene(("meshes", "materials", "images"))
    patch_numpy_for_blender_gltf()
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise ValueError(f"No mesh imported from {path}")
    for obj in meshes:
        matrix_world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = matrix_world
    return meshes


def world_bbox_corners(meshes: list[Any]) -> list[Any]:
    import mathutils  # pylint: disable=import-outside-toplevel

    return [obj.matrix_world @ mathutils.Vector(corner) for obj in meshes for corner in obj.bound_box]


def unique_axis_rotations() -> list[dict[str, Any]]:
    import mathutils  # pylint: disable=import-outside-toplevel

    rotations: dict[tuple[float, ...], dict[str, Any]] = {}
    for pitch, roll, yaw in itertools.product((0, 90, 180, 270), repeat=3):
        matrix = (
            mathutils.Matrix.Rotation(math.radians(yaw), 4, "Z")
            @ mathutils.Matrix.Rotation(math.radians(roll), 4, "Y")
            @ mathutils.Matrix.Rotation(math.radians(pitch), 4, "X")
        )
        key = tuple(round(float(matrix[row][col]), 5) for row in range(3) for col in range(3))
        rotations.setdefault(
            key,
            {
                "degrees": [float(pitch), float(roll), float(yaw)],
                "matrix": matrix,
            },
        )
    return list(rotations.values())


def rotated_dimensions(corners: list[Any], rotation: Any) -> list[float]:
    center = sum(corners[1:], corners[0].copy()) / len(corners)
    rotated = [rotation @ (corner - center) for corner in corners]
    return [
        max(float(point[axis]) for point in rotated) - min(float(point[axis]) for point in rotated)
        for axis in range(3)
    ]


def dimension_fit(source: list[float], target: list[float]) -> dict[str, Any]:
    source = [max(float(value), 1e-9) for value in source]
    target = [max(float(value), 1e-9) for value in target]
    scale = max(target) / max(source)
    fitted = [value * scale for value in source]
    ratios = [max(fitted[index] / target[index], target[index] / fitted[index]) for index in range(3)]
    log_error = math.sqrt(sum(math.log(ratio) ** 2 for ratio in ratios) / 3.0)
    return {
        "uniform_scale": scale,
        "fitted_visual_extent_m": fitted,
        "axis_size_error_ratio": ratios,
        "max_axis_size_error_ratio": max(ratios),
        "rms_log_size_error": log_error,
    }


def rounded(values: list[float], digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def audit_record(record: dict[str, Any], rotations: list[dict[str, Any]], approve_ratio: float, review_ratio: float) -> dict[str, Any]:
    asset_id = str(record["candidate_id"])
    path = Path(str(record["archive_path"]))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    meshes = imported_meshes(path)
    corners = world_bbox_corners(meshes)
    raw_dimensions = rotated_dimensions(corners, rotations[0]["matrix"])
    canonical = [float(value) for value in record["simulation_profile"]["canonical_scale"]["canonical_extent_m"]]

    candidates: list[dict[str, Any]] = []
    for rotation in rotations:
        dimensions = rotated_dimensions(corners, rotation["matrix"])
        fit = dimension_fit(dimensions, canonical)
        candidates.append(
            {
                "visual_alignment_degrees": rotation["degrees"],
                "oriented_source_bbox": dimensions,
                **fit,
            }
        )
    best = min(candidates, key=lambda item: (item["max_axis_size_error_ratio"], item["rms_log_size_error"]))
    ratio = float(best["max_axis_size_error_ratio"])
    if ratio <= approve_ratio:
        status = "approved"
    elif ratio <= review_ratio:
        status = "review"
    else:
        status = "quarantine"

    return {
        "asset_id": asset_id,
        "name": record.get("name"),
        "semantic_category": record.get("semantic_category"),
        "source_path": str(record["archive_path"]),
        "source_mesh_count": len(meshes),
        "source_bbox": rounded(raw_dimensions),
        "canonical_extent_m": rounded(canonical),
        "recommended_visual_alignment_degrees": rounded(best["visual_alignment_degrees"], 3),
        "oriented_source_bbox": rounded(best["oriented_source_bbox"]),
        "uniform_scale": round(float(best["uniform_scale"]), 8),
        "predicted_visual_extent_m": rounded(best["fitted_visual_extent_m"]),
        "axis_size_error_ratio": rounded(best["axis_size_error_ratio"], 4),
        "max_axis_size_error_ratio": round(ratio, 4),
        "rms_log_size_error": round(float(best["rms_log_size_error"]), 6),
        "status": status,
        "reason": "uniform_visual_scale_and_axis-aligned_orientation_must_match_canonical_collision_extent",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approve-ratio", type=float, default=1.35)
    parser.add_argument("--review-ratio", type=float, default=1.60)
    args = parser.parse_args(blender_argv())

    source = load_json(args.source_manifest)
    by_id = {str(record.get("candidate_id")): record for record in source.get("records", [])}
    rotations = unique_axis_rotations()
    records = []
    for asset_id in sorted(by_id):
        print("audit-foreground", asset_id)
        records.append(audit_record(by_id[asset_id], rotations, args.approve_ratio, args.review_ratio))

    report = {
        "version": "physweep_foreground_geometry_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.source_manifest),
        "thresholds": {
            "approve_max_axis_size_error_ratio": args.approve_ratio,
            "review_max_axis_size_error_ratio": args.review_ratio,
        },
        "rotation_candidates": len(rotations),
        "counts": {
            status: sum(record["status"] == status for record in records)
            for status in ("approved", "review", "quarantine")
        },
        "records": records,
    }
    write_json(args.output, report)
    print("output", args.output)
    print("counts", report["counts"])


if __name__ == "__main__":
    main()
