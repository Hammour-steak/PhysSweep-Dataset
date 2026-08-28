#!/usr/bin/env python3
"""Measure dominant upward-facing planes in candidate support GLBs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.core.blender_runtime import blender_world_bounds as bounds
from tools.core.blender_runtime import clear_blender_scene as clear_scene


def args_from_blender() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("assets/manifests/sketchfab_background_admission_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    return parser.parse_args(values)


def import_meshes(path: Path) -> list[Any]:
    import bpy
    import numpy as np

    if "bool" not in np.__dict__:
        np.bool = np.bool_
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise ValueError(f"no mesh in {path}")
    for obj in meshes:
        matrix = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = matrix
    return meshes


def projected_xy_area(points: list[Any]) -> float:
    return abs(
        sum(
            points[index].x * points[(index + 1) % len(points)].y
            - points[(index + 1) % len(points)].x * points[index].y
            for index in range(len(points))
        )
    ) * 0.5


def plane_clusters(meshes: list[Any], low: Any, high: Any) -> list[dict[str, Any]]:
    height = max(float(high.z - low.z), 1.0e-8)
    tolerance = height * 0.0025
    records = []
    for obj in meshes:
        normal_matrix = obj.matrix_world.to_3x3()
        for polygon in obj.data.polygons:
            normal = (normal_matrix @ polygon.normal).normalized()
            if float(normal.z) < 0.94:
                continue
            points = [obj.matrix_world @ obj.data.vertices[index].co for index in polygon.vertices]
            area = projected_xy_area(points)
            if area <= 1.0e-10:
                continue
            records.append(
                {
                    "z": sum(float(point.z) for point in points) / len(points),
                    "area": area,
                    "min_x": min(float(point.x) for point in points),
                    "max_x": max(float(point.x) for point in points),
                    "min_y": min(float(point.y) for point in points),
                    "max_y": max(float(point.y) for point in points),
                }
            )
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for record in records:
        grouped[round((record["z"] - float(low.z)) / tolerance)].append(record)
    total_area = sum(record["area"] for record in records)
    clusters = []
    for members in grouped.values():
        area = sum(record["area"] for record in members)
        z = sum(record["z"] * record["area"] for record in members) / area
        clusters.append(
            {
                "z_world": round(z, 8),
                "z_from_bottom": round(z - float(low.z), 8),
                "height_fraction": round((z - float(low.z)) / height, 8),
                "projected_area": round(area, 8),
                "upward_area_fraction": round(area / max(total_area, 1.0e-12), 8),
                "xy_bounds": [
                    round(min(record["min_x"] for record in members), 6),
                    round(max(record["max_x"] for record in members), 6),
                    round(min(record["min_y"] for record in members), 6),
                    round(max(record["max_y"] for record in members), 6),
                ],
                "polygon_count": len(members),
            }
        )
    clusters.sort(key=lambda record: record["projected_area"], reverse=True)
    return clusters[:16]


def main() -> None:
    args = args_from_blender()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    manifest = json.loads(manifest_path.read_text())
    by_id = {record["candidate_id"]: record for record in manifest["records"]}
    output = []
    for asset_id in args.ids:
        clear_scene()
        record = by_id[asset_id]
        meshes = import_meshes(root / record["archive_path"])
        low, high = bounds(meshes)
        output.append(
            {
                "asset_id": asset_id,
                "name": record["name"],
                "semantic_category": record["semantic_category"],
                "bbox_min": [round(float(value), 8) for value in low],
                "bbox_max": [round(float(value), 8) for value in high],
                "bbox_size": [round(float(value), 8) for value in high - low],
                "dominant_upward_planes": plane_clusters(meshes, low, high),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"records": output}, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
