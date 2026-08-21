#!/usr/bin/env python3
"""Measure upward-facing surface clusters on configured support components."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
import mathutils


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inspect_scene_asset_components import inspection_transform  # noqa: E402
from render_asset_proxy_reviews import bounds, clear_scene, import_meshes  # noqa: E402


def blender_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--normal-z-min", type=float, default=0.94)
    parser.add_argument("--cluster-tolerance-m", type=float, default=0.008)
    return parser.parse_args(values)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def triangle_area(a: Any, b: Any, c: Any) -> float:
    return float((b - a).cross(c - a).length * 0.5)


def world_triangle(obj: Any, triangle: Any) -> list[Any]:
    return [obj.matrix_world @ obj.data.vertices[index].co for index in triangle.vertices]


def upward_triangles(
    meshes: list[Any],
    object_names: list[str],
    transform: Any,
    normal_z_min: float,
) -> list[dict[str, Any]]:
    mesh_by_name = {str(obj.name): obj for obj in meshes}
    result = []
    for name in object_names:
        obj = mesh_by_name[name]
        obj.data.calc_loop_triangles()
        for triangle in obj.data.loop_triangles:
            vertices = [transform @ vertex for vertex in world_triangle(obj, triangle)]
            normal = (vertices[1] - vertices[0]).cross(vertices[2] - vertices[0])
            if normal.length <= 1.0e-12:
                continue
            normal.normalize()
            if float(normal.z) < normal_z_min:
                continue
            area = triangle_area(*vertices)
            if area <= 1.0e-10:
                continue
            result.append(
                {
                    "object_name": name,
                    "z": sum(float(vertex.z) for vertex in vertices) / 3.0,
                    "area": area,
                    "xy_points": [[float(vertex.x), float(vertex.y)] for vertex in vertices],
                }
            )
    return result


def cluster_triangles(
    triangles: list[dict[str, Any]], tolerance: float
) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for triangle in sorted(triangles, key=lambda value: value["z"]):
        target = next(
            (
                cluster
                for cluster in reversed(clusters)
                if abs(float(cluster["z_mean_m"]) - float(triangle["z"])) <= tolerance
            ),
            None,
        )
        if target is None:
            target = {
                "z_area_sum": 0.0,
                "area_m2": 0.0,
                "triangle_count": 0,
                "xy_points": [],
                "object_names": set(),
            }
            clusters.append(target)
        area = float(triangle["area"])
        target["z_area_sum"] += float(triangle["z"]) * area
        target["area_m2"] += area
        target["triangle_count"] += 1
        target["xy_points"].extend(triangle["xy_points"])
        target["object_names"].add(triangle["object_name"])
        target["z_mean_m"] = target["z_area_sum"] / max(target["area_m2"], 1.0e-12)

    result = []
    for cluster in clusters:
        points = cluster.pop("xy_points")
        low_x = min(point[0] for point in points)
        high_x = max(point[0] for point in points)
        low_y = min(point[1] for point in points)
        high_y = max(point[1] for point in points)
        bbox_area = max((high_x - low_x) * (high_y - low_y), 1.0e-12)
        result.append(
            {
                "z_mean_m": round(float(cluster["z_mean_m"]), 7),
                "area_m2": round(float(cluster["area_m2"]), 7),
                "xy_bounds_m": [
                    round(low_x, 7),
                    round(high_x, 7),
                    round(low_y, 7),
                    round(high_y, 7),
                ],
                "bbox_area_m2": round(float(bbox_area), 7),
                "area_to_bbox_ratio": round(float(cluster["area_m2"]) / bbox_area, 7),
                "triangle_count": int(cluster["triangle_count"]),
                "object_names": sorted(cluster["object_names"]),
            }
        )
    return sorted(result, key=lambda item: (-item["area_m2"], -item["z_mean_m"]))


def main() -> None:
    args = blender_args()
    root = args.root.resolve()
    registry = load_json(args.registry.resolve())
    composition = load_json(args.composition.resolve())
    record = next(item for item in registry["records"] if item["asset_id"] == args.asset_id)
    decision = next(item for item in composition["records"] if item["asset_id"] == args.asset_id)
    support_names = decision["component_policy"].get("support_objects", [])
    if not support_names:
        raise ValueError(f"asset has no configured support components: {args.asset_id}")

    clear_scene()
    meshes = import_meshes(root, record)
    mesh_by_name = {str(obj.name): obj for obj in meshes}
    missing = sorted(set(support_names) - set(mesh_by_name))
    if missing:
        raise ValueError(f"missing configured support components: {missing}")

    support_meshes = [mesh_by_name[name] for name in support_names]
    support_low, support_high = bounds(support_meshes)
    support_anchor = mathutils.Vector(
        (
            (support_low.x + support_high.x) * 0.5,
            (support_low.y + support_high.y) * 0.5,
            support_low.z,
        )
    )
    euler = [
        math.radians(float(value))
        for value in record["visual"].get("alignment_euler_degrees", [0.0, 0.0, 0.0])
    ]
    rotation = mathutils.Euler(tuple(euler), "XYZ").to_matrix().to_4x4()
    aligned_transform = rotation @ mathutils.Matrix.Translation(-support_anchor)
    source_triangles = upward_triangles(
        meshes, support_names, aligned_transform, args.normal_z_min
    )
    source_clusters = cluster_triangles(source_triangles, args.cluster_tolerance_m)

    review_transform, normalization_mode = inspection_transform(meshes, record)
    review_triangles = upward_triangles(
        meshes, support_names, review_transform, args.normal_z_min
    )
    review_clusters = cluster_triangles(review_triangles, args.cluster_tolerance_m)
    output = {
        "schema_version": "physweep_asset_component_surface_measurement_v1",
        "asset_id": args.asset_id,
        "asset_name": decision["name"],
        "normalization_mode": normalization_mode,
        "normal_z_min": args.normal_z_min,
        "cluster_tolerance_m": args.cluster_tolerance_m,
        "support_objects": support_names,
        "upward_triangle_count": len(source_triangles),
        "aligned_source_surface_clusters": source_clusters,
        "review_surface_clusters": review_clusters,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(
        json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
