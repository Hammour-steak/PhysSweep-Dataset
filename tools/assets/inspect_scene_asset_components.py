#!/usr/bin/env python3
"""Inspect one curated scene asset at normalized metric scale inside Blender."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
import mathutils


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json
from tools.rendering.render_asset_proxy_reviews import (  # pylint: disable=wrong-import-position
    bounds as mesh_list_bounds,
    clear_scene,
    import_meshes,
    normalized_transform,
    selected_visual_meshes,
)


KEYWORD_GROUPS = {
    "support_surface": ("table", "desk", "bench", "counter", "worktop", "top"),
    "container_or_void": ("sink", "basin", "tray", "drawer", "shelf", "cabinet"),
    "game_piece": ("ball", "cue", "rack"),
    "movable_prop": ("book", "plate", "cup", "bottle", "tool", "lamp", "chair"),
    "room_shell": ("wall", "floor", "ceiling", "door", "window", "room"),
    "fixture_or_obstacle": ("upright", "rail", "leg", "stand", "faucet", "tap", "handle"),
}


def blender_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(values)


def bounds(obj: Any) -> tuple[list[float], list[float]]:
    corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    low = [min(float(corner[axis]) for corner in corners) for axis in range(3)]
    high = [max(float(corner[axis]) for corner in corners) for axis in range(3)]
    return low, high


def parent_chain(obj: Any) -> list[str]:
    result = []
    parent = obj.parent
    while parent is not None:
        result.append(str(parent.name))
        parent = parent.parent
    return result


def keyword_hints(name: str, materials: list[str]) -> list[str]:
    text = " ".join([name, *materials]).lower()
    return [
        group
        for group, keywords in KEYWORD_GROUPS.items()
        if any(keyword in text for keyword in keywords)
    ]


def inspection_transform(meshes: list[Any], record: dict[str, Any]) -> tuple[Any, str]:
    visual = record["visual"]
    kind = str(record["proxy"]["kind"])
    metric_support = kind == "support_compound" and all(
        key in visual
        for key in (
            "source_support_bounds_xy",
            "source_support_plane_z_from_bottom",
            "target_support_size_xy_m",
        )
    )
    if metric_support or "canonical_extent_m" in visual:
        return normalized_transform(meshes, record), "metric_registry_binding"

    low, high = mesh_list_bounds(meshes)
    center = (low + high) * 0.5
    anchor = mathutils.Vector((center.x, center.y, low.z))
    euler = [
        math.radians(float(value))
        for value in visual.get("alignment_euler_degrees", [0.0, 0.0, 0.0])
    ]
    rotation = mathutils.Euler(tuple(euler), "XYZ").to_matrix().to_4x4()
    corners = []
    for x in (low.x, high.x):
        for y in (low.y, high.y):
            for z in (low.z, high.z):
                corners.append(rotation @ (mathutils.Vector((x, y, z)) - anchor))
    aligned_extent = [
        max(float(corner[axis]) for corner in corners)
        - min(float(corner[axis]) for corner in corners)
        for axis in range(3)
    ]
    review_longest_m = {
        "interactive_support": 2.4,
        "render_only_context": 2.4,
        "static_prop": 0.45,
    }[record["asset_role"]]
    scale = review_longest_m / max(aligned_extent)
    return (
        mathutils.Matrix.Scale(scale, 4)
        @ rotation
        @ mathutils.Matrix.Translation(-anchor),
        "review_scale_only_not_metric",
    )


def main() -> None:
    args = blender_args()
    root = args.root.resolve()
    registry = load_json(args.registry.resolve())
    record = next(item for item in registry["records"] if item["asset_id"] == args.asset_id)
    clear_scene()
    meshes = import_meshes(root, record)
    imported_mesh_component_count = len(meshes)
    variant_object_names = [
        str(value) for value in record["visual"].get("variant_object_names", [])
    ]
    variant_source_extents_m = {}
    if variant_object_names:
        by_name = {str(obj.name): obj for obj in meshes}
        missing = sorted(set(variant_object_names) - set(by_name))
        if missing:
            raise ValueError(
                f"missing configured visual variants for {record['asset_id']}: {missing}"
            )
        for name in variant_object_names:
            variant = by_name[name]
            low, high = mesh_list_bounds([variant])
            variant_source_extents_m[name] = [
                round(float(high[axis] - low[axis]), 7) for axis in range(3)
            ]
            normalized_transform([variant], record)
    meshes = selected_visual_meshes(meshes, record)
    selected_object_names = sorted(str(obj.name) for obj in meshes)
    transform, normalization_mode = inspection_transform(meshes, record)
    for obj in meshes:
        obj.matrix_world = transform @ obj.matrix_world
    bpy.context.view_layer.update()

    components = []
    overall_low = [float("inf")] * 3
    overall_high = [float("-inf")] * 3
    for obj in sorted(meshes, key=lambda value: value.name.lower()):
        low, high = bounds(obj)
        extent = [high[axis] - low[axis] for axis in range(3)]
        center = [(low[axis] + high[axis]) * 0.5 for axis in range(3)]
        for axis in range(3):
            overall_low[axis] = min(overall_low[axis], low[axis])
            overall_high[axis] = max(overall_high[axis], high[axis])
        materials = [str(value.name) for value in obj.data.materials if value is not None]
        components.append(
            {
                "object_name": str(obj.name),
                "parent_chain": parent_chain(obj),
                "bounds_low_m": [round(value, 7) for value in low],
                "bounds_high_m": [round(value, 7) for value in high],
                "center_m": [round(value, 7) for value in center],
                "extent_m": [round(value, 7) for value in extent],
                "vertex_count": len(obj.data.vertices),
                "triangle_count": sum(max(0, len(poly.vertices) - 2) for poly in obj.data.polygons),
                "materials": materials,
                "keyword_hints": keyword_hints(str(obj.name), materials),
            }
        )
    overall_extent = [overall_high[axis] - overall_low[axis] for axis in range(3)]
    overall_volume = max(1.0e-12, overall_extent[0] * overall_extent[1] * overall_extent[2])
    for component in components:
        extent = component["extent_m"]
        component["bbox_volume_fraction"] = round(
            float(extent[0] * extent[1] * extent[2]) / overall_volume, 8
        )

    output = {
        "schema_version": "physweep_scene_asset_component_inventory_v1",
        "asset_id": record["asset_id"],
        "name": record["name"],
        "asset_role": record["asset_role"],
        "semantic_category": record["semantic_category"],
        "admission": record["admission"],
        "proxy_kind": record["proxy"]["kind"],
        "normalization_mode": normalization_mode,
        "imported_mesh_component_count": imported_mesh_component_count,
        "selected_object_names": selected_object_names,
        "variant_object_names": variant_object_names,
        "variant_source_extents_m": variant_source_extents_m,
        "mesh_component_count": len(components),
        "overall_bounds_low_m": [round(value, 7) for value in overall_low],
        "overall_bounds_high_m": [round(value, 7) for value in overall_high],
        "overall_extent_m": [round(value, 7) for value in overall_extent],
        "total_vertex_count": sum(item["vertex_count"] for item in components),
        "total_triangle_count": sum(item["triangle_count"] for item in components),
        "components": components,
    }
    write_json(args.output.resolve(), output)
    print(json.dumps({key: output[key] for key in output if key != "components"}, indent=2))


if __name__ == "__main__":
    main()
