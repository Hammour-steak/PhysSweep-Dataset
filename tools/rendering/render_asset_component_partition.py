#!/usr/bin/env python3
"""Render a color-coded component partition for one curated scene asset."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.core.json_io import read_json as load_json
from tools.assets.audit_scene_visual_assets import bbox, look_at, setup_stage  # noqa: E402
from tools.assets.inspect_scene_asset_components import inspection_transform  # noqa: E402
from tools.assets.blender_asset_import import clear_scene, import_meshes  # noqa: E402


ROLE_COLORS = (
    ("support", (0.08, 0.62, 0.22, 1.0)),
    ("context_1", (0.08, 0.34, 0.86, 1.0)),
    ("context_2", (0.95, 0.38, 0.06, 1.0)),
    ("context_3", (0.60, 0.16, 0.78, 1.0)),
)


def blender_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(values)


def make_role_material(name: str, color: tuple[float, float, float, float]) -> Any:
    material = bpy.data.materials.new(f"partition_{name}")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.58
    return material


def assign_material(obj: Any, material: Any) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(material)


def render_views(meshes: list[Any], output: Path) -> None:
    camera, radius = setup_stage(meshes, output)
    low, high = bbox(meshes)
    target = (
        float((low.x + high.x) * 0.5),
        float((low.y + high.y) * 0.5),
        float(low.z + (high.z - low.z) * 0.48),
    )
    distance = radius * 2.35 + 0.7
    height = max(float(high.z) * 0.58, radius * 0.55)
    for yaw in (0, 90, 180, 270):
        angle = math.radians(yaw)
        camera.location = (
            target[0] + math.sin(angle) * distance,
            target[1] - math.cos(angle) * distance,
            height,
        )
        look_at(camera, target)
        bpy.context.scene.render.filepath = str(output / f"partition_yaw_{yaw:03d}.png")
        bpy.ops.render.render(write_still=True)
    camera.location = (target[0], target[1], float(high.z) + distance)
    look_at(camera, target)
    bpy.context.scene.render.filepath = str(output / "partition_top.png")
    bpy.ops.render.render(write_still=True)


def main() -> None:
    args = blender_args()
    root = args.root.resolve()
    registry = load_json(args.registry.resolve())
    composition = load_json(args.composition.resolve())
    record = next(item for item in registry["records"] if item["asset_id"] == args.asset_id)
    decision = next(item for item in composition["records"] if item["asset_id"] == args.asset_id)
    policy = decision["component_policy"]
    if policy.get("mode") != "exact_partition":
        raise ValueError(f"asset does not use exact_partition: {args.asset_id}")

    support_names = list(policy.get("support_objects", []))
    context_names = list(policy.get("separate_context_objects", []))
    declared = support_names + context_names
    if not support_names:
        raise ValueError("exact_partition requires at least one support object")
    if len(declared) != len(set(declared)):
        raise ValueError("component names must be unique across partition roles")

    clear_scene()
    meshes = import_meshes(root, record)
    transform, normalization_mode = inspection_transform(meshes, record)
    for obj in meshes:
        obj.matrix_world = transform @ obj.matrix_world
    bpy.context.view_layer.update()

    imported_names = {str(obj.name) for obj in meshes}
    undeclared = sorted(imported_names - set(declared))
    missing = sorted(set(declared) - imported_names)
    if undeclared or missing:
        raise ValueError(f"partition mismatch: undeclared={undeclared}, missing={missing}")

    role_entries = []
    support_material = make_role_material(*ROLE_COLORS[0])
    for name in support_names:
        obj = next(item for item in meshes if item.name == name)
        assign_material(obj, support_material)
        role_entries.append({"object_name": name, "role": "support"})
    for index, name in enumerate(context_names):
        role_name, color = ROLE_COLORS[1 + index % (len(ROLE_COLORS) - 1)]
        obj = next(item for item in meshes if item.name == name)
        assign_material(obj, make_role_material(f"{role_name}_{index + 1}", color))
        role_entries.append({"object_name": name, "role": role_name})

    output = args.output.resolve()
    render_views(meshes, output)
    manifest = {
        "schema_version": "physweep_asset_component_partition_review_v1",
        "asset_id": args.asset_id,
        "asset_name": decision["name"],
        "normalization_mode": normalization_mode,
        "roles": role_entries,
        "unclassified_objects": [],
        "review_images": [
            *[f"partition_yaw_{yaw:03d}.png" for yaw in (0, 90, 180, 270)],
            "partition_top.png",
        ],
    }
    (output / "partition_review.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
