#!/usr/bin/env python3
"""Measure and render normalized turntables for candidate scene GLBs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256


DEFAULT_IDS = (
    "sketchfab_bg_29a550112e7e41528d7d01697231c11b",
    "sketchfab_bg_8df6adc634bd4004894a5e70565dc52a",
    "sketchfab_bg_96a804790eff41b3b2376f3608fde7df",
    "sketchfab_bg_4bc3743598f7498587468962b3c05e0e",
    "sketchfab_bg_4fa72fdbaa98467bb60d3233d5b36c57",
    "sketchfab_bg_c6ca6667fd7d4381bd184874dcf44075",
    "sketchfab_bg_693529b6817a4632bc812394a48f06b0",
    "sketchfab_bg_b9458e04d30e4b6284d46d35880a9b95",
)

TARGET_HEIGHTS_M = {
    "context_room_corner": 2.8,
    "context_shelf": 2.0,
    "context_wall_window": 2.8,
    "prop_books": 0.24,
    "prop_lamp": 0.52,
    "prop_tableware": 0.12,
    "prop_tray": 0.10,
    "support_game_table": 0.82,
    "support_table_wood": 0.78,
    "support_office_desk": 0.78,
    "support_lab_bench": 0.90,
    "support_kitchen_counter": 0.90,
    "support_workbench": 0.90,
}


def blender_args() -> argparse.Namespace:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="*", default=list(DEFAULT_IDS))
    return parser.parse_args(args)


def reset_scene() -> None:
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def bbox(objects: list[Any]) -> tuple[Any, Any]:
    import mathutils

    low = mathutils.Vector((float("inf"),) * 3)
    high = mathutils.Vector((float("-inf"),) * 3)
    for obj in objects:
        for corner in obj.bound_box:
            point = obj.matrix_world @ mathutils.Vector(corner)
            for axis in range(3):
                low[axis] = min(low[axis], point[axis])
                high[axis] = max(high[axis], point[axis])
    return low, high


def has_image_texture(objects: list[Any]) -> bool:
    for obj in objects:
        for material in obj.data.materials:
            if not material or not material.use_nodes:
                continue
            if any(node.type == "TEX_IMAGE" and node.image for node in material.node_tree.nodes):
                return True
    return False


def import_asset(path: Path) -> list[Any]:
    import bpy
    import numpy as np

    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]

    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    for obj in list(imported):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
            imported.remove(obj)
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise ValueError(f"no mesh in {path}")
    for obj in meshes:
        matrix = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = matrix
    return meshes


def normalize(objects: list[Any], target_height: float) -> tuple[list[float], float]:
    import mathutils

    low, high = bbox(objects)
    source_size = high - low
    scale = target_height / max(float(source_size.z), 1.0e-8)
    bottom_center = mathutils.Vector(((low.x + high.x) / 2.0, (low.y + high.y) / 2.0, low.z))
    transform = mathutils.Matrix.Scale(scale, 4) @ mathutils.Matrix.Translation(-bottom_center)
    for obj in objects:
        obj.matrix_world = transform @ obj.matrix_world
    return [round(float(value), 6) for value in source_size], scale


def look_at(obj: Any, target: tuple[float, float, float]) -> None:
    import mathutils

    obj.rotation_euler = (mathutils.Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def setup_stage(objects: list[Any], output: Path) -> tuple[Any, float]:
    import bpy

    low, high = bbox(objects)
    span = high - low
    radius = max(float(span.x), float(span.y), float(span.z), 0.5)
    bpy.ops.mesh.primitive_plane_add(size=max(radius * 4.0, 6.0), location=(0.0, 0.0, -0.01))
    floor = bpy.context.object
    material = bpy.data.materials.new("audit_floor")
    material.diffuse_color = (0.16, 0.17, 0.18, 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = (0.16, 0.17, 0.18, 1.0)
    principled.inputs["Roughness"].default_value = 0.82
    floor.data.materials.append(material)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.lens = 52.0
    bpy.context.scene.camera = camera

    for name, location, energy, size in (
        ("key", (-3.0, -4.0, 5.0), 850.0, 3.0),
        ("fill", (4.0, 1.0, 3.5), 300.0, 4.0),
    ):
        data = bpy.data.lights.new(name, "AREA")
        data.energy = energy
        data.size = size
        light = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(light)
        light.location = location
        look_at(light, (0.0, 0.0, float(high.z) * 0.45))

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3.0
    scene.eevee.gtao_factor = 1.25
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.world.color = (0.035, 0.04, 0.05)
    scene.view_settings.look = "Medium High Contrast"
    output.mkdir(parents=True, exist_ok=True)
    return camera, radius


def render_turntable(objects: list[Any], output: Path) -> None:
    import bpy
    import mathutils

    camera, radius = setup_stage(objects, output)
    low, high = bbox(objects)
    target_z = max(0.18, float(high.z) * 0.48)
    distance = radius * 2.35 + 0.7
    height = max(float(high.z) * 0.58, radius * 0.55)
    for yaw in (0, 90, 180, 270):
        angle = math.radians(yaw)
        camera.location = (math.sin(angle) * distance, -math.cos(angle) * distance, height)
        look_at(camera, (0.0, 0.0, target_z))
        bpy.context.scene.render.filepath = str(output / f"yaw_{yaw:03d}.png")
        bpy.ops.render.render(write_still=True)


def main() -> None:
    args = blender_args()
    root = args.root.resolve()
    output = args.output.resolve()
    manifest = json.loads((root / "assets/manifests/sketchfab_background_admission_v1.json").read_text())
    records = {record["candidate_id"]: record for record in manifest["records"]}
    report = []
    for asset_id in args.ids:
        reset_scene()
        record = records[asset_id]
        path = root / record["archive_path"]
        if sha256(path) != record["sha256"]:
            raise ValueError(f"hash mismatch: {asset_id}")
        meshes = import_asset(path)
        textured = has_image_texture(meshes)
        target_height = TARGET_HEIGHTS_M[record["semantic_category"]]
        source_size, scale = normalize(meshes, target_height)
        low, high = bbox(meshes)
        asset_output = output / asset_id
        render_turntable(meshes, asset_output)
        report.append(
            {
                "asset_id": asset_id,
                "name": record["name"],
                "semantic_category": record["semantic_category"],
                "path": record["archive_path"],
                "sha256": record["sha256"],
                "license": record["license"],
                "source_bbox_size": source_size,
                "target_height_m": target_height,
                "normalization_scale": round(scale, 9),
                "normalized_bbox_min": [round(float(value), 6) for value in low],
                "normalized_bbox_max": [round(float(value), 6) for value in high],
                "has_image_texture": textured,
                "turntable_dir": str(asset_output.relative_to(root)),
            }
        )
    report_path = output / "audit_report.json"
    report_path.write_text(json.dumps({"records": report}, indent=2) + "\n")
    print(report_path)


if __name__ == "__main__":
    main()
