#!/usr/bin/env python3
"""Render normalized preview images for downloaded Sketchfab background assets."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json


def blender_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def patch_numpy_for_blender_gltf() -> None:
    import numpy as np  # pylint: disable=import-outside-toplevel

    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]


def bbox_for_objects(objects: list[Any]) -> tuple[Any, Any]:
    import mathutils  # pylint: disable=import-outside-toplevel

    mins = mathutils.Vector((float("inf"), float("inf"), float("inf")))
    maxs = mathutils.Vector((float("-inf"), float("-inf"), float("-inf")))
    found = False
    for obj in objects:
        if obj.type != "MESH":
            continue
        found = True
        for corner in obj.bound_box:
            point = obj.matrix_world @ mathutils.Vector(corner)
            mins.x = min(mins.x, point.x)
            mins.y = min(mins.y, point.y)
            mins.z = min(mins.z, point.z)
            maxs.x = max(maxs.x, point.x)
            maxs.y = max(maxs.y, point.y)
            maxs.z = max(maxs.z, point.z)
    if not found:
        mins = mathutils.Vector((0.0, 0.0, 0.0))
        maxs = mathutils.Vector((1.0, 1.0, 1.0))
    return mins, maxs


def has_image_texture(objects: list[Any]) -> bool:
    for obj in objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            mat = slot.material
            if not mat or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.bl_idname == "ShaderNodeTexImage" and getattr(node, "image", None):
                    return True
    return False


def make_solid_material(name: str, color: tuple[float, float, float, float]) -> Any:
    import bpy  # pylint: disable=import-outside-toplevel

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.72
    return mat


def fallback_color(category: str) -> tuple[float, float, float, float]:
    if category.startswith("support_"):
        return (0.55, 0.36, 0.22, 1.0)
    if category.startswith("context_"):
        return (0.58, 0.58, 0.54, 1.0)
    if "lamp" in category:
        return (0.26, 0.25, 0.23, 1.0)
    if "tableware" in category:
        return (0.84, 0.82, 0.76, 1.0)
    return (0.45, 0.35, 0.26, 1.0)


def setup_scene(resolution: tuple[int, int], samples: int) -> None:
    import bpy  # pylint: disable=import-outside-toplevel

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    try:
        scene.cycles.device = "GPU"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        for device in prefs.devices:
            device.use = True
    except Exception:
        pass
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.world = bpy.data.worlds.new("asset_preview_world") if scene.world is None else scene.world
    scene.world.color = (0.045, 0.046, 0.048)
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"

    bpy.ops.object.light_add(type="AREA", location=(-2.0, -2.2, 3.2))
    key = bpy.context.object
    key.data.energy = 620.0
    key.data.size = 4.0

    bpy.ops.object.light_add(type="AREA", location=(2.0, 1.5, 2.2))
    fill = bpy.context.object
    fill.data.energy = 90.0
    fill.data.size = 5.0


def import_asset(record: dict[str, Any]) -> list[Any]:
    patch_numpy_for_blender_gltf()
    import bpy  # pylint: disable=import-outside-toplevel

    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=record["archive_path"])
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    for obj in list(imported):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
            imported.remove(obj)
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh imported for {record['candidate_id']}")
    return meshes


def normalize_asset(meshes: list[Any], record: dict[str, Any]) -> dict[str, Any]:
    import mathutils  # pylint: disable=import-outside-toplevel

    mins, maxs = bbox_for_objects(meshes)
    size = maxs - mins
    category = record.get("classification", {}).get("semantic_category", "")
    role = record.get("classification", {}).get("library_role", "")
    if role == "background_prop":
        target = 1.1
    elif role == "background_context":
        target = 2.4
    else:
        target = 2.9
    scale = target / max(float(size.x), float(size.y), float(size.z), 1e-8)
    bottom_center = mathutils.Vector(((mins.x + maxs.x) / 2.0, (mins.y + maxs.y) / 2.0, mins.z))
    transform = mathutils.Matrix.Scale(scale, 4) @ mathutils.Matrix.Translation(-bottom_center)
    for obj in meshes:
        obj.matrix_world = transform @ obj.matrix_world
    if not has_image_texture(meshes):
        mat = make_solid_material("fallback_preview", fallback_color(category))
        for obj in meshes:
            obj.data.materials.clear()
            obj.data.materials.append(mat)
    new_min, new_max = bbox_for_objects(meshes)
    return {
        "scale": round(float(scale), 8),
        "bbox_min": [round(float(v), 6) for v in new_min],
        "bbox_max": [round(float(v), 6) for v in new_max],
        "has_image_texture": has_image_texture(meshes),
    }


def look_at(camera: Any, target: tuple[float, float, float]) -> None:
    import mathutils  # pylint: disable=import-outside-toplevel

    direction = mathutils.Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera(meshes: list[Any]) -> dict[str, Any]:
    import bpy  # pylint: disable=import-outside-toplevel
    import mathutils  # pylint: disable=import-outside-toplevel

    mins, maxs = bbox_for_objects(meshes)
    center = (mins + maxs) / 2.0
    span = maxs - mins
    target = (float(center.x), float(center.y), float(center.z))
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.lens = 50.0
    camera.data.clip_start = 0.01
    radius = max(0.25, 0.5 * float(span.length))
    limiting_angle = min(float(camera.data.angle_x), float(camera.data.angle_y))
    distance = radius / max(math.sin(limiting_angle * 0.5), 1.0e-4) * 1.18
    direction = mathutils.Vector((0.68, -0.92, 0.48)).normalized()
    position_vector = mathutils.Vector(target) + direction * distance
    position = tuple(float(value) for value in position_vector)
    camera.location = position
    look_at(camera, target)
    bpy.context.scene.camera = camera
    return {"position": [round(v, 4) for v in position], "target": [round(v, 4) for v in target]}


def render_record(record: dict[str, Any], output_dir: Path, resolution: tuple[int, int], samples: int) -> dict[str, Any]:
    import bpy  # pylint: disable=import-outside-toplevel

    setup_scene(resolution, samples)
    meshes = import_asset(record)
    norm = normalize_asset(meshes, record)
    cam = add_camera(meshes)
    image_path = output_dir / "images" / f"{record['candidate_id']}.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(image_path)
    bpy.ops.render.render(write_still=True)
    return {
        "candidate_id": record["candidate_id"],
        "name": record.get("name"),
        "classification": record.get("classification"),
        "image_path": str(image_path),
        "normalization": norm,
        "camera": cam,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render downloaded background asset previews.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", default="640x360")
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args(blender_argv())

    width, height = [int(v) for v in args.resolution.lower().split("x", 1)]
    manifest = load_json(args.manifest)
    records = [
        record
        for record in manifest["records"]
        if record.get("status") in {"downloaded", "exists"} and record.get("archive_kind") == "glb"
    ]
    rendered = []
    for record in records:
        print("render", record["candidate_id"], record.get("name"))
        rendered.append(render_record(record, args.output_dir, (width, height), args.samples))
    out = {
        "render_id": "sketchfab_background_asset_previews_v0",
        "download_manifest": str(args.manifest),
        "output_dir": str(args.output_dir),
        "resolution": args.resolution,
        "samples_per_pixel": args.samples,
        "records": rendered,
    }
    write_json(args.output_dir / "render_manifest.json", out)
    print("manifest:", args.output_dir / "render_manifest.json")


if __name__ == "__main__":
    main()
