#!/usr/bin/env python3
"""Render four normalized review views for visual-environment GLBs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from tools.core.blender_runtime import blender_argv, blender_world_bounds
from tools.core.blender_runtime import patch_numpy_for_blender_gltf as patch_numpy
from tools.rendering.blender_scene import look_at

from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json


def setup_scene(width: int, height: int, samples: int) -> None:
    import bpy  # pylint: disable=import-outside-toplevel

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = samples
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3.0
    scene.eevee.gtao_factor = 1.25
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.25
    scene.world = bpy.data.worlds.new("review_world") if scene.world is None else scene.world
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.12, 0.13, 0.14, 1.0)
    background.inputs["Strength"].default_value = 0.7

    bpy.ops.object.light_add(type="AREA", location=(-3.0, -4.0, 5.0))
    key = bpy.context.object
    key.data.energy = 900.0
    key.data.size = 5.0
    key.data.use_shadow = True
    bpy.ops.object.light_add(type="AREA", location=(4.0, 1.0, 3.5))
    fill = bpy.context.object
    fill.data.energy = 360.0
    fill.data.size = 6.0
    fill.data.use_shadow = True


def import_asset(record: dict[str, Any]) -> list[Any]:
    patch_numpy()
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
        raise RuntimeError(f"No meshes imported for {record['candidate_id']}")
    return meshes


def normalize(meshes: list[Any], target_extent: float) -> tuple[list[Any], list[list[float]]]:
    import mathutils  # pylint: disable=import-outside-toplevel

    minimum, maximum = blender_world_bounds(meshes)
    size = maximum - minimum
    scale = target_extent / max(float(size.x), float(size.y), float(size.z), 1.0e-8)
    center_bottom = mathutils.Vector(
        ((minimum.x + maximum.x) * 0.5, (minimum.y + maximum.y) * 0.5, minimum.z)
    )
    transform = mathutils.Matrix.Scale(scale, 4) @ mathutils.Matrix.Translation(-center_bottom)
    for obj in meshes:
        obj.matrix_world = transform @ obj.matrix_world
    return meshes, [[float(value) for value in row] for row in transform]


def add_camera() -> Any:
    import bpy  # pylint: disable=import-outside-toplevel

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.lens = 48.0
    camera.data.clip_start = 0.01
    camera.data.clip_end = 100.0
    bpy.context.scene.camera = camera
    return camera


def frame_camera(camera: Any, meshes: list[Any]) -> dict[str, list[float]]:
    minimum, maximum = blender_world_bounds(meshes)
    center = (minimum + maximum) * 0.5
    span = maximum - minimum
    radius = max(0.5, 0.5 * float(span.length))
    limiting_angle = min(float(camera.data.angle_x), float(camera.data.angle_y))
    distance = radius / max(math.sin(limiting_angle * 0.5), 1.0e-4) * 1.25
    target = (float(center.x), float(center.y), float(center.z))
    camera.location = (0.0, -distance, max(0.8, float(center.z) + distance * 0.28))
    look_at(camera, target)
    return {
        "position": [round(float(value), 5) for value in camera.location],
        "target": [round(float(value), 5) for value in target],
    }


def render_record(
    record: dict[str, Any], output_dir: Path, width: int, height: int, samples: int
) -> dict[str, Any]:
    import bpy  # pylint: disable=import-outside-toplevel
    import mathutils  # pylint: disable=import-outside-toplevel

    setup_scene(width, height, samples)
    meshes = import_asset(record)
    meshes, transform = normalize(meshes, 3.2)
    base_matrices = {obj.name: obj.matrix_world.copy() for obj in meshes}
    camera = add_camera()
    views = []
    for yaw in (0, 90, 180, 270):
        rotation = mathutils.Matrix.Rotation(math.radians(yaw), 4, "Z")
        for obj in meshes:
            obj.matrix_world = rotation @ base_matrices[obj.name]
        frame = frame_camera(camera, meshes)
        image_path = output_dir / "images" / f"{record['candidate_id']}__yaw{yaw:03d}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)
        views.append({"yaw_degrees": yaw, "image_path": str(image_path), "camera": frame})
    minimum, maximum = blender_world_bounds(meshes)
    return {
        "candidate_id": record["candidate_id"],
        "name": record.get("name"),
        "environment_category": record.get("environment_category"),
        "archive_path": record["archive_path"],
        "sha256": record.get("sha256"),
        "normalization_transform": transform,
        "normalized_bbox_size": [round(float(value), 6) for value in maximum - minimum],
        "views": views,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", default="640x400")
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args(blender_argv())
    width, height = [int(value) for value in args.resolution.lower().split("x", 1)]
    manifest = load_json(args.manifest)
    rendered = []
    for record in manifest["records"]:
        if record.get("archive_kind") != "glb":
            continue
        print("multiview", record["candidate_id"], flush=True)
        rendered.append(render_record(record, args.output_dir, width, height, args.samples))
    output = {
        "version": "physweep_visual_environment_multiview_v1",
        "source_manifest": str(args.manifest),
        "resolution": args.resolution,
        "samples": args.samples,
        "records": rendered,
    }
    write_json(args.output_dir / "render_manifest.json", output)
    print("manifest", args.output_dir / "render_manifest.json")


if __name__ == "__main__":
    main()
