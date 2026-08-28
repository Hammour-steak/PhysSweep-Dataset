#!/usr/bin/env python3
"""Render visual/proxy overlays at three sampled PyBullet probe states."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from tools.assets.blender_asset_import import (
    bounds,
    clear_scene,
    import_meshes,
    normalized_transform,
    selected_visual_meshes,
)
from tools.core.json_io import read_json as load_json
from tools.rendering.blender_scene import look_at


def blender_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="*", default=[])
    return parser.parse_args(values)


def material(name: str, color: tuple[float, float, float, float], roughness: float = 0.5) -> Any:
    import bpy

    result = bpy.data.materials.new(name)
    result.diffuse_color = color
    result.use_nodes = True
    principled = result.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = roughness
    principled.inputs["Alpha"].default_value = color[3]
    if color[3] < 1.0:
        result.blend_method = "BLEND"
        result.show_transparent_back = True
        result.use_screen_refraction = True
    return result


def collider_object(collider: dict[str, Any], proxy_material: Any) -> Any:
    import bpy

    size = [float(v) for v in collider["size_m"]]
    shape = str(collider["shape"])
    if shape == "box":
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.object
        obj.scale = tuple(value * 0.5 for value in size)
    elif shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.5)
        obj = bpy.context.object
        obj.scale = tuple(size)
    elif shape == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=size[0] * 0.5, depth=size[2])
        obj = bpy.context.object
    else:
        raise ValueError(shape)
    obj.name = f"proxy_{collider['id']}"
    obj.location = tuple(float(v) for v in collider["position_m"])
    obj.rotation_euler = tuple(math.radians(float(v)) for v in collider["rotation_euler_degrees"])
    obj.data.materials.append(proxy_material)
    wire = obj.modifiers.new("proxy_wire", "WIREFRAME")
    wire.thickness = 0.004
    return obj


def add_floor(floor_material: Any) -> Any:
    import bpy

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -0.03))
    floor = bpy.context.object
    floor.name = "review_floor"
    floor.scale = (2.4, 2.0, 0.03)
    floor.data.materials.append(floor_material)
    return floor


def add_probe_ball(radius: float, ball_material: Any) -> Any:
    import bpy

    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius)
    ball = bpy.context.object
    ball.name = "probe_ball"
    ball.data.materials.append(ball_material)
    return ball


def setup_stage(objects: list[Any], trajectory: list[dict[str, Any]], output: Path) -> None:
    import bpy
    import mathutils

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3.0
    scene.eevee.gtao_factor = 1.25
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.look = "Medium High Contrast"
    scene.world.color = (0.055, 0.065, 0.075)
    low, high = bounds(objects)
    for sample in trajectory:
        point = sample["position_m"]
        for axis in range(3):
            low[axis] = min(low[axis], float(point[axis]))
            high[axis] = max(high[axis], float(point[axis]))
    center = (low + high) * 0.5
    span = max(float(high.x - low.x), float(high.y - low.y), 0.55)
    height = max(float(high.z - low.z), 0.45)
    bpy.ops.object.camera_add(location=(span * 1.10, -span * 1.40, max(0.75, height * 1.05)))
    camera = bpy.context.object
    camera.data.lens = 52
    look_at(camera, mathutils.Vector((center.x, center.y, max(0.18, center.z * 0.8))))
    scene.camera = camera
    key_data = bpy.data.lights.new("key", "AREA")
    key_data.energy = 850
    key_data.size = 3.0
    key = bpy.data.objects.new("key", key_data)
    bpy.context.collection.objects.link(key)
    key.location = (-1.8, -2.2, 3.2)
    look_at(key, center)
    fill_data = bpy.data.lights.new("fill", "AREA")
    fill_data.energy = 420
    fill_data.size = 2.5
    fill = bpy.data.objects.new("fill", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (2.0, 0.8, 2.0)
    look_at(fill, center)
    output.mkdir(parents=True, exist_ok=True)


def pose_matrix(sample: dict[str, Any]) -> Any:
    import mathutils

    x, y, z, w = [float(v) for v in sample["quaternion_xyzw"]]
    rotation = mathutils.Quaternion((w, x, y, z)).to_matrix().to_4x4()
    return mathutils.Matrix.Translation(tuple(float(v) for v in sample["position_m"])) @ rotation


def render_record(root: Path, output: Path, record: dict[str, Any], probe: dict[str, Any]) -> list[str]:
    import bpy

    clear_scene()
    meshes = import_meshes(root, record)
    meshes = selected_visual_meshes(meshes, record)
    transform = normalized_transform(meshes, record)
    for obj in meshes:
        obj.matrix_world = transform @ obj.matrix_world
    proxy_material = material("proxy_orange", (1.0, 0.18, 0.02, 0.34), 0.42)
    floor_material = material("floor", (0.20, 0.23, 0.25, 1.0), 0.72)
    ball_material = material("probe_ball", (0.10, 0.58, 1.0, 1.0), 0.28)
    proxy_objects = [collider_object(item, proxy_material) for item in record["proxy"]["colliders"]]
    kind = str(record["proxy"]["kind"])
    ball = None
    if kind != "dynamic_rigid":
        ball = add_probe_ball(float(probe.get("probe_radius_m", 0.04)), ball_material)
    add_floor(floor_material)
    samples = probe["samples"]
    setup_stage([*meshes, *proxy_objects], samples, output)
    paths = []
    normalized_mesh_matrices = [obj.matrix_world.copy() for obj in meshes]
    normalized_proxy_matrices = [obj.matrix_world.copy() for obj in proxy_objects]
    labels = ("initial", "contact", "final")
    for label, sample in zip(labels, samples):
        if kind == "dynamic_rigid":
            body = pose_matrix(sample)
            for obj, base in zip(meshes, normalized_mesh_matrices):
                obj.matrix_world = body @ base
            for obj, base in zip(proxy_objects, normalized_proxy_matrices):
                obj.matrix_world = body @ base
        else:
            ball.location = tuple(float(v) for v in sample["position_m"])
            x, y, z, w = [float(v) for v in sample["quaternion_xyzw"]]
            ball.rotation_mode = "QUATERNION"
            ball.rotation_quaternion = (w, x, y, z)
        path = output / f"{label}.png"
        bpy.context.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        paths.append(str(path))
    return paths


def main() -> None:
    args = blender_args()
    root = args.root.resolve()
    registry = load_json(args.registry.resolve())
    probes = load_json(args.probes.resolve())
    by_probe = {str(item["asset_id"]): item for item in probes["records"]}
    report_path = args.output.resolve() / "render_report.json"
    report_by_id = {}
    if args.ids and report_path.exists():
        report_by_id = {
            str(item["asset_id"]): item
            for item in load_json(report_path).get("records", [])
        }
    selected_ids = set(args.ids)
    for record in registry["records"]:
        asset_id = str(record["asset_id"])
        if selected_ids and asset_id not in selected_ids:
            continue
        if not selected_ids and not record["admission"].get("sampling_enabled", False):
            continue
        probe = by_probe[asset_id]
        paths = render_record(root, args.output.resolve() / asset_id, record, probe)
        report_by_id[asset_id] = {
            "asset_id": asset_id,
            "proxy_kind": record["proxy"]["kind"],
            "images": paths,
        }
    report_path.write_text(
        json.dumps(
            {"records": [report_by_id[key] for key in sorted(report_by_id)]},
            indent=2,
        )
        + "\n"
    )
    print(report_path)


if __name__ == "__main__":
    main()
