#!/usr/bin/env python3
"""Render canonical visual/proxy overlays for generated PhysAssets proxies."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from tools.core.blender_runtime import blender_world_bounds as world_bounds
from tools.core.blender_runtime import clear_blender_scene as clear_scene


def args_from_blender() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--views", type=int, choices=(1, 3), default=1)
    return parser.parse_args(values)


def look_at(obj, target) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def proxy_material():
    import bpy
    result = bpy.data.materials.new("proxy_orange")
    result.diffuse_color = (1.0, 0.12, 0.01, 0.38)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = result.diffuse_color
    shader.inputs["Roughness"].default_value = 0.35
    shader.inputs["Alpha"].default_value = 0.38
    result.blend_method = "BLEND"
    result.show_transparent_back = True
    return result


def add_proxy(item, material):
    import bpy
    size = [float(v) for v in item["size_m"]]
    if item["shape"] == "box":
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.object
        obj.scale = tuple(v * 0.5 for v in size)
    elif item["shape"] == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=0.5)
        obj = bpy.context.object
        obj.scale = tuple(size)
    else:
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=size[0] * 0.5, depth=size[2])
        obj = bpy.context.object
    obj.location = tuple(float(v) for v in item["position_m"])
    obj.rotation_euler = tuple(math.radians(float(v)) for v in item["rotation_euler_degrees"])
    obj.data.materials.append(material)
    wire = obj.modifiers.new("proxy_wire", "WIREFRAME")
    wire.thickness = 0.0025
    return obj


def render(record: dict, output: Path, view_count: int) -> None:
    import bpy
    import mathutils
    import numpy as np
    if "bool" not in np.__dict__:
        np.bool = np.bool_
    clear_scene()
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(record["source_glb"]))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    for obj in list(imported):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
            imported.remove(obj)
    meshes = [obj for obj in imported if obj.type == "MESH"]
    for obj in meshes:
        matrix = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = matrix
    low, high = world_bounds(meshes)
    center = (low + high) * 0.5
    longest = max(high[i] - low[i] for i in range(3))
    scale = float(record["transform"]["target_longest_extent_m"]) / float(longest)
    proxy_center = record["transform"]["compound_center_before_recentering_m"]
    transform = mathutils.Matrix.Translation(tuple(-float(v) for v in proxy_center)) @ mathutils.Matrix.Scale(scale, 4) @ mathutils.Matrix.Translation(-center)
    for obj in meshes:
        obj.matrix_world = transform @ obj.matrix_world

    material = proxy_material()
    proxies = [add_proxy(item, material) for item in record["proxy"]["colliders"]]
    bpy.context.view_layer.update()
    low, high = world_bounds([*meshes, *proxies])
    center = (low + high) * 0.5
    span = max(float(high[i] - low[i]) for i in range(3))
    print(record["sample_id"], "review_bounds", tuple(low), tuple(high), "span", span)

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 2.0
    scene.eevee.gtao_factor = 1.2
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.look = "Medium High Contrast"
    scene.world.color = (0.045, 0.055, 0.07)
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.lens = 58
    scene.camera = camera
    light_data = bpy.data.lights.new("key", "AREA")
    light_data.energy = 420
    light_data.size = 2.0
    light = bpy.data.objects.new("key", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (center.x - span, center.y - 1.5 * span, center.z + 2.5 * span)
    look_at(light, center)
    output.parent.mkdir(parents=True, exist_ok=True)
    views = [
        ("front", (1.7, -2.0, 1.25)),
        ("side", (-2.0, -1.3, 1.0)),
        ("top", (0.18, -0.25, 2.7)),
    ][:view_count]
    for label, factors in views:
        camera.location = tuple(center[index] + factors[index] * span for index in range(3))
        look_at(camera, center)
        target = output if view_count == 1 else output.with_name(f"{output.stem}_{label}{output.suffix}")
        scene.render.filepath = str(target)
        bpy.ops.render.render(write_still=True)


def main() -> None:
    args = args_from_blender()
    selected = set(args.ids)
    if args.ids_file:
        for line in args.ids_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                selected.add(str(json.loads(line)["sample_id"]))
    paths = sorted(args.proxy_root.glob("*/proxy.json"), key=lambda p: int(p.parent.name))
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if selected and str(record["sample_id"]) not in selected:
            continue
        render(record, args.output / f"{record['sample_id']}_{record['name'].replace(' ', '_')}.png", args.views)


if __name__ == "__main__":
    main()
