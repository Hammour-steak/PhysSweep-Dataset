#!/usr/bin/env python3
"""Render deterministic interior review views for elongated environment GLBs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy
import mathutils
import numpy as np


if not hasattr(np, "bool"):
    np.bool = bool


def blender_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-length-m", type=float, default=12.0)
    parser.add_argument("--resolution", default="1280x720")
    parser.add_argument("--samples", type=int, default=16)
    return parser.parse_args(values)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.images):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def mesh_bounds(meshes: list[Any]) -> tuple[mathutils.Vector, mathutils.Vector]:
    points = [obj.matrix_world @ mathutils.Vector(corner) for obj in meshes for corner in obj.bound_box]
    low = mathutils.Vector(tuple(min(point[i] for point in points) for i in range(3)))
    high = mathutils.Vector(tuple(max(point[i] for point in points) for i in range(3)))
    return low, high


def horizontal_triangles(meshes: list[Any], normal_z_min: float = 0.94) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for obj in meshes:
        obj.data.calc_loop_triangles()
        for triangle in obj.data.loop_triangles:
            vertices = [obj.matrix_world @ obj.data.vertices[index].co for index in triangle.vertices]
            normal = (vertices[1] - vertices[0]).cross(vertices[2] - vertices[0])
            if normal.length <= 1.0e-12:
                continue
            normal.normalize()
            if float(normal.z) < normal_z_min:
                continue
            area = float((vertices[1] - vertices[0]).cross(vertices[2] - vertices[0]).length * 0.5)
            if area <= 1.0e-9:
                continue
            rows.append(
                {
                    "z": sum(float(vertex.z) for vertex in vertices) / 3.0,
                    "area": area,
                    "min_x": min(float(vertex.x) for vertex in vertices),
                    "max_x": max(float(vertex.x) for vertex in vertices),
                    "min_y": min(float(vertex.y) for vertex in vertices),
                    "max_y": max(float(vertex.y) for vertex in vertices),
                }
            )
    return rows


def cluster_surfaces(rows: list[dict[str, float]], tolerance: float) -> list[dict[str, float]]:
    clusters: list[dict[str, float]] = []
    for row in sorted(rows, key=lambda item: item["z"]):
        cluster = next(
            (item for item in reversed(clusters) if abs(item["z_mean"] - row["z"]) <= tolerance),
            None,
        )
        if cluster is None:
            cluster = {
                "z_area": 0.0,
                "area": 0.0,
                "z_mean": row["z"],
                "min_x": row["min_x"],
                "max_x": row["max_x"],
                "min_y": row["min_y"],
                "max_y": row["max_y"],
                "triangles": 0.0,
            }
            clusters.append(cluster)
        cluster["z_area"] += row["z"] * row["area"]
        cluster["area"] += row["area"]
        cluster["z_mean"] = cluster["z_area"] / cluster["area"]
        cluster["min_x"] = min(cluster["min_x"], row["min_x"])
        cluster["max_x"] = max(cluster["max_x"], row["max_x"])
        cluster["min_y"] = min(cluster["min_y"], row["min_y"])
        cluster["max_y"] = max(cluster["max_y"], row["max_y"])
        cluster["triangles"] += 1.0
    return clusters


def choose_floor(clusters: list[dict[str, float]], low_z: float, high_z: float) -> dict[str, float]:
    ceiling = low_z + 0.45 * (high_z - low_z)
    candidates = [cluster for cluster in clusters if cluster["z_mean"] <= ceiling]
    if not candidates:
        raise ValueError("no low horizontal surface candidate")
    return max(candidates, key=lambda item: item["area"])


def look_at(camera: Any, target: mathutils.Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def add_review_lighting(camera: Any) -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("ReviewWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.08, 0.08, 0.08, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35

    light_data = bpy.data.lights.new("CameraFill", type="AREA")
    light_data.energy = 650.0
    light_data.shape = "DISK"
    light_data.size = 3.0
    light = bpy.data.objects.new("CameraFill", light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = camera.location + mathutils.Vector((0.0, 0.0, 0.4))
    look_at(light, camera.location + camera.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, -4.0)))


def main() -> None:
    args = blender_args()
    width, height = (int(value) for value in args.resolution.lower().split("x", 1))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(args.asset.resolve()))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise ValueError("asset contains no mesh objects")
    low, high = mesh_bounds(meshes)
    size = high - low
    clusters = cluster_surfaces(horizontal_triangles(meshes), max(float(size.z) * 0.002, 1.0e-5))
    floor = choose_floor(clusters, float(low.z), float(high.z))

    floor_dx = floor["max_x"] - floor["min_x"]
    floor_dy = floor["max_y"] - floor["min_y"]
    long_axis = 0 if floor_dx >= floor_dy else 1
    long_extent = floor_dx if long_axis == 0 else floor_dy
    if long_extent <= 1.0e-6:
        raise ValueError("floor candidate has no usable horizontal extent")
    scale = args.target_length_m / long_extent
    floor_center = mathutils.Vector(
        (
            (floor["min_x"] + floor["max_x"]) * 0.5,
            (floor["min_y"] + floor["max_y"]) * 0.5,
            floor["z_mean"],
        )
    )
    transform = mathutils.Matrix.Scale(scale, 4) @ mathutils.Matrix.Translation(-floor_center)
    roots = [obj for obj in bpy.context.scene.objects if obj.parent is None]
    for obj in roots:
        obj.matrix_world = transform @ obj.matrix_world

    normalized_dx = floor_dx * scale
    normalized_dy = floor_dy * scale
    normalized_long = normalized_dx if long_axis == 0 else normalized_dy
    normalized_cross = normalized_dy if long_axis == 0 else normalized_dx
    camera_height = min(max(float(size.z) * scale * 0.22, 1.15), 1.65)

    camera_data = bpy.data.cameras.new("InteriorReviewCamera")
    camera_data.lens = 28.0
    camera_data.clip_start = 0.02
    camera_data.clip_end = max(80.0, normalized_long * 3.0)
    camera = bpy.data.objects.new("InteriorReviewCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = args.samples
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.view_settings.look = "Medium High Contrast"
    add_review_lighting(camera)

    offsets = (-0.34, 0.34, -0.12, 0.12)
    directions = (1.0, -1.0, 1.0, -1.0)
    rendered = []
    for index, (offset, direction) in enumerate(zip(offsets, directions)):
        along = offset * normalized_long
        target_along = along + direction * min(normalized_long * 0.42, 5.0)
        if long_axis == 0:
            camera.location = (along, 0.0, camera_height)
            target = mathutils.Vector((target_along, 0.0, camera_height * 0.72))
        else:
            camera.location = (0.0, along, camera_height)
            target = mathutils.Vector((0.0, target_along, camera_height * 0.72))
        look_at(camera, target)
        for light in (obj for obj in scene.objects if obj.type == "LIGHT"):
            light.location = camera.location + mathutils.Vector((0.0, 0.0, 0.4))
            look_at(light, target)
        path = output_dir / f"internal_view_{index:02d}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        rendered.append(str(path))

    report = {
        "schema_version": "physweep_internal_environment_review_v1",
        "asset": str(args.asset.resolve()),
        "source_bounds": {"min": list(low), "max": list(high), "size": list(size)},
        "floor_candidate": floor,
        "normalization": {
            "scale": scale,
            "long_axis": "x" if long_axis == 0 else "y",
            "length_m": normalized_long,
            "width_m": normalized_cross,
            "camera_height_m": camera_height,
        },
        "horizontal_surface_clusters": sorted(clusters, key=lambda item: item["area"], reverse=True)[:12],
        "renders": rendered,
    }
    (output_dir / "review.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
