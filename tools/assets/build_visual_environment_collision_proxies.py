#!/usr/bin/env python3
"""Build immutable static collision meshes for admitted visual environments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy
import bmesh
import mathutils
import numpy as np

from tools.core.hashing import sha256_file as sha256
from tools.core.blender_runtime import blender_world_bounds
from tools.core.blender_runtime import clear_blender_scene
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json

if "bool" not in np.__dict__:
    np.bool = np.bool_  # type: ignore[attr-defined]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION = "physweep_visual_environment_collision_proxy_v1"


def blender_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--profiles", type=Path, default=Path("configs/scene_mesh_profiles.json")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("assets/proxies/environment")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/visual_environment_collision_proxies.json"),
    )
    parser.add_argument("--maximum-face-count", type=int, default=80000)
    return parser.parse_args(values)


def project_relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def import_meshes(path: Path) -> list[Any]:
    clear_blender_scene(
        ("meshes", "materials", "cameras", "lights", "armatures", "actions")
    )
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj not in before and obj.type == "MESH"
    ]
    if not meshes:
        raise ValueError(f"environment has no mesh: {path}")
    bpy.context.view_layer.update()
    return meshes


def apply_reviewed_shell_edits(meshes: list[Any], asset: dict[str, Any]) -> list[Any]:
    excluded_names = set(str(value) for value in asset.get("exclude_object_names", []))
    excluded_prefixes = tuple(
        str(value) for value in asset.get("exclude_object_name_prefixes", [])
    )
    removed = [
        obj
        for obj in meshes
        if obj.name in excluded_names or obj.name.startswith(excluded_prefixes)
    ]
    for obj in removed:
        bpy.data.objects.remove(obj, do_unlink=True)
    meshes = [obj for obj in meshes if obj not in removed]
    if not meshes:
        raise ValueError(f"shell editing removed every mesh: {asset['asset_id']}")

    selectors = list(asset.get("source_space_face_exclusions", []))
    axis_index = {"x": 0, "y": 1, "z": 2}
    match_counts = [0] * len(selectors)
    retained = []
    for obj in meshes:
        obj.data = obj.data.copy()
        editable = bmesh.new()
        editable.from_mesh(obj.data)
        delete_faces = []
        for face in editable.faces:
            points = [obj.matrix_world @ vertex.co for vertex in face.verts]
            matched = False
            for index, selector in enumerate(selectors):
                axis = axis_index[str(selector["axis"])]
                coordinates = [float(point[axis]) for point in points]
                comparison = str(selector["comparison"])
                selector_matches = (
                    min(coordinates) >= float(selector["value"])
                    if comparison == "at_or_above"
                    else max(coordinates) <= float(selector["value"])
                )
                if selector_matches:
                    match_counts[index] += 1
                    matched = True
            if matched:
                delete_faces.append(face)
        if delete_faces:
            bmesh.ops.delete(editable, geom=delete_faces, context="FACES")
            editable.to_mesh(obj.data)
            obj.data.update()
        editable.free()
        if len(obj.data.polygons):
            retained.append(obj)
        else:
            bpy.data.objects.remove(obj, do_unlink=True)
    if any(count == 0 for count in match_counts):
        missing = [index for index, count in enumerate(match_counts) if count == 0]
        raise ValueError(
            f"shell selectors removed nothing for {asset['asset_id']}: {missing}"
        )
    if not retained:
        raise ValueError(f"shell face editing removed every mesh: {asset['asset_id']}")
    return retained


def triangle_count(meshes: list[Any]) -> int:
    total = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        total += len(mesh.loop_triangles)
        evaluated.to_mesh_clear()
    return total


def add_decimation(meshes: list[Any], maximum_face_count: int) -> tuple[int, float]:
    source_count = triangle_count(meshes)
    if source_count <= maximum_face_count:
        return source_count, 1.0
    ratio = max(0.01, float(maximum_face_count) / float(source_count))
    for obj in meshes:
        if len(obj.data.polygons) < 16:
            continue
        modifier = obj.modifiers.new("PhysSweepCollisionDecimate", "DECIMATE")
        modifier.decimate_type = "COLLAPSE"
        modifier.ratio = ratio
        modifier.use_collapse_triangulate = True
    bpy.context.view_layer.update()
    return source_count, ratio


def evaluated_triangles(
    meshes: list[Any],
    canonical: mathutils.Matrix,
    authoritative_floor_z_m: float,
    floor_exclusion_half_band_m: float,
    floor_minimum_abs_normal_z: float,
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[int, int, int]],
    int,
]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    removed_floor_faces = 0
    for obj in meshes:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        transform = canonical @ evaluated.matrix_world
        transformed = [transform @ vertex.co for vertex in mesh.vertices]
        mesh.calc_loop_triangles()
        referenced: dict[int, int] = {}
        for triangle in mesh.loop_triangles:
            indices = tuple(int(index) for index in triangle.vertices)
            a, b, c = (transformed[index] for index in indices)
            normal = (b - a).cross(c - a)
            if normal.length_squared <= 1.0e-20:
                continue
            centroid_z = (float(a.z) + float(b.z) + float(c.z)) / 3.0
            if (
                abs(float(normal.z)) / normal.length >= floor_minimum_abs_normal_z
                and abs(centroid_z - authoritative_floor_z_m)
                <= floor_exclusion_half_band_m
            ):
                removed_floor_faces += 1
                continue
            remapped = []
            for index in indices:
                if index not in referenced:
                    referenced[index] = len(vertices)
                    vertices.append(
                        tuple(float(value) for value in transformed[index])
                    )
                remapped.append(referenced[index])
            faces.append(tuple(remapped))
        evaluated.to_mesh_clear()
    if not vertices or not faces:
        raise ValueError("environment proxy contains no triangle geometry")
    values = np.asarray(vertices, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("environment proxy contains non-finite vertices")
    return vertices, faces, removed_floor_faces


def write_obj(
    path: Path,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as output:
        output.write("# PhysSweep reviewed visual-environment collision mesh, Z-up\n")
        output.writelines(
            f"v {x:.12g} {y:.12g} {z:.12g}\n" for x, y, z in vertices
        )
        output.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces)


def build_record(
    root: Path,
    output_root: Path,
    profile: dict[str, Any],
    maximum_face_count: int,
) -> dict[str, Any]:
    asset = profile["asset"]
    source = root / str(asset["path"])
    if sha256(source) != str(asset["sha256"]):
        raise ValueError(f"visual hash changed: {asset['asset_id']}")
    meshes = import_meshes(source)
    low, high = blender_world_bounds(meshes)
    expected_size = np.asarray(asset["source_bbox_size"], dtype=np.float64)
    actual_size = np.asarray(tuple(high - low), dtype=np.float64)
    if float(np.max(np.abs(actual_size - expected_size))) > 2.0e-4:
        raise ValueError(f"source bounds changed: {asset['asset_id']}")
    meshes = apply_reviewed_shell_edits(meshes, asset)
    source_faces, ratio = add_decimation(meshes, maximum_face_count)
    axis = {"x": 0, "y": 1, "z": 2}[str(asset["normalization_axis"])]
    scale = float(asset["target_extent_m"]) / max(float(actual_size[axis]), 1.0e-8)
    bottom_center = mathutils.Vector(
        ((low.x + high.x) / 2.0, (low.y + high.y) / 2.0, low.z)
    )
    canonical = (
        mathutils.Matrix.Scale(scale, 4)
        @ mathutils.Matrix.Translation(-bottom_center)
    )
    floor_alignment = asset["floor_alignment"]
    normalized_floor_z_m = (
        float(floor_alignment["source_floor_z"]) - float(low.z)
    ) * scale
    # Decimation can lift a broad floor shell by several centimetres. Remove
    # near-horizontal faces in a scale-bounded band so the analytic floor is
    # the sole contact authority while walls and raised structures remain.
    floor_exclusion_half_band_m = max(
        0.05,
        min(0.08, 0.012 * float(asset["target_extent_m"])),
    )
    floor_minimum_abs_normal_z = 0.94
    vertices, faces, removed_floor_faces = evaluated_triangles(
        meshes,
        canonical,
        normalized_floor_z_m,
        floor_exclusion_half_band_m,
        floor_minimum_abs_normal_z,
    )
    values = np.asarray(vertices, dtype=np.float64)
    proxy_low = values.min(axis=0)
    proxy_high = values.max(axis=0)
    proxy_extent = proxy_high - proxy_low
    asset_id = str(asset["asset_id"])
    proxy_dir = output_root / asset_id
    mesh_path = proxy_dir / "collision.obj"
    write_obj(mesh_path, vertices, faces)
    record = {
        "schema_version": VERSION,
        "profile_id": str(profile["id"]),
        "asset_id": asset_id,
        "source": {
            "visual_path": str(asset["path"]),
            "visual_sha256": str(asset["sha256"]),
            "source_bbox_size": [float(value) for value in actual_size],
        },
        "proxy": {
            "representation": "static_concave_mesh",
            "method": "reviewed_shell_blender_evaluated_triangle_mesh",
            "path": project_relative(root, mesh_path),
            "sha256": sha256(mesh_path),
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "source_face_count": source_faces,
            "authoritative_floor_face_count_removed": removed_floor_faces,
            "decimation_ratio": round(ratio, 9),
            "bounds_min_m": proxy_low.round(9).tolist(),
            "bounds_max_m": proxy_high.round(9).tolist(),
            "extents_m": proxy_extent.round(9).tolist(),
            "flags": ["GEOM_FORCE_CONCAVE_TRIMESH"],
        },
        "transform_contract": {
            "frame": "normalized_visual_asset_local_bottom_center_z_up",
            "scale": round(scale, 12),
            "source_bottom_center": [round(float(value), 12) for value in bottom_center],
            "world_pose_is_frozen_in_scene_metadata": True,
            "authoritative_floor_z_m": round(normalized_floor_z_m, 12),
            "authoritative_floor_exclusion_half_band_m": round(
                floor_exclusion_half_band_m, 12
            ),
            "authoritative_floor_minimum_abs_normal_z": round(
                floor_minimum_abs_normal_z, 12
            ),
        },
        "qa": {
            "reviewed_shell_edits_applied": True,
            "visual_and_proxy_use_same_source_transform": True,
            "finite_geometry": True,
            "remaining_authoritative_floor_band_face_count": 0,
            "maximum_face_count": int(maximum_face_count),
            "global_environment_floor_remains_collision_authority": True,
        },
    }
    write_json(proxy_dir / "proxy.json", record)
    return record


def main() -> None:
    args = blender_args()
    root = args.root.resolve()
    profiles_path = args.profiles if args.profiles.is_absolute() else root / args.profiles
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    if args.maximum_face_count < 1000:
        raise ValueError("maximum face count is too small for a room proxy")
    profiles = load_json(profiles_path)["profiles"]
    records = [
        build_record(root, output_root, profile, int(args.maximum_face_count))
        for profile in profiles
    ]
    manifest = {
        "version": VERSION,
        "policy": {
            "every_admitted_mesh_environment_has_a_static_proxy": True,
            "proxy_is_always_loaded_during_simulation": True,
            "visual_and_collision_world_pose_is_identical": True,
            "raw_visual_glb_is_never_loaded_directly_by_pybullet": True,
            "maximum_face_count_per_environment": int(args.maximum_face_count),
        },
        "source_profiles": project_relative(root, profiles_path),
        "output_root": project_relative(root, output_root),
        "counts": {"profiles": len(records), "built": len(records), "failed": 0},
        "records": records,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest["counts"], ensure_ascii=True))


if __name__ == "__main__":
    main()
