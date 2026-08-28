#!/usr/bin/env python3
"""Build immutable repaired GLBs for PhysAssets visual candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tools.core.blender_runtime import clear_blender_scene
from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json


def blender_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repairs", type=Path, required=True)
    parser.add_argument("--ids", nargs="*", default=[])
    return parser.parse_args(values)


def imported_meshes():
    import bpy

    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def world_bounds(meshes) -> tuple[list[float], list[float]]:
    import mathutils

    low = mathutils.Vector((float("inf"),) * 3)
    high = mathutils.Vector((float("-inf"),) * 3)
    for obj in meshes:
        for corner in obj.bound_box:
            point = obj.matrix_world @ mathutils.Vector(corner)
            for axis in range(3):
                low[axis] = min(low[axis], point[axis])
                high[axis] = max(high[axis], point[axis])
    return list(low), list(high)


def mesh_signature(meshes) -> dict[str, Any]:
    import mathutils

    low, high = world_bounds(meshes)
    triangle_count = 0
    surface_area = 0.0
    for obj in meshes:
        mesh = obj.data
        mesh.calc_loop_triangles()
        for triangle in mesh.loop_triangles:
            points = [obj.matrix_world @ mesh.vertices[index].co for index in triangle.vertices]
            surface_area += float(
                mathutils.geometry.area_tri(points[0], points[1], points[2])
            )
            triangle_count += 1
    return {
        "mesh_count": len(meshes),
        "vertices": sum(len(obj.data.vertices) for obj in meshes),
        "polygons": sum(len(obj.data.polygons) for obj in meshes),
        "triangles": triangle_count,
        "surface_area": surface_area,
        "bounds_min": low,
        "bounds_max": high,
        "extent": [high[index] - low[index] for index in range(3)],
    }


def force_opaque_materials(meshes) -> int:
    changed: set[str] = set()
    for obj in meshes:
        for slot in obj.material_slots:
            material = slot.material
            if material is None or material.name in changed:
                continue
            material.blend_method = "OPAQUE"
            material.diffuse_color[3] = 1.0
            if material.use_nodes and material.node_tree:
                shader = material.node_tree.nodes.get("Principled BSDF")
                if shader is not None:
                    shader.inputs["Alpha"].default_value = 1.0
                    for link in list(shader.inputs["Alpha"].links):
                        material.node_tree.links.remove(link)
            changed.add(material.name)
    return len(changed)


def validate_opaque(meshes) -> None:
    for obj in meshes:
        for slot in obj.material_slots:
            material = slot.material
            if material is None:
                continue
            if material.blend_method != "OPAQUE":
                raise ValueError(f"material is not opaque after reimport: {material.name}")
            if material.use_nodes and material.node_tree:
                shader = material.node_tree.nodes.get("Principled BSDF")
                if shader is not None and shader.inputs["Alpha"].links:
                    raise ValueError(f"material alpha is still linked: {material.name}")


def compare_signatures(before: dict[str, Any], after: dict[str, Any]) -> float:
    if before["mesh_count"] != after["mesh_count"]:
        raise ValueError("repair changed mesh count")
    if before["polygons"] != after["polygons"]:
        raise ValueError("repair changed polygon count")
    if before["triangles"] != after["triangles"]:
        raise ValueError("repair changed triangle count")
    surface_error = abs(float(before["surface_area"]) - float(after["surface_area"]))
    surface_scale = max(float(before["surface_area"]), 1.0e-12)
    if surface_error / surface_scale > 1.0e-6:
        raise ValueError("repair changed world-space surface area")
    errors = []
    for key in ("bounds_min", "bounds_max"):
        errors.extend(abs(float(a) - float(b)) for a, b in zip(before[key], after[key]))
    maximum_error = max(errors, default=0.0)
    if maximum_error > 1.0e-5:
        raise ValueError(f"repair changed world bounds: {maximum_error}")
    return maximum_error


def look_at(obj, target) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def render_review(meshes, output: Path) -> list[str]:
    import bpy
    import mathutils

    low, high = world_bounds(meshes)
    center = (mathutils.Vector(low) + mathutils.Vector(high)) * 0.5
    span = max(high[index] - low[index] for index in range(3))
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = span * 1.5
    scene.eevee.gtao_factor = 1.1
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.look = "Medium High Contrast"
    scene.world.color = (0.055, 0.06, 0.07)

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.lens = 60
    camera.data.clip_start = max(1.0e-6, span * 1.0e-4)
    camera.data.clip_end = max(100.0, span * 20.0)
    scene.camera = camera
    for label, energy, size, position in (
        ("review_key", 700.0, 2.5, (-1.6, -2.2, 2.8)),
        ("review_fill", 320.0, 2.0, (2.2, -0.4, 1.2)),
    ):
        data = bpy.data.lights.new(label, "AREA")
        data.energy = energy * max(span * span, 1.0e-12)
        data.size = size * span
        light = bpy.data.objects.new(label, data)
        bpy.context.collection.objects.link(light)
        light.location = tuple(center[index] + position[index] * span for index in range(3))
        look_at(light, center)

    output.mkdir(parents=True, exist_ok=True)
    views = {
        "front": (1.8, -2.2, 1.25),
        "side": (-2.2, -1.4, 1.1),
        "top": (0.25, -0.35, 2.8),
        "back": (-1.8, 2.2, 1.25),
    }
    paths = []
    for label, factors in views.items():
        camera.location = tuple(center[index] + factors[index] * span for index in range(3))
        look_at(camera, center)
        path = output / f"{label}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        paths.append(str(path))
    return paths


def process_record(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    import bpy

    source = root / str(record["source_path"])
    output = root / str(record["output_path"])
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_source_sha = sha256(source)
    if actual_source_sha != str(record["source_sha256"]):
        raise ValueError(f"source hash mismatch: {record['visual_asset_id']}")

    clear_blender_scene(("meshes", "materials", "images", "cameras", "lights"))
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = imported_meshes()
    if not meshes:
        raise ValueError(f"source has no mesh: {record['visual_asset_id']}")
    before = mesh_signature(meshes)
    operation_counts: dict[str, int] = {}
    for operation in record["operations"]:
        if operation == "force_opaque_materials":
            operation_counts[operation] = force_opaque_materials(meshes)
        else:
            raise ValueError(f"unsupported visual repair operation: {operation}")

    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(output),
        export_format="GLB",
        export_cameras=False,
        export_lights=False,
        export_animations=False,
    )

    clear_blender_scene(("meshes", "materials", "images", "cameras", "lights"))
    bpy.ops.import_scene.gltf(filepath=str(output))
    repaired_meshes = imported_meshes()
    after = mesh_signature(repaired_meshes)
    bounds_error = compare_signatures(before, after)
    if "force_opaque_materials" in record["operations"]:
        validate_opaque(repaired_meshes)
    review_paths = render_review(repaired_meshes, output.parent / "review")
    review_records = [
        {
            "path": str(Path(path).relative_to(root)),
            "sha256": sha256(Path(path)),
        }
        for path in review_paths
    ]
    report = {
        "schema_version": "physweep_object_visual_repair_report_v1",
        "profile_id": str(record["profile_id"]),
        "visual_asset_id": str(record["visual_asset_id"]),
        "source": {
            "path": str(record["source_path"]),
            "sha256": actual_source_sha,
        },
        "admitted_visual": {
            "path": str(record["output_path"]),
            "sha256": sha256(output),
        },
        "operations": list(record["operations"]),
        "operation_counts": operation_counts,
        "reason": str(record["reason"]),
        "verification": {
            "source_signature": before,
            "reimport_signature": after,
            "maximum_bounds_error": bounds_error,
            "material_semantics": "passed",
            "review_views": review_records,
        },
    }
    write_json(output.parent / "repair.json", report)
    return report


def main() -> None:
    import numpy as np

    if "bool" not in np.__dict__:
        np.bool = np.bool_
    args = blender_args()
    root = args.root.resolve()
    manifest_path = args.repairs if args.repairs.is_absolute() else root / args.repairs
    manifest = load_json(manifest_path)
    selected = set(args.ids)
    records = [
        record
        for record in manifest["records"]
        if not selected or str(record["visual_asset_id"]) in selected
    ]
    if selected - {str(record["visual_asset_id"]) for record in records}:
        raise ValueError("requested repair id is not present in the manifest")
    reports = [process_record(root, record) for record in records]
    print(json.dumps({"repaired": len(reports), "reports": reports}, indent=2))


if __name__ == "__main__":
    main()
