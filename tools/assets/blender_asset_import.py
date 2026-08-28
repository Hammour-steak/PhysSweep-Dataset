"""Blender helpers for importing and inspecting admitted scene assets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256


def clear_scene() -> None:
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.materials,
    ):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def bounds(objects: list[Any]) -> tuple[Any, Any]:
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


def import_meshes(root: Path, record: dict[str, Any]) -> list[Any]:
    import bpy
    import numpy as np

    if "bool" not in np.__dict__:
        np.bool = np.bool_
    path = root / record["visual"]["path"]
    if sha256(path) != str(record["visual"]["sha256"]):
        raise ValueError(f"asset hash mismatch: {record['asset_id']}")
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
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
    if not meshes:
        raise ValueError(f"no mesh in asset: {record['asset_id']}")
    return meshes


def selected_visual_meshes(
    meshes: list[Any],
    record: dict[str, Any],
    object_names: list[str] | None = None,
) -> list[Any]:
    """Apply an audited exact-name visual partition without changing raw imports."""
    import bpy

    include_names = object_names or record["visual"].get("include_object_names")
    if not include_names:
        variants = record["visual"].get("variant_object_names", [])
        include_names = variants[:1]
    if not include_names:
        return meshes
    expected = {str(name) for name in include_names}
    by_name = {str(obj.name): obj for obj in meshes}
    missing = sorted(expected - set(by_name))
    if missing:
        raise ValueError(f"missing configured visual components for {record['asset_id']}: {missing}")
    selected = [by_name[name] for name in include_names]
    for obj in meshes:
        if obj not in selected:
            bpy.data.objects.remove(obj, do_unlink=True)
    return selected


def normalized_transform(meshes: list[Any], record: dict[str, Any]) -> Any:
    import mathutils
    import numpy as np

    visual = record["visual"]
    low, high = bounds(meshes)
    center = (low + high) * 0.5
    kind = str(record["proxy"]["kind"])
    anchor = center if kind == "dynamic_rigid" else mathutils.Vector((center.x, center.y, low.z))
    euler = [math.radians(float(v)) for v in visual.get("alignment_euler_degrees", [0, 0, 0])]
    rotation = mathutils.Euler(tuple(euler), "XYZ").to_matrix().to_4x4()
    if kind == "support_compound":
        aligned_bounds = visual.get("source_support_bounds_xy_aligned_relative")
        source_bounds = [
            float(v)
            for v in (
                aligned_bounds
                if aligned_bounds is not None
                else visual["source_support_bounds_xy"]
            )
        ]
        source_z = float(visual["source_support_plane_z_from_bottom"])
        if aligned_bounds is not None:
            corners = [
                mathutils.Vector((x, y, source_z))
                for x in source_bounds[:2]
                for y in source_bounds[2:]
            ]
        else:
            corners = []
            for x in source_bounds[:2]:
                for y in source_bounds[2:]:
                    corners.append(
                        rotation @ (mathutils.Vector((x, y, low.z + source_z)) - anchor)
                    )
        array = np.asarray(corners, dtype=float)
        surface_size = np.ptp(array[:, :2], axis=0)
        surface_center = array[:, :2].mean(axis=0)
        target_xy = np.asarray(visual["target_support_size_xy_m"], dtype=float)
        target_z = float(record["proxy"]["usable_surfaces"][0]["z_m"])
        scales = [target_xy[0] / surface_size[0], target_xy[1] / surface_size[1], target_z / source_z]
        scale = mathutils.Matrix.Diagonal(mathutils.Vector((*scales, 1.0)))
        offset = mathutils.Matrix.Translation(
            (-float(surface_center[0]) * scales[0], -float(surface_center[1]) * scales[1], 0.0)
        )
        return offset @ scale @ rotation @ mathutils.Matrix.Translation(-anchor)
    target = [float(v) for v in visual["canonical_extent_m"]]
    raw_corners = []
    for x in (low.x, high.x):
        for y in (low.y, high.y):
            for z in (low.z, high.z):
                raw_corners.append(rotation @ (mathutils.Vector((x, y, z)) - anchor))
    aligned = np.ptp(np.asarray(raw_corners, dtype=float), axis=0)
    ratios = np.asarray(target, dtype=float) / np.maximum(aligned, 1.0e-9)
    scale_value = float(np.median(ratios))
    if float(np.max(np.abs(ratios - scale_value))) > max(0.006, scale_value * 0.03):
        raise ValueError(f"non-uniform canonical fit: {record['asset_id']} ratios={ratios.tolist()}")
    return mathutils.Matrix.Scale(scale_value, 4) @ rotation @ mathutils.Matrix.Translation(-anchor)


def patch_numpy_for_blender_gltf() -> None:
    """Restore the NumPy alias expected by Blender's bundled glTF importer."""

    import numpy as np  # pylint: disable=import-outside-toplevel

    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]


def mesh_world_bounds(objects: list[Any]) -> tuple[Any, Any]:
    """Return world-space bounds for mesh objects, with a stable empty default."""

    import mathutils  # pylint: disable=import-outside-toplevel

    minimum = mathutils.Vector((float("inf"), float("inf"), float("inf")))
    maximum = mathutils.Vector((float("-inf"), float("-inf"), float("-inf")))
    found = False
    for obj in objects:
        if obj.type != "MESH":
            continue
        found = True
        for corner in obj.bound_box:
            point = obj.matrix_world @ mathutils.Vector(corner)
            minimum.x = min(minimum.x, point.x)
            minimum.y = min(minimum.y, point.y)
            minimum.z = min(minimum.z, point.z)
            maximum.x = max(maximum.x, point.x)
            maximum.y = max(maximum.y, point.y)
            maximum.z = max(maximum.z, point.z)
    if not found:
        minimum = mathutils.Vector((0.0, 0.0, 0.0))
        maximum = mathutils.Vector((1.0, 1.0, 1.0))
    return minimum, maximum


def meshes_have_image_texture(objects: list[Any]) -> bool:
    """Return whether any mesh material contains a loaded image texture."""

    for obj in objects:
        if obj.type != "MESH":
            continue
        for slot in obj.material_slots:
            material = slot.material
            if not material or not material.use_nodes:
                continue
            for node in material.node_tree.nodes:
                if node.bl_idname == "ShaderNodeTexImage" and getattr(
                    node, "image", None
                ):
                    return True
    return False
