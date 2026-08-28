#!/usr/bin/env python3
"""Compile and instantiate immutable static-support proxy bindings."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.hashing import sha256_json as record_sha256
from tools.core.hashing import sha256_json_without_field


BINDING_VERSION = "physweep_static_support_binding_v1"


def _binding_sha256(binding: dict[str, Any]) -> str:
    return sha256_json_without_field(binding, "binding_sha256")


def _vector(value: Any, length: int, label: str, positive: bool = False) -> list[float]:
    result = [float(item) for item in value]
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"invalid {label}: {value}")
    if positive and min(result) <= 0.0:
        raise ValueError(f"non-positive {label}: {value}")
    return result


def _matrix4(value: Any, label: str) -> list[list[float]]:
    result = [[float(item) for item in row] for row in value]
    if (
        len(result) != 4
        or any(len(row) != 4 for row in result)
        or not all(math.isfinite(item) for row in result for item in row)
    ):
        raise ValueError(f"invalid {label}")
    return result


def _matmul4(
    left: list[list[float]], right: list[list[float]]
) -> list[list[float]]:
    return [
        [
            sum(left[row][axis] * right[axis][column] for axis in range(4))
            for column in range(4)
        ]
        for row in range(4)
    ]


def _transform_matrix(
    scale: list[float],
    position: list[float],
    quaternion_xyzw: list[float],
) -> list[list[float]]:
    x, y, z, w = quaternion_xyzw
    rotation = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
    result = [[0.0] * 4 for _ in range(4)]
    for row in range(3):
        for column in range(3):
            result[row][column] = rotation[row][column] * scale[column]
        result[row][3] = position[row]
    result[3][3] = 1.0
    return result


def _usage(record: dict[str, Any], usage_id: str) -> dict[str, Any]:
    matches = [
        usage
        for usage in record["proxy"].get("usages", [])
        if str(usage["id"]) == str(usage_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"support proxy usage is not uniquely declared for "
            f"{record['asset_id']}: {usage_id}"
        )
    usage = matches[0]
    if not bool(usage.get("active", False)):
        raise ValueError(
            f"support proxy usage is inactive for {record['asset_id']}: {usage_id}"
        )
    return usage


def _scaled_safe_surface(
    usage: dict[str, Any],
    target_size: list[float],
    target_center: list[float],
    target_plane: float,
) -> dict[str, Any]:
    reference_size = _vector(
        usage["target_size_xy_m"], 2, "reference target size", positive=True
    )
    reference_center = _vector(
        usage["target_center_xy_m"], 2, "reference target center"
    )
    reference_plane = float(usage["target_support_plane_z_m"])
    safe = usage["safe_surface"]
    safe_center = _vector(safe["center_xy_m"], 2, "safe surface center")
    safe_size = _vector(
        safe["size_xy_m"], 2, "safe surface size", positive=True
    )
    scale_xy = [
        target_size[index] / reference_size[index] for index in range(2)
    ]
    scale_z = target_plane / reference_plane
    return {
        "id": str(safe["id"]),
        "center_xy_m": [
            round(
                target_center[index]
                + (safe_center[index] - reference_center[index])
                * scale_xy[index],
                12,
            )
            for index in range(2)
        ],
        "size_xy_m": [
            round(safe_size[index] * scale_xy[index], 12)
            for index in range(2)
        ],
        "z_m": round(
            target_plane
            + (float(safe["z_m"]) - reference_plane) * scale_z,
            12,
        ),
    }


def compile_static_support_binding(
    record: dict[str, Any],
    *,
    usage_id: str,
    target_size_xy_m: list[float] | None = None,
    target_center_xy_m: list[float] | None = None,
    target_support_plane_z_m: float | None = None,
    maximum_axis_scale_ratio: float | None = None,
) -> dict[str, Any]:
    """Map one canonical support mesh into a concrete metadata support frame."""

    if record["classification"]["body_type"] != "static":
        raise ValueError(f"support proxy is not static: {record['asset_id']}")
    if record["proxy"]["representation"] != "static_concave_mesh":
        raise ValueError(f"support proxy is not an exact mesh: {record['asset_id']}")
    if record["proxy"].get("method") != "blender_evaluated_exact_triangle_mesh":
        raise ValueError(
            f"support proxy is not a Blender-evaluated mesh: {record['asset_id']}"
        )
    usage = _usage(record, usage_id)
    mesh = record["proxy"]["mesh"]
    source = mesh["support_frame"]
    source_size = _vector(source["size_xy"], 2, "source support size", positive=True)
    source_center = _vector(source["center_xy"], 2, "source support center")
    source_plane = float(source["plane_z"])
    target_size = _vector(
        target_size_xy_m
        if target_size_xy_m is not None
        else usage["target_size_xy_m"],
        2,
        "target support size",
        positive=True,
    )
    target_center = _vector(
        target_center_xy_m
        if target_center_xy_m is not None
        else usage["target_center_xy_m"],
        2,
        "target support center",
    )
    target_plane = float(
        target_support_plane_z_m
        if target_support_plane_z_m is not None
        else usage["target_support_plane_z_m"]
    )
    if not math.isfinite(source_plane) or source_plane <= 0.0:
        raise ValueError(f"invalid source support plane: {record['asset_id']}")
    if not math.isfinite(target_plane) or target_plane <= 0.0:
        raise ValueError(f"invalid target support plane: {record['asset_id']}")

    scale = [
        target_size[0] / source_size[0],
        target_size[1] / source_size[1],
        target_plane / source_plane,
    ]
    scale_ratio = max(scale) / min(scale)
    authorized_ratio = float(usage["maximum_axis_scale_ratio"])
    if (
        maximum_axis_scale_ratio is not None
        and float(maximum_axis_scale_ratio) > authorized_ratio + 1.0e-9
    ):
        raise ValueError(
            f"caller scale policy exceeds the catalog authorization for "
            f"{record['asset_id']}: {maximum_axis_scale_ratio} > {authorized_ratio}"
        )
    scale_limit = (
        authorized_ratio
        if maximum_axis_scale_ratio is None
        else float(maximum_axis_scale_ratio)
    )
    if scale_ratio > scale_limit:
        raise ValueError(
            f"support proxy scale ratio exceeds policy for {record['asset_id']}: "
            f"{scale_ratio:.6f} > {scale_limit}"
        )
    base_position = [
        target_center[0] - source_center[0] * scale[0],
        target_center[1] - source_center[1] * scale[1],
        target_plane - source_plane * scale[2],
    ]
    geometry = record["qa"]["geometry"]
    extraction = geometry["extraction"]
    canonical = _matrix4(
        geometry["canonical_transform_after_z_up"],
        "canonical visual transform",
    )
    orientation = [0.0, 0.0, 0.0, 1.0]
    target_transform = _transform_matrix(scale, base_position, orientation)
    safe_surface = _scaled_safe_surface(
        usage, target_size, target_center, target_plane
    )
    binding = {
        "schema_version": BINDING_VERSION,
        "asset_id": str(record["asset_id"]),
        "usage_id": str(usage_id),
        "catalog_record_sha256": record_sha256(record),
        "usage_contract": {
            "source": str(usage["source"]),
            "boundary_behavior": str(usage["boundary_behavior"]),
            "clear_exit_directions_xy": copy.deepcopy(
                usage.get("clear_exit_directions_xy", [])
            ),
            "maximum_axis_scale_ratio": authorized_ratio,
        },
        "classification": {"body_type": "static"},
        "representation": "static_concave_mesh",
        "capabilities": copy.deepcopy(record["capabilities"]),
        "mesh": {
            "path": str(mesh["path"]),
            "sha256": str(mesh["sha256"]),
            "scale": [round(value, 12) for value in scale],
            "base_position_m": [round(value, 12) for value in base_position],
            "base_orientation_quaternion_xyzw": orientation,
            "pybullet_flags": ["GEOM_FORCE_CONCAVE_TRIMESH"],
        },
        "visual": {
            "path": str(record["source"]["visual_path"]),
            "sha256": str(record["source"]["sha256"]),
            "reference_frame": int(extraction["reference_frame"]),
            "selected_object_names": [
                str(value) for value in geometry["selected_node_names"]
            ],
            "canonical_transform_after_z_up": canonical,
            "world_transform_matrix": _matmul4(target_transform, canonical),
            "freeze_policy": "evaluated_mesh_at_reference_frame",
        },
        "source_support_frame": copy.deepcopy(source),
        "target_support_frame": {
            "size_xy_m": target_size,
            "center_xy_m": target_center,
            "plane_z_m": target_plane,
            "safe_surface": safe_surface,
        },
        "diagnostics": {
            "axis_scale_ratio": round(scale_ratio, 9),
            "scale_limit": scale_limit,
        },
    }
    binding["binding_sha256"] = _binding_sha256(binding)
    validate_static_support_binding(binding)
    return binding


def validate_static_support_binding(binding: dict[str, Any]) -> None:
    if binding.get("schema_version") != BINDING_VERSION:
        raise ValueError("unsupported static support binding version")
    if binding.get("representation") != "static_concave_mesh":
        raise ValueError("binding is not an exact static mesh")
    if binding.get("classification", {}).get("body_type") != "static":
        raise ValueError("concave support binding must have a static body type")
    if binding.get("binding_sha256") != _binding_sha256(binding):
        raise ValueError("static support binding hash mismatch")
    if len(str(binding.get("catalog_record_sha256", ""))) != 64:
        raise ValueError("invalid static support catalog record hash")
    mesh = binding["mesh"]
    if len(str(mesh["sha256"])) != 64:
        raise ValueError("invalid support mesh hash")
    _vector(mesh["scale"], 3, "support mesh scale", positive=True)
    _vector(mesh["base_position_m"], 3, "support mesh base position")
    quaternion = _vector(
        mesh["base_orientation_quaternion_xyzw"], 4, "support mesh orientation"
    )
    if not math.isclose(
        sum(value * value for value in quaternion),
        1.0,
        rel_tol=1.0e-7,
        abs_tol=1.0e-7,
    ):
        raise ValueError("support mesh orientation is not normalized")
    if mesh.get("pybullet_flags") != ["GEOM_FORCE_CONCAVE_TRIMESH"]:
        raise ValueError("static concave mesh requires the PyBullet concave flag")
    visual = binding["visual"]
    if len(str(visual["sha256"])) != 64:
        raise ValueError("invalid support visual hash")
    if int(visual["reference_frame"]) < 0:
        raise ValueError("invalid support visual reference frame")
    if not visual.get("selected_object_names"):
        raise ValueError("support visual binding has no selected mesh components")
    _matrix4(
        visual["canonical_transform_after_z_up"],
        "canonical visual transform",
    )
    _matrix4(visual["world_transform_matrix"], "visual world transform")
    target = binding["target_support_frame"]
    size = _vector(target["size_xy_m"], 2, "target support size", positive=True)
    center = _vector(target["center_xy_m"], 2, "target support center")
    safe = target["safe_surface"]
    safe_size = _vector(
        safe["size_xy_m"], 2, "target safe surface size", positive=True
    )
    safe_center = _vector(
        safe["center_xy_m"], 2, "target safe surface center"
    )
    for axis in range(2):
        if (
            abs(safe_center[axis] - center[axis]) + safe_size[axis] / 2.0
            > size[axis] / 2.0 + 1.0e-6
        ):
            raise ValueError("safe support surface exceeds the target footprint")


def validate_static_support_binding_files(
    root: Path,
    binding: dict[str, Any],
    *,
    include_visual: bool = True,
) -> None:
    validate_static_support_binding(binding)
    sources = [("proxy mesh", binding["mesh"])]
    if include_visual:
        sources.append(("visual source", binding["visual"]))
    for label, source in sources:
        path = root / str(source["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != str(source["sha256"]):
            raise ValueError(f"support {label} hash mismatch: {path}")


def create_pybullet_static_support(
    pb: Any, root: Path, binding: dict[str, Any]
) -> int:
    """Create one zero-mass exact support from an immutable binding."""

    validate_static_support_binding_files(root, binding, include_visual=False)
    mesh = binding["mesh"]
    path = root / str(mesh["path"])
    shape = pb.createCollisionShape(
        pb.GEOM_MESH,
        fileName=str(path),
        meshScale=[float(value) for value in mesh["scale"]],
        flags=pb.GEOM_FORCE_CONCAVE_TRIMESH,
    )
    if int(shape) < 0:
        raise RuntimeError(f"PyBullet failed to load support proxy: {path}")
    body = int(
        pb.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=shape,
            basePosition=[float(value) for value in mesh["base_position_m"]],
            baseOrientation=[
                float(value)
                for value in mesh["base_orientation_quaternion_xyzw"]
            ],
        )
    )
    if body < 0:
        raise RuntimeError(f"PyBullet failed to create support body: {path}")
    return body


def blender_import_static_support_visual(
    root: Path,
    binding: dict[str, Any],
    *,
    include_all_source_meshes: bool = False,
) -> tuple[list[Any], Any]:
    """Import, freeze, and place the visual source from the same binding."""

    import bpy  # pylint: disable=import-outside-toplevel
    import mathutils  # pylint: disable=import-outside-toplevel

    validate_static_support_binding_files(root, binding)
    visual = binding["visual"]
    path = root / str(visual["path"])
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    bpy.context.scene.frame_set(int(visual["reference_frame"]))
    bpy.context.view_layer.update()
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not include_all_source_meshes:
        requested = set(str(value) for value in visual["selected_object_names"])
        meshes = [
            obj
            for obj in meshes
            if str(obj.name) in requested or str(obj.data.name) in requested
        ]
        found = {
            name
            for obj in meshes
            for name in (str(obj.name), str(obj.data.name))
            if name in requested
        }
        missing = sorted(requested - found)
        if missing:
            raise ValueError(
                f"support visual components are missing for "
                f"{binding['asset_id']}: {missing}"
            )
    if not meshes:
        raise ValueError(f"support visual contains no selected mesh: {path}")

    depsgraph = bpy.context.evaluated_depsgraph_get()
    frozen = []
    frozen_names = []
    for source in meshes:
        evaluated = source.evaluated_get(depsgraph)
        mesh = bpy.data.meshes.new_from_object(
            evaluated,
            preserve_all_data_layers=True,
            depsgraph=depsgraph,
        )
        obj = bpy.data.objects.new(str(source.name), mesh)
        bpy.context.collection.objects.link(obj)
        obj.matrix_world = evaluated.matrix_world.copy()
        frozen.append(obj)
        frozen_names.append(str(source.name))
    for obj in imported:
        if obj.name in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
    for obj, source_name in zip(frozen, frozen_names):
        obj.name = source_name
        obj["physweep_source_object_name"] = source_name

    root_object = bpy.data.objects.new(
        f"{binding['asset_id']}_immutable_support_root", None
    )
    bpy.context.collection.objects.link(root_object)
    for obj in frozen:
        world = obj.matrix_world.copy()
        obj.parent = root_object
        obj.matrix_world = world
    root_object.matrix_world = mathutils.Matrix(
        tuple(tuple(float(item) for item in row) for row in visual["world_transform_matrix"])
    )
    bpy.context.view_layer.update()
    return frozen, root_object
