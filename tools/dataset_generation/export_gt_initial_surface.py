#!/usr/bin/env python3
"""Export the exact visual t=0 scene surface from bound PhysSweep metadata.

Run this script inside the bundled Blender runtime. It rebuilds only the initial
scene state; no future trajectory is loaded or inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import bpy
import mathutils
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
CONTRACT_DIR = TOOLS_DIR / "dataset_contract"
for candidate in (SCRIPT_DIR, CONTRACT_DIR, TOOLS_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import render_pybullet_rigid as rigid_renderer
from gt_scene_input import (
    DEFAULT_ENVIRONMENT_POINTS,
    DEFAULT_OBJECT_POINTS,
    ENVIRONMENT_SURFACE_POLICY,
    GT_SURFACE_SCHEMA,
    compile_model_scene_condition,
    ground_collider_id,
    interaction_collider_ids,
    sample_metric_surface_indices,
    write_gt_surface,
)


def _argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _normalize(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(lengths, 1e-12)


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    )


def _camera_from_world(camera: dict[str, Any]) -> np.ndarray:
    position = np.asarray(camera["position_m"], dtype=np.float64)
    target = np.asarray(camera["target_m"], dtype=np.float64)
    forward = _normalize((target - position)[None])[0]
    right = _normalize(np.cross(forward, np.asarray([0.0, 0.0, 1.0]))[None])[0]
    up = _normalize(np.cross(right, forward)[None])[0]
    rotation = np.stack([right, up, forward])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ position
    return transform


def _camera_intrinsics(camera: dict[str, Any], width: int, height: int) -> np.ndarray:
    focal = float(camera["focal_length_mm"])
    sensor_width = float(camera["sensor_width_mm"])
    focal_px = focal / sensor_width * width
    return np.asarray(
        [[focal_px, 0.0, width * 0.5], [0.0, focal_px, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return (homogeneous @ transform.T)[:, :3]


def _allocate(total: int, weights: np.ndarray) -> np.ndarray:
    if total <= 0 or len(weights) == 0:
        raise ValueError(
            "point allocation requires a positive total and non-empty weights"
        )
    weights = np.sqrt(np.maximum(weights.astype(np.float64), 1e-12))
    minimum = min(64, max(1, total // (len(weights) * 4)))
    allocation = np.full(len(weights), minimum, dtype=np.int64)
    if int(allocation.sum()) > total:
        allocation[:] = 0
    remaining = total - int(allocation.sum())
    raw = remaining * weights / weights.sum()
    allocation += np.floor(raw).astype(np.int64)
    remainder = total - int(allocation.sum())
    if remainder:
        order = np.argsort(-(raw - np.floor(raw)))
        allocation[order[:remainder]] += 1
    return allocation


def _mesh_area(obj: Any, depsgraph: Any) -> float:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        mesh.calc_loop_triangles()
        vertices = np.asarray(
            [tuple(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices],
            dtype=np.float64,
        )
        triangles = np.asarray(
            [triangle.vertices[:] for triangle in mesh.loop_triangles], dtype=np.int64
        )
        if not len(triangles):
            return 0.0
        values = vertices[triangles]
        return float(
            (
                np.linalg.norm(
                    np.cross(values[:, 1] - values[:, 0], values[:, 2] - values[:, 0]),
                    axis=1,
                )
                * 0.5
            ).sum()
        )
    finally:
        evaluated.to_mesh_clear()


class _ImageCache:
    def __init__(self) -> None:
        self._arrays: dict[int, np.ndarray] = {}

    def load(self, image: Any, maximum_size: int = 768) -> np.ndarray:
        key = int(image.as_pointer())
        if key in self._arrays:
            return self._arrays[key]
        width, height = [int(value) for value in image.size]
        if min(width, height) <= 0:
            raise ValueError(f"Blender image has no pixels: {image.name}")
        if max(width, height) > maximum_size:
            scale = maximum_size / max(width, height)
            image.scale(max(1, round(width * scale)), max(1, round(height * scale)))
            width, height = [int(value) for value in image.size]
        flat = np.empty(width * height * 4, dtype=np.float32)
        image.pixels.foreach_get(flat)
        pixels = flat.reshape(height, width, 4)
        pixels[:, :, :3] = _linear_to_srgb(pixels[:, :, :3])
        self._arrays[key] = pixels
        return pixels


def _downscale_review_images(maximum_size: int = 768) -> dict[str, int]:
    resized = 0
    available = 0
    for image in bpy.data.images:
        width, height = [int(value) for value in image.size]
        if min(width, height) <= 0:
            continue
        available += 1
        if max(width, height) <= maximum_size:
            continue
        scale = maximum_size / max(width, height)
        image.scale(max(1, round(width * scale)), max(1, round(height * scale)))
        resized += 1
    return {"available_image_count": available, "resized_image_count": resized}


def _find_image_node(socket: Any, visited: set[int] | None = None) -> Any | None:
    visited = visited or set()
    for link in socket.links:
        node = link.from_node
        pointer = int(node.as_pointer())
        if pointer in visited:
            continue
        visited.add(pointer)
        if node.bl_idname == "ShaderNodeTexImage" and node.image is not None:
            return node
        for candidate in node.inputs:
            found = _find_image_node(candidate, visited)
            if found is not None:
                return found
    return None


def _base_color_socket(material: Any) -> Any | None:
    if material is None or not material.use_nodes:
        return None
    output = next(
        (
            node
            for node in material.node_tree.nodes
            if node.bl_idname == "ShaderNodeOutputMaterial" and node.is_active_output
        ),
        None,
    )
    candidates = []
    if output is not None and output.inputs.get("Surface"):
        candidates.extend(link.from_node for link in output.inputs["Surface"].links)
    candidates.extend(
        node
        for node in material.node_tree.nodes
        if node.bl_idname == "ShaderNodeBsdfPrincipled"
    )
    for node in candidates:
        if node.bl_idname == "ShaderNodeBsdfPrincipled" and node.inputs.get(
            "Base Color"
        ):
            return node.inputs["Base Color"]
    return None


def _mapped_uv(image_node: Any, uv: np.ndarray) -> np.ndarray:
    vector = image_node.inputs.get("Vector")
    if vector is None or not vector.links:
        return uv
    node = vector.links[0].from_node
    if node.bl_idname != "ShaderNodeMapping":
        return uv
    scale = np.asarray(node.inputs["Scale"].default_value[:2], dtype=np.float64)
    location = np.asarray(node.inputs["Location"].default_value[:2], dtype=np.float64)
    angle = float(node.inputs["Rotation"].default_value[2])
    mapped = uv * scale
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    return mapped @ rotation.T + location


def _sample_image(
    image: np.ndarray, uv: np.ndarray, extension: str = "REPEAT"
) -> np.ndarray:
    height, width = image.shape[:2]
    if extension == "REPEAT":
        coordinates = np.mod(uv, 1.0)
    else:
        coordinates = np.clip(uv, 0.0, 1.0)
    x = np.clip(np.rint(coordinates[:, 0] * (width - 1)).astype(np.int64), 0, width - 1)
    y = np.clip(
        np.rint(coordinates[:, 1] * (height - 1)).astype(np.int64), 0, height - 1
    )
    return image[y, x, :3]


def _material_colors(
    material: Any,
    uv: np.ndarray,
    cache: _ImageCache,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(uv)
    socket = _base_color_socket(material)
    fallback = np.asarray(
        tuple(material.diffuse_color[:3])
        if material is not None
        else (0.55, 0.55, 0.55),
        dtype=np.float32,
    )
    fallback = _linear_to_srgb(fallback)
    colors = np.repeat(fallback[None], count, axis=0)
    valid = np.zeros(count, dtype=np.uint8)
    if socket is None:
        return colors, valid
    image_node = _find_image_node(socket)
    if image_node is None:
        value = np.asarray(socket.default_value[:3], dtype=np.float32)
        return np.repeat(_linear_to_srgb(value)[None], count, axis=0), np.ones(
            count, dtype=np.uint8
        )
    try:
        image = cache.load(image_node.image)
        colors = _sample_image(
            image,
            _mapped_uv(image_node, uv),
            extension=str(image_node.extension),
        )
        valid[:] = 1
    except (RuntimeError, ValueError):
        pass
    return colors.astype(np.float32), valid


def _sample_mesh(
    obj: Any,
    count: int,
    rng: np.random.Generator,
    depsgraph: Any,
    image_cache: _ImageCache,
) -> dict[str, np.ndarray]:
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)
    try:
        mesh.calc_loop_triangles()
        vertices = np.asarray(
            [tuple(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices],
            dtype=np.float64,
        )
        triangles = np.asarray(
            [triangle.vertices[:] for triangle in mesh.loop_triangles], dtype=np.int64
        )
        loops = np.asarray(
            [triangle.loops[:] for triangle in mesh.loop_triangles], dtype=np.int64
        )
        polygon_indices = np.asarray(
            [triangle.polygon_index for triangle in mesh.loop_triangles], dtype=np.int64
        )
        if not len(triangles):
            raise ValueError(f"mesh has no triangles: {obj.name}")
        triangle_vertices = vertices[triangles]
        cross = np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        )
        areas = np.linalg.norm(cross, axis=1) * 0.5
        usable = areas > 1e-12
        if not usable.any():
            raise ValueError(f"mesh has no positive-area triangles: {obj.name}")
        probabilities = areas / areas.sum()
        selected = rng.choice(len(triangles), size=count, replace=True, p=probabilities)
        chosen = triangle_vertices[selected]
        u = rng.random(count)
        v = rng.random(count)
        folded = u + v > 1.0
        u[folded] = 1.0 - u[folded]
        v[folded] = 1.0 - v[folded]
        barycentric = np.column_stack([1.0 - u - v, u, v])
        points = np.einsum("ni,nij->nj", barycentric, chosen)
        normals = _normalize(cross[selected])

        uv = np.zeros((count, 2), dtype=np.float64)
        active_uv = mesh.uv_layers.active
        if active_uv is not None:
            uv_values = np.asarray(
                [tuple(active_uv.data[index].uv) for index in loops[selected].ravel()],
                dtype=np.float64,
            ).reshape(count, 3, 2)
            uv = np.einsum("ni,nij->nj", barycentric, uv_values)
        material_indices = np.asarray(
            [
                mesh.polygons[index].material_index
                for index in polygon_indices[selected]
            ],
            dtype=np.int64,
        )
        colors = np.zeros((count, 3), dtype=np.float32)
        color_valid = np.zeros(count, dtype=np.uint8)
        materials = list(mesh.materials)
        for material_index in np.unique(material_indices):
            mask = material_indices == material_index
            material = (
                materials[int(material_index)]
                if 0 <= material_index < len(materials)
                else None
            )
            colors[mask], color_valid[mask] = _material_colors(
                material, uv[mask], image_cache
            )
        return {
            "xyz_world": points.astype(np.float32),
            "normal_world": normals.astype(np.float32),
            "rgb": colors,
            "rgb_valid": color_valid,
        }
    finally:
        evaluated.to_mesh_clear()


def _sample_payload(
    values: dict[str, np.ndarray],
    count: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    available = len(next(iter(values.values())))
    if available < count:
        raise ValueError(f"point payload has only {available} points; {count} required")
    indices = np.sort(rng.choice(available, size=count, replace=False))
    return {key: value[indices] for key, value in values.items()}


def _concatenate(records: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = records[0].keys()
    return {key: np.concatenate([record[key] for record in records]) for key in keys}


def _load_first_frame(path: Path, cache: _ImageCache) -> np.ndarray:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        return cache.load(
            image, maximum_size=max(int(image.size[0]), int(image.size[1]))
        ).copy()
    finally:
        bpy.data.images.remove(image)


def _render_current_scene_rgb(
    cache: _ImageCache, image_size: tuple[int, int]
) -> np.ndarray:
    """Render the current visibility state and return bottom-origin RGB pixels."""
    scene = bpy.context.scene
    original = {
        "filepath": scene.render.filepath,
        "file_format": scene.render.image_settings.file_format,
        "color_mode": scene.render.image_settings.color_mode,
        "color_depth": scene.render.image_settings.color_depth,
        "resolution_x": scene.render.resolution_x,
        "resolution_y": scene.render.resolution_y,
        "resolution_percentage": scene.render.resolution_percentage,
    }
    with tempfile.TemporaryDirectory(prefix="physweep_gt_environment_") as directory:
        path = Path(directory) / "environment.png"
        try:
            scene.render.filepath = str(path)
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_mode = "RGB"
            scene.render.image_settings.color_depth = "8"
            scene.render.resolution_x, scene.render.resolution_y = image_size
            scene.render.resolution_percentage = 100
            bpy.ops.render.render(write_still=True)
            return _load_first_frame(path, cache)
        finally:
            scene.render.filepath = original["filepath"]
            scene.render.image_settings.file_format = original["file_format"]
            scene.render.image_settings.color_mode = original["color_mode"]
            scene.render.image_settings.color_depth = original["color_depth"]
            scene.render.resolution_x = original["resolution_x"]
            scene.render.resolution_y = original["resolution_y"]
            scene.render.resolution_percentage = original["resolution_percentage"]


def _set_hidden(obj: Any, hidden: bool) -> None:
    obj.hide_render = hidden
    obj.hide_viewport = hidden
    obj.hide_set(hidden)
    bpy.context.view_layer.update()


def _initial_frame_mask(
    xyz_world: np.ndarray,
    camera_from_world: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    clip_start: float,
    clip_end: float,
) -> np.ndarray:
    xyz_camera = _transform_points(xyz_world, camera_from_world)
    safe_z = np.maximum(xyz_camera[:, 2], 1e-6)
    u = intrinsics[0, 0] * xyz_camera[:, 0] / safe_z + intrinsics[0, 2]
    v = intrinsics[1, 2] - intrinsics[1, 1] * xyz_camera[:, 1] / safe_z
    width, height = image_size
    margin_px = 0.5
    return (
        (xyz_camera[:, 2] > max(0.03, clip_start))
        & (xyz_camera[:, 2] < clip_end)
        & (u >= margin_px)
        & (u < width - margin_px)
        & (v >= margin_px)
        & (v < height - margin_px)
    )


def _filtered_payload(
    payload: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    return {key: value[mask] for key, value in payload.items()}


def _ground_candidate_mask(
    payload: dict[str, np.ndarray],
    descriptor: dict[str, Any],
    camera_from_world: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    clip_start: float,
    clip_end: float,
) -> np.ndarray:
    contract = descriptor["ground_contract"]
    expected_normal = np.asarray(contract["plane_normal_world"], dtype=np.float64)
    orientation_mask = np.abs(payload["normal_world"] @ expected_normal) >= 0.8
    mask = (
        _initial_frame_mask(
            payload["xyz_world"],
            camera_from_world,
            intrinsics,
            image_size,
            clip_start,
            clip_end,
        )
        & orientation_mask
    )
    xyz = payload["xyz_world"].astype(np.float64)
    plane_point = np.asarray(contract["plane_point_world_m"], dtype=np.float64)
    plane_normal = np.asarray(contract["plane_normal_world"], dtype=np.float64)
    plane_distance = np.abs((xyz - plane_point) @ plane_normal)
    mask &= plane_distance <= float(contract["plane_tolerance_m"])
    return mask


def _analytic_contact_plane_contract(collider: dict[str, Any]) -> dict[str, Any]:
    rotation = mathutils.Euler(
        tuple(
            math.radians(float(value)) for value in collider["rotation_euler_degrees"]
        ),
        "XYZ",
    ).to_matrix()
    normal = np.asarray(rotation @ mathutils.Vector((0.0, 0.0, 1.0)))
    normal = _normalize(normal[None])[0]
    position = np.asarray(collider["position_m"], dtype=np.float64)
    plane_point = position + normal * float(collider["size_m"][2]) * 0.5
    return {
        "plane_point_world_m": plane_point.tolist(),
        "plane_normal_world": normal.tolist(),
        "plane_tolerance_m": 0.004,
    }


def _sample_analytic_box_top(
    collider: dict[str, Any],
    count: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Sample the exact top contact face of a box collider."""
    size = np.asarray(collider["size_m"], dtype=np.float64)
    position = np.asarray(collider["position_m"], dtype=np.float64)
    rotation = np.asarray(
        mathutils.Euler(
            tuple(
                math.radians(float(value))
                for value in collider["rotation_euler_degrees"]
            ),
            "XYZ",
        ).to_matrix(),
        dtype=np.float64,
    )
    local = np.empty((count, 3), dtype=np.float64)
    local[:, 0] = rng.uniform(-size[0] * 0.5, size[0] * 0.5, count)
    local[:, 1] = rng.uniform(-size[1] * 0.5, size[1] * 0.5, count)
    local[:, 2] = size[2] * 0.5
    xyz_world = local @ rotation.T + position
    normal = _normalize((rotation @ np.asarray([0.0, 0.0, 1.0]))[None])[0]
    return {
        "xyz_world": xyz_world.astype(np.float32),
        "normal_world": np.repeat(normal[None], count, axis=0).astype(np.float32),
        "rgb": np.zeros((count, 3), dtype=np.float32),
        "rgb_valid": np.zeros(count, dtype=np.uint8),
    }


def _sample_environment_first_hits(
    camera_from_world: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    camera_position: np.ndarray,
    clip_start: float,
    clip_end: float,
    rendered_rgb: np.ndarray,
    target: int,
    rng: np.random.Generator,
    depsgraph: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Sample the first visible environment surface at unique camera pixels."""
    width, height = image_size
    candidate_count = min(width * height, max(target * 3, target + 4096))
    pixels = rng.choice(width * height, size=candidate_count, replace=False)
    rotation_world_from_camera = camera_from_world[:3, :3].T
    scene = bpy.context.scene
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    for flat in pixels:
        pixel_y, pixel_x = divmod(int(flat), width)
        ray_camera = np.asarray(
            [
                (pixel_x + 0.5 - intrinsics[0, 2]) / intrinsics[0, 0],
                (intrinsics[1, 2] - pixel_y - 0.5) / intrinsics[1, 1],
                1.0,
            ],
            dtype=np.float64,
        )
        direction_world = _normalize((ray_camera @ rotation_world_from_camera.T)[None])[
            0
        ]
        ray_start = max(0.03, clip_start) + 1e-4
        ray_origin = camera_position + direction_world * ray_start
        hit, location, normal, _, hit_object, _ = scene.ray_cast(
            depsgraph,
            mathutils.Vector(tuple(float(value) for value in ray_origin)),
            mathutils.Vector(tuple(float(value) for value in direction_world)),
            distance=max(0.0, clip_end - ray_start),
        )
        if not hit:
            continue
        location_array = np.asarray(location, dtype=np.float64)
        location_camera = _transform_points(
            location_array[None], camera_from_world
        )[0]
        if location_camera[2] <= max(0.03, clip_start):
            continue
        static_id = str(hit_object.get("physweep_static_id", ""))
        if not static_id:
            raise ValueError(f"visible environment mesh has no static id: {hit_object.name}")
        rgb = rendered_rgb[height - 1 - pixel_y, pixel_x, :3]
        records.append(
            (
                location_array.astype(np.float32),
                np.asarray(normal, dtype=np.float32),
                np.asarray(rgb, dtype=np.float32),
                static_id,
            )
        )
    minimum_context_points = min(target, 4096)
    if len(records) < minimum_context_points:
        raise ValueError(
            f"initial-camera first-hit sampling produced {len(records)} points; "
            f"{minimum_context_points} required for context"
        )
    output_count = min(target, len(records))
    candidate_xyz_world = np.stack([record[0] for record in records])
    candidate_scene_part_id = np.asarray(
        [record[3] for record in records], dtype=np.str_
    )
    selected, metric_sampling = sample_metric_surface_indices(
        candidate_xyz_world,
        candidate_scene_part_id,
        output_count,
        rng,
        retained_part_ids=tuple(sorted(set(candidate_scene_part_id.tolist()))),
    )
    xyz_world = candidate_xyz_world[selected]
    normal_world = _normalize(np.stack([records[index][1] for index in selected]))
    rgb = np.stack([records[index][2] for index in selected])
    scene_part_id = candidate_scene_part_id[selected]
    return {
        "xyz_world": xyz_world.astype(np.float32),
        "normal_world": normal_world.astype(np.float32),
        "rgb": rgb.astype(np.float32),
        "rgb_valid": np.ones(output_count, dtype=np.uint8),
        "body_id": np.zeros(output_count, dtype=np.int16),
        "scene_part_id": scene_part_id,
    }, {
        "method": "unique_camera_pixel_first_hit_then_metric_surface_equalization",
        "candidate_pixel_count": int(candidate_count),
        "hit_pixel_count": len(records),
        "metric_sampling": metric_sampling,
        "visible_scene_part_point_counts": {
            part_id: int(np.count_nonzero(scene_part_id == part_id))
            for part_id in sorted(set(scene_part_id.tolist()))
        },
        "requested_point_count": int(target),
        "output_point_count": int(output_count),
    }


def _sample_ground_completion_candidates(
    descriptors: list[dict[str, Any]],
    ground_id: str,
    ground_collider: dict[str, Any],
    ground_contract: dict[str, Any],
    camera_from_world: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    clip_start: float,
    clip_end: float,
    target: int,
    rng: np.random.Generator,
    depsgraph: Any,
    image_cache: _ImageCache,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Sample the single semantic ground's top plane inside the initial frame."""
    static_descriptors = [item for item in descriptors if item["ground_completion"]]
    if len(static_descriptors) > 1:
        raise ValueError(
            f"scene has multiple rendered semantic grounds: {len(static_descriptors)}"
        )
    descriptor = {
        "static_id": ground_id,
        "ground_contract": ground_contract,
    }
    if static_descriptors:
        rendered = static_descriptors[0]
        descriptor.update(rendered)
        if _mesh_area(rendered["object"], depsgraph) <= 1e-12:
            raise ValueError("semantic ground mesh has no surface area")

        def sample_candidates(count: int) -> dict[str, np.ndarray]:
            return _sample_mesh(
                rendered["object"], count, rng, depsgraph, image_cache
            )

        ground_source = "rendered_ground_mesh"
        ground_mesh = str(rendered["object"].name)
    else:

        def sample_candidates(count: int) -> dict[str, np.ndarray]:
            return _sample_analytic_box_top(ground_collider, count, rng)

        ground_source = "exact_simulation_box_collider"
        ground_mesh = None

    pilot_count = max(2048, min(4096, target // 6))
    pilot = sample_candidates(pilot_count)
    mask = _ground_candidate_mask(
        pilot,
        descriptor,
        camera_from_world,
        intrinsics,
        image_size,
        clip_start,
        clip_end,
    )
    accepted = _filtered_payload(pilot, mask)
    if not len(accepted["xyz_world"]):
        raise ValueError("semantic ground has no top plane in the initial frame")
    payloads = [accepted]
    accepted_count = len(accepted["xyz_world"])
    acceptance = float(mask.mean())
    total_candidate_count = pilot_count
    attempts = 0
    while accepted_count < target and attempts < 8:
        needed = target - accepted_count
        batch_count = max(2048, math.ceil(needed * 1.35 / max(acceptance, 1e-4)))
        batch_count = min(batch_count, 262144)
        batch = sample_candidates(batch_count)
        total_candidate_count += batch_count
        mask = _ground_candidate_mask(
            batch,
            descriptor,
            camera_from_world,
            intrinsics,
            image_size,
            clip_start,
            clip_end,
        )
        current = _filtered_payload(batch, mask)
        if len(current["xyz_world"]):
            payloads.append(current)
            accepted_count += len(current["xyz_world"])
        acceptance = max(1e-4, 0.5 * acceptance + 0.5 * float(mask.mean()))
        attempts += 1
    minimum_required = min(target, DEFAULT_ENVIRONMENT_POINTS)
    if accepted_count < minimum_required:
        raise ValueError(
            f"ground sampling produced {accepted_count} points; "
            f"{minimum_required} required"
        )
    output_count = min(target, accepted_count)
    output = _sample_payload(_concatenate(payloads), output_count, rng)
    output["body_id"] = np.zeros(output_count, dtype=np.int16)
    ground_part_id = str(descriptor["static_id"])
    output["scene_part_id"] = np.full(
        output_count, ground_part_id, dtype=f"<U{len(ground_part_id)}"
    )
    report = {
        "method": "complete_single_semantic_ground_top_plane_inside_initial_frame",
        "ground_source": ground_source,
        "ground_mesh": ground_mesh,
        "candidate_point_count": int(total_candidate_count),
        "requested_point_count": int(target),
        "minimum_required_point_count": int(minimum_required),
        "output_point_count": int(output_count),
        "minimum_ground_normal_alignment": 0.8,
    }
    return output, report


def _ground_contract(
    metadata: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    collider_id = ground_collider_id(metadata)
    colliders = {
        str(record["id"]): record
        for record in metadata["simulation"]["support"]["colliders"]
    }
    collider = colliders[collider_id]
    if str(collider.get("primitive")) != "box":
        raise ValueError(f"ground {collider_id} must use a box collider")
    if collider.get("render_replaced_by_solid_wedge") or collider.get(
        "replaced_by_static_support_binding"
    ):
        raise ValueError(f"ground {collider_id} cannot use a replacement visual")
    return collider_id, collider, _analytic_contact_plane_contract(collider)


def _visibility(
    xyz_world: np.ndarray,
    xyz_camera: np.ndarray,
    camera_position: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    clip_start: float,
    depsgraph: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width, height = image_size
    safe_z = np.maximum(xyz_camera[:, 2], 1e-6)
    u = intrinsics[0, 0] * xyz_camera[:, 0] / safe_z + intrinsics[0, 2]
    v = intrinsics[1, 2] - intrinsics[1, 1] * xyz_camera[:, 1] / safe_z
    in_frame = (
        (xyz_camera[:, 2] > max(0.03, clip_start))
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )
    visible = np.zeros(len(xyz_world), dtype=np.uint8)
    scene = bpy.context.scene
    for index in np.flatnonzero(in_frame):
        delta = xyz_world[index].astype(np.float64) - camera_position
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-9:
            continue
        direction_array = delta / distance
        ray_start = max(0.03, clip_start) + 1e-4
        if distance <= ray_start:
            continue
        ray_origin = camera_position + direction_array * ray_start
        direction = mathutils.Vector(
            tuple(float(value) for value in direction_array)
        )
        tolerance = max(0.0005, distance * 0.0001)
        hit, location, *_ = scene.ray_cast(
            depsgraph,
            mathutils.Vector(tuple(float(value) for value in ray_origin)),
            direction,
            distance=distance - ray_start + tolerance,
        )
        if (
            hit
            and np.linalg.norm(
                np.asarray(location, dtype=np.float64) - xyz_world[index]
            )
            <= tolerance
        ):
            visible[index] = 1
    return visible, u, v


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--scene-id")
    parser.add_argument("--first-frame", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--scene-glb-output", type=Path)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--source-object-points", type=int, default=8192)
    parser.add_argument("--source-environment-points", type=int, default=24576)
    parser.add_argument(
        "--source-ground-completion-candidates", type=int, default=24576
    )
    parser.add_argument(
        "--model-object-points", type=int, default=DEFAULT_OBJECT_POINTS
    )
    parser.add_argument(
        "--model-environment-points", type=int, default=DEFAULT_ENVIRONMENT_POINTS
    )
    return parser.parse_args(_argv())


def main() -> None:
    args = _parse_args()
    started = time.perf_counter()
    global PROJECT_ROOT
    PROJECT_ROOT = args.project_root.resolve()
    rigid_renderer.PROJECT_ROOT = PROJECT_ROOT
    metadata_path = _resolve(str(args.metadata))
    first_frame_path = _resolve(str(args.first_frame))
    source_output = _resolve(str(args.source_output))
    model_output = _resolve(str(args.model_output))
    scene_glb_output = (
        _resolve(str(args.scene_glb_output))
        if args.scene_glb_output is not None
        else None
    )
    report_output = _resolve(str(args.report_output))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "physweep_pybullet_rigid_bound_metadata_v1":
        raise ValueError("GT scene export requires bound rigid metadata")
    if not first_frame_path.is_file():
        raise FileNotFoundError(first_frame_path)
    scene_id = str(args.scene_id or metadata["scene_id"])

    visual = metadata["visualization"]
    render_config = dict(visual["render"])
    rigid_renderer.clear_scene()
    rigid_renderer.setup_scene(render_config)
    materials = rigid_renderer.build_static_scene(metadata, visual)
    dynamic_record = metadata["simulation"]["objects"][0]
    dynamic = rigid_renderer.create_dynamic_primitive(
        dynamic_record, materials["dynamic_object"]
    )
    state = dynamic_record["initial_state"]
    dynamic.location = tuple(float(value) for value in state["position_m"])
    dynamic.rotation_mode = "QUATERNION"
    dynamic.rotation_quaternion = tuple(
        float(value)
        for value in state.get("orientation_quaternion_wxyz", [1.0, 0.0, 0.0, 0.0])
    )
    dynamic["physweep_object_id"] = str(dynamic_record["id"])
    bpy.context.scene.frame_set(int(render_config["frame_start"]))
    bpy.context.view_layer.update()

    ground_id, ground_collider, ground_contract = _ground_contract(metadata)
    interaction_ids = interaction_collider_ids(metadata)
    descriptors = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        if obj == dynamic:
            category = "object"
        else:
            category = "environment"
        static_role = str(obj.get("physweep_static_role", ""))
        static_primitive = str(obj.get("physweep_static_primitive", ""))
        static_id = str(obj.get("physweep_static_id", ""))
        descriptors.append(
            {
                "object": obj,
                "category": category,
                "static_role": static_role,
                "static_primitive": static_primitive,
                "static_id": static_id,
                "ground_contract": ground_contract if static_id == ground_id else None,
                "ground_completion": category == "environment"
                and static_id == ground_id,
            }
        )
    if not any(item["category"] == "object" for item in descriptors):
        raise ValueError("rebuilt scene has no dynamic object mesh")
    if not any(item["category"] == "environment" for item in descriptors):
        raise ValueError("rebuilt scene has no environment mesh")
    image_cache = _ImageCache()
    rng = np.random.default_rng(args.seed)
    camera = visual["camera"]
    camera_from_world = _camera_from_world(camera)
    first_frame_image = bpy.data.images.load(
        str(first_frame_path), check_existing=False
    )
    width, height = [int(value) for value in first_frame_image.size]
    bpy.data.images.remove(first_frame_image)
    intrinsics = _camera_intrinsics(camera, width, height)
    def transform_to_camera(payload: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        xyz_camera = _transform_points(payload["xyz_world"], camera_from_world)
        normals_camera = _normalize(
            payload["normal_world"] @ camera_from_world[:3, :3].T
        )
        payload["xyz"] = xyz_camera.astype(np.float32)
        payload["normal"] = normals_camera.astype(np.float32)
        return payload

    camera_position = np.asarray(camera["position_m"], dtype=np.float64)
    _set_hidden(dynamic, True)
    try:
        hidden_depsgraph = bpy.context.evaluated_depsgraph_get()
        environment_rgb = _render_current_scene_rgb(image_cache, (width, height))
        visible_environment, visible_environment_report = (
            _sample_environment_first_hits(
                camera_from_world,
                intrinsics,
                (width, height),
                camera_position,
                float(camera["clip_start_m"]),
                float(camera["clip_end_m"]),
                environment_rgb,
                int(args.source_environment_points),
                rng,
                hidden_depsgraph,
            )
        )
        ground_candidates, ground_report = (
            _sample_ground_completion_candidates(
                descriptors,
                ground_id,
                ground_collider,
                ground_contract,
                camera_from_world,
                intrinsics,
                (width, height),
                float(camera["clip_start_m"]),
                float(camera["clip_end_m"]),
                int(args.source_ground_completion_candidates),
                rng,
                hidden_depsgraph,
                image_cache,
            )
        )
        ground_candidates = transform_to_camera(ground_candidates)
        ground_visible, _, _ = _visibility(
            ground_candidates["xyz_world"],
            ground_candidates["xyz"],
            camera_position,
            intrinsics,
            (width, height),
            float(camera["clip_start_m"]),
            hidden_depsgraph,
        )
        ground_completion = _filtered_payload(
            ground_candidates, ~ground_visible.astype(bool)
        )
    finally:
        _set_hidden(dynamic, False)

    depsgraph = bpy.context.evaluated_depsgraph_get()
    object_target = int(args.source_object_points)
    object_candidate_count = max(object_target * 3, object_target + 2048)
    object_descriptors = [item for item in descriptors if item["category"] == "object"]
    areas = np.asarray(
        [_mesh_area(item["object"], depsgraph) for item in object_descriptors],
        dtype=np.float64,
    )
    if (areas <= 0).all():
        raise ValueError("controlled object mesh has no surface area")
    allocation = _allocate(object_candidate_count, np.maximum(areas, 1e-9))
    object_records = []
    for descriptor, count in zip(object_descriptors, allocation):
        payload = _sample_mesh(
            descriptor["object"], int(count), rng, depsgraph, image_cache
        )
        length = len(payload["xyz_world"])
        payload.update(
            {
                "body_id": np.ones(length, dtype=np.int16),
                "scene_part_id": np.full(
                    length,
                    str(dynamic_record["id"]),
                    dtype=f"<U{len(str(dynamic_record['id']))}",
                ),
            }
        )
        object_records.append(payload)
    object_payload = transform_to_camera(_concatenate(object_records))
    object_output = _sample_payload(object_payload, object_target, rng)
    visible_environment = transform_to_camera(visible_environment)
    object_output["ground_completion_mask"] = np.zeros(object_target, dtype=np.uint8)
    visible_environment["ground_completion_mask"] = np.zeros(
        len(visible_environment["xyz_world"]), dtype=np.uint8
    )
    finalized = [object_output, visible_environment]
    if len(ground_completion["xyz_world"]):
        ground_completion["ground_completion_mask"] = np.ones(
            len(ground_completion["xyz_world"]), dtype=np.uint8
        )
        finalized.append(ground_completion)
    surface = _concatenate(finalized)
    sampling_report = {
        "object": {
            "method": "complete_mesh_area_sampling",
            "candidate_point_count": object_candidate_count,
            "output_point_count": object_target,
        },
        "environment_first_hit": visible_environment_report,
        "ground_completion": {
            **ground_report,
            "ground_collider_id": ground_id,
            "ground_visual_id": ground_id,
            "hidden_completion_point_count": len(ground_completion["xyz_world"]),
        },
    }

    visible, projected_u, projected_v = _visibility(
        surface["xyz_world"],
        surface["xyz"],
        np.asarray(camera["position_m"], dtype=np.float64),
        intrinsics,
        (width, height),
        float(camera["clip_start_m"]),
        depsgraph,
    )
    frame_rgb = _load_first_frame(first_frame_path, image_cache)
    visible_indices = np.flatnonzero(visible)
    if len(visible_indices):
        x = np.clip(
            np.rint(projected_u[visible_indices]).astype(np.int64), 0, width - 1
        )
        y_top = np.clip(
            np.rint(projected_v[visible_indices]).astype(np.int64), 0, height - 1
        )
        y_bottom = height - 1 - y_top
        surface["rgb"][visible_indices] = frame_rgb[y_bottom, x, :3]
        surface["rgb_valid"][visible_indices] = 1
    surface["visible_mask"] = visible.astype(np.uint8)

    glb_report = None
    if scene_glb_output is not None:
        image_report = _downscale_review_images()
        scene_glb_output.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.export_scene.gltf(
            filepath=str(scene_glb_output),
            export_format="GLB",
            export_cameras=False,
            export_lights=False,
            export_animations=False,
            export_apply=True,
            export_image_format="JPEG",
        )
        glb_report = {
            "path": str(scene_glb_output),
            "sha256": _sha256(scene_glb_output),
            "size_bytes": int(scene_glb_output.stat().st_size),
            **image_report,
        }

    source_metadata = {
        "schema": GT_SURFACE_SCHEMA,
        "scene_id": scene_id,
        "bound_metadata_scene_id": str(metadata["scene_id"]),
        "source": "simulation_gt_complete_object_complete_ground_plus_camera_first_hit_visible_non_ground_at_t0",
        "source_metadata": {
            "path": str(metadata_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(metadata_path),
        },
        "first_frame": {
            "path": str(first_frame_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(first_frame_path),
        },
        "coordinate_frame": "camera_right_up_forward",
        "units": "meters",
        "controlled_object_id": str(dynamic_record["id"]),
        "complete_object_surface": True,
        "camera_first_hit_visible_non_ground_surface": True,
        "ground_surface_completion": True,
        "hidden_non_ground_surface_completion": False,
        "metric_environment_density": True,
        "environment_surface_policy": ENVIRONMENT_SURFACE_POLICY,
        "controlled_object_removal": "visibility_only_hide_at_t0",
        "ground_collider_id": ground_id,
        "interaction_collider_ids": list(interaction_ids),
        "visible_environment_policy": "all_bound_scene_geometry_rendered_at_t0",
        "ground_collider_contract": ground_contract,
        "ground_completion_scope": "hidden_semantic_ground_top_plane_inside_initial_frame",
        "ground_completion_min_normal_alignment": 0.8,
        "visibility_mask_reference": "original_t0_scene",
        "visibility_is_annotation_not_filter": True,
        "future_trajectory_read": False,
        "local_context": {
            "type": ENVIRONMENT_SURFACE_POLICY,
            "image_size_px": [width, height],
            "camera_clip_start_m": float(camera["clip_start_m"]),
            "camera_clip_end_m": float(camera["clip_end_m"]),
        },
        "visibility_method": "blender_scene_first_hit_raycast_from_bound_camera",
        "rgb_policy": "rendered_first_hits_and_original_visible_object_use_image_rgb_hidden_ground_uses_bound_mesh_rgb_or_unknown_collider_rgb",
        "surface_sampling": sampling_report,
        "approximations": [],
    }
    arrays = {
        **{
            name: surface[name]
            for name in (
                "xyz",
                "xyz_world",
                "normal",
                "rgb",
                "rgb_valid",
                "body_id",
                "scene_part_id",
                "visible_mask",
                "ground_completion_mask",
            )
        },
        "camera_from_world": camera_from_world.astype(np.float32),
        "camera_intrinsics": intrinsics.astype(np.float32),
        "image_size_px": np.asarray([width, height], dtype=np.int32),
        "controlled_object_id": np.asarray(str(dynamic_record["id"])),
    }
    source_report = write_gt_surface(source_output, arrays, source_metadata)
    model_report = compile_model_scene_condition(
        source_output,
        model_output,
        seed=args.seed,
        object_points=args.model_object_points,
        environment_points=args.model_environment_points,
        bound_metadata=metadata,
    )
    report = {
        "schema": "physweep.gt_initial_surface_export.v2",
        "scene_id": scene_id,
        "source_surface": {
            "path": str(source_output),
            "sha256": _sha256(source_output),
            **source_report,
        },
        "model_scene": {
            "path": str(model_output),
            "sha256": _sha256(model_output),
            **model_report,
        },
        "scene_glb": glb_report,
        "future_trajectory_read": False,
        "wall_time_s": round(time.perf_counter() - started, 6),
    }
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
