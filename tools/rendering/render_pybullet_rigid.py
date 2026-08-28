#!/usr/bin/env python3
"""Render one bound PhysSweep PyBullet trajectory in Blender."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import bpy
import mathutils
import numpy as np

if "bool" not in np.__dict__:
    np.bool = np.bool_  # type: ignore[attr-defined]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.rendering import render_sketchfab_background_compositions as composition
from tools.rendering.appearance_adaptation import (
    apply_material_lightness_adaptation,
    choose_rendered_frame_exposure_adjustment,
)
from tools.rendering.blender_render_settings import configure_render_engine
from tools.assets.static_support_proxy import blender_import_static_support_visual
from tools.rendering.video_encoding import configure_h264_output, normalize_h264_container
from tools.dataset_contract.trajectory_contract import object_trajectory_view



def configure_project_root(root: Path) -> Path:
    global PROJECT_ROOT
    PROJECT_ROOT = root.resolve()
    return PROJECT_ROOT


def blender_argv() -> list[str]:
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def setup_scene(render: dict[str, Any]) -> None:
    scene = bpy.context.scene
    configure_render_engine(scene, render)
    scene.render.resolution_x = int(render["resolution_x"])
    scene.render.resolution_y = int(render["resolution_y"])
    scene.render.resolution_percentage = int(render["resolution_percentage"])
    scene.render.fps = int(render["fps"])
    scene.frame_start = int(render["frame_start"])
    scene.frame_end = int(render["frame_end"])
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    color = render["color_management"]
    scene.view_settings.view_transform = str(color["view_transform"])
    scene.view_settings.look = str(color["look"])
    scene.view_settings.exposure = float(color["exposure"])
    scene.view_settings.gamma = float(color["gamma"])
    for view_layer in scene.view_layers:
        if hasattr(view_layer, "use_pass_object_index"):
            view_layer.use_pass_object_index = True


def material_from_binding(
    name: str, binding: dict[str, Any], dimensions_m: list[float]
) -> Any:
    record = dict(binding["record"])
    record["path"] = str(resolve_project_path(str(record["path"])))
    if not Path(record["path"]).exists():
        raise FileNotFoundError(record["path"])
    fallback = tuple(
        float(value)
        for value in binding.get("semantic_color_srgb", [0.45, 0.42, 0.38, 1.0])
    )
    material = composition.make_strict_polyhaven_material(
        name,
        record,
        fallback,
        texture_scale=float(binding["texture_scale"]),
        semantic_color_mix=float(binding.get("semantic_color_mix", 0.0)),
    )
    del dimensions_m
    for node in material.node_tree.nodes:
        if node.bl_idname == "ShaderNodeMapping":
            node.inputs["Scale"].default_value = (
                float(binding["texture_scale"]),
                float(binding["texture_scale"]),
                float(binding["texture_scale"]),
            )
            for source in material.node_tree.nodes:
                if source.bl_idname == "ShaderNodeTexCoord":
                    for link in list(material.node_tree.links):
                        if (
                            link.to_node == node
                            and link.to_socket == node.inputs["Vector"]
                        ):
                            material.node_tree.links.remove(link)
                    material.node_tree.links.new(
                        source.outputs["UV"], node.inputs["Vector"]
                    )
                    break
    return material


def create_box(
    name: str,
    size: list[float],
    position: list[float],
    rotation: list[float],
    material: Any,
) -> Any:
    bpy.ops.mesh.primitive_cube_add(
        size=1.0, location=tuple(float(value) for value in position)
    )
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = tuple(float(value) for value in size)
    obj.rotation_euler = tuple(math.radians(float(value)) for value in rotation)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.cube_project(cube_size=1.0, correct_aspect=True)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.data.materials.append(material)
    return obj


def create_solid_wedge(
    record: dict[str, Any], surface_material: Any, structure_material: Any
) -> Any:
    if str(record.get("slope_axis")) != "y":
        raise ValueError("solid wedge currently requires slope_axis=y")
    width, length = [float(value) for value in record["size_xy_m"]]
    base_z = float(record["base_z_m"])
    high_z = float(record["high_top_z_m"])
    if min(width, length) <= 0.0 or high_z <= base_z:
        raise ValueError("solid wedge dimensions must be positive")
    half_x = width / 2.0
    half_y = length / 2.0
    vertices = [
        (-half_x, -half_y, base_z),
        (half_x, -half_y, base_z),
        (half_x, half_y, base_z),
        (-half_x, half_y, base_z),
        (-half_x, half_y, high_z),
        (half_x, half_y, high_z),
    ]
    faces = [
        (0, 1, 2, 3),
        (0, 4, 5, 1),
        (3, 2, 5, 4),
        (0, 3, 4),
        (1, 5, 2),
    ]
    mesh = bpy.data.meshes.new(f"{record['id']}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(str(record["id"]), mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(structure_material)
    obj.data.materials.append(surface_material)
    for polygon in obj.data.polygons:
        polygon.material_index = 1 if polygon.index == 1 else 0
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")
    bevel = obj.modifiers.new("edge_bevel", "BEVEL")
    bevel.width = min(width, length, high_z - base_z) * 0.012
    bevel.segments = 2
    return obj


def _create_dynamic_cuboid(record: dict[str, Any], material: Any) -> Any:
    size = [float(value) for value in record["geometry"]["size_m"]]
    obj = create_box(
        "dynamic_object_a", size, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], material
    )
    bevel = obj.modifiers.new("edge_bevel", "BEVEL")
    bevel.width = min(size) * 0.045
    bevel.segments = 3
    return obj


def _create_dynamic_sphere(record: dict[str, Any], material: Any) -> Any:
    size = [float(value) for value in record["geometry"]["size_m"]]
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=64, ring_count=32, radius=size[0] / 2.0, location=(0.0, 0.0, 0.0)
    )
    obj = bpy.context.object
    obj.name = "dynamic_object_a"
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def _create_dynamic_cylinder(record: dict[str, Any], material: Any) -> Any:
    size = [float(value) for value in record["geometry"]["size_m"]]
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=64,
        radius=max(size[0], size[1]) / 2.0,
        depth=size[2],
        location=(0.0, 0.0, 0.0),
    )
    obj = bpy.context.object
    obj.name = "dynamic_object_a"
    obj.data.materials.append(material)
    bevel = obj.modifiers.new("edge_bevel", "BEVEL")
    bevel.width = min(size) * 0.035
    bevel.segments = 3
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def _create_dynamic_mesh(record: dict[str, Any], material: Any) -> Any:
    profile = record["visual_profile"]
    path = resolve_project_path(str(profile["path"]))
    if not path.exists():
        raise FileNotFoundError(path)
    if sha256(path) != str(profile["sha256"]):
        raise ValueError(f"dynamic visual mesh hash mismatch: {path}")
    before = set(bpy.context.scene.objects)
    bpy.context.scene.frame_set(0)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported_objects = [obj for obj in bpy.context.scene.objects if obj not in before]
    imported = [obj for obj in imported_objects if obj.type == "MESH"]
    if not imported:
        raise ValueError(f"visual mesh contains no renderable mesh: {path}")
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in imported:
        world_matrix = obj.matrix_world.copy()
        evaluated = obj.evaluated_get(depsgraph)
        baked_mesh = bpy.data.meshes.new_from_object(
            evaluated,
            preserve_all_data_layers=True,
            depsgraph=depsgraph,
        )
        obj.data = baked_mesh
        obj.animation_data_clear()
        for constraint in list(obj.constraints):
            obj.constraints.remove(constraint)
        for modifier in list(obj.modifiers):
            obj.modifiers.remove(modifier)
        obj.parent = None
        obj.matrix_world = world_matrix
    for source_object in imported_objects:
        if source_object not in imported:
            bpy.data.objects.remove(source_object, do_unlink=True)
    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    for obj in imported:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = imported[0]
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.join()
    obj = bpy.context.object
    bpy.context.view_layer.update()
    obj.rotation_mode = "XYZ"
    print(f"mesh_extent_imported={list(obj.dimensions)}")
    coordinate_frame = str(profile["alignment_coordinate_frame"])
    if coordinate_frame == "raw_gltf_z_up":
        obj.rotation_euler = (math.radians(-90.0), 0.0, 0.0)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
        bpy.context.view_layer.update()
    elif coordinate_frame != "blender_imported_z_up":
        raise ValueError(
            f"unsupported mesh alignment coordinate frame: {coordinate_frame}"
        )
    print(f"mesh_extent_coordinate_restored={list(obj.dimensions)}")
    obj.rotation_euler = tuple(
        math.radians(float(value)) for value in profile["alignment_euler_degrees"]
    )
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)
    bpy.context.view_layer.update()
    print(f"mesh_extent_aligned={list(obj.dimensions)}")
    target = np.asarray(record["geometry"]["size_m"], dtype=np.float64)
    source = np.maximum(np.asarray(obj.dimensions, dtype=np.float64), 1.0e-8)
    scale_by_axis = target / source
    uniform_scale = float(np.median(scale_by_axis))
    predicted = source * uniform_scale
    relative_error = np.abs(predicted - target) / np.maximum(target, 1.0e-8)
    if float(relative_error.max()) > 0.06:
        raise ValueError(
            f"audited visual and collision extents diverge for {profile['asset_id']}: "
            f"target={target.tolist()} predicted={predicted.tolist()}"
        )
    obj.scale = tuple(float(value) * uniform_scale for value in obj.scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    local_bounds = np.asarray(
        [list(corner) for corner in obj.bound_box], dtype=np.float64
    )
    local_center = (local_bounds.min(axis=0) + local_bounds.max(axis=0)) / 2.0
    local_center_vector = tuple(float(value) for value in local_center)
    for vertex in obj.data.vertices:
        vertex.co -= mathutils.Vector(local_center_vector)
    obj.location = (0.0, 0.0, 0.0)
    obj.name = "dynamic_object_a"
    obj["physweep_visual_asset_id"] = str(profile["asset_id"])
    obj["physweep_source_animation_baked_frame"] = 0
    obj["physweep_collision_extent_m"] = target.tolist()
    obj["physweep_visual_extent_m"] = predicted.tolist()
    if not obj.data.materials:
        obj.data.materials.append(material)
    return obj


DYNAMIC_PRIMITIVE_BUILDERS = {
    "cuboid": _create_dynamic_cuboid,
    "sphere": _create_dynamic_sphere,
    "cylinder": _create_dynamic_cylinder,
}

DYNAMIC_VISUAL_BUILDERS = {
    "primitive": lambda record, material: DYNAMIC_PRIMITIVE_BUILDERS[
        str(record["visual_profile"]["primitive"])
    ](record, material),
    "mesh": _create_dynamic_mesh,
}


def create_dynamic_primitive(record: dict[str, Any], material: Any) -> Any:
    visual_type = str(record.get("visual_profile", {"type": "primitive"})["type"])
    if "visual_profile" not in record:
        record = {
            **record,
            "visual_profile": {
                "type": "primitive",
                "primitive": record["geometry"]["type"],
            },
        }
    try:
        return DYNAMIC_VISUAL_BUILDERS[visual_type](record, material)
    except KeyError as exc:
        raise ValueError(f"unsupported dynamic visual type: {visual_type}") from exc


def normalize_imported_transparency_shadows(meshes: list[Any]) -> int:
    """Keep Eevee shadow evaluation consistent with imported alpha blending."""
    shadow_mode_by_blend_mode = {
        "BLEND": "HASHED",
        "HASHED": "HASHED",
        "CLIP": "CLIP",
    }
    changed = 0
    seen = set()
    for obj in meshes:
        for material in obj.data.materials:
            if material is None or material.as_pointer() in seen:
                continue
            seen.add(material.as_pointer())
            desired = shadow_mode_by_blend_mode.get(str(material.blend_method))
            if desired is not None and str(material.shadow_method) == "OPAQUE":
                material.shadow_method = desired
                changed += 1
    return changed


def create_static_mesh(record: dict[str, Any]) -> list[Any]:
    import bmesh

    path = resolve_project_path(str(record["path"]))
    if not path.exists():
        raise FileNotFoundError(path)
    if sha256(path) != str(record["sha256"]):
        raise ValueError(f"static visual mesh hash mismatch: {path}")

    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    for obj in list(imported):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
            imported.remove(obj)
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise ValueError(f"static visual mesh contains no renderable mesh: {path}")
    for obj in meshes:
        matrix = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = matrix

    low, high = composition.bbox_for_objects(meshes)
    source_size = np.asarray(high - low, dtype=np.float64)
    expected_size = np.asarray(record["source_bbox_size"], dtype=np.float64)
    relative_error = np.abs(source_size - expected_size) / np.maximum(
        expected_size, 1.0e-8
    )
    if float(relative_error.max()) > 0.005:
        raise ValueError(
            f"static visual source bounds changed for {record['asset_id']}: "
            f"expected={expected_size.tolist()} actual={source_size.tolist()}"
        )
    excluded_names = {str(value) for value in record.get("exclude_object_names", [])}
    excluded_prefixes = tuple(
        str(value) for value in record.get("exclude_object_name_prefixes", [])
    )
    imported_names = {obj.name for obj in meshes}
    missing_names = excluded_names - imported_names
    missing_prefixes = [
        prefix
        for prefix in excluded_prefixes
        if not any(name.startswith(prefix) for name in imported_names)
    ]
    if missing_names or missing_prefixes:
        raise ValueError(
            f"static visual shell selectors changed for {record['asset_id']}: "
            f"names={sorted(missing_names)} prefixes={missing_prefixes}"
        )
    excluded_meshes = [
        obj
        for obj in meshes
        if obj.name in excluded_names
        or any(obj.name.startswith(prefix) for prefix in excluded_prefixes)
    ]
    for obj in excluded_meshes:
        bpy.data.objects.remove(obj, do_unlink=True)
    meshes = [obj for obj in meshes if obj not in excluded_meshes]
    if not meshes:
        raise ValueError(
            f"static visual shell editing removed every mesh: {record['asset_id']}"
        )
    face_exclusions = list(record.get("source_space_face_exclusions", []))
    axis_index = {"x": 0, "y": 1, "z": 2}
    normalized_face_exclusions = []
    for selector in face_exclusions:
        if not isinstance(selector, dict) or set(selector) != {
            "axis",
            "comparison",
            "value",
        }:
            raise ValueError(
                f"invalid static visual source-space face selector: {record['asset_id']}"
            )
        selector_axis = str(selector["axis"])
        comparison = str(selector["comparison"])
        if selector_axis not in axis_index or comparison not in {
            "at_or_above",
            "at_or_below",
        }:
            raise ValueError(
                f"unsupported static visual source-space face selector: {record['asset_id']}"
            )
        normalized_face_exclusions.append(
            {
                "axis_index": axis_index[selector_axis],
                "comparison": comparison,
                "value": float(selector["value"]),
            }
        )
    excluded_face_count = 0
    exclusion_match_counts = [0] * len(normalized_face_exclusions)
    if normalized_face_exclusions:
        retained_meshes = []
        for obj in meshes:
            obj.data = obj.data.copy()
            mesh = obj.data
            editable = bmesh.new()
            editable.from_mesh(mesh)
            faces = []
            for face in editable.faces:
                world_vertices = [obj.matrix_world @ vertex.co for vertex in face.verts]
                matched = False
                for selector_index, selector in enumerate(normalized_face_exclusions):
                    coordinates = [
                        float(vertex[selector["axis_index"]])
                        for vertex in world_vertices
                    ]
                    if selector["comparison"] == "at_or_above":
                        selector_matches = min(coordinates) >= selector["value"]
                    else:
                        selector_matches = max(coordinates) <= selector["value"]
                    if selector_matches:
                        exclusion_match_counts[selector_index] += 1
                        matched = True
                if matched:
                    faces.append(face)
            excluded_face_count += len(faces)
            if faces:
                bmesh.ops.delete(editable, geom=faces, context="FACES")
                editable.to_mesh(mesh)
                mesh.update()
            editable.free()
            if len(mesh.polygons):
                retained_meshes.append(obj)
            else:
                bpy.data.objects.remove(obj, do_unlink=True)
        meshes = retained_meshes
        unmatched_selectors = [
            index
            for index, match_count in enumerate(exclusion_match_counts)
            if match_count == 0
        ]
        if unmatched_selectors:
            raise ValueError(
                f"static visual shell face selectors removed nothing for "
                f"{record['asset_id']}: {unmatched_selectors}"
            )
        if not meshes:
            raise ValueError(
                f"static visual shell face threshold removed every mesh: {record['asset_id']}"
            )
    axis = {"x": 0, "y": 1, "z": 2}[str(record["normalization_axis"])]
    scale = float(record["target_extent_m"]) / max(float(source_size[axis]), 1.0e-8)
    bottom_center = mathutils.Vector(
        ((low.x + high.x) / 2.0, (low.y + high.y) / 2.0, low.z)
    )
    rotation = (
        mathutils.Euler(
            tuple(
                math.radians(float(value)) for value in record["rotation_euler_degrees"]
            ),
            "XYZ",
        )
        .to_matrix()
        .to_4x4()
    )
    transform = (
        mathutils.Matrix.Translation(
            tuple(float(value) for value in record["position_m"])
        )
        @ rotation
        @ mathutils.Matrix.Scale(scale, 4)
        @ mathutils.Matrix.Translation(-bottom_center)
    )
    for obj in meshes:
        obj.matrix_world = transform @ obj.matrix_world
        obj.name = f"{record['id']}_{obj.name}"
        obj["physweep_scene_visual_asset_id"] = str(record["asset_id"])
        obj["physweep_collision_enabled"] = False
        obj["physweep_excluded_shell_mesh_count"] = len(excluded_meshes)
        obj["physweep_excluded_shell_face_count"] = excluded_face_count
    normalized_transparency_shadow_count = normalize_imported_transparency_shadows(
        meshes
    )
    for obj in meshes:
        obj["physweep_normalized_transparency_shadow_count"] = (
            normalized_transparency_shadow_count
        )
    if bool(
        record.get("requires_image_texture", False)
    ) and not composition.object_has_image_texture(meshes):
        raise ValueError(
            f"static visual mesh has no image texture: {record['asset_id']}"
        )
    return meshes


def create_exact_support_visual(
    record: dict[str, Any], material_bindings: dict[str, Any]
) -> list[Any]:
    binding = record["binding"]
    meshes, _ = blender_import_static_support_visual(PROJECT_ROOT, binding)
    target = binding["target_support_frame"]
    target_xy = [float(value) for value in target["size_xy_m"]]
    target_support_plane = float(target["plane_z_m"])
    for obj in meshes:
        obj.name = f"{record['id']}_{obj.name}"
        obj["physweep_support_visual_asset_id"] = str(binding["asset_id"])
        obj["physweep_collision_enabled"] = False

    if str(record["material_policy"]) == "support_surface_pbr_override":
        role = str(record["material_role"])
        material = material_from_binding(
            f"physweep_{role}_{record['id']}",
            material_bindings[role],
            [target_xy[0], target_xy[1], target_support_plane],
        )
        for obj in meshes:
            obj.data.materials.clear()
            obj.data.materials.append(material)
    elif bool(
        record.get("requires_image_texture", False)
    ) and not composition.object_has_image_texture(meshes):
        raise ValueError(
            f"support visual mesh has no image texture: {binding['asset_id']}"
        )
    return meshes


def tag_static_meshes(meshes: list[Any], record: dict[str, Any]) -> None:
    material_role = str(record.get("material_role", "embedded_asset"))
    for obj in meshes:
        obj["physweep_static_id"] = str(record["id"])
        obj["physweep_static_role"] = str(record["role"])
        obj["physweep_material_role"] = material_role
        obj["physweep_static_primitive"] = str(record["primitive"])


def look_at(obj: Any, target: list[float]) -> None:
    import mathutils

    obj.rotation_euler = (
        (mathutils.Vector(tuple(float(value) for value in target)) - obj.location)
        .to_track_quat("-Z", "Y")
        .to_euler()
    )


def add_camera(binding: dict[str, Any]) -> Any:
    bpy.ops.object.camera_add(
        location=tuple(float(value) for value in binding["position_m"])
    )
    camera = bpy.context.object
    camera.name = "physweep_camera"
    camera.data.lens = float(binding["focal_length_mm"])
    camera.data.sensor_width = float(binding["sensor_width_mm"])
    camera.data.clip_start = float(binding["clip_start_m"])
    camera.data.clip_end = float(binding["clip_end_m"])
    camera.data.dof.use_dof = False
    look_at(camera, binding["target_m"])
    bpy.context.scene.camera = camera
    return camera


def add_area_light(name: str, binding: dict[str, Any]) -> Any:
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = float(binding["energy_w"])
    light_data.size = float(binding["size_m"])
    light_data.use_shadow = bool(binding.get("cast_shadow", True))
    if hasattr(light_data, "use_contact_shadow"):
        light_data.use_contact_shadow = bool(binding.get("contact_shadow", False))
    if "contact_shadow_bias_m" in binding:
        light_data.contact_shadow_bias = float(binding["contact_shadow_bias_m"])
    if "contact_shadow_distance_m" in binding:
        light_data.contact_shadow_distance = float(binding["contact_shadow_distance_m"])
    if "contact_shadow_thickness" in binding:
        light_data.contact_shadow_thickness = float(binding["contact_shadow_thickness"])
    light = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light)
    light.location = tuple(float(value) for value in binding["position_m"])
    look_at(light, binding["target_m"])
    return light


def apply_hdri(binding: dict[str, Any]) -> None:
    path = resolve_project_path(str(binding["path"]))
    if not path.exists():
        raise FileNotFoundError(path)
    if sha256(path) != str(binding["sha256"]):
        raise ValueError(f"HDRI hash mismatch: {path}")
    world = bpy.context.scene.world or bpy.data.worlds.new("PhysSweepWorld")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = float(binding["strength"])
    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(str(path), check_existing=True)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value[2] = math.radians(
        float(binding["rotation_degrees"])
    )
    texcoord = nodes.new("ShaderNodeTexCoord")
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    links.new(environment.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])


def build_static_scene(
    metadata: dict[str, Any], visual: dict[str, Any]
) -> dict[str, Any]:
    material_bindings = visual["materials"]
    static_objects = [
        *visual["static_objects"],
        *visual["environment"]["static_background_objects"],
    ]
    ids = [str(record["id"]) for record in static_objects]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate static object id in bound metadata")
    support_objects = []
    for record in static_objects:
        if not record.get("visible", True):
            continue
        if record["primitive"] == "mesh":
            created = create_static_mesh(record)
            tag_static_meshes(created, record)
            if record["role"] not in {"environment_floor", "room_wall"}:
                support_objects.extend(created)
            continue
        if record["primitive"] == "exact_support_visual":
            created = create_exact_support_visual(record, material_bindings)
            tag_static_meshes(created, record)
            support_objects.extend(created)
            continue
        if record["primitive"] == "solid_wedge":
            surface_role = str(record["material_role"])
            structure_role = str(record["structure_material_role"])
            for role in (surface_role, structure_role):
                if role not in material_bindings:
                    raise ValueError(f"missing material role: {role}")
            dimensions = [
                float(record["size_xy_m"][0]),
                float(record["size_xy_m"][1]),
                float(record["high_top_z_m"]) - float(record["base_z_m"]),
            ]
            surface_material = material_from_binding(
                f"physweep_{surface_role}_{record['id']}",
                material_bindings[surface_role],
                dimensions,
            )
            structure_material = material_from_binding(
                f"physweep_{structure_role}_{record['id']}",
                material_bindings[structure_role],
                dimensions,
            )
            created = create_solid_wedge(
                record, surface_material, structure_material
            )
            tag_static_meshes([created], record)
            support_objects.append(created)
            continue
        if record["primitive"] != "box":
            raise ValueError(f"unsupported static primitive: {record['primitive']}")
        role = str(record["material_role"])
        if role not in material_bindings:
            raise ValueError(f"missing material role: {role}")
        material = material_from_binding(
            f"physweep_{role}_{record['id']}",
            material_bindings[role],
            [float(value) for value in record["size_m"]],
        )
        obj = create_box(
            str(record["id"]),
            record["size_m"],
            record["position_m"],
            record["rotation_euler_degrees"],
            material,
        )
        tag_static_meshes([obj], record)
        if record["role"] not in {"environment_floor", "room_wall"}:
            support_objects.append(obj)
            bevel = obj.modifiers.new("edge_bevel", "BEVEL")
            bevel.width = min(float(value) for value in record["size_m"]) * 0.025
            bevel.segments = 2
    add_camera(visual["camera"])
    add_area_light("key_area", visual["environment"]["key_light"])
    add_area_light("fill_area", visual["environment"]["fill_light"])
    apply_hdri(visual["hdri"])
    dynamic_size = metadata["simulation"]["objects"][0]["geometry"]["size_m"]
    return {
        "dynamic_object": material_from_binding(
            "physweep_dynamic_object",
            material_bindings["dynamic_object"],
            [float(value) for value in dynamic_size],
        ),
        "support_objects": support_objects,
    }


def add_dynamic_animation(
    metadata: dict[str, Any], trajectory: dict[str, np.ndarray], material: Any
) -> Any:
    record = metadata["simulation"]["objects"][0]
    object_id = str(record["object_id"])
    positions = np.asarray(trajectory[f"{object_id}__position_m"], dtype=np.float64)
    quaternions = np.asarray(
        trajectory[f"{object_id}__quaternion_wxyz"], dtype=np.float64
    )
    obj = create_dynamic_primitive(record, material)
    object_identity = metadata.get("object_identity", {})
    mask_records = object_identity.get("instance_masks", {}).get("objects", {})
    mask_record = mask_records.get(object_id, {})
    obj.pass_index = int(mask_record.get("instance_id", 1))
    obj["physweep_object_id"] = object_id
    obj["physweep_mask_instance_id"] = obj.pass_index
    obj.rotation_mode = "QUATERNION"
    frame_start = int(metadata["visualization"]["render"]["frame_start"])
    for index, (position, quaternion) in enumerate(zip(positions, quaternions)):
        frame = frame_start + index
        obj.location = tuple(float(value) for value in position)
        obj.rotation_quaternion = tuple(float(value) for value in quaternion)
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    if obj.animation_data and obj.animation_data.action:
        for curve in obj.animation_data.action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"
    return obj


def validate_instance_mask_output(
    output: dict[str, Any] | None,
    frames: list[int],
) -> dict[str, Any] | None:
    """Reject malformed compositor output before it enters a dataset."""
    if output is None:
        return None
    object_reports = {}
    for object_id, record in output["objects"].items():
        paths = [
            Path(record["directory"]) / f"frame_{frame:04d}.png"
            for frame in frames
        ]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing instance masks: {missing[:3]}")
        probe_indices = sorted(
            {0, len(frames) // 4, len(frames) // 2, 3 * len(frames) // 4, len(frames) - 1}
        )
        occupancies = []
        soft_edge_fractions = []
        for index in probe_indices:
            path = paths[index]
            image = bpy.data.images.load(str(path), check_existing=False)
            try:
                width, height = [int(value) for value in image.size]
                rgba = np.asarray(image.pixels[:], dtype=np.float32).reshape(
                    height, width, 4
                )
                alpha = rgba[:, :, 3]
                if (
                    not np.isfinite(alpha).all()
                    or float(alpha.min()) < 0.0
                    or float(alpha.max()) > 1.0
                ):
                    raise ValueError(f"instance mask alpha is invalid: {path}")
                occupancies.append(float(np.mean(alpha > 1.0e-6)))
                soft_edge_fractions.append(
                    float(np.mean((alpha > 1.0e-6) & (alpha < 1.0 - 1.0e-6)))
                )
            finally:
                bpy.data.images.remove(image)
        if not 0.0 < occupancies[0] < 1.0:
            raise ValueError(
                f"initial instance mask must be nonempty and non-full: {object_id}"
            )
        object_reports[str(object_id)] = {
            "frame_count": len(frames),
            "pixel_probe_frames": [frames[index] for index in probe_indices],
            "nonempty_probe_count": sum(value > 0.0 for value in occupancies),
            "minimum_occupancy_fraction": round(min(occupancies), 9),
            "maximum_occupancy_fraction": round(max(occupancies), 9),
            "maximum_soft_edge_fraction": round(max(soft_edge_fractions), 9),
        }
    return {
        "policy_version": "physweep_antialiased_silhouette_validation_v1",
        "objects": object_reports,
    }


def render_unoccluded_instance_masks(
    render: dict[str, Any], metadata: dict[str, Any], dynamic: Any
) -> dict[str, Any]:
    """Render a conservative silhouette tube using the bound camera and trajectory."""
    mask_dir = resolve_project_path(str(render["instance_mask_dir"]))
    object_id = str(metadata["simulation"]["objects"][0]["object_id"])
    object_dir = mask_dir / object_id
    object_dir.mkdir(parents=True, exist_ok=True)
    for stale_mask in object_dir.glob("frame_*.png"):
        stale_mask.unlink()
    scene = bpy.context.scene
    scene.render.use_compositing = False
    scene.use_nodes = False
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    if scene.render.engine == "BLENDER_EEVEE":
        scene.eevee.taa_render_samples = 1
    for obj in scene.objects:
        if obj.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"}:
            obj.hide_render = obj != dynamic
    material = bpy.data.materials.new("physweep_motion_mask")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    dynamic.data.materials.clear()
    dynamic.data.materials.append(material)
    scene.render.filepath = str(object_dir / "frame_")
    scene.frame_set(int(render["frame_start"]))
    bpy.ops.render.render(animation=True)
    return {
        "encoding": "rgba_alpha_antialiased_silhouette_mask",
        "occlusion_policy": "unoccluded_dynamic_silhouette",
        "path_layout": "object_id_subdirectories",
        "directory": str(mask_dir),
        "filename_pattern": "frame_{frame:04d}.png",
        "objects": {
            object_id: {
                "instance_id": int(dynamic.pass_index),
                "directory": str(object_dir),
            }
        },
    }


def configure_video_output(
    render: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    scene = bpy.context.scene
    path = resolve_project_path(str(render["video_path"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(path)
    frame_count = int(render["frame_end"]) - int(render["frame_start"]) + 1
    encoding = configure_h264_output(
        scene,
        fps=int(render["fps"]),
        frame_count=frame_count,
    )
    return path, encoding


def render_inspection_frames(render: dict[str, Any]) -> list[Path]:
    scene = bpy.context.scene
    frame_dir = resolve_project_path(str(render["inspection_frame_dir"]))
    frame_dir.mkdir(parents=True, exist_ok=True)
    original_format = scene.render.image_settings.file_format
    original_path = scene.render.filepath
    paths = []
    scene.render.image_settings.file_format = "PNG"
    for frame in render["inspection_frames"]:
        scene.frame_set(int(frame))
        path = frame_dir / f"frame_{int(frame):04d}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        paths.append(path)
    scene.render.image_settings.file_format = original_format
    scene.render.filepath = original_path
    return paths


def rendered_png_statistics(path: Path) -> dict[str, float]:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = [int(value) for value in image.size]
        rgba = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
        rgb = np.clip(rgba[:, :, :3], 0.0, 1.0)
        luma = 255.0 * (
            0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        )
        return {
            "mean_luma": round(float(luma.mean()), 4),
            "luma_std": round(float(luma.std()), 4),
            "highlight_fraction_above_0_90": round(float(np.mean(luma > 229.5)), 7),
            "clipped_dark_fraction": round(float(np.mean(luma <= 2.0)), 7),
            "clipped_light_fraction": round(float(np.mean(luma >= 253.0)), 7),
        }
    finally:
        bpy.data.images.remove(image)


def adapt_rendered_frame_exposure(
    render: dict[str, Any],
    maximum_adjustments: int = 3,
) -> tuple[list[Path], dict[str, Any]]:
    scene = bpy.context.scene
    initial_exposure = float(scene.view_settings.exposure)
    cumulative_correction = 0.0
    attempts = []
    inspection_paths: list[Path] = []
    for attempt_index in range(maximum_adjustments + 1):
        inspection_paths = render_inspection_frames(render)
        statistics = [
            {
                "frame": int(frame),
                **rendered_png_statistics(path),
            }
            for frame, path in zip(render["inspection_frames"], inspection_paths)
        ]
        adjustment = choose_rendered_frame_exposure_adjustment(
            statistics,
            cumulative_correction,
        )
        delta = float(adjustment["applied_delta_ev"])
        correction_applied = delta != 0.0 and attempt_index < maximum_adjustments
        attempts.append(
            {
                "attempt": attempt_index,
                "exposure_ev": round(float(scene.view_settings.exposure), 6),
                "frames": statistics,
                "decision": adjustment,
                "correction_applied": correction_applied,
            }
        )
        if not correction_applied:
            break
        scene.view_settings.exposure += delta
        cumulative_correction += delta
    return inspection_paths, {
        "policy_version": attempts[-1]["decision"]["policy_version"],
        "inspection_frames": [int(value) for value in render["inspection_frames"]],
        "initial_exposure_ev": round(initial_exposure, 6),
        "final_exposure_ev": round(float(scene.view_settings.exposure), 6),
        "cumulative_correction_ev": round(cumulative_correction, 6),
        "adjustment_count": max(0, len(attempts) - 1),
        "final_probe_within_targets": not attempts[-1]["decision"]["reasons"],
        "attempts": attempts,
    }


def render(
    metadata_path: Path,
    first_frame_only: bool = False,
    mask_only: bool = False,
    instance_mask_dir: str | None = None,
    mask_resolution: tuple[int, int] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    metadata = load_json(metadata_path)
    if metadata["schema_version"] != "physweep_pybullet_rigid_bound_metadata_v1":
        raise ValueError("renderer requires bound PyBullet rigid metadata")
    source_metadata_path = resolve_project_path(
        str(metadata["source_metadata"]["path"])
    )
    if sha256(source_metadata_path) != str(metadata["source_metadata"]["sha256"]):
        raise ValueError("source metadata changed after visual binding")
    simulation_record_path = resolve_project_path(
        str(metadata["simulation_record"]["path"])
    )
    if sha256(simulation_record_path) != str(metadata["simulation_record"]["sha256"]):
        raise ValueError("simulation record changed after visual binding")
    trajectory_path = resolve_project_path(str(metadata["trajectory"]["path"]))
    if sha256(trajectory_path) != str(metadata["trajectory"]["sha256"]):
        raise ValueError("trajectory hash mismatch")
    with np.load(trajectory_path) as source:
        trajectory = {key: source[key] for key in source.files}
    trajectory = object_trajectory_view(metadata, trajectory)
    visual = metadata["visualization"]
    render_config = dict(visual["render"])
    if instance_mask_dir is not None:
        render_config["instance_mask_dir"] = instance_mask_dir
    if mask_only:
        if not render_config.get("instance_mask_dir"):
            raise ValueError("mask-only rendering requires an instance mask directory")
        if mask_resolution is not None:
            render_config["resolution_x"] = int(mask_resolution[0])
            render_config["resolution_y"] = int(mask_resolution[1])
        render_config["resolution_percentage"] = 100
        render_config["samples"] = 1
    expected_frames = (
        int(render_config["frame_end"]) - int(render_config["frame_start"]) + 1
    )
    if int(trajectory["time_s"].shape[0]) != expected_frames:
        raise ValueError("render and trajectory frame counts differ")
    clear_scene()
    setup_scene(render_config)
    materials = build_static_scene(metadata, visual)
    dynamic = add_dynamic_animation(metadata, trajectory, materials["dynamic_object"])
    if dynamic is None:
        raise RuntimeError("dynamic object was not created")
    if mask_only:
        scene = bpy.context.scene
        instance_mask_output = render_unoccluded_instance_masks(
            render_config, metadata, dynamic
        )
        instance_mask_output["validation"] = validate_instance_mask_output(
            instance_mask_output,
            list(
                range(
                    int(render_config["frame_start"]),
                    int(render_config["frame_end"]) + 1,
                )
            ),
        )
        record = {
            "schema_version": "physweep_pybullet_render_record_v1",
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
            "scene_id": metadata["scene_id"],
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256(metadata_path),
            "trajectory_path": str(trajectory_path),
            "trajectory_sha256": sha256(trajectory_path),
            "video_path": None,
            "video_sha256": None,
            "inspection_frames": [],
            "blender_version": bpy.app.version_string,
            "render_engine": scene.render.engine,
            "video_encoding": None,
            "instance_mask_output": instance_mask_output,
            "lighting_adaptation": None,
            "render_scope": "instance_masks_only",
            "mask_resolution": [
                int(render_config["resolution_x"]),
                int(render_config["resolution_y"]),
            ],
            "wall_time_s": round(time.perf_counter() - started, 6),
        }
        record_path = (
            resolve_project_path(str(render_config["instance_mask_dir"]))
            / "render_record.json"
        )
        write_json(record_path, record)
        return record
    lighting_adaptation = apply_material_lightness_adaptation(
        bpy.context.scene,
        [dynamic],
        materials["support_objects"],
    )
    if first_frame_only:
        render_config["inspection_frames"] = [int(render_config["frame_start"])]
    inspection_paths, rendered_frame_adaptation = adapt_rendered_frame_exposure(
        render_config
    )
    lighting_adaptation["rendered_frame_exposure"] = rendered_frame_adaptation
    video_path, video_encoding = configure_video_output(render_config)
    if first_frame_only:
        video_sha = None
        instance_mask_output = None
        mask_validation = None
    else:
        bpy.context.scene.frame_set(int(render_config["frame_start"]))
        bpy.ops.render.render(animation=True)
        normalize_h264_container(video_path)
        video_sha = sha256(video_path)
        instance_mask_output = render_unoccluded_instance_masks(
            render_config, metadata, dynamic
        )
        mask_validation = validate_instance_mask_output(
            instance_mask_output,
            list(
                range(
                    int(render_config["frame_start"]),
                    int(render_config["frame_end"]) + 1,
                )
            ),
        )
    if instance_mask_output is not None:
        instance_mask_output["validation"] = mask_validation
    record = {
        "schema_version": "physweep_pybullet_render_record_v1",
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "scene_id": metadata["scene_id"],
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": sha256(trajectory_path),
        "video_path": str(video_path) if not first_frame_only else None,
        "video_sha256": video_sha,
        "inspection_frames": [str(path) for path in inspection_paths],
        "blender_version": bpy.app.version_string,
        "render_engine": bpy.context.scene.render.engine,
        "video_encoding": video_encoding,
        "instance_mask_output": instance_mask_output,
        "lighting_adaptation": lighting_adaptation,
        "render_scope": "first_frame_only" if first_frame_only else "full_animation",
        "wall_time_s": round(time.perf_counter() - started, 6),
    }
    record_path = (
        resolve_project_path(str(render_config["inspection_frame_dir"]))
        / "render_record.json"
    )
    write_json(record_path, record)
    return record


def parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must look like 320x180")
    width, height = (int(part) for part in parts)
    if min(width, height) <= 0:
        raise argparse.ArgumentTypeError("resolution must be positive")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--first-frame-only", action="store_true")
    parser.add_argument("--mask-only", action="store_true")
    parser.add_argument("--instance-mask-dir")
    parser.add_argument("--mask-resolution", type=parse_resolution)
    return parser.parse_args(blender_argv())


def main() -> None:
    args = parse_args()
    configure_project_root(args.root)
    if args.first_frame_only and args.mask_only:
        raise ValueError("first-frame-only and mask-only are mutually exclusive")
    record = render(
        args.metadata.resolve(),
        first_frame_only=args.first_frame_only,
        mask_only=args.mask_only,
        instance_mask_dir=args.instance_mask_dir,
        mask_resolution=args.mask_resolution,
    )
    print(json.dumps(record, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
