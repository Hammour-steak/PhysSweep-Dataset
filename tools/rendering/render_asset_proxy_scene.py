#!/usr/bin/env python3
"""Render one asset-only proxy scene from immutable metadata and trajectory."""

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
from bpy_extras.object_utils import world_to_camera_view

if "bool" not in np.__dict__:
    np.bool = np.bool_  # type: ignore[attr-defined]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json
from tools.assets.blender_asset_import import (  # pylint: disable=wrong-import-position
    clear_scene,
    import_meshes,
    normalized_transform,
    selected_visual_meshes,
)
from tools.rendering.blender_scene import look_at
from tools.rendering.blender_render_settings import configure_render_engine
from tools.dataset_contract.immutable_scene_contract import validate_simulation_record
from tools.assets.static_support_proxy import blender_import_static_support_visual
from tools.rendering.video_encoding import configure_h264_output, normalize_h264_container
from tools.dataset_contract.trajectory_contract import adapter_trajectory_view
from tools.rendering.appearance_adaptation import apply_material_lightness_adaptation
from tools.core.camera_geometry import blocker_safe_seeded_view_order, seeded_view_order
from tools.rendering.specialized_render_evidence import (
    render_instance_mask_record,
    render_instance_masks,
    render_implementation,
)


def configure_project_root(root: Path) -> Path:
    global PROJECT_ROOT
    PROJECT_ROOT = root.resolve()
    return PROJECT_ROOT


def blender_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--inspection-frame-dir", type=Path)
    parser.add_argument("--mask-only", action="store_true")
    parser.add_argument("--instance-mask-dir", type=Path)
    return parser.parse_args(values)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def setup_scene(metadata: dict[str, Any]) -> None:
    scene = bpy.context.scene
    lighting = metadata["render"]["environment"]["lighting"]
    color = lighting["color_management"]
    width, height = [int(value) for value in metadata["render"]["resolution"]]
    configure_render_engine(scene, metadata["render"])
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.fps = int(metadata["physics"]["output_fps"])
    scene.frame_start = 1
    scene.frame_end = int(metadata["physics"]["frame_count"])
    scene.view_settings.view_transform = str(color["view_transform"])
    scene.view_settings.look = str(color["look"])
    scene.view_settings.exposure = float(color["exposure"])
    scene.view_settings.gamma = float(color["gamma"])
    world = scene.world or bpy.data.worlds.new("AssetOnlyWorld")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.075, 0.085, 0.10, 1.0)
    background.inputs["Strength"].default_value = 0.45


def apply_hdri(binding: dict[str, Any]) -> None:
    path = resolve(str(binding["path"]))
    if not path.exists():
        raise FileNotFoundError(path)
    if sha256(path) != str(binding["sha256"]):
        raise ValueError(f"HDRI hash mismatch: {path}")
    world = bpy.context.scene.world or bpy.data.worlds.new("AssetOnlyWorld")
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
    mapping.inputs["Rotation"].default_value[2] = math.radians(float(binding["rotation_degrees"]))
    texcoord = nodes.new("ShaderNodeTexCoord")
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    links.new(environment.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])


def imported_at_proxy_frame(
    record: dict[str, Any], visual_object_names: list[str] | None = None
) -> list[Any]:
    meshes = import_meshes(PROJECT_ROOT, record)
    meshes = selected_visual_meshes(meshes, record, visual_object_names)
    transform = normalized_transform(meshes, record)
    contact_offset = record["visual"].get(
        "contact_alignment_offset_m", [0.0, 0.0, 0.0]
    )
    transform = (
        mathutils.Matrix.Translation(
            tuple(float(value) for value in contact_offset)
        )
        @ transform
    )
    for obj in meshes:
        obj.matrix_world = transform @ obj.matrix_world
    return meshes


def add_support(
    binding: dict[str, Any], *, include_all_source_meshes: bool = False
) -> list[Any]:
    meshes, root_object = blender_import_static_support_visual(
        PROJECT_ROOT,
        binding,
        include_all_source_meshes=include_all_source_meshes,
    )
    root_object["physweep_asset_role"] = "support"
    root_object["physweep_asset_id"] = binding["asset_id"]
    root_object["physweep_binding_sha256"] = binding["binding_sha256"]
    for obj in meshes:
        obj["physweep_asset_role"] = "support"
        obj["physweep_asset_id"] = binding["asset_id"]
        obj["physweep_binding_sha256"] = binding["binding_sha256"]
    return meshes


def add_static_prop(record: dict[str, Any], binding: dict[str, Any]) -> list[Any]:
    meshes = imported_at_proxy_frame(record, binding.get("visual_object_names"))
    transform = (
        mathutils.Matrix.Translation(tuple(float(value) for value in binding["position_m"]))
        @ mathutils.Matrix.Rotation(math.radians(float(binding["yaw_degrees"])), 4, "Z")
    )
    for obj in meshes:
        obj.matrix_world = transform @ obj.matrix_world
        obj["physweep_asset_role"] = "static_prop"
        obj["physweep_asset_id"] = record["asset_id"]
    return meshes


def add_dynamic(record: dict[str, Any], trajectory: dict[str, np.ndarray]) -> list[Any]:
    meshes = imported_at_proxy_frame(record)
    bpy.ops.object.empty_add(type="PLAIN_AXES", location=(0.0, 0.0, 0.0))
    root = bpy.context.object
    root.name = "dynamic_asset_root"
    for obj in meshes:
        obj.parent = root
        obj["physweep_asset_role"] = "dynamic"
        obj["physweep_asset_id"] = record["asset_id"]
    root.rotation_mode = "QUATERNION"
    for index, (position, quaternion) in enumerate(
        zip(trajectory["position_m"], trajectory["quaternion_xyzw"]), start=1
    ):
        root.location = tuple(float(value) for value in position)
        x, y, z, w = [float(value) for value in quaternion]
        root.rotation_quaternion = (w, x, y, z)
        root.keyframe_insert(data_path="location", frame=index)
        root.keyframe_insert(data_path="rotation_quaternion", frame=index)
    if root.animation_data and root.animation_data.action:
        for curve in root.animation_data.action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"
    return meshes


def pbr_material(name: str, binding: dict[str, Any], repeat: float) -> Any:
    channels = binding["channels"]
    for channel in ("base_color", "roughness", "normal"):
        path = resolve(str(channels[channel]["path"]))
        if not path.exists() or sha256(path) != str(channels[channel]["sha256"]):
            raise ValueError(f"invalid PBR {channel} binding: {path}")
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    nodes = result.node_tree.nodes
    links = result.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (repeat, repeat, repeat)
    links.new(texcoord.outputs["UV"], mapping.inputs["Vector"])
    base_color = nodes.new("ShaderNodeTexImage")
    base_color.image = bpy.data.images.load(str(resolve(channels["base_color"]["path"])), check_existing=True)
    roughness = nodes.new("ShaderNodeTexImage")
    roughness.image = bpy.data.images.load(str(resolve(channels["roughness"]["path"])), check_existing=True)
    roughness.image.colorspace_settings.name = "Non-Color"
    normal_texture = nodes.new("ShaderNodeTexImage")
    normal_texture.image = bpy.data.images.load(str(resolve(channels["normal"]["path"])), check_existing=True)
    normal_texture.image.colorspace_settings.name = "Non-Color"
    normal = nodes.new("ShaderNodeNormalMap")
    normal.inputs["Strength"].default_value = 0.55
    for texture in (base_color, roughness, normal_texture):
        links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
    links.new(base_color.outputs["Color"], shader.inputs["Base Color"])
    links.new(roughness.outputs["Color"], shader.inputs["Roughness"])
    links.new(normal_texture.outputs["Color"], normal.inputs["Color"])
    links.new(normal.outputs["Normal"], shader.inputs["Normal"])
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])
    return result


def add_environment(binding: dict[str, Any], camera: dict[str, Any]) -> list[Any]:
    room = binding["room"]
    half_extent = float(room["half_extent_m"])
    height = float(room["height_m"])
    repeat_per_m = float(room["texture_repeat_per_m"])
    floor_material = pbr_material(
        "room_floor_pbr", room["floor_material"], 2.0 * half_extent * repeat_per_m
    )
    wall_material = pbr_material(
        "room_wall_pbr", room["wall_material"], 2.0 * half_extent * repeat_per_m
    )
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, -0.002))
    floor = bpy.context.object
    floor.name = "room_floor"
    floor.scale = (2.0 * half_extent, 2.0 * half_extent, 1.0)
    floor.data.materials.append(floor_material)

    camera_side = mathutils.Vector((camera["position_m"][0], camera["position_m"][1]))
    camera_side.normalize()
    wall_specs = (
        ((0.0, half_extent, 0.5 * height), 0.0, (0.0, 1.0)),
        ((0.0, -half_extent, 0.5 * height), 180.0, (0.0, -1.0)),
        ((half_extent, 0.0, 0.5 * height), 90.0, (1.0, 0.0)),
        ((-half_extent, 0.0, 0.5 * height), -90.0, (-1.0, 0.0)),
    )
    result = [floor]
    for index, (location, yaw, side) in enumerate(wall_specs):
        if camera_side.dot(mathutils.Vector(side)) > 0.1:
            continue
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=location)
        wall = bpy.context.object
        wall.name = f"room_wall_{index:02d}"
        wall.rotation_euler = (math.radians(90.0), 0.0, math.radians(yaw))
        wall.scale = (2.0 * half_extent, height, 1.0)
        wall.data.materials.append(wall_material)
        result.append(wall)
    return result


def add_lighting(
    target: mathutils.Vector,
    span: float,
    binding: dict[str, Any],
) -> None:
    scale_rules = binding["light_scale"]
    reference_span = float(scale_rules["reference_span_m"])
    scale = (max(span, 1.0e-6) / reference_span) ** float(
        scale_rules["exponent"]
    )
    scale = max(
        float(scale_rules["minimum"]),
        min(scale, float(scale_rules["maximum"])),
    )
    for record in binding["area_lights"]:
        name = str(record["name"])
        data = bpy.data.lights.new(name, "AREA")
        data.energy = float(record["energy_w"]) * scale
        data.size = float(record["size_m"])
        data.use_shadow = True
        if hasattr(data, "use_contact_shadow"):
            data.use_contact_shadow = True
        light = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(light)
        light.location = target + mathutils.Vector(record["offset_m"])
        look_at(light, target)


def aabb_corners(low: np.ndarray, high: np.ndarray) -> list[mathutils.Vector]:
    return [
        mathutils.Vector((float(x), float(y), float(z)))
        for x in (low[0], high[0])
        for y in (low[1], high[1])
        for z in (low[2], high[2])
    ]


def add_edge_transition_camera(
    metadata: dict[str, Any],
    trajectory: dict[str, np.ndarray],
    dynamic_record: dict[str, Any],
    specialized_views: dict[str, Any],
) -> dict[str, Any]:
    """Frame the support edge, fall, landing, and final resting pose together."""

    surface = metadata["physics"]["support_surface"]
    observation = metadata["camera_request"]["observation"]
    focus_event = observation["focus_event"]
    if (
        str(focus_event["type"]) != "collider_contact"
        or str(focus_event["collider_id"]) != "environment_floor"
    ):
        raise ValueError("edge-transition camera requires environment-floor contact")

    positions = np.asarray(trajectory["position_m"], dtype=np.float64)
    contacts = np.flatnonzero(
        np.asarray(trajectory["ground_contact"], dtype=np.int64) > 0
    )
    if not contacts.size:
        raise ValueError("edge-transition camera requires a ground-contact event")
    if not bool(observation.get("include_final_settled_pose", True)):
        raise ValueError("edge-transition camera requires the final settled pose")
    contact_index = int(contacts[0])
    post_event_frames = max(2, int(focus_event.get("post_event_frames", 2)))
    primary_end = min(positions.shape[0], contact_index + post_event_frames + 1)

    # Fit real trajectory samples instead of corners of one global AABB. The
    # latter combines positions and heights that never occur at the same time.
    primary_indices = np.arange(primary_end, dtype=int)
    full_indices = np.linspace(
        0,
        positions.shape[0] - 1,
        min(25, positions.shape[0]),
        dtype=int,
    )
    framing_indices = np.unique(
        np.concatenate(
            [primary_indices, full_indices, np.asarray([positions.shape[0] - 1])]
        )
    )
    dynamic_extent = np.asarray(
        dynamic_record["visual"]["canonical_extent_m"], dtype=np.float64
    )
    framing_half_extent = 0.55 * dynamic_extent
    framing_points: list[mathutils.Vector] = []
    for center in positions[framing_indices]:
        framing_points.extend(
            aabb_corners(center - framing_half_extent, center + framing_half_extent)
        )

    exit_direction = np.asarray(
        metadata["physics"]["initial_state"]["calculation"][
            "clear_exit_direction_xy"
        ],
        dtype=np.float64,
    )
    exit_direction /= np.linalg.norm(exit_direction)
    lateral = np.asarray([-exit_direction[1], exit_direction[0]], dtype=np.float64)
    surface_center_xy = np.asarray(surface["center_xy_m"], dtype=np.float64)
    surface_half_size = 0.5 * np.asarray(surface["size_xy_m"], dtype=np.float64)
    edge_distances = []
    for axis in range(2):
        component = float(exit_direction[axis])
        if abs(component) <= 1.0e-8:
            continue
        boundary = surface_center_xy[axis] + math.copysign(
            surface_half_size[axis], component
        )
        distance = (boundary - surface_center_xy[axis]) / component
        if distance >= 0.0:
            edge_distances.append(distance)
    if not edge_distances:
        raise ValueError("edge-transition camera cannot resolve the support edge")
    edge_xy = surface_center_xy + exit_direction * min(edge_distances)
    lateral_span = min(0.30, 0.35 * float(np.max(surface_half_size)))
    inner_xy = edge_xy - exit_direction * min(0.28, 0.30 * min(surface["size_xy_m"]))
    support_z = float(surface["z_m"])
    for xy in (
        edge_xy - lateral * lateral_span,
        edge_xy + lateral * lateral_span,
        inner_xy,
    ):
        framing_points.append(mathutils.Vector((float(xy[0]), float(xy[1]), support_z)))

    framing_array = np.asarray(
        [[float(point.x), float(point.y), float(point.z)] for point in framing_points],
        dtype=np.float64,
    )
    content_low = framing_array.min(axis=0)
    content_high = framing_array.max(axis=0)
    target = 0.5 * (content_low + content_high)
    focus_span = float(np.max(content_high - content_low))

    scene_digest = hashlib.sha256(str(metadata["scene_id"]).encode("utf-8")).digest()
    exit_azimuth = math.degrees(
        math.atan2(float(exit_direction[1]), float(exit_direction[0]))
    )
    profile = str(metadata["physics"]["motion_profile"])
    view_rule = specialized_views[profile]
    relative_azimuths = seeded_view_order(
        [float(value) for value in view_rule["relative_azimuth_degrees"]],
        scene_digest,
        4,
    )
    elevation_candidates = seeded_view_order(
        [float(value) for value in view_rule["elevation_degrees"]],
        scene_digest,
        5,
    )
    azimuth_candidates = [exit_azimuth + value for value in relative_azimuths]
    lens_candidates = [
        float(value)
        for value in observation.get("focal_length_candidates_mm", [40.0, 36.0])
    ]
    framing_margin = float(observation.get("minimum_transition_margin_ndc", 0.075))
    maximum_distance = float(observation.get("maximum_camera_distance_m", 4.8))
    minimum_distance = max(
        1.65,
        min(
            3.20,
            max(
                1.12 * float(np.max(content_high[:2] - content_low[:2])) + 0.48,
                2.00 * float(content_high[2] - content_low[2]) + 0.48,
            ),
        ),
    )

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "asset_edge_transition_camera"
    camera.data.sensor_width = 36.0
    camera.data.clip_start = 0.03
    camera.data.clip_end = 100.0
    bpy.context.scene.camera = camera

    def set_pose(
        distance: float, azimuth_degrees: float, elevation_degrees: float
    ) -> mathutils.Vector:
        azimuth = math.radians(azimuth_degrees)
        elevation = math.radians(elevation_degrees)
        horizontal = distance * math.cos(elevation)
        location = mathutils.Vector(
            (
                float(target[0]) + math.cos(azimuth) * horizontal,
                float(target[1]) + math.sin(azimuth) * horizontal,
                float(target[2]) + distance * math.sin(elevation),
            )
        )
        camera.location = location
        look_at(camera, mathutils.Vector(target))
        bpy.context.view_layer.update()
        return location

    selected: dict[str, Any] | None = None
    for lens_mm in lens_candidates:
        camera.data.lens = lens_mm
        admissible: list[dict[str, Any]] = []
        for azimuth_degrees in azimuth_candidates:
            for elevation_degrees in elevation_candidates:
                distance = minimum_distance
                for iteration in range(16):
                    location = set_pose(
                        distance, azimuth_degrees, elevation_degrees
                    )
                    projected = [
                        world_to_camera_view(bpy.context.scene, camera, point)
                        for point in framing_points
                    ]
                    if all(
                        point.z > 0.0
                        and framing_margin <= point.x <= 1.0 - framing_margin
                        and framing_margin <= point.y <= 1.0 - framing_margin
                        for point in projected
                    ):
                        admissible.append(
                            {
                                "location": location.copy(),
                                "distance": distance,
                                "azimuth_degrees": azimuth_degrees,
                                "elevation_degrees": elevation_degrees,
                                "lens_mm": lens_mm,
                                "iteration": iteration,
                                "projected": projected,
                            }
                        )
                        break
                    next_distance = min(maximum_distance, distance * 1.08)
                    if next_distance <= distance + 1.0e-6:
                        break
                    distance = next_distance
        if admissible:
            selected = admissible[0]
            break
    if selected is None:
        raise ValueError("edge-transition camera cannot satisfy its framing contract")

    camera.data.lens = float(selected["lens_mm"])
    location = set_pose(
        float(selected["distance"]),
        float(selected["azimuth_degrees"]),
        float(selected["elevation_degrees"]),
    )
    projected_centers = [
        world_to_camera_view(
            bpy.context.scene,
            camera,
            mathutils.Vector((float(point[0]), float(point[1]), float(point[2]))),
        )
        for point in positions
    ]
    center_visible = [
        point.z > 0.0
        and framing_margin <= point.x <= 1.0 - framing_margin
        and framing_margin <= point.y <= 1.0 - framing_margin
        for point in projected_centers
    ]
    minimum_full_fraction = float(
        observation.get("minimum_full_trajectory_center_visible_fraction", 0.80)
    )
    full_fraction = float(np.mean(center_visible))
    if full_fraction + 1.0e-9 < minimum_full_fraction:
        raise ValueError("edge-transition camera misses too much of the trajectory")

    landing_projection = projected_centers[contact_index]
    final_projection = projected_centers[-1]
    return {
        "solver_version": "asset_edge_transition_camera_v3",
        "observation_intent": str(observation["intent"]),
        "structure_context": str(observation["structure_context"]),
        "position_m": [round(float(value), 6) for value in location],
        "target_m": [round(float(value), 6) for value in target],
        "focal_length_mm": float(selected["lens_mm"]),
        "focus_span_m": round(focus_span, 6),
        "azimuth_degrees": float(selected["azimuth_degrees"]),
        "elevation_degrees": float(selected["elevation_degrees"]),
        "prop_placed_behind_primary_target": False,
        "dynamic_extent_m": [round(float(value), 6) for value in dynamic_extent],
        "projection_fit_iterations": int(selected["iteration"]),
        "framing_fit_passed": True,
        "framing_margin_ndc": framing_margin,
        "ground_contact_frame": contact_index,
        "primary_end_frame": primary_end - 1,
        "full_trajectory_center_visible_fraction": round(full_fraction, 6),
        "landing_center_ndc": [
            round(float(landing_projection.x), 6),
            round(float(landing_projection.y), 6),
        ],
        "final_center_ndc": [
            round(float(final_projection.x), 6),
            round(float(final_projection.y), 6),
        ],
    }


def add_camera(
    metadata: dict[str, Any],
    trajectory: dict[str, np.ndarray],
    dynamic_record: dict[str, Any],
    specialized_views: dict[str, Any],
) -> dict[str, Any]:
    frozen = metadata.get("camera_binding")
    if frozen is not None:
        bpy.ops.object.camera_add()
        camera = bpy.context.object
        camera.name = "asset_only_camera"
        camera.data.lens = float(frozen["focal_length_mm"])
        camera.data.sensor_width = 36.0
        camera.data.clip_start = 0.03
        camera.data.clip_end = 100.0
        camera.location = tuple(float(value) for value in frozen["position_m"])
        look_at(camera, mathutils.Vector(frozen["target_m"]))
        bpy.context.scene.camera = camera
        return dict(frozen)
    surface = metadata["physics"]["support_surface"]
    positions = np.asarray(trajectory["position_m"], dtype=np.float64)
    observation = metadata["camera_request"]["observation"]
    if str(observation["structure_context"]) == "edge_and_landing":
        return add_edge_transition_camera(
            metadata, trajectory, dynamic_record, specialized_views
        )
    focus_event = observation["focus_event"]
    event_type = str(focus_event["type"])
    if event_type == "fraction":
        primary_end = max(
            2,
            min(
                positions.shape[0],
                int(math.ceil(positions.shape[0] * float(focus_event["fraction"]))),
            ),
        )
    elif event_type in {"primary_support_contact", "collider_contact"}:
        channel = "support_contact"
        if event_type == "collider_contact":
            if str(focus_event["collider_id"]) != "environment_floor":
                raise ValueError("asset camera supports only environment-floor collider focus")
            channel = "ground_contact"
        contacts = np.flatnonzero(np.asarray(trajectory[channel], dtype=np.int64) > 0)
        if not contacts.size:
            raise ValueError(f"asset camera observation event did not occur: {channel}")
        primary_end = min(
            positions.shape[0],
            int(contacts[0]) + max(2, int(focus_event.get("post_event_frames", 2))),
        )
    else:
        raise ValueError(f"unsupported asset camera focus event: {event_type}")
    primary = positions[:primary_end]
    dynamic_extent = np.asarray(
        dynamic_record["visual"]["canonical_extent_m"], dtype=np.float64
    )
    dynamic_half_extent = 0.5 * dynamic_extent
    content_low = primary.min(axis=0) - dynamic_half_extent
    content_high = primary.max(axis=0) + dynamic_half_extent
    trajectory_center = 0.5 * (primary.min(axis=0) + primary.max(axis=0))
    surface_center = np.asarray([*surface["center_xy_m"], surface["z_m"]], dtype=np.float64)
    target = 0.82 * trajectory_center + 0.18 * surface_center
    vertical_span = max(0.0, float(content_high[2] - content_low[2]))
    sx, sy = [float(value) for value in surface["size_xy_m"]]
    dynamic_span = content_high[:2] - content_low[:2]
    focus_x = min(sx * 0.72, max(float(dynamic_span[0]) + 0.52, 0.92))
    focus_y = min(sy * 0.92, max(float(dynamic_span[1]) + 0.38, 0.52))
    prop = metadata["physics"].get("static_prop")
    if prop:
        prop_center = np.asarray(prop["position_m"][:2], dtype=np.float64)
        prop_extent = np.asarray(prop["world_aabb_extent_m"], dtype=np.float64)
        prop_low = np.asarray(
            [
                prop_center[0] - 0.5 * prop_extent[0],
                prop_center[1] - 0.5 * prop_extent[1],
                float(prop["position_m"][2]),
            ],
            dtype=np.float64,
        )
        prop_high = np.asarray(
            [
                prop_center[0] + 0.5 * prop_extent[0],
                prop_center[1] + 0.5 * prop_extent[1],
                float(prop["position_m"][2]) + prop_extent[2],
            ],
            dtype=np.float64,
        )
        combined_low = np.minimum(
            content_low[:2],
            prop_low[:2],
        )
        combined_high = np.maximum(
            content_high[:2],
            prop_high[:2],
        )
        combined_span = combined_high - combined_low
        content_low = np.minimum(content_low, prop_low)
        content_high = np.maximum(content_high, prop_high)
        target[:2] = (
            0.68 * trajectory_center[:2]
            + 0.22 * prop_center
            + 0.10 * surface_center[:2]
        )
        vertical_span = float(content_high[2] - content_low[2])
        focus_x = min(sx * 0.92, max(float(combined_span[0]) + 0.38, focus_x))
        focus_y = min(sy * 0.96, max(float(combined_span[1]) + 0.30, focus_y))
    target[2] = float(0.5 * (content_low[2] + content_high[2]))
    focus_span = max(focus_x, focus_y * 1.45)
    profile = str(metadata["physics"]["motion_profile"])
    view_rule = specialized_views[profile]
    variants = [float(value) for value in view_rule["azimuth_degrees"]]
    scene_digest = hashlib.sha256(str(metadata["scene_id"]).encode("utf-8")).digest()
    ordered = seeded_view_order(variants, scene_digest, 0)
    if prop:
        prop_offset = np.asarray(prop["position_m"][:2], dtype=np.float64) - target[:2]
        ordered = blocker_safe_seeded_view_order(
            variants,
            scene_digest,
            0,
            prop_offset,
        )
    azimuth_degrees = ordered[0]
    azimuth = math.radians(azimuth_degrees)
    elevation_degrees = seeded_view_order(
        [float(value) for value in view_rule["elevation_degrees"]],
        scene_digest,
        1,
    )[0]
    elevation = math.radians(elevation_degrees)
    distance = max(
        1.65,
        min(
            3.30 if prop else 3.05,
            max(
                1.52 * focus_span + 0.36,
                2.20 * vertical_span + 0.50,
            ),
        ),
    )
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = "asset_only_camera"
    camera.data.lens = 48.0
    camera.data.sensor_width = 36.0
    camera.data.clip_start = 0.03
    camera.data.clip_end = 100.0
    bpy.context.scene.camera = camera

    def set_pose(value: float) -> mathutils.Vector:
        horizontal = value * math.cos(elevation)
        location = mathutils.Vector(
            (
                float(target[0]) + math.cos(azimuth) * horizontal,
                float(target[1]) + math.sin(azimuth) * horizontal,
                float(target[2]) + value * math.sin(elevation),
            )
        )
        camera.location = location
        look_at(camera, mathutils.Vector(target))
        bpy.context.view_layer.update()
        return location

    context_half = np.asarray([0.5 * focus_x, 0.5 * focus_y, 0.0], dtype=np.float64)
    context_low = np.asarray(
        [
            target[0] - context_half[0],
            target[1] - context_half[1],
            float(surface["z_m"]),
        ]
    )
    context_high = np.asarray(
        [
            target[0] + context_half[0],
            target[1] + context_half[1],
            float(surface["z_m"]),
        ]
    )
    framing_points = aabb_corners(content_low, content_high)
    framing_points.extend(aabb_corners(context_low, context_high))
    location = set_pose(distance)
    projection_iterations = 0
    max_distance = 3.65 if prop else 3.40
    for projection_iterations in range(9):
        projected = [
            world_to_camera_view(bpy.context.scene, camera, point)
            for point in framing_points
        ]
        if all(
            point.z > 0.0
            and 0.055 <= point.x <= 0.945
            and 0.075 <= point.y <= 0.925
            for point in projected
        ):
            break
        next_distance = min(max_distance, distance * 1.10)
        if next_distance <= distance + 1.0e-6:
            break
        distance = next_distance
        location = set_pose(distance)
    return {
        "solver_version": "asset_motion_structure_camera_v2",
        "observation_intent": str(observation["intent"]),
        "structure_context": str(observation["structure_context"]),
        "position_m": [round(float(value), 6) for value in location],
        "target_m": [round(float(value), 6) for value in target],
        "focal_length_mm": 48.0,
        "focus_span_m": round(float(focus_span), 6),
        "azimuth_degrees": azimuth_degrees,
        "elevation_degrees": elevation_degrees,
        "prop_placed_behind_primary_target": bool(prop),
        "dynamic_extent_m": [round(float(value), 6) for value in dynamic_extent],
        "projection_fit_iterations": int(projection_iterations),
    }


def render(
    metadata_path: Path,
    video_path_override: Path | None = None,
    frame_dir_override: Path | None = None,
    *,
    mask_only: bool = False,
    instance_mask_dir: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    metadata = load_json(metadata_path)
    if metadata["schema_version"] != "physweep_asset_proxy_scene_v3":
        raise ValueError("unsupported metadata schema")
    registry_path = resolve(metadata["registry"]["path"])
    if sha256(registry_path) != metadata["registry"]["sha256"]:
        raise ValueError("asset proxy registry hash mismatch")
    semantic_rules_path = resolve(metadata["semantic_rules"]["path"])
    if sha256(semantic_rules_path) != metadata["semantic_rules"]["sha256"]:
        raise ValueError("asset semantic rules hash mismatch")
    composition_rules_path = resolve(metadata["composition_rules"]["path"])
    if sha256(composition_rules_path) != metadata["composition_rules"]["sha256"]:
        raise ValueError("asset scene composition rules hash mismatch")
    visual_rules_path = resolve(metadata["visual_rules"]["path"])
    if sha256(visual_rules_path) != metadata["visual_rules"]["sha256"]:
        raise ValueError("visual sampling rules hash mismatch")
    catalog_path = resolve(metadata["physical_proxy_catalog"]["path"])
    if sha256(catalog_path) != metadata["physical_proxy_catalog"]["sha256"]:
        raise ValueError("physical proxy catalog hash mismatch")
    records = {record["asset_id"]: record for record in load_json(registry_path)["records"]}
    simulation_record, trajectory_path, _ = validate_simulation_record(
        root=PROJECT_ROOT,
        metadata_path=metadata_path,
        metadata=metadata,
    )
    with np.load(trajectory_path) as source:
        trajectory = {key: source[key] for key in source.files}
    trajectory = adapter_trajectory_view(trajectory)
    dynamic_record = records[metadata["assets"]["dynamic_asset_id"]]
    prop_id = metadata["assets"]["static_prop_asset_id"]
    clear_scene()
    setup_scene(metadata)
    if not mask_only:
        apply_hdri(metadata["render"]["environment"])
    support_objects = add_support(metadata["physics"]["static_support_binding"])
    prop_objects = []
    if prop_id:
        prop_objects = add_static_prop(
            records[prop_id], metadata["physics"]["static_prop"]
        )
    dynamic_objects = add_dynamic(dynamic_record, trajectory)
    visual_rules = load_json(visual_rules_path)
    camera = add_camera(
        metadata,
        trajectory,
        dynamic_record,
        visual_rules["specialized_camera_views"],
    )
    dynamic_ids = [
        str(record["object_id"])
        for record in metadata["object_identity"]["objects"]
        if str(record["role"]) == "dynamic"
    ]
    if len(dynamic_ids) != 1:
        raise ValueError("asset proxy renderer requires exactly one dynamic identity")
    if mask_only:
        if frame_dir_override is None or instance_mask_dir is None:
            raise ValueError(
                "mask-only rendering requires an inspection record directory and mask directory"
            )
        frame_dir = frame_dir_override.resolve()
        frame_dir.mkdir(parents=True, exist_ok=True)
        record = render_instance_mask_record(
            root=PROJECT_ROOT,
            metadata_path=metadata_path,
            metadata=metadata,
            camera=camera,
            dynamic_objects={dynamic_ids[0]: dynamic_objects},
            mask_root=instance_mask_dir,
            renderer_path=Path(__file__),
        )
        write_json(frame_dir / "render_record.json", record)
        print(json.dumps(record, indent=2))
        return record
    add_environment(metadata["render"]["environment"], camera)
    add_lighting(
        mathutils.Vector(camera["target_m"]),
        camera["focus_span_m"],
        metadata["render"]["environment"]["lighting"],
    )
    lighting_adaptation = apply_material_lightness_adaptation(
        bpy.context.scene,
        dynamic_objects,
        [*support_objects, *prop_objects],
    )
    scene = bpy.context.scene
    frame_dir = (
        frame_dir_override.resolve()
        if frame_dir_override is not None
        else resolve(metadata["render"]["inspection_frame_dir"])
    )
    frame_dir.mkdir(parents=True, exist_ok=True)
    inspection_frames = [1, (scene.frame_end + 1) // 2, scene.frame_end]
    inspection_paths = []
    scene.render.image_settings.file_format = "PNG"
    for frame in inspection_frames:
        scene.frame_set(frame)
        path = frame_dir / f"frame_{frame:04d}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        inspection_paths.append(path)
    video_path = (
        video_path_override.resolve()
        if video_path_override is not None
        else resolve(metadata["render"]["video_path"])
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(video_path)
    video_encoding = configure_h264_output(
        scene,
        fps=int(metadata["physics"]["output_fps"]),
        frame_count=int(metadata["physics"]["frame_count"]),
    )
    render_samples = int(scene.eevee.taa_render_samples)
    bpy.ops.render.render(animation=True)
    normalize_h264_container(video_path)
    mask_path = metadata["object_identity"]["instance_masks"].get("path")
    instance_mask_output = None
    if instance_mask_dir is not None or (
        isinstance(mask_path, str) and bool(mask_path)
    ):
        instance_mask_output = render_instance_masks(
            root=PROJECT_ROOT,
            metadata=metadata,
            dynamic_objects={dynamic_ids[0]: dynamic_objects},
            mask_root_override=instance_mask_dir,
        )
    record = {
        "schema_version": "physweep_asset_proxy_render_record_v1",
        "scene_id": metadata["scene_id"],
        "metadata_sha256": sha256(metadata_path),
        "metadata_path": str(metadata_path),
        "render_output_overridden": bool(
            video_path_override is not None or frame_dir_override is not None
        ),
        "support_binding_sha256": metadata["physics"]["static_support_binding"][
            "binding_sha256"
        ],
        "video_path": str(video_path),
        "video_sha256": sha256(video_path),
        "inspection_frames": [str(path) for path in inspection_paths],
        "camera": camera,
        "lighting_adaptation": lighting_adaptation,
        "render_engine": scene.render.engine,
        "render_samples": render_samples,
        "video_encoding": video_encoding,
        "instance_mask_output": instance_mask_output,
        "implementation": render_implementation(Path(__file__)),
        "wall_time_s": round(time.perf_counter() - started, 6),
    }
    write_json(frame_dir / "render_record.json", record)
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    args = blender_args()
    configure_project_root(args.root)
    render(
        args.metadata.resolve(),
        args.video_path,
        args.inspection_frame_dir,
        mask_only=args.mask_only,
        instance_mask_dir=args.instance_mask_dir,
    )
