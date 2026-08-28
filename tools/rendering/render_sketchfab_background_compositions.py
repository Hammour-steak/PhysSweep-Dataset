#!/usr/bin/env python3
"""Render first-frame previews for Sketchfab background composition metadata."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

from tools.core.blender_runtime import blender_argv, patch_numpy_for_blender_gltf
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json
from tools.physics.support_structure import table_structure_boxes
from tools.physics.inclined_support import inclined_plane_geometry
from tools.assets.blender_asset_import import (
    mesh_world_bounds as bbox_for_objects,
    meshes_have_image_texture as object_has_image_texture,
)
from tools.rendering.blender_scene import look_at
from tools.rendering.lighting_quality import floor_glare_guard


DEFAULT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHADOW_LIGHTING_RULE = {
    "rule_id": "texture_visible_natural_soft_shadow_default_v4",
    "key_light": {
        "type": "AREA",
        "location": [-2.2, -2.4, 4.0],
        "energy": 350.0,
        "size": 4.0,
    },
    "fill_light": {
        "type": "AREA",
        "location": [2.0, 1.6, 2.6],
        "energy": 36.0,
        "size": 8.8,
    },
    "hdri_strength_multiplier": 0.24,
    "world_color": [0.018, 0.019, 0.021],
}


def resolve_project_asset_paths(value: Any, root: Path = DEFAULT_ROOT) -> Any:
    if isinstance(value, list):
        return [resolve_project_asset_paths(item, root) for item in value]
    if isinstance(value, dict):
        return {key: resolve_project_asset_paths(item, root) for key, item in value.items()}
    if isinstance(value, str) and (value.startswith("assets/") or value.startswith("configs/")):
        return str(root / value)
    return value


def merged_shadow_lighting_rule(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    rule = json.loads(json.dumps(DEFAULT_SHADOW_LIGHTING_RULE))
    if metadata:
        override = metadata.get("render", {}).get("shadow_lighting_rule", {})
        if isinstance(override, dict):
            for key, value in override.items():
                if isinstance(value, dict) and isinstance(rule.get(key), dict):
                    rule[key].update(value)
                else:
                    rule[key] = value
    return rule


def vector_list(vec: Any) -> list[float]:
    return [round(float(vec.x), 6), round(float(vec.y), 6), round(float(vec.z), 6)]


def make_solid_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float = 0.74,
    emission_strength: float = 0.0,
) -> Any:
    import bpy  # pylint: disable=import-outside-toplevel

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = color
    principled = mat.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = color
        principled.inputs["Roughness"].default_value = roughness
        if "Specular" in principled.inputs:
            principled.inputs["Specular"].default_value = 0.25
        if emission_strength > 0.0:
            if "Emission" in principled.inputs:
                principled.inputs["Emission"].default_value = color
            if "Emission Strength" in principled.inputs:
                principled.inputs["Emission Strength"].default_value = emission_strength
    return mat


def make_subtle_wall_material(
    name: str,
    color: tuple[float, float, float, float],
    noise_scale: float = 28.0,
    noise_strength: float = 0.10,
    bump_strength: float = 0.025,
) -> Any:
    import bpy  # pylint: disable=import-outside-toplevel

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = color
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    principled = nodes.get("Principled BSDF")
    if not principled:
        return mat

    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.92
    if "Specular" in principled.inputs:
        principled.inputs["Specular"].default_value = 0.12

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = noise_scale
    noise.inputs["Detail"].default_value = 9.0
    noise.inputs["Roughness"].default_value = 0.58

    ramp = nodes.new("ShaderNodeValToRGB")
    low = tuple(max(0.0, color[i] * (1.0 - noise_strength)) for i in range(3)) + (color[3],)
    high = tuple(min(1.0, color[i] * (1.0 + noise_strength * 0.65)) for i in range(3)) + (color[3],)
    ramp.color_ramp.elements[0].position = 0.20
    ramp.color_ramp.elements[0].color = low
    ramp.color_ramp.elements[1].position = 1.00
    ramp.color_ramp.elements[1].color = high
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], principled.inputs["Base Color"])

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = bump_strength
    bump.inputs["Distance"].default_value = 0.08
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    return mat


REQUIRED_SURFACE_GEOMETRY_KEYS = (
    "edge_model",
    "smooth_shading",
    "bevel_width_m",
    "bevel_segments",
    "weighted_normals",
    "mesh_resolution",
)


def require_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Metadata field `{label}` must be an object")
    return value


def require_keys(record: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in record]
    if missing:
        raise ValueError(f"Metadata field `{label}` is missing required keys: {missing}")


def strict_srgb_color(values: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        raise ValueError(f"Metadata field `{label}` must be RGB or RGBA")
    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]) if len(values) > 3 else 1.0,
    )


def srgb_color(values: list[Any] | tuple[Any, ...], fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) < 3:
        return fallback
    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]) if len(values) > 3 else 1.0,
    )


def polyhaven_texture_path(material_record: dict[str, Any] | None, stems: tuple[str, ...]) -> Path | None:
    if not material_record:
        return None
    base = Path(str(material_record.get("path", "")))
    texture_dir = base / "textures" if base.is_dir() else base.parent / "textures"
    if not texture_dir.exists():
        return None
    for stem in stems:
        matches = sorted(texture_dir.glob(f"*{stem}*"))
        if matches:
            return matches[0]
    return None


def make_polyhaven_material(
    name: str,
    material_record: dict[str, Any] | None,
    fallback_color_value: tuple[float, float, float, float],
    texture_scale: float = 2.5,
    minimum_roughness: float | None = None,
    maximum_specular: float | None = None,
) -> Any:
    import bpy  # pylint: disable=import-outside-toplevel

    diffuse_path = polyhaven_texture_path(material_record, ("_diff_", "_albedo_", "_basecolor_", "_base_color_", "_col_", "_color_"))
    if not diffuse_path:
        fallback = make_subtle_wall_material(
            name, fallback_color_value, noise_strength=0.08, bump_strength=0.018
        )
        principled = fallback.node_tree.nodes.get("Principled BSDF")
        if principled:
            if minimum_roughness is not None:
                principled.inputs["Roughness"].default_value = max(
                    float(principled.inputs["Roughness"].default_value),
                    max(0.0, min(1.0, float(minimum_roughness))),
                )
            if maximum_specular is not None and "Specular" in principled.inputs:
                principled.inputs["Specular"].default_value = min(
                    float(principled.inputs["Specular"].default_value),
                    max(0.0, min(1.0, float(maximum_specular))),
                )
        return fallback

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = fallback_color_value
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    principled = nodes.get("Principled BSDF")
    if not principled:
        return mat

    if minimum_roughness is not None:
        minimum_roughness = max(0.0, min(1.0, float(minimum_roughness)))
    if maximum_specular is not None:
        maximum_specular = max(0.0, min(1.0, float(maximum_specular)))
    if "Specular" in principled.inputs:
        principled.inputs["Specular"].default_value = min(
            0.18, maximum_specular if maximum_specular is not None else 0.18
        )
    principled.inputs["Roughness"].default_value = max(
        0.78, minimum_roughness if minimum_roughness is not None else 0.0
    )

    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (texture_scale, texture_scale, texture_scale)
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])

    diffuse = nodes.new("ShaderNodeTexImage")
    diffuse.image = bpy.data.images.load(str(diffuse_path))
    diffuse.extension = "REPEAT"
    links.new(mapping.outputs["Vector"], diffuse.inputs["Vector"])
    links.new(diffuse.outputs["Color"], principled.inputs["Base Color"])

    roughness_path = polyhaven_texture_path(material_record, ("_rough_", "_roughness_"))
    if roughness_path:
        roughness = nodes.new("ShaderNodeTexImage")
        roughness.image = bpy.data.images.load(str(roughness_path))
        roughness.image.colorspace_settings.name = "Non-Color"
        roughness.extension = "REPEAT"
        links.new(mapping.outputs["Vector"], roughness.inputs["Vector"])
        if minimum_roughness is None:
            links.new(roughness.outputs["Color"], principled.inputs["Roughness"])
        else:
            roughness_floor = nodes.new("ShaderNodeMath")
            roughness_floor.operation = "MAXIMUM"
            roughness_floor.inputs[1].default_value = minimum_roughness
            links.new(roughness.outputs["Color"], roughness_floor.inputs[0])
            links.new(roughness_floor.outputs["Value"], principled.inputs["Roughness"])

    normal_path = polyhaven_texture_path(material_record, ("_nor_gl_", "_normal_", "_nor_"))
    if normal_path:
        normal = nodes.new("ShaderNodeTexImage")
        normal.image = bpy.data.images.load(str(normal_path))
        normal.image.colorspace_settings.name = "Non-Color"
        normal.extension = "REPEAT"
        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.inputs["Strength"].default_value = 0.18
        links.new(mapping.outputs["Vector"], normal.inputs["Vector"])
        links.new(normal.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])

    return mat


def make_strict_polyhaven_material(
    name: str,
    material_record: dict[str, Any],
    fallback_color_value: tuple[float, float, float, float],
    texture_scale: float,
    semantic_color_mix: float = 0.0,
    material_coordinate_attribute: str | None = None,
) -> Any:
    import bpy  # pylint: disable=import-outside-toplevel

    require_keys(material_record, ("asset_source", "asset_id", "path"), f"{name}.poly_haven_material")
    if material_record["asset_source"] != "poly_haven_curated_v2":
        raise ValueError(f"{name}.poly_haven_material.asset_source must be poly_haven_curated_v2")
    diffuse_path = polyhaven_texture_path(material_record, ("_diff_", "_albedo_", "_basecolor_", "_base_color_", "_col_", "_color_"))
    if not diffuse_path:
        raise ValueError(f"Poly Haven material `{material_record.get('asset_id')}` has no diffuse/basecolor texture")

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = fallback_color_value
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    principled = nodes.get("Principled BSDF")
    if not principled:
        raise RuntimeError(f"Blender material `{name}` has no Principled BSDF")

    if "Specular" in principled.inputs:
        principled.inputs["Specular"].default_value = 0.18
    principled.inputs["Roughness"].default_value = 0.78

    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (texture_scale, texture_scale, texture_scale)
    if material_coordinate_attribute:
        texcoord = nodes.new("ShaderNodeAttribute")
        texcoord.attribute_name = material_coordinate_attribute
        links.new(texcoord.outputs["Vector"], mapping.inputs["Vector"])
    else:
        texcoord = nodes.new("ShaderNodeTexCoord")
        links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])

    diffuse = nodes.new("ShaderNodeTexImage")
    diffuse.image = bpy.data.images.load(str(diffuse_path))
    diffuse.extension = "REPEAT"
    links.new(mapping.outputs["Vector"], diffuse.inputs["Vector"])
    if semantic_color_mix > 0.0:
        semantic_color = nodes.new("ShaderNodeRGB")
        semantic_color.outputs["Color"].default_value = fallback_color_value
        color_mix = nodes.new("ShaderNodeMixRGB")
        color_mix.blend_type = "MIX"
        color_mix.inputs["Fac"].default_value = max(0.0, min(1.0, semantic_color_mix))
        links.new(diffuse.outputs["Color"], color_mix.inputs["Color1"])
        links.new(semantic_color.outputs["Color"], color_mix.inputs["Color2"])
        links.new(color_mix.outputs["Color"], principled.inputs["Base Color"])
    else:
        links.new(diffuse.outputs["Color"], principled.inputs["Base Color"])

    roughness_path = polyhaven_texture_path(material_record, ("_rough_", "_roughness_"))
    if roughness_path:
        roughness = nodes.new("ShaderNodeTexImage")
        roughness.image = bpy.data.images.load(str(roughness_path))
        roughness.image.colorspace_settings.name = "Non-Color"
        roughness.extension = "REPEAT"
        links.new(mapping.outputs["Vector"], roughness.inputs["Vector"])
        links.new(roughness.outputs["Color"], principled.inputs["Roughness"])

    normal_path = polyhaven_texture_path(material_record, ("_nor_gl_", "_normal_", "_nor_"))
    if normal_path:
        normal = nodes.new("ShaderNodeTexImage")
        normal.image = bpy.data.images.load(str(normal_path))
        normal.image.colorspace_settings.name = "Non-Color"
        normal.extension = "REPEAT"
        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.inputs["Strength"].default_value = 0.34
        links.new(mapping.outputs["Vector"], normal.inputs["Vector"])
        links.new(normal.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])
    return mat


def make_object_specific_pbr_material(name: str, record: dict[str, Any]) -> Any:
    import bpy  # pylint: disable=import-outside-toplevel

    require_keys(
        record,
        (
            "profile_id",
            "pattern",
            "base_color",
            "roughness",
            "metallic",
            "variation_strength",
            "bump_strength",
            "pattern_scale",
        ),
        f"{name}.object_specific_pbr",
    )
    base_color = strict_srgb_color(record["base_color"], f"{name}.object_specific_pbr.base_color")
    variation = max(0.0, min(0.45, float(record["variation_strength"])))
    pattern = str(record["pattern"])

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.diffuse_color = base_color
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    principled = nodes.get("Principled BSDF")
    if not principled:
        raise RuntimeError(f"Blender material `{name}` has no Principled BSDF")
    principled.inputs["Roughness"].default_value = float(record["roughness"])
    principled.inputs["Metallic"].default_value = float(record["metallic"])
    if "Specular" in principled.inputs:
        principled.inputs["Specular"].default_value = 0.28

    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = float(record["pattern_scale"])
    noise.inputs["Detail"].default_value = 6.0 if pattern == "paper_fiber" else 3.5
    noise.inputs["Roughness"].default_value = 0.62
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    pattern_fac = noise.outputs["Fac"]

    if pattern in {"wood_grain", "painted_metal_bands", "rolling_contrast_bands"}:
        wave = nodes.new("ShaderNodeTexWave")
        wave.wave_type = "BANDS"
        wave.bands_direction = "X" if pattern == "wood_grain" else "Z"
        wave.inputs["Scale"].default_value = float(record["pattern_scale"])
        wave.inputs["Distortion"].default_value = 5.5 if pattern == "wood_grain" else (0.05 if pattern == "rolling_contrast_bands" else 0.35)
        wave.inputs["Detail"].default_value = 4.0 if pattern == "wood_grain" else 2.0
        links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
        if pattern == "rolling_contrast_bands":
            pattern_fac = wave.outputs["Color"]
        else:
            mix = nodes.new("ShaderNodeMixRGB")
            mix.blend_type = "MULTIPLY"
            mix.inputs["Fac"].default_value = 0.72 if pattern == "wood_grain" else 0.38
            links.new(noise.outputs["Fac"], mix.inputs["Color1"])
            links.new(wave.outputs["Color"], mix.inputs["Color2"])
            pattern_fac = mix.outputs["Color"]

    ramp = nodes.new("ShaderNodeValToRGB")
    low = tuple(max(0.0, base_color[i] * (1.0 - variation)) for i in range(3)) + (base_color[3],)
    high = tuple(min(1.0, base_color[i] * (1.0 + variation * 0.72)) for i in range(3)) + (base_color[3],)
    ramp.color_ramp.elements[0].position = 0.18
    ramp.color_ramp.elements[0].color = low
    ramp.color_ramp.elements[1].position = 0.86
    ramp.color_ramp.elements[1].color = high
    if pattern == "rolling_contrast_bands":
        ramp.color_ramp.interpolation = "CONSTANT"
        ramp.color_ramp.elements[0].position = 0.44
        ramp.color_ramp.elements[0].color = tuple(max(0.0, base_color[i] * 0.22) for i in range(3)) + (base_color[3],)
        ramp.color_ramp.elements[1].position = 0.56
        ramp.color_ramp.elements[1].color = tuple(min(1.0, base_color[i] * 1.18) for i in range(3)) + (base_color[3],)
    links.new(pattern_fac, ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], principled.inputs["Base Color"])

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = float(record["bump_strength"])
    bump.inputs["Distance"].default_value = 0.035
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    return mat


def room_wall_indices_to_skip(camera_azimuth_degrees: float) -> list[int]:
    azimuth = math.radians(camera_azimuth_degrees)
    camera_side = (math.cos(azimuth), math.sin(azimuth))
    wall_sides = ((0.0, 1.0), (0.0, -1.0), (1.0, 0.0), (-1.0, 0.0))
    return [
        index
        for index, side in enumerate(wall_sides)
        if side[0] * camera_side[0] + side[1] * camera_side[1] > 0.05
    ]


def add_procedural_backdrop(
    metadata: dict[str, Any],
    support_top_z: float,
    camera_azimuth_degrees: float | None = None,
    lighting_rule: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    import bpy  # pylint: disable=import-outside-toplevel

    backdrop = metadata.get("environment", {}).get("backdrop", {})
    if backdrop.get("enabled", True) is False:
        return None

    wall_color = srgb_color(backdrop.get("wall_color", []), (0.50, 0.51, 0.50, 1.0))
    floor_color = srgb_color(backdrop.get("floor_color", []), (0.36, 0.34, 0.31, 1.0))
    camera_view = metadata.get("camera", {}).get("view_rule", {})
    azimuth_degrees = float(
        camera_azimuth_degrees
        if camera_azimuth_degrees is not None
        else camera_view.get("azimuth_degrees", -55.2)
    )
    azimuth = math.radians(azimuth_degrees)
    follow_camera_view = bool(backdrop.get("follow_camera_view", False))
    wall_distance = float(backdrop.get("wall_distance_m", 2.15))
    wall_y = float(backdrop.get("wall_y", 3.2))
    wall_width = float(backdrop.get("wall_width", 8.0))
    wall_height = float(backdrop.get("wall_height", 4.0))
    floor_width = float(backdrop.get("floor_width", 12.0))
    floor_depth = float(backdrop.get("floor_depth", 10.0))
    secondary_surface = metadata.get("simulation", {}).get(
        "secondary_contact_surface", {}
    )
    if secondary_surface:
        if secondary_surface.get("renderer_binding") != "environment.backdrop.floor":
            raise ValueError("secondary contact surface renderer binding is unsupported")
        floor_z = float(secondary_surface["plane_z_m"])
    else:
        floor_z = float(backdrop.get("floor_z_m", 0.0))
    room_walls = bool(backdrop.get("room_walls", True))
    floor_center_y = 0.0
    skipped_near_walls: list[int] = []

    bpy.context.scene.world.color = wall_color[:3]

    environment = metadata.get("environment", {})
    wall_mat = make_polyhaven_material(
        "procedural_backdrop_wall",
        environment.get("backdrop_material"),
        wall_color,
        texture_scale=float(backdrop.get("wall_texture_scale", 2.8)),
    )
    glare_guard = floor_glare_guard(lighting_rule)
    floor_mat = make_polyhaven_material(
        "procedural_backdrop_floor",
        environment.get("floor_material"),
        floor_color,
        texture_scale=float(backdrop.get("floor_texture_scale", 3.2)),
        minimum_roughness=(
            float(glare_guard["minimum_roughness"]) if glare_guard else None
        ),
        maximum_specular=(
            float(glare_guard["maximum_specular"]) if glare_guard else None
        ),
    )

    if room_walls:
        wall_locations = [
            ((0.0, wall_y, wall_height / 2.0), 0.0),
            ((0.0, -wall_y, wall_height / 2.0), math.radians(180.0)),
            ((wall_y, 0.0, wall_height / 2.0), math.radians(90.0)),
            ((-wall_y, 0.0, wall_height / 2.0), math.radians(-90.0)),
        ]
        near_wall_indices = set(room_wall_indices_to_skip(azimuth_degrees))
        walls = []
        for wall_index, (wall_location, wall_rotation_z) in enumerate(wall_locations):
            if wall_index in near_wall_indices:
                skipped_near_walls.append(wall_index)
                continue
            bpy.ops.mesh.primitive_plane_add(size=1.0, location=wall_location)
            wall = bpy.context.object
            wall.name = f"procedural_backdrop_wall_{wall_index:02d}"
            wall.rotation_euler[0] = math.radians(90.0)
            wall.rotation_euler[2] = wall_rotation_z
            wall.scale = (wall_width, wall_height, 1.0)
            wall.data.materials.append(wall_mat)
            walls.append(wall)
        wall_location = wall_locations[0][0]
    elif follow_camera_view:
        wall_location = (-math.cos(azimuth) * wall_distance, -math.sin(azimuth) * wall_distance, wall_height / 2.0)
        wall_rotation_z = azimuth
        floor_center_y = 0.0
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=wall_location)
        wall = bpy.context.object
        wall.name = "procedural_backdrop_wall"
        wall.rotation_euler[0] = math.radians(90.0)
        wall.rotation_euler[2] = wall_rotation_z
        wall.scale = (wall_width, wall_height, 1.0)
        wall.data.materials.append(wall_mat)
    else:
        wall_location = (0.0, wall_y, wall_height / 2.0)
        wall_rotation_z = 0.0
        floor_center_y = wall_y - floor_depth / 2.0
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=wall_location)
        wall = bpy.context.object
        wall.name = "procedural_backdrop_wall"
        wall.rotation_euler[0] = math.radians(90.0)
        wall.rotation_euler[2] = wall_rotation_z
        wall.scale = (wall_width, wall_height, 1.0)
        wall.data.materials.append(wall_mat)

    bpy.ops.mesh.primitive_plane_add(
        size=1.0, location=(0.0, floor_center_y, floor_z)
    )
    floor = bpy.context.object
    floor.name = "procedural_backdrop_floor"
    floor.scale = (floor_width, floor_depth, 1.0)
    floor.data.materials.append(floor_mat)

    return {
        "wall_y": round(wall_y, 4),
        "wall_location": [round(float(v), 4) for v in wall_location],
        "wall_follows_camera_view": follow_camera_view,
        "room_walls": room_walls,
        "skipped_near_walls": skipped_near_walls,
        "camera_azimuth_degrees": round(azimuth_degrees, 4),
        "wall_width": round(wall_width, 4),
        "wall_height": round(wall_height, 4),
        "floor_width": round(floor_width, 4),
        "floor_depth": round(floor_depth, 4),
        "floor_z_m": round(floor_z, 6),
        "wall_color": [round(float(v), 4) for v in wall_color],
        "floor_color": [round(float(v), 4) for v in floor_color],
        "floor_glare_guard": glare_guard,
    }


def create_procedural_support(support: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
    import bpy  # pylint: disable=import-outside-toplevel

    placement = support.get("placement", {})
    target = placement.get("target_size", [2.8, 1.2, 0.78])
    target_x = float(target[0])
    target_y = float(target[1])
    support_top_z = float(placement.get("support_top_z", target[2] if len(target) > 2 else 0.78))
    thickness = float(placement.get("thickness", 0.08))
    position = placement.get("position", [0.0, 0.0, support_top_z])
    yaw = math.radians(float(placement.get("yaw_degrees", 0.0)))
    rotation = placement.get("rotation_degrees", [0.0, 0.0, 0.0])
    if not isinstance(rotation, list) or len(rotation) != 3:
        rotation = [0.0, 0.0, 0.0]
    pitch = math.radians(float(rotation[0]))
    roll = math.radians(float(rotation[1]))
    support_shape = str(placement.get("support_shape", "rectangular_slab"))

    if support_shape == "inclined_ramp":
        ramp_geometry = inclined_plane_geometry(placement)
        mesh = bpy.data.meshes.new(f"support_{support.get('asset_id', 'procedural_poly_haven')}_mesh")
        center_x = float(position[0])
        center_y = float(position[1])
        half_x = target_x / 2.0
        half_y = target_y / 2.0
        low_top = float(ramp_geometry["low_top_z_m"])
        high_top = float(ramp_geometry["high_top_z_m"])
        bottom_z = low_top - thickness
        vertices = [
            (center_x - half_x, center_y - half_y, bottom_z),
            (center_x + half_x, center_y - half_y, bottom_z),
            (center_x + half_x, center_y + half_y, bottom_z),
            (center_x - half_x, center_y + half_y, bottom_z),
            (center_x - half_x, center_y - half_y, low_top),
            (center_x + half_x, center_y - half_y, low_top),
            (center_x + half_x, center_y + half_y, high_top),
            (center_x - half_x, center_y + half_y, high_top),
        ]
        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (3, 7, 4, 0),
        ]
        mesh.from_pydata(vertices, [], faces)
        mesh.update()
        slab = bpy.data.objects.new(f"support_{support.get('asset_id', 'procedural_poly_haven')}", mesh)
        bpy.context.collection.objects.link(slab)
        bpy.context.view_layer.objects.active = slab
    else:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(float(position[0]), float(position[1]), support_top_z - thickness / 2.0))
        slab = bpy.context.object
        slab.name = f"support_{support.get('asset_id', 'procedural_poly_haven')}"
        slab.dimensions = (target_x, target_y, thickness)
        bpy.context.view_layer.objects.active = slab
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        slab.rotation_euler = (pitch, roll, yaw)
    material_record = metadata.get("materials", {}).get(support.get("material_assignment_key", "floor_or_contact_surface"))
    support_texture_scale = float(placement.get("poly_haven_texture_scale", placement.get("texture_scale", 2.6)))
    support_material = make_polyhaven_material(
        "procedural_support_polyhaven",
        material_record,
        (0.50, 0.42, 0.33, 1.0),
        texture_scale=support_texture_scale,
    )
    slab.data.materials.append(support_material)

    objects = [slab]
    if support_shape == "tray_surface":
        rail_height = float(placement.get("rail_height_m", 0.10))
        rail_width = float(placement.get("rail_width_m", 0.045))
        rail_z = support_top_z + rail_height / 2.0
        rail_specs = [
            ((0.0, target_y / 2.0 + rail_width / 2.0, rail_z), (target_x, rail_width, rail_height)),
            ((0.0, -target_y / 2.0 - rail_width / 2.0, rail_z), (target_x, rail_width, rail_height)),
        ]
        if support_shape == "tray_surface":
            rail_specs.extend(
                [
                    ((target_x / 2.0 + rail_width / 2.0, 0.0, rail_z), (rail_width, target_y + 2.0 * rail_width, rail_height)),
                    ((-target_x / 2.0 - rail_width / 2.0, 0.0, rail_z), (rail_width, target_y + 2.0 * rail_width, rail_height)),
                ]
            )
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        for rail_index, (rail_location, rail_dimensions) in enumerate(rail_specs):
            local_x, local_y, local_z = rail_location
            world_location = (
                float(position[0]) + local_x * cos_yaw - local_y * sin_yaw,
                float(position[1]) + local_x * sin_yaw + local_y * cos_yaw,
                local_z,
            )
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=world_location)
            rail = bpy.context.object
            rail.name = f"{slab.name}_rail_{rail_index:02d}"
            rail.dimensions = rail_dimensions
            bpy.context.view_layer.objects.active = rail
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            rail.rotation_euler[2] = yaw
            rail.data.materials.append(support_material)
            objects.append(rail)

    structure_boxes = []
    if support_shape != "inclined_ramp" and placement.get("support_structure", {}).get("profile_id"):
        structure_boxes = table_structure_boxes(placement)
        leg_material = make_solid_material("procedural_support_leg_material", (0.25, 0.20, 0.15, 1.0), roughness=0.78)
        for box in structure_boxes:
            bpy.ops.mesh.primitive_cube_add(size=1.0, location=box["center_m"])
            structure_object = bpy.context.object
            structure_object.name = f"{slab.name}_{box['id']}"
            structure_object.dimensions = tuple(float(value) for value in box["dimensions_m"])
            bpy.context.view_layer.objects.active = structure_object
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            structure_object.rotation_euler[2] = math.radians(float(box["yaw_degrees"]))
            structure_object.data.materials.append(leg_material)
            objects.append(structure_object)

    mins, maxs = bbox_for_objects(objects)
    return {
        "asset_id": support.get("asset_id"),
        "name": support.get("name"),
        "role": support.get("composition_role", support.get("role")),
        "slot": support.get("slot"),
        "mesh_count": len(objects),
        "has_image_texture": bool(polyhaven_texture_path(material_record, ("_diff_", "_albedo_", "_basecolor_", "_base_color_"))),
        "source_bbox_size": [round(target_x, 6), round(target_y, 6), round(thickness, 6)],
        "target_size": [round(target_x, 6), round(target_y, 6), round(thickness, 6)],
        "scale": 1.0,
        "auto_yaw_degrees": 0.0,
        "rotation_degrees": [round(float(rotation[0]), 4), round(float(rotation[1]), 4), round(math.degrees(yaw), 4)],
        "final_bbox_min": vector_list(mins),
        "final_bbox_max": vector_list(maxs),
        "support_structure": {
            "profile_id": placement.get("support_structure", {}).get("profile_id"),
            "box_count": len(structure_boxes),
            "boxes": structure_boxes,
        },
        "objects": objects,
    }


def fallback_color(asset: dict[str, Any]) -> tuple[float, float, float, float]:
    category = asset.get("semantic_category") or asset.get("role") or ""
    role = asset.get("composition_role") or asset.get("slot") or ""
    if "table" in category or "bench" in category or "support" in role:
        return (0.52, 0.34, 0.20, 1.0)
    if "wall" in category or "context" in role:
        return (0.58, 0.57, 0.53, 1.0)
    if "lamp" in category:
        return (0.25, 0.24, 0.22, 1.0)
    if "books" in category:
        return (0.34, 0.24, 0.18, 1.0)
    if "tableware" in category:
        return (0.86, 0.82, 0.75, 1.0)
    if asset.get("slot") == "object_a":
        return (0.80, 0.16, 0.12, 1.0)
    if asset.get("slot") == "object_b":
        return (0.12, 0.26, 0.70, 1.0)
    if asset.get("slot") == "object_c":
        return (0.92, 0.76, 0.14, 1.0)
    return (0.55, 0.42, 0.31, 1.0)


def light_location(light_rule: dict[str, Any], fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    location = light_rule.get("location", fallback)
    if not isinstance(location, list) or len(location) < 3:
        return fallback
    return (float(location[0]), float(location[1]), float(location[2]))


def setup_scene(resolution: tuple[int, int], samples: int, lighting_rule: dict[str, Any]) -> None:
    import bpy  # pylint: disable=import-outside-toplevel

    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    if hasattr(scene.render, "use_persistent_data"):
        scene.render.use_persistent_data = True
    configure_cycles_device(bpy)

    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    scene.world = bpy.data.worlds.new("composition_world") if scene.world is None else scene.world
    world_color = lighting_rule.get("world_color", DEFAULT_SHADOW_LIGHTING_RULE["world_color"])
    scene.world.color = tuple(float(value) for value in world_color[:3])

    key_rule = lighting_rule.get("key_light", DEFAULT_SHADOW_LIGHTING_RULE["key_light"])
    bpy.ops.object.light_add(type=str(key_rule.get("type", "AREA")), location=light_location(key_rule, (-1.8, -2.4, 3.8)))
    key = bpy.context.object
    key.name = "directional_softbox_key"
    key.data.energy = float(key_rule.get("energy", 520.0))
    key.data.size = float(key_rule.get("size", 1.15))

    fill_rule = lighting_rule.get("fill_light", DEFAULT_SHADOW_LIGHTING_RULE["fill_light"])
    bpy.ops.object.light_add(type=str(fill_rule.get("type", "AREA")), location=light_location(fill_rule, (2.0, 1.6, 2.6)))
    fill = bpy.context.object
    fill.name = "weak_softbox_fill"
    fill.data.energy = float(fill_rule.get("energy", 12.0))
    fill.data.size = float(fill_rule.get("size", 6.0))


def configure_cycles_device(bpy: Any) -> None:
    scene = bpy.context.scene
    requested = os.environ.get("PHYSWEEP_CYCLES_DEVICE", "OPTIX").upper()
    if requested not in {"OPTIX", "CUDA", "CPU"}:
        requested = "OPTIX"
    if requested == "CPU":
        scene.cycles.device = "CPU"
        print("PhysSweep render device: CPU")
        return

    prefs = bpy.context.preferences.addons["cycles"].preferences
    chosen_type = None
    for candidate in ([requested] if requested in {"OPTIX", "CUDA"} else []) + ["CUDA"]:
        try:
            prefs.compute_device_type = candidate
            prefs.get_devices()
        except Exception as exc:  # pragma: no cover - Blender runtime branch
            print(f"PhysSweep render device warning: {candidate} unavailable: {exc}")
            continue
        gpu_devices = [device for device in prefs.devices if device.type == candidate]
        if gpu_devices:
            chosen_type = candidate
            break

    if not chosen_type:
        scene.cycles.device = "CPU"
        print("PhysSweep render device: CPU fallback; no CUDA/OPTIX devices found")
        return

    prefs.compute_device_type = chosen_type
    prefs.get_devices()
    enabled = []
    for device in prefs.devices:
        device.use = device.type == chosen_type
        if device.use:
            enabled.append(device.name)
    scene.cycles.device = "GPU"
    print(f"PhysSweep render device: {chosen_type} GPU, enabled={enabled}")


def apply_environment_lighting(metadata: dict[str, Any], lighting_rule: dict[str, Any]) -> None:
    import bpy  # pylint: disable=import-outside-toplevel

    lighting = metadata.get("environment", {}).get("lighting", {})
    hdri_path = Path(str(lighting.get("path", "")))
    if not hdri_path.exists():
        return

    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputWorld")
    background = nodes.new("ShaderNodeBackground")
    background.inputs["Strength"].default_value = float(lighting.get("strength", 0.55)) * float(
        lighting_rule.get("hdri_strength_multiplier", 0.32)
    )

    environment = nodes.new("ShaderNodeTexEnvironment")
    environment.image = bpy.data.images.load(str(hdri_path), check_existing=True)
    mapping = nodes.new("ShaderNodeMapping")
    texcoord = nodes.new("ShaderNodeTexCoord")
    mapping.inputs["Rotation"].default_value[2] = math.radians(float(lighting.get("rotation", 0.0)))

    links.new(texcoord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], environment.inputs["Vector"])
    links.new(environment.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])


def proxy_dimensions(proxy_geometry: dict[str, Any]) -> list[float]:
    scale = proxy_geometry.get("scale", {})
    geometry_id = proxy_geometry.get("geometry_id", "")
    if geometry_id == "sphere":
        r = float(scale.get("radius", 0.18))
        return [2 * r, 2 * r, 2 * r]
    if geometry_id in {"cuboid", "book_proxy"}:
        return [
            float(scale.get("x", scale.get("overall_x", 0.5))),
            float(scale.get("y", scale.get("overall_y", 0.4))),
            float(scale.get("z", scale.get("overall_z", 0.3))),
        ]
    if geometry_id in {"bottle_proxy", "mug_proxy", "bowl_proxy"}:
        r = float(scale.get("radius", 0.16))
        return [2 * r, 2 * r, float(scale.get("height", 0.42))]
    if geometry_id == "toy_car_proxy":
        return [float(scale.get("x", 0.7)), float(scale.get("y", 0.34)), float(scale.get("z", 0.24))]
    if geometry_id in {"L_block", "T_block", "cross_block"}:
        return [
            float(scale.get("overall_x", 0.7)),
            float(scale.get("overall_y", 0.35)),
            float(scale.get("thickness", 0.12)),
        ]
    return [0.5, 0.4, 0.3]


def target_size_for_asset(asset: dict[str, Any]) -> list[float]:
    placement = asset.get("placement", {})
    if "target_size" in placement:
        return [float(v) for v in placement["target_size"]]
    return proxy_dimensions(asset.get("proxy_geometry", {}))


def auto_yaw_for_asset(asset: dict[str, Any], source_size: Any) -> float:
    target = target_size_for_asset(asset)
    role = asset.get("composition_role") or asset.get("role") or ""
    if "support" not in role:
        return 0.0
    sx, sy = max(float(source_size.x), 1e-8), max(float(source_size.y), 1e-8)
    source_aspect = max(sx, sy) / max(min(sx, sy), 1e-8)
    target_aspect = max(target[0], target[1]) / max(min(target[0], target[1]), 1e-8)
    if source_aspect < 1.25 or target_aspect < 1.25:
        return 0.0
    source_long_axis = 0 if sx >= sy else 1
    target_long_axis = 0 if target[0] >= target[1] else 1
    if source_long_axis == target_long_axis:
        return 0.0
    return math.pi / 2.0 if source_long_axis == 1 else -math.pi / 2.0


def rotated_size_for_yaw(source_size: Any, yaw: float) -> tuple[float, float, float]:
    if abs(abs(yaw) - math.pi / 2.0) < 1e-4:
        return float(source_size.y), float(source_size.x), float(source_size.z)
    return float(source_size.x), float(source_size.y), float(source_size.z)


def scale_for_asset(asset: dict[str, Any], source_size: tuple[float, float, float]) -> float:
    target = target_size_for_asset(asset)
    sx, sy, sz = max(source_size[0], 1e-8), max(source_size[1], 1e-8), max(source_size[2], 1e-8)
    category = asset.get("semantic_category") or ""
    role = asset.get("composition_role") or asset.get("role") or ""
    if "support" in role:
        return min(target[0] / sx, target[1] / sy)
    if "context" in role or category.startswith("context_"):
        return min(target[0] / sx, target[2] / sz)
    return max(target) / max(sx, sy, sz)


def import_glb(asset: dict[str, Any]) -> list[Any]:
    patch_numpy_for_blender_gltf()
    import bpy  # pylint: disable=import-outside-toplevel

    glb_path = Path(asset["glb_path"])
    if not glb_path.exists():
        raise FileNotFoundError(glb_path)

    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    for obj in list(imported):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
            imported.remove(obj)
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError(f"No mesh object imported from {glb_path}")
    for obj in meshes:
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = world
    return meshes


def create_procedural_foreground(asset: dict[str, Any], prefix: str) -> list[Any]:
    import bpy  # pylint: disable=import-outside-toplevel

    placement = require_dict(asset.get("placement"), f"{prefix}.placement")
    require_keys(placement, ("target_size",), f"{prefix}.placement")
    target = [float(v) for v in placement["target_size"]]
    if len(target) != 3:
        raise ValueError(f"Metadata field `{prefix}.placement.target_size` must have three values")
    primitive = require_dict(asset.get("primitive"), f"{prefix}.primitive")
    require_keys(primitive, ("shape",), f"{prefix}.primitive")
    shape = str(primitive["shape"])
    visual_material = require_dict(asset.get("visual_material"), f"{prefix}.visual_material")
    require_keys(
        visual_material,
        ("material_model", "color", "roughness", "material_hint", "surface_geometry"),
        f"{prefix}.visual_material",
    )
    color = strict_srgb_color(visual_material["color"], f"{prefix}.visual_material.color")
    material_model = str(visual_material["material_model"])
    material_name = f"{prefix}_{asset.get('slot', 'primitive')}_material"
    object_specific_record: dict[str, Any] | None = None
    if material_model == "poly_haven_curated_v2_material":
        require_keys(
            visual_material,
            ("poly_haven_material", "poly_haven_texture_scale"),
            f"{prefix}.visual_material",
        )
        material = make_strict_polyhaven_material(
            material_name,
            require_dict(visual_material["poly_haven_material"], f"{prefix}.visual_material.poly_haven_material"),
            color,
            texture_scale=float(visual_material["poly_haven_texture_scale"]),
            semantic_color_mix=float(visual_material.get("semantic_color_mix", 0.84)),
        )
    elif material_model == "object_specific_pbr_v1":
        object_specific_record = require_dict(
            visual_material.get("object_specific_pbr"), f"{prefix}.visual_material.object_specific_pbr"
        )
        material = make_object_specific_pbr_material(
            material_name,
            object_specific_record,
        )
    else:
        raise ValueError(
            f"{prefix}.visual_material.material_model must be poly_haven_curated_v2_material "
            "or object_specific_pbr_v1"
        )

    surface_geometry = require_dict(visual_material["surface_geometry"], f"{prefix}.visual_material.surface_geometry")
    require_keys(surface_geometry, REQUIRED_SURFACE_GEOMETRY_KEYS, f"{prefix}.visual_material.surface_geometry")
    mesh_resolution = require_dict(surface_geometry["mesh_resolution"], f"{prefix}.visual_material.surface_geometry.mesh_resolution")
    require_keys(mesh_resolution, ("sphere_segments", "sphere_rings", "cylinder_vertices"), f"{prefix}.visual_material.surface_geometry.mesh_resolution")

    if shape == "sphere":
        radius = max(float(target[0]), float(target[1]), float(target[2])) / 2.0
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=int(mesh_resolution["sphere_segments"]),
            ring_count=int(mesh_resolution["sphere_rings"]),
            radius=radius,
            location=(0.0, 0.0, radius),
        )
    elif shape in {"cylinder", "bottle_proxy"}:
        radius = max(float(target[0]), float(target[1])) / 2.0
        depth = float(target[2])
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=int(mesh_resolution["cylinder_vertices"]),
            radius=radius,
            depth=depth,
            location=(0.0, 0.0, depth / 2.0),
        )
    else:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, float(target[2]) / 2.0))
        obj = bpy.context.object
        obj.dimensions = (float(target[0]), float(target[1]), float(target[2]))
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    obj = bpy.context.object
    obj.name = f"{prefix}_{asset.get('asset_id', asset.get('slot', 'procedural_primitive'))}"
    smooth_shading = bool(surface_geometry["smooth_shading"])
    if smooth_shading:
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.shade_smooth()
    bevel_width = float(surface_geometry["bevel_width_m"])
    bevel_segments = int(surface_geometry["bevel_segments"])
    weighted_normals = bool(surface_geometry["weighted_normals"])
    if shape != "sphere" and bevel_width > 0.0:
        obj.data.use_auto_smooth = True
        bevel = obj.modifiers.new("procedural_edge_softening", "BEVEL")
        bevel.width = bevel_width
        bevel.segments = max(1, bevel_segments)
        bevel.affect = "EDGES"
        if weighted_normals:
            normal = obj.modifiers.new("procedural_weighted_normals", "WEIGHTED_NORMAL")
            normal.keep_sharp = True
    obj.data.materials.append(material)
    objects = [obj]
    if object_specific_record and object_specific_record.get("profile_id") == "painted_metal_can_v1":
        radius = max(float(target[0]), float(target[1])) / 2.0
        depth = float(target[2])
        lid_depth = min(0.009, depth * 0.045)
        lid_material = make_solid_material(
            f"{material_name}_metal_lids",
            (0.36, 0.38, 0.40, 1.0),
            roughness=0.26,
        )
        lid_principled = lid_material.node_tree.nodes.get("Principled BSDF")
        if lid_principled:
            lid_principled.inputs["Metallic"].default_value = 0.82
        for lid_index, lid_z in enumerate((lid_depth / 2.0, depth - lid_depth / 2.0)):
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=int(mesh_resolution["cylinder_vertices"]),
                radius=radius * 1.012,
                depth=lid_depth,
                location=(0.0, 0.0, lid_z),
            )
            lid = bpy.context.object
            lid.name = f"{obj.name}_lid_{lid_index}"
            lid.data.materials.append(lid_material)
            objects.append(lid)
    return objects


def footprint_points(mins: Any, maxs: Any) -> list[tuple[float, float]]:
    center_x = (float(mins.x) + float(maxs.x)) / 2.0
    center_y = (float(mins.y) + float(maxs.y)) / 2.0
    span_x = float(maxs.x) - float(mins.x)
    span_y = float(maxs.y) - float(mins.y)
    if span_x < 1e-5 or span_y < 1e-5:
        return [(center_x, center_y)]
    inset_x = span_x * 0.28
    inset_y = span_y * 0.28
    return [
        (center_x, center_y),
        (float(mins.x) + inset_x, float(mins.y) + inset_y),
        (float(mins.x) + inset_x, float(maxs.y) - inset_y),
        (float(maxs.x) - inset_x, float(mins.y) + inset_y),
        (float(maxs.x) - inset_x, float(maxs.y) - inset_y),
    ]


def raycast_support_height(
    support_objects: list[Any],
    x: float,
    y: float,
    fallback_z: float,
    max_delta: float = 0.36,
) -> float | None:
    # Ramps can differ from their semantic center plane by 20cm+.
    # Keep the gate wide enough for inclined supports while still rejecting
    # unrelated geometry far above or below the contact surface.
    import mathutils  # pylint: disable=import-outside-toplevel

    origin_world = mathutils.Vector((x, y, fallback_z + 3.0))
    direction_world = mathutils.Vector((0.0, 0.0, -1.0))
    candidates: list[float] = []
    for obj in support_objects:
        if obj.type != "MESH":
            continue
        matrix_inv = obj.matrix_world.inverted()
        origin_local = matrix_inv @ origin_world
        direction_local = matrix_inv.to_3x3() @ direction_world
        if direction_local.length < 1e-8:
            continue
        hit, location, normal, _face_index = obj.ray_cast(origin_local, direction_local.normalized(), distance=6.0)
        if not hit:
            continue
        world_location = obj.matrix_world @ location
        world_normal = (obj.matrix_world.to_3x3() @ normal).normalized()
        if float(world_normal.z) < 0.35:
            continue
        z = float(world_location.z)
        if fallback_z - max_delta <= z <= fallback_z + max_delta:
            candidates.append(z)
    if not candidates:
        return None
    return max(candidates)


def calculated_contact_height(
    support_objects: list[Any],
    object_mins: Any,
    object_maxs: Any,
    fallback_z: float,
    contact_surface_mode: str = "support_mesh_raycast",
) -> tuple[float, int, str]:
    if contact_surface_mode == "semantic_plane_only":
        return fallback_z, 0, "semantic_support_plane"
    points = footprint_points(object_mins, object_maxs)
    heights = [
        height
        for x, y in points
        if (height := raycast_support_height(support_objects, x, y, fallback_z)) is not None
    ]
    if len(heights) != len(points):
        raise ValueError(
            f"Support-mesh contact raycast resolved {len(heights)}/{len(points)} footprint points. "
            "Fix the support geometry, normals, or placement instead of falling back to a flat semantic plane."
        )
    return max(heights), len(heights), "support_mesh_raycast"


def clamp_xy_to_safe_surface(
    position: list[float],
    asset: dict[str, Any],
    safe_bounds: dict[str, list[float]] | None,
) -> tuple[list[float], bool]:
    if not safe_bounds:
        return position, False

    x_bounds = safe_bounds.get("x")
    y_bounds = safe_bounds.get("y")
    if not x_bounds or not y_bounds or len(x_bounds) != 2 or len(y_bounds) != 2:
        return position, False

    target = target_size_for_asset(asset)
    footprint_x = max(float(target[0]), float(target[1])) / 2.0 + 0.035
    footprint_y = max(float(target[0]), float(target[1])) / 2.0 + 0.035
    min_x = float(x_bounds[0]) + footprint_x
    max_x = float(x_bounds[1]) - footprint_x
    min_y = float(y_bounds[0]) + footprint_y
    max_y = float(y_bounds[1]) - footprint_y
    if min_x > max_x:
        min_x = max_x = (float(x_bounds[0]) + float(x_bounds[1])) / 2.0
    if min_y > max_y:
        min_y = max_y = (float(y_bounds[0]) + float(y_bounds[1])) / 2.0

    clamped = list(position)
    original_x = float(clamped[0])
    original_y = float(clamped[1])
    clamped[0] = min(max(original_x, min_x), max_x)
    clamped[1] = min(max(original_y, min_y), max_y)
    return clamped, abs(clamped[0] - original_x) > 1e-6 or abs(clamped[1] - original_y) > 1e-6


def place_asset(
    asset: dict[str, Any],
    support_top_z: float,
    prefix: str,
    support_objects: list[Any] | None = None,
    default_contact_epsilon: float = 0.0,
    support_safe_bounds: dict[str, list[float]] | None = None,
    support_contact_surface_mode: str = "support_mesh_raycast",
) -> dict[str, Any]:
    import mathutils  # pylint: disable=import-outside-toplevel

    is_procedural = asset.get("asset_source") == "procedural_rigid_primitive_v1"
    meshes = create_procedural_foreground(asset, prefix) if is_procedural else import_glb(asset)
    for obj in meshes:
        obj.name = f"{prefix}_{asset.get('asset_id', asset.get('slot', 'asset'))}_{obj.name}"

    mins, maxs = bbox_for_objects(meshes)
    source_size = maxs - mins
    bottom_center = mathutils.Vector(((mins.x + maxs.x) / 2.0, (mins.y + maxs.y) / 2.0, mins.z))
    auto_yaw = auto_yaw_for_asset(asset, source_size)
    scale = scale_for_asset(asset, rotated_size_for_yaw(source_size, auto_yaw))
    placement = asset.get("placement", {})
    position = placement.get("position", [0.0, 0.0, 0.0])
    anchor = placement.get("anchor", "origin")
    xy_clamped = False
    if anchor == "support_top":
        position, xy_clamped = clamp_xy_to_safe_surface(list(position), asset, support_safe_bounds)
    z = float(position[2] if len(position) > 2 else 0.0)
    contact_epsilon = float(placement.get("contact_epsilon_m", default_contact_epsilon))
    if anchor == "support_top":
        z = support_top_z
    elif anchor == "behind_support":
        z = 0.0
        contact_epsilon = 0.0
    rotation = placement.get("rotation_degrees", [0.0, 0.0, 0.0])
    if not isinstance(rotation, list) or len(rotation) != 3:
        rotation = [0.0, 0.0, 0.0]
    visual_alignment = placement.get("visual_alignment_degrees", [0.0, 0.0, 0.0])
    if not isinstance(visual_alignment, list) or len(visual_alignment) != 3:
        raise ValueError(f"Metadata field `{prefix}.placement.visual_alignment_degrees` must have three values")
    pitch = math.radians(float(rotation[0]))
    roll = math.radians(float(rotation[1]))
    yaw = math.radians(float(placement.get("yaw_degrees", 0.0))) + math.radians(float(rotation[2]))
    visual_pitch = math.radians(float(visual_alignment[0]))
    visual_roll = math.radians(float(visual_alignment[1]))
    visual_yaw = math.radians(float(visual_alignment[2])) + auto_yaw
    target_position = mathutils.Vector((float(position[0]), float(position[1]), z))

    transform = (
        mathutils.Matrix.Translation(target_position)
        @ mathutils.Matrix.Rotation(yaw, 4, "Z")
        @ mathutils.Matrix.Rotation(roll, 4, "Y")
        @ mathutils.Matrix.Rotation(pitch, 4, "X")
        @ mathutils.Matrix.Rotation(visual_yaw, 4, "Z")
        @ mathutils.Matrix.Rotation(visual_roll, 4, "Y")
        @ mathutils.Matrix.Rotation(visual_pitch, 4, "X")
        @ mathutils.Matrix.Scale(scale, 4)
        @ mathutils.Matrix.Translation(-bottom_center)
    )
    for obj in meshes:
        obj.matrix_world = transform @ obj.matrix_world

    # Pitch/roll rotations can change the bottom point. Settle tabletop/floor
    # objects back onto their intended support plane after applying rotations.
    if anchor in {"support_top", "origin"}:
        settled_min, _ = bbox_for_objects(meshes)
        settle = z - float(settled_min.z)
        if abs(settle) > 1e-6:
            settle_transform = mathutils.Matrix.Translation((0.0, 0.0, settle))
            for obj in meshes:
                obj.matrix_world = settle_transform @ obj.matrix_world

    contact_z = z
    contact_sample_count = 0
    contact_source = "placement_z"
    if anchor == "support_top" and support_objects:
        contact_min, contact_max = bbox_for_objects(meshes)
        contact_z, contact_sample_count, contact_source = calculated_contact_height(
            support_objects,
            contact_min,
            contact_max,
            support_top_z,
            support_contact_surface_mode,
        )
        target_min_z = contact_z + contact_epsilon
        contact_shift = target_min_z - float(contact_min.z)
        if abs(contact_shift) > 1e-6:
            contact_transform = mathutils.Matrix.Translation((0.0, 0.0, contact_shift))
            for obj in meshes:
                obj.matrix_world = contact_transform @ obj.matrix_world

    has_texture = object_has_image_texture(meshes)
    if not has_texture and not is_procedural:
        raise ValueError(
            f"Non-procedural asset `{asset.get('asset_id', prefix)}` has no image texture. "
            "Fix the asset metadata or curation instead of using a renderer fallback."
        )

    final_min, final_max = bbox_for_objects(meshes)
    return {
        "asset_id": asset.get("asset_id"),
        "name": asset.get("name"),
        "role": asset.get("composition_role", asset.get("role")),
        "slot": asset.get("slot"),
        "mesh_count": len(meshes),
        "has_image_texture": has_texture,
        "source_bbox_size": vector_list(source_size),
        "target_size": [round(v, 6) for v in target_size_for_asset(asset)],
        "scale": round(float(scale), 8),
        "auto_yaw_degrees": round(math.degrees(auto_yaw), 4),
        "rotation_degrees": [round(float(v), 4) for v in rotation],
        "visual_alignment_degrees": [round(float(v), 4) for v in visual_alignment],
        "contact_support_z": round(float(contact_z), 6),
        "contact_epsilon_m": round(float(contact_epsilon), 6),
        "contact_sample_count": contact_sample_count,
        "contact_source": contact_source,
        "safe_surface_bounds": support_safe_bounds,
        "xy_clamped_to_safe_surface": xy_clamped,
        "final_bbox_min": vector_list(final_min),
        "final_bbox_max": vector_list(final_max),
        "objects": meshes,
    }


def add_camera(focus_meshes: list[Any], support_top_z: float, camera_config: dict[str, Any]) -> dict[str, Any]:
    import bpy  # pylint: disable=import-outside-toplevel

    mins, maxs = bbox_for_objects(focus_meshes)
    center = (mins + maxs) / 2.0
    span = maxs - mins
    distance_rule = camera_config.get("distance_rule", {})
    height_rule = camera_config.get("height_rule", {})
    target_offset = camera_config.get("target_offset_m", [0.0, -0.04, 0.0])
    if not isinstance(target_offset, list) or len(target_offset) < 3:
        target_offset = [0.0, -0.04, 0.0]

    focal_length_mm = float(camera_config.get("focal_length_mm", 45.0))
    span_x = max(float(span.x), float(distance_rule.get("min_focus_span_x_m", 0.8)))
    span_y = max(float(span.y), float(distance_rule.get("min_focus_span_y_m", 0.6)))
    span_xy = max(span_x, span_y)
    target_z = max(float(support_top_z) + 0.22, min(float(center.z), float(support_top_z) + 0.55))
    target = (
        float(center.x) + float(target_offset[0]),
        float(center.y) + float(target_offset[1]),
        target_z + float(target_offset[2]),
    )
    min_distance = float(distance_rule.get("min_distance_m", 2.15))
    max_distance = float(distance_rule.get("max_distance_m", 4.6))
    base_distance = float(distance_rule.get("base_distance_m", 1.25))
    span_scale = float(distance_rule.get("span_scale", 1.05))
    distance = max(min_distance, min(max_distance, base_distance + span_xy * span_scale))
    view_rule = camera_config.get("view_rule", {})
    azimuth_degrees = float(view_rule.get("azimuth_degrees", -55.2))
    azimuth = math.radians(azimuth_degrees)
    min_height = float(height_rule.get("min_height_above_target_m", 0.78))
    height_scale = float(height_rule.get("distance_height_scale", 0.34))
    position = (
        target[0] + distance * math.cos(azimuth),
        target[1] + distance * math.sin(azimuth),
        target[2] + max(min_height, distance * height_scale),
    )
    bpy.ops.object.camera_add(location=position)
    camera = bpy.context.object
    camera.name = "main_composition_camera"
    camera.data.lens = focal_length_mm
    camera.data.sensor_width = 32.0
    look_at(camera, target)
    bpy.context.scene.camera = camera
    return {
        "position": [round(v, 4) for v in position],
        "target": [round(v, 4) for v in target],
        "distance": round(distance, 4),
        "bbox_min": vector_list(mins),
        "bbox_max": vector_list(maxs),
        "rule": {
            "min_distance_m": round(min_distance, 4),
            "base_distance_m": round(base_distance, 4),
            "span_scale": round(span_scale, 4),
            "azimuth_degrees": round(azimuth_degrees, 4),
            "min_height_above_target_m": round(min_height, 4),
            "distance_height_scale": round(height_scale, 4),
        },
    }


def collect_assets(metadata: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    background = metadata["background"]
    support = background["support_asset"]
    static_visual = list(background.get("context_assets", [])) + list(background.get("static_props", []))
    foreground = list(metadata.get("foreground", {}).get("objects", []))
    return support, static_visual, foreground


def intended_support_top_z(support: dict[str, Any], support_record: dict[str, Any]) -> float:
    placement = support.get("placement", {})
    mode = placement.get("support_top_mode", "target_height")
    if mode == "explicit" and "support_top_z" in placement:
        return float(placement["support_top_z"])
    if mode == "bbox_max":
        return float(support_record["final_bbox_max"][2])
    target_size = placement.get("target_size")
    if isinstance(target_size, list) and len(target_size) >= 3:
        return float(target_size[2])
    return float(support_record["final_bbox_max"][2])


def render_metadata(
    metadata_path: Path,
    output_dir: Path | None,
    resolution: tuple[int, int],
    samples: int,
    root: Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    import bpy  # pylint: disable=import-outside-toplevel

    metadata = resolve_project_asset_paths(load_json(metadata_path), root)
    lighting_rule = merged_shadow_lighting_rule(metadata)
    setup_scene(resolution, samples, lighting_rule)
    support, static_visual, foreground = collect_assets(metadata)

    records: list[dict[str, Any]] = []
    if support.get("asset_source") == "procedural_poly_haven_support_v1":
        support_record = create_procedural_support(support, metadata)
    else:
        support_record = place_asset(support, 0.0, "support")
    support_top_z = intended_support_top_z(support, support_record)
    backdrop_record = add_procedural_backdrop(
        metadata, support_top_z, lighting_rule=lighting_rule
    )
    apply_environment_lighting(metadata, lighting_rule)
    records.append({k: v for k, v in support_record.items() if k != "objects"})

    render_config = metadata.get("render", {})
    default_contact_epsilon = float(render_config.get("contact_epsilon_m", render_config.get("contact_clearance_m", 0.0015)))
    support_safe_bounds = support.get("placement", {}).get("safe_surface_bounds")
    support_contact_surface_mode = support.get("placement", {}).get("contact_surface_mode", "support_mesh_raycast")
    context_meshes: list[Any] = []
    prop_meshes: list[Any] = []
    for index, asset in enumerate(static_visual, start=1):
        record = place_asset(
            asset,
            support_top_z,
            f"static_{index:02d}",
            support_record["objects"],
            default_contact_epsilon,
            support_safe_bounds,
            support_contact_surface_mode,
        )
        if str(asset.get("composition_role", "")).startswith("background"):
            context_meshes.extend(record["objects"])
        else:
            prop_meshes.extend(record["objects"])
        records.append({k: v for k, v in record.items() if k != "objects"})

    foreground_meshes: list[Any] = []
    for index, asset in enumerate(foreground, start=1):
        record = place_asset(
            asset,
            support_top_z,
            f"fg_{index:02d}",
            support_record["objects"],
            default_contact_epsilon,
            support_safe_bounds,
            support_contact_surface_mode,
        )
        foreground_meshes.extend(record["objects"])
        records.append({k: v for k, v in record.items() if k != "objects"})

    focus_meshes = prop_meshes + foreground_meshes
    if not focus_meshes:
        focus_meshes = support_record["objects"]
    camera_record = add_camera(focus_meshes, support_top_z, metadata.get("camera", {}))

    dataset_image = Path(metadata["outputs"]["first_frame"])
    dataset_image.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.filepath = str(dataset_image)
    bpy.ops.render.render(write_still=True)

    preview_image = dataset_image
    if output_dir is not None:
        preview_dir = output_dir / "first_frames"
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_image = preview_dir / f"{metadata['scene_id']}.png"
        shutil.copy2(dataset_image, preview_image)

    scene_manifest = {
        "scene_id": metadata["scene_id"],
        "metadata_path": str(metadata_path),
        "dataset_image": str(dataset_image),
        "preview_image": str(preview_image),
        "support_top_z": round(float(support_top_z), 6),
        "shadow_lighting_rule": lighting_rule,
        "backdrop": backdrop_record,
        "camera": camera_record,
        "assets": records,
    }
    write_json(Path(metadata["outputs"]["render_manifest"]), scene_manifest)
    return scene_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Render first frames for background composition metadata.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resolution", default="1280x720")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(blender_argv())

    width, height = [int(v) for v in args.resolution.lower().split("x", 1)]
    manifest = load_json(args.manifest)
    output_dir = args.output_dir.resolve() if args.output_dir else None
    records = []
    for index, sample in enumerate(manifest["samples"], start=1):
        if args.limit and index > args.limit:
            break
        print("render", sample["scene_id"])
        records.append(render_metadata(Path(sample["metadata_path"]), output_dir, (width, height), args.samples, args.root))

    render_manifest = {
        "render_id": "sketchfab_background_compositions_first_frames_v0",
        "metadata_manifest": str(args.manifest),
        "output_dir": str(output_dir) if output_dir else None,
        "resolution": args.resolution,
        "samples_per_pixel": args.samples,
        "records": [{k: v for k, v in record.items() if k != "assets"} for record in records],
    }
    if output_dir:
        write_json(output_dir / "render_manifest.json", render_manifest)
        print("manifest:", output_dir / "render_manifest.json")
    else:
        print(json.dumps(render_manifest, indent=2))


if __name__ == "__main__":
    main()
