"""Shared Blender scene, material and CLI helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from tools.core.blender_runtime import blender_argv, look_at


def parse_scene_render_args(
    description: str | None,
    *,
    project_root: Path | None = None,
) -> argparse.Namespace:
    """Parse the common metadata/video arguments used by Blender scene renderers."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--metadata", type=Path, required=True)
    if project_root is not None:
        parser.add_argument("--root", type=Path, default=project_root)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--inspection-frame-dir", type=Path)
    values = blender_argv() if "--" in sys.argv else []
    return parser.parse_args(values)


def add_bound_lights(metadata: dict[str, Any]) -> None:
    """Create metadata-bound lights aimed at the declared camera target."""

    import bpy  # pylint: disable=import-outside-toplevel

    target = metadata["camera"]["target_m"]
    for binding in metadata["render"]["lights"]:
        light_data = bpy.data.lights.new(str(binding["id"]), str(binding["type"]))
        light_data.energy = float(binding["energy_w"])
        light_data.size = float(binding["size_m"])
        light_data.color = tuple(float(value) for value in binding["color_rgb"])
        light = bpy.data.objects.new(str(binding["id"]), light_data)
        bpy.context.collection.objects.link(light)
        light.location = tuple(float(value) for value in binding["position_m"])
        look_at(light, target)


def require_keys(record: dict[str, Any], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in record]
    if missing:
        raise ValueError(f"Metadata field `{label}` is missing required keys: {missing}")


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
