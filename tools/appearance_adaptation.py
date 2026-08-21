"""Deterministic material-lightness analysis and restrained lighting adaptation."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable


POLICY_VERSION = "physweep_material_lightness_adaptation_v1"
RENDERED_FRAME_POLICY_VERSION = "physweep_rendered_frame_exposure_adaptation_v1"
LIGHT_LUMINANCE = 0.42
DARK_LUMINANCE = 0.12
LIGHT_PIXEL_THRESHOLD = 0.50
MIN_MEAN_LUMA = 12.0
MAX_MEAN_LUMA = 245.0
MIN_LUMA_STD = 8.0
MIN_MEAN_GRADIENT = 0.7
MAX_CLIPPED_FRACTION = 0.35
PROBE_MIN_MEAN_LUMA = 13.0
PROBE_MAX_MEAN_LUMA = 242.0
PROBE_MIN_LUMA_STD = 8.4
TARGET_LOW_MEAN_LUMA = 16.0
TARGET_HIGH_MEAN_LUMA = 238.0
TARGET_LUMA_STD = 8.8
MAX_EXPOSURE_STEP_EV = 0.35
MAX_TOTAL_EXPOSURE_CORRECTION_EV = 0.70
_IMAGE_STATS_CACHE: dict[tuple[int, int], tuple[float, float]] = {}


def _luminance(red: float, green: float, blue: float) -> float:
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _sample_image(image: Any, grid_size: int = 24) -> tuple[float, float]:
    width, height = [int(value) for value in image.size]
    if width <= 0 or height <= 0:
        raise ValueError(f"image has no pixels: {image.name}")
    cache_key = (int(image.as_pointer()), grid_size)
    if cache_key in _IMAGE_STATS_CACHE:
        return _IMAGE_STATS_CACHE[cache_key]
    sample = image.copy()
    try:
        sample.scale(grid_size, grid_size)
        pixels = list(sample.pixels[:])
        luminances = []
        for row in range(grid_size):
            for column in range(grid_size):
                offset = 4 * (row * grid_size + column)
                luminances.append(
                    _luminance(
                        float(pixels[offset]),
                        float(pixels[offset + 1]),
                        float(pixels[offset + 2]),
                    )
                )
        mean = sum(luminances) / len(luminances)
        light_fraction = sum(
            value >= LIGHT_PIXEL_THRESHOLD for value in luminances
        ) / len(luminances)
    finally:
        import bpy

        bpy.data.images.remove(sample)
    result = (mean, light_fraction)
    _IMAGE_STATS_CACHE[cache_key] = result
    return result


def _upstream_images(socket: Any, visited: set[int] | None = None) -> list[Any]:
    visited = set() if visited is None else visited
    images = []
    for link in socket.links:
        node = link.from_node
        node_key = id(node)
        if node_key in visited:
            continue
        visited.add(node_key)
        if node.bl_idname == "ShaderNodeTexImage" and node.image is not None:
            images.append(node.image)
            continue
        for input_socket in node.inputs:
            if input_socket.is_linked:
                images.extend(_upstream_images(input_socket, visited))
    return images


def _base_color_images(material: Any) -> list[Any]:
    if material is None or not material.use_nodes or material.node_tree is None:
        return []
    images = []
    for node in material.node_tree.nodes:
        if node.bl_idname != "ShaderNodeBsdfPrincipled":
            continue
        images.extend(_upstream_images(node.inputs["Base Color"]))
    unique_images = []
    seen = set()
    for image in images:
        image_key = id(image)
        if image_key not in seen:
            seen.add(image_key)
            unique_images.append(image)
    return unique_images


def _material_lightness(material: Any) -> tuple[float, float, str]:
    images = _base_color_images(material)
    if images:
        samples = [_sample_image(image) for image in images]
        return (
            sum(value[0] for value in samples) / len(samples),
            sum(value[1] for value in samples) / len(samples),
            "base_color_texture_pixels",
        )
    if material is not None and material.use_nodes and material.node_tree is not None:
        for node in material.node_tree.nodes:
            if node.bl_idname == "ShaderNodeBsdfPrincipled":
                color = node.inputs["Base Color"].default_value
                luminance = _luminance(
                    float(color[0]), float(color[1]), float(color[2])
                )
                return luminance, float(luminance >= LIGHT_PIXEL_THRESHOLD), "base_color"
    return 0.18, 0.0, "neutral_fallback"


def analyze_objects(objects: Iterable[Any]) -> dict[str, Any]:
    material_weights: dict[Any, float] = defaultdict(float)
    for obj in objects:
        if obj is None or getattr(obj, "type", None) != "MESH":
            continue
        slots = list(obj.material_slots)
        if not slots:
            continue
        polygon_counts: dict[int, int] = defaultdict(int)
        for polygon in obj.data.polygons:
            polygon_counts[int(polygon.material_index)] += 1
        for index, slot in enumerate(slots):
            if slot.material is not None:
                material_weights[slot.material] += max(1, polygon_counts.get(index, 0))
    if not material_weights:
        return {
            "mean_linear_luminance": 0.18,
            "light_pixel_fraction": 0.0,
            "classification": "medium",
            "material_count": 0,
            "measurement": "neutral_fallback",
        }
    total_weight = sum(material_weights.values())
    weighted_luminance = 0.0
    weighted_light_fraction = 0.0
    measurements = set()
    for material, weight in material_weights.items():
        luminance, light_fraction, measurement = _material_lightness(material)
        weighted_luminance += weight * luminance
        weighted_light_fraction += weight * light_fraction
        measurements.add(measurement)
    luminance = weighted_luminance / total_weight
    light_fraction = weighted_light_fraction / total_weight
    if luminance >= LIGHT_LUMINANCE or light_fraction >= 0.40:
        classification = "light"
    elif luminance <= DARK_LUMINANCE:
        classification = "dark"
    else:
        classification = "medium"
    return {
        "mean_linear_luminance": round(luminance, 6),
        "light_pixel_fraction": round(light_fraction, 6),
        "classification": classification,
        "material_count": len(material_weights),
        "measurement": sorted(measurements),
    }


def choose_adaptation(
    dynamic: dict[str, Any], support: dict[str, Any]
) -> dict[str, Any]:
    dynamic_luminance = float(dynamic["mean_linear_luminance"])
    support_luminance = float(support["mean_linear_luminance"])
    contrast = abs(dynamic_luminance - support_luminance)
    dynamic_class = str(dynamic["classification"])
    support_class = str(support["classification"])
    if dynamic_class == "light" and support_class == "light":
        decision = ("both_light", -0.45, 0.72, 0.72)
    elif support_luminance >= 0.72:
        decision = ("very_light_support", -0.52, 0.60, 0.64)
    elif support_class == "light" and contrast < 0.20:
        decision = ("light_low_contrast", -0.32, 0.78, 0.78)
    elif support_class == "light" or dynamic_class == "light":
        decision = ("one_light", -0.18, 0.86, 0.84)
    elif dynamic_class == "dark" and support_class == "dark":
        decision = ("both_dark", 0.12, 1.05, 1.10)
    else:
        decision = ("balanced", 0.0, 1.0, 1.0)
    label, exposure_delta, world_scale, fill_scale = decision
    return {
        "policy_version": POLICY_VERSION,
        "decision": label,
        "dynamic": dynamic,
        "support_asset": support,
        "dynamic_support_luminance_contrast": round(contrast, 6),
        "exposure_delta_ev": exposure_delta,
        "world_strength_scale": world_scale,
        "fill_light_scale": fill_scale,
    }


def frame_statistics_within_fixed_limits(statistics: dict[str, Any]) -> bool:
    return (
        MIN_MEAN_LUMA <= float(statistics["mean_luma"]) <= MAX_MEAN_LUMA
        and float(statistics["luma_std"]) >= MIN_LUMA_STD
        and float(statistics["mean_gradient"]) >= MIN_MEAN_GRADIENT
        and float(statistics["clipped_dark_fraction"]) <= MAX_CLIPPED_FRACTION
        and float(statistics["clipped_light_fraction"]) <= MAX_CLIPPED_FRACTION
    )


def choose_rendered_frame_exposure_adjustment(
    frame_statistics: Iterable[dict[str, Any]],
    cumulative_correction_ev: float = 0.0,
) -> dict[str, Any]:
    statistics = list(frame_statistics)
    if not statistics:
        raise ValueError("at least one rendered-frame statistic is required")
    minimum_mean = min(float(record["mean_luma"]) for record in statistics)
    maximum_mean = max(float(record["mean_luma"]) for record in statistics)
    minimum_std = min(float(record["luma_std"]) for record in statistics)
    reasons = []
    desired_scale = 1.0
    if minimum_mean < PROBE_MIN_MEAN_LUMA:
        reasons.append("low_mean_luma")
        desired_scale = max(
            desired_scale,
            TARGET_LOW_MEAN_LUMA / max(minimum_mean, 1.0e-6),
        )
    if minimum_std < PROBE_MIN_LUMA_STD:
        reasons.append("low_luma_contrast")
        desired_scale = max(
            desired_scale,
            TARGET_LUMA_STD / max(minimum_std, 1.0e-6),
        )
    if maximum_mean > PROBE_MAX_MEAN_LUMA:
        reasons.append("high_mean_luma")
        desired_scale = min(desired_scale, TARGET_HIGH_MEAN_LUMA / maximum_mean)
    requested_delta = math.log2(desired_scale) if reasons else 0.0
    lower_total = -MAX_TOTAL_EXPOSURE_CORRECTION_EV
    upper_total = MAX_TOTAL_EXPOSURE_CORRECTION_EV
    bounded_delta = max(
        -MAX_EXPOSURE_STEP_EV,
        min(MAX_EXPOSURE_STEP_EV, requested_delta),
    )
    bounded_delta = max(
        lower_total - float(cumulative_correction_ev),
        min(upper_total - float(cumulative_correction_ev), bounded_delta),
    )
    if abs(bounded_delta) < 1.0e-4:
        bounded_delta = 0.0
    return {
        "policy_version": RENDERED_FRAME_POLICY_VERSION,
        "reasons": reasons,
        "minimum_mean_luma": round(minimum_mean, 6),
        "maximum_mean_luma": round(maximum_mean, 6),
        "minimum_luma_std": round(minimum_std, 6),
        "requested_delta_ev": round(requested_delta, 6),
        "applied_delta_ev": round(bounded_delta, 6),
        "cumulative_correction_ev": round(
            float(cumulative_correction_ev) + bounded_delta,
            6,
        ),
    }


def apply_material_lightness_adaptation(
    scene: Any,
    dynamic_objects: Iterable[Any],
    support_objects: Iterable[Any],
) -> dict[str, Any]:
    report = choose_adaptation(
        analyze_objects(dynamic_objects),
        analyze_objects(support_objects),
    )
    scene.view_settings.exposure += float(report["exposure_delta_ev"])
    if scene.world is not None and scene.world.use_nodes:
        for node in scene.world.node_tree.nodes:
            if node.bl_idname == "ShaderNodeBackground":
                node.inputs["Strength"].default_value *= float(
                    report["world_strength_scale"]
                )
    for light in scene.objects:
        if light.type != "LIGHT":
            continue
        name = str(light.name).lower()
        if "fill" in name or "rim" in name:
            light.data.energy *= float(report["fill_light_scale"])
    report["result_exposure_ev"] = round(float(scene.view_settings.exposure), 6)
    return report
