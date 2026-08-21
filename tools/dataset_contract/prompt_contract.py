"""Deterministic natural-language prompts for PhysSweep video training."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


PROMPT_TEMPLATE_VERSION = "physweep_initial_event_prompt_v2"

SUPPORT_LABELS = {
    "lab_bench": "laboratory bench",
    "kitchen_counter": "kitchen counter",
    "wood_tabletop": "wooden tabletop",
    "concrete_floor_mat": "concrete floor",
    "wood_floor": "wooden floor",
    "rubber_floor": "rubber floor",
    "ground_ramp_short_steep": "short steep ramp",
    "ground_ramp_long_shallow": "long shallow ramp",
    "ground_channel_ramp": "channel ramp",
    "ground_ramp_standard": "ramp",
    "raised_channel_ramp": "raised channel ramp",
    "raised_ramp_short_steep": "short steep raised ramp",
    "raised_ramp_long_shallow": "long shallow raised ramp",
    "raised_ramp_standard": "raised ramp",
    "wood_tray": "wooden tray",
    "low_pedestal": "low pedestal",
}

SUPPORTED_MOTION_FAMILIES = {
    "drop_fall_1obj",
    "edge_fall_1obj",
    "projectile_1obj",
    "arc_projectile_1obj",
    "slide_push_1obj",
    "roll_or_slide_1obj",
    "slope_slide_down_1obj",
    "slope_slide_up_1obj",
    "wall_impact_1obj",
    "ramp_to_flat_1obj",
    "bounce_1obj",
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _humanize(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _object_label(metadata: Mapping[str, Any], object_id: str) -> str:
    dimensions = _mapping(
        _mapping(metadata.get("semantic_sampling")).get("five_dimensions")
    )
    foreground = _mapping(dimensions.get("foreground_object"))
    raw_label = foreground.get("object_type")
    if not raw_label:
        simulation = _mapping(metadata.get("simulation"))
        for record in simulation.get("objects", []):
            if not isinstance(record, Mapping):
                continue
            if str(record.get("object_id")) == object_id:
                raw_label = record.get("semantic_type")
                break
    if not raw_label:
        identity = _mapping(metadata.get("object_identity"))
        for record in identity.get("objects", []):
            if not isinstance(record, Mapping):
                continue
            if str(record.get("object_id")) == object_id:
                raw_label = record.get("semantic_label")
                break

    label = str(raw_label or "object").strip().lower()
    label = re.sub(r"^physassets[_ ]+\d+[_ ]+", "", label)
    label = re.sub(r"^physassets[_ ]+", "", label)
    label = _humanize(label) or "object"
    aliases = {
        "remote": "remote control",
        "drink box": "drink carton",
    }
    return aliases.get(label, label)


def _support_label(dimensions: Mapping[str, Any]) -> str:
    interaction = _mapping(dimensions.get("support_interaction"))
    support_type = str(interaction.get("support_type") or "support surface")
    return SUPPORT_LABELS.get(support_type, _humanize(support_type))


def _motion_description(
    family: str,
    support: str,
) -> tuple[str, str]:
    if family not in SUPPORTED_MOTION_FAMILIES:
        raise ValueError(
            f"motion family needs an explicit natural-language prompt: {family}"
        )
    if family == "drop_fall_1obj":
        return f"released above the {support} under gravity", support
    if family == "bounce_1obj":
        return f"released above the {support} under gravity", support
    if family == "projectile_1obj":
        return f"launched horizontally above the {support}", support
    if family == "arc_projectile_1obj":
        return f"launched upward and forward above the {support}", support
    if family == "edge_fall_1obj":
        return f"given an initial velocity across the {support} near its edge", support
    if family == "slide_push_1obj":
        return f"given a short initial push across the {support}", support
    if family == "roll_or_slide_1obj":
        return f"set in motion across the {support}", support
    if family == "slope_slide_down_1obj":
        return f"released on the {support} with a downhill initial state", support
    if family == "slope_slide_up_1obj":
        return f"launched uphill on the {support}", support
    if family == "wall_impact_1obj":
        return (
            f"given an initial velocity toward a fixed wall across the {support}",
            f"{support} and fixed wall",
        )
    if family == "ramp_to_flat_1obj":
        return (
            f"released on the {support} with a downhill initial state above flat ground",
            f"{support} and flat ground",
        )
    raise AssertionError(f"unhandled motion family: {family}")


def build_training_prompt(metadata: Mapping[str, Any], object_id: str) -> str:
    """Compile one base-scene prompt shared by every one-factor sweep level."""
    dimensions = _mapping(
        _mapping(metadata.get("semantic_sampling")).get("five_dimensions")
    )
    motion = _mapping(dimensions.get("motion"))
    family = str(motion.get("family") or "")
    label = _object_label(metadata, object_id)
    support = _support_label(dimensions)
    action, stable_context = _motion_description(family, support)
    prompt = (
        f"Static-camera video of exactly one {label} {action}. "
        f"The same single {label}, {stable_context}, lighting, and background "
        "remain consistent throughout."
    )
    forbidden = ("physassets", "1obj", "_")
    if any(token in prompt.lower() for token in forbidden):
        raise ValueError(f"compiled prompt contains an internal token: {prompt}")
    return prompt
