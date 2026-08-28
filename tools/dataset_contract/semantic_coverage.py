"""Coverage summaries derived only from published semantic metadata."""

from __future__ import annotations

from collections import Counter
from typing import Any


def semantic_coverage_counts(
    scenes: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Summarize the five semantic sampling dimensions without sampler imports."""

    dimensions = [scene["semantic_sampling"]["five_dimensions"] for scene in scenes]
    fields = {
        "motion": ("motion", "family"),
        "object": ("foreground_object", "object_type"),
        "visual_type": ("foreground_object", "visual_type"),
        "scene_class": ("support_interaction", "scene_class"),
        "support": ("support_interaction", "support_type"),
        "support_geometry_variant": (
            "support_interaction",
            "geometry_variant_id",
        ),
        "scene_visual": ("support_interaction", "scene_visual_profile"),
        "scene_visual_type": ("support_interaction", "scene_visual_type"),
        "support_visual": ("support_interaction", "support_visual_profile"),
        "support_visual_type": ("support_interaction", "support_visual_type"),
        "camera": ("camera_observation", "camera_profile"),
        "surface_family": ("appearance_lighting", "surface_family"),
        "environment_category": (
            "appearance_lighting",
            "environment_category",
        ),
    }
    return {
        label: dict(Counter(item[group][field] for item in dimensions))
        for label, (group, field) in fields.items()
    }
