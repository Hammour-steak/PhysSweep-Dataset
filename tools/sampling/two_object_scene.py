"""Scene admission and deterministic environment binding for 2obj."""

from __future__ import annotations

import copy
from typing import Any

from tools.assets.environment_collision import (
    compile_environment_binding,
    validate_environment_binding,
)
from tools.core.camera_geometry import deterministic_pair_side_azimuths
from tools.core.rigid_geometry import finite_vector


_CONTRACT_FIELDS = {"schema_version", "allowed_scene_classes"}


def bind_two_object_scene(
    metadata: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    """Admit one flat host and orient its environment to the pair side view."""

    if set(contract) != _CONTRACT_FIELDS or contract.get("schema_version") != (
        "physweep_two_object_scene_compatibility_v1"
    ):
        raise ValueError("unsupported two-object scene-compatibility contract")
    allowed = contract.get("allowed_scene_classes")
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(not isinstance(value, str) or not value for value in allowed)
        or len(allowed) != len(set(allowed))
    ):
        raise ValueError("two-object allowed scene classes are invalid")
    scene = copy.deepcopy(metadata)
    support = scene.get("simulation", {}).get("support")
    if not isinstance(support, dict):
        raise ValueError("two-object scene lacks a support contract")
    scene_class = str(support.get("scene_class", ""))
    if scene_class not in set(allowed):
        raise ValueError(
            f"two-object scene class is not admitted: {scene_class}"
        )
    slope = float(support["surface_frame"]["slope_angle_degrees"])
    if abs(slope) > 1.0e-8:
        raise ValueError("two-object scene requires a flat support")
    bounds = support.get("safe_surface_bounds")
    if not isinstance(bounds, dict) or set(bounds) != {"x", "y"}:
        raise ValueError("two-object scene lacks safe surface bounds")
    x0, x1, y0, y1 = finite_vector(
        [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]],
        4,
        "two-object safe surface bounds",
    )
    if x1 <= x0 or y1 <= y0:
        raise ValueError("two-object safe surface bounds are empty")
    validate_environment_binding(scene)
    interaction = scene["simulation"].get("interaction")
    if not isinstance(interaction, dict):
        raise ValueError("two-object scene lacks an interaction contract")
    preferred_azimuth, _ = deterministic_pair_side_azimuths(
        str(scene["scene_id"]), interaction["approach_axis_xy"]
    )
    scene["environment_binding"] = compile_environment_binding(
        scene, [], azimuth_override_degrees=preferred_azimuth
    )
    validate_environment_binding(scene)
    interaction["scene_compatibility"] = {
        "schema_version": contract["schema_version"],
        "scene_class": scene_class,
        "environment_binding_policy": "recompiled_for_preferred_pair_side",
    }
    return scene
