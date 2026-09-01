"""Physical-scene admission and deterministic environment binding for 2obj."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from tools.assets.environment_collision import (
    compile_environment_binding,
    validate_environment_binding,
)
from tools.core.camera_geometry import pair_view_azimuth_degrees
from tools.core.json_io import read_json
from tools.core.rigid_geometry import finite_vector


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TWO_OBJECT_SCENE_RULES = (
    PROJECT_ROOT / "configs" / "two_object_scene_rules.json"
)
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "host_eligibility",
    "physical_rules",
    "visual_environment_coverage",
}
_HOST_FIELDS = {
    "required_collider_roles",
    "allowed_collider_roles",
    "allowed_visual_types",
    "camera_envelope_policy",
}
_REQUIRED_RULE_FIELDS = {
    "id",
    "scene_class",
    "support_shape",
    "maximum_abs_slope_degrees",
    "allowed_structure_families",
    "allowed_kinematic_regimes",
    "allowed_interaction_classes",
    "allowed_camera_view_families",
    "camera_view_family_overrides",
}
_OPTIONAL_RULE_FIELDS = {
    "object_camera_plane_readability_by_view_family",
    "pair_camera_geometry_view_families",
}
_ENVIRONMENT_FIELDS = {"metadata_key", "categories", "selection_policy"}
_MOTION_NEUTRAL_HOST_ROLES = {
    "primary_support",
    "support_structure",
    "environment_floor",
}
_KINEMATIC_REGIMES = {
    "supported_supported",
    "airborne_supported",
    "airborne_airborne",
}
_INTERACTION_CLASSES = {"interacting", "independent"}
_SCENE_SUPPORT_SHAPES = {
    "ground_flat": "rectangular_slab",
    "raised_flat": "rectangular_slab",
    "ground_feature": "inclined_ramp",
}


def load_two_object_scene_rules(
    path: Path = DEFAULT_TWO_OBJECT_SCENE_RULES,
) -> dict[str, Any]:
    """Load and validate the separately versioned 2obj scene rules."""

    rules = read_json(path)
    validate_two_object_scene_rules(rules)
    return rules


def resolved_two_object_scene_rules(
    rules: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return either the validated explicit contract or the validated default."""

    if rules is None:
        return load_two_object_scene_rules()
    validate_two_object_scene_rules(rules)
    return rules


def validate_two_object_scene_rules(contract: dict[str, Any]) -> None:
    """Reject ambiguous, overlapping, or motion-active scene declarations."""

    if (
        not isinstance(contract, dict)
        or set(contract) != _TOP_LEVEL_FIELDS
        or contract.get("schema_version") != "physweep_two_object_scene_rules_v2"
    ):
        raise ValueError("unsupported two-object scene-rules contract")
    host = contract.get("host_eligibility")
    if not isinstance(host, dict) or set(host) != _HOST_FIELDS:
        raise ValueError("two-object host eligibility is incomplete")
    required_roles = host.get("required_collider_roles")
    allowed_roles = host.get("allowed_collider_roles")
    visual_types = host.get("allowed_visual_types")
    if (
        required_roles != ["primary_support"]
        or not isinstance(allowed_roles, list)
        or set(allowed_roles) != _MOTION_NEUTRAL_HOST_ROLES
        or len(allowed_roles) != len(set(allowed_roles))
        or visual_types != ["procedural_room"]
        or host.get("camera_envelope_policy") != "unbounded_only"
    ):
        raise ValueError("two-object host eligibility must remain motion-neutral")

    rules = contract.get("physical_rules")
    if (
        not isinstance(rules, list)
        or len(rules) < 2
        or any(
            not isinstance(rule, dict)
            or not _REQUIRED_RULE_FIELDS.issubset(rule)
            or not set(rule).issubset(
                _REQUIRED_RULE_FIELDS | _OPTIONAL_RULE_FIELDS
            )
            for rule in rules
        )
    ):
        raise ValueError("two-object physical scene rules are incomplete")
    ids = [rule["id"] for rule in rules]
    if (
        any(not isinstance(value, str) or not value for value in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("two-object physical scene rule ids must be unique")
    admitted_families: set[tuple[str, str, str]] = set()
    admitted_regimes: set[str] = set()
    admitted_interaction_classes: set[str] = set()
    for rule in rules:
        scene_class = rule["scene_class"]
        maximum_slope = rule["maximum_abs_slope_degrees"]
        families = rule["allowed_structure_families"]
        regimes = rule["allowed_kinematic_regimes"]
        interaction_classes = rule["allowed_interaction_classes"]
        camera_view_families = rule["allowed_camera_view_families"]
        camera_overrides = rule["camera_view_family_overrides"]
        camera_plane_readability = rule.get(
            "object_camera_plane_readability_by_view_family"
        )
        pair_camera_families = rule.get(
            "pair_camera_geometry_view_families"
        )
        if (
            not isinstance(scene_class, str)
            or scene_class not in _SCENE_SUPPORT_SHAPES
            or rule["support_shape"] != _SCENE_SUPPORT_SHAPES[scene_class]
            or isinstance(maximum_slope, bool)
            or not isinstance(maximum_slope, (int, float))
            or not math.isfinite(float(maximum_slope))
            or (
                rule["support_shape"] == "rectangular_slab"
                and float(maximum_slope) != 0.0
            )
            or (
                rule["support_shape"] == "inclined_ramp"
                and not 5.0 <= float(maximum_slope) <= 30.0
            )
            or not isinstance(families, list)
            or not families
            or len(families) != len(set(families))
            or any(not isinstance(value, str) or not value for value in families)
            or not isinstance(regimes, list)
            or not regimes
            or not set(regimes).issubset(_KINEMATIC_REGIMES)
            or len(regimes) != len(set(regimes))
            or not isinstance(interaction_classes, list)
            or not interaction_classes
            or not set(interaction_classes).issubset(_INTERACTION_CLASSES)
            or len(interaction_classes) != len(set(interaction_classes))
            or not isinstance(camera_view_families, list)
            or not camera_view_families
            or any(
                not isinstance(family, str) or not family
                for family in camera_view_families
            )
            or len(camera_view_families) != len(set(camera_view_families))
            or not isinstance(camera_overrides, dict)
            or any(
                not isinstance(motion_id, str)
                or not motion_id
                or not isinstance(allowed, list)
                or not allowed
                or len(allowed) != len(set(allowed))
                or not set(allowed).issubset(camera_view_families)
                for motion_id, allowed in camera_overrides.items()
            )
            or (
                camera_overrides
                and {
                    family
                    for allowed in camera_overrides.values()
                    for family in allowed
                }
                != set(camera_view_families)
            )
            or (
                rule["support_shape"] == "inclined_ramp"
                and (
                    not isinstance(camera_plane_readability, dict)
                    or not camera_plane_readability
                    or set(camera_plane_readability)
                    != set(camera_view_families)
                    or any(
                        not isinstance(family, str)
                        or not family
                        or not isinstance(readability, dict)
                        or set(readability)
                        != {"geometry_size_axes", "minimum_extent_m"}
                        or not isinstance(
                            readability.get("geometry_size_axes"), list
                        )
                        or len(readability.get("geometry_size_axes", [])) != 2
                        or len(set(readability["geometry_size_axes"])) != 2
                        or any(
                            isinstance(axis, bool)
                            or not isinstance(axis, int)
                            or axis not in {0, 1, 2}
                            for axis in readability.get(
                                "geometry_size_axes", []
                            )
                        )
                        or isinstance(readability.get("minimum_extent_m"), bool)
                        or not isinstance(
                            readability.get("minimum_extent_m"), (int, float)
                        )
                        or not math.isfinite(
                            float(readability.get("minimum_extent_m", math.nan))
                        )
                        or float(readability.get("minimum_extent_m", 0.0))
                        <= 0.0
                        for family, readability in camera_plane_readability.items()
                    )
                    or not isinstance(pair_camera_families, list)
                    or not pair_camera_families
                    or len(pair_camera_families)
                    != len(set(pair_camera_families))
                    or set(pair_camera_families)
                    != set(camera_view_families)
                )
            )
            or (
                rule["support_shape"] != "inclined_ramp"
                and (
                    camera_plane_readability is not None
                    or pair_camera_families is not None
                )
            )
        ):
            raise ValueError("two-object physical scene rule is invalid")
        for family in families:
            key = (scene_class, str(rule["support_shape"]), str(family))
            if key in admitted_families:
                raise ValueError("two-object physical scene rules overlap")
            admitted_families.add(key)
        admitted_regimes.update(map(str, regimes))
        admitted_interaction_classes.update(map(str, interaction_classes))
    if admitted_regimes != _KINEMATIC_REGIMES:
        raise ValueError("two-object physical scene rules leave a regime unreachable")
    if admitted_interaction_classes != _INTERACTION_CLASSES:
        raise ValueError(
            "two-object physical scene rules leave an interaction class unreachable"
        )

    environment = contract.get("visual_environment_coverage")
    if not isinstance(environment, dict) or set(environment) != _ENVIRONMENT_FIELDS:
        raise ValueError("two-object visual-environment coverage is incomplete")
    categories = environment.get("categories")
    rule_ids = set(ids)
    if (
        environment.get("metadata_key")
        != "appearance.scene_visual.environment_category"
        or environment.get("selection_policy")
        != "balanced_feasible_within_scene_camera_and_source_pair"
        or not isinstance(categories, list)
        or len(categories) < 2
        or any(
            not isinstance(record, dict)
            or set(record) != {"id", "allowed_scene_rules"}
            or not isinstance(record["id"], str)
            or not record["id"]
            or not isinstance(record["allowed_scene_rules"], list)
            or not record["allowed_scene_rules"]
            or len(record["allowed_scene_rules"])
            != len(set(record["allowed_scene_rules"]))
            or not set(record["allowed_scene_rules"]).issubset(rule_ids)
            for record in categories
        )
        or len({record["id"] for record in categories}) != len(categories)
        or any(
            not any(rule_id in record["allowed_scene_rules"] for record in categories)
            for rule_id in rule_ids
        )
    ):
        raise ValueError("two-object visual-environment categories are invalid")


def allowed_scene_classes(contract: dict[str, Any]) -> tuple[str, ...]:
    """Return physical scene classes in stable first-declaration order."""

    validate_two_object_scene_rules(contract)
    return tuple(
        dict.fromkeys(str(rule["scene_class"]) for rule in contract["physical_rules"])
    )


def allowed_camera_view_families(
    rule: dict[str, Any], motion_pattern: str | None = None
) -> tuple[str, ...]:
    """Return the declared camera pool for one rule and motion pattern."""

    allowed = tuple(map(str, rule["allowed_camera_view_families"]))
    overrides = rule["camera_view_family_overrides"]
    if motion_pattern is None or not overrides:
        return allowed
    return tuple(map(str, overrides.get(str(motion_pattern), ())))


def resolve_scene_rule(
    contract: dict[str, Any],
    support: dict[str, Any],
    kinematic_regime: str | None = None,
    interaction_class: str | None = None,
    camera_view_family_id: str | None = None,
    motion_pattern: str | None = None,
) -> dict[str, Any] | None:
    """Resolve one unambiguous physical rule for an existing support host."""

    validate_two_object_scene_rules(contract)
    scene_class = str(support.get("scene_class", ""))
    support_shape = str(support.get("support_shape", ""))
    structure_family = str(support.get("structure_family", ""))
    surface_frame = support.get("surface_frame")
    if (
        not isinstance(surface_frame, dict)
        or "slope_angle_degrees" not in surface_frame
    ):
        return None
    raw_slope = surface_frame["slope_angle_degrees"]
    if (
        isinstance(raw_slope, bool)
        or not isinstance(raw_slope, (int, float))
        or not math.isfinite(float(raw_slope))
    ):
        return None
    slope = float(raw_slope)
    matches = [
        rule
        for rule in contract["physical_rules"]
        if str(rule["scene_class"]) == scene_class
        and str(rule["support_shape"]) == support_shape
        and structure_family in set(rule["allowed_structure_families"])
        and abs(slope) <= float(rule["maximum_abs_slope_degrees"]) + 1.0e-12
        and (
            kinematic_regime is None
            or kinematic_regime in set(rule["allowed_kinematic_regimes"])
        )
        and (
            interaction_class is None
            or interaction_class in set(rule["allowed_interaction_classes"])
        )
        and (
            motion_pattern is None
            or bool(allowed_camera_view_families(rule, motion_pattern))
        )
        and (
            camera_view_family_id is None
            or camera_view_family_id
            in set(allowed_camera_view_families(rule, motion_pattern))
        )
    ]
    if len(matches) > 1:
        raise ValueError("two-object support resolves to overlapping scene rules")
    return matches[0] if matches else None


def bind_two_object_scene(
    metadata: dict[str, Any],
    contract: dict[str, Any],
    expected_rule_id: str | None = None,
) -> dict[str, Any]:
    """Admit one host and orient it to the declared pair-relative view."""

    validate_two_object_scene_rules(contract)
    scene = copy.deepcopy(metadata)
    support = scene.get("simulation", {}).get("support")
    interaction = scene.get("simulation", {}).get("interaction")
    if not isinstance(support, dict) or not isinstance(interaction, dict):
        raise ValueError("two-object scene lacks support or interaction contracts")
    rule = resolve_scene_rule(
        contract,
        support,
        str(interaction.get("kinematic_regime", "")),
        str(interaction.get("interaction_class", "")),
        str(interaction.get("camera_view_family_id", "")),
        str(interaction.get("motion_pattern", "")),
    )
    if rule is None:
        raise ValueError("two-object host has no compatible physical scene rule")
    rule_id = str(rule["id"])
    if expected_rule_id is not None and rule_id != str(expected_rule_id):
        raise ValueError("two-object host changed its selected physical scene rule")
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
    preferred_azimuth = pair_view_azimuth_degrees(
        interaction["approach_axis_xyz"],
        interaction["camera_relative_azimuth_degrees"],
    )
    scene_visual = scene.get("appearance", {}).get("scene_visual")
    if not isinstance(scene_visual, dict):
        raise ValueError("two-object host lacks a scene visual")
    scene_visual.pop("camera_context", None)
    composition = scene_visual.get("composition")
    if isinstance(composition, dict):
        composition.pop("camera", None)
    scene["environment_binding"] = compile_environment_binding(
        scene, [], azimuth_override_degrees=preferred_azimuth
    )
    validate_environment_binding(scene)
    interaction["scene_compatibility"] = {
        "schema_version": "physweep_two_object_scene_compatibility_v2",
        "scene_rule_id": rule_id,
        "scene_class": str(rule["scene_class"]),
        "environment_binding_policy": "recompiled_for_declared_pair_view",
    }
    return scene
