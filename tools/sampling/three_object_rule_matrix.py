"""Validate the compact production-facing three-object rule matrix."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


MATRIX_PATH = Path("configs/three_object_rule_matrix.json")
FAMILY_IDS = ("generic", "billiards", "pinball", "marble_run")
ROLES = ("P", "Q", "R")
SHAPES = ("sphere", "cuboid", "cylinder")
GENERIC_TEMPLATES = (
    "chain_transfer",
    "successive_hits",
    "converging_hits",
    "pair_control",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_reference(root: Path, reference: dict[str, Any]) -> Path:
    if set(reference) != {"path", "sha256"}:
        raise ValueError("rule reference must contain only path and sha256")
    path = (root / str(reference["path"])).resolve()
    if root != path and root not in path.parents:
        raise ValueError("rule reference escapes the project root")
    if not path.is_file() or _sha256(path) != reference["sha256"]:
        raise ValueError(f"changed or missing rule reference: {reference['path']}")
    return path


def _by_id(rows: Any, expected: tuple[str, ...], label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} must be a list of objects")
    ids = tuple(str(row.get("id", "")) for row in rows)
    if ids != expected or len(set(ids)) != len(ids):
        raise ValueError(f"{label} ids or ordering changed")
    return {row["id"]: row for row in rows}


def _validate_shape_roles(value: Any, *, allowed: dict[str, set[str]]) -> None:
    if not isinstance(value, dict) or tuple(value) != ROLES:
        raise ValueError("shape_by_role must explicitly cover P, Q and R")
    for role in ROLES:
        shapes = value[role]
        if not isinstance(shapes, list) or not shapes or len(shapes) != len(set(shapes)):
            raise ValueError("role shape choices must be distinct and nonempty")
        if set(shapes) != allowed[role]:
            raise ValueError(f"shape capability changed for role {role}")


def _validate_camera(camera: Any, expected_ids: tuple[str, str, str], max_elevation: float) -> None:
    if not isinstance(camera, dict) or camera.get("policy") != "base_only_fixed_group":
        raise ValueError("camera must be admitted on the base and frozen for the group")
    views = camera.get("view_families")
    if not isinstance(views, list) or len(views) != 3:
        raise ValueError("camera must expose exactly three requested view families")
    if tuple(view.get("id") for view in views) != expected_ids:
        raise ValueError("camera view family ids changed")
    azimuths = []
    for view in views:
        azimuth = view.get("azimuth_degrees")
        elevation = view.get("elevation_degrees")
        if (
            type(azimuth) not in (int, float)
            or type(elevation) not in (int, float)
            or not math.isfinite(float(azimuth))
            or not 0 < float(elevation) <= max_elevation
        ):
            raise ValueError("camera angle is outside the admitted family scope")
        azimuths.append(float(azimuth))
    if len(set(azimuths)) != 3:
        raise ValueError("camera view families need distinct azimuths")
    focal = camera.get("focal_length_mm")
    if not isinstance(focal, list) or len(focal) < 2 or any(float(value) <= 0 for value in focal):
        raise ValueError("camera focal choices are incomplete")


def _validate_asset_scope(root: Path, reference: dict[str, Any]) -> dict[str, Any]:
    path = _resolved_reference(root, reference)
    scope = _load(path)
    if scope.get("schema_version") != "physweep_three_object_asset_scope_v1":
        raise ValueError("unsupported three-object asset scope")
    generic = scope.get("generic_visuals", {})
    ids = generic.get("asset_ids")
    if not isinstance(ids, list) or len(ids) != 84 or len(set(ids)) != 84:
        raise ValueError("generic visual inventory must contain 84 distinct ids")
    if generic.get("count_by_proxy_shape") != {"cuboid": 35, "cylinder": 27, "sphere": 22}:
        raise ValueError("generic proxy-shape inventory changed")

    dynamic = scope.get("dynamic_assets", {})
    nonball = dynamic.get("true_nonball_role_safe")
    sphere = dynamic.get("sphere_assets")
    excluded = dynamic.get("excluded")
    if not all(isinstance(rows, list) for rows in (nonball, sphere, excluded)):
        raise ValueError("dynamic asset groups must be lists")
    if (len(nonball), len(sphere), len(excluded)) != (10, 1, 5):
        raise ValueError("dynamic asset inventory counts changed")
    all_asset_ids = [row.get("asset_id") for rows in (nonball, sphere, excluded) for row in rows]
    if "" in all_asset_ids or None in all_asset_ids or len(all_asset_ids) != len(set(all_asset_ids)):
        raise ValueError("dynamic asset ids are missing or overlap")
    if any(row.get("shape") not in {"cuboid", "cylinder"} for row in nonball):
        raise ValueError("role-safe nonball assets must use cuboid or cylinder envelopes")
    if any(row.get("proxy") not in {"cuboid", "cylinder", "compound"} for row in nonball):
        raise ValueError("role-safe nonball proxy type changed")
    if any(row.get("shape") != "sphere" or row.get("proxy") != "sphere" for row in sphere):
        raise ValueError("sphere asset scope changed")
    policy = dynamic.get("policy", {})
    if (
        policy.get("true_nonball_motion") != "flat pair_control only"
        or policy.get("true_nonball_layout") != "axis_aligned"
        or policy.get("allowed_roles") != list(ROLES)
        or policy.get("maximum_true_assets_per_scene") != 1
    ):
        raise ValueError("dynamic asset compatibility was widened")

    supports = scope.get("exact_support_assets", {})
    support_ids = supports.get("allowed_asset_ids")
    excluded_supports = supports.get("excluded")
    if (
        not isinstance(support_ids, list)
        or len(support_ids) != 18
        or len(set(support_ids)) != 18
        or not isinstance(excluded_supports, list)
        or len(excluded_supports) != 1
        or excluded_supports[0].get("asset_id") in set(support_ids)
    ):
        raise ValueError("exact support asset scope changed")
    return {
        "path": reference["path"],
        "sha256": reference["sha256"],
        "generic_visual_count": len(ids),
        "role_safe_nonball_count": len(nonball),
        "exact_support_count": len(support_ids),
    }


def _validate_generic(family: dict[str, Any]) -> dict[str, Any]:
    subcases = _by_id(
        family.get("subcases"), ("flat", "inclined", "exact_mesh_support"), "generic subcase"
    )
    flat = subcases["flat"]
    if flat.get("supports") != ["ground_flat", "raised_flat"]:
        raise ValueError("generic flat support scope changed")
    templates = _by_id(flat.get("motion_templates"), GENERIC_TEMPLATES, "generic template")
    sphere = {role: {"sphere"} for role in ROLES}
    mixed_r = {"P": {"sphere"}, "Q": {"sphere"}, "R": set(SHAPES)}
    multi = {role: set(SHAPES) for role in ROLES}

    expected_variants = {
        "chain_transfer": ("axis_aligned", "angled_chain_left"),
        "successive_hits": (
            "axis_aligned", "offset_successive_left", "offset_successive_right"
        ),
        "converging_hits": ("axis_aligned",),
        "pair_control": (
            "axis_aligned_spheres",
            "axis_aligned_mixed_R",
            "axis_aligned_multi_mesh",
            "crossing_control_left",
            "crossing_control_right",
        ),
    }
    for template_id in GENERIC_TEMPLATES:
        variants = _by_id(
            templates[template_id].get("variants"), expected_variants[template_id], f"{template_id} variant"
        )
        if template_id != "pair_control":
            for variant in variants.values():
                _validate_shape_roles(variant.get("shape_by_role"), allowed=sphere)

    pair = _by_id(
        templates["pair_control"]["variants"], expected_variants["pair_control"], "pair_control variant"
    )
    _validate_shape_roles(pair["axis_aligned_spheres"].get("shape_by_role"), allowed=sphere)
    _validate_shape_roles(pair["axis_aligned_mixed_R"].get("shape_by_role"), allowed=mixed_r)
    mesh = pair["axis_aligned_multi_mesh"]
    _validate_shape_roles(mesh.get("shape_by_role"), allowed=multi)
    if (
        mesh.get("maximum_sphere_count") != 1
        or mesh.get("true_asset_roles") != list(ROLES)
        or mesh.get("maximum_true_assets_per_scene") != 1
        or not str(mesh.get("admission", "")).startswith("base_physics_camera_media_passed")
    ):
        raise ValueError("multi-mesh pair capability changed")
    for variant_id in ("crossing_control_left", "crossing_control_right"):
        _validate_shape_roles(pair[variant_id].get("shape_by_role"), allowed=mixed_r)
    if flat.get("motion_axes") != ["world_x", "world_y"] or flat.get("target_roles") != list(ROLES):
        raise ValueError("generic flat role or axis coverage changed")
    _validate_camera(flat.get("camera"), ("front_oblique", "side_oblique", "rear_oblique"), 35)
    if flat["camera"].get("diversity_policy") != "balance all three selected frozen families inside every sizeable generic quota":
        raise ValueError("generic flat camera diversity must use selected frozen families")
    required_flat_diversity = {
        "motion_template", "layout_variant", "motion_axis", "shape_pattern",
        "target_role", "support_class", "camera_family", "environment_category",
        "visual_asset_id",
    }
    if set(flat.get("diversity_dimensions", [])) != required_flat_diversity:
        raise ValueError("generic flat diversity dimensions changed")

    inclined = subcases["inclined"]
    inclined_templates = _by_id(inclined.get("motion_templates"), ("pair_control",), "inclined template")
    inclined_variants = _by_id(
        inclined_templates["pair_control"].get("variants"), ("axis_aligned",), "inclined variant"
    )
    _validate_shape_roles(inclined_variants["axis_aligned"].get("shape_by_role"), allowed=sphere)
    if inclined.get("supports") != ["ground_ramp_long_shallow", "raised_ramp_long_shallow"]:
        raise ValueError("inclined support scope changed")
    if inclined.get("slope_degrees") != [8.0, 12.0] or inclined.get("target_roles") != list(ROLES):
        raise ValueError("inclined numeric or target-role scope changed")
    if inclined.get("material_policy") != {"contact_friction_minimum": 0.32, "rolling_friction": 0.018}:
        raise ValueError("inclined material policy changed")
    _validate_camera(inclined.get("camera"), ("front_oblique", "side_oblique", "rear_oblique"), 50)
    if inclined["camera"].get("diversity_policy") != "balance all three selected frozen families; do not inherit flat crossing layouts":
        raise ValueError("inclined camera diversity must use selected frozen families")

    support = subcases["exact_mesh_support"]
    _validate_shape_roles(support.get("shape_by_role"), allowed=sphere)
    if (
        support.get("supports") != "three_object_asset_scope.exact_support_assets"
        or support.get("motion_templates") != ["chain_transfer", "pair_control"]
        or support.get("motion_axes") != ["world_x", "world_y"]
        or support.get("target_roles") != list(ROLES)
    ):
        raise ValueError("exact mesh support compatibility changed")
    _validate_camera(support.get("camera"), ("front_oblique", "side_oblique", "rear_oblique"), 35)
    if support["camera"].get("diversity_policy") != "balance all three selected frozen families inside every sizeable generic quota":
        raise ValueError("exact support camera diversity must use selected frozen families")
    return {
        "subcase_count": len(subcases),
        "motion_template_count": len(templates),
        "flat_variant_count": sum(len(template["variants"]) for template in templates.values()),
    }


def _validate_special(family: dict[str, Any]) -> None:
    family_id = family["id"]
    expected = {
        "billiards": ("chain_transfer", "source_pool_table", None),
        "pinball": ("pair_control", "exact_analytic_passive_pinfield", 12),
        "marble_run": ("ordered_contacts", "three_segment_source_fixture_v1", 24),
    }[family_id]
    if family.get("motion_templates") != [expected[0]] or family.get("fixture") != expected[1]:
        raise ValueError(f"{family_id} motion or fixture scope changed")
    sphere = {role: {"sphere"} for role in ROLES}
    _validate_shape_roles(family.get("shape_by_role"), allowed=sphere)
    if family.get("target_roles") != list(ROLES):
        raise ValueError(f"{family_id} target-role scope changed")
    if expected[2] is not None and family.get("finite_initial_state_capacity") != expected[2]:
        raise ValueError(f"{family_id} finite initial-state capacity changed")
    if family_id == "billiards":
        if family.get("camera") != "generic.flat.camera":
            raise ValueError("billiards should reuse the generic camera policy")
    elif family_id == "pinball":
        _validate_camera(family.get("camera"), ("pinfield_front", "pinfield_left", "pinfield_right"), 18)
    else:
        _validate_camera(family.get("camera"), ("track_front", "track_left", "track_right"), 65)
    dimensions = set(family.get("diversity_dimensions", []))
    if not {"target_role", "camera_family"} <= dimensions:
        raise ValueError(f"{family_id} omits target or camera diversity")
    if family_id != "billiards" and not {"background_profile", "palette_permutation"} <= dimensions:
        raise ValueError(f"{family_id} omits appearance diversity")


def validate_rule_matrix(root: Path, matrix: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    if matrix.get("schema_version") != "physweep_three_object_rule_matrix_v3":
        raise ValueError("unsupported three-object rule matrix schema")
    if matrix.get("status") != "simplified_validated_pending_quota":
        raise ValueError("simplified rule matrix status changed")
    expected_group = {
        "object_count": 3,
        "object_ids": ["object_a", "object_b", "object_c"],
        "roles": list(ROLES),
        "sweep_target_object_id": "object_a",
        "members_per_group": 13,
        "base_members": 1,
        "sweep_members": 12,
        "sweep_axes": ["contact_friction", "contact_restitution", "mass_kg"],
        "sweep_values_per_axis": 4,
        "masks": False,
    }
    if matrix.get("group_contract") != expected_group:
        raise ValueError("three-object group contract changed")
    common = matrix.get("common_sampling_contract", {})
    if (
        common.get("camera_policy") != "admit_on_base_then_freeze_for_all_13_members"
        or common.get("camera_request_policy") != "requested_family_is_a_preference_and_may_fallback_during_base_admission"
        or common.get("camera_diversity_accounting") != "count_selected_frozen_family_after_base_admission"
        or common.get("camera_deficit_policy") != "replace_inside_the_same_physical_family_and_subcase_without_changing_the_physics_cell"
        or common.get("failure_policy") != "bounded_candidate_replacement_inside_the_same_frozen_cell"
        or common.get("public_sample_files") != ["metadata.json", "trajectory.npz", "video.mp4"]
    ):
        raise ValueError("common production contract changed")

    asset_report = _validate_asset_scope(root, matrix.get("asset_scope", {}))
    families = _by_id(matrix.get("families"), FAMILY_IDS, "family")
    if any(family.get("quota_key") != family_id for family_id, family in families.items()):
        raise ValueError("family quota keys must equal stable family ids")
    generic_report = _validate_generic(families["generic"])
    references = []
    for reference in families["generic"].get("implementation_refs", []):
        _resolved_reference(root, reference)
        references.append(reference["path"])
    if len(references) != 3 or len(set(references)) != 3:
        raise ValueError("generic implementation references changed")
    for family_id in FAMILY_IDS[1:]:
        reference = families[family_id].get("rule_ref", {})
        _resolved_reference(root, reference)
        references.append(reference["path"])
        _validate_special(families[family_id])

    overlay = matrix.get("production_quota_overlay", {})
    if (
        overlay.get("status") != "not_created_waiting_for_user_allocation"
        or overlay.get("required_family_keys") != list(FAMILY_IDS)
        or any(key in matrix for key in ("base_count", "family_base_counts", "candidate_budget"))
    ):
        raise ValueError("production quota was added to the rule-only matrix")
    required_unsupported = {
        "independent_three",
        "near_simultaneous_three_body_contact",
        "airborne_three_object_interaction",
        "generic_multi_mesh_chain_transfer",
        "non_axisymmetric_or_off_center_dynamic_compound_assets",
        "active_pinball_mechanisms",
        "post_physics_visual_rebinding",
        "sweep_outcome_based_group_replacement",
    }
    if not required_unsupported <= set(matrix.get("unsupported", [])):
        raise ValueError("unsupported physical scopes are incomplete")
    return {
        "status": "passed",
        "schema_version": matrix["schema_version"],
        "family_count": len(families),
        "production_quota_present": False,
        "asset_scope": asset_report,
        "generic": generic_report,
        "implementation_reference_count": len(references),
    }


def load_and_validate_rule_matrix(
    root: Path, path: Path | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    matrix_path = (root / (path or MATRIX_PATH)).resolve()
    if root != matrix_path and root not in matrix_path.parents:
        raise ValueError("rule matrix escapes the project root")
    matrix = _load(matrix_path)
    return matrix, validate_rule_matrix(root, matrix)
