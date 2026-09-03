#!/usr/bin/env python3
"""Build the bounded 2obj motion matrix from reviewed 1obj object sources."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json, write_json
from tools.dataset_contract.object_identity_contract import attach_object_identity
from tools.motion_rules.two_object import apply_two_object_motion
from tools.scene_rules.two_object import (
    DEFAULT_TWO_OBJECT_SCENE_RULES,
    allowed_camera_view_families,
    allowed_scene_classes,
    bind_two_object_scene,
    load_two_object_scene_rules,
    resolved_two_object_scene_rules,
)
from tools.sampling.object_collection import compile_object_collection_scene


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "physweep_two_object"
DEFAULT_MATRIX = PROJECT_ROOT / "configs" / "two_object_sampling_matrix.json"
_MATRIX_FIELDS = {
    "schema_version",
    "objects",
    "shape_families",
    "shape_motion_compatibility",
    "candidate_pool",
    "coverage_plan",
    "shared_physics",
    "pair_observation",
    "motion_intents",
    "policy",
}
_CANDIDATE_POOL_FIELDS = {
    "schema_version",
    "source_release",
    "object_source_families",
    "object_eligibility",
    "pair_eligibility",
}
_SOURCE_RELEASE_FIELDS = {
    "released_base_manifest_schema_version",
    "generation_manifest_schema_version",
    "sample_kind",
}
_OBJECT_ELIGIBILITY_FIELDS = {
    "body_model",
    "required_pose_profile",
    "scale_bins",
    "visual_profile_key",
    "distinct_source_scenes",
    "asset_proxy_policy",
    "asset_proxy_maximum_aabb_center_offset_m",
    "asset_scale_bin_maximum_extent_m",
}
_COVERAGE_PLAN_FIELDS = {
    "schema_version",
    "seed",
    "role_ordered_scale_pairs",
    "scene_motion_scale_compatibility",
    "role_ordered_shape_pairs",
    "camera_view_families",
    "role_ordered_source_family_pairs",
    "replicates_per_cell",
    "minimum_interacting_fraction",
    "selection_policy",
}
_SELECTION_POLICY = {
    "source_pair_uniqueness": "unordered_without_replacement",
    "host_uniqueness": "balanced_bounded_reuse",
    "host_must_differ_from_object_sources": True,
    "object_visual_profile_coverage": "balanced_eligible",
    "object_source_family_coverage": "balanced_feasible_role_ordered_pairs",
    "host_visual_profile_coverage": "all_eligible",
    "deterministic_ranking": "sha256_seeded",
}
_POLICY_FIELDS = {
    "post_contact_outcome_is_not_preclassified",
    "one_factor_sweep_keeps_both_initial_states_fixed",
    "initial_contact_is_deferred",
    "airborne_airborne_is_explicitly_bounded",
}


def _validated_intents(
    matrix: dict[str, Any], scene_rules: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    resolved_scene_rules = resolved_two_object_scene_rules(scene_rules)
    if set(matrix) != _MATRIX_FIELDS or (
        matrix.get("schema_version") != "physweep_two_object_sampling_matrix_v14"
    ):
        raise ValueError("unsupported two-object sampling matrix")
    objects = matrix.get("objects")
    if not isinstance(objects, dict) or set(objects) != {"schema_version", "roles"}:
        raise ValueError("two-object matrix lacks an object collection")
    if objects.get("schema_version") != "physweep_object_collection_v1":
        raise ValueError("unsupported object collection")
    roles = objects.get("roles")
    if (
        not isinstance(roles, list)
        or len(roles) != 2
        or any(
            not isinstance(role, dict) or set(role) != {"object_id"}
            for role in roles
        )
        or len({str(role["object_id"]).strip() for role in roles}) != 2
        or any(not str(role["object_id"]).strip() for role in roles)
    ):
        raise ValueError("two-object roles must contain two unique object ids")
    shape_contract = matrix.get("shape_families")
    if (
        not isinstance(shape_contract, dict)
        or set(shape_contract) != {"schema_version", "families"}
        or shape_contract.get("schema_version")
        != "physweep_two_object_shape_families_v1"
    ):
        raise ValueError("two-object shape-family contract is incomplete")
    families = shape_contract.get("families")
    family_fields = {
        "id",
        "geometry_type",
        "stable_pose_profile",
        "supported_motion_mode",
    }
    if (
        not isinstance(families, list)
        or any(not isinstance(record, dict) or set(record) != family_fields for record in families)
    ):
        raise ValueError("two-object shape families are invalid")
    family_ids = [str(record["id"]) for record in families]
    family_id_set = set(family_ids)
    geometry_types = [str(record["geometry_type"]) for record in families]
    expected_motion_modes = {
        "sphere": "rolling",
        "cuboid": "sliding_upright",
        "cylinder": "sliding_upright",
    }
    if (
        family_id_set != set(expected_motion_modes)
        or len(family_ids) != len(set(family_ids))
        or geometry_types != family_ids
        or any(record["stable_pose_profile"] != "support_normal" for record in families)
        or any(
            record["supported_motion_mode"]
            != expected_motion_modes[str(record["id"])]
            for record in families
        )
    ):
        raise ValueError("two-object shape families contradict rigid geometry")
    candidate_pool = matrix.get("candidate_pool")
    if (
        not isinstance(candidate_pool, dict)
        or set(candidate_pool) != _CANDIDATE_POOL_FIELDS
        or candidate_pool.get("schema_version")
        != "physweep_two_object_candidate_pool_v4"
    ):
        raise ValueError("two-object candidate-pool contract is incomplete")
    source = candidate_pool.get("source_release")
    if not isinstance(source, dict) or set(source) != _SOURCE_RELEASE_FIELDS:
        raise ValueError("two-object source-release contract is incomplete")
    if any(not isinstance(value, str) or not value for value in source.values()):
        raise ValueError("two-object source-release values must be nonempty strings")
    source_families = candidate_pool.get("object_source_families")
    if (
        not isinstance(source_families, list)
        or source_families
        != [
            {
                "id": "generic",
                "generation_metadata_schema_version": (
                    "physweep_pybullet_rigid_metadata_v1"
                ),
            },
            {
                "id": "asset",
                "generation_metadata_schema_version": (
                    "physweep_asset_proxy_scene_v3"
                ),
            },
        ]
    ):
        raise ValueError("two-object object-source families are invalid")
    eligibility = candidate_pool.get("object_eligibility")
    if (
        not isinstance(eligibility, dict)
        or set(eligibility) != _OBJECT_ELIGIBILITY_FIELDS
        or eligibility.get("body_model") != "rigid_body"
        or eligibility.get("required_pose_profile") != "support_normal"
        or eligibility.get("visual_profile_key") != "visual_profile.id"
        or eligibility.get("distinct_source_scenes") is not True
        or eligibility.get("asset_proxy_policy")
        != "centered_primitive_or_upright_axisymmetric_compound"
        or eligibility.get("asset_proxy_maximum_aabb_center_offset_m") != 0.001
    ):
        raise ValueError("two-object object-eligibility contract is invalid")
    scale_bins = eligibility.get("scale_bins")
    if (
        not isinstance(scale_bins, list)
        or not scale_bins
        or any(not isinstance(value, str) or not value for value in scale_bins)
        or len(scale_bins) != len(set(scale_bins))
    ):
        raise ValueError("two-object scale bins are invalid")
    asset_scale_limits = eligibility.get("asset_scale_bin_maximum_extent_m")
    if (
        not isinstance(asset_scale_limits, dict)
        or list(asset_scale_limits) != scale_bins
        or asset_scale_limits != {"small": 0.18, "medium": 0.23, "large": None}
    ):
        raise ValueError("two-object asset scale-bin contract is invalid")
    if candidate_pool.get("pair_eligibility") != {
        "maximum_interacting_mass_ratio": 50.0,
        "maximum_interacting_geometry_aspect_ratio": 6.0,
    }:
        raise ValueError("two-object pair-eligibility contract is invalid")
    coverage = matrix.get("coverage_plan")
    if (
        not isinstance(coverage, dict)
        or set(coverage) != _COVERAGE_PLAN_FIELDS
        or coverage.get("schema_version")
        != "physweep_two_object_coverage_plan_v4"
    ):
        raise ValueError("two-object coverage plan is incomplete")
    seed = coverage.get("seed")
    replicates = coverage.get("replicates_per_cell")
    minimum_interacting_fraction = coverage.get("minimum_interacting_fraction")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 1
        or isinstance(minimum_interacting_fraction, bool)
        or not isinstance(minimum_interacting_fraction, (int, float))
        or not math.isfinite(float(minimum_interacting_fraction))
        or not 0.0 < float(minimum_interacting_fraction) < 1.0
    ):
        raise ValueError("two-object coverage seed and replicates are invalid")
    scale_pairs = coverage.get("role_ordered_scale_pairs")
    if not isinstance(scale_pairs, list) or any(
        not isinstance(record, dict)
        or set(record) != {"id", "object_a", "object_b"}
        for record in scale_pairs
    ):
        raise ValueError("two-object ordered scale pairs are invalid")
    pair_ids = [str(record["id"]) for record in scale_pairs]
    actual_pairs = {
        (str(record["object_a"]), str(record["object_b"]))
        for record in scale_pairs
    }
    expected_pairs = {(left, right) for left in scale_bins for right in scale_bins}
    if (
        any(not pair_id for pair_id in pair_ids)
        or len(pair_ids) != len(set(pair_ids))
        or len(actual_pairs) != len(scale_pairs)
        or actual_pairs != expected_pairs
    ):
        raise ValueError("two-object ordered scale pairs must cover the full product")
    shape_pairs = coverage.get("role_ordered_shape_pairs")
    if not isinstance(shape_pairs, list) or any(
        not isinstance(record, dict)
        or set(record) != {"id", "object_a", "object_b"}
        for record in shape_pairs
    ):
        raise ValueError("two-object ordered shape pairs are invalid")
    shape_pair_ids = [str(record["id"]) for record in shape_pairs]
    shape_pair_id_set = set(shape_pair_ids)
    actual_shape_pairs = {
        (str(record["object_a"]), str(record["object_b"]))
        for record in shape_pairs
    }
    expected_shape_pairs = {
        (left, right) for left in family_ids for right in family_ids
    }
    if (
        any(not pair_id for pair_id in shape_pair_ids)
        or len(shape_pair_ids) != len(set(shape_pair_ids))
        or len(actual_shape_pairs) != len(shape_pairs)
        or actual_shape_pairs != expected_shape_pairs
    ):
        raise ValueError("two-object ordered shape pairs must cover the full product")
    view_families = coverage.get("camera_view_families")
    view_fields = {
        "id",
        "relative_azimuth_degrees",
        "preferred_elevation_degrees",
        "minimum_elevation_degrees",
        "maximum_elevation_degrees",
    }
    if (
        not isinstance(view_families, list)
        or len(view_families) < 4
        or any(
            not isinstance(record, dict) or set(record) != view_fields
            for record in view_families
        )
    ):
        raise ValueError("two-object camera-view families are invalid")
    view_ids = [str(record["id"]) for record in view_families]
    if (
        any(not value for value in view_ids)
        or len(view_ids) != len(set(view_ids))
        or any(
            not -180.0 <= float(record["relative_azimuth_degrees"]) <= 180.0
            or not 0.0
            < float(record["minimum_elevation_degrees"])
            <= float(record["preferred_elevation_degrees"])
            <= float(record["maximum_elevation_degrees"])
            < 90.0
            for record in view_families
        )
    ):
        raise ValueError("two-object camera-view family values are invalid")
    source_pairs = coverage.get("role_ordered_source_family_pairs")
    source_family_ids = [str(record["id"]) for record in source_families]
    expected_source_pairs = {
        (left, right) for left in source_family_ids for right in source_family_ids
    }
    if (
        not isinstance(source_pairs, list)
        or any(
            not isinstance(record, dict)
            or set(record) != {"id", "object_a", "object_b"}
            for record in source_pairs
        )
        or len({str(record["id"]) for record in source_pairs}) != len(source_pairs)
        or {
            (str(record["object_a"]), str(record["object_b"]))
            for record in source_pairs
        }
        != expected_source_pairs
    ):
        raise ValueError("two-object source-family pairs must cover the full product")
    selection = coverage.get("selection_policy")
    if not isinstance(selection, dict) or (
        set(selection)
        != {
            *_SELECTION_POLICY,
            "maximum_object_source_reuse",
            "maximum_host_source_reuse",
        }
    ):
        raise ValueError("two-object selection policy is incomplete")
    if any(selection.get(key) != value for key, value in _SELECTION_POLICY.items()):
        raise ValueError("two-object selection policy may not be weakened")
    if (
        selection.get("maximum_object_source_reuse") != 2
        or selection.get("maximum_host_source_reuse") != 2
    ):
        raise ValueError("two-object source reuse limits may not be weakened")
    policy = matrix.get("policy")
    if not isinstance(policy, dict) or set(policy) != _POLICY_FIELDS:
        raise ValueError("two-object sampling policy is incomplete")
    if not all(bool(policy[field]) for field in _POLICY_FIELDS):
        raise ValueError("two-object sampling policy may not be weakened")
    intents = matrix.get("motion_intents")
    if not isinstance(intents, list) or not intents:
        raise ValueError("two-object matrix contains no motion intents")
    if any(not isinstance(intent, dict) for intent in intents):
        raise ValueError("two-object motion intents must be records")
    ids = [str(intent.get("id", "")) for intent in intents]
    if any(not motion_id for motion_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("two-object motion intent ids must be unique")
    admitted_motion_contracts = {
        (str(regime), str(interaction_class))
        for rule in resolved_scene_rules["physical_rules"]
        for regime in rule["allowed_kinematic_regimes"]
        for interaction_class in rule["allowed_interaction_classes"]
    }
    if any(
        (
            str(intent.get("kinematic_regime", "")),
            str(intent.get("interaction_class", "")),
        )
        not in admitted_motion_contracts
        for intent in intents
    ):
        raise ValueError("two-object motion intent has no compatible scene rule")
    intent_id_set = set(ids)
    scene_motion_scale_compatibility = coverage.get(
        "scene_motion_scale_compatibility"
    )
    compatibility_fields = {
        "scene_class",
        "motion_ids",
        "allowed_scale_pair_ids",
    }
    declared_scene_classes = set(allowed_scene_classes(resolved_scene_rules))
    declared_scale_pair_ids = set(pair_ids)
    constrained_scene_motions: set[tuple[str, str]] = set()
    if not isinstance(scene_motion_scale_compatibility, list) or any(
        not isinstance(record, dict)
        or set(record) != compatibility_fields
        or str(record["scene_class"]) not in declared_scene_classes
        or not isinstance(record["motion_ids"], list)
        or not record["motion_ids"]
        or any(str(value) not in intent_id_set for value in record["motion_ids"])
        or len(record["motion_ids"]) != len(set(map(str, record["motion_ids"])))
        or not isinstance(record["allowed_scale_pair_ids"], list)
        or not record["allowed_scale_pair_ids"]
        or any(
            str(value) not in declared_scale_pair_ids
            for value in record["allowed_scale_pair_ids"]
        )
        or len(record["allowed_scale_pair_ids"])
        != len(set(map(str, record["allowed_scale_pair_ids"])))
        for record in scene_motion_scale_compatibility
    ):
        raise ValueError(
            "two-object scene-motion-scale compatibility is invalid"
        )
    for record in scene_motion_scale_compatibility:
        scene_class = str(record["scene_class"])
        for motion_id in map(str, record["motion_ids"]):
            key = (scene_class, motion_id)
            if key in constrained_scene_motions:
                raise ValueError(
                    "two-object scene-motion-scale compatibility overlaps"
                )
            intent = next(value for value in intents if str(value["id"]) == motion_id)
            if not any(
                str(rule["scene_class"]) == scene_class
                and str(intent["kinematic_regime"])
                in rule["allowed_kinematic_regimes"]
                and str(intent["interaction_class"])
                in rule["allowed_interaction_classes"]
                and bool(allowed_camera_view_families(rule, motion_id))
                for rule in resolved_scene_rules["physical_rules"]
            ):
                raise ValueError(
                    "two-object scale constraint names an incompatible scene motion"
                )
            constrained_scene_motions.add(key)
    compatibility = matrix.get("shape_motion_compatibility")
    if (
        not isinstance(compatibility, dict)
        or set(compatibility) != {"schema_version", "pair_sets", "rules"}
        or compatibility.get("schema_version")
        != "physweep_two_object_shape_motion_compatibility_v1"
    ):
        raise ValueError("two-object shape-motion compatibility is incomplete")
    pair_sets = compatibility.get("pair_sets")
    if (
        not isinstance(pair_sets, dict)
        or not pair_sets
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(values, list)
            or not values
            or any(str(value) not in shape_pair_id_set for value in values)
            or len(values) != len(set(map(str, values)))
            for name, values in pair_sets.items()
        )
    ):
        raise ValueError("two-object shape-pair sets are invalid")
    compatibility_rules = compatibility.get("rules")
    if (
        not isinstance(compatibility_rules, list)
        or any(
            not isinstance(record, dict)
            or set(record) != {"motion_id", "shape_pair_set_id"}
            or str(record["motion_id"]) not in intent_id_set
            or str(record["shape_pair_set_id"]) not in pair_sets
            for record in compatibility_rules
        )
        or [str(record["motion_id"]) for record in compatibility_rules] != ids
    ):
        raise ValueError("two-object shape-motion rules must follow motion order")
    admitted_pairs = {
        str(pair_id)
        for record in compatibility_rules
        for pair_id in pair_sets[str(record["shape_pair_set_id"])]
    }
    if admitted_pairs != shape_pair_id_set:
        raise ValueError("two-object compatibility leaves a shape pair unreachable")
    return intents


def compatible_shape_pair_ids(
    matrix: dict[str, Any], motion_id: str
) -> tuple[str, ...]:
    """Return the explicitly admitted ordered shape pairs for one motion."""

    rules = matrix["shape_motion_compatibility"]
    record = next(
        (
            value
            for value in rules["rules"]
            if str(value["motion_id"]) == motion_id
        ),
        None,
    )
    if record is None:
        raise ValueError(f"unknown two-object motion intent: {motion_id}")
    return tuple(
        str(value)
        for value in rules["pair_sets"][str(record["shape_pair_set_id"])]
    )


def compatible_scale_pair_ids(
    matrix: dict[str, Any], motion_id: str, scene_class: str
) -> tuple[str, ...]:
    """Return scale pairs admitted for one motion in one scene class."""

    declared = tuple(
        str(record["id"])
        for record in matrix["coverage_plan"]["role_ordered_scale_pairs"]
    )
    matches = [
        record
        for record in matrix["coverage_plan"][
            "scene_motion_scale_compatibility"
        ]
        if str(record["scene_class"]) == scene_class
        and motion_id in set(map(str, record["motion_ids"]))
    ]
    if not matches:
        return declared
    if len(matches) != 1:
        raise ValueError("overlapping two-object scene-motion-scale compatibility")
    return tuple(map(str, matches[0]["allowed_scale_pair_ids"]))


def shape_pair_id(
    matrix: dict[str, Any], object_a_shape: str, object_b_shape: str
) -> str:
    """Resolve one ordered shape pair without deriving ids from string syntax."""

    matches = [
        str(record["id"])
        for record in matrix["coverage_plan"]["role_ordered_shape_pairs"]
        if str(record["object_a"]) == object_a_shape
        and str(record["object_b"]) == object_b_shape
    ]
    if len(matches) != 1:
        raise ValueError("two-object geometry does not name one declared shape pair")
    return matches[0]


def camera_view_family(
    matrix: dict[str, Any], family_id: str | None = None
) -> dict[str, Any]:
    """Resolve one declared pair-relative camera family."""

    families = matrix["coverage_plan"]["camera_view_families"]
    resolved_id = str(families[0]["id"]) if family_id is None else str(family_id)
    matches = [record for record in families if str(record["id"]) == resolved_id]
    if len(matches) != 1:
        raise ValueError(f"unknown two-object camera-view family: {resolved_id}")
    return matches[0]


def build_two_object_scene(
    host_template: dict[str, Any],
    matrix: dict[str, Any],
    motion_id: str,
    object_templates: Sequence[dict[str, Any]] | None = None,
    sample_index: int | None = None,
    camera_view_family_id: str | None = None,
    scene_rules: dict[str, Any] | None = None,
    scene_rule_id: str | None = None,
) -> dict[str, Any]:
    """Compose two sources and apply one declared initial-state intent."""

    resolved_scene_rules = resolved_two_object_scene_rules(scene_rules)
    intents = _validated_intents(matrix, resolved_scene_rules)
    matches = [intent for intent in intents if str(intent["id"]) == motion_id]
    if len(matches) != 1:
        raise ValueError(f"unknown two-object motion intent: {motion_id}")
    declared_index = intents.index(matches[0]) + 1
    resolved_index = declared_index if sample_index is None else sample_index
    if isinstance(resolved_index, bool) or not isinstance(resolved_index, int):
        raise ValueError("sample index must be an integer")
    if resolved_index < 1:
        raise ValueError("sample index must be positive")
    sources = (
        tuple(object_templates)
        if object_templates is not None
        else (host_template, host_template)
    )
    pair_scene = compile_object_collection_scene(
        host_template,
        sources,
        matrix["objects"]["roles"],
    )
    shapes = [
        str(record.get("geometry", {}).get("type", ""))
        for record in pair_scene["simulation"]["objects"]
    ]
    resolved_shape_pair = shape_pair_id(matrix, shapes[0], shapes[1])
    if resolved_shape_pair not in compatible_shape_pair_ids(matrix, motion_id):
        raise ValueError(
            f"shape pair {resolved_shape_pair} is incompatible with {motion_id}"
        )
    scene = apply_two_object_motion(
        pair_scene,
        matrix["shape_families"],
        matrix["shared_physics"],
        matrix["pair_observation"],
        camera_view_family(matrix, camera_view_family_id),
        matches[0],
    )
    scene = bind_two_object_scene(
        scene, resolved_scene_rules, expected_rule_id=scene_rule_id
    )
    scene["dataset_id"] = DATASET_ID
    scene["dataset_stage"] = "two_object_base_candidate"
    scene["sample_index"] = resolved_index
    attach_object_identity(scene)
    return scene


def build_two_object_matrix(
    host_template: dict[str, Any],
    matrix: dict[str, Any],
    object_templates: Sequence[dict[str, Any]] | None = None,
    motion_ids: Sequence[str] | None = None,
    scene_rules: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build all or a declared subset of matrix rows in stable matrix order."""

    resolved_scene_rules = resolved_two_object_scene_rules(scene_rules)
    intents = _validated_intents(matrix, resolved_scene_rules)
    declared_ids = [str(intent["id"]) for intent in intents]
    requested_ids = (
        declared_ids if motion_ids is None else [str(value) for value in motion_ids]
    )
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("two-object motion intents may not be repeated")
    selected = set(requested_ids)
    unknown = sorted(selected.difference(declared_ids))
    if unknown:
        raise ValueError(f"unknown two-object motion intents: {unknown}")
    if not selected:
        raise ValueError("at least one two-object motion intent is required")
    return [
        build_two_object_scene(
            host_template,
            matrix,
            motion_id,
            object_templates,
            scene_rules=resolved_scene_rules,
        )
        for motion_id in declared_ids
        if motion_id in selected
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--object-a-template", type=Path)
    parser.add_argument("--object-b-template", type=Path)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--scene-rules", type=Path, default=DEFAULT_TWO_OBJECT_SCENE_RULES
    )
    parser.add_argument("--motion-id", action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    host_path = args.template.resolve()
    object_paths = [
        (args.object_a_template or host_path).resolve(),
        (args.object_b_template or host_path).resolve(),
    ]
    matrix_path = args.matrix.resolve()
    scene_rules_path = args.scene_rules.resolve()
    output_dir = args.output_dir.resolve()
    try:
        host_relative = host_path.relative_to(root)
        object_relatives = [path.relative_to(root) for path in object_paths]
        matrix_relative = matrix_path.relative_to(root)
        scene_rules_relative = scene_rules_path.relative_to(root)
        output_dir.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "two-object matrix inputs and output must remain under --root"
        ) from error
    matrix = read_json(matrix_path)
    scene_rules = load_two_object_scene_rules(scene_rules_path)
    scenes = build_two_object_matrix(
        read_json(host_path),
        matrix,
        [read_json(path) for path in object_paths],
        args.motion_id,
        scene_rules,
    )
    samples = []
    for scene in scenes:
        metadata_path = output_dir / "scenes" / scene["scene_id"] / "metadata.json"
        write_json(metadata_path, scene)
        samples.append(
            {
                "scene_id": scene["scene_id"],
                "metadata_path": str(metadata_path.relative_to(root)),
                "metadata_sha256": sha256(metadata_path),
            }
        )
    object_ids = [
        str(role["object_id"]) for role in matrix["objects"]["roles"]
    ]
    object_sources = [
        {
            "object_id": object_id,
            "path": str(relative),
            "sha256": sha256(path),
        }
        for object_id, path, relative in zip(
            object_ids, object_paths, object_relatives
        )
    ]
    manifest = {
        "schema_version": "physweep_pybullet_base_manifest_v1",
        "dataset_id": DATASET_ID,
        "sample_count": len(samples),
        "matrix": {
            "path": str(matrix_relative),
            "sha256": sha256(matrix_path),
        },
        "scene_rules": {
            "path": str(scene_rules_relative),
            "sha256": sha256(scene_rules_path),
        },
        "host_template": {
            "path": str(host_relative),
            "sha256": sha256(host_path),
        },
        "object_sources": object_sources,
        "samples": samples,
        "status": "sampled_pending_simulation",
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
