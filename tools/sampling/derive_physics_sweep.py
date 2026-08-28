#!/usr/bin/env python3
"""Derive immutable one-factor PyBullet sweep metadata from frozen base scenes."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json
from tools.core.sweep_values import (
    allowed_sweep_domain,
    resolve_sweep_domain,
    round_sweep_value,
    sweep_values,
)
from tools.dataset_contract.object_identity_contract import attach_object_identity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/physics_sweep.json"


def validate_output_dir(root: Path, output_dir: Path) -> None:
    try:
        relative = output_dir.relative_to(root)
    except ValueError as error:
        raise ValueError("sweep output must be inside the project root") from error
    if len(relative.parts) < 2:
        raise ValueError("refusing to replace a top-level project directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"sweep output is not empty: {output_dir}; use a clean output directory"
        )


def _objects(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    objects = metadata.get("simulation", {}).get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("simulation.objects must be a list")
    return objects


def _schema(metadata: dict[str, Any]) -> str:
    return str(metadata.get("schema_version", ""))


def _identity_objects(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    records = metadata.get("object_identity", {}).get("objects", [])
    if not isinstance(records, list):
        raise ValueError("object_identity.objects must be a list")
    dynamic = [record for record in records if record.get("role") == "dynamic"]
    if not dynamic:
        raise ValueError("metadata contains no dynamic object identity")
    return dynamic


def _dynamic_objects(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    objects = _objects(metadata)
    if objects:
        return objects
    return _identity_objects(metadata)


def _object_id(obj: dict[str, Any], index: int) -> str:
    """Return the required stable object binding."""
    value = obj.get("object_id")
    if value in (None, ""):
        raise ValueError(f"simulation.objects[{index}] has no object_id")
    return str(value)


def _target_object_indices(
    metadata: dict[str, Any],
    target_object_ids: list[str] | None = None,
    target_object_indices: list[int] | None = None,
) -> list[int]:
    objects = _dynamic_objects(metadata)
    ids = [_object_id(obj, index) for index, obj in enumerate(objects)]
    if len(set(ids)) != len(ids):
        raise ValueError("dynamic object identifiers must be unique")
    if target_object_ids and target_object_indices:
        raise ValueError("choose target object ids or indices, not both")
    if target_object_ids:
        lookup = {object_id: index for index, object_id in enumerate(ids)}
        missing = [object_id for object_id in target_object_ids if object_id not in lookup]
        if missing:
            raise ValueError(f"unknown target object id(s): {', '.join(missing)}")
        return [lookup[object_id] for object_id in target_object_ids]
    if target_object_indices:
        if any(index < 0 or index >= len(objects) for index in target_object_indices):
            raise ValueError("target object index is outside simulation.objects")
        return list(dict.fromkeys(target_object_indices))
    return list(range(len(objects)))


def _asset_id(metadata: dict[str, Any], object_index: int) -> str | None:
    objects = _dynamic_objects(metadata)
    obj = objects[object_index]
    for value in (
        obj.get("asset_id"),
        obj.get("visual_profile", {}).get("asset_id"),
        metadata.get("assets", {}).get("dynamic_asset_id"),
    ):
        if value:
            return str(value)
    return None


def _runtime_material(
    metadata: dict[str, Any],
    root: Path,
    registry: dict[str, Any],
    object_index: int,
) -> dict[str, float]:
    schema = _schema(metadata)
    objects = _objects(metadata)
    if objects:
        material = objects[object_index].get("material", {})
        return {
            "mass_kg": float(material["mass_kg"]),
            "contact_friction": float(material["contact_friction"]),
            "contact_restitution": float(material["contact_restitution"]),
        }
    if schema == "physweep_asset_proxy_scene_v3":
        asset_id = _asset_id(metadata, object_index)
        record = registry.get(str(asset_id))
        if record is None:
            raise ValueError(f"asset proxy scene has no registry record: {asset_id}")
        material = record.get("proxy", {}).get("material", {})
        return {
            "mass_kg": float(metadata["physics"]["mass_kg"]),
            "contact_friction": float(material["friction"]),
            "contact_restitution": float(material["restitution"]),
        }
    if schema == "physweep_billiards_scene_v4":
        backend_path = root / str(metadata["physics"]["backend_config"]["path"])
        backend = load_json(backend_path)
        dynamics = backend["billiards_rules"]["ball_dynamics"]
        return {
            "mass_kg": float(metadata["physics"]["ball_mass_kg"]),
            "contact_friction": float(dynamics["lateral_friction"]),
            "contact_restitution": float(dynamics["restitution"]),
        }
    raise ValueError(f"unsupported base metadata schema: {schema!r}")


def _set_runtime_material(
    metadata: dict[str, Any],
    object_index: int,
    material: dict[str, float],
) -> str:
    schema = _schema(metadata)
    objects = _objects(metadata)
    if objects:
        obj = objects[object_index]
        obj["material"].update(material)
        return f"simulation.objects[{object_index}].material"
    physics = metadata.setdefault("physics", {})
    physics["runtime_material"] = copy.deepcopy(material)
    if schema == "physweep_asset_proxy_scene_v3":
        physics["mass_kg"] = material["mass_kg"]
        return "physics.runtime_material"
    if schema == "physweep_billiards_scene_v4":
        physics["ball_mass_kg"] = material["mass_kg"]
        return "physics.runtime_material"
    raise ValueError(f"unsupported base metadata schema: {schema!r}")


def _clear_parent_outputs(metadata: dict[str, Any]) -> None:
    """Remove generated parent artifacts that must be recomputed for a sweep."""
    physics = metadata.get("physics")
    if isinstance(physics, dict):
        for key in ("trajectory_path", "audit_path", "simulation_record_path"):
            physics.pop(key, None)
    render = metadata.get("render")
    if isinstance(render, dict):
        for key in ("video_path", "inspection_frame_dir"):
            render.pop(key, None)


def _record_ids(metadata: dict[str, Any], object_index: int) -> list[str]:
    obj = _dynamic_objects(metadata)[object_index]
    visual = obj.get("visual_profile", {})
    candidates = (
        obj.get("asset_id"),
        visual.get("asset_id"),
        visual.get("id"),
        obj.get("semantic_type"),
    )
    return list(dict.fromkeys(str(value) for value in candidates if value))


def _load_prior_indexes(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    profiles_path = root / str(config["prior_sources"]["physassets_profiles"])
    registry_path = root / str(config["prior_sources"]["asset_proxy_registry"])
    profiles: dict[str, Any] = {}
    registry: dict[str, Any] = {}
    if profiles_path.exists():
        document = load_json(profiles_path)
        profiles = {str(item["id"]): item for item in document.get("profiles", [])}
    if registry_path.exists():
        document = load_json(registry_path)
        registry = {str(item["asset_id"]): item for item in document.get("records", [])}
    return profiles, registry


def resolve_prior_provenance(
    root: Path,
    config: dict[str, Any],
    base_manifest: dict[str, Any] | None,
) -> dict[str, dict[str, str]]:
    declared = {
        str(record.get("path")): str(record.get("sha256"))
        for record in (base_manifest or {}).get("dependencies", {}).values()
        if isinstance(record, dict) and record.get("path") and record.get("sha256")
    }
    result: dict[str, dict[str, str]] = {}
    for name, relative_path in config.get("prior_sources", {}).items():
        path = root / str(relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"missing sweep prior source: {path}")
        digest = sha256(path)
        frozen_digest = declared.get(str(relative_path))
        if frozen_digest is not None and frozen_digest != digest:
            raise ValueError(
                f"base manifest dependency changed: {relative_path}"
            )
        result[str(name)] = {"path": str(relative_path), "sha256": digest}
    return result


def _mass_bounds(
    metadata: dict[str, Any],
    profiles: dict[str, Any],
    registry: dict[str, Any],
    object_index: int,
) -> list[float] | None:
    objects = _objects(metadata)
    if objects:
        bounds = objects[object_index].get("material", {}).get("mass_range_kg")
        if bounds and len(bounds) == 2:
            return [float(bounds[0]), float(bounds[1])]
    for asset_id in _record_ids(metadata, object_index):
        profile = profiles.get(asset_id)
        if profile:
            bounds = profile.get("physics", {}).get("mass_range_kg")
            if bounds and len(bounds) == 2:
                return [float(bounds[0]), float(bounds[1])]
        record = registry.get(asset_id)
        if record:
            bounds = record.get("proxy", {}).get("mass_range_kg")
            if bounds and len(bounds) == 2:
                return [float(bounds[0]), float(bounds[1])]
    return None


def _friction_domain(
    metadata: dict[str, Any],
    axis_rules: dict[str, Any],
    base_value: float,
    axis: str,
    object_index: int,
) -> list[float] | None:
    if axis != "contact_friction":
        return None
    if _schema(metadata) != "physweep_pybullet_rigid_metadata_v1":
        return None
    obj = _objects(metadata)[object_index]
    expected = obj.get("expected_motion", {})
    motion = str(expected.get("motion_family", ""))
    motion_family = motion.rsplit("_", 1)[0] if motion.endswith("obj") else motion
    event_requiring_motion = {
        "slide_push",
        "roll_or_slide",
        "ramp_to_flat",
        "slope_slide_down",
        "slope_slide_up",
        "wall_impact",
        "edge_fall",
    }
    if motion_family not in event_requiring_motion:
        return None
    try:
        support_frame = metadata["simulation"]["support"]["surface_frame"]
        normal = [float(value) for value in support_frame["normal"]]
        gravity = [
            float(value)
            for value in metadata["simulation"]["world"]["gravity_m_s2"]
        ]
        velocity = [
            float(value)
            for value in obj["initial_state"][
                "linear_velocity_m_s"
            ]
        ]
        minimum_distance = float(
            expected.get(
                "minimum_uphill_displacement_m",
                expected["minimum_displacement_m"],
            )
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{motion_family} lacks the analytic support-frame contract "
            "required for a friction sweep"
        ) from error
    normal_length = math.sqrt(sum(value * value for value in normal))
    if normal_length <= 1.0e-9:
        raise ValueError("friction sweep support normal must be nonzero")
    if minimum_distance <= 1.0e-6:
        raise ValueError("friction sweep minimum displacement must be positive")
    normal = [value / normal_length for value in normal]
    normal_velocity = sum(v * n for v, n in zip(velocity, normal))
    tangent_velocity = [
        v - normal_velocity * n for v, n in zip(velocity, normal)
    ]
    speed = math.sqrt(sum(value * value for value in tangent_velocity))
    gravity_tangent = [
        g - sum(g2 * n for g2, n in zip(gravity, normal)) * n
        for g, n in zip(gravity, normal)
    ]
    normal_acceleration = math.sqrt(
        sum((g - gt) ** 2 for g, gt in zip(gravity, gravity_tangent))
    )
    if normal_acceleration <= 1.0e-6:
        raise ValueError("friction sweep requires gravity normal to the support")
    slope_angle = math.radians(
        abs(float(support_frame.get("slope_angle_degrees", 0.0)))
    )
    slope_transition_limit = math.tan(slope_angle)
    if speed <= 1.0e-6:
        if motion_family != "slope_slide_down" or slope_transition_limit <= 1.0e-6:
            raise ValueError(
                f"{motion_family} friction sweep requires tangential initial motion"
            )
        theoretical_max = slope_transition_limit
    else:
        direction = [value / speed for value in tangent_velocity]
        gravity_tangent_component = sum(
            g * d for g, d in zip(gravity, direction)
        )
        theoretical_max = (
            speed * speed / (2.0 * minimum_distance) + gravity_tangent_component
        ) / normal_acceleration
    static_hold_limit = (
        slope_transition_limit
        if bool(expected.get("must_reverse_downhill"))
        else 0.0
    )
    configured_low, configured_high = [
        float(value) for value in axis_rules["domain"]
    ]
    if "transition_margin" in axis_rules:
        # A high-friction endpoint deliberately crosses the calculated motion
        # transition; downstream audits still enforce hard physical checks.
        transition_threshold = theoretical_max
        if static_hold_limit > 1.0e-6:
            transition_threshold = max(transition_threshold, static_hold_limit)
        margin = float(axis_rules["transition_margin"])
        feasible_high = max(base_value, transition_threshold * margin)
    else:
        if static_hold_limit > 1.0e-6:
            theoretical_max = min(theoretical_max, static_hold_limit)
        safety = float(axis_rules.get("motion_feasibility_safety", 0.85))
        feasible_high = max(base_value, theoretical_max * safety)
    # Keep the original feasible domain when it already contains the base on
    # both sides. Only widen an endpoint when the feasible motion bound would
    # otherwise place the base at an edge and violate the centered policy.
    range_policy = axis_rules.get("range_policy", {})
    if (
        range_policy.get("mode") == "relative_multipliers"
        and feasible_high <= base_value
    ):
        requested_high = base_value * float(range_policy["upper_multiplier"])
        feasible_high = max(feasible_high, requested_high)
    return [configured_low, min(configured_high, feasible_high)]


def validate_base(
    metadata: dict[str, Any],
    max_objects: int = 3,
    supported_schemas: set[str] | None = None,
) -> None:
    schema = _schema(metadata)
    supported = supported_schemas or {
        "physweep_pybullet_rigid_metadata_v1",
        "physweep_asset_proxy_scene_v3",
        "physweep_billiards_scene_v4",
        "physweep_passive_pinball_scene_v1",
        "physweep_marble_run_scene_v1",
    }
    if schema not in supported:
        raise ValueError(f"unsupported base metadata schema: {schema!r}")
    if schema == "physweep_pybullet_rigid_metadata_v1":
        dataset_stage = str(metadata.get("dataset_stage", ""))
        if not dataset_stage.endswith("_base_candidate"):
            raise ValueError("input metadata is not a base candidate")
    if "sweep" in metadata:
        raise ValueError("input already contains a sweep binding")
    objects = _dynamic_objects(metadata)
    if not objects or len(objects) > max_objects:
        raise ValueError(
            f"the object-bound rigid sweep supports 1..{max_objects} dynamic objects"
        )
    identifiers = [_object_id(obj, index) for index, obj in enumerate(objects)]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("dynamic object identifiers must be unique")
    if _objects(metadata):
        for obj in objects:
            if obj.get("body_model") != "rigid_body":
                raise ValueError("the active rigid sweep supports rigid_body records only")
            material = obj.get("material", {})
            for field in ("mass_kg", "contact_friction", "contact_restitution"):
                if field not in material:
                    raise ValueError(
                        f"base metadata lacks runtime material field: {field}"
                    )


def derive_one(
    base: dict[str, Any],
    base_path: Path,
    root: Path,
    config: dict[str, Any],
    config_path: Path,
    axis: str,
    level_index: int,
    profiles: dict[str, Any],
    registry: dict[str, Any],
    target_object_index: int = 0,
) -> dict[str, Any]:
    max_objects = int(config.get("max_dynamic_objects", 3))
    validate_base(
        base,
        max_objects=max_objects,
        supported_schemas={str(value) for value in config["base_schema_versions"]},
    )
    objects = _dynamic_objects(base)
    if target_object_index < 0 or target_object_index >= len(objects):
        raise IndexError(f"invalid target object index: {target_object_index}")
    target_object_id = _object_id(objects[target_object_index], target_object_index)
    axis_rules = config["axes"][axis]
    runtime_material = _runtime_material(
        base, root, registry, target_object_index
    )
    base_value = float(runtime_material[axis])
    mass_bounds = _mass_bounds(base, profiles, registry, target_object_index)
    domain_override = _friction_domain(
        base, axis_rules, base_value, axis, target_object_index
    )
    schema_domain = axis_rules.get("schema_domains", {}).get(_schema(base))
    if schema_domain is not None:
        domain_override = [float(value) for value in schema_domain]
    allowed_domain = allowed_sweep_domain(
        base_value,
        axis_rules,
        mass_bounds,
        axis,
        domain_override,
    )
    resolved_domain = resolve_sweep_domain(
        base_value,
        axis_rules,
        allowed_domain,
    )
    values = sweep_values(
        base_value,
        axis_rules,
        mass_bounds,
        axis,
        domain_override=domain_override,
        endpoint_policy=config.get("endpoint_policy"),
    )
    if level_index < 0 or level_index >= len(values):
        raise IndexError(f"invalid sweep level {level_index} for {axis}")
    rounded_base = round_sweep_value(base_value)
    base_level_index = values.index(rounded_base)
    requested_base_policy = str(
        config.get("endpoint_policy", {}).get(
            "base_value_policy", "preserve_exactly_at_nearest_position"
        )
    )
    if requested_base_policy == "preserve_exactly_at_middle_position":
        if base_level_index != len(values) // 2:
            raise ValueError(
                f"{axis} derived values do not place the base at the middle level"
            )
        applied_base_policy = requested_base_policy
    else:
        applied_base_policy = requested_base_policy

    derived = copy.deepcopy(base)
    _clear_parent_outputs(derived)
    derived["scene_id"] = (
        f"{base['scene_id']}__sweep_{target_object_id}_{axis}_{level_index:02d}"
    )
    derived["dataset_stage"] = "object_physics_sweep_candidate"
    resolved_material = copy.deepcopy(runtime_material)
    resolved_material[axis] = values[level_index]
    material_path = _set_runtime_material(
        derived, target_object_index, resolved_material
    )
    object_field = str(config["axes"][axis]["object_field"])
    field = f"{material_path}.{axis}"
    resolved_object_physics = [
        {
            "object_id": _object_id(obj, index),
            "object_index": index,
            "material": (
                copy.deepcopy(resolved_material)
                if index == target_object_index
                else _runtime_material(base, root, registry, index)
            ),
        }
        for index, obj in enumerate(_dynamic_objects(derived))
    ]
    derived["sweep"] = {
        "schema_version": config["version"],
        "kind": "sweep",
        "mode": "one_factor",
        "parent_scene_id": str(base["scene_id"]),
        "parent_metadata_path": str(base_path.relative_to(root)),
        "parent_metadata_sha256": sha256(base_path),
        "target_object_id": target_object_id,
        "target_object_index": target_object_index,
        "parameter": axis,
        "axis": axis,
        "level_index": level_index,
        "level_count": len(values),
        "value": values[level_index],
        "base_value": rounded_base,
        "base_level_index": base_level_index,
        "base_value_policy_applied": applied_base_policy,
        "domain": [round_sweep_value(value) for value in resolved_domain],
        "allowed_domain": [round_sweep_value(value) for value in allowed_domain],
        "object_field": object_field,
        "overridden_field": field,
        "source_schema_version": _schema(base),
        "resolved_object_physics": resolved_object_physics,
        "resolved_state_policy": "all_dynamic_objects_serialized_in_metadata",
        "config_path": str(config_path.relative_to(root)),
        "config_sha256": sha256(config_path),
        "initial_state_policy": "copied_from_base_unchanged",
        "visual_policy": "copied_from_base_unchanged",
        "endpoint_policy": copy.deepcopy(
            config.get("endpoint_policy", {
                "algorithm": "axis_domain_then_interpolate",
                "normalized_positions": axis_rules["level_positions"],
                "base_value_policy": "preserve_exactly_at_nearest_position",
                "edge_policy": "reject_if_middle_impossible",
                "interpolation_policy": "axis_scale",
            })
        ),
    }
    attach_object_identity(derived)
    return derived


def normalize_canonical_base(derived: dict[str, Any]) -> dict[str, Any]:
    """Turn the single retained middle level into an unbound canonical base."""
    result = copy.deepcopy(derived)
    sweep = result["sweep"]
    if round_sweep_value(float(sweep["value"])) != round_sweep_value(
        float(sweep["base_value"])
    ):
        raise ValueError("canonical base must use the exact parent base value")
    parent_scene_id = str(sweep["parent_scene_id"])
    result["scene_id"] = f"{parent_scene_id}__base"
    result["dataset_stage"] = "object_physics_canonical_base"
    result["sweep"] = {
        "schema_version": sweep["schema_version"],
        "kind": "base",
        "mode": "one_factor_reference",
        "parent_scene_id": parent_scene_id,
        "parent_metadata_path": sweep["parent_metadata_path"],
        "parent_metadata_sha256": sweep["parent_metadata_sha256"],
        "target_object_id": None,
        "target_object_index": None,
        "parameter": None,
        "value": None,
        "source_schema_version": sweep["source_schema_version"],
        "resolved_object_physics": copy.deepcopy(sweep["resolved_object_physics"]),
        "resolved_state_policy": sweep["resolved_state_policy"],
        "config_path": sweep["config_path"],
        "config_sha256": sweep["config_sha256"],
        "initial_state_policy": sweep["initial_state_policy"],
        "visual_policy": sweep["visual_policy"],
    }
    attach_object_identity(result)
    return result


def collect_inputs(
    root: Path,
    base: list[Path] | None,
    base_dir: Path | None,
    base_manifest: Path | None,
) -> list[Path]:
    if base_manifest is not None and (base or base_dir is not None):
        raise ValueError("choose --base-manifest or direct base inputs, not both")

    def under_root(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    paths: list[Path] = []
    if base_manifest is not None:
        base_manifest = under_root(base_manifest)
        manifest = load_json(base_manifest)
        records = manifest.get("records")
        if not isinstance(records, list):
            raise ValueError("base manifest has no records list")
        declared_count = int(manifest.get("sample_count", len(records)))
        if declared_count != len(records):
            raise ValueError("base manifest sample count does not match records")
        for record in records:
            metadata_path = record.get("metadata_path")
            if not metadata_path:
                raise ValueError("base manifest record has no metadata_path")
            paths.append(root / str(metadata_path))
    if base:
        paths.extend(under_root(path) for path in base)
    if base_dir is not None:
        base_dir = under_root(base_dir)
        paths.extend(sorted(base_dir.rglob("metadata.json")))
    unique = sorted({path.resolve() for path in paths})
    return [path for path in unique if path.is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base", type=Path, action="append")
    parser.add_argument("--base-dir", type=Path)
    parser.add_argument(
        "--base-manifest",
        type=Path,
        help="Read the exact frozen base records instead of scanning a directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--axis", choices=["mass_kg", "contact_friction", "contact_restitution"], action="append")
    parser.add_argument(
        "--target-object-id",
        action="append",
        help="Restrict derivation to one or more object ids; defaults to every object.",
    )
    parser.add_argument(
        "--target-object-index",
        action="append",
        type=int,
        help="Restrict derivation to object indices; defaults to every object.",
    )
    parser.add_argument("--max-inputs", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (root / args.config).resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir.is_absolute()
        else (root / args.output_dir).resolve()
    )
    if not args.dry_run:
        validate_output_dir(root, output_dir)
    config = load_json(config_path)
    axes = args.axis or list(config["axes"])
    for axis in axes:
        if axis not in config["axes"]:
            raise ValueError(f"unsupported sweep axis: {axis}")
    configured_base_axis = str(config.get("canonical_base_axis", axes[0]))
    canonical_base_axis = (
        configured_base_axis if configured_base_axis in axes else axes[0]
    )
    inputs = collect_inputs(root, args.base, args.base_dir, args.base_manifest)
    if not inputs:
        raise ValueError("no base metadata inputs were provided")
    if args.max_inputs is not None:
        if args.max_inputs <= 0:
            raise ValueError("--max-inputs must be positive")
        inputs = inputs[: args.max_inputs]
    if args.target_object_id and args.target_object_index:
        raise ValueError("choose --target-object-id or --target-object-index, not both")
    resolved_base_manifest = None
    base_manifest_document = None
    if args.base_manifest is not None:
        resolved_base_manifest = (
            args.base_manifest.resolve()
            if args.base_manifest.is_absolute()
            else (root / args.base_manifest).resolve()
        )
        base_manifest_document = load_json(resolved_base_manifest)
    prior_provenance = resolve_prior_provenance(
        root, config, base_manifest_document
    )
    profiles, registry = _load_prior_indexes(root, config)

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for base_path in inputs:
        base = load_json(base_path)
        target_indices = _target_object_indices(
            base,
            target_object_ids=args.target_object_id,
            target_object_indices=args.target_object_index,
        )
        canonical_emitted = False
        for target_object_index in target_indices:
            for axis in axes:
                for level_index in range(int(config["axes"][axis]["level_count"])):
                    try:
                        derived = derive_one(
                            base,
                            base_path,
                            root,
                            config,
                            config_path,
                            axis,
                            level_index,
                            profiles,
                            registry,
                            target_object_index=target_object_index,
                        )
                    except (IndexError, KeyError, TypeError, ValueError) as error:
                        if not args.dry_run:
                            raise
                        errors.append(
                            {
                                "base": str(base_path.relative_to(root)),
                                "target_object_index": str(target_object_index),
                                "axis": axis,
                                "level_index": str(level_index),
                                "error": str(error),
                            }
                        )
                        continue
                    is_base_level = level_index == int(
                        derived["sweep"]["base_level_index"]
                    )
                    if is_base_level:
                        if axis != canonical_base_axis or canonical_emitted:
                            continue
                        derived = normalize_canonical_base(derived)
                        canonical_emitted = True
                    if args.dry_run:
                        records.append(
                            {
                                "parent": str(base_path.relative_to(root)),
                                "scene_id": derived["scene_id"],
                                "target_object_id": derived["sweep"]["target_object_id"],
                                "target_object_index": derived["sweep"][
                                    "target_object_index"
                                ],
                                "source_schema_version": derived["sweep"][
                                    "source_schema_version"
                                ],
                                "kind": derived["sweep"]["kind"],
                                "axis": derived["sweep"].get("axis"),
                                "level_index": derived["sweep"].get("level_index"),
                                "value": derived["sweep"].get("value"),
                            }
                        )
                        continue
                    relative_base = base_path.relative_to(root)
                    target = (
                        output_dir
                        / str(base["scene_id"])
                        / derived["scene_id"]
                        / "metadata.json"
                    )
                    write_json(target, derived)
                    records.append(
                        {
                            "path": str(target.relative_to(root)),
                            "metadata_sha256": sha256(target),
                            "parent": str(relative_base),
                            "scene_id": derived["scene_id"],
                            "target_object_id": derived["sweep"]["target_object_id"],
                            "target_object_index": derived["sweep"][
                                "target_object_index"
                            ],
                            "source_schema_version": derived["sweep"][
                                "source_schema_version"
                            ],
                            "kind": derived["sweep"]["kind"],
                            "axis": derived["sweep"].get("axis"),
                            "level_index": derived["sweep"].get("level_index"),
                            "value": derived["sweep"].get("value"),
                        }
                    )
        if not canonical_emitted:
            raise ValueError(
                f"canonical base was not emitted for {base_path.relative_to(root)}"
            )

    schema_counts: dict[str, int] = {}
    for record in records:
        schema = str(record["source_schema_version"])
        schema_counts[schema] = schema_counts.get(schema, 0) + 1
    base_manifest_provenance = None
    if resolved_base_manifest is not None:
        base_manifest_provenance = {
            "path": str(resolved_base_manifest.relative_to(root)),
            "sha256": sha256(resolved_base_manifest),
        }
    manifest = {
        "schema_version": "physweep_physics_sweep_manifest_v2",
        "dataset_id": output_dir.parent.name if output_dir.name == "metadata" else output_dir.name,
        "config": {
            "path": str(config_path.relative_to(root)),
            "sha256": sha256(config_path),
        },
        "implementation": {
            "path": str(Path(__file__).resolve().relative_to(root)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "prior_sources": prior_provenance,
        "base_count": len(inputs),
        "base_manifest": base_manifest_provenance,
        "target_policy": "one_factor_per_object",
        "target_object_id_filter": args.target_object_id,
        "target_object_index_filter": args.target_object_index,
        "axis_count": len(axes),
        "canonical_base_axis": canonical_base_axis,
        "derived_count": len(records),
        "sample_count": len(records),
        "source_schema_counts": schema_counts,
        "error_count": len(errors),
        "records": records,
        "errors": errors,
    }
    if not args.dry_run:
        write_json(output_dir / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "base_count": len(inputs),
                "derived_count": len(records),
                "error_count": len(errors),
                "dry_run": bool(args.dry_run),
                "errors": errors[:20],
            },
            indent=2,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
