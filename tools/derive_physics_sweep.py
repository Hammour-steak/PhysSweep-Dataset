#!/usr/bin/env python3
"""Derive immutable one-factor PyBullet sweep metadata from frozen base scenes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from object_identity_contract import attach_object_identity
except ModuleNotFoundError:  # import when the script is loaded as tools.* in tests
    from tools.object_identity_contract import attach_object_identity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/physics_sweep.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested_get(value: dict[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if "[" in part:
            name, index_text = part[:-1].split("[", 1)
            current = current[name][int(index_text)]
        else:
            current = current[part]
    return current


def _nested_set(value: dict[str, Any], path: str, replacement: Any) -> None:
    parts = path.split(".")
    current: Any = value
    for part in parts[:-1]:
        if "[" in part:
            name, index_text = part[:-1].split("[", 1)
            current = current[name][int(index_text)]
        else:
            current = current[part]
    final = parts[-1]
    if "[" in final:
        name, index_text = final[:-1].split("[", 1)
        current[name][int(index_text)] = replacement
    else:
        current[final] = replacement


def _objects(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    objects = metadata.get("simulation", {}).get("objects", [])
    if not isinstance(objects, list):
        raise ValueError("simulation.objects must be a list")
    return objects


def _object_id(obj: dict[str, Any], index: int) -> str:
    """Return the required stable object binding."""
    value = obj.get("object_id")
    if value in (None, ""):
        raise ValueError(f"simulation.objects[{index}] has no object_id")
    return str(value)


def _object_field_path(index: int, object_field: str) -> str:
    return f"simulation.objects[{index}].{object_field}"


def _target_object_indices(
    metadata: dict[str, Any],
    target_object_ids: list[str] | None = None,
    target_object_indices: list[int] | None = None,
) -> list[int]:
    objects = _objects(metadata)
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


def _record_ids(metadata: dict[str, Any], object_index: int) -> set[str]:
    obj = _objects(metadata)[object_index]
    visual = obj.get("visual_profile", {})
    return {
        str(obj.get("semantic_type", "")),
        str(visual.get("id", "")),
        str(visual.get("asset_id", "")),
    } - {""}


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


def _mass_bounds(
    metadata: dict[str, Any],
    profiles: dict[str, Any],
    registry: dict[str, Any],
    object_index: int,
) -> list[float] | None:
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
    except (KeyError, TypeError, ValueError):
        return None
    normal_length = math.sqrt(sum(value * value for value in normal))
    if normal_length <= 1.0e-9 or minimum_distance <= 1.0e-6:
        return None
    normal = [value / normal_length for value in normal]
    normal_velocity = sum(v * n for v, n in zip(velocity, normal))
    tangent_velocity = [
        v - normal_velocity * n for v, n in zip(velocity, normal)
    ]
    speed = math.sqrt(sum(value * value for value in tangent_velocity))
    if speed <= 1.0e-6:
        return None
    direction = [value / speed for value in tangent_velocity]
    gravity_tangent_component = sum(g * d for g, d in zip(gravity, direction))
    gravity_tangent = [
        g - sum(g2 * n for g2, n in zip(gravity, normal)) * n
        for g, n in zip(gravity, normal)
    ]
    normal_acceleration = math.sqrt(
        sum((g - gt) ** 2 for g, gt in zip(gravity, gravity_tangent))
    )
    if normal_acceleration <= 1.0e-6:
        return None
    theoretical_max = (
        speed * speed / (2.0 * minimum_distance) + gravity_tangent_component
    ) / normal_acceleration
    static_hold_limit = 0.0
    if bool(expected.get("must_reverse_downhill")):
        slope_angle = math.radians(
            abs(float(support_frame.get("slope_angle_degrees", 0.0)))
        )
        static_hold_limit = math.tan(slope_angle)
    configured_low, configured_high = [
        float(value) for value in axis_rules["domain"]
    ]
    if "transition_margin" in axis_rules:
        # v3 deliberately crosses the motion transition. A high-friction
        # sample may stop before the planned event; the trajectory audit
        # records that as an advisory while retaining hard physical checks.
        transition_threshold = theoretical_max
        if static_hold_limit > 1.0e-6:
            transition_threshold = max(transition_threshold, static_hold_limit)
        margin = float(axis_rules["transition_margin"])
        feasible_high = max(base_value, transition_threshold * margin)
    else:
        # Preserve the conservative endpoint behavior of v1/v2 metadata.
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


def _round_value(value: float) -> float:
    return round(float(value), 6)


def _allowed_domain(
    base_value: float,
    axis_rules: dict[str, Any],
    mass_bounds: list[float] | None,
    axis: str,
    domain_override: list[float] | None,
) -> list[float]:
    if axis == "mass_kg":
        if mass_bounds is None:
            return [base_value * 0.5, base_value * 2.0]
        return [float(mass_bounds[0]), float(mass_bounds[1])]
    if domain_override is not None:
        return [float(domain_override[0]), float(domain_override[1])]
    return [float(value) for value in axis_rules["domain"]]


def _resolve_sweep_domain(
    base_value: float,
    axis_rules: dict[str, Any],
    allowed_domain: list[float],
) -> list[float]:
    allowed_low, allowed_high = allowed_domain
    policy = axis_rules.get("range_policy", {})
    mode = str(policy.get("mode", "global"))
    if mode == "relative_multipliers":
        low = max(allowed_low, base_value * float(policy["lower_multiplier"]))
        high = min(allowed_high, base_value * float(policy["upper_multiplier"]))
    elif mode == "linear_symmetric_span":
        span = max(
            float(policy["minimum_absolute_span"]),
            abs(base_value) * float(policy["relative_span"]),
        )
        low = max(allowed_low, base_value - span)
        high = min(allowed_high, base_value + span)
    elif mode == "global":
        low, high = allowed_low, allowed_high
    else:
        raise ValueError(f"unsupported range policy: {mode}")

    if low < high and low <= base_value <= high:
        return [low, high]
    return [allowed_low, allowed_high]


def _sweep_values(
    base_value: float,
    axis_rules: dict[str, Any],
    mass_bounds: list[float] | None,
    axis: str,
    domain_override: list[float] | None = None,
    endpoint_policy: dict[str, Any] | None = None,
) -> list[float]:
    allowed_domain = _allowed_domain(
        base_value,
        axis_rules,
        mass_bounds,
        axis,
        domain_override,
    )
    low, high = _resolve_sweep_domain(base_value, axis_rules, allowed_domain)
    if not (math.isfinite(base_value) and low < high):
        raise ValueError(f"invalid sweep domain for {axis}: {low}, {high}")
    if not (low <= base_value <= high):
        raise ValueError(
            f"base value {base_value} is outside the declared {axis} domain "
            f"[{low}, {high}]"
        )

    endpoint_policy = endpoint_policy or {}
    fixed_positions = [
        float(value)
        for value in endpoint_policy.get(
            "normalized_positions", axis_rules["level_positions"]
        )
    ]
    expected_count = int(endpoint_policy.get("level_count", axis_rules["level_count"]))
    if len(fixed_positions) != expected_count:
        raise ValueError(f"{axis} has inconsistent level count")
    scale = axis_rules["scale"]
    if scale == "log" and (low <= 0.0 or high <= 0.0 or base_value <= 0.0):
        raise ValueError(f"log sweep requires positive {axis} bounds")
    if scale not in {"linear", "log"}:
        raise ValueError(f"unsupported sweep scale: {scale}")

    def interpolate(start: float, end: float, fraction: float) -> float:
        if scale == "log":
            return start * math.exp(fraction * math.log(end / start))
        return start + fraction * (end - start)

    base_policy = endpoint_policy.get(
        "base_value_policy", "preserve_exactly_at_nearest_position"
    )
    middle_policy = base_policy == "preserve_exactly_at_middle_position"
    middle_index = expected_count // 2
    is_interior_base = low < base_value < high

    if middle_policy and expected_count % 2 == 1 and is_interior_base:
        if middle_index == 0:
            raise ValueError(f"{axis} middle policy needs at least three levels")
        if (
            fixed_positions[0] != 0.0
            or fixed_positions[middle_index] != 0.5
            or fixed_positions[-1] != 1.0
            or any(left >= right for left, right in zip(fixed_positions, fixed_positions[1:]))
        ):
            raise ValueError(
                f"{axis} middle policy requires ordered positions with a 0.5 center"
            )
        values = []
        for index, position in enumerate(fixed_positions):
            if index <= middle_index:
                fraction = position / fixed_positions[middle_index]
                value = interpolate(low, base_value, fraction)
            else:
                fraction = (
                    position - fixed_positions[middle_index]
                ) / (fixed_positions[-1] - fixed_positions[middle_index])
                value = interpolate(base_value, high, fraction)
            values.append(_round_value(value))
        values[middle_index] = _round_value(base_value)
    else:
        if middle_policy and is_interior_base:
            raise ValueError(
                f"{axis} middle policy requires an odd number of sweep levels"
            )
        if middle_policy and not is_interior_base:
            edge_policy = endpoint_policy.get("edge_policy")
            raise ValueError(
                f"{axis} base cannot occupy the middle level under edge policy "
                f"{edge_policy!r}; widen the declared domain or reject this base"
            )

        if scale == "log":
            base_position = math.log(base_value / low) / math.log(high / low)
        else:
            base_position = (base_value - low) / (high - low)
        base_position = min(1.0, max(0.0, base_position))
        nearest = min(
            range(len(fixed_positions)),
            key=lambda index: abs(fixed_positions[index] - base_position),
        )
        fixed_positions[nearest] = base_position
        positions = sorted(fixed_positions)
        values = [
            _round_value(interpolate(low, high, position))
            for position in positions
        ]
        closest = min(
            range(len(values)), key=lambda index: abs(values[index] - base_value)
        )
        values[closest] = _round_value(base_value)
    if len(set(values)) != len(values):
        raise ValueError(f"{axis} domain is too narrow for five distinct levels")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError(f"{axis} values are not strictly ordered after rounding")
    return values


def validate_base(metadata: dict[str, Any], max_objects: int = 3) -> None:
    if metadata.get("schema_version") != "physweep_pybullet_rigid_metadata_v1":
        raise ValueError("input is not a current PyBullet rigid base metadata record")
    dataset_stage = str(metadata.get("dataset_stage", ""))
    if not dataset_stage.endswith("_base_candidate"):
        raise ValueError("input metadata is not a base candidate")
    if "sweep" in metadata:
        raise ValueError("input already contains a sweep binding")
    objects = _objects(metadata)
    if not objects or len(objects) > max_objects:
        raise ValueError(
            f"the object-bound rigid sweep supports 1..{max_objects} dynamic objects"
        )
    identifiers = [_object_id(obj, index) for index, obj in enumerate(objects)]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("dynamic object identifiers must be unique")
    for obj in objects:
        if obj.get("body_model") != "rigid_body":
            raise ValueError("the active rigid sweep supports rigid_body records only")
        material = obj.get("material", {})
        for field in ("mass_kg", "contact_friction", "contact_restitution"):
            if field not in material:
                raise ValueError(f"base metadata lacks runtime material field: {field}")


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
    validate_base(base, max_objects=max_objects)
    objects = _objects(base)
    if target_object_index < 0 or target_object_index >= len(objects):
        raise IndexError(f"invalid target object index: {target_object_index}")
    target_object_id = _object_id(objects[target_object_index], target_object_index)
    axis_rules = config["axes"][axis]
    object_field_value = axis_rules.get("object_field")
    if object_field_value is None:
        object_field_value = str(axis_rules["metadata_field"]).replace(
            "simulation.objects[0].", ""
        )
    object_field = str(object_field_value)
    field = _object_field_path(target_object_index, object_field)
    base_value = float(_nested_get(base, field))
    mass_bounds = _mass_bounds(base, profiles, registry, target_object_index)
    domain_override = _friction_domain(
        base, axis_rules, base_value, axis, target_object_index
    )
    allowed_domain = _allowed_domain(
        base_value,
        axis_rules,
        mass_bounds,
        axis,
        domain_override,
    )
    resolved_domain = _resolve_sweep_domain(
        base_value,
        axis_rules,
        allowed_domain,
    )
    values = _sweep_values(
        base_value,
        axis_rules,
        mass_bounds,
        axis,
        domain_override=domain_override,
        endpoint_policy=config.get("endpoint_policy"),
    )
    if level_index < 0 or level_index >= len(values):
        raise IndexError(f"invalid sweep level {level_index} for {axis}")
    rounded_base = _round_value(base_value)
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
    derived["scene_id"] = (
        f"{base['scene_id']}__sweep_{target_object_id}_{axis}_{level_index:02d}"
    )
    derived["dataset_stage"] = "object_physics_sweep_candidate"
    for index, obj in enumerate(derived["simulation"]["objects"]):
        obj.setdefault("object_id", _object_id(obj, index))
    _nested_set(derived, field, values[level_index])
    resolved_object_physics = [
        {
            "object_id": _object_id(obj, index),
            "object_index": index,
            "material": copy.deepcopy(obj.get("material", {})),
        }
        for index, obj in enumerate(derived["simulation"]["objects"])
    ]
    derived["sweep"] = {
        "schema_version": config["version"],
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
        "domain": [_round_value(value) for value in resolved_domain],
        "allowed_domain": [_round_value(value) for value in allowed_domain],
        "object_field": object_field,
        "overridden_field": field,
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


def collect_inputs(base: list[Path] | None, base_dir: Path | None) -> list[Path]:
    paths: list[Path] = []
    if base:
        paths.extend(base)
    if base_dir is not None:
        paths.extend(sorted(base_dir.rglob("metadata.json")))
    unique = sorted({path.resolve() for path in paths})
    return [path for path in unique if path.is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base", type=Path, action="append")
    parser.add_argument("--base-dir", type=Path)
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
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    config = load_json(config_path)
    axes = args.axis or list(config["axes"])
    for axis in axes:
        if axis not in config["axes"]:
            raise ValueError(f"unsupported sweep axis: {axis}")
    configured_base_axis = str(config.get("canonical_base_axis", axes[0]))
    canonical_base_axis = (
        configured_base_axis if configured_base_axis in axes else axes[0]
    )
    inputs = collect_inputs(args.base, args.base_dir)
    if not inputs:
        raise ValueError("no base metadata inputs were provided")
    if args.max_inputs is not None:
        if args.max_inputs <= 0:
            raise ValueError("--max-inputs must be positive")
        inputs = inputs[: args.max_inputs]
    if args.target_object_id and args.target_object_index:
        raise ValueError("choose --target-object-id or --target-object-index, not both")
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
                    if (
                        axis != canonical_base_axis
                        and level_index == int(derived["sweep"]["base_level_index"])
                    ):
                        continue
                    if args.dry_run:
                        records.append(
                            {
                                "parent": str(base_path.relative_to(root)),
                                "scene_id": derived["scene_id"],
                                "target_object_id": derived["sweep"]["target_object_id"],
                                "target_object_index": target_object_index,
                                "axis": axis,
                                "level_index": level_index,
                                "value": derived["sweep"]["value"],
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
                            "parent": str(relative_base),
                            "scene_id": derived["scene_id"],
                            "target_object_id": derived["sweep"]["target_object_id"],
                            "target_object_index": target_object_index,
                            "axis": axis,
                            "level_index": level_index,
                            "value": derived["sweep"]["value"],
                        }
                    )

    manifest = {
        "schema_version": "physweep_physics_sweep_manifest_v1",
        "dataset_id": output_dir.parent.name if output_dir.name == "metadata" else output_dir.name,
        "config": {
            "path": str(config_path.relative_to(root)),
            "sha256": sha256(config_path),
        },
        "base_count": len(inputs),
        "target_policy": "one_factor_per_object",
        "target_object_id_filter": args.target_object_id,
        "target_object_index_filter": args.target_object_index,
        "axis_count": len(axes),
        "canonical_base_axis": canonical_base_axis,
        "derived_count": len(records),
        "sample_count": len(records),
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
