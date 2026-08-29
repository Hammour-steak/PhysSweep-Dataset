#!/usr/bin/env python3
"""Shared object identity contract for PhysSweep metadata and annotations.

The contract deliberately keeps semantic identity separate from array order:
``object_id`` is the only cross-modal join key.  Text mentions, trajectories,
instance masks, and one-factor sweeps all refer to that key.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


OBJECT_IDENTITY_SCHEMA_VERSION = "physweep_object_identity_v1"


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def require_simulation_objects(
    metadata: Mapping[str, Any],
    supported_counts: Iterable[int],
    consumer: str,
) -> list[dict[str, Any]]:
    """Return explicit simulation objects after checking an adapter capability."""

    simulation = metadata.get("simulation")
    records = simulation.get("objects") if isinstance(simulation, Mapping) else None
    if not isinstance(records, list) or any(
        not isinstance(record, dict) for record in records
    ):
        raise ValueError(f"{consumer} requires simulation.objects records")
    allowed = tuple(sorted({int(value) for value in supported_counts}))
    if len(records) not in allowed:
        raise ValueError(
            f"{consumer} supports dynamic object counts {list(allowed)}, "
            f"received {len(records)}"
        )
    return records


def require_single_simulation_object(
    metadata: Mapping[str, Any], consumer: str
) -> dict[str, Any]:
    """Return the sole object for an explicitly one-object adapter."""

    return require_simulation_objects(metadata, (1,), consumer)[0]


def canonical_object_id(record: Mapping[str, Any], index: int) -> str:
    """Return the required cross-modal object identity."""
    value = record.get("object_id")
    if value is None or not str(value).strip():
        raise ValueError(f"object record {index} has no object_id")
    return str(value)


def _simulation_objects(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    simulation = _as_mapping(metadata.get("simulation"))
    records = _as_list(simulation.get("objects"))
    if records:
        return [dict(record) for record in records if isinstance(record, Mapping)]

    # Billiards uses initial_states rather than simulation.objects.
    physics = _as_mapping(metadata.get("physics"))
    records = _as_list(physics.get("initial_states"))
    if records:
        return [dict(record) for record in records if isinstance(record, Mapping)]

    # Asset-review scenes use a compact physics block instead of simulation.objects.
    assets = _as_mapping(metadata.get("assets"))
    dynamic_asset_id = assets.get("dynamic_asset_id")
    if dynamic_asset_id:
        return [
            {
                "object_id": "object_a",
                "semantic_type": (
                    metadata.get("semantic_type")
                    or metadata.get("object_type")
                    or metadata.get("dynamic_asset_name")
                    or "dynamic asset"
                ),
                "asset_id": str(dynamic_asset_id),
                "role": "dynamic",
            }
        ]
    return []


def _is_dynamic(record: Mapping[str, Any]) -> bool:
    role = str(record.get("role", record.get("body_model", "dynamic"))).lower()
    return not bool(record.get("is_dynamic") is False or role in {"static", "support", "environment"})


def _humanize(value: Any) -> str:
    text = str(value or "object").strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower() or "object"


def _semantic_label(record: Mapping[str, Any], index: int) -> str:
    visual = _as_mapping(record.get("visual_profile"))
    for key in (
        "semantic_type",
        "semantic_label",
        "object_type",
        "label",
        "asset_id",
    ):
        if record.get(key):
            return _humanize(record[key])
    for key in ("asset_id",):
        if visual.get(key):
            return _humanize(visual[key])
    return f"object {index + 1}"


def _motion_phrase(metadata: Mapping[str, Any]) -> str:
    semantic = _as_mapping(metadata.get("semantic_sampling"))
    dimensions = _as_mapping(semantic.get("five_dimensions"))
    motion = _as_mapping(dimensions.get("motion"))
    family = str(
        motion.get("family")
        or semantic.get("motion_family")
        or _as_mapping(metadata.get("semantics")).get("motion")
        or _as_mapping(metadata.get("physics")).get("motion_profile")
        or _as_mapping(metadata.get("semantics")).get("profile")
        or "moves"
    )
    phrases = {
        "drop_fall": "falls under gravity",
        "edge_fall": "falls from an edge",
        "projectile": "travels through the air",
        "arc_projectile": "follows an arcing trajectory",
        "slide_push": "slides across the support",
        "roll_or_slide": "rolls or slides across the support",
        "slope_slide_down": "moves down the slope",
        "slope_slide_up": "moves up the slope and returns",
        "wall_impact": "impacts a wall",
        "ramp_to_flat": "moves from the ramp onto the flat support",
        "resting_push": "is pushed across the support",
        "diagonal_push": "is pushed diagonally across the support",
        "free_roll": "rolls freely across the support",
        "single_ball_free_roll": "rolls freely across the support",
        "single_ball_rail_rebound": "rebounds from the rail",
        "three_ball_collision": "collides with the other balls",
        "surface_hit_rest_2obj": "collide after one moves toward the other",
        "surface_head_on_2obj": "collide while moving toward each other",
        "surface_crossing_2obj": "collide along crossing surface paths",
        "surface_catch_up_2obj": "collide while moving in the same direction",
        "air_drop_hit_supported_2obj": "collide as one falls toward the other",
        "air_projectile_hit_supported_2obj": (
            "collide as one travels through the air"
        ),
        "surface_single_independent_2obj": "do not contact each other",
        "surface_dual_independent_2obj": "do not contact each other",
        "air_supported_independent_2obj": "do not contact each other",
    }
    return phrases.get(family, f"moves in the {family.replace('_', ' ')} scenario")


def _support_label(metadata: Mapping[str, Any]) -> str | None:
    simulation = _as_mapping(metadata.get("simulation"))
    support = _as_mapping(simulation.get("support"))
    if support.get("label"):
        return _humanize(support["label"])
    assets = _as_mapping(metadata.get("assets"))
    if assets.get("support_asset_id"):
        return _humanize(assets["support_asset_id"])
    return None


def _trajectory_keys(object_id: str) -> dict[str, str]:
    return {
        "position_m": f"{object_id}__position_m",
        "quaternion_wxyz": f"{object_id}__quaternion_wxyz",
        "linear_velocity_m_s": f"{object_id}__linear_velocity_m_s",
        "angular_velocity_rad_s": f"{object_id}__angular_velocity_rad_s",
    }


def _is_billiards_layout(metadata: Mapping[str, Any]) -> bool:
    simulation = _as_mapping(metadata.get("simulation"))
    physics = _as_mapping(metadata.get("physics"))
    return not _as_list(simulation.get("objects")) and bool(
        _as_list(physics.get("initial_states"))
    )


def _source_asset_id(record: Mapping[str, Any]) -> str | None:
    visual = _as_mapping(record.get("visual_profile"))
    for value in (record.get("asset_id"), visual.get("asset_id")):
        if value:
            return str(value)
    return None


def _normalise_records(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = _simulation_objects(metadata)
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(source):
        object_id = canonical_object_id(record, index)
        if object_id in seen:
            raise ValueError(f"duplicate object_id: {object_id}")
        seen.add(object_id)
        result.append(
            {
                "object_id": object_id,
                "object_index": index,
                "role": "dynamic" if _is_dynamic(record) else "static",
                "semantic_label": _semantic_label(record, index),
                "asset_id": _source_asset_id(record),
            }
        )
    if not result:
        raise ValueError("metadata contains no object records for identity binding")
    return result


def build_object_identity(
    metadata: Mapping[str, Any],
    *,
    trajectory_path: str | None = None,
    mask_path: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic cross-modal identity block without file I/O."""
    records = _normalise_records(metadata)
    dynamic = [record for record in records if record["role"] == "dynamic"]
    if not dynamic:
        raise ValueError("object identity requires at least one dynamic object")

    mentions = []
    trajectory_objects: dict[str, Any] = {}
    mask_objects: dict[str, Any] = {}
    object_records = []
    billiards_layout = _is_billiards_layout(metadata)
    for mask_instance_id, record in enumerate(records, start=1):
        object_id = str(record["object_id"])
        mention = f"the {record['semantic_label']}"
        trajectory_key = object_id if record["role"] == "dynamic" else None
        trajectory_array_index = (
            int(record["object_index"]) if billiards_layout and record["role"] == "dynamic" else None
        )
        object_records.append(
            {
                **record,
                "text_mention": mention,
                "trajectory_key": trajectory_key,
                "trajectory_array_index": trajectory_array_index,
                "mask_key": object_id,
                "mask_instance_id": mask_instance_id,
            }
        )
        mentions.append(
            {
                "object_id": object_id,
                "text": mention,
                "role": record["role"],
            }
        )
        mask_objects[object_id] = {
            "instance_id": mask_instance_id,
            "key": object_id,
        }
        if record["role"] == "dynamic":
            if billiards_layout:
                trajectory_objects[object_id] = {
                    "array_index": int(record["object_index"]),
                    "position_m": "position_m",
                    "quaternion_xyzw": "quaternion_xyzw",
                    "linear_velocity_m_s": "linear_velocity_m_s",
                }
            else:
                trajectory_objects[object_id] = {
                    "position_m": _trajectory_keys(object_id)["position_m"],
                    "quaternion_wxyz": _trajectory_keys(object_id)["quaternion_wxyz"],
                    "linear_velocity_m_s": _trajectory_keys(object_id)["linear_velocity_m_s"],
                    "angular_velocity_rad_s": _trajectory_keys(object_id)["angular_velocity_rad_s"],
                }

    labels = [str(record["semantic_label"]) for record in dynamic]
    subject = " and ".join(f"the {label}" for label in labels)
    support = _support_label(metadata)
    caption = f"{subject} {_motion_phrase(metadata)}"
    if support:
        caption += f" on the {support}"
    caption += "."

    identity = {
        "schema_version": OBJECT_IDENTITY_SCHEMA_VERSION,
        "object_order": [str(record["object_id"]) for record in records],
        "objects": object_records,
        "text": {
            "caption": caption,
            "object_mentions": mentions,
            "template_version": "physweep_object_caption_v1",
        },
        "trajectory": {
            "format": "npz",
            "layout": "frame_object_channel" if billiards_layout else "one_array_per_object",
            "path": trajectory_path,
            "path_policy": "bound_simulation_record" if trajectory_path is None else "metadata_relative",
            "objects": trajectory_objects,
        },
        "instance_masks": {
            "encoding": "rgba_alpha_antialiased_silhouette_mask",
            "path": mask_path,
            "path_policy": "bound_render_manifest" if mask_path is None else "metadata_relative",
            "path_layout": "object_id_subdirectories",
            "filename_pattern": "frame_{frame:04d}.png",
            "objects": mask_objects,
        },
    }
    sweep = _as_mapping(metadata.get("sweep"))
    target = sweep.get("target_object_id")
    if target is not None:
        identity["sweep_target"] = {
            "object_id": str(target),
            "validated": str(target) in set(identity["object_order"]),
        }
    return identity


def attach_object_identity(
    metadata: dict[str, Any],
    *,
    trajectory_path: str | None = None,
    mask_path: str | None = None,
) -> dict[str, Any]:
    """Attach the canonical cross-modal identity contract."""
    metadata["object_identity"] = build_object_identity(
        metadata,
        trajectory_path=trajectory_path,
        mask_path=mask_path,
    )
    sweep_target = metadata["object_identity"].get("sweep_target")
    if sweep_target is not None and not bool(sweep_target.get("validated")):
        raise ValueError(
            "sweep target_object_id is not present in object_identity.object_order: "
            f"{sweep_target.get('object_id')}"
        )
    return metadata


def validate_object_identity(
    metadata: Mapping[str, Any],
    *,
    trajectory_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate the identity block and optionally validate actual NPZ keys."""
    errors: list[str] = []
    identity = _as_mapping(metadata.get("object_identity"))
    if identity.get("schema_version") != OBJECT_IDENTITY_SCHEMA_VERSION:
        errors.append("missing or unsupported object_identity.schema_version")
    records = _as_list(identity.get("objects"))
    ids = [str(_as_mapping(record).get("object_id", "")) for record in records]
    if not ids or any(not value for value in ids):
        errors.append("object_identity.objects must contain non-empty object_id values")
    if len(ids) != len(set(ids)):
        errors.append("object_identity.objects contains duplicate object_id values")
    order = [str(value) for value in _as_list(identity.get("object_order"))]
    if order != ids:
        errors.append("object_order does not match object record order")

    dynamic_ids = [
        str(_as_mapping(record).get("object_id"))
        for record in records
        if str(_as_mapping(record).get("role")) == "dynamic"
    ]
    mentions = _as_list(_as_mapping(identity.get("text")).get("object_mentions"))
    mentioned_ids = [str(_as_mapping(item).get("object_id", "")) for item in mentions]
    if set(mentioned_ids) != set(ids):
        errors.append("text.object_mentions must cover every identity object exactly once")
    if len(mentioned_ids) != len(set(mentioned_ids)):
        errors.append("text.object_mentions contains duplicate object_id values")

    trajectory = _as_mapping(identity.get("trajectory"))
    trajectory_objects = _as_mapping(trajectory.get("objects"))
    if set(str(key) for key in trajectory_objects) != set(dynamic_ids):
        errors.append("trajectory.objects must cover exactly the dynamic object IDs")
    if trajectory_keys is not None:
        available = {str(key) for key in trajectory_keys}
        for object_id in dynamic_ids:
            declared = _as_mapping(trajectory_objects.get(object_id))
            required_fields = [("position_m",), ("quaternion_wxyz", "quaternion_xyzw")]
            for fields in required_fields:
                if not any(declared.get(field) in available for field in fields):
                    errors.append(
                        f"trajectory key missing for {object_id}: one of {fields}"
                    )

    masks = _as_mapping(identity.get("instance_masks"))
    mask_objects = _as_mapping(masks.get("objects"))
    if set(str(key) for key in mask_objects) != set(ids):
        errors.append("instance_masks.objects must cover every identity object")
    instance_ids = [int(_as_mapping(value).get("instance_id", 0)) for value in mask_objects.values()]
    if any(value <= 0 for value in instance_ids) or len(instance_ids) != len(set(instance_ids)):
        errors.append("instance mask IDs must be positive and unique")

    sweep = _as_mapping(metadata.get("sweep"))
    target = sweep.get("target_object_id")
    if target is not None and str(target) not in set(ids):
        errors.append(f"sweep target_object_id is not in object_order: {target}")

    if errors:
        raise ValueError("; ".join(errors))
    return {
        "schema_version": OBJECT_IDENTITY_SCHEMA_VERSION,
        "object_count": len(ids),
        "dynamic_object_count": len(dynamic_ids),
        "object_ids": ids,
        "dynamic_object_ids": dynamic_ids,
        "trajectory_validated": trajectory_keys is not None,
    }
