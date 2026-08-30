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

    # Published samples store resolved objects under physics.objects.
    records = _as_list(physics.get("objects"))
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


def _public_object_label(value: Any) -> str:
    """Return a human label while keeping source ids in structured provenance."""

    label = _humanize(value)
    if re.fullmatch(r"physassets\s+\d+", label):
        return "object"
    label = re.sub(r"^physassets\s+\d+\s+", "", label)
    label = re.sub(r"^physassets\s+", "", label)
    aliases = {
        "remote": "remote control",
        "drink box": "drink carton",
    }
    return aliases.get(label, label) or "object"


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
            return _public_object_label(record[key])
    for key in ("asset_id",):
        if visual.get(key):
            return _public_object_label(visual[key])
    return f"object {index + 1}"


def _motion_family(metadata: Mapping[str, Any]) -> str:
    semantic = _as_mapping(metadata.get("semantic_sampling"))
    dimensions = _as_mapping(semantic.get("five_dimensions"))
    motion = _as_mapping(dimensions.get("motion"))
    semantics = _as_mapping(metadata.get("semantics"))
    compact_motion = semantics.get("motion")
    if isinstance(compact_motion, Mapping):
        compact_motion = compact_motion.get("family")
    return str(
        motion.get("family")
        or semantic.get("motion_family")
        or compact_motion
        or _as_mapping(metadata.get("physics")).get("motion_profile")
        or semantics.get("profile")
        or ""
    )


def _support_label(metadata: Mapping[str, Any]) -> str | None:
    dimensions = _as_mapping(
        _as_mapping(metadata.get("semantic_sampling")).get("five_dimensions")
    )
    semantic_support = _as_mapping(dimensions.get("support_interaction"))
    compact_support = _as_mapping(
        _as_mapping(metadata.get("semantics")).get("support")
    )
    simulation = _as_mapping(metadata.get("simulation"))
    support = _as_mapping(simulation.get("support"))
    fixture = _as_mapping(_as_mapping(metadata.get("physics")).get("fixture"))
    raw_support = (
        semantic_support.get("support_type")
        or compact_support.get("support_type")
        or support.get("label")
        or support.get("semantic_type")
        or fixture.get("id")
    )
    if raw_support:
        aliases = {
            "concrete floor mat": "concrete floor",
            "ground channel ramp": "channel ramp",
            "ground ramp long shallow": "long shallow ramp",
            "ground ramp short steep": "short steep ramp",
            "ground ramp standard": "ramp",
            "indoor long floor": "indoor floor",
            "long lab bench": "long laboratory bench",
            "long wood table": "long wooden table",
            "lab bench": "laboratory bench",
            "raised channel ramp": "raised channel ramp",
            "raised ramp long shallow": "long shallow raised ramp",
            "raised ramp short steep": "short steep raised ramp",
            "raised ramp standard": "raised ramp",
            "wood floor": "wooden floor",
            "wood tray": "wooden tray",
            "wood tabletop": "wooden tabletop",
        }
        label = _humanize(raw_support)
        if re.search(r"(?:sketchfab|physassets)\s+[0-9a-f]{8,}", label):
            return None
        return aliases.get(label, label)
    assets = _as_mapping(metadata.get("assets"))
    if assets.get("support_asset_id"):
        label = _humanize(assets["support_asset_id"])
        if not re.search(r"(?:sketchfab|physassets)\s+[0-9a-f]{8,}", label):
            return label
    return None


def _environment_label(metadata: Mapping[str, Any]) -> str | None:
    dimensions = _as_mapping(
        _as_mapping(metadata.get("semantic_sampling")).get("five_dimensions")
    )
    appearance = _as_mapping(dimensions.get("appearance_lighting"))
    category = appearance.get("environment_category")
    if not category:
        category = _as_mapping(
            _as_mapping(metadata.get("semantics")).get("appearance")
        ).get("environment_category")
    if not category:
        category = _as_mapping(metadata.get("appearance")).get(
            "environment_category"
        )
    if not category:
        render_environment = _as_mapping(
            _as_mapping(metadata.get("render")).get("environment")
        )
        category = render_environment.get("role")
    if not category:
        return None
    aliases = {
        "garage workshop": "workshop",
        "home office": "indoor room",
        "lab studio": "laboratory studio",
        "minimal": "minimal indoor setting",
        "outdoor courtyard": "outdoor courtyard",
        "indoor neutral": "indoor environment",
        "studio soft": "studio setting",
    }
    label = _humanize(category)
    return aliases.get(label, label)


def _environment_prefix(metadata: Mapping[str, Any]) -> str:
    environment = _environment_label(metadata)
    if not environment:
        return ""
    article = "an" if environment[0] in "aeiou" else "a"
    return f"In {article} {environment}, "


def _one_object_caption(
    metadata: Mapping[str, Any], dynamic: list[dict[str, Any]]
) -> str:
    if len(dynamic) != 1:
        raise ValueError("one-object caption requires exactly one dynamic object")
    subject = f"the {dynamic[0]['semantic_label']}"
    support = _support_label(metadata) or "support surface"
    family = _motion_family(metadata)
    aliases = {
        "drop_fall": "drop_fall_1obj",
        "edge_fall": "edge_fall_1obj",
        "projectile": "projectile_1obj",
        "arc_projectile": "arc_projectile_1obj",
        "slide_push": "slide_push_1obj",
        "roll_or_slide": "roll_or_slide_1obj",
        "slope_slide_down": "slope_slide_down_1obj",
        "slope_slide_up": "slope_slide_up_1obj",
        "wall_impact": "wall_impact_1obj",
        "ramp_to_flat": "ramp_to_flat_1obj",
        "bounce": "bounce_1obj",
    }
    family = aliases.get(family, family)
    clauses = {
        "drop_fall_1obj": f"{subject} falls under gravity onto the {support}",
        "edge_fall_1obj": (
            f"{subject} moves across the {support} and falls from its edge"
        ),
        "projectile_1obj": (
            f"{subject} is launched horizontally through the air above the {support}"
        ),
        "arc_projectile_1obj": (
            f"{subject} is launched upward and forward above the {support}"
        ),
        "slide_push_1obj": (
            f"{subject} receives a short initial push and slides across the {support}"
        ),
        "roll_or_slide_1obj": f"{subject} rolls or slides across the {support}",
        "slope_slide_down_1obj": f"{subject} moves downhill along the {support}",
        "slope_slide_up_1obj": (
            f"{subject} is launched uphill along the {support} and returns downward"
        ),
        "wall_impact_1obj": (
            f"{subject} moves across the {support} and strikes a fixed wall"
        ),
        "ramp_to_flat_1obj": (
            f"{subject} moves down the {support} and continues onto flat ground"
        ),
        "bounce_1obj": (
            f"{subject} falls under gravity and bounces on the {support}"
        ),
        "vertical_drop": f"{subject} falls under gravity onto the {support}",
        "resting_push": (
            f"{subject} moves across the {support} after an initial push"
        ),
        "diagonal_push": (
            f"{subject} moves diagonally across the {support} after an initial push"
        ),
        "edge_exit": (
            f"{subject} moves across the {support}, leaves its edge, and falls"
        ),
        "workbench_clear_zone_drop": (
            f"{subject} falls under gravity onto a clear area of the workbench"
        ),
        "workbench_long_axis_push": (
            f"{subject} moves along the long axis of the workbench after an initial push"
        ),
        "single_ball_free_roll": (
            f"{subject} rolls freely across a billiards table without touching a rail"
        ),
        "single_ball_rail_rebound": (
            f"{subject} strikes a rail and rebounds across a billiards table"
        ),
        "dense_pinfield_descent": (
            f"{subject} descends through a dense passive peg field into a catch bin"
        ),
        "offset_pinfield_descent": (
            f"{subject} descends from an offset start through a passive peg field "
            "into a catch bin"
        ),
        "early_release_chain": (
            f"{subject} starts upstream and travels through a four-segment passive "
            "marble-run channel"
        ),
        "late_release_chain": (
            f"{subject} starts farther downstream and travels through a four-segment "
            "passive marble-run channel"
        ),
    }
    if family not in clauses:
        raise ValueError(f"one-object motion needs an explicit caption: {family}")
    return _environment_prefix(metadata) + clauses[family] + "."


def _two_object_caption(
    metadata: Mapping[str, Any], dynamic: list[dict[str, Any]]
) -> str | None:
    """Describe a declared 2obj initial event without predicting its outcome."""

    if len(dynamic) != 2:
        return None
    simulation = _as_mapping(metadata.get("simulation"))
    interaction = _as_mapping(simulation.get("interaction"))
    pattern = str(interaction.get("motion_pattern") or "")
    if not pattern:
        return None
    dimensions = _as_mapping(
        _as_mapping(metadata.get("semantic_sampling")).get("five_dimensions")
    )
    semantic_motion = _as_mapping(dimensions.get("motion"))
    semantic_family = str(semantic_motion.get("family") or "")
    if semantic_family and semantic_family != pattern:
        raise ValueError("two-object motion semantics disagree with interaction")

    left = f"the {dynamic[0]['semantic_label']}"
    right = f"the {dynamic[1]['semantic_label']}"
    support = _support_label(metadata) or "support surface"
    clauses = {
        "surface_hit_rest_2obj": (
            f"{left} moves across the {support} and collides with {right}, "
            "which starts at rest"
        ),
        "surface_head_on_2obj": (
            f"{left} and {right} move toward each other across the {support} "
            "and collide"
        ),
        "surface_crossing_2obj": (
            f"{left} and {right} move along crossing paths across the {support} "
            "and collide"
        ),
        "surface_catch_up_2obj": (
            f"{left} catches up with {right} while both move in the same "
            f"direction across the {support}, and they collide"
        ),
        "air_drop_hit_supported_2obj": (
            f"{left} falls under gravity and collides with {right}, which "
            f"starts at rest on the {support}"
        ),
        "air_projectile_hit_supported_2obj": (
            f"{left} is launched upward and forward, then collides with "
            f"{right}, which starts at rest on the {support}"
        ),
        "surface_single_independent_2obj": (
            f"{left} moves across the {support} while {right} remains at rest; "
            "they do not contact"
        ),
        "surface_dual_independent_2obj": (
            f"{left} and {right} move separately across the {support} without "
            "contacting each other"
        ),
        "air_supported_independent_2obj": (
            f"{left} falls under gravity beside {right}, which remains at rest "
            f"on the {support}; they do not contact"
        ),
    }
    if pattern not in clauses:
        raise ValueError(
            f"two-object motion needs an explicit caption: {pattern}"
        )
    return _environment_prefix(metadata) + clauses[pattern] + "."


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
    semantic_records = _as_list(
        _as_mapping(metadata.get("semantics")).get("objects")
    )
    semantic_labels = {
        str(record.get("object_id")): _public_object_label(
            record.get("semantic_label")
        )
        for record in semantic_records
        if isinstance(record, Mapping)
        and record.get("object_id")
        and record.get("semantic_label")
    }
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
                "semantic_label": semantic_labels.get(
                    object_id, _semantic_label(record, index)
                ),
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

    if len(dynamic) == 1:
        caption = _one_object_caption(metadata, dynamic)
    else:
        caption = _two_object_caption(metadata, dynamic)
    if caption is None:
        family = _motion_family(metadata)
        if family != "three_ball_collision":
            raise ValueError(
                f"multi-object motion needs an explicit caption: {family}"
            )
        labels = [str(record["semantic_label"]) for record in dynamic]
        subject = " and ".join(f"the {label}" for label in labels)
        caption = f"{subject} collide with one another on a billiards table."

    identity = {
        "schema_version": OBJECT_IDENTITY_SCHEMA_VERSION,
        "object_order": [str(record["object_id"]) for record in records],
        "objects": object_records,
        "text": {
            "caption": caption,
            "object_mentions": mentions,
            "template_version": "physweep_object_caption_v3",
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
