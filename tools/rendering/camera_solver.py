"""Deterministic camera geometry and search for rigid-object binders."""

from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Sequence

import numpy as np

from tools.core.camera_geometry import (
    camera_azimuth_offsets,
    deterministic_pair_side_azimuths,
    inclined_surface_side_readability,
    pair_approach_axis_xy,
)
from tools.dataset_contract.object_identity_contract import (
    require_simulation_objects,
)


SUPPORTED_DYNAMIC_OBJECT_COUNTS = (1, 2)


def _aabb_corners(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float64,
    )


def _trajectory_aabb_corners(
    lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    if (
        lower.ndim != 2
        or lower.shape[1] != 3
        or upper.shape != lower.shape
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or bool(np.any(lower > upper))
    ):
        raise ValueError("trajectory AABBs must be finite ordered (F, 3) arrays")
    return np.stack(
        [
            np.column_stack(
                [
                    lower[:, axis] if bit == 0 else upper[:, axis]
                    for axis, bit in enumerate(bits)
                ]
            )
            for bits in (
                (0, 0, 0),
                (0, 0, 1),
                (0, 1, 0),
                (0, 1, 1),
                (1, 0, 0),
                (1, 0, 1),
                (1, 1, 0),
                (1, 1, 1),
            )
        ],
        axis=1,
    )


def _frame_visibility_mask(
    projected: np.ndarray, margin_ndc: float
) -> np.ndarray:
    return np.logical_and.reduce(
        [
            projected[..., 0] >= margin_ndc,
            projected[..., 0] <= 1.0 - margin_ndc,
            projected[..., 1] >= margin_ndc,
            projected[..., 1] <= 1.0 - margin_ndc,
            projected[..., 2] > 0.1,
        ]
    )


def _projected_span(points: np.ndarray) -> float:
    return max(float(np.ptp(points[:, 0])), float(np.ptp(points[:, 1])))


def _two_object_camera_state(
    metadata: dict[str, Any], trajectory: dict[str, np.ndarray]
) -> tuple[
    list[dict[str, Any]],
    list[str],
    dict[str, Any],
    str,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    objects = require_simulation_objects(metadata, (2,), __name__)
    object_ids = [str(obj["object_id"]) for obj in objects]
    interaction = metadata["simulation"].get("interaction")
    if not isinstance(interaction, dict):
        raise ValueError("joint camera requires simulation.interaction")
    interaction_class = str(interaction.get("interaction_class", ""))
    if interaction_class not in {"interacting", "independent"}:
        raise ValueError("joint camera requires a declared interaction class")
    positions = np.stack(
        [
            np.asarray(trajectory[f"{object_id}__position_m"], dtype=np.float64)
            for object_id in object_ids
        ],
        axis=1,
    )
    lower = np.stack(
        [
            np.asarray(trajectory[f"{object_id}__aabb_min_m"], dtype=np.float64)
            for object_id in object_ids
        ],
        axis=1,
    )
    upper = np.stack(
        [
            np.asarray(trajectory[f"{object_id}__aabb_max_m"], dtype=np.float64)
            for object_id in object_ids
        ],
        axis=1,
    )
    if (
        positions.ndim != 3
        or positions.shape[1:] != (2, 3)
        or positions.shape != lower.shape
        or positions.shape != upper.shape
        or positions.shape[0] < 2
        or not np.isfinite(positions).all()
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or bool(np.any(lower > upper))
    ):
        raise ValueError("joint camera trajectories must be finite ordered arrays")
    return (
        objects,
        object_ids,
        interaction,
        interaction_class,
        positions,
        lower,
        upper,
    )


def _two_object_observation_contract(
    interaction: dict[str, Any],
) -> dict[str, float]:
    values = {
        "maximum_side_deviation": float(
            interaction["maximum_camera_side_deviation_degrees"]
        ),
        "minimum_elevation": float(
            interaction["minimum_camera_elevation_degrees"]
        ),
        "preferred_elevation": float(
            interaction["preferred_camera_elevation_degrees"]
        ),
        "maximum_elevation": float(
            interaction["maximum_camera_elevation_degrees"]
        ),
        "maximum_camera_distance": float(interaction["maximum_camera_distance_m"]),
        "maximum_distance_above_minimum": float(
            interaction["maximum_camera_distance_above_minimum_m"]
        ),
        "envelope_margin": float(interaction["full_motion_envelope_margin_ndc"]),
        "preferred_envelope_span": float(
            interaction["preferred_full_motion_envelope_span_ndc"]
        ),
        "minimum_object_span": float(
            interaction["minimum_per_object_median_span_ndc"]
        ),
        "minimum_unoccluded": float(
            interaction["minimum_per_object_unoccluded_fraction"]
        ),
        "minimum_keyframe_separation_ratio": float(
            interaction[
                "minimum_pair_keyframe_projected_center_separation_to_radius_sum_ratio"
            ]
        ),
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("joint camera observation values must be finite")
    if not 0.0 < values["maximum_side_deviation"] <= 45.0:
        raise ValueError("camera side deviation must lie in (0, 45]")
    if not (
        0.0
        < values["minimum_elevation"]
        <= values["preferred_elevation"]
        <= values["maximum_elevation"]
        < 90.0
    ):
        raise ValueError("joint camera elevations are invalid")
    if (
        values["maximum_camera_distance"] <= 0.0
        or values["maximum_distance_above_minimum"] <= 0.0
        or values["maximum_distance_above_minimum"]
        >= values["maximum_camera_distance"]
    ):
        raise ValueError("joint camera distance limits are invalid")
    maximum_envelope_span = 1.0 - 2.0 * values["envelope_margin"]
    if (
        not 0.0 <= values["envelope_margin"] < 0.20
        or not 0.0
        < values["preferred_envelope_span"]
        <= maximum_envelope_span
        or not 0.0
        < values["minimum_object_span"]
        <= values["preferred_envelope_span"]
        or not 0.0 < values["minimum_unoccluded"] <= 1.0
        or not 0.0 < values["minimum_keyframe_separation_ratio"] <= 1.0
    ):
        raise ValueError("joint full-motion envelope constraints are invalid")
    values["maximum_envelope_span"] = maximum_envelope_span
    return values


def _two_object_elevation_candidates(
    contract: dict[str, float],
) -> tuple[float, ...]:
    """Try the preferred elevation, then deterministic interior alternatives."""

    preferred = contract["preferred_elevation"]
    minimum = contract["minimum_elevation"]
    maximum = contract["maximum_elevation"]
    candidates = (
        preferred,
        0.5 * (preferred + maximum),
        0.5 * (preferred + minimum),
        0.25 * preferred + 0.75 * maximum,
        0.25 * preferred + 0.75 * minimum,
    )
    return tuple(dict.fromkeys(candidates))


def _camera_group_id(metadata: dict[str, Any]) -> str:
    sweep = metadata.get("sweep")
    if not isinstance(sweep, dict):
        return str(metadata["scene_id"])
    parent_id = sweep.get("parent_scene_id")
    if parent_id:
        return str(parent_id)
    return str(metadata["scene_id"])


def audit_two_object_camera(
    metadata: dict[str, Any],
    trajectory: dict[str, np.ndarray],
    camera: dict[str, Any],
) -> dict[str, Any]:
    """Audit one fixed camera against both complete object trajectories."""

    (
        objects,
        object_ids,
        interaction,
        interaction_class,
        positions,
        lower,
        upper,
    ) = _two_object_camera_state(metadata, trajectory)
    contract = _two_object_observation_contract(interaction)
    position = np.asarray(camera["position_m"], dtype=np.float64)
    target = np.asarray(camera["target_m"], dtype=np.float64)
    lens = float(camera["focal_length_mm"])
    sensor_width = float(camera["sensor_width_mm"])
    if (
        position.shape != (3,)
        or target.shape != (3,)
        or not np.isfinite(position).all()
        or not np.isfinite(target).all()
        or not np.isfinite(lens)
        or not np.isfinite(sensor_width)
        or lens <= 0.0
        or sensor_width <= 0.0
    ):
        raise ValueError("joint camera parameters must be finite and positive")
    view_vector = position - target
    distance = float(np.linalg.norm(view_vector))
    horizontal_distance = float(np.linalg.norm(view_vector[:2]))
    if distance <= 1.0e-8 or horizontal_distance <= 1.0e-8:
        raise ValueError("joint camera view vector is degenerate")
    elevation = math.degrees(math.atan2(float(view_vector[2]), horizontal_distance))
    if not (
        contract["minimum_elevation"] - 1.0e-6
        <= elevation
        <= contract["maximum_elevation"] + 1.0e-6
    ):
        raise ValueError(
            f"joint camera elevation is outside the contract: {elevation:.6f}"
        )
    if distance > contract["maximum_camera_distance"] + 1.0e-6:
        raise ValueError(
            f"joint camera exceeds its distance limit: {distance:.6f}"
        )
    approach_axis = np.asarray(
        pair_approach_axis_xy(interaction["approach_axis_xyz"]),
        dtype=np.float64,
    )
    horizontal_view = view_vector[:2] / horizontal_distance
    side_deviation = math.degrees(
        math.asin(float(np.clip(abs(horizontal_view @ approach_axis), 0.0, 1.0)))
    )
    if side_deviation > contract["maximum_side_deviation"] + 1.0e-6:
        raise ValueError(
            "joint camera is not sufficiently side-on: "
            f"deviation={side_deviation:.6f}"
        )

    aspect = 16.0 / 9.0
    envelope_margin = contract["envelope_margin"]
    blockers = camera_occlusion_colliders(metadata)
    envelope_lower = lower.min(axis=(0, 1))
    envelope_upper = upper.max(axis=(0, 1))
    envelope_corners = _aabb_corners(envelope_lower, envelope_upper)
    projected_envelope = project_points(
        envelope_corners, position, target, lens, sensor_width, aspect
    )
    envelope_visible = _frame_visibility_mask(projected_envelope, envelope_margin)
    envelope_visible_fraction = float(envelope_visible.mean())
    if not bool(envelope_visible.all()):
        raise ValueError(
            "joint camera does not contain the complete full-motion envelope: "
            f"visible_fraction={envelope_visible_fraction:.6f}"
        )
    envelope_projected_span = _projected_span(projected_envelope)
    per_object: dict[str, dict[str, float]] = {}
    for object_index, object_id in enumerate(object_ids):
        projected_centers = project_points(
            positions[:, object_index], position, target, lens, sensor_width, aspect
        )
        center_visible = float(
            _frame_visibility_mask(projected_centers, envelope_margin).mean()
        )
        object_corners = _trajectory_aabb_corners(
            lower[:, object_index], upper[:, object_index]
        )
        projected_corners = project_points(
            object_corners.reshape(-1, 3),
            position,
            target,
            lens,
            sensor_width,
            aspect,
        ).reshape(object_corners.shape)
        aabb_visible = _frame_visibility_mask(projected_corners, envelope_margin)
        aabb_visible_fraction = float(aabb_visible.mean())
        projected_spans = np.maximum(
            np.ptp(projected_corners[..., 0], axis=1),
            np.ptp(projected_corners[..., 1], axis=1),
        )
        median_span = float(np.median(projected_spans))
        static_unoccluded = unoccluded_fraction(
            position, positions[:, object_index], blockers
        )
        per_object[object_id] = {
            "full_center_visible_fraction": center_visible,
            "full_motion_aabb_visible_fraction": aabb_visible_fraction,
            "initial_aabb_visible_fraction": float(aabb_visible[0].mean()),
            "initial_span_ndc": float(projected_spans[0]),
            "median_span_ndc": median_span,
            "static_unoccluded_fraction": static_unoccluded,
        }
        if (
            not bool(aabb_visible.all())
            or median_span < contract["minimum_object_span"]
            or static_unoccluded < contract["minimum_unoccluded"]
        ):
            raise ValueError(
                f"joint camera violates per-object visibility for {object_id}: "
                f"{per_object[object_id]}"
            )

    pair_contacts = np.asarray(
        trajectory[f"{object_ids[0]}__object_contact_count__{object_ids[1]}"],
        dtype=np.int32,
    )
    if pair_contacts.shape != (positions.shape[0],):
        raise ValueError("pair-contact trajectory length is inconsistent")
    collision_indices = np.flatnonzero(pair_contacts > 0)
    if interaction_class == "interacting":
        if not collision_indices.size:
            raise ValueError("joint camera requires the declared pair collision")
        pair_keyframe = int(collision_indices[0])
        pair_keyframe_kind = "first_contact"
    else:
        if collision_indices.size:
            raise ValueError("independent joint camera received a pair collision")
        pair_keyframe = int(
            np.argmin(np.linalg.norm(positions[:, 1] - positions[:, 0], axis=1))
        )
        pair_keyframe_kind = "closest_approach"
    keyframe_projection = project_points(
        positions[pair_keyframe], position, target, lens, sensor_width, aspect
    )
    if not bool(_frame_visibility_mask(keyframe_projection, envelope_margin).all()):
        raise ValueError("joint camera does not contain both pair keyframe centers")
    projected_keyframe_separation = float(
        np.linalg.norm(keyframe_projection[1, :2] - keyframe_projection[0, :2])
    )
    separation_direction = (
        keyframe_projection[1, :2] - keyframe_projection[0, :2]
    ) / max(projected_keyframe_separation, 1.0e-8)
    direction_scale = math.sqrt(
        float(separation_direction[0]) ** 2
        + (aspect * float(separation_direction[1])) ** 2
    )
    keyframe_projected_radii = []
    for object_index, obj in enumerate(objects):
        geometry = obj.get("geometry", {})
        size = np.asarray(geometry.get("size_m"), dtype=np.float64)
        if (
            geometry.get("type") != "sphere"
            or size.shape != (3,)
            or not np.isfinite(size).all()
            or bool(np.any(size <= 0.0))
            or not np.allclose(size, size[0], atol=1.0e-8, rtol=0.0)
        ):
            raise ValueError("joint pair-keyframe visibility requires spheres")
        depth = float(keyframe_projection[object_index, 2])
        keyframe_projected_radii.append(
            (lens / sensor_width)
            * (0.5 * float(size[0]))
            * direction_scale
            / depth
        )
    keyframe_separation_ratio = projected_keyframe_separation / max(
        sum(keyframe_projected_radii), 1.0e-8
    )
    if keyframe_separation_ratio < contract["minimum_keyframe_separation_ratio"]:
        raise ValueError(
            "joint camera visually overlaps the two objects at the pair keyframe: "
            f"ratio={keyframe_separation_ratio:.6f}"
        )
    return {
        "object_count": 2,
        "per_object_visibility": per_object,
        "joint_motion_envelope_world_bounds_m": {
            "min": [round(float(value), 6) for value in envelope_lower],
            "max": [round(float(value), 6) for value in envelope_upper],
        },
        "joint_motion_envelope_visible_fraction": round(
            envelope_visible_fraction, 6
        ),
        "joint_motion_envelope_span_ndc": round(envelope_projected_span, 6),
        "pair_keyframe_kind": pair_keyframe_kind,
        "pair_keyframe_index": pair_keyframe,
        "pair_keyframe_projected_center_separation_ndc": round(
            projected_keyframe_separation, 6
        ),
        "pair_keyframe_projected_center_separation_to_radius_sum_ratio": round(
            keyframe_separation_ratio, 6
        ),
        "pair_keyframe_projected_silhouette_radii_ndc": [
            round(value, 6) for value in keyframe_projected_radii
        ],
        "side_view_deviation_degrees": round(side_deviation, 6),
        "camera_elevation_degrees": round(elevation, 6),
        "camera_distance_m": round(distance, 6),
    }


def _solve_two_object_camera(
    metadata: dict[str, Any],
    trajectory: dict[str, np.ndarray],
    rules: dict[str, Any],
    *,
    audit_members: Sequence[
        tuple[dict[str, Any], dict[str, np.ndarray]]
    ] = (),
) -> dict[str, Any]:
    """Frame and audit the complete spatiotemporal envelope of both objects."""

    (
        objects,
        object_ids,
        interaction,
        interaction_class,
        positions,
        lower,
        upper,
    ) = _two_object_camera_state(metadata, trajectory)
    contract = _two_object_observation_contract(interaction)
    maximum_side_deviation = contract["maximum_side_deviation"]
    minimum_elevation = contract["minimum_elevation"]
    preferred_elevation = contract["preferred_elevation"]
    maximum_elevation = contract["maximum_elevation"]
    maximum_camera_distance = contract["maximum_camera_distance"]
    maximum_distance_above_minimum = contract["maximum_distance_above_minimum"]
    envelope_margin = contract["envelope_margin"]
    preferred_envelope_span = contract["preferred_envelope_span"]
    minimum_object_span = contract["minimum_object_span"]
    maximum_envelope_span = contract["maximum_envelope_span"]
    camera_group_id = _camera_group_id(metadata)
    joint_lower = lower.min(axis=1)
    joint_upper = upper.max(axis=1)
    envelope_lower = joint_lower.min(axis=0)
    envelope_upper = joint_upper.max(axis=0)
    envelope_size = envelope_upper - envelope_lower
    if bool(np.any(envelope_size <= 0.0)):
        raise ValueError("joint full-motion envelope must have positive size")
    envelope_center = 0.5 * (envelope_lower + envelope_upper)
    virtual_id = "camera_joint_dynamic_envelope"
    virtual_metadata = copy.deepcopy(metadata)
    virtual_metadata["scene_id"] = camera_group_id
    virtual_object = copy.deepcopy(objects[0])
    virtual_object["object_id"] = virtual_id
    virtual_object["geometry"] = {
        "type": "cuboid",
        "size_m": envelope_size.tolist(),
    }
    virtual_metadata["simulation"]["objects"] = [virtual_object]
    virtual_request = virtual_metadata["camera_request"]
    joint_observation = virtual_request["observation"]
    joint_observation.update(
        {
            "intent": "joint_full_motion_envelope",
            "focus_event": {"type": "fraction", "fraction": 1.0},
            "structure_context": "horizontal_surface",
            "preferred_object_span_ndc": preferred_envelope_span,
            "minimum_median_object_span_ndc": minimum_object_span,
        }
    )
    virtual_request.update(
        {
            "minimum_full_trajectory_center_visible_fraction": 1.0,
            "minimum_primary_trajectory_center_visible_fraction": 1.0,
            "full_trajectory_camera_target_blend": 1.0,
            "minimum_initial_object_span_ndc": minimum_object_span,
            "minimum_initial_object_visible_fraction": 1.0,
            "initial_object_center_margin_ndc": envelope_margin,
            "initial_object_corner_margin_ndc": envelope_margin,
            "maximum_initial_object_span_ndc": maximum_envelope_span,
            "soft_maximum_focus_span_ndc": preferred_envelope_span,
            "maximum_focus_span_ndc": maximum_envelope_span,
            "minimum_camera_elevation_degrees": minimum_elevation,
            "maximum_camera_elevation_degrees": maximum_elevation,
            "maximum_camera_distance_m": maximum_camera_distance,
            "maximum_camera_distance_above_minimum_m": (
                maximum_distance_above_minimum
            ),
            "maximum_local_azimuth_deviation_degrees": maximum_side_deviation,
            "minimum_primary_trajectory_unoccluded_fraction": 0.0,
            "minimum_full_trajectory_unoccluded_fraction": 0.0,
            "allow_partial_exit": False,
        }
    )
    virtual_trajectory = dict(trajectory)
    virtual_trajectory[f"{virtual_id}__position_m"] = np.tile(
        envelope_center, (positions.shape[0], 1)
    )
    virtual_trajectory[f"{virtual_id}__aabb_min_m"] = np.tile(
        envelope_lower, (positions.shape[0], 1)
    )
    virtual_trajectory[f"{virtual_id}__aabb_max_m"] = np.tile(
        envelope_upper, (positions.shape[0], 1)
    )
    profile = str(virtual_metadata["camera_request"]["profile"])
    failures = []
    camera = None
    selected_azimuth = None
    selected_elevation = None
    selected_member_diagnostics: list[dict[str, Any]] = []
    side_azimuths = deterministic_pair_side_azimuths(
        camera_group_id, interaction["approach_axis_xyz"]
    )
    for candidate_elevation in _two_object_elevation_candidates(contract):
        for candidate_azimuth in side_azimuths:
            joint_rules = copy.deepcopy(rules)
            matching_profiles = [
                record
                for record in joint_rules["axes"]["camera_axis"]
                if str(record["label"]) == profile
            ]
            if len(matching_profiles) != 1:
                raise ValueError(f"joint camera profile is not unique: {profile}")
            view_rule = matching_profiles[0]["overrides"]["view_rule"]
            view_rule["azimuth_degrees"] = candidate_azimuth
            view_rule["elevation_degrees"] = candidate_elevation
            try:
                candidate = solve_camera(
                    virtual_metadata, virtual_trajectory, joint_rules
                )
                candidate_diagnostics = audit_two_object_camera(
                    metadata, trajectory, candidate
                )
                member_diagnostics = []
                for member_metadata, member_trajectory in audit_members:
                    try:
                        member_diagnostics.append(
                            audit_two_object_camera(
                                member_metadata, member_trajectory, candidate
                            )
                        )
                    except ValueError as error:
                        raise ValueError(
                            "camera group member failed: "
                            f"{member_metadata['scene_id']}: {error}"
                        ) from error
            except ValueError as error:
                failures.append(
                    f"azimuth={candidate_azimuth:.6f},"
                    f"elevation={candidate_elevation:.6f}: {error}"
                )
                continue
            camera = candidate
            camera["diagnostics"].update(candidate_diagnostics)
            selected_azimuth = candidate_azimuth
            selected_elevation = candidate_elevation
            selected_member_diagnostics = member_diagnostics
            break
        if camera is not None:
            break
    if camera is None or selected_azimuth is None or selected_elevation is None:
        raise ValueError(
            "joint camera could not solve a contract candidate; "
            + " | ".join(failures)
        )
    camera["solver_version"] = (
        "joint_full_motion_envelope_group_camera_v2"
        if audit_members
        else "joint_full_motion_envelope_camera_v2"
    )
    camera["diagnostics"]["pair_selected_side_azimuth_degrees"] = round(
        selected_azimuth, 6
    )
    camera["diagnostics"]["pair_selected_elevation_degrees"] = round(
        selected_elevation, 6
    )
    camera["diagnostics"]["pair_camera_candidate_failure_count"] = len(failures)
    if audit_members:
        representative = selected_member_diagnostics[0]
        camera["diagnostics"].update(
            {
                key: value
                for key, value in representative.items()
                if key.startswith("pair_keyframe_")
            }
        )
        camera["diagnostics"]["camera_group"] = {
            "group_id": camera_group_id,
            "member_count": len(audit_members),
            "all_members_audited": True,
            "minimum_joint_motion_envelope_visible_fraction": min(
                record["joint_motion_envelope_visible_fraction"]
                for record in selected_member_diagnostics
            ),
            "minimum_per_object_median_span_ndc": {
                object_id: round(
                    min(
                        record["per_object_visibility"][object_id][
                            "median_span_ndc"
                        ]
                        for record in selected_member_diagnostics
                    ),
                    6,
                )
                for object_id in object_ids
            },
            "minimum_per_object_unoccluded_fraction": {
                object_id: round(
                    min(
                        record["per_object_visibility"][object_id][
                            "static_unoccluded_fraction"
                        ]
                        for record in selected_member_diagnostics
                    ),
                    6,
                )
                for object_id in object_ids
            },
        }
    return camera


def solve_two_object_camera_group(
    metadata: dict[str, Any],
    members: Sequence[tuple[dict[str, Any], dict[str, np.ndarray]]],
    rules: dict[str, Any],
) -> dict[str, Any]:
    """Freeze one camera that passes every trajectory in a one-factor group."""

    if not members:
        raise ValueError("two-object camera group has no members")
    base_objects, base_ids, base_interaction, base_class, *_ = (
        _two_object_camera_state(metadata, members[0][1])
    )
    base_contract = _two_object_observation_contract(base_interaction)
    group_id = _camera_group_id(metadata)
    base_camera_request = metadata["camera_request"]
    base_support = metadata["simulation"]["support"]
    binding_sha256 = str(
        metadata.get("environment_binding", {}).get("binding_sha256", "")
    )
    if not binding_sha256:
        raise ValueError("two-object camera group lacks an environment binding")
    base_static_objects = [
        {
            "object_id": obj["object_id"],
            "geometry": obj["geometry"],
            "initial_state": obj["initial_state"],
        }
        for obj in base_objects
    ]
    normalized = []
    seen_members = set()
    for member_metadata, member_trajectory in sorted(
        members,
        key=lambda item: (
            item[0].get("sweep", {}).get("kind") != "base",
            str(item[0]["scene_id"]),
        ),
    ):
        member_id = str(member_metadata["scene_id"])
        if member_id in seen_members:
            raise ValueError(f"duplicate camera group member: {member_id}")
        seen_members.add(member_id)
        (
            member_objects,
            member_ids,
            member_interaction,
            member_class,
            *_,
        ) = _two_object_camera_state(member_metadata, member_trajectory)
        member_static_objects = [
            {
                "object_id": obj["object_id"],
                "geometry": obj["geometry"],
                "initial_state": obj["initial_state"],
            }
            for obj in member_objects
        ]
        if (
            _camera_group_id(member_metadata) != group_id
            or member_ids != base_ids
            or member_class != base_class
            or member_static_objects != base_static_objects
            or member_metadata["camera_request"] != base_camera_request
            or member_metadata["simulation"]["support"] != base_support
            or _two_object_observation_contract(member_interaction)
            != base_contract
            or not np.allclose(
                member_interaction["approach_axis_xyz"],
                base_interaction["approach_axis_xyz"],
                atol=1.0e-12,
                rtol=0.0,
            )
            or str(
                member_metadata.get("environment_binding", {}).get(
                    "binding_sha256", ""
                )
            )
            != binding_sha256
        ):
            raise ValueError(
                f"incompatible two-object camera group member: {member_id}"
            )
        normalized.append((member_metadata, member_trajectory))

    trajectory_keys = [
        key
        for object_id in base_ids
        for key in (
            f"{object_id}__position_m",
            f"{object_id}__aabb_min_m",
            f"{object_id}__aabb_max_m",
        )
    ]
    trajectory_keys.append(
        f"{base_ids[0]}__object_contact_count__{base_ids[1]}"
    )
    aggregate = {
        key: np.concatenate(
            [trajectory[key] for _, trajectory in normalized], axis=0
        )
        for key in trajectory_keys
    }
    return _solve_two_object_camera(
        metadata,
        aggregate,
        rules,
        audit_members=normalized,
    )


def project_points(
    points: np.ndarray,
    position: np.ndarray,
    target: np.ndarray,
    lens_mm: float,
    sensor_width_mm: float,
    aspect: float,
) -> np.ndarray:
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1.0e-8:
        right = np.asarray([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = points - position
    depth = relative @ forward
    horizontal = relative @ right
    vertical = relative @ up
    half_width = np.maximum(depth * sensor_width_mm / (2.0 * lens_mm), 1.0e-8)
    half_height = half_width / aspect
    return np.column_stack(
        [
            0.5 + horizontal / (2.0 * half_width),
            0.5 + vertical / (2.0 * half_height),
            depth,
        ]
    )


def image_center_visibility_mask(projected: np.ndarray) -> np.ndarray:
    """Return whether projected centers lie inside the actual image."""

    projected = np.asarray(projected, dtype=np.float64)
    if projected.ndim != 2 or projected.shape[1] != 3:
        raise ValueError("projected centers must have shape (N, 3)")
    return np.logical_and.reduce(
        [
            projected[:, 0] >= 0.0,
            projected[:, 0] <= 1.0,
            projected[:, 1] >= 0.0,
            projected[:, 1] <= 1.0,
            projected[:, 2] > 0.1,
        ]
    )


def camera_inside_structural_envelope(
    metadata: dict[str, Any],
    position: np.ndarray,
) -> bool:
    """Keep cameras inside structures whose walls define the usable space."""

    support = metadata["simulation"]["support"]
    envelope = support.get("camera_envelope")
    if envelope is None:
        return True
    if str(envelope.get("type")) != "paired_parallel_walls":
        raise ValueError("unsupported structural camera envelope")
    motion_axis = str(envelope["motion_axis"])
    if motion_axis not in {"x", "y"}:
        raise ValueError("structural camera-envelope axis must be x or y")
    cross_axis = 1 if motion_axis == "x" else 0
    wall_ids = {str(value) for value in envelope["collider_ids"]}
    walls = [
        collider
        for collider in support.get("colliders", [])
        if str(collider.get("id")) in wall_ids
    ]
    if len(wall_ids) != 2 or len(walls) != 2:
        raise ValueError("paired-wall camera envelope requires two colliders")
    clearance_m = float(envelope["clearance_m"])
    if clearance_m <= 0.0:
        raise ValueError("structural camera-envelope clearance must be positive")
    lower_inner = -math.inf
    upper_inner = math.inf
    ceiling = math.inf
    for wall in walls:
        center = np.asarray(wall["position_m"], dtype=np.float64)
        size = np.asarray(wall["size_m"], dtype=np.float64)
        if center[cross_axis] < 0.0:
            lower_inner = max(
                lower_inner, float(center[cross_axis] + 0.5 * size[cross_axis])
            )
        else:
            upper_inner = min(
                upper_inner, float(center[cross_axis] - 0.5 * size[cross_axis])
            )
        ceiling = min(ceiling, float(center[2] + 0.5 * size[2]))
    position = np.asarray(position, dtype=np.float64)
    return bool(
        lower_inner + clearance_m <= position[cross_axis] <= upper_inner - clearance_m
        and position[2] <= ceiling - clearance_m
    )


def camera_target_centers(
    focus_points: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return primary-observation and complete-trajectory target centers."""

    focus_points = np.asarray(focus_points, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if focus_points.ndim != 2 or focus_points.shape[1] != 3:
        raise ValueError("focus points must have shape (N, 3)")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("trajectory positions must have shape (N, 3)")
    primary_target = (focus_points.min(axis=0) + focus_points.max(axis=0)) / 2.0
    full_target = (positions.min(axis=0) + positions.max(axis=0)) / 2.0
    return primary_target, full_target


def camera_azimuth_degrees(rules: dict[str, Any], profile: str) -> float:
    for record in rules["axes"]["camera_axis"]:
        if str(record["label"]) == profile:
            return float(record["overrides"]["view_rule"]["azimuth_degrees"])
    raise ValueError(f"unknown camera profile: {profile}")


def camera_elevation_degrees(rules: dict[str, Any], profile: str) -> float:
    for record in rules["axes"]["camera_axis"]:
        if str(record["label"]) == profile:
            return float(record["overrides"]["view_rule"]["elevation_degrees"])
    raise ValueError(f"unknown camera profile: {profile}")


def sampled_motion_points(
    trajectory: dict[str, np.ndarray],
    object_id: str,
    observation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(trajectory[f"{object_id}__position_m"], dtype=np.float64)
    lower = np.asarray(trajectory[f"{object_id}__aabb_min_m"], dtype=np.float64)
    upper = np.asarray(trajectory[f"{object_id}__aabb_max_m"], dtype=np.float64)
    event = observation["focus_event"]
    event_type = str(event["type"])
    post_event_frames = max(2, int(event.get("post_event_frames", 2)))

    def first_contact_index(key: str) -> int:
        if key not in trajectory:
            raise ValueError(f"camera observation has no trajectory channel: {key}")
        contacts = np.flatnonzero(np.asarray(trajectory[key], dtype=np.int64) > 0)
        if not contacts.size:
            raise ValueError(f"camera observation event did not occur: {key}")
        return int(contacts[0])

    key_indices = [0]
    if event_type == "fraction":
        fraction = float(event["fraction"])
        if not 0.0 < fraction <= 1.0:
            raise ValueError("camera observation fraction must be in (0, 1]")
        primary_end = max(2, min(len(positions), int(math.ceil(len(positions) * fraction))))
    elif event_type == "primary_support_contact":
        event_index = first_contact_index(
            f"{object_id}__primary_support_contact_count"
        )
        primary_end = min(len(positions), event_index + post_event_frames)
        key_indices.append(event_index)
    elif event_type == "collider_contact":
        event_index = first_contact_index(
            f"{object_id}__collider_contact_count__{event['collider_id']}"
        )
        primary_end = min(len(positions), event_index + post_event_frames)
        key_indices.append(event_index)
    elif event_type == "first_contact":
        contact_keys = [f"{object_id}__primary_support_contact_count"]
        contact_keys.extend(
            key
            for key in trajectory
            if key.startswith(f"{object_id}__collider_contact_count__")
        )
        occurred = []
        for key in contact_keys:
            contacts = np.flatnonzero(np.asarray(trajectory[key], dtype=np.int64) > 0)
            if contacts.size:
                occurred.append(int(contacts[0]))
        if not occurred:
            raise ValueError("camera observation expected a contact event")
        event_index = min(occurred)
        primary_end = min(len(positions), event_index + post_event_frames)
        key_indices.append(event_index)
    else:
        raise ValueError(f"unknown camera focus event: {event_type}")

    key_indices.append(primary_end - 1)
    if str(observation["intent"]) == "ballistic_arc":
        key_indices.append(int(np.argmax(positions[:primary_end, 2])))
    sampled = np.linspace(0, primary_end - 1, min(18, primary_end), dtype=int)
    indices = np.unique(np.concatenate([sampled, np.asarray(key_indices, dtype=int)]))
    points = []
    for index in indices:
        points.append(positions[index])
        for x in (lower[index, 0], upper[index, 0]):
            for y in (lower[index, 1], upper[index, 1]):
                for z in (lower[index, 2], upper[index, 2]):
                    points.append([x, y, z])
    return np.asarray(points, dtype=np.float64), positions, indices


def support_context_points(
    metadata: dict[str, Any],
    azimuth_degrees: float,
    focus_xy: np.ndarray,
    observation: dict[str, Any],
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    support = metadata["simulation"]["support"]
    outward = np.asarray(
        [math.cos(math.radians(azimuth_degrees)), math.sin(math.radians(azimuth_degrees))],
        dtype=np.float64,
    )
    lateral = np.asarray([-outward[1], outward[0]], dtype=np.float64)
    bounds = support["safe_surface_bounds"]
    lower = np.asarray([float(bounds["x"][0]), float(bounds["y"][0])])
    upper = np.asarray([float(bounds["x"][1]), float(bounds["y"][1])])
    center = np.clip(np.asarray(focus_xy, dtype=np.float64), lower, upper)

    ray_limits = []
    for axis in range(2):
        if abs(outward[axis]) <= 1.0e-8:
            continue
        boundary = upper[axis] if outward[axis] > 0.0 else lower[axis]
        ray_limits.append((boundary - center[axis]) / outward[axis])
    positive_limits = [value for value in ray_limits if value >= 0.0]
    edge_distance = min(positive_limits) if positive_limits else 0.0
    edge = np.clip(center + outward * edge_distance * 0.88, lower, upper)
    trajectory_span = float(np.linalg.norm(np.ptp(positions[:, :2], axis=0)))
    local_span = min(0.60, max(0.18, trajectory_span * 0.25))
    side_a = np.clip(center + lateral * local_span, lower, upper)
    side_b = np.clip(center - lateral * local_span, lower, upper)
    inward = np.clip(center - outward * local_span, lower, upper)

    slope_angle = math.radians(float(support["surface_frame"]["slope_angle_degrees"]))
    support_top = float(support["surface_center_z_m"])

    def surface_point(xy: np.ndarray) -> list[float]:
        return [
            float(xy[0]),
            float(xy[1]),
            support_top + float(xy[1]) * math.tan(slope_angle) + 0.01,
        ]

    points = [surface_point(value) for value in (center, edge, side_a, side_b, inward)]
    # A horizontal support is identified by the local contact patch, not by a
    # distant boundary of an arbitrarily large floor plane.
    required_indices = [0, 2, 3]
    context = str(observation["structure_context"])

    if (
        context == "horizontal_surface"
        and str(support.get("scene_class")) == "raised_flat"
        and support.get("motion_axis") in {"x", "y"}
        and "maximum_planar_trajectory_distance_m" in support
    ):
        axis = 0 if support["motion_axis"] == "x" else 1
        cross_axis = 1 - axis
        support_center = np.clip(np.zeros(2, dtype=np.float64), lower, upper)
        axis_half_span = min(
            0.45 * float(upper[axis] - lower[axis]),
            0.5 * float(support["maximum_planar_trajectory_distance_m"]),
        )
        cross_half_span = 0.30 * float(upper[cross_axis] - lower[cross_axis])
        long_axis_anchors = []
        for sign in (-1.0, 1.0):
            value = support_center.copy()
            value[axis] += sign * axis_half_span
            long_axis_anchors.append(np.clip(value, lower, upper))
        for sign in (-1.0, 1.0):
            value = support_center.copy()
            value[cross_axis] += sign * cross_half_span
            long_axis_anchors.append(np.clip(value, lower, upper))
        required_indices = list(
            range(len(points), len(points) + 1 + len(long_axis_anchors))
        )
        points.extend(
            surface_point(value)
            for value in (support_center, *long_axis_anchors)
        )

    if context in {"inclined_surface", "ramp_and_landing"}:
        visual_geometry = support.get("visual_geometry")
        if (
            isinstance(visual_geometry, dict)
            and str(visual_geometry.get("primitive")) == "solid_wedge"
        ):
            width, length = [
                float(value) for value in visual_geometry["size_xy_m"]
            ]
            low_z = float(visual_geometry["base_z_m"]) + 0.01
            high_z = float(visual_geometry["high_top_z_m"]) + 0.01
            wedge_corners = [
                [-width / 2.0, -length / 2.0, low_z],
                [width / 2.0, -length / 2.0, low_z],
                [-width / 2.0, length / 2.0, high_z],
                [width / 2.0, length / 2.0, high_z],
            ]
            required_indices = list(
                range(len(points), len(points) + len(wedge_corners))
            )
            points.extend(wedge_corners)
        else:
            low = np.asarray([center[0], lower[1]], dtype=np.float64)
            high = np.asarray([center[0], upper[1]], dtype=np.float64)
            required_indices = [len(points), len(points) + 1]
            points.extend([surface_point(low), surface_point(high)])

    collider_id = observation["focus_event"].get("collider_id")
    if context == "impact_boundary":
        collider = next(
            (
                value
                for value in support["colliders"]
                if str(value["id"]) == str(collider_id)
            ),
            None,
        )
        if collider is None:
            raise ValueError(f"camera context collider is missing: {collider_id}")
        size = np.asarray(collider["size_m"], dtype=np.float64)
        transform = rotation_matrix_xyz_degrees(collider["rotation_euler_degrees"])
        origin = np.asarray(collider["position_m"], dtype=np.float64)
        local = np.asarray(
            [
                [0.0, 0.0, size[2] / 2.0],
                [-size[0] / 2.0, 0.0, size[2] / 2.0],
                [size[0] / 2.0, 0.0, size[2] / 2.0],
            ],
            dtype=np.float64,
        )
        required_indices = list(range(len(points), len(points) + len(local)))
        points.extend((local @ transform.T + origin).tolist())

    if context == "ramp_and_landing":
        collider = next(
            (
                value
                for value in support["colliders"]
                if str(value["id"]) == str(collider_id)
            ),
            None,
        )
        if collider is None:
            raise ValueError(f"camera context collider is missing: {collider_id}")
        size = np.asarray(collider["size_m"], dtype=np.float64)
        transform = rotation_matrix_xyz_degrees(collider["rotation_euler_degrees"])
        origin = np.asarray(collider["position_m"], dtype=np.float64)
        local_landing = transform.T @ (positions[-1] - origin)
        local_landing[:2] = np.clip(
            local_landing[:2],
            -size[:2] / 2.0,
            size[:2] / 2.0,
        )
        local_landing[2] = size[2] / 2.0 + 0.01
        landing = local_landing @ transform.T + origin
        required_indices.append(len(points))
        points.append(landing.tolist())

    if context == "edge_and_landing":
        travel = positions[-1, :2] - positions[0, :2]
        if np.linalg.norm(travel) > 1.0e-8:
            travel /= np.linalg.norm(travel)
        else:
            travel = outward
        limits = []
        for axis in range(2):
            if abs(travel[axis]) <= 1.0e-8:
                continue
            boundary = upper[axis] if travel[axis] > 0.0 else lower[axis]
            limits.append((boundary - center[axis]) / travel[axis])
        positive = [value for value in limits if value >= 0.0]
        edge_xy = np.clip(
            center + travel * (min(positive) if positive else 0.0), lower, upper
        )
        landing = positions[-1].copy()
        landing[2] = min(float(landing[2]), support_top)
        required_indices = [len(points), len(points) + 1]
        points.extend([surface_point(edge_xy), landing.tolist()])

    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(required_indices, dtype=int),
    )


def rotation_matrix_xyz_degrees(euler_degrees: list[float]) -> np.ndarray:
    x, y, z = np.radians(np.asarray(euler_degrees, dtype=np.float64))
    rx = np.asarray([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.asarray([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.asarray([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def segment_intersects_box(
    start: np.ndarray, end: np.ndarray, collider: dict[str, Any]
) -> bool:
    """Return whether an oriented box blocks the open segment to a target point."""

    center = np.asarray(collider["position_m"], dtype=np.float64)
    half = np.asarray(collider["size_m"], dtype=np.float64) / 2.0
    rotation = rotation_matrix_xyz_degrees(collider["rotation_euler_degrees"])
    local_start = rotation.T @ (np.asarray(start, dtype=np.float64) - center)
    local_end = rotation.T @ (np.asarray(end, dtype=np.float64) - center)
    direction = local_end - local_start
    lower_t, upper_t = 0.0, 0.985
    for axis in range(3):
        if abs(direction[axis]) <= 1.0e-10:
            if local_start[axis] < -half[axis] or local_start[axis] > half[axis]:
                return False
            continue
        first = (-half[axis] - local_start[axis]) / direction[axis]
        second = (half[axis] - local_start[axis]) / direction[axis]
        near, far = sorted((float(first), float(second)))
        lower_t = max(lower_t, near)
        upper_t = min(upper_t, far)
        if lower_t > upper_t:
            return False
    return upper_t >= 0.0 and lower_t <= 0.985


def segments_intersect_box(
    start: np.ndarray, ends: np.ndarray, collider: dict[str, Any]
) -> np.ndarray:
    """Vectorized equivalent of segment_intersects_box for one camera pose."""

    points = np.asarray(ends, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("segment endpoints must have shape (N, 3)")
    center = np.asarray(collider["position_m"], dtype=np.float64)
    half = np.asarray(collider["size_m"], dtype=np.float64) / 2.0
    rotation = rotation_matrix_xyz_degrees(collider["rotation_euler_degrees"])
    local_start = rotation.T @ (np.asarray(start, dtype=np.float64) - center)
    local_ends = (points - center) @ rotation
    directions = local_ends - local_start
    lower_t = np.zeros(len(points), dtype=np.float64)
    upper_t = np.full(len(points), 0.985, dtype=np.float64)
    possible = np.ones(len(points), dtype=bool)
    for axis in range(3):
        direction = directions[:, axis]
        parallel = np.abs(direction) <= 1.0e-10
        possible &= ~(
            parallel
            & (
                (local_start[axis] < -half[axis])
                | (local_start[axis] > half[axis])
            )
        )
        nonparallel = ~parallel
        if np.any(nonparallel):
            first = (-half[axis] - local_start[axis]) / direction[nonparallel]
            second = (half[axis] - local_start[axis]) / direction[nonparallel]
            lower_t[nonparallel] = np.maximum(
                lower_t[nonparallel], np.minimum(first, second)
            )
            upper_t[nonparallel] = np.minimum(
                upper_t[nonparallel], np.maximum(first, second)
            )
        possible &= lower_t <= upper_t
    return possible & (upper_t >= 0.0) & (lower_t <= 0.985)


def unoccluded_fraction(
    camera_position: np.ndarray,
    points: np.ndarray,
    colliders: list[dict[str, Any]],
) -> float:
    blockers = [
        collider
        for collider in colliders
        if bool(collider.get("visible", True))
        and collider.get("primitive") == "box"
    ]
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        return 0.0
    blocked = np.zeros(len(points), dtype=bool)
    for collider in blockers:
        blocked |= segments_intersect_box(camera_position, points, collider)
        if np.all(blocked):
            break
    return float((~blocked).mean())


def camera_occlusion_colliders(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every frozen box that can visibly block the selected camera."""

    support_blockers = [
        collider
        for collider in metadata["simulation"]["support"]["colliders"]
        if bool(collider.get("visible", True))
        and bool(collider.get("occludes_camera", False))
        and collider.get("primitive") == "box"
    ]
    environment_blockers = [
        collider
        for collider in metadata.get("environment_binding", {}).get(
            "colliders", []
        )
        if bool(collider.get("visible", True))
        and bool(collider.get("collision_enabled", True))
        and collider.get("primitive") == "box"
    ]
    return [*support_blockers, *environment_blockers]


def solve_camera(
    metadata: dict[str, Any], trajectory: dict[str, np.ndarray], rules: dict[str, Any]
) -> dict[str, Any]:
    objects = require_simulation_objects(
        metadata, SUPPORTED_DYNAMIC_OBJECT_COUNTS, __name__
    )
    if len(objects) == 2:
        return _solve_two_object_camera(metadata, trajectory, rules)
    request = metadata["camera_request"]
    profile = str(request["profile"])
    observation = request["observation"]
    obj = objects[0]
    object_id = str(obj["object_id"])
    focus_points, positions, center_indices = sampled_motion_points(
        trajectory,
        object_id,
        observation,
    )
    azimuth_base = camera_azimuth_degrees(rules, profile)
    is_ramp = metadata["simulation"]["support"]["support_shape"] == "inclined_ramp"
    minimum_ramp_side_readability = float(
        rules["camera_observation"][
            "minimum_inclined_surface_side_readability"
        ]
    )
    object_size = np.asarray(
        obj["geometry"]["size_m"], dtype=np.float64
    )
    object_lower = np.asarray(
        trajectory[f"{object_id}__aabb_min_m"], dtype=np.float64
    )
    object_upper = np.asarray(
        trajectory[f"{object_id}__aabb_max_m"], dtype=np.float64
    )
    initial_corners = np.asarray(
        [
            [x, y, z]
            for x in (object_lower[0, 0], object_upper[0, 0])
            for y in (object_lower[0, 1], object_upper[0, 1])
            for z in (object_lower[0, 2], object_upper[0, 2])
        ],
        dtype=np.float64,
    )
    support_top = float(metadata["simulation"]["support"]["surface_center_z_m"])
    primary_target, full_target = camera_target_centers(focus_points, positions)
    structure_target = None
    if str(observation["structure_context"]) in {
        "ramp_and_landing",
        "edge_and_landing",
    }:
        context_points, required_anchor_indices = support_context_points(
            metadata,
            camera_azimuth_degrees(rules, profile),
            primary_target[:2],
            observation,
            positions[center_indices],
        )
        framing_points = np.vstack(
            [positions[center_indices], context_points[required_anchor_indices]]
        )
        structure_target = (
            framing_points.min(axis=0) + framing_points.max(axis=0)
        ) / 2.0
    requested_target_blend = float(request["full_trajectory_camera_target_blend"])
    requested_lens = min(44.0, float(request["focal_length_mm"]))
    scene_visual = metadata["appearance"]["scene_visual"]
    camera_context = scene_visual.get("camera_context", {})
    composition = scene_visual.get("composition")
    reviewed_camera = (
        composition.get("camera", {})
        if isinstance(composition, dict)
        and str(composition.get("review_status")) == "approved"
        else {}
    )
    azimuth_deviation_limits = [
        float(value)
        for value in (
            request.get("maximum_local_azimuth_deviation_degrees"),
            reviewed_camera.get("maximum_local_azimuth_deviation_degrees")
            if reviewed_camera
            else None,
        )
        if value is not None
    ]
    maximum_local_azimuth_deviation = (
        min(azimuth_deviation_limits) if azimuth_deviation_limits else None
    )
    reviewed_preferred_elevation = (
        float(reviewed_camera["preferred_elevation_degrees"])
        if reviewed_camera
        else None
    )
    context_depth_offset = float(
        reviewed_camera["target_depth_offset_m"]
        if reviewed_camera
        else camera_context.get("depth_offset_m", 0.0)
    )
    context_lateral_offset = float(
        reviewed_camera["target_lateral_offset_m"]
        if reviewed_camera
        else camera_context.get("lateral_offset_m", 0.0)
    )
    context_target_z_offset = float(
        reviewed_camera["target_z_offset_m"]
        if reviewed_camera
        else camera_context.get("target_z_offset_m", 0.0)
    )
    context_focal_cap = (
        float(reviewed_camera["focal_length_cap_mm"])
        if reviewed_camera
        else camera_context.get("focal_length_cap_mm")
    )
    if context_focal_cap is not None:
        requested_lens = min(requested_lens, float(context_focal_cap))
    focal_length_candidates = [requested_lens]
    focal_length_candidates.extend(
        value for value in (40.0, 36.0, 32.0, 28.0) if value < requested_lens
    )
    sensor_width = 32.0
    aspect = 16.0 / 9.0
    planar_object_size = max(float(object_size[0]), float(object_size[1]))
    object_size_scale = min(1.60, max(0.80, math.sqrt(planar_object_size / 0.18)))
    preferred_span = float(observation["preferred_object_span_ndc"]) * object_size_scale
    minimum_median_object_span = float(
        observation["minimum_median_object_span_ndc"]
    )
    minimum_center_fraction = float(
        request.get("minimum_primary_trajectory_center_visible_fraction", 0.75)
    )
    minimum_full_center_fraction = float(
        request["minimum_full_trajectory_center_visible_fraction"]
    )
    if str(observation["intent"]) == "ballistic_arc":
        minimum_full_center_fraction = max(
            minimum_full_center_fraction,
            float(
                rules["camera_observation"][
                    "minimum_ballistic_full_trajectory_center_visible_fraction"
                ]
            ),
        )
    minimum_object_span = float(request["minimum_initial_object_span_ndc"])
    minimum_initial_object_visible_fraction = float(
        request.get("minimum_initial_object_visible_fraction", 0.75)
    )
    initial_object_center_margin = float(
        request.get("initial_object_center_margin_ndc", 0.05)
    )
    initial_object_corner_margin = float(
        request.get("initial_object_corner_margin_ndc", 0.0)
    )
    if not 0.0 <= initial_object_center_margin < 0.5:
        raise ValueError("initial object center margin must be in [0, 0.5)")
    if not 0.0 <= initial_object_corner_margin < 0.5:
        raise ValueError("initial object corner margin must be in [0, 0.5)")
    soft_maximum_focus_span = float(request["soft_maximum_focus_span_ndc"])
    maximum_focus_span = float(request["maximum_focus_span_ndc"])
    focus_span_penalty_weight = float(request["focus_span_penalty_weight"])
    if not 0.0 < soft_maximum_focus_span < maximum_focus_span:
        raise ValueError("camera focus-span limits must be positive and ordered")
    maximum_camera_distance = float(request["maximum_camera_distance_m"])
    support_size = np.asarray(metadata["simulation"]["support"]["size_m"][:2], dtype=np.float64)
    support_diagonal = float(np.linalg.norm(support_size))
    minimum_camera_distance = max(
        float(request.get("minimum_camera_distance_floor_m", 1.45)),
        float(request.get("minimum_camera_distance_offset_m", 0.50))
        + support_diagonal
        * float(request.get("minimum_camera_distance_support_diagonal_scale", 0.36)),
    )
    if reviewed_camera:
        minimum_camera_distance = max(
            minimum_camera_distance,
            float(reviewed_camera["minimum_distance_m"]),
        )
    distance_allowance = float(
        request.get("maximum_camera_distance_above_minimum_m", 1.70)
    )
    if camera_context:
        distance_allowance += (
            1.50 * context_depth_offset + 0.50 * context_target_z_offset
        )
    maximum_camera_distance = min(
        maximum_camera_distance,
        minimum_camera_distance + distance_allowance,
    )
    if reviewed_camera:
        maximum_camera_distance = min(
            maximum_camera_distance,
            float(reviewed_camera["maximum_distance_m"]),
        )
    if maximum_camera_distance < minimum_camera_distance:
        raise ValueError(
            "reviewed environment camera corridor cannot frame support for "
            f"{metadata['scene_id']}: minimum={minimum_camera_distance:.6f} "
            f"maximum={maximum_camera_distance:.6f}"
        )
    preferred_camera_distance = minimum_camera_distance + float(
        request.get("preferred_camera_distance_offset_m", 0.25)
    )
    distance_penalty_weight = float(request.get("camera_distance_penalty_weight", 0.08))
    maximum_object_span = float(request.get("maximum_initial_object_span_ndc", 0.22))
    requested_minimum_elevation = request.get("minimum_camera_elevation_degrees")
    requested_maximum_elevation = request.get("maximum_camera_elevation_degrees")
    minimum_elevation_candidates = [
        float(value)
        for value in (
            requested_minimum_elevation,
            camera_context.get("minimum_elevation_degrees"),
            reviewed_camera.get("minimum_elevation_degrees")
            if reviewed_camera
            else None,
        )
        if value is not None
    ]
    maximum_elevation_candidates = [
        float(value)
        for value in (
            requested_maximum_elevation,
            camera_context.get("maximum_elevation_degrees"),
            reviewed_camera.get("maximum_elevation_degrees")
            if reviewed_camera
            else None,
        )
        if value is not None
    ]
    minimum_camera_elevation = (
        max(minimum_elevation_candidates) if minimum_elevation_candidates else None
    )
    maximum_camera_elevation = (
        min(maximum_elevation_candidates) if maximum_elevation_candidates else None
    )
    if (
        minimum_camera_elevation is not None
        and maximum_camera_elevation is not None
        and minimum_camera_elevation > maximum_camera_elevation
    ):
        raise ValueError(
            "camera elevation constraints do not overlap for "
            f"{metadata['scene_id']}: motion_min={requested_minimum_elevation}, "
            f"motion_max={requested_maximum_elevation}, "
            "environment_min="
            f"{camera_context.get('minimum_elevation_degrees')}, "
            "environment_max="
            f"{camera_context.get('maximum_elevation_degrees')}"
        )
    minimum_context_fraction = float(
        request.get("minimum_support_context_visible_fraction", 0.80)
    )
    minimum_anchor_fraction = float(
        observation["minimum_anchor_visible_fraction"]
    )
    minimum_anchor_unoccluded = float(
        observation["minimum_anchor_unoccluded_fraction"]
    )
    minimum_primary_unoccluded = float(
        request.get("minimum_primary_trajectory_unoccluded_fraction", 0.90)
    )
    minimum_full_unoccluded = float(
        request.get("minimum_full_trajectory_unoccluded_fraction", 0.70)
    )
    colliders = camera_occlusion_colliders(metadata)
    has_camera_blockers = bool(colliders)
    full_occlusion_indices = np.unique(
        np.linspace(0, len(positions) - 1, min(18, len(positions)), dtype=int)
    )
    def evaluate_target(
        motion_target: np.ndarray,
        target_blend: float,
        focal_length_mm: float,
        wide_azimuth: bool = False,
        absolute_elevations: tuple[float, ...] | None = None,
        target_mode: str = "motion",
    ) -> list[dict[str, Any]]:
        result = []
        azimuth_offsets = camera_azimuth_offsets(
            maximum_deviation_degrees=maximum_local_azimuth_deviation,
            wide=wide_azimuth,
            has_camera_blockers=has_camera_blockers,
        )
        for azimuth_offset in azimuth_offsets:
            azimuth = math.radians(azimuth_base + azimuth_offset)
            outward_direction = np.asarray(
                [math.cos(azimuth), math.sin(azimuth)], dtype=np.float64
            )
            lateral_direction = np.asarray(
                [-outward_direction[1], outward_direction[0]], dtype=np.float64
            )
            target = np.asarray(motion_target, dtype=np.float64).copy()
            target[:2] += (
                -outward_direction * context_depth_offset
                + lateral_direction * context_lateral_offset
            )
            target[2] += context_target_z_offset
            ramp_side_readability = inclined_surface_side_readability(
                azimuth_base + azimuth_offset
            )
            if (
                is_ramp
                and ramp_side_readability < minimum_ramp_side_readability
            ):
                continue
            context_points, required_anchor_indices = support_context_points(
                metadata,
                azimuth_base + azimuth_offset,
                motion_target[:2],
                observation,
                positions[center_indices],
            )
            base_elevation = (
                reviewed_preferred_elevation
                if reviewed_preferred_elevation is not None
                else camera_elevation_degrees(rules, profile)
            )
            if is_ramp:
                base_elevation = min(base_elevation, 42.0)
            context_elevations = (
                (base_elevation, base_elevation + 10.0, base_elevation - 8.0,
                 base_elevation + 18.0, base_elevation - 14.0)
                if camera_context
                else (base_elevation, base_elevation + 10.0,
                      base_elevation - 8.0, base_elevation + 18.0)
            )
            elevation_candidates = (
                tuple(dict.fromkeys(float(value) for value in absolute_elevations))
                if absolute_elevations is not None
                else context_elevations
            )
            for elevation_degrees in elevation_candidates:
                if (
                    minimum_camera_elevation is not None
                    and elevation_degrees < float(minimum_camera_elevation)
                ):
                    continue
                if (
                    maximum_camera_elevation is not None
                    and elevation_degrees > float(maximum_camera_elevation)
                ):
                    continue
                elevation_offset = elevation_degrees - base_elevation
                elevation = math.radians(elevation_degrees)
                distance_samples = max(
                    48,
                    int(round((maximum_camera_distance - minimum_camera_distance) / 0.05)) + 1,
                )
                for distance in np.linspace(
                    minimum_camera_distance, maximum_camera_distance, distance_samples
                ):
                    horizontal_distance = float(distance) * math.cos(elevation)
                    position = target + np.asarray(
                        [
                            horizontal_distance * math.cos(azimuth),
                            horizontal_distance * math.sin(azimuth),
                            float(distance) * math.sin(elevation),
                        ]
                    )
                    if not camera_inside_structural_envelope(metadata, position):
                        continue
                    projected_focus = project_points(
                        focus_points,
                        position,
                        target,
                        focal_length_mm,
                        sensor_width,
                        aspect,
                    )
                    projected_centers = project_points(
                        positions[center_indices],
                        position,
                        target,
                        focal_length_mm,
                        sensor_width,
                        aspect,
                    )
                    projected_full_centers = project_points(
                        positions,
                        position,
                        target,
                        focal_length_mm,
                        sensor_width,
                        aspect,
                    )
                    projected_context = project_points(
                        context_points,
                        position,
                        target,
                        focal_length_mm,
                        sensor_width,
                        aspect,
                    )
                    projected_initial_corners = project_points(
                        initial_corners,
                        position,
                        target,
                        focal_length_mm,
                        sensor_width,
                        aspect,
                    )
                    center_inside = image_center_visibility_mask(projected_centers)
                    center_fraction = float(center_inside.mean())
                    full_center_inside = image_center_visibility_mask(
                        projected_full_centers
                    )
                    full_center_fraction = float(full_center_inside.mean())
                    initial_inside = bool(
                        initial_object_center_margin
                        <= projected_centers[0, 0]
                        <= 1.0 - initial_object_center_margin
                        and initial_object_center_margin
                        <= projected_centers[0, 1]
                        <= 1.0 - initial_object_center_margin
                        and projected_centers[0, 2] > 0.1
                    )
                    initial_corner_inside = np.logical_and.reduce(
                        [
                            projected_initial_corners[:, 0]
                            >= initial_object_corner_margin,
                            projected_initial_corners[:, 0]
                            <= 1.0 - initial_object_corner_margin,
                            projected_initial_corners[:, 1]
                            >= initial_object_corner_margin,
                            projected_initial_corners[:, 1]
                            <= 1.0 - initial_object_corner_margin,
                            projected_initial_corners[:, 2] > 0.1,
                        ]
                    )
                    initial_object_visible_fraction = float(initial_corner_inside.mean())
                    initial_object_inside = (
                        initial_object_visible_fraction
                        >= minimum_initial_object_visible_fraction
                    )
                    context_inside = np.logical_and.reduce(
                        [
                            projected_context[:, 0] >= -0.12,
                            projected_context[:, 0] <= 1.12,
                            projected_context[:, 1] >= -0.12,
                            projected_context[:, 1] <= 1.12,
                            projected_context[:, 2] > 0.1,
                        ]
                    )
                    context_fraction = float(context_inside.mean())
                    anchor_fraction = float(
                        context_inside[required_anchor_indices].mean()
                    )
                    object_span = max(
                        float(np.ptp(projected_initial_corners[:, 0])),
                        float(np.ptp(projected_initial_corners[:, 1])),
                    )
                    primary_object_spans = []
                    for index in center_indices:
                        corners = np.asarray(
                            [
                                [x, y, z]
                                for x in (object_lower[index, 0], object_upper[index, 0])
                                for y in (object_lower[index, 1], object_upper[index, 1])
                                for z in (object_lower[index, 2], object_upper[index, 2])
                            ],
                            dtype=np.float64,
                        )
                        projected = project_points(
                            corners,
                            position,
                            target,
                            focal_length_mm,
                            sensor_width,
                            aspect,
                        )
                        primary_object_spans.append(
                            max(
                                float(np.ptp(projected[:, 0])),
                                float(np.ptp(projected[:, 1])),
                            )
                        )
                    median_object_span = float(np.median(primary_object_spans))
                    focus_span = max(
                        float(np.ptp(projected_focus[:, 0])),
                        float(np.ptp(projected_focus[:, 1])),
                    )
                    primary_unoccluded = unoccluded_fraction(
                        position, positions[center_indices], colliders
                    )
                    full_unoccluded = unoccluded_fraction(
                        position, positions[full_occlusion_indices], colliders
                    )
                    target_unoccluded = unoccluded_fraction(
                        position, target[None, :], colliders
                    )
                    required_anchors_unoccluded = unoccluded_fraction(
                        position,
                        context_points[required_anchor_indices],
                        colliders,
                    )
                    admissible = (
                        initial_inside
                        and initial_object_inside
                        and center_fraction >= minimum_center_fraction
                        and full_center_fraction >= minimum_full_center_fraction
                        and context_fraction >= minimum_context_fraction
                        and anchor_fraction >= minimum_anchor_fraction
                        and focus_span <= maximum_focus_span
                        and object_span >= minimum_object_span
                        and object_span <= maximum_object_span
                        and median_object_span >= minimum_median_object_span
                        and primary_unoccluded >= minimum_primary_unoccluded
                        and full_unoccluded >= minimum_full_unoccluded
                        and required_anchors_unoccluded
                        >= minimum_anchor_unoccluded
                    )
                    score = (
                        abs(median_object_span - preferred_span) * 5.0
                        + (1.0 - center_fraction) * 2.0
                        + (1.0 - full_center_fraction) * 0.65
                        + (1.0 - context_fraction) * 0.75
                        + (1.0 - anchor_fraction) * 1.25
                        + (1.0 - primary_unoccluded) * 3.0
                        + (1.0 - full_unoccluded) * 0.8
                        + (1.0 - required_anchors_unoccluded) * 1.25
                        + max(0.0, focus_span - soft_maximum_focus_span)
                        * focus_span_penalty_weight
                        + abs(float(distance) - preferred_camera_distance)
                        * distance_penalty_weight
                        + abs(azimuth_offset) * 0.001
                        + abs(elevation_offset) * 0.015
                        + (1.0 - ramp_side_readability) * (0.8 if is_ramp else 0.0)
                    )
                    result.append(
                        {
                            "admissible": admissible,
                            "score": score,
                            "position": position,
                            "target": target,
                            "motion_target": motion_target,
                            "target_blend": target_blend,
                            "target_mode": target_mode,
                            "distance": float(distance),
                            "focal_length_mm": float(focal_length_mm),
                            "azimuth_degrees": azimuth_base + azimuth_offset,
                            "elevation_degrees": math.degrees(elevation),
                            "center_fraction": center_fraction,
                            "full_center_fraction": full_center_fraction,
                            "context_fraction": context_fraction,
                            "anchor_fraction": anchor_fraction,
                            "object_span": object_span,
                            "median_object_span": median_object_span,
                            "initial_object_visible_fraction": initial_object_visible_fraction,
                            "focus_span": focus_span,
                            "primary_unoccluded_fraction": primary_unoccluded,
                            "full_unoccluded_fraction": full_unoccluded,
                            "target_unoccluded_fraction": target_unoccluded,
                            "required_structure_anchors_unoccluded_fraction": (
                                required_anchors_unoccluded
                            ),
                            "ramp_side_readability": ramp_side_readability,
                            "constraints": {
                                "initial_center_inside": initial_inside,
                                "initial_object_inside": initial_object_inside,
                                "primary_center_fraction": center_fraction >= minimum_center_fraction,
                                "full_center_fraction": full_center_fraction >= minimum_full_center_fraction,
                                "support_context_fraction": context_fraction >= minimum_context_fraction,
                                "required_structure_anchor_fraction": anchor_fraction >= minimum_anchor_fraction,
                                "focus_span": focus_span <= maximum_focus_span,
                                "object_span": object_span >= minimum_object_span,
                                "maximum_object_span": object_span <= maximum_object_span,
                                "median_object_span": median_object_span >= minimum_median_object_span,
                                "primary_unoccluded_fraction": primary_unoccluded >= minimum_primary_unoccluded,
                                "full_unoccluded_fraction": full_unoccluded >= minimum_full_unoccluded,
                                "required_structure_anchors_unoccluded_fraction": (
                                    required_anchors_unoccluded
                                    >= minimum_anchor_unoccluded
                                ),
                            },
                        }
                    )
        return result

    candidates: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []
    attempted_blends = []
    attempted_focal_lengths = []
    used_safety_elevation_fallback = False

    def search_focal_length(focal_length_mm: float) -> list[dict[str, Any]]:
        nonlocal used_safety_elevation_fallback
        for target_blend in dict.fromkeys(
            (requested_target_blend, requested_target_blend / 2.0, 0.0)
        ):
            attempted_blends.append(target_blend)
            motion_target = (
                (1.0 - target_blend) * primary_target
                + target_blend * full_target
            )
            motion_target[2] = max(
                max(0.18, support_top - 0.45),
                min(float(motion_target[2]), support_top + 0.78),
            )
            target_candidates = evaluate_target(
                motion_target,
                target_blend,
                focal_length_mm,
            )
            candidates.extend(target_candidates)
            accepted = [
                record for record in target_candidates if record["admissible"]
            ]
            if accepted:
                return accepted
        if structure_target is not None:
            target_candidates = evaluate_target(
                structure_target,
                requested_target_blend,
                focal_length_mm,
                wide_azimuth=True,
                target_mode="motion_and_required_structure",
            )
            candidates.extend(target_candidates)
            accepted = [
                record for record in target_candidates if record["admissible"]
            ]
            if accepted:
                return accepted
        for initial_bias in (0.20, 0.35, 0.50):
            attempted_blends.append(-initial_bias)
            motion_target = (
                (1.0 - initial_bias) * primary_target
                + initial_bias * positions[0]
            )
            motion_target[2] = max(
                max(0.18, support_top - 0.45),
                min(float(motion_target[2]), support_top + 0.78),
            )
            target_candidates = evaluate_target(
                motion_target,
                -initial_bias,
                focal_length_mm,
                wide_azimuth=initial_bias >= 0.35,
            )
            candidates.extend(target_candidates)
            accepted = [
                record for record in target_candidates if record["admissible"]
            ]
            if accepted:
                return accepted
        safety_elevations = (20.0, 24.0, 27.0, 32.0, 38.0, 42.0, 48.0)
        for target_blend in dict.fromkeys(
            (requested_target_blend, requested_target_blend / 2.0, 0.0)
        ):
            motion_target = (
                (1.0 - target_blend) * primary_target
                + target_blend * full_target
            )
            motion_target[2] = max(
                max(0.18, support_top - 0.45),
                min(float(motion_target[2]), support_top + 0.78),
            )
            target_candidates = evaluate_target(
                motion_target,
                target_blend,
                focal_length_mm,
                wide_azimuth=True,
                absolute_elevations=safety_elevations,
            )
            candidates.extend(target_candidates)
            accepted = [
                record for record in target_candidates if record["admissible"]
            ]
            if accepted:
                used_safety_elevation_fallback = True
                return accepted
        return []

    for focal_length_mm in focal_length_candidates:
        attempted_focal_lengths.append(focal_length_mm)
        admitted = search_focal_length(focal_length_mm)
        if admitted:
            break
    if not candidates:
        raise ValueError(
            f"camera solver evaluated no poses for {metadata['scene_id']}"
        )
    best = min(admitted or candidates, key=lambda record: record["score"])
    camera_variant_index = 0
    if admitted:
        # Geometry decides which poses are valid. The scene seed only chooses
        # among near-equivalent valid poses, preserving reproducibility while
        # avoiding a systematic collapse to the nominal camera angle.
        quality_limit = float(best["score"]) + 0.65
        quality_pool = [
            record for record in admitted if float(record["score"]) <= quality_limit
        ]
        scene_digest = hashlib.sha256(
            f"generic-camera:{metadata['scene_id']}".encode("utf-8")
        ).digest()
        azimuths = sorted(
            {round(float(record["azimuth_degrees"]), 6) for record in quality_pool}
        )
        elevations = sorted(
            {round(float(record["elevation_degrees"]), 6) for record in quality_pool}
        )
        preferred_azimuth = azimuths[int(scene_digest[0]) % len(azimuths)]
        preferred_elevation = elevations[int(scene_digest[1]) % len(elevations)]
        ranked_pool = sorted(
            quality_pool,
            key=lambda record: (
                abs(float(record["azimuth_degrees"]) - preferred_azimuth) / 24.0
                + abs(float(record["elevation_degrees"]) - preferred_elevation) / 8.0,
                float(record["score"]),
            ),
        )
        best = ranked_pool[0]
        camera_variant_index = next(
            index
            for index, candidate in enumerate(quality_pool)
            if candidate is best
        )
    if not best["admissible"]:
        largest_object = max(candidates, key=lambda record: record["object_span"])
        closest_to_object_threshold = min(
            candidates,
            key=lambda record: abs(record["object_span"] - minimum_object_span),
        )
        best_structure_target = min(
            (
                record
                for record in candidates
                if record["target_mode"] == "motion_and_required_structure"
            ),
            key=lambda record: record["score"],
            default=None,
        )
        best_anchor_coverage = min(
            candidates,
            key=lambda record: (-record["anchor_fraction"], record["score"]),
        )
        raise ValueError(
            f"camera solver found no admissible pose for {metadata['scene_id']}: "
            f"best={best}; closest_object_threshold={closest_to_object_threshold}; "
            f"largest_object={largest_object}; "
            f"best_structure_target={best_structure_target}; "
            f"best_anchor_coverage={best_anchor_coverage}"
        )
    return {
        "solver_version": "motion_structure_camera_v12",
        "profile": profile,
        "observation_intent": str(observation["intent"]),
        "structure_context": str(observation["structure_context"]),
        "position_m": [round(float(value), 6) for value in best["position"]],
        "target_m": [round(float(value), 6) for value in best["target"]],
        "focal_length_mm": round(float(best["focal_length_mm"]), 6),
        "sensor_width_mm": sensor_width,
        "clip_start_m": 0.05,
        "clip_end_m": 100.0,
        "diagnostics": {
            "deterministic_variant_index": camera_variant_index,
            "selected_azimuth_degrees": round(float(best["azimuth_degrees"]), 6),
            "selected_elevation_degrees": round(float(best["elevation_degrees"]), 6),
            "evaluated_candidates": len(candidates),
            "attempted_full_trajectory_target_blends": [
                round(float(value), 6) for value in attempted_blends
            ],
            "attempted_focal_lengths_mm": [
                round(float(value), 6) for value in attempted_focal_lengths
            ],
            "selected_focal_length_mm": round(
                float(best["focal_length_mm"]), 6
            ),
            "selected_full_trajectory_target_blend": round(
                float(best["target_blend"]), 6
            ),
            "selected_target_mode": str(best["target_mode"]),
            "motion_target_m": [
                round(float(value), 6) for value in best["motion_target"]
            ],
            "environment_camera_context": {
                "enabled": bool(camera_context),
                "depth_offset_m": round(context_depth_offset, 6),
                "lateral_offset_m": round(context_lateral_offset, 6),
                "target_z_offset_m": round(context_target_z_offset, 6),
                "focal_length_cap_mm": (
                    round(float(context_focal_cap), 6)
                    if context_focal_cap is not None
                    else None
                ),
                "minimum_elevation_degrees": (
                    round(float(minimum_camera_elevation), 6)
                    if minimum_camera_elevation is not None
                    else None
                ),
                "maximum_elevation_degrees": (
                    round(float(maximum_camera_elevation), 6)
                    if maximum_camera_elevation is not None
                    else None
                ),
                "reviewed_camera_corridor": (
                    {
                        "preferred_local_azimuth_degrees": round(
                            float(
                                reviewed_camera[
                                    "preferred_local_azimuth_degrees"
                                ]
                            ),
                            6,
                        ),
                        "maximum_local_azimuth_deviation_degrees": round(
                            float(maximum_local_azimuth_deviation), 6
                        ),
                        "preferred_elevation_degrees": round(
                            float(reviewed_preferred_elevation), 6
                        ),
                        "minimum_distance_m": round(
                            float(reviewed_camera["minimum_distance_m"]), 6
                        ),
                        "maximum_distance_m": round(
                            float(reviewed_camera["maximum_distance_m"]), 6
                        ),
                        "target_depth_offset_m": round(
                            float(reviewed_camera["target_depth_offset_m"]), 6
                        ),
                        "target_lateral_offset_m": round(
                            float(reviewed_camera["target_lateral_offset_m"]),
                            6,
                        ),
                        "target_z_offset_m": round(
                            float(reviewed_camera["target_z_offset_m"]), 6
                        ),
                        "focal_length_cap_mm": round(
                            float(reviewed_camera["focal_length_cap_mm"]), 6
                        ),
                    }
                    if reviewed_camera
                    else None
                ),
            },
            "used_safety_elevation_fallback": used_safety_elevation_fallback,
            "score": round(float(best["score"]), 6),
            "distance_m": round(float(best["distance"]), 6),
            "minimum_distance_m": round(float(minimum_camera_distance), 6),
            "preferred_distance_m": round(float(preferred_camera_distance), 6),
            "maximum_distance_m": round(float(maximum_camera_distance), 6),
            "azimuth_degrees": round(float(best["azimuth_degrees"]), 6),
            "elevation_degrees": round(float(best["elevation_degrees"]), 6),
            "ramp_side_readability": round(
                float(best["ramp_side_readability"]), 6
            ),
            "primary_center_visible_fraction": round(float(best["center_fraction"]), 6),
            "primary_center_evaluated_frame_count": int(len(center_indices)),
            "full_trajectory_center_visible_fraction": round(
                float(best["full_center_fraction"]), 6
            ),
            "full_trajectory_center_evaluated_frame_count": int(len(positions)),
            "center_visibility_bounds_ndc": [0.0, 1.0],
            "support_context_visible_fraction": round(float(best["context_fraction"]), 6),
            "required_structure_anchor_visible_fraction": round(
                float(best["anchor_fraction"]), 6
            ),
            "initial_object_span_ndc": round(float(best["object_span"]), 6),
            "initial_object_visible_fraction": round(
                float(best["initial_object_visible_fraction"]), 6
            ),
            "preferred_object_span_ndc": round(float(preferred_span), 6),
            "median_primary_object_span_ndc": round(
                float(best["median_object_span"]), 6
            ),
            "focus_span_ndc": round(float(best["focus_span"]), 6),
            "primary_trajectory_unoccluded_fraction": round(
                float(best["primary_unoccluded_fraction"]), 6
            ),
            "full_trajectory_unoccluded_fraction": round(
                float(best["full_unoccluded_fraction"]), 6
            ),
            "target_unoccluded_fraction": round(
                float(best["target_unoccluded_fraction"]), 6
            ),
            "required_structure_anchors_unoccluded_fraction": round(
                float(best["required_structure_anchors_unoccluded_fraction"]),
                6,
            ),
            "minimum_required_structure_anchor_unoccluded_fraction": round(
                minimum_anchor_unoccluded, 6
            ),
        },
    }
