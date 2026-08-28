#!/usr/bin/env python3
"""Trajectory metrics and semantic QA for PhysSweep rigid simulations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from tools.motion_rules.one_object import MotionAuditContext, audit_motion
from tools.motion_rules.one_object.common import (
    distance_lower_bound as _distance_lower_bound,
    distance_upper_bound as _distance_upper_bound,
)
from tools.physics.physics_invariants import PROXY_SHAPE_CODE, additional_physics_invariants


def _horizontal_displacement(positions: np.ndarray) -> float:
    delta = positions[-1, :2] - positions[0, :2]
    return float(np.linalg.norm(delta))


def _path_length(positions: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def _first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return int(indices[0]) if indices.size else None


def active_motion_duration_s(
    linear_velocity_m_s: np.ndarray,
    time_s: np.ndarray,
    speed_threshold_m_s: float,
) -> float:
    velocities = np.asarray(linear_velocity_m_s, dtype=np.float64)
    times = np.asarray(time_s, dtype=np.float64)
    if velocities.ndim != 2 or velocities.shape[1] != 3:
        raise ValueError("linear velocity must have shape [frame, 3]")
    if times.ndim != 1 or times.shape[0] != velocities.shape[0]:
        raise ValueError("time samples must match linear velocity frames")
    if float(speed_threshold_m_s) <= 0.0:
        raise ValueError("active speed threshold must be positive")
    active = np.flatnonzero(
        np.linalg.norm(velocities, axis=1) >= float(speed_threshold_m_s)
    )
    return float(times[active[-1]] - times[0]) if active.size else 0.0


def audit_support_transition_contract(
    metadata: dict[str, Any],
    expected: dict[str, Any],
    primary_contacts: np.ndarray,
    destination_contacts: np.ndarray | None,
    destination_contact_index: int | None,
    check: Any,
) -> None:
    support = metadata["simulation"]["support"]
    contract = support.get("transition_contract")
    if contract is None:
        return
    required_keys = {
        "version",
        "type",
        "source_collider_id",
        "destination_collider_id",
        "boundary_point_m",
        "outward_direction_xy",
        "source_boundary_height_m",
        "destination_surface_height_m",
        "height_drop_m",
        "intermediate_phase",
        "required_contact_sequence",
        "allow_source_recontact_after_destination",
    }
    missing = sorted(required_keys.difference(contract))
    if missing:
        raise ValueError(f"transition contract is incomplete: {missing}")
    colliders = {str(value["id"]): value for value in support["colliders"]}
    source_id = str(contract["source_collider_id"])
    destination_id = str(contract["destination_collider_id"])
    if source_id not in colliders or destination_id not in colliders:
        raise ValueError("transition contract references an unknown collider")
    if colliders[source_id]["role"] != "primary_support":
        raise ValueError("transition source is not the primary support")
    if colliders[destination_id]["role"] not in {
        "landing_surface",
        "environment_floor",
    }:
        raise ValueError("transition destination is not a landing surface")
    if str(expected.get("transition_contract_version")) != str(contract["version"]):
        raise ValueError("expected motion targets a different transition contract")
    if str(expected.get("required_collider_contact_id")) != destination_id:
        raise ValueError("expected motion targets a different destination collider")
    if contract["required_contact_sequence"] != ["source", "destination"]:
        raise ValueError("unsupported transition contact sequence")
    outward = np.asarray(contract["outward_direction_xy"], dtype=np.float64)
    if outward.shape != (2,) or not np.isfinite(outward).all():
        raise ValueError("transition outward direction must be a finite 2D vector")
    outward_norm = float(np.linalg.norm(outward))
    if abs(outward_norm - 1.0) > 1.0e-6:
        raise ValueError("transition outward direction must be normalized")
    boundary = np.asarray(contract["boundary_point_m"], dtype=np.float64)
    if boundary.shape != (3,) or not np.isfinite(boundary).all():
        raise ValueError("transition boundary point must be finite 3D")
    source_height = float(contract["source_boundary_height_m"])
    destination_height = float(contract["destination_surface_height_m"])
    height_drop = float(contract["height_drop_m"])
    if abs(float(boundary[2]) - source_height) > 1.0e-6:
        raise ValueError("transition boundary height is inconsistent")
    if abs(height_drop - max(0.0, source_height - destination_height)) > 1.0e-6:
        raise ValueError("transition height drop is inconsistent")
    destination = colliders[destination_id]
    destination_top = float(destination["position_m"][2]) + 0.5 * float(
        destination["size_m"][2]
    )
    if abs(destination_height - destination_top) > 1.0e-6:
        raise ValueError("transition destination height differs from its collider")
    transition_type = str(contract["type"])
    phase = str(contract["intermediate_phase"])
    expected_phase = {
        "raised_edge_to_floor": "airborne",
        "incline_to_horizontal": "continuous_contact",
    }.get(transition_type)
    if expected_phase is None or phase != expected_phase:
        raise ValueError("transition type and intermediate phase disagree")
    if transition_type == "raised_edge_to_floor":
        source = colliders[source_id]
        source_top = float(source["position_m"][2]) + 0.5 * float(
            source["size_m"][2]
        )
        if abs(source_height - source_top) > 1.0e-6:
            raise ValueError("raised transition height differs from its source collider")
    initial_velocity = np.asarray(
        metadata["simulation"]["objects"][0]["initial_state"][
            "linear_velocity_m_s"
        ][:2],
        dtype=np.float64,
    )
    initial_speed_xy = float(np.linalg.norm(initial_velocity))
    if initial_speed_xy <= 1.0e-8 or float(initial_velocity @ outward) <= 0.0:
        raise ValueError("transition initial velocity does not point toward the boundary")
    source_indices = np.flatnonzero(primary_contacts > 0)
    contact_order_valid = bool(
        source_indices.size
        and destination_contact_index is not None
        and int(source_indices[0]) < destination_contact_index
    )
    check(
        "transition_contact_order",
        contact_order_valid,
        {
            "first_source_contact_frame": (
                int(source_indices[0]) if source_indices.size else None
            ),
            "first_destination_contact_frame": destination_contact_index,
        },
        "source contact before destination contact",
    )
    if destination_contacts is None or destination_contact_index is None:
        return
    destination_only = np.flatnonzero(
        (destination_contacts > 0) & (primary_contacts == 0)
    )
    destination_only_index = (
        int(destination_only[0]) if destination_only.size else None
    )
    check(
        "transition_destination_only_contact",
        destination_only_index is not None,
        destination_only_index,
        "at_least_one_destination_only_frame",
    )
    if not bool(contract["allow_source_recontact_after_destination"]):
        no_source_recontact = bool(
            destination_only_index is not None
            and not np.any(primary_contacts[destination_only_index + 1 :] > 0)
        )
        check(
            "transition_no_source_recontact",
            no_source_recontact,
            no_source_recontact,
            True,
        )
    if phase == "airborne":
        source_before_destination = source_indices[
            source_indices < destination_contact_index
        ]
        last_source = (
            int(source_before_destination[-1])
            if source_before_destination.size
            else None
        )
        airborne_frames = (
            max(0, destination_contact_index - last_source - 1)
            if last_source is not None
            else 0
        )
        check(
            "transition_airborne_phase",
            airborne_frames >= 1,
            airborne_frames,
            1,
        )
    elif phase != "continuous_contact":
        raise ValueError(f"unsupported transition intermediate phase: {phase}")


def audit_trajectory(metadata: dict[str, Any], trajectory: dict[str, np.ndarray]) -> dict[str, Any]:
    obj = metadata["simulation"]["objects"][0]
    object_id = str(obj["object_id"])
    positions = np.asarray(trajectory[f"{object_id}__position_m"], dtype=np.float64)
    velocities = np.asarray(trajectory[f"{object_id}__linear_velocity_m_s"], dtype=np.float64)
    angular = np.asarray(trajectory[f"{object_id}__angular_velocity_rad_s"], dtype=np.float64)
    quaternions = np.asarray(
        trajectory[f"{object_id}__quaternion_wxyz"], dtype=np.float64
    )
    primary_contacts = np.asarray(
        trajectory[f"{object_id}__primary_support_contact_count"], dtype=np.int64
    )
    all_contacts = np.asarray(trajectory[f"{object_id}__all_contact_count"], dtype=np.int64)
    contact_distance = np.asarray(
        trajectory[f"{object_id}__minimum_contact_distance_m"], dtype=np.float64
    )
    expected = obj["expected_motion"]
    motion = str(expected["motion_family"])
    finite = bool(
        np.isfinite(positions).all()
        and np.isfinite(velocities).all()
        and np.isfinite(angular).all()
    )
    first_primary_contact = _first_true(primary_contacts > 0)
    max_penetration = max(0.0, -float(np.min(contact_distance)))
    speed = np.linalg.norm(velocities, axis=1)
    angular_speed = np.linalg.norm(angular, axis=1)
    displacement_xy = _horizontal_displacement(positions)
    path_length_xy = float(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum())
    path_length_3d = _path_length(positions)
    support_fraction = float(np.mean(primary_contacts > 0))
    limits = metadata.get("qa", {}).get("limits", {})
    absolute_penetration_limit = float(
        limits.get("maximum_trajectory_penetration_m", 0.008)
    )
    relative_penetration_limit = float(
        limits.get("maximum_trajectory_penetration_fraction_of_min_extent", 0.10)
    )
    maximum_initial_penetration = float(
        limits.get("maximum_initial_penetration_m", 0.0005)
    )
    absolute_distance_tolerance = float(
        limits.get("audit_distance_absolute_tolerance_m", 0.0001)
    )
    relative_distance_tolerance = float(
        limits.get("audit_distance_relative_tolerance_fraction", 0.02)
    )
    dimensionless_ratio_tolerance = float(
        limits.get("audit_dimensionless_ratio_tolerance", 0.02)
    )
    extremum_interval_error_multiplier = float(
        limits.get("audit_extremum_interval_error_multiplier", 2.0)
    )
    configured_maximum_linear_speed = float(
        limits.get("maximum_linear_speed_m_s", 5.2)
    )
    gravity = abs(
        float(metadata.get("simulation", {}).get("world", {}).get(
            "gravity_m_s2", [0.0, 0.0, -9.81]
        )[2])
    )
    initial_speed = float(speed[0])
    gravitational_drop = max(0.0, float(positions[0, 2] - positions[:, 2].min()))
    energy_consistent_speed = math.sqrt(
        initial_speed * initial_speed + 2.0 * gravity * gravitational_drop
    )
    maximum_linear_speed = max(
        configured_maximum_linear_speed,
        1.15 * energy_consistent_speed,
    )
    maximum_angular_speed = float(limits.get("maximum_angular_speed_rad_s", 120.0))
    maximum_surface_speed = float(limits.get("maximum_rotational_surface_speed_m_s", 4.5))
    object_size = np.asarray(obj["geometry"]["size_m"], dtype=np.float64)
    maximum_penetration = min(
        absolute_penetration_limit,
        relative_penetration_limit * float(np.min(object_size)),
    )
    initial_penetration = max(0.0, -float(contact_distance[0]))
    contact_radius = float(np.min(object_size) / 2.0)
    rotational_surface_speed = angular_speed * contact_radius
    active_duration_s = active_motion_duration_s(
        velocities,
        np.asarray(trajectory["time_s"], dtype=np.float64),
        float(expected.get("active_speed_threshold_m_s", 0.03)),
    )
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, value: Any, threshold: Any) -> None:
        checks.append(
            {
                "id": check_id,
                "passed": bool(passed),
                "value": value,
                "threshold": threshold,
            }
        )

    check("finite_state", finite, finite, True)
    check(
        "bounded_initial_penetration",
        initial_penetration
        <= _distance_upper_bound(
            maximum_initial_penetration,
            absolute_distance_tolerance,
            relative_distance_tolerance,
        ),
        round(initial_penetration, 6),
        _distance_upper_bound(
            maximum_initial_penetration,
            absolute_distance_tolerance,
            relative_distance_tolerance,
        ),
    )
    check(
        "bounded_penetration",
        max_penetration
        <= _distance_upper_bound(
            maximum_penetration,
            absolute_distance_tolerance,
            relative_distance_tolerance,
        ),
        round(max_penetration, 6),
        _distance_upper_bound(
            maximum_penetration,
            absolute_distance_tolerance,
            relative_distance_tolerance,
        ),
    )
    check(
        "bounded_linear_speed",
        float(speed.max()) <= maximum_linear_speed,
        round(float(speed.max()), 6),
        maximum_linear_speed,
    )
    check(
        "bounded_angular_speed",
        float(angular_speed.max()) <= maximum_angular_speed,
        round(float(angular_speed.max()), 6),
        maximum_angular_speed,
    )
    check(
        "bounded_rotational_surface_speed",
        float(rotational_surface_speed.max()) <= maximum_surface_speed,
        round(float(rotational_surface_speed.max()), 6),
        maximum_surface_speed,
    )
    minimum_displacement = float(expected.get("minimum_displacement_m", 0.0))
    if minimum_displacement > 0.0:
        check(
            "visible_motion",
            path_length_3d
            >= _distance_lower_bound(
                minimum_displacement,
                absolute_distance_tolerance,
                relative_distance_tolerance,
            ),
            round(path_length_3d, 6),
            round(
                _distance_lower_bound(
                    minimum_displacement,
                    absolute_distance_tolerance,
                    relative_distance_tolerance,
                ),
                6,
            ),
        )
    minimum_active_duration_s = float(
        expected.get("minimum_active_duration_s", 0.0)
    )
    if minimum_active_duration_s > 0.0:
        check(
            "useful_active_duration",
            active_duration_s + 1.0e-9 >= minimum_active_duration_s,
            round(active_duration_s, 6),
            minimum_active_duration_s,
        )
    if expected.get("must_contact_primary_support"):
        check(
            "primary_support_contact",
            first_primary_contact is not None,
            first_primary_contact,
            "at_least_one_frame",
        )

    required_collider_id = expected.get("required_collider_contact_id")
    required_contacts = None
    required_contact_index = None
    if required_collider_id:
        required_key = f"{object_id}__collider_contact_count__{required_collider_id}"
        if required_key not in trajectory:
            raise ValueError(f"trajectory is missing required collider contacts: {required_key}")
        required_contacts = np.asarray(trajectory[required_key], dtype=np.int64)
        required_contact_index = _first_true(required_contacts > 0)
        check(
            "required_collider_contact",
            required_contact_index is not None,
            required_contact_index,
            str(required_collider_id),
        )

    audit_support_transition_contract(
        metadata,
        expected,
        primary_contacts,
        required_contacts,
        required_contact_index,
        check,
    )

    audit_motion(
        MotionAuditContext(
            metadata=metadata,
            trajectory=trajectory,
            obj=obj,
            object_id=object_id,
            expected=expected,
            motion=motion,
            positions=positions,
            velocities=velocities,
            angular=angular,
            primary_contacts=primary_contacts,
            all_contacts=all_contacts,
            speed=speed,
            angular_speed=angular_speed,
            support_fraction=support_fraction,
            first_primary_contact=first_primary_contact,
            required_contacts=required_contacts,
            required_contact_index=required_contact_index,
            limits=limits,
            absolute_distance_tolerance=absolute_distance_tolerance,
            relative_distance_tolerance=relative_distance_tolerance,
            dimensionless_ratio_tolerance=dimensionless_ratio_tolerance,
            extremum_interval_error_multiplier=extremum_interval_error_multiplier,
            gravity=gravity,
            check=check,
        )
    )

    runtime_dynamics_key = f"{object_id}__runtime_dynamics"
    runtime_support_key = f"{object_id}__runtime_support_dynamics"
    runtime_support = np.asarray(trajectory[runtime_support_key], dtype=np.float64)
    support_dynamics = metadata["simulation"]["support"]["dynamics"]
    expected_support = np.tile(
        np.asarray(
            [
                support_dynamics["lateral_friction"],
                support_dynamics["restitution"],
            ],
            dtype=np.float64,
        ),
        (runtime_support.shape[0], 1),
    )
    support_dynamics_error = float(np.max(np.abs(runtime_support - expected_support)))
    parameter_tolerance = float(limits.get("parameter_match_absolute_tolerance", 1.0e-6))
    check(
        "pybullet_support_dynamics_match_metadata",
        support_dynamics_error <= parameter_tolerance,
        round(support_dynamics_error, 9),
        parameter_tolerance,
    )
    geometry_type = str(obj["geometry"]["type"])
    additional = additional_physics_invariants(
        motion=motion,
        time_s=np.asarray(trajectory["time_s"], dtype=np.float64),
        positions=positions,
        quaternions_wxyz=quaternions,
        linear_velocity=velocities,
        angular_velocity=angular,
        contact_mask=all_contacts > 0,
        shape=geometry_type,
        size_m=np.asarray(obj["geometry"]["size_m"], dtype=np.float64),
        mass_kg=float(obj["material"]["mass_kg"]),
        gravity_m_s2=np.asarray(
            metadata["simulation"]["world"]["gravity_m_s2"], dtype=np.float64
        ),
        initial_state=obj["initial_state"],
        material=obj["material"],
        quality=limits,
        runtime_dynamics=np.asarray(
            trajectory[runtime_dynamics_key], dtype=np.float64
        ),
        runtime_inertia_diagonal_kg_m2=np.asarray(
            trajectory[f"{object_id}__runtime_inertia_diagonal_kg_m2"],
            dtype=np.float64,
        ),
        expected_proxy_shape_codes=np.asarray(
            [PROXY_SHAPE_CODE[geometry_type]], dtype=np.int32
        ),
        expected_proxy_dimensions_m=np.asarray(
            [obj["geometry"]["size_m"]], dtype=np.float64
        ),
        expected_proxy_positions_m=np.zeros((1, 3), dtype=np.float64),
        expected_proxy_quaternions_xyzw=np.asarray(
            [[0.0, 0.0, 0.0, 1.0]], dtype=np.float64
        ),
        runtime_proxy_shape_codes=np.asarray(
            trajectory[f"{object_id}__runtime_proxy_shape_codes"], dtype=np.int32
        ),
        runtime_proxy_dimensions_m=np.asarray(
            trajectory[f"{object_id}__runtime_proxy_dimensions_m"], dtype=np.float64
        ),
        runtime_proxy_positions_m=np.asarray(
            trajectory[f"{object_id}__runtime_proxy_positions_m"], dtype=np.float64
        ),
        runtime_proxy_quaternions_xyzw=np.asarray(
            trajectory[f"{object_id}__runtime_proxy_quaternions_xyzw"],
            dtype=np.float64,
        ),
        coulomb_friction_utilization=np.asarray(
            trajectory[f"{object_id}__maximum_coulomb_friction_utilization"],
            dtype=np.float64,
        ),
        validate_primitive_inertia_geometry=True,
    )
    checks.extend(additional["checks"])

    advisory_ids = set()
    if "sweep" in metadata:
        # A material sweep may legitimately change the visible motion mode:
        # low restitution can remove the rebound and strong dissipation can
        # make the object settle before the base scene's duration target. The
        # same applies when friction prevents a planned edge exit or contact
        # event; physical validity checks remain hard failures below.
        advisory_ids = {
            "useful_active_duration",
            "visible_rebound",
            "wall_post_impact_rebound_speed",
            "required_collider_contact",
            "primary_support_exit",
            "edge_vertical_drop",
            "transition_contact_order",
            "transition_destination_only_contact",
            "transition_no_source_recontact",
            "minimum_post_transition_travel",
            "bounded_angular_speed",
            "visible_motion",
            "ramp_transition_downhill_extent",
            "no_unplanned_environment_contact",
            "downhill_reversal",
            "uphill_extent",
            "downhill_extent",
        }
        # A supported object on an incline can remain slow while gravity's
        # tangential component keeps it creeping downhill.  That is a valid
        # outcome of a friction sweep, so the generic flat-rest heuristic is
        # advisory for incline motion families rather than a rejection.
        if motion in {
            "slope_slide_down_1obj",
            "slope_slide_up_1obj",
            "ramp_to_flat_1obj",
        }:
            advisory_ids.add("bounded_rest_state_drift")
        # The restitution ratio is a useful diagnostic for a sphere's normal
        # impact, but it is not a material identity test for oblique box/case
        # contacts: contact point, orientation, and spin can redirect energy.
        # Keep penetration and energy invariants hard while making this one
        # check advisory for non-spherical sweep impacts.
        if (
            str(obj["geometry"]["type"]) != "sphere"
            and motion in {"bounce_1obj", "wall_impact_1obj"}
        ):
            advisory_ids.add("bounce_response_matches_restitution")
    advisory_checks = []
    for record in checks:
        if record["id"] in advisory_ids and not record["passed"]:
            record["severity"] = "advisory"
            advisory_checks.append(record)
    passed = all(
        record["passed"] or record.get("severity") == "advisory"
        for record in checks
    )
    return {
        "schema_version": "physweep_rigid_trajectory_audit_v1",
        "scene_id": metadata["scene_id"],
        "passed": passed,
        "checks": checks,
        "advisories": advisory_checks,
        "metrics": {
            "frame_count": int(positions.shape[0]),
            "horizontal_displacement_m": round(displacement_xy, 6),
            "horizontal_path_length_m": round(path_length_xy, 6),
            "path_length_3d_m": round(path_length_3d, 6),
            "vertical_range_m": round(float(np.ptp(positions[:, 2])), 6),
            "maximum_linear_speed_m_s": round(float(speed.max()), 6),
            "maximum_angular_speed_rad_s": round(float(angular_speed.max()), 6),
            "maximum_rotational_surface_speed_m_s": round(
                float(rotational_surface_speed.max()), 6
            ),
            "active_motion_duration_s": round(active_duration_s, 6),
            "primary_support_contact_fraction": round(support_fraction, 6),
            "first_primary_support_contact_frame": first_primary_contact,
            "maximum_penetration_m": round(max_penetration, 6),
            **additional["metrics"],
        },
    }


def compact_failure_ids(audit: dict[str, Any]) -> list[str]:
    return [
        str(record["id"])
        for record in audit["checks"]
        if not record["passed"] and record.get("severity") != "advisory"
    ]


def compact_advisory_ids(audit: dict[str, Any]) -> list[str]:
    return [
        str(record["id"])
        for record in audit["checks"]
        if record.get("severity") == "advisory"
    ]


def validate_trajectory_contract(metadata: dict[str, Any], trajectory: dict[str, np.ndarray]) -> None:
    object_id = str(metadata["simulation"]["objects"][0]["object_id"])
    required = {
        "time_s",
        f"{object_id}__position_m",
        f"{object_id}__quaternion_wxyz",
        f"{object_id}__linear_velocity_m_s",
        f"{object_id}__angular_velocity_rad_s",
        f"{object_id}__aabb_min_m",
        f"{object_id}__aabb_max_m",
        f"{object_id}__primary_support_contact_count",
        f"{object_id}__all_contact_count",
        f"{object_id}__minimum_contact_distance_m",
        f"{object_id}__total_normal_force_n",
        f"{object_id}__maximum_coulomb_friction_utilization",
    }
    required_collider_id = metadata["simulation"]["objects"][0][
        "expected_motion"
    ].get("required_collider_contact_id")
    if required_collider_id:
        required.add(
            f"{object_id}__collider_contact_count__{required_collider_id}"
        )
    missing = required.difference(trajectory)
    if missing:
        raise ValueError(f"trajectory is missing keys: {sorted(missing)}")
    runtime_required = {
        f"{object_id}__runtime_dynamics",
        f"{object_id}__runtime_inertia_diagonal_kg_m2",
        f"{object_id}__runtime_support_dynamics",
        f"{object_id}__runtime_proxy_shape_codes",
        f"{object_id}__runtime_proxy_dimensions_m",
        f"{object_id}__runtime_proxy_positions_m",
        f"{object_id}__runtime_proxy_quaternions_xyzw",
    }
    runtime_missing = runtime_required.difference(trajectory)
    if runtime_missing:
        raise ValueError(
            f"trajectory is missing runtime descriptors: {sorted(runtime_missing)}"
        )
    frame_count = int(metadata["simulation"]["time"]["frame_count"])
    for key in required:
        if int(np.asarray(trajectory[key]).shape[0]) != frame_count:
            raise ValueError(f"trajectory frame count mismatch for {key}")
