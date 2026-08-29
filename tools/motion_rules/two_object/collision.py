"""Semantic audit for a deterministic collision between two rigid objects."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from tools.core.rigid_geometry import (
    PROXY_SHAPE_CODE,
    finite_vector,
    positive_vector,
)


_RULE_FIELDS = {
    "schema_version",
    "reference_motion",
    "initial_surface_gap_m",
    "initial_velocity_x_m_s",
    "contact_friction",
    "contact_restitution",
    "minimum_displacement_m",
    "minimum_support_contact_fraction",
    "interaction_audit",
}
_AUDIT_FIELDS = {
    "minimum_initial_clearance_m",
    "minimum_approach_axis_alignment",
    "minimum_pre_contact_closing_speed_m_s",
    "maximum_first_contact_time_s",
    "minimum_post_contact_separation_m",
    "maximum_collision_window_momentum_change_fraction",
    "maximum_camera_side_deviation_degrees",
    "minimum_collision_projected_separation_to_span_ratio",
}


def _first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return int(indices[0]) if indices.size else None


def _path_length(positions: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def _surface_gap(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    size_a: np.ndarray,
    size_b: np.ndarray,
) -> np.ndarray:
    radius_a = 0.5 * float(np.max(size_a))
    radius_b = 0.5 * float(np.max(size_b))
    return np.linalg.norm(positions_b - positions_a, axis=1) - radius_a - radius_b


def apply_two_sphere_collision(
    pair_scene: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Place and configure an ordered pair of independently selected spheres."""

    if set(config) != _RULE_FIELDS or (
        config.get("schema_version") != "physweep_two_sphere_collision_v1"
    ):
        raise ValueError("unsupported two-sphere collision rule")
    interaction_audit = config.get("interaction_audit")
    if (
        not isinstance(interaction_audit, dict)
        or set(interaction_audit) != _AUDIT_FIELDS
    ):
        raise ValueError("two-sphere collision audit fields are incomplete")
    scene = copy.deepcopy(pair_scene)
    objects = scene.get("simulation", {}).get("objects", [])
    if len(objects) != 2 or any(not isinstance(obj, dict) for obj in objects):
        raise ValueError("two-sphere collision requires one ordered object pair")
    object_ids = [str(obj.get("object_id", "")) for obj in objects]
    if any(not object_id for object_id in object_ids) or len(set(object_ids)) != 2:
        raise ValueError("two-sphere collision object ids must be unique")

    radii = []
    for obj in objects:
        geometry = obj.get("geometry", {})
        if geometry.get("type") != "sphere":
            raise ValueError("two-sphere collision requires sphere candidates")
        dimensions = positive_vector(
            geometry["size_m"], 3, f"{obj['object_id']} sphere dimensions"
        )
        if max(dimensions) - min(dimensions) > 1.0e-8:
            raise ValueError("sphere candidate dimensions must be isotropic")
        radii.append(0.5 * dimensions[0])

    support = scene["simulation"]["support"]
    slope = float(support["surface_frame"]["slope_angle_degrees"])
    if abs(slope) > 1.0e-8:
        raise ValueError("two-sphere collision requires a flat support")
    bounds = support["safe_surface_bounds"]
    placement = scene.get("environment_binding", {}).get("placement", {})
    if placement.get("action_anchor_rule") == "initial_object_xy":
        scene_anchor = finite_vector(
            placement["scene_anchor_m"][:2], 2, "environment scene anchor"
        )
        center_x, center_y = scene_anchor
    else:
        center_x = 0.5 * (float(bounds["x"][0]) + float(bounds["x"][1]))
        center_y = 0.5 * (float(bounds["y"][0]) + float(bounds["y"][1]))

    surface_gap = finite_vector(
        [config["initial_surface_gap_m"]], 1, "initial sphere surface gap"
    )[0]
    if surface_gap <= 0.0:
        raise ValueError("initial sphere surface gap must be positive")
    center_distance = radii[0] + radii[1] + surface_gap
    positions_x = [center_x - 0.5 * center_distance, center_x + 0.5 * center_distance]
    margin = 0.02
    if (
        positions_x[0] < float(bounds["x"][0]) + radii[0] + margin
        or positions_x[1] > float(bounds["x"][1]) - radii[1] - margin
        or center_y < float(bounds["y"][0]) + max(radii) + margin
        or center_y > float(bounds["y"][1]) - max(radii) - margin
    ):
        raise ValueError("host support is too small for the selected object pair")
    positions_z = [
        float(support["surface_center_z_m"]) + radius + 0.0005
        for radius in radii
    ]
    velocities_x = finite_vector(
        config["initial_velocity_x_m_s"], 2, "two-object velocities"
    )
    if velocities_x[0] <= velocities_x[1]:
        raise ValueError("two-object velocities must form a closing pair")

    contact_friction, contact_restitution, minimum_displacement = finite_vector(
        [
            config["contact_friction"],
            config["contact_restitution"],
            config["minimum_displacement_m"],
        ],
        3,
        "two-object physical rule values",
    )
    minimum_support_fraction = finite_vector(
        [config["minimum_support_contact_fraction"]],
        1,
        "minimum support contact fraction",
    )[0]
    if contact_friction < 0.0:
        raise ValueError("contact friction must be nonnegative")
    if not 0.0 <= contact_restitution <= 1.0:
        raise ValueError("contact restitution must lie in [0, 1]")
    if minimum_displacement <= 0.0:
        raise ValueError("minimum displacement must be positive")
    if not 0.0 < minimum_support_fraction <= 1.0:
        raise ValueError("minimum support contact fraction must lie in (0, 1]")

    motion = str(config["reference_motion"])
    scene["scene_id"] = f"{scene['scene_id']}__two_sphere_collision"
    dimensions = scene["semantic_sampling"]["five_dimensions"]
    dimensions["motion"] = {
        "family": motion,
        "subtype": "direct_pair_collision",
        "direction": "positive_x",
        "direction_angle_degrees": 0.0,
        "trajectory_extent": "medium",
        "initial_position_zone": "opposed_pair",
    }
    expected_common = {
        "motion_family": motion,
        "contact_mode": "supported_pair_collision",
        "must_contact_primary_support": True,
        "minimum_displacement_m": minimum_displacement,
        "minimum_support_contact_fraction": minimum_support_fraction,
    }
    for index, obj in enumerate(objects):
        material = copy.deepcopy(obj["material"])
        material["contact_friction"] = contact_friction
        material["contact_restitution"] = contact_restitution
        obj["material"] = material
        obj["initial_state"] = {
            "pose_profile": "support_normal",
            "position_m": [positions_x[index], center_y, positions_z[index]],
            "orientation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "linear_velocity_m_s": [velocities_x[index], 0.0, 0.0],
            "angular_velocity_rad_s": [
                0.0,
                velocities_x[index] / radii[index],
                0.0,
            ],
        }
        obj["expected_motion"] = {
            **expected_common,
            "required_object_contact_id": object_ids[1 - index],
        }
    scene["simulation"]["interaction"] = {
        "type": "pairwise_collision",
        "object_ids": object_ids,
        "approach_axis_xy": [1.0, 0.0],
        **copy.deepcopy(interaction_audit),
    }
    return scene


def audit_pair_collision(
    metadata: dict[str, Any], trajectory: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Audit one declared pair-collision event without invoking 1obj heuristics."""

    objects = metadata.get("simulation", {}).get("objects")
    if (
        not isinstance(objects, list)
        or len(objects) != 2
        or any(not isinstance(obj, dict) for obj in objects)
    ):
        raise ValueError("pair-collision audit requires two simulation objects")
    object_ids = [str(obj["object_id"]) for obj in objects]
    interaction = metadata["simulation"].get("interaction")
    if not isinstance(interaction, dict):
        raise ValueError("two-object metadata requires simulation.interaction")
    if interaction.get("type") != "pairwise_collision":
        raise ValueError("unsupported two-object interaction type")
    if list(interaction.get("object_ids", [])) != object_ids:
        raise ValueError("interaction object order differs from simulation.objects")
    for index, obj in enumerate(objects):
        other_object_id = object_ids[1 - index]
        if (
            obj.get("expected_motion", {}).get("required_object_contact_id")
            != other_object_id
        ):
            raise ValueError(
                f"{object_ids[index]} does not require contact with {other_object_id}"
            )
        geometry = obj.get("geometry", {})
        dimensions = np.asarray(geometry.get("size_m", []), dtype=np.float64)
        if (
            geometry.get("type") != "sphere"
            or dimensions.shape != (3,)
            or not np.isfinite(dimensions).all()
            or np.any(dimensions <= 0.0)
            or float(np.ptp(dimensions)) > 1.0e-8
        ):
            raise ValueError(
                "pair-collision audit currently requires isotropic sphere geometry"
            )

    time_s = np.asarray(trajectory["time_s"], dtype=np.float64)
    positions = {
        object_id: np.asarray(
            trajectory[f"{object_id}__position_m"], dtype=np.float64
        )
        for object_id in object_ids
    }
    velocities = {
        object_id: np.asarray(
            trajectory[f"{object_id}__linear_velocity_m_s"], dtype=np.float64
        )
        for object_id in object_ids
    }
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

    limits = metadata.get("qa", {}).get("limits", {})
    parameter_tolerance = float(
        limits.get("parameter_match_absolute_tolerance", 1.0e-6)
    )
    maximum_initial_penetration = float(
        limits.get("maximum_initial_penetration_m", 0.0005)
    )
    maximum_penetration = float(
        limits.get("maximum_trajectory_penetration_m", 0.008)
    )
    maximum_linear_speed = float(limits.get("maximum_linear_speed_m_s", 5.2))
    maximum_angular_speed = float(
        limits.get("maximum_angular_speed_rad_s", 120.0)
    )
    support_dynamics = metadata["simulation"]["support"]["dynamics"]
    expected_support = np.asarray(
        [
            support_dynamics["lateral_friction"],
            support_dynamics["restitution"],
        ],
        dtype=np.float64,
    )

    for obj in objects:
        object_id = str(obj["object_id"])
        position = positions[object_id]
        velocity = velocities[object_id]
        angular = np.asarray(
            trajectory[f"{object_id}__angular_velocity_rad_s"], dtype=np.float64
        )
        quaternion = np.asarray(
            trajectory[f"{object_id}__quaternion_wxyz"], dtype=np.float64
        )
        contact_distance = np.asarray(
            trajectory[f"{object_id}__minimum_contact_distance_m"],
            dtype=np.float64,
        )
        primary_contacts = np.asarray(
            trajectory[f"{object_id}__primary_support_contact_count"],
            dtype=np.int32,
        )
        expected = obj["expected_motion"]
        finite = bool(
            np.isfinite(position).all()
            and np.isfinite(velocity).all()
            and np.isfinite(angular).all()
            and np.isfinite(quaternion).all()
        )
        check(f"{object_id}__finite_state", finite, finite, True)
        check(
            f"{object_id}__initial_position_matches_metadata",
            bool(
                np.allclose(
                    position[0],
                    obj["initial_state"]["position_m"],
                    atol=parameter_tolerance,
                    rtol=0.0,
                )
            ),
            float(
                np.max(
                    np.abs(
                        position[0]
                        - np.asarray(
                            obj["initial_state"]["position_m"], dtype=np.float64
                        )
                    )
                )
            ),
            parameter_tolerance,
        )
        check(
            f"{object_id}__initial_linear_velocity_matches_metadata",
            bool(
                np.allclose(
                    velocity[0],
                    obj["initial_state"]["linear_velocity_m_s"],
                    atol=parameter_tolerance,
                    rtol=0.0,
                )
            ),
            float(
                np.max(
                    np.abs(
                        velocity[0]
                        - np.asarray(
                            obj["initial_state"]["linear_velocity_m_s"],
                            dtype=np.float64,
                        )
                    )
                )
            ),
            parameter_tolerance,
        )
        runtime_dynamics = np.asarray(
            trajectory[f"{object_id}__runtime_dynamics"], dtype=np.float64
        )
        expected_dynamics = np.asarray(
            [
                obj["material"]["mass_kg"],
                obj["material"]["contact_friction"],
                obj["material"]["contact_restitution"],
                obj["material"]["rolling_friction"],
                obj["material"]["spinning_friction"],
            ],
            dtype=np.float64,
        )
        dynamics_error = float(np.max(np.abs(runtime_dynamics - expected_dynamics)))
        check(
            f"{object_id}__pybullet_dynamics_match_metadata",
            dynamics_error <= parameter_tolerance,
            dynamics_error,
            parameter_tolerance,
        )
        runtime_support = np.asarray(
            trajectory[f"{object_id}__runtime_support_dynamics"], dtype=np.float64
        )
        support_error = float(
            np.max(np.abs(runtime_support - expected_support[None, :]))
        )
        check(
            f"{object_id}__pybullet_support_dynamics_match_metadata",
            support_error <= parameter_tolerance,
            support_error,
            parameter_tolerance,
        )
        inertia = np.asarray(
            trajectory[f"{object_id}__runtime_inertia_diagonal_kg_m2"],
            dtype=np.float64,
        )
        check(
            f"{object_id}__runtime_inertia_is_finite_and_positive",
            bool(np.isfinite(inertia).all() and np.all(inertia > 0.0)),
            inertia.tolist(),
            "finite and positive",
        )
        geometry_type = str(obj["geometry"]["type"])
        proxy_codes = np.asarray(
            trajectory[f"{object_id}__runtime_proxy_shape_codes"], dtype=np.int32
        )
        proxy_dimensions = np.asarray(
            trajectory[f"{object_id}__runtime_proxy_dimensions_m"],
            dtype=np.float64,
        )
        check(
            f"{object_id}__collision_proxy_matches_definition",
            bool(
                proxy_codes.shape == (1,)
                and int(proxy_codes[0]) == PROXY_SHAPE_CODE[geometry_type]
                and proxy_dimensions.shape == (1, 3)
                and np.allclose(
                    proxy_dimensions[0],
                    obj["geometry"]["size_m"],
                    atol=parameter_tolerance,
                    rtol=0.0,
                )
            ),
            {
                "shape_codes": proxy_codes.tolist(),
                "dimensions_m": proxy_dimensions.tolist(),
            },
            {
                "shape_code": PROXY_SHAPE_CODE[geometry_type],
                "dimensions_m": obj["geometry"]["size_m"],
            },
        )
        initial_penetration = max(0.0, -float(contact_distance[0]))
        trajectory_penetration = max(0.0, -float(np.min(contact_distance)))
        check(
            f"{object_id}__bounded_initial_penetration",
            initial_penetration <= maximum_initial_penetration,
            initial_penetration,
            maximum_initial_penetration,
        )
        check(
            f"{object_id}__bounded_penetration",
            trajectory_penetration <= maximum_penetration,
            trajectory_penetration,
            maximum_penetration,
        )
        speed = np.linalg.norm(velocity, axis=1)
        angular_speed = np.linalg.norm(angular, axis=1)
        check(
            f"{object_id}__bounded_linear_speed",
            float(speed.max()) <= maximum_linear_speed,
            float(speed.max()),
            maximum_linear_speed,
        )
        check(
            f"{object_id}__bounded_angular_speed",
            float(angular_speed.max()) <= maximum_angular_speed,
            float(angular_speed.max()),
            maximum_angular_speed,
        )
        minimum_displacement = float(expected.get("minimum_displacement_m", 0.0))
        path_length = _path_length(position)
        check(
            f"{object_id}__visible_motion",
            path_length >= minimum_displacement,
            path_length,
            minimum_displacement,
        )
        if expected.get("must_contact_primary_support", False):
            check(
                f"{object_id}__primary_support_contact",
                bool(np.any(primary_contacts > 0)),
                _first_true(primary_contacts > 0),
                "at least one frame",
            )
        minimum_support_fraction = float(
            expected.get("minimum_support_contact_fraction", 0.0)
        )
        if minimum_support_fraction > 0.0:
            support_fraction = float(np.mean(primary_contacts > 0))
            check(
                f"{object_id}__primary_support_contact_fraction",
                support_fraction >= minimum_support_fraction,
                support_fraction,
                minimum_support_fraction,
            )

    object_a, object_b = objects
    object_id_a, object_id_b = object_ids
    contact_a = np.asarray(
        trajectory[f"{object_id_a}__object_contact_count__{object_id_b}"],
        dtype=np.int32,
    )
    contact_b = np.asarray(
        trajectory[f"{object_id_b}__object_contact_count__{object_id_a}"],
        dtype=np.int32,
    )
    contact_mask = (contact_a > 0) | (contact_b > 0)
    first_contact = _first_true(contact_mask)
    check(
        "pair_contact_channels_are_reciprocal",
        bool(np.array_equal(contact_a, contact_b)),
        int(np.count_nonzero(contact_a != contact_b)),
        0,
    )
    check(
        "required_pair_collision",
        first_contact is not None,
        first_contact,
        "at least one frame",
    )
    size_a = np.asarray(object_a["geometry"]["size_m"], dtype=np.float64)
    size_b = np.asarray(object_b["geometry"]["size_m"], dtype=np.float64)
    gap = _surface_gap(
        positions[object_id_a], positions[object_id_b], size_a, size_b
    )
    minimum_initial_clearance = float(
        interaction.get("minimum_initial_clearance_m", 0.02)
    )
    check(
        "pair_initial_clearance",
        float(gap[0]) >= minimum_initial_clearance,
        float(gap[0]),
        minimum_initial_clearance,
    )
    approach_axis = np.asarray(interaction["approach_axis_xy"], dtype=np.float64)
    if (
        approach_axis.shape != (2,)
        or not np.isfinite(approach_axis).all()
        or float(np.linalg.norm(approach_axis)) <= 1.0e-8
    ):
        raise ValueError("pair interaction requires a finite nonzero approach axis")
    approach_axis /= float(np.linalg.norm(approach_axis))
    initial_axis = positions[object_id_b][0, :2] - positions[object_id_a][0, :2]
    initial_axis /= max(float(np.linalg.norm(initial_axis)), 1.0e-12)
    approach_alignment = float(initial_axis @ approach_axis)
    minimum_approach_alignment = float(
        interaction["minimum_approach_axis_alignment"]
    )
    check(
        "pair_approach_axis_alignment",
        approach_alignment >= minimum_approach_alignment,
        approach_alignment,
        minimum_approach_alignment,
    )
    if first_contact is not None:
        maximum_first_contact_time = float(
            interaction.get("maximum_first_contact_time_s", time_s[-1])
        )
        check(
            "pair_collision_time",
            float(time_s[first_contact]) <= maximum_first_contact_time,
            float(time_s[first_contact]),
            maximum_first_contact_time,
        )
        pre_index = max(0, first_contact - 1)
        separation = (
            positions[object_id_b][pre_index] - positions[object_id_a][pre_index]
        )
        separation /= max(float(np.linalg.norm(separation)), 1.0e-12)
        relative_velocity = (
            velocities[object_id_b][pre_index] - velocities[object_id_a][pre_index]
        )
        closing_speed = -float(relative_velocity @ separation)
        minimum_closing_speed = float(
            interaction.get("minimum_pre_contact_closing_speed_m_s", 0.05)
        )
        check(
            "pair_pre_contact_approach",
            closing_speed >= minimum_closing_speed,
            closing_speed,
            minimum_closing_speed,
        )
        post_indices = np.flatnonzero(
            np.arange(time_s.size) > first_contact
        )
        post_gap = float(np.max(gap[post_indices])) if post_indices.size else 0.0
        minimum_post_separation = float(
            interaction.get("minimum_post_contact_separation_m", 0.02)
        )
        check(
            "pair_post_contact_separation",
            post_gap >= minimum_post_separation,
            post_gap,
            minimum_post_separation,
        )

    masses = np.asarray(
        [float(obj["material"]["mass_kg"]) for obj in objects], dtype=np.float64
    )
    momentum_reference_index = max(0, (first_contact or 0) - 1)
    momentum_result_index = min(
        time_s.size - 1, (first_contact or 0) + 2
    )
    reference_momentum = (
        masses[0] * velocities[object_id_a][momentum_reference_index, :2]
        + masses[1] * velocities[object_id_b][momentum_reference_index, :2]
    )
    combined_momentum = (
        masses[0] * velocities[object_id_a][momentum_result_index, :2]
        + masses[1] * velocities[object_id_b][momentum_result_index, :2]
    )
    momentum_scale = max(float(np.linalg.norm(reference_momentum)), 1.0e-9)
    momentum_change_fraction = float(
        np.linalg.norm(combined_momentum - reference_momentum) / momentum_scale
    )
    maximum_momentum_change = float(
        interaction.get("maximum_collision_window_momentum_change_fraction", 0.25)
    )
    check(
        "bounded_collision_window_pair_momentum_change",
        momentum_change_fraction <= maximum_momentum_change,
        momentum_change_fraction,
        maximum_momentum_change,
    )

    advisory_ids: set[str] = set()
    sweep = metadata.get("sweep")
    if isinstance(sweep, dict) and sweep.get("kind") == "sweep":
        parameter = sweep.get("parameter")
        if parameter == "contact_restitution":
            advisory_ids.add("pair_post_contact_separation")
        if parameter == "contact_friction":
            advisory_ids.add("bounded_collision_window_pair_momentum_change")
    advisories = []
    for record in checks:
        if record["id"] in advisory_ids and not record["passed"]:
            record["severity"] = "advisory"
            advisories.append(record)
    passed = all(
        record["passed"] or record.get("severity") == "advisory"
        for record in checks
    )
    return {
        "schema_version": "physweep_rigid_trajectory_audit_v1",
        "scene_id": metadata["scene_id"],
        "passed": passed,
        "checks": checks,
        "advisories": advisories,
        "metrics": {
            "frame_count": int(time_s.size),
            "object_count": 2,
            "first_pair_contact_frame": first_contact,
            "minimum_pair_surface_gap_m": float(np.min(gap)),
            "collision_window_pair_momentum_change_fraction": (
                momentum_change_fraction
            ),
        },
    }
