"""Semantic audit for interacting and independent two-object motion."""

from __future__ import annotations

from typing import Any

import numpy as np

from tools.core.camera_geometry import pair_approach_axis_xy
from tools.core.rigid_geometry import (
    declared_collision_descriptors,
    primitive_support_radius_m,
    quaternion_matrix_wxyz,
)


def _first_true(values: np.ndarray) -> int | None:
    indices = np.flatnonzero(values)
    return int(indices[0]) if indices.size else None


def _path_length(positions: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def _center_axis_separation_margin(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    quaternions_a: np.ndarray,
    quaternions_b: np.ndarray,
    object_a: dict[str, Any],
    object_b: dict[str, Any],
) -> np.ndarray:
    """Return a conservative separation margin along the pair center axis."""

    delta = positions_b - positions_a
    distance = np.linalg.norm(delta, axis=1)
    directions = np.zeros_like(delta)
    valid = distance > 1.0e-12
    directions[valid] = delta[valid] / distance[valid, None]
    directions[~valid, 0] = 1.0

    def radius(
        obj: dict[str, Any],
        quaternions: np.ndarray,
        world_directions: np.ndarray,
    ) -> np.ndarray:
        geometry = obj["geometry"]
        return np.asarray(
            [
                primitive_support_radius_m(
                    str(geometry["type"]),
                    list(geometry["size_m"]),
                    (
                        np.asarray(quaternion_matrix_wxyz(quaternion)).T
                        @ direction
                    ).tolist(),
                )
                for quaternion, direction in zip(quaternions, world_directions)
            ],
            dtype=np.float64,
        )

    return distance - radius(object_a, quaternions_a, directions) - radius(
        object_b, quaternions_b, -directions
    )


def audit_pair_motion(
    metadata: dict[str, Any], trajectory: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Audit the declared pair-contact requirement and per-object invariants."""

    objects = metadata.get("simulation", {}).get("objects")
    if (
        not isinstance(objects, list)
        or len(objects) != 2
        or any(not isinstance(obj, dict) for obj in objects)
    ):
        raise ValueError("pair-motion audit requires two simulation objects")
    object_ids = [str(obj["object_id"]) for obj in objects]
    interaction = metadata["simulation"].get("interaction")
    if not isinstance(interaction, dict):
        raise ValueError("two-object metadata requires simulation.interaction")
    interaction_type = str(interaction.get("type", ""))
    if interaction_type not in {"pairwise_collision", "pairwise_independent"}:
        raise ValueError("unsupported two-object interaction type")
    interacting = interaction_type == "pairwise_collision"
    expected_class = "interacting" if interacting else "independent"
    if interaction.get("interaction_class") != expected_class:
        raise ValueError("pair interaction type and class disagree")
    expected_requirement = "must_contact" if interacting else "must_not_contact"
    if interaction.get("contact_requirement") != expected_requirement:
        raise ValueError("pair interaction has a contradictory contact requirement")
    if list(interaction.get("object_ids", [])) != object_ids:
        raise ValueError("interaction object order differs from simulation.objects")
    motion = (
        metadata.get("semantic_sampling", {})
        .get("five_dimensions", {})
        .get("motion")
    )
    if not isinstance(motion, dict):
        raise ValueError("two-object metadata requires semantic motion")
    if motion.get("family") != interaction.get("motion_pattern"):
        raise ValueError("semantic motion and interaction pattern disagree")
    if motion.get("interaction_class") != expected_class:
        raise ValueError("semantic motion and interaction class disagree")
    if motion.get("kinematic_regime") != interaction.get("kinematic_regime"):
        raise ValueError("semantic motion and kinematic regime disagree")
    expected_object_motions = {
        object_id: objects[index].get("expected_motion", {}).get("motion_family")
        for index, object_id in enumerate(object_ids)
    }
    if motion.get("object_motions") != expected_object_motions:
        raise ValueError("semantic and per-object motion families disagree")
    for index, obj in enumerate(objects):
        other_object_id = object_ids[1 - index]
        relation_key = (
            "required_object_contact_id"
            if interacting
            else "forbidden_object_contact_id"
        )
        if obj.get("expected_motion", {}).get(relation_key) != other_object_id:
            raise ValueError(
                f"{object_ids[index]} has no valid relation to {other_object_id}"
            )
        geometry = obj.get("geometry", {})
        dimensions = np.asarray(geometry.get("size_m", []), dtype=np.float64)
        geometry_type = str(geometry.get("type", ""))
        if (
            geometry_type not in {"sphere", "cuboid", "cylinder"}
            or dimensions.shape != (3,)
            or not np.isfinite(dimensions).all()
            or np.any(dimensions <= 0.0)
            or (
                geometry_type == "sphere"
                and float(np.ptp(dimensions)) > 1.0e-8
            )
            or (
                geometry_type == "cylinder"
                and abs(float(dimensions[0] - dimensions[1])) > 1.0e-8
            )
        ):
            raise ValueError("pair-motion audit received invalid primitive geometry")

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
    maximum_independent_rest_path_length = float(
        interaction["maximum_independent_rest_path_length_m"]
    )
    minimum_pre_contact_arc_ascent = float(
        interaction["minimum_pre_contact_arc_vertical_ascent_m"]
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
        declared_proxy = declared_collision_descriptors(obj)
        expected_proxy = {
            key: np.asarray(
                value, dtype=np.int32 if key == "shape_codes" else np.float64
            )
            for key, value in declared_proxy.items()
        }
        proxy_codes = np.asarray(
            trajectory[f"{object_id}__runtime_proxy_shape_codes"], dtype=np.int32
        )
        proxy_dimensions = np.asarray(
            trajectory[f"{object_id}__runtime_proxy_dimensions_m"],
            dtype=np.float64,
        )
        proxy_positions = np.asarray(
            trajectory[f"{object_id}__runtime_proxy_positions_m"],
            dtype=np.float64,
        )
        proxy_quaternions = np.asarray(
            trajectory[f"{object_id}__runtime_proxy_quaternions_xyzw"],
            dtype=np.float64,
        )
        check(
            f"{object_id}__collision_proxy_matches_definition",
            bool(
                np.array_equal(proxy_codes, expected_proxy["shape_codes"])
                and proxy_dimensions.shape
                == expected_proxy["dimensions_m"].shape
                and proxy_positions.shape == expected_proxy["positions_m"].shape
                and proxy_quaternions.shape
                == expected_proxy["quaternions_xyzw"].shape
                and np.allclose(
                    proxy_dimensions,
                    expected_proxy["dimensions_m"],
                    atol=parameter_tolerance,
                    rtol=0.0,
                )
                and np.allclose(
                    proxy_positions,
                    expected_proxy["positions_m"],
                    atol=parameter_tolerance,
                    rtol=0.0,
                )
                and all(
                    min(
                        float(np.linalg.norm(runtime - expected)),
                        float(np.linalg.norm(runtime + expected)),
                    )
                    <= parameter_tolerance
                    for runtime, expected in zip(
                        proxy_quaternions,
                        expected_proxy["quaternions_xyzw"],
                        strict=True,
                    )
                )
            ),
            {
                "shape_codes": proxy_codes.tolist(),
                "dimensions_m": proxy_dimensions.tolist(),
                "positions_m": proxy_positions.tolist(),
                "quaternions_xyzw": proxy_quaternions.tolist(),
            },
            {
                "shape_codes": expected_proxy["shape_codes"].tolist(),
                "dimensions_m": expected_proxy["dimensions_m"].tolist(),
                "positions_m": expected_proxy["positions_m"].tolist(),
                "quaternions_xyzw": expected_proxy["quaternions_xyzw"].tolist(),
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
        minimum_displacement = float(expected["minimum_displacement_m"])
        path_length = _path_length(position)
        check(
            f"{object_id}__visible_motion",
            path_length >= minimum_displacement,
            path_length,
            minimum_displacement,
        )
        motion_family = str(expected["motion_family"])
        if not interacting and motion_family == "rest":
            check(
                f"{object_id}__bounded_independent_rest_path_length",
                path_length <= maximum_independent_rest_path_length,
                path_length,
                maximum_independent_rest_path_length,
            )
        if motion_family == "arc_projectile_1obj":
            pre_contact_stop = (
                first_contact + 1 if first_contact is not None else position.shape[0]
            )
            pre_contact_position = position[:pre_contact_stop]
            vertical_ascent = float(
                np.max(pre_contact_position[:, 2]) - pre_contact_position[0, 2]
            )
            check(
                f"{object_id}__pre_contact_arc_vertical_ascent",
                vertical_ascent >= minimum_pre_contact_arc_ascent,
                vertical_ascent,
                minimum_pre_contact_arc_ascent,
            )
        if bool(expected["must_contact_primary_support"]):
            check(
                f"{object_id}__primary_support_contact",
                bool(np.any(primary_contacts > 0)),
                _first_true(primary_contacts > 0),
                "at least one frame",
            )
        minimum_support_fraction = float(
            expected["minimum_support_contact_fraction"]
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
    check(
        "pair_contact_channels_are_reciprocal",
        bool(np.array_equal(contact_a, contact_b)),
        int(np.count_nonzero(contact_a != contact_b)),
        0,
    )
    check_id = (
        "required_pair_collision" if interacting else "forbidden_pair_collision"
    )
    check(
        check_id,
        first_contact is not None if interacting else first_contact is None,
        first_contact,
        "at least one frame" if interacting else "no frames",
    )
    separation_margin = _center_axis_separation_margin(
        positions[object_id_a],
        positions[object_id_b],
        np.asarray(
            trajectory[f"{object_id_a}__quaternion_wxyz"], dtype=np.float64
        ),
        np.asarray(
            trajectory[f"{object_id_b}__quaternion_wxyz"], dtype=np.float64
        ),
        object_a,
        object_b,
    )
    minimum_initial_clearance = float(
        interaction["minimum_initial_clearance_m"]
    )
    check(
        "pair_initial_clearance",
        float(separation_margin[0]) >= minimum_initial_clearance,
        float(separation_margin[0]),
        minimum_initial_clearance,
    )
    pair_approach_axis_xy(interaction["approach_axis_xyz"])
    if interacting and first_contact is not None:
        maximum_first_contact_time = float(
            interaction["maximum_first_contact_time_s"]
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
            interaction["minimum_pre_contact_closing_speed_m_s"]
        )
        check(
            "pair_pre_contact_approach",
            closing_speed >= minimum_closing_speed,
            closing_speed,
            minimum_closing_speed,
        )
    advisories: list[dict[str, Any]] = []
    passed = all(record["passed"] for record in checks)
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
            "minimum_pair_center_axis_separation_margin_m": float(
                np.min(separation_margin)
            ),
            "interaction_class": interaction["interaction_class"],
            "motion_pattern": interaction["motion_pattern"],
        },
    }
