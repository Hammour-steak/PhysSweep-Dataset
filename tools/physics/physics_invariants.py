#!/usr/bin/env python3
"""Shared numerical invariants for deterministic rigid-body trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np

from tools.core.rigid_geometry import PROXY_SHAPE_CODE, quaternion_matrix_wxyz


def runtime_collision_descriptors(
    pybullet: Any, body: int
) -> dict[str, np.ndarray]:
    shape_codes: list[int] = []
    dimensions: list[list[float]] = []
    positions: list[list[float]] = []
    quaternions: list[list[float]] = []
    for record in pybullet.getCollisionShapeData(body, -1):
        geometry_type = int(record[2])
        raw_dimensions = [float(value) for value in record[3]]
        if geometry_type == pybullet.GEOM_BOX:
            shape_codes.append(PROXY_SHAPE_CODE["box"])
            normalized_dimensions = raw_dimensions
        elif geometry_type == pybullet.GEOM_SPHERE:
            shape_codes.append(PROXY_SHAPE_CODE["sphere"])
            normalized_dimensions = [2.0 * raw_dimensions[0]] * 3
        elif geometry_type == pybullet.GEOM_CYLINDER:
            shape_codes.append(PROXY_SHAPE_CODE["cylinder"])
            normalized_dimensions = [
                2.0 * raw_dimensions[1],
                2.0 * raw_dimensions[1],
                raw_dimensions[0],
            ]
        else:
            raise ValueError(f"unsupported runtime collision geometry: {geometry_type}")
        dimensions.append(normalized_dimensions)
        positions.append([float(value) for value in record[5]])
        quaternions.append([float(value) for value in record[6]])
    return {
        "shape_codes": np.asarray(shape_codes, dtype=np.int32),
        "dimensions_m": np.asarray(dimensions, dtype=np.float64),
        "positions_m": np.asarray(positions, dtype=np.float64),
        "quaternions_xyzw": np.asarray(quaternions, dtype=np.float64),
    }


def maximum_coulomb_utilization(
    contacts: list[Any],
    dynamic_friction: float,
    static_friction_by_body: dict[int, float],
) -> float:
    values: list[float] = []
    for record in contacts:
        normal_force = float(record[9])
        if normal_force <= 1.0e-9:
            continue
        static_friction = static_friction_by_body.get(int(record[2]))
        if static_friction is None:
            continue
        friction_force = float(np.hypot(float(record[10]), float(record[12])))
        combined_friction = min(
            10.0, abs(float(dynamic_friction) * float(static_friction))
        )
        capacity = combined_friction * normal_force
        utilization = (
            friction_force / capacity
            if capacity > 1.0e-12
            else (0.0 if friction_force <= 1.0e-12 else float("inf"))
        )
        values.append(utilization)
    return max(values, default=0.0)


def principal_inertia(shape: str, size_m: np.ndarray, mass_kg: float) -> np.ndarray:
    x, y, z = np.asarray(size_m, dtype=np.float64)
    if shape == "sphere":
        value = 0.4 * mass_kg * (x / 2.0) ** 2
        return np.repeat(value, 3)
    if shape == "cylinder":
        radius = max(x, y) / 2.0
        transverse = mass_kg * (3.0 * radius * radius + z * z) / 12.0
        return np.asarray([transverse, transverse, 0.5 * mass_kg * radius * radius])
    return mass_kg * np.asarray([y * y + z * z, x * x + z * z, x * x + y * y]) / 12.0


def quaternion_error_xyzw(left: np.ndarray, right: np.ndarray) -> float:
    left_value = np.asarray(left, dtype=np.float64)
    right_value = np.asarray(right, dtype=np.float64)
    left_value /= max(float(np.linalg.norm(left_value)), 1.0e-12)
    right_value /= max(float(np.linalg.norm(right_value)), 1.0e-12)
    return min(
        float(np.linalg.norm(left_value - right_value)),
        float(np.linalg.norm(left_value + right_value)),
    )


def additional_physics_invariants(
    *,
    motion: str,
    time_s: np.ndarray,
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
    contact_mask: np.ndarray,
    shape: str,
    size_m: np.ndarray,
    mass_kg: float,
    gravity_m_s2: float | np.ndarray,
    initial_state: dict[str, Any],
    material: dict[str, Any],
    quality: dict[str, Any],
    runtime_dynamics: np.ndarray | None = None,
    runtime_inertia_diagonal_kg_m2: np.ndarray | None = None,
    expected_proxy_shape_codes: np.ndarray | None = None,
    expected_proxy_dimensions_m: np.ndarray | None = None,
    expected_proxy_positions_m: np.ndarray | None = None,
    expected_proxy_quaternions_xyzw: np.ndarray | None = None,
    runtime_proxy_shape_codes: np.ndarray | None = None,
    runtime_proxy_dimensions_m: np.ndarray | None = None,
    runtime_proxy_positions_m: np.ndarray | None = None,
    runtime_proxy_quaternions_xyzw: np.ndarray | None = None,
    coulomb_friction_utilization: np.ndarray | None = None,
    validate_primitive_inertia_geometry: bool = False,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, value: Any, threshold: Any) -> None:
        checks.append(
            {"id": check_id, "passed": bool(passed), "value": value, "threshold": threshold}
        )

    parameter_tolerance = float(quality.get("parameter_match_absolute_tolerance", 1.0e-6))
    initial_position = np.asarray(initial_state["position_m"], dtype=np.float64)
    if "orientation_quaternion_wxyz" in initial_state:
        initial_quaternion_value = initial_state["orientation_quaternion_wxyz"]
    else:
        xyzw = initial_state["orientation_quaternion_xyzw"]
        initial_quaternion_value = [xyzw[3], *xyzw[:3]]
    initial_quaternion = np.asarray(initial_quaternion_value, dtype=np.float64)
    initial_linear = np.asarray(initial_state["linear_velocity_m_s"], dtype=np.float64)
    initial_angular = np.asarray(initial_state["angular_velocity_rad_s"], dtype=np.float64)
    quaternion_error = min(
        float(np.linalg.norm(quaternions_wxyz[0] - initial_quaternion)),
        float(np.linalg.norm(quaternions_wxyz[0] + initial_quaternion)),
    )
    check(
        "initial_position_matches_metadata",
        bool(np.allclose(positions[0], initial_position, atol=parameter_tolerance, rtol=0.0)),
        round(float(np.max(np.abs(positions[0] - initial_position))), 9),
        parameter_tolerance,
    )
    check(
        "initial_orientation_matches_metadata",
        quaternion_error <= parameter_tolerance,
        round(quaternion_error, 9),
        parameter_tolerance,
    )
    check(
        "initial_linear_velocity_matches_metadata",
        bool(np.allclose(linear_velocity[0], initial_linear, atol=parameter_tolerance, rtol=0.0)),
        round(float(np.max(np.abs(linear_velocity[0] - initial_linear))), 9),
        parameter_tolerance,
    )
    check(
        "initial_angular_velocity_matches_metadata",
        bool(np.allclose(angular_velocity[0], initial_angular, atol=parameter_tolerance, rtol=0.0)),
        round(float(np.max(np.abs(angular_velocity[0] - initial_angular))), 9),
        parameter_tolerance,
    )

    check("positive_mass", mass_kg > 0.0, mass_kg, "> 0")
    friction = float(material.get("contact_friction", material.get("friction", 0.0)))
    restitution = float(material.get("contact_restitution", material.get("restitution", 0.0)))
    check("friction_in_physical_range", 0.0 <= friction <= 2.0, friction, [0.0, 2.0])
    check("restitution_in_physical_range", 0.0 <= restitution <= 1.0, restitution, [0.0, 1.0])
    if runtime_dynamics is not None:
        expected_runtime = np.asarray(
            [
                mass_kg,
                friction,
                restitution,
                float(material.get("rolling_friction", 0.0)),
                float(material.get("spinning_friction", 0.0)),
            ]
        )
        runtime_error = float(np.max(np.abs(runtime_dynamics - expected_runtime)))
        check(
            "pybullet_dynamics_match_metadata",
            runtime_error <= parameter_tolerance,
            round(runtime_error, 9),
            parameter_tolerance,
        )

    proxy_values = (
        expected_proxy_shape_codes,
        expected_proxy_dimensions_m,
        expected_proxy_positions_m,
        expected_proxy_quaternions_xyzw,
        runtime_proxy_shape_codes,
        runtime_proxy_dimensions_m,
        runtime_proxy_positions_m,
        runtime_proxy_quaternions_xyzw,
    )
    if any(value is not None for value in proxy_values):
        if not all(value is not None for value in proxy_values):
            raise ValueError("proxy descriptor audit requires every expected and runtime field")
        expected_codes = np.asarray(expected_proxy_shape_codes, dtype=np.int32)
        runtime_codes = np.asarray(runtime_proxy_shape_codes, dtype=np.int32)
        expected_dimensions = np.asarray(expected_proxy_dimensions_m, dtype=np.float64)
        runtime_dimensions = np.asarray(runtime_proxy_dimensions_m, dtype=np.float64)
        expected_positions = np.asarray(expected_proxy_positions_m, dtype=np.float64)
        runtime_positions = np.asarray(runtime_proxy_positions_m, dtype=np.float64)
        expected_quaternions = np.asarray(
            expected_proxy_quaternions_xyzw, dtype=np.float64
        )
        runtime_quaternions = np.asarray(
            runtime_proxy_quaternions_xyzw, dtype=np.float64
        )
        expected_count = int(expected_codes.shape[0])
        runtime_count = int(runtime_codes.shape[0])
        check(
            "collision_proxy_count_matches_definition",
            runtime_count == expected_count,
            runtime_count,
            expected_count,
        )
        if runtime_count == expected_count:
            check(
                "collision_proxy_types_match_definition",
                bool(np.array_equal(runtime_codes, expected_codes)),
                runtime_codes.tolist(),
                expected_codes.tolist(),
            )
            dimension_error = float(
                np.max(np.abs(runtime_dimensions - expected_dimensions))
            )
            position_error = float(
                np.max(np.abs(runtime_positions - expected_positions))
            )
            orientation_error = max(
                (
                    quaternion_error_xyzw(runtime_value, expected_value)
                    for runtime_value, expected_value in zip(
                        runtime_quaternions, expected_quaternions
                    )
                ),
                default=0.0,
            )
            check(
                "collision_proxy_dimensions_match_definition",
                dimension_error <= parameter_tolerance,
                round(dimension_error, 9),
                parameter_tolerance,
            )
            check(
                "collision_proxy_positions_match_definition",
                position_error <= parameter_tolerance,
                round(position_error, 9),
                parameter_tolerance,
            )
            check(
                "collision_proxy_orientations_match_definition",
                orientation_error <= parameter_tolerance,
                round(orientation_error, 9),
                parameter_tolerance,
            )
    analytic_inertia = principal_inertia(shape, size_m, mass_kg)
    if runtime_inertia_diagonal_kg_m2 is None:
        inertia = analytic_inertia
    else:
        inertia = np.asarray(runtime_inertia_diagonal_kg_m2, dtype=np.float64)
        check(
            "runtime_inertia_is_finite_and_positive",
            bool(np.isfinite(inertia).all() and np.all(inertia > 0.0)),
            [round(float(value), 10) for value in inertia],
            "three finite positive principal moments",
        )
        if validate_primitive_inertia_geometry:
            inertia_relative_error = float(
                np.max(
                    np.abs(inertia - analytic_inertia)
                    / np.maximum(np.abs(analytic_inertia), 1.0e-12)
                )
            )
            inertia_limit = float(
                quality.get("maximum_primitive_inertia_relative_error", 1.0e-6)
            )
            check(
                "primitive_inertia_matches_geometry",
                inertia_relative_error <= inertia_limit,
                round(inertia_relative_error, 9),
                inertia_limit,
            )
    omega_body = np.asarray(
        [
            np.asarray(quaternion_matrix_wxyz(q)).T @ omega
            for q, omega in zip(quaternions_wxyz, angular_velocity)
        ]
    )
    kinetic = 0.5 * mass_kg * np.sum(linear_velocity * linear_velocity, axis=1)
    rotational = 0.5 * np.sum(omega_body * omega_body * inertia[None, :], axis=1)
    gravity = np.asarray(gravity_m_s2, dtype=np.float64)
    if gravity.ndim == 0:
        gravity = np.asarray([0.0, 0.0, float(gravity)], dtype=np.float64)
    potential_raw = -mass_kg * (positions @ gravity)
    potential = potential_raw - float(np.min(potential_raw))
    energy = kinetic + rotational + potential
    energy_gain = float(np.max(energy) - energy[0])
    characteristic_length = max(float(np.min(size_m)), 1.0e-6)
    energy_scale = max(
        float(energy[0]),
        mass_kg * float(np.linalg.norm(gravity)) * characteristic_length,
        1.0e-9,
    )
    energy_limit = (
        float(quality.get("maximum_unforced_energy_gain_fraction", 0.05))
        * energy_scale
        + float(quality.get("maximum_unforced_energy_gain_j_per_kg", 0.01))
        * mass_kg
    )
    check(
        "bounded_unforced_mechanical_energy_gain",
        energy_gain <= energy_limit,
        round(energy_gain, 7),
        round(energy_limit, 7),
    )

    contact_indices = np.flatnonzero(np.asarray(contact_mask, dtype=bool))
    first_contact = int(contact_indices[0]) if contact_indices.size else len(time_s)
    airborne_count = first_contact
    gravity_motions = {
        "drop_fall_1obj", "projectile_1obj", "arc_projectile_1obj", "bounce_1obj",
        "vertical_drop", "workbench_clear_zone_drop",
    }
    if motion in gravity_motions and airborne_count >= 5:
        fit = np.polyfit(time_s[:airborne_count], linear_velocity[:airborne_count, 2], 1)
        gravity_z = float(gravity[2])
        gravity_error = abs(float(fit[0]) - gravity_z) / max(abs(gravity_z), 1.0e-9)
        gravity_limit = float(quality.get("maximum_gravity_fit_relative_error", 0.12))
        check(
            "airborne_gravity_fit",
            gravity_error <= gravity_limit,
            round(gravity_error, 7),
            gravity_limit,
        )

    if motion in {"projectile_1obj", "arc_projectile_1obj"} and airborne_count >= 6:
        airborne_time = time_s[:airborne_count] - float(time_s[0])
        theoretical = (
            positions[0]
            + airborne_time[:, None] * linear_velocity[0]
            + 0.5 * airborne_time[:, None] ** 2 * gravity[None, :]
        )
        position_error = np.linalg.norm(
            positions[:airborne_count] - theoretical, axis=1
        )
        rmse = float(np.sqrt(np.mean(position_error * position_error)))
        maximum_error = float(np.max(position_error))
        horizontal_velocity_drift = float(
            np.max(
                np.linalg.norm(
                    linear_velocity[:airborne_count, :2]
                    - linear_velocity[0, :2],
                    axis=1,
                )
            )
        )
        rmse_limit = float(
            quality.get("maximum_ballistic_theory_rmse_m", 0.02)
        )
        maximum_error_limit = float(
            quality.get("maximum_ballistic_theory_error_m", 0.04)
        )
        horizontal_drift_limit = float(
            quality.get("maximum_airborne_horizontal_velocity_drift_m_s", 0.08)
        )
        check(
            "ballistic_theory_position_rmse",
            rmse <= rmse_limit,
            round(rmse, 7),
            rmse_limit,
        )
        check(
            "ballistic_theory_maximum_position_error",
            maximum_error <= maximum_error_limit,
            round(maximum_error, 7),
            maximum_error_limit,
        )
        check(
            "airborne_horizontal_velocity_drift",
            horizontal_velocity_drift <= horizontal_drift_limit,
            round(horizontal_velocity_drift, 7),
            horizontal_drift_limit,
        )

    if coulomb_friction_utilization is not None:
        utilization = np.asarray(coulomb_friction_utilization, dtype=np.float64)
        finite_utilization = utilization[np.isfinite(utilization)]
        if finite_utilization.size:
            maximum_utilization = float(np.max(finite_utilization))
            utilization_limit = float(
                quality.get("maximum_coulomb_friction_utilization", 1.05)
            )
            check(
                "coulomb_friction_force_within_limit",
                maximum_utilization <= utilization_limit,
                round(maximum_utilization, 7),
                utilization_limit,
            )
            kinetic_sliding_motions = {
                "slide_push_1obj",
                "slope_slide_down_1obj",
                "slope_slide_up_1obj",
                "ramp_to_flat_1obj",
            }
            if motion in kinetic_sliding_motions and shape != "sphere":
                minimum_utilization = float(
                    quality.get("minimum_kinetic_sliding_friction_utilization", 0.70)
                )
                check(
                    "kinetic_sliding_activates_friction_limit",
                    maximum_utilization >= minimum_utilization,
                    round(maximum_utilization, 7),
                    minimum_utilization,
                )

    rest_frames = int(quality.get("rest_drift_window_frames", 12))
    rest_speed = float(quality.get("rest_linear_speed_threshold_m_s", 0.06))
    rest_angular_speed = float(
        quality.get("rest_angular_speed_threshold_rad_s", 0.10)
    )
    if len(time_s) >= rest_frames:
        window = slice(len(time_s) - rest_frames, len(time_s))
        final_speed = np.linalg.norm(linear_velocity[window], axis=1)
        final_angular_speed = np.linalg.norm(angular_velocity[window], axis=1)
        if (
            bool(np.asarray(contact_mask, dtype=bool)[window].all())
            and float(final_speed.max()) <= rest_speed
            and float(final_angular_speed.max()) <= rest_angular_speed
        ):
            drift = float(
                np.max(np.linalg.norm(positions[window] - positions[len(time_s) - rest_frames], axis=1))
            )
            drift_limit = float(quality.get("maximum_rest_drift_m", 0.012))
            check("bounded_rest_state_drift", drift <= drift_limit, round(drift, 7), drift_limit)

    return {
        "checks": checks,
        "metrics": {
            "maximum_mechanical_energy_j": round(float(np.max(energy)), 7),
            "mechanical_energy_gain_j": round(energy_gain, 7),
            "mechanical_energy_reference_j": round(energy_scale, 7),
            "mechanical_energy_gain_limit_j": round(energy_limit, 7),
            "airborne_frames_for_fit": airborne_count,
        },
    }
