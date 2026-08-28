"""PyBullet simulation and trajectory audits for admitted asset proxies."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as pb

from tools.assets.static_support_proxy import create_pybullet_static_support
from tools.motion_rules.one_object import asset_motion_group
from tools.physics.physics_invariants import (
    PROXY_SHAPE_CODE,
    additional_physics_invariants,
    maximum_coulomb_utilization,
    runtime_collision_descriptors,
)
from tools.physics.physics_time_step import simulation_hz_for_min_extent
from tools.physics.rigid_trajectory import active_motion_duration_s


ASSET_AUDIT_VERSION = "physweep_asset_profile_audit_v3"


def asset_motion_usefulness(
    backend: dict[str, Any], profile: str
) -> dict[str, float]:
    try:
        minimum_duration = float(
            backend["asset_proxy_rules"]["quality"][
                "minimum_active_duration_s_by_profile"
            ][profile]
        )
    except KeyError as error:
        raise ValueError(
            f"asset motion profile lacks a useful-duration contract: {profile}"
        ) from error
    return {
        "minimum_active_duration_s": minimum_duration,
        "active_speed_threshold_m_s": float(
            backend["quality"]["active_speed_threshold_m_s"]
        ),
    }


def quaternion(euler_degrees: list[float]) -> tuple[float, float, float, float]:
    return pb.getQuaternionFromEuler([math.radians(float(value)) for value in euler_degrees])


def compound_shape(colliders: list[dict[str, Any]]) -> int:
    types: list[int] = []
    half_extents: list[list[float]] = []
    radii: list[float] = []
    lengths: list[float] = []
    positions: list[list[float]] = []
    orientations: list[tuple[float, float, float, float]] = []
    for collider in colliders:
        shape = str(collider["shape"])
        size = [float(value) for value in collider["size_m"]]
        types.append({"box": pb.GEOM_BOX, "sphere": pb.GEOM_SPHERE, "cylinder": pb.GEOM_CYLINDER}[shape])
        half_extents.append([value * 0.5 for value in size])
        if shape == "sphere":
            radii.append(0.5 * max(size))
        elif shape == "cylinder":
            radii.append(0.5 * max(size[:2]))
        else:
            radii.append(0.0)
        lengths.append(size[2] if shape == "cylinder" else 0.0)
        positions.append([float(value) for value in collider["position_m"]])
        orientations.append(quaternion(collider["rotation_euler_degrees"]))
    return int(
        pb.createCollisionShapeArray(
            shapeTypes=types,
            halfExtents=half_extents,
            radii=radii,
            lengths=lengths,
            collisionFramePositions=positions,
            collisionFrameOrientations=orientations,
        )
    )


def create_body(record: dict[str, Any], mass: float, position: list[float]) -> int:
    return int(
        pb.createMultiBody(
            baseMass=float(mass),
            baseCollisionShapeIndex=compound_shape(record["proxy"]["colliders"]),
            basePosition=position,
        )
    )


def local_bounds(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    client = pb.connect(pb.DIRECT)
    if client < 0:
        raise RuntimeError("PyBullet DIRECT connection failed during proxy measurement")
    try:
        body = create_body(record, 0.0, [0.0, 0.0, 0.0])
        low, high = pb.getAABB(body)
        return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)
    finally:
        pb.disconnect(client)


def audit_asset_trajectory(
    profile: str,
    arrays: dict[str, np.ndarray],
    minimum_contact_distance: float,
    initial_penetration_m: float,
    initial_prop_contacts: int,
    proxy_extent_m: list[float],
    surface: dict[str, Any],
    quality: dict[str, Any],
    asset_rules: dict[str, Any],
    initial: dict[str, Any] | None = None,
    mass: float | None = None,
    material: dict[str, Any] | None = None,
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
    runtime_static_dynamics_error: float | None = None,
    expected_motion: dict[str, float] | None = None,
) -> dict[str, Any]:
    asset_motion_group(profile)
    profile_quality = asset_rules["quality"]
    positions = arrays["position_m"]
    linear = arrays["linear_velocity_m_s"]
    angular = arrays["angular_velocity_rad_s"]
    support = arrays["support_contact"].astype(bool)
    ground = arrays["ground_contact"].astype(bool)
    prop = arrays["prop_contact"].astype(bool)
    displacement = np.linalg.norm(positions - positions[0], axis=1)
    linear_speed = np.linalg.norm(linear, axis=1)
    angular_speed = np.linalg.norm(angular, axis=1)
    contact_radius_m = 0.5 * min(float(value) for value in proxy_extent_m)
    rotational_surface_speed = angular_speed * contact_radius_m
    expected_motion = expected_motion or asset_motion_usefulness(
        {"asset_proxy_rules": asset_rules, "quality": quality}, profile
    )
    active_duration_s = active_motion_duration_s(
        linear,
        arrays["time_s"],
        float(expected_motion["active_speed_threshold_m_s"]),
    )
    finite = all(
        np.isfinite(value).all()
        for key, value in arrays.items()
        if key != "maximum_coulomb_friction_utilization"
    )
    penetration = max(0.0, -float(minimum_contact_distance))
    minimum_proxy_extent_m = min(float(value) for value in proxy_extent_m)
    penetration_limit = min(
        float(quality["maximum_trajectory_penetration_m"]),
        float(quality["maximum_trajectory_penetration_fraction_of_min_extent"])
        * minimum_proxy_extent_m,
    )
    penetration_tolerance = max(
        float(quality.get("audit_distance_absolute_tolerance_m", 0.0)),
        penetration_limit
        * float(quality.get("audit_distance_relative_tolerance_fraction", 0.0)),
    )
    checks: dict[str, bool] = {
        "finite_trajectory": bool(finite),
        "initial_penetration_within_limit": initial_penetration_m
        <= float(quality["maximum_initial_penetration_m"]),
        "no_initial_prop_overlap": initial_prop_contacts == 0,
        "no_unplanned_static_prop_contact": not bool(prop.any()),
        "contact_observed": bool(support.any() or ground.any()),
        "penetration_within_limit": penetration
        <= penetration_limit + penetration_tolerance,
        "visible_motion": float(displacement.max())
        >= float(profile_quality["minimum_visible_motion_m"]),
        "linear_speed_within_limit": float(linear_speed.max())
        <= float(profile_quality["maximum_linear_speed_m_s"]),
        "angular_speed_within_limit": float(angular_speed.max())
        <= float(profile_quality["maximum_angular_speed_rad_s"]),
        "rotational_surface_speed_within_limit": float(
            rotational_surface_speed.max()
        )
        <= float(quality["maximum_rotational_surface_speed_m_s"]),
        "world_bounds_valid": float(np.min(positions[:, 2]))
        >= float(profile_quality["minimum_world_z_m"]),
        "useful_active_duration": active_duration_s + 1.0e-9
        >= float(expected_motion["minimum_active_duration_s"]),
    }
    first_support_indices = np.flatnonzero(support)
    first_support = int(first_support_indices[0]) if first_support_indices.size else None
    initial_direction = np.asarray(linear[0, :2], dtype=np.float64).copy()
    initial_speed_xy = float(np.linalg.norm(initial_direction))
    if initial_speed_xy > 1.0e-8:
        initial_direction /= initial_speed_xy
    projected = (positions[:, :2] - positions[0, :2]) @ initial_direction

    if profile in {"vertical_drop", "workbench_clear_zone_drop"}:
        contact_index = first_support if first_support is not None else len(positions) - 1
        lateral_drift = float(
            np.linalg.norm(positions[contact_index, :2] - positions[0, :2])
        )
        vertical_drop = float(positions[0, 2] - positions[: contact_index + 1, 2].min())
        checks.update(
            {
                "drop_starts_airborne": first_support is not None
                and first_support
                >= int(profile_quality["drop_minimum_airborne_frames"])
                and not support[:first_support].any(),
                "drop_contacts_primary_support": first_support is not None,
                "drop_lateral_drift_within_limit": lateral_drift
                <= float(quality["maximum_unforced_lateral_drift_m"]),
                "drop_vertical_extent": vertical_drop
                >= float(profile_quality["drop_minimum_vertical_extent_m"]),
            }
        )
        if profile == "workbench_clear_zone_drop":
            center_x, center_y = [float(value) for value in surface["center_xy_m"]]
            size_x, size_y = [float(value) for value in surface["size_xy_m"]]
            footprint_radius = 0.5 * max(
                float(proxy_extent_m[0]), float(proxy_extent_m[1])
            )
            checks["workbench_initial_footprint_in_safe_zone"] = bool(
                abs(float(positions[0, 0]) - center_x)
                <= 0.5 * size_x - footprint_radius
                and abs(float(positions[0, 1]) - center_y)
                <= 0.5 * size_y - footprint_radius
            )
    elif profile in {"resting_push", "diagonal_push"}:
        checks.update(
            {
                "push_has_initial_support_contact": bool(
                    support[
                        : int(profile_quality["push_initial_contact_frames"])
                    ].any()
                ),
                "push_sustains_support_contact": float(np.mean(support))
                >= float(profile_quality["push_minimum_support_contact_fraction"]),
                "push_follows_initial_direction": float(projected.max())
                >= float(profile_quality["push_minimum_projected_displacement_m"]),
            }
        )
        if profile == "diagonal_push":
            delta = positions[:, :2] - positions[0, :2]
            checks["diagonal_push_has_two_axis_motion"] = bool(
                float(np.max(np.abs(delta[:, 0])))
                >= float(profile_quality["diagonal_minimum_x_displacement_m"])
                and float(np.max(np.abs(delta[:, 1])))
                >= float(profile_quality["diagonal_minimum_y_displacement_m"])
            )
    elif profile == "edge_exit":
        exit_index = None
        if first_support is not None:
            exit_frames = int(profile_quality["edge_exit_consecutive_frames"])
            for index in range(first_support + 1, len(support) - exit_frames + 1):
                if not support[index : index + exit_frames].any():
                    exit_index = index
                    break
        vertical_drop = float(positions[0, 2] - positions[:, 2].min())
        checks.update(
            {
                "edge_has_initial_support_contact": int(support.sum())
                >= int(profile_quality["edge_minimum_initial_support_frames"]),
                "edge_exits_primary_support": exit_index is not None,
                "edge_contacts_ground_after_exit": exit_index is not None
                and bool(ground[exit_index:].any()),
                "edge_vertical_drop": vertical_drop
                >= max(
                    float(profile_quality["edge_minimum_vertical_drop_m"]),
                    float(surface["z_m"])
                    * float(
                        profile_quality[
                            "edge_minimum_support_height_drop_fraction"
                        ]
                    ),
                ),
            }
        )
    elif profile == "workbench_long_axis_push":
        center_x, center_y = [float(value) for value in surface["center_xy_m"]]
        size_x, size_y = [float(value) for value in surface["size_xy_m"]]
        footprint_radius = 0.5 * max(
            float(proxy_extent_m[0]), float(proxy_extent_m[1])
        )
        checks.update(
            {
                "workbench_initial_footprint_in_safe_zone": bool(
                    abs(float(positions[0, 0]) - center_x)
                    <= 0.5 * size_x - footprint_radius
                    and abs(float(positions[0, 1]) - center_y)
                    <= 0.5 * size_y - footprint_radius
                ),
                "workbench_push_has_initial_support_contact": bool(
                    support[
                        : int(profile_quality["push_initial_contact_frames"])
                    ].any()
                ),
                "workbench_push_sustains_support_contact": float(np.mean(support))
                >= float(
                    profile_quality["workbench_minimum_support_contact_fraction"]
                ),
                "workbench_push_follows_long_axis": float(projected.max())
                >= float(
                    profile_quality["workbench_minimum_projected_displacement_m"]
                ),
            }
        )
    else:
        raise ValueError(f"unsupported audited motion profile: {profile}")

    additional = {"checks": [], "metrics": {}}
    if initial is not None and mass is not None and material is not None:
        quaternions_wxyz = np.column_stack(
            [
                arrays["quaternion_xyzw"][:, 3],
                arrays["quaternion_xyzw"][:, 0],
                arrays["quaternion_xyzw"][:, 1],
                arrays["quaternion_xyzw"][:, 2],
            ]
        )
        effective_material = {
            "friction": float(material["friction"]),
            "restitution": float(material["restitution"]),
            "rolling_friction": float(material["rolling_friction"]),
            "spinning_friction": float(material["spinning_friction"]),
        }
        additional = additional_physics_invariants(
            motion=profile,
            time_s=arrays["time_s"],
            positions=positions,
            quaternions_wxyz=quaternions_wxyz,
            linear_velocity=linear,
            angular_velocity=angular,
            contact_mask=support | ground | prop,
            shape="cuboid",
            size_m=np.asarray(proxy_extent_m, dtype=np.float64),
            mass_kg=mass,
            gravity_m_s2=-9.81,
            initial_state=initial,
            material=effective_material,
            quality=quality,
            runtime_dynamics=runtime_dynamics,
            runtime_inertia_diagonal_kg_m2=runtime_inertia_diagonal_kg_m2,
            expected_proxy_shape_codes=expected_proxy_shape_codes,
            expected_proxy_dimensions_m=expected_proxy_dimensions_m,
            expected_proxy_positions_m=expected_proxy_positions_m,
            expected_proxy_quaternions_xyzw=expected_proxy_quaternions_xyzw,
            runtime_proxy_shape_codes=runtime_proxy_shape_codes,
            runtime_proxy_dimensions_m=runtime_proxy_dimensions_m,
            runtime_proxy_positions_m=runtime_proxy_positions_m,
            runtime_proxy_quaternions_xyzw=runtime_proxy_quaternions_xyzw,
            coulomb_friction_utilization=arrays.get(
                "maximum_coulomb_friction_utilization"
            ),
        )
        if runtime_static_dynamics_error is not None:
            additional["checks"].append(
                {
                    "id": "pybullet_static_dynamics_match_configuration",
                    "passed": runtime_static_dynamics_error
                    <= float(quality["parameter_match_absolute_tolerance"]),
                    "value": round(runtime_static_dynamics_error, 9),
                    "threshold": float(
                        quality["parameter_match_absolute_tolerance"]
                    ),
                }
            )
    checks.update(
        {record["id"]: bool(record["passed"]) for record in additional["checks"]}
    )

    return {
        "passed": all(checks.values()),
        "profile": profile,
        "checks": checks,
        "maximum_displacement_m": round(float(displacement.max()), 6),
        "maximum_linear_speed_m_s": round(float(linear_speed.max()), 6),
        "maximum_angular_speed_rad_s": round(float(angular_speed.max()), 6),
        "maximum_rotational_surface_speed_m_s": round(
            float(rotational_surface_speed.max()), 6
        ),
        "active_motion_duration_s": round(active_duration_s, 6),
        "minimum_active_motion_duration_s": float(
            expected_motion["minimum_active_duration_s"]
        ),
        "minimum_contact_distance_m": round(float(minimum_contact_distance), 7),
        "maximum_penetration_m": round(penetration, 7),
        "penetration_limit_m": round(penetration_limit, 7),
        "penetration_tolerance_m": round(penetration_tolerance, 7),
        "initial_penetration_m": round(initial_penetration_m, 7),
        "support_contact_frames": int(support.sum()),
        "ground_contact_frames": int(ground.sum()),
        "prop_contact_frames": int(prop.sum()),
        "additional_invariant_details": additional,
    }


def simulate_scene(
    root: Path,
    dynamic: dict[str, Any],
    static_support_binding: dict[str, Any],
    prop_record: dict[str, Any] | None,
    prop_binding: dict[str, Any] | None,
    initial: dict[str, Any],
    mass: float,
    duration_s: float,
    output_fps: int,
    profile: str,
    backend: dict[str, Any],
    expected_motion: dict[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    asset_rules = backend["asset_proxy_rules"]
    engine_rules = asset_rules["engine"]
    contact_rules = asset_rules["contact"]
    proxy_low, proxy_high = local_bounds(dynamic)
    minimum_proxy_extent_m = float(np.min(proxy_high - proxy_low))
    simulation_hz = simulation_hz_for_min_extent(
        backend["engine"], minimum_proxy_extent_m
    )
    steps_per_frame = simulation_hz // output_fps
    frame_count = int(round(duration_s * output_fps)) + 1
    client = pb.connect(pb.DIRECT)
    if client < 0:
        raise RuntimeError("PyBullet DIRECT connection failed")
    try:
        pb.resetSimulation()
        pb.setGravity(0.0, 0.0, -9.81)
        pb.setTimeStep(1.0 / simulation_hz)
        pb.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / simulation_hz,
            numSolverIterations=int(backend["engine"]["solver_iterations"]),
            deterministicOverlappingPairs=1,
            restitutionVelocityThreshold=float(
                engine_rules["restitution_velocity_threshold_m_s"]
            ),
            enableConeFriction=int(bool(engine_rules["enable_cone_friction"])),
            useSplitImpulse=int(bool(engine_rules["use_split_impulse"])),
        )
        ground_shape = pb.createCollisionShape(pb.GEOM_PLANE)
        ground = int(pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=ground_shape))
        pb.changeDynamics(
            ground,
            -1,
            lateralFriction=float(contact_rules["ground"]["lateral_friction"]),
            restitution=float(contact_rules["ground"]["restitution"]),
        )
        support_body = create_pybullet_static_support(
            pb, root, static_support_binding
        )
        pb.changeDynamics(
            support_body,
            -1,
            lateralFriction=float(contact_rules["support"]["lateral_friction"]),
            restitution=float(contact_rules["support"]["restitution"]),
        )
        prop_body = None
        if prop_record and prop_binding:
            prop_body = create_body(prop_record, 0.0, [float(value) for value in prop_binding["position_m"]])
            yaw = quaternion([0.0, 0.0, float(prop_binding["yaw_degrees"])])
            pb.resetBasePositionAndOrientation(prop_body, prop_binding["position_m"], yaw)
            pb.changeDynamics(
                prop_body,
                -1,
                lateralFriction=float(
                    contact_rules["static_prop"]["lateral_friction"]
                ),
                restitution=float(contact_rules["static_prop"]["restitution"]),
            )
        body = create_body(dynamic, mass, [float(value) for value in initial["position_m"]])
        pb.resetBasePositionAndOrientation(
            body,
            initial["position_m"],
            initial["orientation_quaternion_xyzw"],
        )
        material = dynamic["proxy"].get("material", {})
        dynamic_defaults = contact_rules["dynamic_defaults"]
        pb.changeDynamics(
            body,
            -1,
            lateralFriction=float(material.get("friction", 0.45)),
            restitution=float(material.get("restitution", 0.15)),
            linearDamping=float(dynamic_defaults["linear_damping"]),
            angularDamping=float(dynamic_defaults["angular_damping"]),
            rollingFriction=float(dynamic_defaults["rolling_friction"]),
            spinningFriction=float(dynamic_defaults["spinning_friction"]),
            contactProcessingThreshold=float(
                dynamic_defaults["contact_processing_threshold_m"]
            ),
        )
        dynamics_info = pb.getDynamicsInfo(body, -1)
        runtime_proxy = runtime_collision_descriptors(pb, body)
        runtime_inertia = np.asarray(dynamics_info[2], dtype=np.float64)
        runtime_dynamics = np.asarray(
            [
                dynamics_info[0],
                dynamics_info[1],
                dynamics_info[5],
                dynamics_info[6],
                dynamics_info[7],
            ],
            dtype=np.float64,
        )
        pb.resetBaseVelocity(
            body,
            linearVelocity=initial["linear_velocity_m_s"],
            angularVelocity=initial["angular_velocity_rad_s"],
        )
        pb.performCollisionDetection()
        initial_contacts = list(pb.getContactPoints(bodyA=body))
        static_friction_by_body = {
            ground: float(contact_rules["ground"]["lateral_friction"]),
            support_body: float(contact_rules["support"]["lateral_friction"]),
        }
        if prop_body is not None:
            static_friction_by_body[prop_body] = float(
                contact_rules["static_prop"]["lateral_friction"]
            )
        runtime_static_values = []
        expected_static_values = []
        for static_body, rule_name in (
            (ground, "ground"),
            (support_body, "support"),
            (prop_body, "static_prop"),
        ):
            if static_body is None:
                continue
            info = pb.getDynamicsInfo(static_body, -1)
            runtime_static_values.append([float(info[1]), float(info[5])])
            expected_static_values.append(
                [
                    float(contact_rules[rule_name]["lateral_friction"]),
                    float(contact_rules[rule_name]["restitution"]),
                ]
            )
        runtime_static_dynamics_error = float(
            np.max(
                np.abs(
                    np.asarray(runtime_static_values, dtype=np.float64)
                    - np.asarray(expected_static_values, dtype=np.float64)
                )
            )
        )
        initial_prop_contacts = (
            len(pb.getContactPoints(bodyA=body, bodyB=prop_body)) if prop_body else 0
        )
        initial_penetration_m = max(
            0.0,
            -min((float(contact[8]) for contact in initial_contacts), default=0.0),
        )
        positions = []
        quaternions = []
        linear_velocities = []
        angular_velocities = []
        aabb_minimums = []
        aabb_maximums = []
        minimum_contact_distance = 0.0
        support_contact_flags = [
            any(int(contact[2]) == support_body for contact in initial_contacts)
        ]
        ground_contact_flags = [
            any(int(contact[2]) == ground for contact in initial_contacts)
        ]
        prop_contact_flags = [
            prop_body is not None
            and any(int(contact[2]) == prop_body for contact in initial_contacts)
        ]
        friction_utilization = [
            maximum_coulomb_utilization(
                initial_contacts,
                float(material.get("friction", 0.45)),
                static_friction_by_body,
            )
        ]
        for frame in range(frame_count):
            position, orientation = pb.getBasePositionAndOrientation(body)
            linear, angular = pb.getBaseVelocity(body)
            aabb_minimum, aabb_maximum = pb.getAABB(body)
            positions.append(position)
            quaternions.append(orientation)
            linear_velocities.append(linear)
            angular_velocities.append(angular)
            aabb_minimums.append(aabb_minimum)
            aabb_maximums.append(aabb_maximum)
            if frame == frame_count - 1:
                break
            support_contact = False
            ground_contact = False
            prop_contact = False
            frame_friction_utilization = 0.0
            for _ in range(steps_per_frame):
                pb.stepSimulation()
                contacts = list(pb.getContactPoints(bodyA=body))
                if contacts:
                    minimum_contact_distance = min(
                        minimum_contact_distance,
                        min(float(contact[8]) for contact in contacts),
                    )
                support_contact |= any(int(contact[2]) == support_body for contact in contacts)
                ground_contact |= any(int(contact[2]) == ground for contact in contacts)
                prop_contact |= prop_body is not None and any(int(contact[2]) == prop_body for contact in contacts)
                step_utilization = maximum_coulomb_utilization(
                    contacts,
                    float(material.get("friction", 0.45)),
                    static_friction_by_body,
                )
                frame_friction_utilization = max(
                    frame_friction_utilization, step_utilization
                )
            support_contact_flags.append(support_contact)
            ground_contact_flags.append(ground_contact)
            prop_contact_flags.append(prop_contact)
            friction_utilization.append(frame_friction_utilization)
    finally:
        pb.disconnect(client)

    arrays = {
        "time_s": np.arange(frame_count, dtype=np.float64) / float(output_fps),
        "position_m": np.asarray(positions, dtype=np.float64),
        "quaternion_xyzw": np.asarray(quaternions, dtype=np.float64),
        "linear_velocity_m_s": np.asarray(linear_velocities, dtype=np.float64),
        "angular_velocity_rad_s": np.asarray(angular_velocities, dtype=np.float64),
        "aabb_min_m": np.asarray(aabb_minimums, dtype=np.float64),
        "aabb_max_m": np.asarray(aabb_maximums, dtype=np.float64),
        "support_contact": np.asarray(support_contact_flags, dtype=np.int8),
        "ground_contact": np.asarray(ground_contact_flags, dtype=np.int8),
        "prop_contact": np.asarray(prop_contact_flags, dtype=np.int8),
        "maximum_coulomb_friction_utilization": np.asarray(
            friction_utilization, dtype=np.float64
        ),
        "runtime_dynamics": runtime_dynamics,
        "runtime_inertia_diagonal_kg_m2": runtime_inertia,
        "runtime_proxy_shape_codes": runtime_proxy["shape_codes"],
        "runtime_proxy_dimensions_m": runtime_proxy["dimensions_m"],
        "runtime_proxy_positions_m": runtime_proxy["positions_m"],
        "runtime_proxy_quaternions_xyzw": runtime_proxy["quaternions_xyzw"],
    }
    expected_proxy_shape_codes = np.asarray(
        [
            PROXY_SHAPE_CODE[str(collider["shape"])]
            for collider in dynamic["proxy"]["colliders"]
        ],
        dtype=np.int32,
    )
    expected_proxy_dimensions = np.asarray(
        [collider["size_m"] for collider in dynamic["proxy"]["colliders"]],
        dtype=np.float64,
    )
    expected_proxy_positions = np.asarray(
        [collider["position_m"] for collider in dynamic["proxy"]["colliders"]],
        dtype=np.float64,
    )
    expected_proxy_quaternions = np.asarray(
        [
            quaternion(collider["rotation_euler_degrees"])
            for collider in dynamic["proxy"]["colliders"]
        ],
        dtype=np.float64,
    )
    audit = audit_asset_trajectory(
        profile,
        arrays,
        minimum_contact_distance,
        initial_penetration_m,
        initial_prop_contacts,
        [float(value) for value in proxy_high - proxy_low],
        static_support_binding["target_support_frame"]["safe_surface"],
        backend["quality"],
        asset_rules,
        initial,
        mass,
        {
            "friction": float(material.get("friction", 0.45)),
            "restitution": float(material.get("restitution", 0.15)),
            "rolling_friction": float(dynamic_defaults["rolling_friction"]),
            "spinning_friction": float(dynamic_defaults["spinning_friction"]),
        },
        runtime_dynamics,
        runtime_inertia,
        expected_proxy_shape_codes,
        expected_proxy_dimensions,
        expected_proxy_positions,
        expected_proxy_quaternions,
        runtime_proxy["shape_codes"],
        runtime_proxy["dimensions_m"],
        runtime_proxy["positions_m"],
        runtime_proxy["quaternions_xyzw"],
        runtime_static_dynamics_error,
        expected_motion,
      )
    audit["audit_version"] = ASSET_AUDIT_VERSION
    audit["simulation_hz"] = simulation_hz
    audit["collision_authority"] = "exact_static_proxy"
    audit["support_binding_sha256"] = static_support_binding[
        "binding_sha256"
    ]
    return arrays, audit
