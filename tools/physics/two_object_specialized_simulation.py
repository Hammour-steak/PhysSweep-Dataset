#!/usr/bin/env python3
"""Numerical adapters for two spheres in frozen specialized fixtures."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tools.assets.static_support_proxy import create_pybullet_static_support
from tools.core.paths import join_project_path


def _pybullet() -> Any:
    try:
        import pybullet as pb  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise RuntimeError("pybullet is required for specialized simulation") from error
    return pb


def _engine(scene: dict[str, Any]) -> dict[str, Any]:
    payload = scene["adapter_payload"]
    backend = payload["backend"]
    if scene["backend_binding"]["adapter_id"] == "billiards_two_object_v1":
        return backend["billiards_rules"]["engine"]
    backend_physics = backend.get("physics", {})
    return (
        backend_physics
        if "solver_iterations" in backend_physics
        else scene["source_metadata"]["physics"]["engine"]
    )


def _configure_world(pb: Any, scene: dict[str, Any]) -> None:
    frequency = int(scene["time"]["simulation_hz"])
    engine = _engine(scene)
    pb.resetSimulation()
    pb.setGravity(*[float(value) for value in scene["world"]["gravity_m_s2"]])
    pb.setTimeStep(1.0 / frequency)
    pb.setPhysicsEngineParameter(
        fixedTimeStep=1.0 / frequency,
        numSolverIterations=int(engine["solver_iterations"]),
        deterministicOverlappingPairs=1,
        restitutionVelocityThreshold=float(
            engine["restitution_velocity_threshold_m_s"]
        ),
        enableConeFriction=int(bool(engine["enable_cone_friction"])),
        useSplitImpulse=int(bool(engine["use_split_impulse"])),
    )


def _billiards_fixture(
    pb: Any, scene: dict[str, Any], root: Path
) -> tuple[dict[int, str], Callable[[int, tuple[Any, ...]], bool]]:
    payload = scene["adapter_payload"]
    support = int(
        create_pybullet_static_support(pb, root, payload["static_support_binding"])
    )
    rules = payload["backend"]["billiards_rules"]
    pb.changeDynamics(
        support,
        -1,
        lateralFriction=float(rules["support_dynamics"]["lateral_friction"]),
        restitution=float(rules["support_dynamics"]["restitution"]),
    )
    bed_z = float(
        payload["static_support_binding"]["target_support_frame"]
        ["safe_surface"]["z_m"]
    )
    rail_height = float(rules["quality"]["minimum_rail_contact_height_above_bed_m"])

    def is_rail(body: int, contact: tuple[Any, ...]) -> bool:
        return body == support and float(contact[6][2]) > bed_z + rail_height

    return {support: "pool_table"}, is_rail


def _pinball_fixture(
    pb: Any, scene: dict[str, Any], _root: Path
) -> tuple[dict[int, str], Callable[[int, tuple[Any, ...]], bool]]:
    fixture = scene["adapter_payload"]["fixture"]
    material = fixture["material"]
    shapes: dict[tuple[Any, ...], int] = {}
    bodies: dict[int, str] = {}
    for collider in fixture["colliders"]:
        if collider["shape"] == "box":
            key = ("box", *[float(value) for value in collider["half_extents_m"]])
            if key not in shapes:
                shapes[key] = int(
                    pb.createCollisionShape(pb.GEOM_BOX, halfExtents=list(key[1:]))
                )
        elif collider["shape"] == "cylinder":
            key = (
                "cylinder",
                float(collider["radius_m"]),
                float(collider["length_m"]),
            )
            if key not in shapes:
                shapes[key] = int(
                    pb.createCollisionShape(
                        pb.GEOM_CYLINDER, radius=key[1], height=key[2]
                    )
                )
        else:
            raise ValueError(f"unsupported pinball collider: {collider['shape']}")
        body = int(
            pb.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=shapes[key],
                basePosition=[float(value) for value in collider["position_m"]],
                baseOrientation=[
                    float(value)
                    for value in collider["orientation_quaternion_xyzw"]
                ],
            )
        )
        bodies[body] = str(collider["id"])
        pb.changeDynamics(
            body,
            -1,
            lateralFriction=float(material["contact_friction"]),
            restitution=float(material["contact_restitution"]),
        )
    return bodies, lambda _body, _contact: False


def _marble_fixture(
    pb: Any, scene: dict[str, Any], root: Path
) -> tuple[dict[int, str], Callable[[int, tuple[Any, ...]], bool]]:
    fixture = scene["adapter_payload"]["fixture"]
    bodies: dict[int, str] = {}
    mesh_material = fixture["mesh_material"]
    mesh_shapes: dict[tuple[str, tuple[float, ...]], int] = {}
    for component in fixture["mesh_components"]:
        path = join_project_path(root, str(component["collision"]["path"]))
        scale = tuple(float(value) for value in component["mesh_scale"])
        key = (str(path), scale)
        if key not in mesh_shapes:
            mesh_shapes[key] = int(
                pb.createCollisionShape(
                    pb.GEOM_MESH,
                    fileName=str(path),
                    meshScale=list(scale),
                    flags=pb.GEOM_FORCE_CONCAVE_TRIMESH,
                )
            )
        body = int(
            pb.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=mesh_shapes[key],
                basePosition=[float(value) for value in component["base_position_m"]],
                baseOrientation=[
                    float(value)
                    for value in component["base_orientation_quaternion_xyzw"]
                ],
            )
        )
        bodies[body] = str(component["id"])
        pb.changeDynamics(
            body,
            -1,
            lateralFriction=float(mesh_material["contact_friction"]),
            restitution=float(mesh_material["contact_restitution"]),
        )
    analytic_material = fixture["analytic_material"]
    box_shapes: dict[tuple[float, ...], int] = {}
    for collider in fixture["analytic_colliders"]:
        extents = tuple(float(value) for value in collider["half_extents_m"])
        if extents not in box_shapes:
            box_shapes[extents] = int(
                pb.createCollisionShape(pb.GEOM_BOX, halfExtents=list(extents))
            )
        body = int(
            pb.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=box_shapes[extents],
                basePosition=[float(value) for value in collider["position_m"]],
            )
        )
        bodies[body] = str(collider["id"])
        pb.changeDynamics(
            body,
            -1,
            lateralFriction=float(analytic_material["contact_friction"]),
            restitution=float(analytic_material["contact_restitution"]),
        )
    return bodies, lambda _body, _contact: False


def _create_spheres(pb: Any, scene: dict[str, Any]) -> tuple[list[int], np.ndarray, np.ndarray]:
    bodies = []
    runtime_material = []
    runtime_inertia = []
    billiards = scene["backend_binding"]["adapter_id"] == "billiards_two_object_v1"
    ball_rules = (
        scene["adapter_payload"]["backend"]["billiards_rules"]["ball_dynamics"]
        if billiards
        else {}
    )
    shapes: dict[float, int] = {}
    for record in scene["objects"]:
        proxy = record["collision_proxy"]
        radius = (
            float(proxy["radius_m"])
            if "radius_m" in proxy
            else float(proxy["size_m"][0]) / 2.0
        )
        if radius not in shapes:
            shapes[radius] = int(pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius))
        material = record["material"]
        initial = record["initial_state"]
        orientation = initial.get(
            "orientation_quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]
        )
        body = int(
            pb.createMultiBody(
                baseMass=float(material["mass_kg"]),
                baseCollisionShapeIndex=shapes[radius],
                basePosition=[float(value) for value in initial["position_m"]],
                baseOrientation=[float(value) for value in orientation],
            )
        )
        pb.resetBaseVelocity(
            body,
            linearVelocity=[float(value) for value in initial["linear_velocity_m_s"]],
            angularVelocity=[float(value) for value in initial["angular_velocity_rad_s"]],
        )
        pb.changeDynamics(
            body,
            -1,
            lateralFriction=float(material["contact_friction"]),
            restitution=float(material["contact_restitution"]),
            rollingFriction=float(material.get("rolling_friction", ball_rules.get("rolling_friction", 0.0))),
            spinningFriction=float(material.get("spinning_friction", ball_rules.get("spinning_friction", 0.0))),
            linearDamping=float(material.get("linear_damping", ball_rules.get("linear_damping", 0.0))),
            angularDamping=float(material.get("angular_damping", ball_rules.get("angular_damping", 0.0))),
            contactProcessingThreshold=float(ball_rules.get("contact_processing_threshold_m", 0.0)),
        )
        info = pb.getDynamicsInfo(body, -1)
        runtime_material.append([float(info[0]), float(info[1]), float(info[5])])
        runtime_inertia.append([float(value) for value in info[2]])
        bodies.append(body)
    return (
        bodies,
        np.asarray(runtime_material, dtype=np.float64),
        np.asarray(runtime_inertia, dtype=np.float64),
    )


def simulate_two_object_specialized(
    scene: dict[str, Any], root: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run one two-sphere scene and audit its declared interaction semantics."""

    if len(scene["objects"]) != 2:
        raise ValueError("specialized two-object simulation requires two objects")
    adapter = str(scene["backend_binding"]["adapter_id"])
    fixture_builder = {
        "billiards_two_object_v1": _billiards_fixture,
        "passive_pinball_two_object_v1": _pinball_fixture,
        "marble_run_two_object_v1": _marble_fixture,
    }.get(adapter)
    if fixture_builder is None:
        raise ValueError(f"unsupported specialized two-object adapter: {adapter}")
    pb = _pybullet()
    frame_count = int(scene["time"]["frame_count"])
    output_fps = int(scene["time"]["output_fps"])
    simulation_hz = int(scene["time"]["simulation_hz"])
    if simulation_hz % output_fps:
        raise ValueError("specialized simulation frequency is not frame aligned")
    steps_per_frame = simulation_hz // output_fps
    client = pb.connect(pb.DIRECT)
    if client < 0:
        raise RuntimeError("PyBullet DIRECT connection failed")
    try:
        _configure_world(pb, scene)
        fixture_bodies, is_rail = fixture_builder(pb, scene, root)
        bodies, runtime_material, runtime_inertia = _create_spheres(pb, scene)
        positions = np.zeros((frame_count, 2, 3), dtype=np.float64)
        orientations = np.zeros((frame_count, 2, 4), dtype=np.float64)
        linear = np.zeros((frame_count, 2, 3), dtype=np.float64)
        angular = np.zeros((frame_count, 2, 3), dtype=np.float64)
        contact_count = np.zeros((frame_count, 2), dtype=np.int32)
        path_lengths = np.zeros(2, dtype=np.float64)
        previous = np.asarray(
            [record["initial_state"]["position_m"] for record in scene["objects"]],
            dtype=np.float64,
        )
        touched = [set(), set()]
        first_pair_step: int | None = None
        first_rail_step: int | None = None
        minimum_contact_distance = 0.0
        maximum_speed = 0.0

        def observe(frame: int) -> None:
            for index, body in enumerate(bodies):
                position, orientation = pb.getBasePositionAndOrientation(body)
                velocity, spin = pb.getBaseVelocity(body)
                positions[frame, index] = position
                orientations[frame, index] = orientation
                linear[frame, index] = velocity
                angular[frame, index] = spin
                contact_count[frame, index] = len(pb.getContactPoints(bodyA=body))

        observe(0)
        total_steps = (frame_count - 1) * steps_per_frame
        for step in range(1, total_steps + 1):
            pb.stepSimulation()
            pair_contacts = pb.getContactPoints(bodyA=bodies[0], bodyB=bodies[1])
            if pair_contacts and first_pair_step is None:
                first_pair_step = step
            for contact in pair_contacts:
                minimum_contact_distance = min(
                    minimum_contact_distance, float(contact[8])
                )
            for index, body in enumerate(bodies):
                position = np.asarray(
                    pb.getBasePositionAndOrientation(body)[0], dtype=np.float64
                )
                path_lengths[index] += float(np.linalg.norm(position - previous[index]))
                previous[index] = position
                maximum_speed = max(
                    maximum_speed,
                    float(np.linalg.norm(pb.getBaseVelocity(body)[0])),
                )
                for contact in pb.getContactPoints(bodyA=body):
                    other = int(contact[2])
                    if other in fixture_bodies:
                        touched[index].add(fixture_bodies[other])
                        if first_rail_step is None and is_rail(other, contact):
                            first_rail_step = step
                    minimum_contact_distance = min(
                        minimum_contact_distance, float(contact[8])
                    )
            if step % steps_per_frame == 0:
                observe(step // steps_per_frame)
    finally:
        pb.disconnect(client)

    quality = scene["adapter_payload"]["quality"]
    first_pair_time = (
        None if first_pair_step is None else first_pair_step / float(simulation_hz)
    )
    checks = {
        "finite_trajectory": all(
            np.isfinite(value).all()
            for value in (positions, orientations, linear, angular)
        ),
        "two_dynamic_objects": len(bodies) == 2,
        "pair_contact_observed": first_pair_step is not None,
        "first_pair_contact_within_limit": first_pair_time is not None
        and first_pair_time <= float(quality["maximum_first_pair_contact_time_s"]),
        "maximum_penetration": -minimum_contact_distance
        <= float(quality["maximum_penetration_m"]),
    }
    if adapter == "billiards_two_object_v1":
        checks["rail_contact_before_pair_contact_absent"] = not bool(
            quality.get("rail_contact_before_pair_contact_is_forbidden", False)
        ) or first_rail_step is None or (
            first_pair_step is not None and first_rail_step >= first_pair_step
        )
    elif adapter == "passive_pinball_two_object_v1":
        checks["minimum_path_length_per_object"] = bool(
            np.all(path_lengths >= float(quality["minimum_path_length_per_object_m"]))
        )
        checks["minimum_distinct_fixture_contacts_per_object"] = all(
            len(records)
            >= int(quality["minimum_distinct_fixture_contacts_per_object"])
            for records in touched
        )
    else:
        required = set(str(value) for value in quality["required_track_contact_ids"])
        checks["minimum_path_length_per_object"] = bool(
            np.all(path_lengths >= float(quality["minimum_path_length_per_object_m"]))
        )
        checks["required_track_contacts_per_object"] = all(
            required <= records for records in touched
        )
    arrays = {
        "time_s": np.arange(frame_count, dtype=np.float64) / float(output_fps),
        "position_m": positions,
        "quaternion_wxyz": orientations[:, :, [3, 0, 1, 2]],
        "linear_velocity_m_s": linear,
        "angular_velocity_rad_s": angular,
        "contact_count": contact_count,
        "runtime_material": runtime_material,
        "inertia_diagonal_kg_m2": runtime_inertia,
        "adapter__quaternion_xyzw": orientations,
    }
    audit = {
        "schema_version": "physweep_two_object_specialized_audit_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "first_pair_contact_time_s": (
                None if first_pair_time is None else round(first_pair_time, 9)
            ),
            "maximum_penetration_m": round(-minimum_contact_distance, 9),
            "path_length_m": [round(float(value), 9) for value in path_lengths],
            "maximum_linear_speed_m_s": round(maximum_speed, 9),
            "fixture_contacts": [sorted(records) for records in touched],
        },
    }
    return arrays, audit
