#!/usr/bin/env python3
"""Shared bounded sphere execution in frozen specialized fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from tools.physics.contact_events import ContactEventCollector

from tools.assets.static_support_proxy import create_pybullet_static_support
from tools.core.hashing import sha256_file
from tools.core.paths import join_project_path
from tools.physics.contact_configuration import contact_processing_threshold
from tools.physics.solver_configuration import (
    apply_solver_configuration, resolved_solver_configuration,
)


def _pybullet() -> Any:
    try:
        import pybullet as pb  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise RuntimeError("pybullet is required for specialized simulation") from error
    return pb


def _configure_world(pb: Any, scene: dict[str, Any]) -> dict[str, Any]:
    frequency = int(scene["time"]["simulation_hz"])
    engine = resolved_solver_configuration(
        scene["backend_binding"]["adapter_id"], scene.get("source_metadata", {}), scene,
    )
    pb.resetSimulation()
    pb.setGravity(*[float(value) for value in scene["world"]["gravity_m_s2"]])
    pb.setTimeStep(1.0 / frequency)
    return apply_solver_configuration(pb, engine)


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
    mesh_shapes: dict[tuple[str, str, tuple[float, ...]], int] = {}
    for component in fixture["mesh_components"]:
        path = join_project_path(root, str(component["collision"]["path"]))
        expected = str(component["collision"]["sha256"])
        scale = tuple(float(value) for value in component["mesh_scale"])
        key = (str(path), expected, scale)
        if key not in mesh_shapes:
            if sha256_file(path) != expected:
                raise ValueError(f"marble-run collision hash changed: {component['id']}")
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


def _create_spheres(
    pb: Any, scene: dict[str, Any], *, execution: list[dict[str, Any]] | None = None
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    bodies = []
    runtime_material = []
    runtime_inertia = []
    runtime_extras = []
    threshold = contact_processing_threshold(scene)
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
            rollingFriction=float(material["rolling_friction"]),
            spinningFriction=float(material["spinning_friction"]),
            linearDamping=float(material["linear_damping"]),
            angularDamping=float(material["angular_damping"]),
            contactProcessingThreshold=threshold,
        )
        if execution is not None:
            execution.append({"object_id": record["object_id"], "contactProcessingThreshold": threshold})
        info = pb.getDynamicsInfo(body, -1)
        runtime_material.append([float(info[0]), float(info[1]), float(info[5])])
        runtime_inertia.append([float(value) for value in info[2]])
        # PyBullet exposes rolling/spinning friction through getDynamicsInfo;
        # damping has no getter, so retain the values passed to changeDynamics.
        runtime_extras.append([
            float(info[6]), float(info[7]),
            float(material["linear_damping"]), float(material["angular_damping"]),
        ])
        bodies.append(body)
    return (
        bodies,
        np.asarray(runtime_material, dtype=np.float64),
        np.asarray(runtime_inertia, dtype=np.float64),
        np.asarray(runtime_extras, dtype=np.float64),
    )


def simulate_specialized_spheres(
    scene: dict[str, Any], root: Path, *, fixture_builder: Callable,
    pybullet_factory: Callable = _pybullet,
    configure_world: Callable = _configure_world,
    create_spheres: Callable = _create_spheres,
    event_distance_tolerance_m: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Execute 2/3 spheres; family/interaction admission belongs to the caller.

    The three-sphere path requires complete substep event evidence. It is not
    registered as a production adapter merely by this execution primitive.
    """
    count = len(scene["objects"])
    allowed = {
        "billiards_two_object_v1": 2, "passive_pinball_two_object_v1": 2,
        "marble_run_two_object_v1": 2, "billiards_three_object_v1": 3,
        "passive_pinball_three_object_v1": 3, "marble_run_three_object_v1": 3,
    }
    if allowed.get(scene["backend_binding"]["adapter_id"]) != count:
        raise ValueError("specialized adapter and object count do not match")
    if count == 3 and (event_distance_tolerance_m is None or
            not np.isfinite(event_distance_tolerance_m) or event_distance_tolerance_m < 0):
        raise ValueError("three-sphere execution requires a finite contact tolerance")
    pb = pybullet_factory()
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
        solver_execution = configure_world(pb, scene)
        fixture_bodies, is_rail = fixture_builder(pb, scene, root)
        contact_execution: list[dict[str, Any]] = []
        bodies, runtime_material, runtime_inertia, runtime_extras = create_spheres(
            pb, scene, execution=contact_execution,
        )
        if len(bodies) != count:
            raise ValueError("sphere execution body count differs from declared objects")
        runtime_proxies = []
        runtime_support = []
        if count == 3:
            for body in bodies:
                shapes = pb.getCollisionShapeData(body, -1)
                if len(shapes) != 1:
                    raise ValueError('three-sphere execution requires one collider per body')
                shape = shapes[0]
                runtime_proxies.append({'shape_type': int(shape[2]), 'dimensions_m': list(shape[3]),
                    'local_position_m': list(shape[5]), 'local_quaternion_xyzw': list(shape[6])})
            for body, fixture_id in fixture_bodies.items():
                info = pb.getDynamicsInfo(body, -1)
                runtime_support.append({'fixture_id': fixture_id, 'mass_kg': float(info[0]),
                    'lateral_friction': float(info[1]), 'restitution': float(info[5])})
        positions = np.zeros((frame_count, count, 3), dtype=np.float64)
        orientations = np.zeros((frame_count, count, 4), dtype=np.float64)
        linear = np.zeros((frame_count, count, 3), dtype=np.float64)
        angular = np.zeros((frame_count, count, 3), dtype=np.float64)
        contact_count = np.zeros((frame_count, count), dtype=np.int32)
        path_lengths = np.zeros(count, dtype=np.float64)
        previous = np.asarray(
            [record["initial_state"]["position_m"] for record in scene["objects"]],
            dtype=np.float64,
        )
        touched = [set() for _ in bodies]
        first_pair_step: int | None = None
        first_rail_step: int | None = None
        minimum_contact_distance = 0.0
        maximum_speed = 0.0
        interval_contact_count = np.zeros(count, dtype=np.int32)
        interval_minimum_distance = np.zeros(count, dtype=np.float64)
        minimum_distances = np.zeros((frame_count, count), dtype=np.float64)

        def observe(frame: int) -> None:
            for index, body in enumerate(bodies):
                position, orientation = pb.getBasePositionAndOrientation(body)
                velocity, spin = pb.getBaseVelocity(body)
                positions[frame, index] = position
                orientations[frame, index] = orientation
                linear[frame, index] = velocity
                angular[frame, index] = spin
                contact_count[frame, index] = interval_contact_count[index]
                minimum_distances[frame, index] = interval_minimum_distance[index]

        collector = None
        if count == 3:
            collector = ContactEventCollector(
                [obj["object_id"] for obj in scene["objects"]], simulation_hz,
                float(event_distance_tolerance_m),
            )

        def observe_events(step: int) -> None:
            nonlocal minimum_contact_distance
            if collector is None:
                return
            contacts = {}
            for i in range(count):
                for j in range(i + 1, count):
                    points = pb.getContactPoints(bodyA=bodies[i], bodyB=bodies[j])
                    if points:
                        distance = min(float(point[8]) for point in points)
                        contacts[(scene["objects"][i]["object_id"], scene["objects"][j]["object_id"])] = distance
                        minimum_contact_distance = min(minimum_contact_distance, distance)
            collector.observe(step, contacts, {
                obj["object_id"]: list(pb.getBaseVelocity(body)[0])
                for obj, body in zip(scene["objects"], bodies)
            })

        pb.performCollisionDetection()
        interval_contact_count[:] = [len(pb.getContactPoints(bodyA=body)) for body in bodies]
        if count == 3:
            for index, body in enumerate(bodies):
                interval_minimum_distance[index] = min([0., *[float(c[8]) for c in pb.getContactPoints(bodyA=body)]])
        observe(0)
        observe_events(0)
        interval_contact_count.fill(0)
        interval_minimum_distance.fill(0)
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
                contacts = pb.getContactPoints(bodyA=body)
                if count == 3:
                    interval_minimum_distance[index] = min([interval_minimum_distance[index], *[float(c[8]) for c in contacts]])
                interval_contact_count[index] = max(interval_contact_count[index], len(contacts))
                for contact in contacts:
                    other = int(contact[2])
                    if other in fixture_bodies:
                        touched[index].add(fixture_bodies[other])
                        if first_rail_step is None and is_rail(other, contact):
                            first_rail_step = step
                    minimum_contact_distance = min(
                        minimum_contact_distance, float(contact[8])
                    )
            observe_events(step)
            if step % steps_per_frame == 0:
                observe(step // steps_per_frame)
                interval_contact_count.fill(0)
                interval_minimum_distance.fill(0)
    finally:
        pb.disconnect(client)

    arrays = {
        "time_s": np.arange(frame_count, dtype=np.float64) / float(output_fps),
        "position_m": positions,
        "quaternion_wxyz": orientations[:, :, [3, 0, 1, 2]],
        "linear_velocity_m_s": linear,
        "angular_velocity_rad_s": angular,
        "contact_count": contact_count,
        "runtime_material": runtime_material,
        "runtime_material_extras": runtime_extras,
        "inertia_diagonal_kg_m2": runtime_inertia,
        "adapter__quaternion_xyzw": orientations,
    }
    if count == 3:
        arrays['adapter__minimum_contact_distance_m'] = minimum_distances
    return arrays, {
        "solver_execution": solver_execution,
        "contact_processing_execution": contact_execution,
        "first_pair_step": first_pair_step, "first_rail_step": first_rail_step,
        "minimum_contact_distance": minimum_contact_distance,
        "maximum_speed": maximum_speed, "path_lengths": path_lengths,
        "touched": touched,
        "interaction_events": collector.evidence() if collector is not None else None,
        "runtime_proxies": runtime_proxies,
        "runtime_support": runtime_support,
    }
