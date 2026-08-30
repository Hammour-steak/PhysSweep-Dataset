#!/usr/bin/env python3
"""Run one immutable PhysSweep rigid metadata scene in PyBullet DIRECT mode."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json
from tools.dataset_contract.object_identity_contract import (
    require_simulation_objects,
)
from tools.physics.physics_invariants import (
    maximum_coulomb_utilization,
    runtime_collision_descriptors,
)
from tools.core.rigid_geometry import (
    declared_collision_descriptors,
    quaternion_xyzw_from_wxyz,
)
from tools.physics.rigid_trajectory import (
    audit_trajectory,
    compact_advisory_ids,
    compact_failure_ids,
    validate_trajectory_contract,
)
from tools.assets.static_support_proxy import create_pybullet_static_support
from tools.assets.environment_collision import (
    create_pybullet_environment_bodies,
    validate_environment_binding,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_DYNAMIC_OBJECT_COUNTS = (1, 2)


def create_box_body(pb: Any, collider: dict[str, Any]) -> int:
    size = [float(value) for value in collider["size_m"]]
    orientation = pb.getQuaternionFromEuler(
        [
            float(value) * np.pi / 180.0
            for value in collider["rotation_euler_degrees"]
        ]
    )
    shape = pb.createCollisionShape(
        pb.GEOM_BOX, halfExtents=[value / 2.0 for value in size]
    )
    return int(
        pb.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=shape,
            basePosition=[float(value) for value in collider["position_m"]],
            baseOrientation=orientation,
        )
    )


def create_dynamic_body(pb: Any, record: dict[str, Any]) -> int:
    geometry = record["geometry"]
    shape_type = str(geometry["type"])
    size = [float(value) for value in geometry["size_m"]]
    collision_profile = record["collision_profile"]
    if collision_profile.get("type") == "compound":
        declared = declared_collision_descriptors(record)
        dimensions = np.asarray(declared["dimensions_m"], dtype=np.float64)
        shape_codes = np.asarray(declared["shape_codes"], dtype=np.int32)
        shape = pb.createCollisionShapeArray(
            shapeTypes=[
                {
                    1: pb.GEOM_BOX,
                    2: pb.GEOM_SPHERE,
                    3: pb.GEOM_CYLINDER,
                }[int(code)]
                for code in shape_codes
            ],
            halfExtents=(dimensions * 0.5).tolist(),
            radii=[
                float(dimensions[index, 0] * 0.5)
                if int(code) in {2, 3}
                else 0.0
                for index, code in enumerate(shape_codes)
            ],
            lengths=[
                float(dimensions[index, 2]) if int(code) == 3 else 0.0
                for index, code in enumerate(shape_codes)
            ],
            collisionFramePositions=declared["positions_m"],
            collisionFrameOrientations=declared["quaternions_xyzw"],
        )
    elif shape_type == "cuboid":
        shape = pb.createCollisionShape(
            pb.GEOM_BOX, halfExtents=[value / 2.0 for value in size]
        )
    elif shape_type == "sphere":
        shape = pb.createCollisionShape(pb.GEOM_SPHERE, radius=size[0] / 2.0)
    elif shape_type == "cylinder":
        shape = pb.createCollisionShape(
            pb.GEOM_CYLINDER, radius=max(size[0], size[1]) / 2.0, height=size[2]
        )
    else:
        raise ValueError(f"unsupported dynamic primitive: {shape_type}")
    initial = record["initial_state"]
    material = record["material"]
    body = int(
        pb.createMultiBody(
            baseMass=float(material["mass_kg"]),
            baseCollisionShapeIndex=shape,
            basePosition=[float(value) for value in initial["position_m"]],
            baseOrientation=quaternion_xyzw_from_wxyz(
                initial["orientation_quaternion_wxyz"]
            ),
        )
    )
    pb.changeDynamics(
        body,
        -1,
        lateralFriction=float(material["contact_friction"]),
        restitution=float(material["contact_restitution"]),
        linearDamping=float(material["linear_damping"]),
        angularDamping=float(material["angular_damping"]),
        rollingFriction=float(material["rolling_friction"]),
        spinningFriction=float(material["spinning_friction"]),
        ccdSweptSphereRadius=min(size) * 0.22,
        contactProcessingThreshold=0.0,
    )
    pb.resetBaseVelocity(
        body,
        linearVelocity=[float(value) for value in initial["linear_velocity_m_s"]],
        angularVelocity=[float(value) for value in initial["angular_velocity_rad_s"]],
    )
    return body


def capture_frame(
    pb: Any,
    body: int,
    contact_interval: dict[str, Any],
) -> dict[str, Any]:
    position, quaternion_xyzw = pb.getBasePositionAndOrientation(body)
    linear_velocity, angular_velocity = pb.getBaseVelocity(body)
    aabb_min, aabb_max = pb.getAABB(body)
    return {
        "position_m": position,
        "quaternion_wxyz": [
            float(quaternion_xyzw[3]),
            float(quaternion_xyzw[0]),
            float(quaternion_xyzw[1]),
            float(quaternion_xyzw[2]),
        ],
        "linear_velocity_m_s": linear_velocity,
        "angular_velocity_rad_s": angular_velocity,
        "aabb_min_m": aabb_min,
        "aabb_max_m": aabb_max,
        "primary_support_contact_count": int(contact_interval["primary_support_contact_count"]),
        "all_contact_count": int(contact_interval["all_contact_count"]),
        "minimum_contact_distance_m": float(contact_interval["minimum_contact_distance_m"]),
        "total_normal_force_n": float(contact_interval["total_normal_force_n"]),
        "maximum_coulomb_friction_utilization": float(
            contact_interval["maximum_coulomb_friction_utilization"]
        ),
        "collider_contact_counts": dict(contact_interval["collider_contact_counts"]),
        "object_contact_counts": dict(contact_interval["object_contact_counts"]),
    }


def contact_interval_record(
    pb: Any,
    body: int,
    static_bodies: dict[str, int],
    dynamic_friction: float,
    static_friction_by_body: dict[int, float],
    other_dynamic_bodies: dict[str, int],
    dynamic_friction_by_body: dict[int, float],
) -> dict[str, Any]:
    contacts = list(pb.getContactPoints(bodyA=body))
    collider_ids = {body_id: collider_id for collider_id, body_id in static_bodies.items()}
    collider_contact_counts = {collider_id: 0 for collider_id in static_bodies}
    object_ids = {body_id: object_id for object_id, body_id in other_dynamic_bodies.items()}
    object_contact_counts = {object_id: 0 for object_id in other_dynamic_bodies}
    for record in contacts:
        other_body = int(record[2])
        collider_id = collider_ids.get(other_body)
        if collider_id is not None:
            collider_contact_counts[collider_id] += 1
        other_object_id = object_ids.get(other_body)
        if other_object_id is not None:
            object_contact_counts[other_object_id] += 1
    distances = [float(record[8]) for record in contacts]
    forces = [float(record[9]) for record in contacts]
    return {
        "primary_support_contact_count": collider_contact_counts["support"],
        "all_contact_count": len(contacts),
        "minimum_contact_distance_m": min(distances) if distances else 0.0,
        "total_normal_force_n": sum(forces),
        "maximum_coulomb_friction_utilization": maximum_coulomb_utilization(
            contacts,
            dynamic_friction,
            {**static_friction_by_body, **dynamic_friction_by_body},
        ),
        "collider_contact_counts": collider_contact_counts,
        "object_contact_counts": object_contact_counts,
    }


def empty_contact_interval(
    static_bodies: dict[str, int], other_dynamic_bodies: dict[str, int]
) -> dict[str, Any]:
    return {
        "primary_support_contact_count": 0,
        "all_contact_count": 0,
        "minimum_contact_distance_m": 0.0,
        "total_normal_force_n": 0.0,
        "maximum_coulomb_friction_utilization": 0.0,
        "collider_contact_counts": {
            collider_id: 0 for collider_id in static_bodies
        },
        "object_contact_counts": {
            object_id: 0 for object_id in other_dynamic_bodies
        },
    }


def merge_contact_intervals(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["primary_support_contact_count"] = max(
        int(target["primary_support_contact_count"]),
        int(source["primary_support_contact_count"]),
    )
    target["all_contact_count"] = max(
        int(target["all_contact_count"]), int(source["all_contact_count"])
    )
    target["minimum_contact_distance_m"] = min(
        float(target["minimum_contact_distance_m"]),
        float(source["minimum_contact_distance_m"]),
    )
    target["total_normal_force_n"] = max(
        float(target["total_normal_force_n"]), float(source["total_normal_force_n"])
    )
    source_utilization = float(source["maximum_coulomb_friction_utilization"])
    target_utilization = float(target["maximum_coulomb_friction_utilization"])
    if np.isfinite(source_utilization):
        target["maximum_coulomb_friction_utilization"] = (
            source_utilization
            if not np.isfinite(target_utilization)
            else max(target_utilization, source_utilization)
        )
    for collider_id, count in source["collider_contact_counts"].items():
        target["collider_contact_counts"][collider_id] = max(
            int(target["collider_contact_counts"][collider_id]), int(count)
        )
    for object_id, count in source["object_contact_counts"].items():
        target["object_contact_counts"][object_id] = max(
            int(target["object_contact_counts"][object_id]), int(count)
        )


def simulate(metadata: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import pybullet as pb  # pylint: disable=import-outside-toplevel

    if metadata["schema_version"] != "physweep_pybullet_rigid_metadata_v1":
        raise ValueError("unsupported rigid metadata schema")
    if metadata["simulation"]["backend"]["id"] != "pybullet_rigid_v1":
        raise ValueError("metadata does not declare the PyBullet rigid backend")
    objects = require_simulation_objects(
        metadata, SUPPORTED_DYNAMIC_OBJECT_COUNTS, __name__
    )
    if any(obj["body_model"] != "rigid_body" for obj in objects):
        raise ValueError("PyBullet rigid v1 requires rigid-body objects")

    simulation = metadata["simulation"]
    time_config = simulation["time"]
    nominal_simulation_hz = int(time_config["simulation_hz"])
    output_fps = int(time_config["output_fps"])
    if nominal_simulation_hz % output_fps != 0:
        raise ValueError("simulation_hz must be divisible by output_fps")
    exact_static_binding = simulation["support"].get("exact_static_binding")
    contact_modes = {
        str(obj.get("expected_motion", {}).get("contact_mode", ""))
        for obj in objects
    }
    # Concave static meshes need a finer step only during ballistic impact.
    integration_substep_factor = (
        2
        if exact_static_binding is not None and "ballistic_then_contact" in contact_modes
        else 1
    )
    simulation_hz = nominal_simulation_hz * integration_substep_factor
    steps_per_frame = simulation_hz // output_fps
    frame_count = int(time_config["frame_count"])
    solver = simulation["solver"]
    client = pb.connect(pb.DIRECT)
    if client < 0:
        raise RuntimeError("PyBullet DIRECT connection failed")
    try:
        pb.resetSimulation()
        gravity = [float(value) for value in simulation["world"]["gravity_m_s2"]]
        pb.setGravity(*gravity)
        pb.setTimeStep(1.0 / simulation_hz)
        pb.setPhysicsEngineParameter(
            numSolverIterations=int(solver["iterations"]),
            deterministicOverlappingPairs=int(bool(solver["deterministic_overlapping_pairs"])),
            restitutionVelocityThreshold=float(solver["restitution_velocity_threshold_m_s"]),
            contactBreakingThreshold=float(solver["contact_breaking_threshold_m"]),
            enableConeFriction=1,
            useSplitImpulse=1,
            solverResidualThreshold=0.0,
        )
        support_dynamics = simulation["support"]["dynamics"]
        static_bodies: dict[str, int] = {}
        static_friction_by_body: dict[int, float] = {}
        for collider in simulation["support"]["colliders"]:
            if not bool(collider.get("collision_enabled", True)):
                continue
            body = create_box_body(pb, collider)
            pb.changeDynamics(
                body,
                -1,
                lateralFriction=float(support_dynamics["lateral_friction"]),
                restitution=float(support_dynamics["restitution"]),
            )
            static_bodies[str(collider["id"])] = body
            static_friction_by_body[body] = float(
                support_dynamics["lateral_friction"]
            )
        exact_binding = simulation["support"].get("exact_static_binding")
        if exact_binding is not None:
            if "support" in static_bodies:
                raise ValueError(
                    "exact support binding conflicts with an analytic support body"
                )
            body = create_pybullet_static_support(pb, PROJECT_ROOT, exact_binding)
            pb.changeDynamics(
                body,
                -1,
                lateralFriction=float(support_dynamics["lateral_friction"]),
                restitution=float(support_dynamics["restitution"]),
            )
            static_bodies["support"] = body
            static_friction_by_body[body] = float(
                support_dynamics["lateral_friction"]
            )
        if "support" not in static_bodies:
            raise ValueError("simulation has no authoritative primary support body")
        environment_binding = validate_environment_binding(metadata)
        environment_bodies = create_pybullet_environment_bodies(
            pb, PROJECT_ROOT, environment_binding
        )
        duplicate_ids = set(static_bodies) & set(environment_bodies)
        if duplicate_ids:
            raise ValueError(
                f"environment collider ids conflict with support: {sorted(duplicate_ids)}"
            )
        environment_dynamics = environment_binding["dynamics"]
        support_friction = float(support_dynamics["lateral_friction"])
        support_restitution = float(support_dynamics["restitution"])
        if (
            str(environment_dynamics["policy"]) != "inherit_primary_support"
            or float(environment_dynamics["lateral_friction"]) != support_friction
            or float(environment_dynamics["restitution"]) != support_restitution
        ):
            raise ValueError("environment dynamics diverge from the primary support")
        for collider_id, environment_body in environment_bodies.items():
            pb.changeDynamics(
                environment_body,
                -1,
                lateralFriction=support_friction,
                restitution=support_restitution,
            )
            static_bodies[collider_id] = environment_body
            static_friction_by_body[environment_body] = support_friction
        object_by_id = {str(obj["object_id"]): obj for obj in objects}
        if len(object_by_id) != len(objects):
            raise ValueError("dynamic object ids must be unique")
        dynamic_bodies = {
            object_id: create_dynamic_body(pb, obj)
            for object_id, obj in object_by_id.items()
        }
        other_dynamic_bodies_by_object = {
            object_id: {
                other_id: other_body
                for other_id, other_body in dynamic_bodies.items()
                if other_id != object_id
            }
            for object_id in dynamic_bodies
        }
        dynamic_friction_by_body = {
            dynamic_bodies[object_id]: float(obj["material"]["contact_friction"])
            for object_id, obj in object_by_id.items()
        }
        runtime_by_object: dict[str, dict[str, Any]] = {}
        for object_id, body in dynamic_bodies.items():
            dynamics_info = pb.getDynamicsInfo(body, -1)
            runtime_by_object[object_id] = {
                "proxy": runtime_collision_descriptors(pb, body),
                "dynamics": np.asarray(
                    [
                        dynamics_info[0],
                        dynamics_info[1],
                        dynamics_info[5],
                        dynamics_info[6],
                        dynamics_info[7],
                    ],
                    dtype=np.float64,
                ),
                "inertia": np.asarray(dynamics_info[2], dtype=np.float64),
            }
        runtime_support_dynamics = np.asarray(
            [
                [
                    pb.getDynamicsInfo(static_body, -1)[1],
                    pb.getDynamicsInfo(static_body, -1)[5],
                ]
                for static_body in static_bodies.values()
            ],
            dtype=np.float64,
        )
        records_by_object: dict[str, list[dict[str, Any]]] = {
            object_id: [] for object_id in object_by_id
        }
        pb.performCollisionDetection()
        initial_environment_contacts = [
            contact
            for body in dynamic_bodies.values()
            for environment_body in environment_bodies.values()
            for contact in pb.getContactPoints(bodyA=body, bodyB=environment_body)
        ]
        dynamic_body_items = list(dynamic_bodies.items())
        initial_object_contacts = [
            contact
            for index, (_, body_a) in enumerate(dynamic_body_items)
            for _, body_b in dynamic_body_items[index + 1 :]
            for contact in pb.getContactPoints(bodyA=body_a, bodyB=body_b)
        ]
        maximum_initial_penetration = float(
            metadata["qa"]["limits"]["maximum_initial_penetration_m"]
        )
        if any(
            float(contact[8]) < -maximum_initial_penetration
            for contact in initial_environment_contacts + initial_object_contacts
        ):
            raise ValueError("dynamic objects initially penetrate another collision body")
        for object_id, body in dynamic_bodies.items():
            other_bodies = other_dynamic_bodies_by_object[object_id]
            initial_interval = empty_contact_interval(static_bodies, other_bodies)
            merge_contact_intervals(
                initial_interval,
                contact_interval_record(
                    pb,
                    body,
                    static_bodies,
                    float(object_by_id[object_id]["material"]["contact_friction"]),
                    static_friction_by_body,
                    other_bodies,
                    dynamic_friction_by_body,
                ),
            )
            records_by_object[object_id].append(
                capture_frame(pb, body, initial_interval)
            )
        for _frame in range(1, frame_count):
            intervals = {
                object_id: empty_contact_interval(
                    static_bodies, other_dynamic_bodies_by_object[object_id]
                )
                for object_id in dynamic_bodies
            }
            for _ in range(steps_per_frame):
                pb.stepSimulation()
                for object_id, body in dynamic_bodies.items():
                    merge_contact_intervals(
                        intervals[object_id],
                        contact_interval_record(
                            pb,
                            body,
                            static_bodies,
                            float(
                                object_by_id[object_id]["material"]["contact_friction"]
                            ),
                            static_friction_by_body,
                            other_dynamic_bodies_by_object[object_id],
                            dynamic_friction_by_body,
                        ),
                    )
            for object_id, body in dynamic_bodies.items():
                records_by_object[object_id].append(
                    capture_frame(pb, body, intervals[object_id])
                )
    finally:
        pb.disconnect(client)

    trajectory: dict[str, np.ndarray] = {
        "time_s": np.arange(frame_count, dtype=np.float64) / float(output_fps),
    }
    vector_keys = (
        "position_m",
        "quaternion_wxyz",
        "linear_velocity_m_s",
        "angular_velocity_rad_s",
        "aabb_min_m",
        "aabb_max_m",
    )
    scalar_float_keys = (
        "minimum_contact_distance_m",
        "total_normal_force_n",
        "maximum_coulomb_friction_utilization",
    )
    scalar_int_keys = ("primary_support_contact_count", "all_contact_count")
    for object_id in object_by_id:
        runtime = runtime_by_object[object_id]
        runtime_proxy = runtime["proxy"]
        trajectory[f"{object_id}__runtime_dynamics"] = runtime["dynamics"]
        trajectory[f"{object_id}__runtime_inertia_diagonal_kg_m2"] = runtime[
            "inertia"
        ]
        trajectory[f"{object_id}__runtime_support_dynamics"] = runtime_support_dynamics
        trajectory[f"{object_id}__runtime_proxy_shape_codes"] = runtime_proxy[
            "shape_codes"
        ]
        trajectory[f"{object_id}__runtime_proxy_dimensions_m"] = runtime_proxy[
            "dimensions_m"
        ]
        trajectory[f"{object_id}__runtime_proxy_positions_m"] = runtime_proxy[
            "positions_m"
        ]
        trajectory[f"{object_id}__runtime_proxy_quaternions_xyzw"] = runtime_proxy[
            "quaternions_xyzw"
        ]
        records = records_by_object[object_id]
        for key in vector_keys:
            trajectory[f"{object_id}__{key}"] = np.asarray(
                [record[key] for record in records], dtype=np.float64
            )
        for key in scalar_float_keys:
            trajectory[f"{object_id}__{key}"] = np.asarray(
                [record[key] for record in records], dtype=np.float64
            )
        for key in scalar_int_keys:
            trajectory[f"{object_id}__{key}"] = np.asarray(
                [record[key] for record in records], dtype=np.int32
            )
        for collider_id in static_bodies:
            trajectory[
                f"{object_id}__collider_contact_count__{collider_id}"
            ] = np.asarray(
                [record["collider_contact_counts"][collider_id] for record in records],
                dtype=np.int32,
            )
        for other_object_id in object_by_id:
            if other_object_id == object_id:
                continue
            trajectory[
                f"{object_id}__object_contact_count__{other_object_id}"
            ] = np.asarray(
                [
                    record["object_contact_counts"][other_object_id]
                    for record in records
                ],
                dtype=np.int32,
            )
    validate_trajectory_contract(metadata, trajectory)
    audit = audit_trajectory(metadata, trajectory)
    return trajectory, audit


def run(metadata_path: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    metadata = load_json(metadata_path)
    trajectory, audit = simulate(metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectory.npz"
    np.savez_compressed(trajectory_path, **trajectory)
    audit_path = output_dir / "trajectory_audit.json"
    write_json(audit_path, audit)
    record = {
        "schema_version": "physweep_pybullet_simulation_record_v1",
        "scene_id": metadata["scene_id"],
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": sha256(trajectory_path),
        "audit_path": str(audit_path),
        "audit_sha256": sha256(audit_path),
        "audit_passed": bool(audit["passed"]),
        "failed_checks": compact_failure_ids(audit),
        "advisory_checks": compact_advisory_ids(audit),
        "wall_time_s": round(time.perf_counter() - started, 6),
    }
    write_json(output_dir / "simulation_record.json", record)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = run(args.metadata.resolve(), args.output_dir.resolve())
    print(json.dumps(record, indent=2, ensure_ascii=True))
    if not record["audit_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
