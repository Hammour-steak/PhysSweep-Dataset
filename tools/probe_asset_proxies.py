#!/usr/bin/env python3
"""Run deterministic PyBullet probes for every admitted asset proxy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pybullet as pb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 240.0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def quaternion(euler_degrees: list[float]) -> tuple[float, float, float, float]:
    return pb.getQuaternionFromEuler([math.radians(float(v)) for v in euler_degrees])


def compound_shape(colliders: list[dict[str, Any]]) -> int:
    types = []
    half_extents = []
    radii = []
    lengths = []
    positions = []
    orientations = []
    for collider in colliders:
        shape = str(collider["shape"])
        size = [float(v) for v in collider["size_m"]]
        types.append({"box": pb.GEOM_BOX, "sphere": pb.GEOM_SPHERE, "cylinder": pb.GEOM_CYLINDER}[shape])
        half_extents.append([value * 0.5 for value in size])
        radii.append(size[0] * 0.5 if shape != "box" else 0.0)
        lengths.append(size[2] if shape == "cylinder" else 0.0)
        positions.append([float(v) for v in collider["position_m"]])
        orientations.append(quaternion(collider["rotation_euler_degrees"]))
    return pb.createCollisionShapeArray(
        shapeTypes=types,
        halfExtents=half_extents,
        radii=radii,
        lengths=lengths,
        collisionFramePositions=positions,
        collisionFrameOrientations=orientations,
    )


def contact_summary(body: int, other: int) -> tuple[int, float]:
    contacts = pb.getContactPoints(bodyA=body, bodyB=other)
    penetration = min((float(item[8]) for item in contacts), default=0.0)
    return len(contacts), penetration


def simulate(body: int, other: int, steps: int = 720) -> dict[str, Any]:
    samples = []
    first_contact = None
    max_contacts = 0
    deepest = 0.0
    for step in range(steps):
        pb.stepSimulation()
        count, penetration = contact_summary(body, other)
        max_contacts = max(max_contacts, count)
        deepest = min(deepest, penetration)
        position, orientation = pb.getBasePositionAndOrientation(body)
        if count and first_contact is None:
            first_contact = {
                "step": step,
                "position_m": [round(float(v), 7) for v in position],
                "quaternion_xyzw": [round(float(v), 7) for v in orientation],
            }
        if step in {0, steps - 1}:
            samples.append(
                {
                    "step": step,
                    "position_m": [round(float(v), 7) for v in position],
                    "quaternion_xyzw": [round(float(v), 7) for v in orientation],
                }
            )
    position, orientation = pb.getBasePositionAndOrientation(body)
    velocity, angular_velocity = pb.getBaseVelocity(body)
    final_contacts, final_penetration = contact_summary(body, other)
    final_aabb = pb.getAABB(body)
    return {
        "samples": [samples[0], first_contact or samples[0], samples[-1]],
        "first_contact_step": None if first_contact is None else first_contact["step"],
        "max_contact_count": max_contacts,
        "deepest_contact_distance_m": round(deepest, 7),
        "final_contact_count": final_contacts,
        "final_contact_distance_m": round(final_penetration, 7),
        "final_aabb_min_m": [round(float(v), 7) for v in final_aabb[0]],
        "final_aabb_max_m": [round(float(v), 7) for v in final_aabb[1]],
        "final_position_m": [round(float(v), 7) for v in position],
        "final_quaternion_xyzw": [round(float(v), 7) for v in orientation],
        "final_linear_speed_m_s": round(math.sqrt(sum(float(v) ** 2 for v in velocity)), 7),
        "final_angular_speed_rad_s": round(
            math.sqrt(sum(float(v) ** 2 for v in angular_velocity)), 7
        ),
    }


def reset_world() -> int:
    pb.resetSimulation()
    pb.setGravity(0.0, 0.0, -9.81)
    pb.setTimeStep(DT)
    pb.setPhysicsEngineParameter(numSolverIterations=80, fixedTimeStep=DT)
    plane = pb.createCollisionShape(pb.GEOM_PLANE)
    return pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=plane)


def proxy_top(colliders: list[dict[str, Any]]) -> float:
    return max(
        float(item["position_m"][2]) + 0.5 * max(float(v) for v in item["size_m"])
        for item in colliders
    )


def probe_dynamic(record: dict[str, Any]) -> dict[str, Any]:
    plane = reset_world()
    colliders = record["proxy"]["colliders"]
    shape = compound_shape(colliders)
    start_z = max(0.35, proxy_top(colliders) + 0.35)
    body = pb.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=shape, basePosition=[0, 0, start_z])
    material = record["proxy"].get("material", {})
    pb.changeDynamics(
        body,
        -1,
        lateralFriction=float(material.get("friction", 0.5)),
        restitution=float(material.get("restitution", 0.2)),
        linearDamping=0.04,
        angularDamping=0.04,
    )
    result = simulate(body, plane)
    result["probe"] = "dynamic_drop"
    result["passed"] = (
        result["max_contact_count"] > 0
        and result["final_aabb_min_m"][2] >= -0.006
        and result["final_contact_distance_m"] >= -0.006
    )
    return result


def probe_static(record: dict[str, Any]) -> dict[str, Any]:
    reset_world()
    colliders = record["proxy"]["colliders"]
    proxy_body = pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=compound_shape(colliders))
    target = max(colliders, key=lambda item: float(item["position_m"][2]) + float(item["size_m"][2]))
    radius = 0.035
    sphere = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius)
    start = [
        float(target["position_m"][0]),
        float(target["position_m"][1]),
        float(target["position_m"][2]) + 0.5 * float(target["size_m"][2]) + 0.22,
    ]
    body = pb.createMultiBody(baseMass=0.08, baseCollisionShapeIndex=sphere, basePosition=start)
    result = simulate(body, proxy_body, steps=480)
    result["probe"] = "static_prop_drop"
    result["probe_radius_m"] = radius
    result["passed"] = (
        result["max_contact_count"] > 0
        and result["final_aabb_min_m"][2] >= -0.006
    )
    return result


def probe_support(record: dict[str, Any]) -> dict[str, Any]:
    reset_world()
    proxy_body = pb.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=compound_shape(record["proxy"]["colliders"]),
    )
    surface = record["proxy"]["usable_surfaces"][0]
    radius = 0.045
    sphere = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius)
    center = [float(v) for v in surface["center_xy_m"]]
    start = [center[0], center[1], float(surface["z_m"]) + 0.28]
    body = pb.createMultiBody(baseMass=0.12, baseCollisionShapeIndex=sphere, basePosition=start)
    pb.changeDynamics(body, -1, lateralFriction=0.45, restitution=0.18)
    result = simulate(body, proxy_body, steps=480)
    expected_center_z = float(surface["z_m"]) + radius
    result["probe"] = "support_center_drop"
    result["probe_radius_m"] = radius
    result["expected_final_center_z_m"] = round(expected_center_z, 7)
    result["passed"] = (
        result["max_contact_count"] > 0
        and result["final_contact_distance_m"] >= -0.006
        and abs(result["final_position_m"][2] - expected_center_z) <= 0.018
    )
    return result


def probe_record(record: dict[str, Any]) -> dict[str, Any]:
    kind = str(record["proxy"]["kind"])
    if kind == "dynamic_rigid":
        result = probe_dynamic(record)
    elif kind == "static_compound":
        result = probe_static(record)
    elif kind == "support_compound":
        result = probe_support(record)
    else:
        return {"asset_id": record["asset_id"], "proxy_kind": kind, "probe": "skipped", "passed": True}
    return {"asset_id": record["asset_id"], "proxy_kind": kind, **result}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_ROOT / "configs/asset_proxy_registry.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-disabled", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_json(args.registry)
    connection = pb.connect(pb.DIRECT)
    if connection < 0:
        raise RuntimeError("failed to start PyBullet DIRECT")
    try:
        records = [
            item
            for item in registry["records"]
            if args.include_disabled or bool(item["admission"].get("sampling_enabled", False))
        ]
        results = [probe_record(item) for item in records]
    finally:
        pb.disconnect()
    physical_results = [item for item in results if item["probe"] != "skipped"]
    report = {
        "version": "physweep_asset_proxy_probe_report_v1",
        "registry_version": registry["version"],
        "counts": {
            "records": len(results),
            "tested": len(physical_results),
            "skipped": len(results) - len(physical_results),
            "passed": sum(bool(item["passed"]) for item in physical_results),
            "failed": sum(not bool(item["passed"]) for item in physical_results),
        },
        "records": results,
    }
    write_json(args.output, report)
    print(json.dumps(report["counts"], indent=2, ensure_ascii=True))
    if report["counts"]["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
