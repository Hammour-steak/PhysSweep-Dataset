#!/usr/bin/env python3
"""Run deterministic geometry and PyBullet probes for the proxy catalog."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as pb

from physical_proxy_catalog import (
    load_catalog,
    read_jsonl,
    validate_record,
    write_json,
)
from static_support_proxy import (
    compile_static_support_binding,
    create_pybullet_static_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DT = 1.0 / 480.0
STATIC_SUPPORT_HZ = 1920
MINIMUM_SAFE_SURFACE_FLAT_FRACTION = 0.95
SUPPORT_PLANE_TOLERANCE_M = 0.004


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quaternion(euler_degrees: list[float]) -> tuple[float, float, float, float]:
    return pb.getQuaternionFromEuler(
        [math.radians(float(value)) for value in euler_degrees]
    )


def analytic_shape(colliders: list[dict[str, Any]]) -> int:
    types = []
    half_extents = []
    radii = []
    lengths = []
    positions = []
    orientations = []
    for collider in colliders:
        shape = str(collider["shape"])
        size = [float(value) for value in collider["size_m"]]
        types.append(
            {"box": pb.GEOM_BOX, "sphere": pb.GEOM_SPHERE, "cylinder": pb.GEOM_CYLINDER}[
                shape
            ]
        )
        half_extents.append([value * 0.5 for value in size])
        radii.append(size[0] * 0.5 if shape != "box" else 0.0)
        lengths.append(size[2] if shape == "cylinder" else 0.0)
        positions.append([float(value) for value in collider["position_m"]])
        orientations.append(quaternion(collider["rotation_euler_degrees"]))
    return pb.createCollisionShapeArray(
        shapeTypes=types,
        halfExtents=half_extents,
        radii=radii,
        lengths=lengths,
        collisionFramePositions=positions,
        collisionFrameOrientations=orientations,
    )


def reset_world(time_step: float = DT) -> int:
    pb.resetSimulation()
    pb.setGravity(0.0, 0.0, -9.81)
    pb.setTimeStep(float(time_step))
    pb.setPhysicsEngineParameter(
        fixedTimeStep=float(time_step),
        numSolverIterations=150,
        deterministicOverlappingPairs=1,
    )
    plane_shape = pb.createCollisionShape(pb.GEOM_PLANE)
    return pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=plane_shape)


def probe_analytic(record: dict[str, Any]) -> dict[str, Any]:
    plane = reset_world()
    colliders = record["proxy"]["colliders"]
    shape = analytic_shape(colliders)
    extent = max(
        abs(float(collider["position_m"][2])) + max(float(v) for v in collider["size_m"])
        for collider in colliders
    )
    body = pb.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=shape,
        basePosition=[0.0, 0.0, max(0.35, extent + 0.25)],
    )
    pb.changeDynamics(
        body,
        -1,
        lateralFriction=0.45,
        restitution=0.08,
        linearDamping=0.04,
        angularDamping=0.04,
    )
    contacted = False
    deepest = 0.0
    for _ in range(1440):
        pb.stepSimulation()
        contacts = pb.getContactPoints(bodyA=body, bodyB=plane)
        contacted = contacted or bool(contacts)
        deepest = min(deepest, min((float(item[8]) for item in contacts), default=0.0))
    position, _ = pb.getBasePositionAndOrientation(body)
    velocity, angular = pb.getBaseVelocity(body)
    final_contacts = pb.getContactPoints(bodyA=body, bodyB=plane)
    final_penetration = min(
        (float(item[8]) for item in final_contacts), default=0.0
    )
    passed = bool(
        contacted
        and final_penetration >= -0.006
        and float(position[2]) >= -0.05
        and np.linalg.norm(velocity) < 2.0
        and np.linalg.norm(angular) < 100.0
    )
    return {
        "probe": "analytic_drop",
        "passed": passed,
        "contacted": contacted,
        "deepest_transient_contact_m": deepest,
        "final_contact_distance_m": final_penetration,
        "final_position_m": [float(value) for value in position],
        "final_linear_speed_m_s": float(np.linalg.norm(velocity)),
        "final_angular_speed_rad_s": float(np.linalg.norm(angular)),
    }


def _world_with_support(
    root: Path, binding: dict[str, Any]
) -> tuple[int, int]:
    ground = reset_world(1.0 / STATIC_SUPPORT_HZ)
    support = create_pybullet_static_support(pb, root, binding)
    pb.changeDynamics(support, -1, lateralFriction=0.48, restitution=0.05)
    pb.changeDynamics(ground, -1, lateralFriction=0.55, restitution=0.02)
    return ground, support


def _surface_rays(
    body: int, usage: dict[str, Any]
) -> tuple[dict[str, Any], list[list[float]]]:
    safe = usage["safe_surface"]
    center = np.asarray(safe["center_xy_m"], dtype=np.float64)
    size = np.asarray(safe["size_xy_m"], dtype=np.float64)
    plane = float(usage["target_support_plane_z_m"])
    xs = np.linspace(center[0] - 0.46 * size[0], center[0] + 0.46 * size[0], 25)
    ys = np.linspace(center[1] - 0.46 * size[1], center[1] + 0.46 * size[1], 17)
    points = [(float(x), float(y)) for y in ys for x in xs]
    rays = pb.rayTestBatch(
        [[x, y, plane + 0.65] for x, y in points],
        [[x, y, max(-0.05, plane - 0.40)] for x, y in points],
        numThreads=0,
    )
    flat_points: list[list[float]] = []
    support_heights: list[float] = []
    low_count = 0
    high_count = 0
    for point, result in zip(points, rays):
        if int(result[0]) != body:
            low_count += 1
            continue
        height = float(result[3][2])
        support_heights.append(height)
        if abs(height - plane) <= SUPPORT_PLANE_TOLERANCE_M:
            flat_points.append([point[0], point[1], height])
        elif height < plane - SUPPORT_PLANE_TOLERANCE_M:
            low_count += 1
        else:
            high_count += 1
    flat_fraction = len(flat_points) / len(points)
    report = {
        "ray_count": len(points),
        "support_hit_count": len(support_heights),
        "flat_support_count": len(flat_points),
        "flat_support_fraction": flat_fraction,
        "support_plane_tolerance_m": SUPPORT_PLANE_TOLERANCE_M,
        "below_or_missing_fraction": low_count / len(points),
        "above_plane_fraction": high_count / len(points),
        "height_min_m": None if not support_heights else min(support_heights),
        "height_median_m": (
            None if not support_heights else float(np.median(support_heights))
        ),
        "height_max_m": None if not support_heights else max(support_heights),
        "minimum_flat_support_fraction": MINIMUM_SAFE_SURFACE_FLAT_FRACTION,
        "passed": bool(
            flat_fraction >= MINIMUM_SAFE_SURFACE_FLAT_FRACTION
        ),
    }
    return report, flat_points


def _nearest_flat_point(
    points: list[list[float]], target_xy: list[float]
) -> list[float]:
    target = np.asarray(target_xy, dtype=np.float64)
    return min(
        points,
        key=lambda point: float(
            np.linalg.norm(np.asarray(point[:2], dtype=np.float64) - target)
        ),
    )


def _probe_drop(
    root: Path,
    binding: dict[str, Any],
    point: list[float],
    shape_kind: str,
) -> dict[str, Any]:
    _, support = _world_with_support(root, binding)
    if shape_kind == "sphere":
        half_height = 0.035
        shape = pb.createCollisionShape(pb.GEOM_SPHERE, radius=half_height)
    elif shape_kind == "box":
        half_height = 0.026
        shape = pb.createCollisionShape(
            pb.GEOM_BOX, halfExtents=[0.036, 0.036, half_height]
        )
    else:
        raise ValueError(f"unsupported drop probe shape: {shape_kind}")
    start = [float(point[0]), float(point[1]), float(point[2]) + 0.34]
    body = pb.createMultiBody(
        baseMass=0.12, baseCollisionShapeIndex=shape, basePosition=start
    )
    pb.changeDynamics(
        body,
        -1,
        lateralFriction=0.35,
        restitution=0.08,
        linearDamping=0.04,
        angularDamping=0.04,
        ccdSweptSphereRadius=0.012,
        contactProcessingThreshold=0.0,
    )
    contacted = False
    deepest = 0.0
    for _ in range(int(2.5 * STATIC_SUPPORT_HZ)):
        pb.stepSimulation()
        contacts = pb.getContactPoints(bodyA=body, bodyB=support)
        contacted = contacted or bool(contacts)
        deepest = min(deepest, min((float(item[8]) for item in contacts), default=0.0))
    position, _ = pb.getBasePositionAndOrientation(body)
    velocity, _ = pb.getBaseVelocity(body)
    horizontal_drift = float(
        np.linalg.norm(np.asarray(position[:2]) - np.asarray(start[:2]))
    )
    expected_z = float(point[2]) + half_height
    physics_passed = bool(
        contacted
        and deepest >= -0.012
        and abs(float(position[2]) - expected_z) <= 0.025
        and np.linalg.norm(velocity) <= 0.25
    )
    motion_semantics_passed = bool(physics_passed and horizontal_drift <= 0.04)
    return {
        "passed": physics_passed,
        "physics_passed": physics_passed,
        "stable_vertical_drop": motion_semantics_passed,
        "shape": shape_kind,
        "contacted": contacted,
        "deepest_contact_m": deepest,
        "final_position_m": [float(value) for value in position],
        "expected_final_z_m": expected_z,
        "horizontal_drift_m": horizontal_drift,
        "final_speed_m_s": float(np.linalg.norm(velocity)),
    }


def _probe_short_slide(
    root: Path,
    binding: dict[str, Any],
    usage: dict[str, Any],
    flat_points: list[list[float]],
) -> dict[str, Any]:
    _, support = _world_with_support(root, binding)
    safe = usage["safe_surface"]
    center = [float(value) for value in safe["center_xy_m"]]
    size = [float(value) for value in safe["size_xy_m"]]
    axis = int(np.argmax(size))
    direction = np.zeros(2, dtype=np.float64)
    direction[axis] = 1.0
    start_target = np.asarray(center, dtype=np.float64) - direction * min(
        0.16, 0.18 * size[axis]
    )
    start_point = _nearest_flat_point(flat_points, start_target.tolist())
    half = [0.032, 0.032, 0.024]
    shape = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=half)
    start = [start_point[0], start_point[1], start_point[2] + half[2] + 0.002]
    body = pb.createMultiBody(
        baseMass=0.18, baseCollisionShapeIndex=shape, basePosition=start
    )
    pb.changeDynamics(
        body,
        -1,
        lateralFriction=0.18,
        restitution=0.03,
        linearDamping=0.02,
        angularDamping=0.05,
        ccdSweptSphereRadius=0.010,
        contactProcessingThreshold=0.0,
    )
    velocity = [float(direction[0] * 0.65), float(direction[1] * 0.65), 0.0]
    pb.resetBaseVelocity(body, linearVelocity=velocity)
    minimum_z = float("inf")
    contacted = False
    for _ in range(STATIC_SUPPORT_HZ):
        pb.stepSimulation()
        position, _ = pb.getBasePositionAndOrientation(body)
        minimum_z = min(minimum_z, float(position[2]))
        contacted = contacted or bool(pb.getContactPoints(bodyA=body, bodyB=support))
    final, _ = pb.getBasePositionAndOrientation(body)
    displacement = np.asarray(final, dtype=np.float64) - np.asarray(start)
    forward = float(np.dot(displacement[:2], direction))
    lateral = float(abs(displacement[1 - axis]))
    plane = float(usage["target_support_plane_z_m"])
    passed = bool(
        contacted
        and forward >= 0.04
        and lateral <= 0.12
        and minimum_z >= plane - 0.10
        and np.isfinite(displacement).all()
    )
    return {
        "passed": passed,
        "axis": "x" if axis == 0 else "y",
        "start_position_m": start,
        "final_position_m": [float(value) for value in final],
        "forward_displacement_m": forward,
        "lateral_displacement_m": lateral,
        "minimum_center_z_m": minimum_z,
        "contacted": contacted,
    }


def _probe_boundary(
    root: Path,
    binding: dict[str, Any],
    usage: dict[str, Any],
    flat_points: list[list[float]],
) -> dict[str, Any]:
    _, support = _world_with_support(root, binding)
    target = binding["target_support_frame"]
    center = np.asarray(target["center_xy_m"], dtype=np.float64)
    size = np.asarray(target["size_xy_m"], dtype=np.float64)
    directions = usage.get("clear_exit_directions_xy", [])
    if directions:
        direction = np.asarray(directions[0], dtype=np.float64)
        direction /= np.linalg.norm(direction)
    else:
        direction = np.zeros(2, dtype=np.float64)
        direction[int(np.argmax(size))] = 1.0
    point = max(
        flat_points,
        key=lambda value: float(np.dot(np.asarray(value[:2]) - center, direction)),
    )
    radius = 0.03
    shape = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius)
    start = [float(point[0]), float(point[1]), float(point[2]) + radius + 0.002]
    body = pb.createMultiBody(
        baseMass=0.1, baseCollisionShapeIndex=shape, basePosition=start
    )
    pb.changeDynamics(
        body,
        -1,
        lateralFriction=0.12,
        restitution=0.05,
        rollingFriction=0.001,
        linearDamping=0.01,
        angularDamping=0.01,
        ccdSweptSphereRadius=0.010,
        contactProcessingThreshold=0.0,
    )
    pb.resetBaseVelocity(
        body,
        linearVelocity=[float(direction[0] * 1.35), float(direction[1] * 1.35), 0.0],
    )
    positions = []
    support_contacts = 0
    for _ in range(2 * STATIC_SUPPORT_HZ):
        pb.stepSimulation()
        position, _ = pb.getBasePositionAndOrientation(body)
        positions.append(position)
        support_contacts += int(bool(pb.getContactPoints(bodyA=body, bodyB=support)))
    values = np.asarray(positions, dtype=np.float64)
    relative = values[:, :2] - center
    outside = bool(
        np.any(np.any(np.abs(relative) > size * 0.5 + radius, axis=1))
    )
    below = bool(
        np.any(values[:, 2] < float(target["plane_z_m"]) - 0.08)
    )
    behavior = str(usage["boundary_behavior"])
    if behavior == "open":
        passed = bool(outside and below)
    else:
        passed = bool(not below and support_contacts > 0)
    return {
        "passed": passed,
        "expected_behavior": behavior,
        "direction_xy": direction.tolist(),
        "left_visual_footprint": outside,
        "descended_below_support": below,
        "support_contact_steps": support_contacts,
        "minimum_center_z_m": float(values[:, 2].min()),
        "final_position_m": values[-1].tolist(),
    }


def probe_static_usage(
    root: Path, record: dict[str, Any], usage: dict[str, Any]
) -> dict[str, Any]:
    binding = compile_static_support_binding(
        record,
        target_size_xy_m=usage["target_size_xy_m"],
        target_center_xy_m=usage["target_center_xy_m"],
        target_support_plane_z_m=float(usage["target_support_plane_z_m"]),
        usage_id=str(usage["id"]),
        maximum_axis_scale_ratio=float(usage["maximum_axis_scale_ratio"]),
    )
    _, body = _world_with_support(root, binding)
    rays, flat_points = _surface_rays(body, usage)
    aabb = pb.getAABB(body)
    finite_aabb = bool(np.isfinite(np.asarray(aabb, dtype=np.float64)).all())
    geometry_passed = bool(
        rays["passed"]
        and finite_aabb
        and min(aabb[0]) > -100.0
        and max(aabb[1]) < 100.0
    )
    if not geometry_passed:
        return {
            "probe": "static_support_scene_contract",
            "usage_id": str(usage["id"]),
            "passed": False,
            "binding": binding,
            "surface_rays": rays,
            "scaled_aabb": [list(aabb[0]), list(aabb[1])],
            "failure": "support_surface_geometry_gate",
        }
    safe_center = [float(value) for value in usage["safe_surface"]["center_xy_m"]]
    center_point = _nearest_flat_point(flat_points, safe_center)
    nonround_drop = _probe_drop(root, binding, center_point, "box")
    round_drop = _probe_drop(root, binding, center_point, "sphere")
    slide = _probe_short_slide(root, binding, usage, flat_points)
    boundary = _probe_boundary(root, binding, usage, flat_points)
    passed = bool(
        nonround_drop["passed"] and slide["passed"] and boundary["passed"]
    )
    admitted_dynamic_classes = ["nonround_rigid"]
    if round_drop["physics_passed"]:
        admitted_dynamic_classes.append("round_rigid")
    return {
        "probe": "static_support_scene_contract",
        "usage_id": str(usage["id"]),
        "passed": passed,
        "binding": binding,
        "surface_rays": rays,
        "drop_nonround": nonround_drop,
        "drop_round": round_drop,
        "short_slide": slide,
        "boundary": boundary,
        "admitted_dynamic_classes": admitted_dynamic_classes,
        "motion_compatibility": {
            "stable_vertical_drop_nonround": bool(
                nonround_drop["stable_vertical_drop"]
            ),
            "stable_vertical_drop_round": bool(
                round_drop["stable_vertical_drop"]
            ),
            "surface_slide_nonround": bool(slide["passed"]),
            "boundary_behavior": str(usage["boundary_behavior"]),
        },
        "scaled_aabb": [list(aabb[0]), list(aabb[1])],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--catalog", type=Path, default=Path("assets/proxies/catalog.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("assets/proxies/objects/validation.json")
    )
    parser.add_argument(
        "--records",
        type=Path,
        help="probe unpublished candidate records without mutating the active catalog",
    )
    parser.add_argument(
        "--all-proxy-ready",
        action="store_true",
        help="re-probe every inherited proxy instead of active records only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    catalog_path = args.catalog if args.catalog.is_absolute() else root / args.catalog
    if args.records is None:
        manifest, records = load_catalog(root, catalog_path)
        records_sha256 = str(manifest["records_sha256"])
    else:
        records_path = args.records if args.records.is_absolute() else root / args.records
        records = read_jsonl(records_path)
        for record in records:
            validate_record(record, root)
        manifest = None
        records_sha256 = sha256(records_path)
    selected = [
        record
        for record in records
        if record["proxy"]["representation"] == "static_concave_mesh"
        or (
            record["proxy"]["representation"] == "analytic_compound"
            and record["admission"][
                "proxy_ready" if args.all_proxy_ready else "active_matrix_selected"
            ]
        )
    ]
    connection = pb.connect(pb.DIRECT)
    if connection < 0:
        raise RuntimeError("failed to connect to PyBullet DIRECT")
    try:
        results = []
        for record in selected:
            representation = record["proxy"]["representation"]
            if representation == "analytic_compound":
                result = probe_analytic(record)
                results.append(
                    {
                        "asset_id": record["asset_id"],
                        "representation": representation,
                        **result,
                    }
                )
                continue
            for usage in record["proxy"]["usages"]:
                result = probe_static_usage(root, record, usage)
                results.append(
                    {
                        "asset_id": record["asset_id"],
                        "representation": representation,
                        **result,
                    }
                )
    finally:
        pb.disconnect()
    report = {
        "version": "physweep_physical_proxy_validation_v2",
        "implementation": {
            "path": str(Path(__file__).resolve().relative_to(root)).replace(
                "\\", "/"
            ),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "catalog_path": str(catalog_path.relative_to(root)).replace("\\", "/"),
        "catalog_records_sha256": records_sha256,
        "scope": "all_proxy_ready" if args.all_proxy_ready else "active_and_static_mesh",
        "counts": {
            "tested": len(results),
            "passed": sum(bool(item["passed"]) for item in results),
            "failed": sum(not bool(item["passed"]) for item in results),
        },
        "records": results,
    }
    output = args.output if args.output.is_absolute() else root / args.output
    write_json(output, report)
    if manifest is not None:
        updated_manifest = copy.deepcopy(manifest)
        updated_manifest["validation"] = {
            "path": str(output.relative_to(root)).replace("\\", "/"),
            "sha256": sha256(output),
            "counts": copy.deepcopy(report["counts"]),
            "scope": report["scope"],
        }
        write_json(catalog_path, updated_manifest)
    print(json.dumps(report["counts"], indent=2, ensure_ascii=True))
    if report["counts"]["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
