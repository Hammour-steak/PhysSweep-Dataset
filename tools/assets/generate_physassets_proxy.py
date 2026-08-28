#!/usr/bin/env python3
"""Generate and validate a compound convex PyBullet proxy for one GLB asset."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import coacd
import numpy as np
import pybullet as bullet
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        meshes = [loaded]
    else:
        meshes = [
            geometry
            for geometry in loaded.dump()
            if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0
        ]
    if not meshes:
        raise ValueError("source contains no triangle meshes")
    mesh = trimesh.util.concatenate(meshes)
    mesh.remove_unreferenced_vertices()
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("source contains non-finite vertices")
    return mesh


def normalize_mesh(
    mesh: trimesh.Trimesh,
    target_extent_m: float,
) -> tuple[trimesh.Trimesh, dict]:
    # glTF/GLB uses Y-up. PhysSweep and PyBullet use Z-up.
    up_axis_transform = trimesh.transformations.rotation_matrix(
        math.pi / 2.0,
        [1.0, 0.0, 0.0],
    )
    rotated = mesh.copy()
    rotated.apply_transform(up_axis_transform)
    source_bounds = np.asarray(rotated.bounds, dtype=np.float64)
    source_extents = source_bounds[1] - source_bounds[0]
    longest = float(source_extents.max())
    if not math.isfinite(longest) or longest <= 1e-9:
        raise ValueError("source has invalid extents")
    source_origin = np.array(
        [
            (source_bounds[0, 0] + source_bounds[1, 0]) * 0.5,
            (source_bounds[0, 1] + source_bounds[1, 1]) * 0.5,
            (source_bounds[0, 2] + source_bounds[1, 2]) * 0.5,
        ],
        dtype=np.float64,
    )
    scale = target_extent_m / longest
    normalized = rotated.copy()
    normalized.vertices = (normalized.vertices - source_origin) * scale
    normalized.remove_unreferenced_vertices()
    return normalized, {
        "proxy_frame": "compound_center_of_mass_z_up",
        "source_up_axis": "y",
        "target_up_axis": "z",
        "up_axis_transform": up_axis_transform.tolist(),
        "rotated_source_origin": source_origin.tolist(),
        "uniform_scale": scale,
        "source_extents": source_extents.tolist(),
        "normalized_extents_m": normalized.extents.tolist(),
        "target_longest_extent_m": target_extent_m,
    }


def recenter_on_compound_com(
    mesh: trimesh.Trimesh,
    parts: list[trimesh.Trimesh],
) -> tuple[trimesh.Trimesh, list[trimesh.Trimesh], np.ndarray]:
    volumes = np.asarray([abs(part.volume) for part in parts], dtype=np.float64)
    centers = np.asarray([part.center_mass for part in parts], dtype=np.float64)
    valid = np.isfinite(volumes) & (volumes > 1e-12)
    if valid.any() and np.isfinite(centers[valid]).all():
        compound_com = np.average(centers[valid], axis=0, weights=volumes[valid])
    else:
        compound_com = np.asarray(mesh.bounding_box.centroid, dtype=np.float64)
    centered_mesh = mesh.copy()
    centered_mesh.apply_translation(-compound_com)
    centered_parts = []
    for part in parts:
        centered = part.copy()
        centered.apply_translation(-compound_com)
        centered_parts.append(centered)
    return centered_mesh, centered_parts, compound_com


def decompose(
    mesh: trimesh.Trimesh,
    threshold: float,
    max_hulls: int,
    max_vertices_per_hull: int,
) -> tuple[list[trimesh.Trimesh], float]:
    coacd.set_log_level("error")
    started = time.monotonic()
    parts_raw = coacd.run_coacd(
        coacd.Mesh(
            np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int32),
        ),
        threshold=threshold,
        max_convex_hull=max_hulls,
        preprocess_mode="auto",
        preprocess_resolution=40,
        resolution=1200,
        mcts_nodes=16,
        mcts_iterations=80,
        mcts_max_depth=3,
        merge=True,
        decimate=True,
        max_ch_vertex=max_vertices_per_hull,
        seed=0,
    )
    elapsed = time.monotonic() - started
    parts = [
        trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        for vertices, faces in parts_raw
        if len(vertices) >= 4 and len(faces) >= 4
    ]
    if not parts:
        raise ValueError("convex decomposition produced no valid parts")
    return parts, elapsed


def export_parts(parts: list[trimesh.Trimesh], output_dir: Path) -> list[Path]:
    collision_dir = output_dir / "collision"
    if collision_dir.exists():
        shutil.rmtree(collision_dir)
    collision_dir.mkdir(parents=True)
    paths = []
    for index, part in enumerate(parts):
        path = collision_dir / f"part_{index:03d}.obj"
        part.export(path, file_type="obj")
        paths.append(path)
    return paths


def make_collision_shape(part_paths: list[Path]) -> int:
    count = len(part_paths)
    return bullet.createCollisionShapeArray(
        shapeTypes=[bullet.GEOM_MESH] * count,
        fileNames=[str(path) for path in part_paths],
        meshScales=[[1.0, 1.0, 1.0]] * count,
        collisionFramePositions=[[0.0, 0.0, 0.0]] * count,
        collisionFrameOrientations=[[0.0, 0.0, 0.0, 1.0]] * count,
    )


def finite_vector(values: tuple[float, ...] | list[float]) -> bool:
    return bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())


def run_drop_test(
    collision_shape: int,
    orientation: tuple[float, float, float, float],
    steps: int = 720,
) -> dict:
    body = bullet.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=collision_shape,
        basePosition=[0.0, 0.0, 0.8],
        baseOrientation=orientation,
    )
    contacted = False
    minimum_z = math.inf
    maximum_z = -math.inf
    for _ in range(steps):
        bullet.stepSimulation()
        position, _ = bullet.getBasePositionAndOrientation(body)
        minimum_z = min(minimum_z, float(position[2]))
        maximum_z = max(maximum_z, float(position[2]))
        contacted = contacted or bool(bullet.getContactPoints(bodyA=body))
    position, rotation = bullet.getBasePositionAndOrientation(body)
    linear_velocity, angular_velocity = bullet.getBaseVelocity(body)
    aabb_min, aabb_max = bullet.getAABB(body)
    passed = bool(
        finite_vector(position)
        and finite_vector(rotation)
        and finite_vector(linear_velocity)
        and finite_vector(angular_velocity)
        and contacted
        and aabb_min[2] >= -0.005
        and aabb_max[2] <= 2.0
        and np.linalg.norm(linear_velocity) <= 0.08
        and np.linalg.norm(angular_velocity) <= 0.35
    )
    result = {
        "passed": passed,
        "contacted": contacted,
        "final_position": list(position),
        "final_orientation": list(rotation),
        "final_linear_speed": float(np.linalg.norm(linear_velocity)),
        "final_angular_speed": float(np.linalg.norm(angular_velocity)),
        "final_aabb_min": list(aabb_min),
        "final_aabb_max": list(aabb_max),
        "center_z_range": [minimum_z, maximum_z],
    }
    bullet.removeBody(body)
    return result


def run_slide_test(collision_shape: int) -> dict:
    body = bullet.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=collision_shape,
        basePosition=[0.0, 0.0, 0.5],
    )
    bullet.changeDynamics(body, -1, lateralFriction=0.35, restitution=0.1)
    for _ in range(600):
        bullet.stepSimulation()
    start, _ = bullet.getBasePositionAndOrientation(body)
    bullet.resetBaseVelocity(body, linearVelocity=[1.0, 0.0, 0.0])
    minimum_aabb_z = math.inf
    for _ in range(480):
        bullet.stepSimulation()
        aabb_min, _ = bullet.getAABB(body)
        minimum_aabb_z = min(minimum_aabb_z, float(aabb_min[2]))
    end, _ = bullet.getBasePositionAndOrientation(body)
    linear_velocity, angular_velocity = bullet.getBaseVelocity(body)
    displacement = float(np.linalg.norm(np.asarray(end[:2]) - np.asarray(start[:2])))
    passed = bool(
        finite_vector(end)
        and finite_vector(linear_velocity)
        and finite_vector(angular_velocity)
        and minimum_aabb_z >= -0.005
        and 0.01 <= displacement <= 3.0
        and abs(float(end[2]) - float(start[2])) <= 0.05
    )
    result = {
        "passed": passed,
        "horizontal_displacement_m": displacement,
        "minimum_aabb_z": minimum_aabb_z,
        "final_linear_speed": float(np.linalg.norm(linear_velocity)),
        "final_angular_speed": float(np.linalg.norm(angular_velocity)),
    }
    bullet.removeBody(body)
    return result


def validate_proxy(part_paths: list[Path]) -> dict:
    client = bullet.connect(bullet.DIRECT)
    try:
        bullet.resetSimulation()
        bullet.setGravity(0.0, 0.0, -9.81)
        bullet.setTimeStep(1.0 / 240.0)
        plane_shape = bullet.createCollisionShape(
            bullet.GEOM_BOX,
            halfExtents=[2.0, 2.0, 0.025],
        )
        bullet.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=plane_shape,
            basePosition=[0.0, 0.0, -0.025],
        )
        collision_shape = make_collision_shape(part_paths)
        if collision_shape < 0:
            raise RuntimeError("PyBullet failed to create compound collision shape")
        orientations = [
            (0.0, 0.0, 0.0, 1.0),
            tuple(bullet.getQuaternionFromEuler([0.71, 0.37, 0.19])),
            tuple(bullet.getQuaternionFromEuler([1.23, 0.83, 0.51])),
        ]
        drops = [run_drop_test(collision_shape, orientation) for orientation in orientations]
        slide = run_slide_test(collision_shape)
        return {
            "passed": all(test["passed"] for test in drops) and slide["passed"],
            "drop_tests": drops,
            "slide_test": slide,
        }
    finally:
        bullet.disconnect(client)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--object-name", default="")
    parser.add_argument("--material", default="")
    parser.add_argument("--target-extent-m", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--max-hulls", type=int, default=16)
    parser.add_argument("--max-vertices-per-hull", type=int, default=64)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    mesh = load_mesh(args.source)
    normalized, transform = normalize_mesh(mesh, args.target_extent_m)
    parts, decomposition_seconds = decompose(
        normalized,
        args.threshold,
        args.max_hulls,
        args.max_vertices_per_hull,
    )
    normalized, parts, compound_com = recenter_on_compound_com(normalized, parts)
    transform["compound_com_before_recentering_m"] = compound_com.tolist()
    transform["canonical_bounds_m"] = normalized.bounds.tolist()
    transform["placement_bottom_offset_m"] = float(-normalized.bounds[0, 2])
    part_paths = export_parts(parts, args.output)
    validation = validate_proxy(part_paths)
    part_metrics = [
        {
            "path": str(path.relative_to(args.output)),
            "vertices": int(len(part.vertices)),
            "faces": int(len(part.faces)),
            "volume_m3": float(abs(part.volume)),
        }
        for path, part in zip(part_paths, parts)
    ]
    record = {
        "schema_version": "physweep_generated_collision_proxy_v1",
        "sample_id": str(args.sample_id),
        "objaverse_uid": str(args.uid),
        "object_name": args.object_name,
        "material": args.material,
        "source_glb": str(args.source),
        "method": "coacd_compound_convex",
        "parameters": {
            "threshold": args.threshold,
            "max_hulls": args.max_hulls,
            "max_vertices_per_hull": args.max_vertices_per_hull,
        },
        "transform": transform,
        "source_mesh": {
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
        },
        "proxy": {
            "part_count": len(parts),
            "parts": part_metrics,
        },
        "validation": validation,
        "timing": {
            "decomposition_seconds": decomposition_seconds,
            "total_seconds": time.monotonic() - started,
        },
        "admission": "passed" if validation["passed"] else "failed_validation",
    }
    (args.output / "proxy.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
