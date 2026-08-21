#!/usr/bin/env python3
"""Fit and validate a stable primitive PyBullet proxy for one PhysAssets GLB."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pybullet as pb
import trimesh


SPHERE_WORDS = {"ball", "basketball", "football", "soccer", "tennis", "sphere"}
BOX_WORDS = {
    "box", "book", "crate", "die", "block", "carton", "kit", "notebook",
    "magazine", "wallet", "pack", "brick", "case", "remote", "phone",
}
CYLINDER_WORDS = {
    "bottle", "barrel", "can", "cup", "bowl", "pot",
    "jar", "cylinder", "glass", "goblet", "roll",
    "vase", "trash", "spray",
    "vial",
}
COMPLEX_CONTAINER_WORDS = {"mug", "bucket", "pitcher", "jug", "teapot"}
ROD_WORDS = {
    "pencil", "pen", "stick", "rod", "screw", "screwdriver", "flashlight",
    "log", "bat",
}


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    geometries = [loaded] if isinstance(loaded, trimesh.Trimesh) else list(loaded.dump())
    meshes = [g for g in geometries if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    if not meshes:
        raise ValueError("source contains no triangle meshes")
    mesh = trimesh.util.concatenate(meshes)
    mesh.remove_unreferenced_vertices()
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("source contains non-finite vertices")
    return mesh


def canonicalize(mesh: trimesh.Trimesh, target_extent_m: float) -> tuple[trimesh.Trimesh, dict]:
    rotation = trimesh.transformations.rotation_matrix(math.pi / 2.0, [1.0, 0.0, 0.0])
    result = mesh.copy()
    result.apply_transform(rotation)
    bounds = np.asarray(result.bounds, dtype=np.float64)
    center = bounds.mean(axis=0)
    extents = bounds[1] - bounds[0]
    longest = float(extents.max())
    if not math.isfinite(longest) or longest <= 1e-9:
        raise ValueError("source has invalid extents")
    scale = target_extent_m / longest
    result.vertices = (result.vertices - center) * scale
    return result, {
        "source_up_axis": "y",
        "target_up_axis": "z",
        "up_axis_transform": rotation.tolist(),
        "rotated_source_center": center.tolist(),
        "uniform_scale": scale,
        "target_longest_extent_m": target_extent_m,
    }


def tokens(name: str) -> set[str]:
    return set(name.lower().replace("-", " ").replace("_", " ").split())


def collider(shape: str, size: list[float], position: list[float], rotation=None, cid="body") -> dict:
    return {
        "id": cid,
        "shape": shape,
        "size_m": [float(v) for v in size],
        "position_m": [float(v) for v in position],
        "rotation_euler_degrees": [0.0, 0.0, 0.0] if rotation is None else rotation,
    }


def fit_box(mesh: trimesh.Trimesh) -> list[dict]:
    bounds = np.asarray(mesh.bounds)
    return [collider("box", (bounds[1] - bounds[0]).tolist(), bounds.mean(axis=0).tolist())]


def fit_sphere(mesh: trimesh.Trimesh) -> list[dict]:
    center = np.asarray(mesh.bounds).mean(axis=0)
    radius = float(np.linalg.norm(np.asarray(mesh.vertices) - center, axis=1).max())
    diameter = 2.0 * radius
    return [collider("sphere", [diameter] * 3, center.tolist())]


def fit_axis_cylinder(mesh: trimesh.Trimesh, axis: int) -> list[dict]:
    vertices = np.asarray(mesh.vertices)
    lo = float(vertices[:, axis].min())
    hi = float(vertices[:, axis].max())
    center = np.asarray(mesh.bounds).mean(axis=0)
    radial_axes = [i for i in range(3) if i != axis]
    radius = float(np.linalg.norm(vertices[:, radial_axes] - center[radial_axes], axis=1).max())
    rotation = {0: [0.0, 90.0, 0.0], 1: [90.0, 0.0, 0.0], 2: [0.0, 0.0, 0.0]}[axis]
    return [collider("cylinder", [2 * radius, 2 * radius, hi - lo], center.tolist(), rotation)]


def fit_profile_cylinders(mesh: trimesh.Trimesh, bands: int = 3) -> list[dict]:
    vertices = np.asarray(mesh.vertices)
    z_edges = np.linspace(vertices[:, 2].min(), vertices[:, 2].max(), bands + 1)
    result = []
    for index in range(bands):
        upper = vertices[:, 2] <= z_edges[index + 1] if index == bands - 1 else vertices[:, 2] < z_edges[index + 1]
        selected = vertices[(vertices[:, 2] >= z_edges[index]) & upper]
        if len(selected) < 4:
            continue
        xy_center = (selected[:, :2].min(axis=0) + selected[:, :2].max(axis=0)) * 0.5
        radius = float(np.linalg.norm(selected[:, :2] - xy_center, axis=1).max())
        height = float(z_edges[index + 1] - z_edges[index])
        result.append(collider(
            "cylinder", [2 * radius, 2 * radius, height],
            [float(xy_center[0]), float(xy_center[1]), float((z_edges[index] + z_edges[index + 1]) * 0.5)],
            cid=f"profile_{index}",
        ))
    return result


def primitive_volume(item: dict) -> float:
    size = item["size_m"]
    if item["shape"] == "box":
        return float(np.prod(size))
    if item["shape"] == "sphere":
        return 4.0 * math.pi * (size[0] * 0.5) ** 3 / 3.0
    return math.pi * (size[0] * 0.5) ** 2 * size[2]


def choose_proxy(mesh: trimesh.Trimesh, name: str) -> tuple[str, list[dict]]:
    words = tokens(name)
    extents = np.asarray(mesh.extents)
    axis = int(np.argmax(extents))
    if words & SPHERE_WORDS and float(extents.max() / extents.min()) <= 1.65:
        return "semantic_sphere", fit_sphere(mesh)
    if words & BOX_WORDS:
        return "semantic_box", fit_box(mesh)
    if words & COMPLEX_CONTAINER_WORDS:
        return "review_complex_container", fit_axis_cylinder(mesh, 2)
    if words & CYLINDER_WORDS:
        return "semantic_upright_cylinder", fit_axis_cylinder(mesh, 2)
    if words & ROD_WORDS:
        return "semantic_axis_cylinder", fit_axis_cylinder(mesh, axis)

    candidates = {
        "geometric_box": fit_box(mesh),
        "geometric_sphere": fit_sphere(mesh),
        "geometric_axis_cylinder": fit_axis_cylinder(mesh, axis),
    }
    method = min(candidates, key=lambda key: sum(primitive_volume(x) for x in candidates[key]))
    return method, candidates[method]


def recenter(mesh: trimesh.Trimesh, colliders: list[dict]) -> tuple[trimesh.Trimesh, np.ndarray]:
    volumes = np.asarray([primitive_volume(c) for c in colliders])
    positions = np.asarray([c["position_m"] for c in colliders])
    center = np.average(positions, axis=0, weights=volumes)
    result = mesh.copy()
    result.apply_translation(-center)
    for item in colliders:
        item["position_m"] = (np.asarray(item["position_m"]) - center).tolist()
    return result, center


def make_shape(colliders: list[dict]) -> int:
    if len(colliders) == 1:
        item = colliders[0]
        size = item["size_m"]
        common = {
            "collisionFramePosition": item["position_m"],
            "collisionFrameOrientation": pb.getQuaternionFromEuler(
                [math.radians(v) for v in item["rotation_euler_degrees"]]
            ),
        }
        if item["shape"] == "box":
            return pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[v * 0.5 for v in size], **common)
        if item["shape"] == "sphere":
            return pb.createCollisionShape(pb.GEOM_SPHERE, radius=size[0] * 0.5, **common)
        return pb.createCollisionShape(pb.GEOM_CYLINDER, radius=size[0] * 0.5, height=size[2], **common)
    shape_types, half_extents, radii, lengths, positions, orientations = [], [], [], [], [], []
    for item in colliders:
        shape_types.append({"box": pb.GEOM_BOX, "sphere": pb.GEOM_SPHERE, "cylinder": pb.GEOM_CYLINDER}[item["shape"]])
        size = item["size_m"]
        half_extents.append([v * 0.5 for v in size])
        radii.append(size[0] * 0.5 if item["shape"] != "box" else 0.0)
        lengths.append(size[2] if item["shape"] == "cylinder" else 0.0)
        positions.append(item["position_m"])
        orientations.append(pb.getQuaternionFromEuler([math.radians(v) for v in item["rotation_euler_degrees"]]))
    return pb.createCollisionShapeArray(
        shapeTypes=shape_types, halfExtents=half_extents, radii=radii, lengths=lengths,
        collisionFramePositions=positions, collisionFrameOrientations=orientations,
    )


def run_probe(shape: int, orientation: tuple[float, float, float, float], angular_limit: float) -> dict:
    body = pb.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=shape, basePosition=[0, 0, 0.35], baseOrientation=orientation)
    pb.changeDynamics(body, -1, lateralFriction=0.4, restitution=0.08, linearDamping=0.04, angularDamping=0.04)
    contacted = False
    deepest = 0.0
    for _ in range(1440):
        pb.stepSimulation()
        contacts = pb.getContactPoints(bodyA=body)
        contacted = contacted or bool(contacts)
        deepest = min(deepest, min((float(c[8]) for c in contacts), default=0.0))
    position, _ = pb.getBasePositionAndOrientation(body)
    velocity, angular = pb.getBaseVelocity(body)
    aabb = pb.getAABB(body)
    final_contacts = pb.getContactPoints(bodyA=body)
    final_deepest = min((float(c[8]) for c in final_contacts), default=0.0)
    # Broadphase AABBs are conservative for rotating cylinders; contact distance
    # is the authoritative penetration measurement.
    passed = bool(contacted and final_deepest >= -0.006 and np.linalg.norm(velocity) < 2.0 and np.linalg.norm(angular) < angular_limit)
    pb.removeBody(body)
    return {"passed": passed, "contacted": contacted, "deepest_transient_contact_m": deepest, "final_contact_m": final_deepest, "final_aabb_min_z_m": aabb[0][2], "final_position_m": list(position), "final_linear_speed_m_s": float(np.linalg.norm(velocity)), "final_angular_speed_rad_s": float(np.linalg.norm(angular)), "angular_speed_limit_rad_s": angular_limit}


def run_slide_probe(shape: int, angular_limit: float) -> dict:
    body = pb.createMultiBody(baseMass=1.0, baseCollisionShapeIndex=shape, basePosition=[0, 0, 0.35])
    pb.changeDynamics(body, -1, lateralFriction=0.35, restitution=0.05, linearDamping=0.04, angularDamping=0.04)
    for _ in range(720):
        pb.stepSimulation()
    start, _ = pb.getBasePositionAndOrientation(body)
    pb.resetBaseVelocity(body, linearVelocity=[1.0, 0.0, 0.0])
    contacted = False
    deepest = 0.0
    for _ in range(480):
        pb.stepSimulation()
        contacts = pb.getContactPoints(bodyA=body)
        contacted = contacted or bool(contacts)
        deepest = min(deepest, min((float(c[8]) for c in contacts), default=0.0))
    end, _ = pb.getBasePositionAndOrientation(body)
    velocity, angular = pb.getBaseVelocity(body)
    final_contacts = pb.getContactPoints(bodyA=body)
    final_deepest = min((float(c[8]) for c in final_contacts), default=0.0)
    displacement = float(np.linalg.norm(np.asarray(end[:2]) - np.asarray(start[:2])))
    passed = bool(contacted and final_deepest >= -0.006 and 0.01 <= displacement <= 3.0 and np.linalg.norm(velocity) < 2.0 and np.linalg.norm(angular) < angular_limit)
    pb.removeBody(body)
    return {"passed": passed, "contacted": contacted, "deepest_transient_contact_m": deepest, "final_contact_m": final_deepest, "horizontal_displacement_m": displacement, "final_linear_speed_m_s": float(np.linalg.norm(velocity)), "final_angular_speed_rad_s": float(np.linalg.norm(angular)), "angular_speed_limit_rad_s": angular_limit}


def validate(colliders: list[dict]) -> dict:
    pb.connect(pb.DIRECT)
    try:
        pb.resetSimulation()
        pb.setGravity(0, 0, -9.81)
        pb.setTimeStep(1.0 / 240.0)
        pb.setPhysicsEngineParameter(numSolverIterations=80)
        plane = pb.createCollisionShape(pb.GEOM_PLANE)
        pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=plane)
        shape = make_shape(colliders)
        minimum_radius = max(0.001, min(min(c["size_m"]) * 0.5 for c in colliders))
        angular_limit = max(25.0, 4.0 / minimum_radius)
        orientations = [(0, 0, 0, 1), pb.getQuaternionFromEuler([0.71, 0.37, 0.19]), pb.getQuaternionFromEuler([1.23, 0.83, 0.51])]
        probes = [run_probe(shape, q, angular_limit) for q in orientations]
        slide = run_slide_probe(shape, angular_limit)
        return {"passed": all(p["passed"] for p in probes) and slide["passed"], "drop_tests": probes, "slide_test": slide}
    finally:
        pb.disconnect()


def fit_quality(mesh: trimesh.Trimesh, colliders: list[dict], method: str) -> dict:
    proxy_volume = sum(primitive_volume(c) for c in colliders)
    try:
        hull_volume = float(abs(mesh.convex_hull.volume))
    except Exception:
        hull_volume = 0.0
    ratio = proxy_volume / hull_volume if hull_volume > 1e-12 else math.inf
    semantic_fit = method.startswith("semantic_")
    ratio_limits = {
        "semantic_sphere": 1.35,
        "semantic_box": 1.8,
        "semantic_upright_cylinder": 1.8,
        "semantic_axis_cylinder": 1.8,
    }
    ratio_limit = ratio_limits.get(method, 1.8)
    return {
        "proxy_volume_m3": proxy_volume,
        "visual_convex_hull_volume_m3": hull_volume,
        "proxy_to_visual_hull_volume_ratio": ratio,
        "review_ratio_limit": ratio_limit,
        "semantic_template_matched": semantic_fit,
        "requires_visual_review": (
            not semantic_fit or not math.isfinite(ratio) or ratio > ratio_limit
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--object-name", default="")
    parser.add_argument("--material", default="")
    parser.add_argument("--target-extent-m", type=float, default=0.2)
    args = parser.parse_args()

    started = time.monotonic()
    source = load_mesh(args.source)
    mesh, transform = canonicalize(source, args.target_extent_m)
    method, colliders = choose_proxy(mesh, args.object_name)
    mesh, center = recenter(mesh, colliders)
    transform.update({
        "proxy_frame": "compound_volume_center_z_up",
        "compound_center_before_recentering_m": center.tolist(),
        "canonical_visual_bounds_m": mesh.bounds.tolist(),
        "placement_bottom_offset_m": float(-mesh.bounds[0, 2]),
    })
    quality = fit_quality(mesh, colliders, method)
    validation = validate(colliders)
    admission = "passed" if validation["passed"] and not quality["requires_visual_review"] else "needs_review"
    record = {
        "schema_version": "physweep_generated_primitive_proxy_v1",
        "asset_id": f"physassets_{args.sample_id}",
        "sample_id": str(args.sample_id),
        "objaverse_uid": args.uid,
        "name": args.object_name,
        "source_glb": str(args.source),
        "source_material_label": args.material,
        "transform": transform,
        "proxy": {"kind": "dynamic_rigid", "body_type": "dynamic", "method": method, "colliders": colliders},
        "fit_quality": quality,
        "validation": validation,
        "admission": admission,
        "timing_seconds": time.monotonic() - started,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "proxy.json").write_text(json.dumps(record, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=True))


if __name__ == "__main__":
    main()
