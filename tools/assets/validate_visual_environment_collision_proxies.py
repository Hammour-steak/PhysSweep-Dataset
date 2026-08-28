#!/usr/bin/env python3
"""Validate static visual-environment meshes in PyBullet DIRECT mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION = "physweep_visual_environment_collision_validation_v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_record(pb: Any, root: Path, record: dict[str, Any]) -> dict[str, Any]:
    pb.resetSimulation()
    proxy = record["proxy"]
    path = root / str(proxy["path"])
    started = time.perf_counter()
    shape = pb.createCollisionShape(
        pb.GEOM_MESH,
        fileName=str(path.resolve()),
        flags=pb.GEOM_FORCE_CONCAVE_TRIMESH,
    )
    body = int(pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=shape))
    pb.performCollisionDetection()
    load_time = time.perf_counter() - started
    low, high = pb.getAABB(body)
    extents = [float(high[index] - low[index]) for index in range(3)]
    extent_error = max(
        abs(extents[index] - float(proxy["extents_m"][index]))
        for index in range(3)
    )
    margin_x = min(0.06, 0.04 * float(high[0] - low[0]))
    margin_y = min(0.06, 0.04 * float(high[1] - low[1]))
    xs = [
        low[0] + margin_x + (high[0] - low[0] - 2.0 * margin_x) * index / 12.0
        for index in range(13)
    ]
    ys = [
        low[1] + margin_y + (high[1] - low[1] - 2.0 * margin_y) * index / 12.0
        for index in range(13)
    ]
    starts = [(x, y, high[2] + 0.5) for x in xs for y in ys]
    ends = [(x, y, low[2] - 0.5) for x in xs for y in ys]
    hits = pb.rayTestBatch(starts, ends)
    mesh_hits = [hit for hit in hits if int(hit[0]) == body]
    upward_hits = [hit for hit in mesh_hits if float(hit[4][2]) >= 0.45]
    contact_observed = False
    maximum_penetration_m = 0.0
    final_position = None
    if upward_hits:
        center_xy = [(low[0] + high[0]) / 2.0, (low[1] + high[1]) / 2.0]
        selected = min(
            upward_hits,
            key=lambda hit: (
                (float(hit[3][0]) - center_xy[0]) ** 2
                + (float(hit[3][1]) - center_xy[1]) ** 2,
                -float(hit[4][2]),
            ),
        )
        hit_position = [float(value) for value in selected[3]]
        radius = 0.04
        sphere_shape = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius)
        sphere = int(
            pb.createMultiBody(
                baseMass=0.05,
                baseCollisionShapeIndex=sphere_shape,
                basePosition=[hit_position[0], hit_position[1], hit_position[2] + 0.28],
            )
        )
        pb.changeDynamics(
            sphere,
            -1,
            lateralFriction=0.5,
            restitution=0.0,
            ccdSweptSphereRadius=radius * 0.5,
            contactProcessingThreshold=0.0,
        )
        pb.setGravity(0.0, 0.0, -9.81)
        pb.setTimeStep(1.0 / 1800.0)
        pb.setPhysicsEngineParameter(
            numSolverIterations=100,
            deterministicOverlappingPairs=1,
            contactBreakingThreshold=0.001,
        )
        for _ in range(1800):
            pb.stepSimulation()
            contacts = pb.getContactPoints(bodyA=sphere, bodyB=body)
            if contacts:
                contact_observed = True
                maximum_penetration_m = max(
                    maximum_penetration_m,
                    max(0.0, max(-float(contact[8]) for contact in contacts)),
                )
        final_position = [
            round(float(value), 9)
            for value in pb.getBasePositionAndOrientation(sphere)[0]
        ]
        pb.removeBody(sphere)
    checks = {
        "source_hash": sha256(path) == str(proxy["sha256"]),
        "pybullet_shape_created": shape >= 0 and body >= 0,
        "extent_match": extent_error <= 1.0e-5,
        "ray_coverage": len(mesh_hits) > 0,
        "upward_contact_surface": len(upward_hits) > 0,
        "dynamic_contact_response": contact_observed,
        "bounded_contact_penetration": maximum_penetration_m <= 0.012,
        "load_time": load_time <= 1.0,
    }
    pb.removeBody(body)
    return {
        "profile_id": str(record["profile_id"]),
        "asset_id": str(record["asset_id"]),
        "passed": all(checks.values()),
        "checks": checks,
        "load_time_s": round(load_time, 6),
        "extent_maximum_absolute_error_m": round(extent_error, 12),
        "ray_hits": len(mesh_hits),
        "upward_ray_hits": len(upward_hits),
        "contact_observed": contact_observed,
        "maximum_contact_penetration_m": round(maximum_penetration_m, 9),
        "drop_probe_final_position_m": final_position,
        "ray_count": len(hits),
        "face_count": int(proxy["face_count"]),
        "vertex_count": int(proxy["vertex_count"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/visual_environment_collision_proxies.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/visual_environment_v6/environment_collision_validation_v1.json"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_path = args.output if args.output.is_absolute() else root / args.output
    document = load_json(manifest_path)
    import pybullet as pb  # pylint: disable=import-outside-toplevel

    client = pb.connect(pb.DIRECT)
    if client < 0:
        raise RuntimeError("PyBullet DIRECT connection failed")
    try:
        records = [validate_record(pb, root, record) for record in document["records"]]
    finally:
        pb.disconnect(client)
    failed = [record for record in records if not record["passed"]]
    report = {
        "version": VERSION,
        "proxy_manifest": str(manifest_path.relative_to(root)).replace("\\", "/"),
        "proxy_manifest_sha256": sha256(manifest_path),
        "scope": "all_admitted_visual_environment_static_meshes",
        "counts": {
            "tested": len(records),
            "passed": len(records) - len(failed),
            "failed": len(failed),
        },
        "records": records,
    }
    write_json(output_path, report)
    print(json.dumps(report["counts"], ensure_ascii=True))
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
