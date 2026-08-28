#!/usr/bin/env python3
"""Generate immutable metadata and trajectories for specialized billiards scenes."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pybullet as pb

from tools.assets.visual_environment_binding import choose_environment
from tools.core.hashing import relative_file_binding as file_binding
from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json
from tools.core.paths import resolve_project_path as project_path
from tools.dataset_contract.immutable_scene_contract import freeze_metadata, write_simulation_record
from tools.dataset_contract.object_identity_contract import attach_object_identity
from tools.assets.physical_proxy_catalog import load_catalog, records_by_id
from tools.physics.physics_time_step import simulation_hz_for_geometry
from tools.assets.static_support_proxy import (
    compile_static_support_binding,
    create_pybullet_static_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BILLIARDS_AUDIT_VERSION = "physweep_billiards_audit_v2"
PROFILE_FAMILIES = {
    "single_ball_free_roll": "billiards_single_ball",
    "single_ball_rail_rebound": "billiards_single_ball",
    "three_ball_collision": "billiards_collision",
}
PROFILE_DESCRIPTIONS = {
    "single_ball_free_roll": "One cue ball rolls freely across the table without touching a rail.",
    "single_ball_rail_rebound": "One cue ball strikes a rail and rebounds on the table.",
    "three_ball_collision": "A cue ball strikes two object balls in the central table region.",
}


def billiards_camera(
    seed: int,
    profile: str,
    specialized_views: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose a profile-specific, table-safe view reproducibly from the scene seed."""
    if specialized_views is None:
        specialized_views = load_json(
            PROJECT_ROOT / "configs/visual_sampling.json"
        )["specialized_camera_views"]
    view_rule = specialized_views[profile]
    rng = random.Random(f"billiards-camera:{profile}:{int(seed)}")
    base_x, base_y = 2.72, -3.18
    radius = math.hypot(base_x, base_y)
    base_angle = math.atan2(base_y, base_x)
    yaw_degrees = rng.choice(view_rule["yaw_offset_degrees"])
    distance_scale = rng.choice((0.98, 1.0, 1.02))
    angle = base_angle + math.radians(yaw_degrees)
    target_x = rng.uniform(-0.08, 0.08)
    target_y = rng.uniform(-0.06, 0.06)
    target_z = rng.choice((0.68, 0.70, 0.72))
    elevation_degrees = float(rng.choice(view_rule["elevation_degrees"]))
    horizontal_radius = radius * distance_scale
    height_m = target_z + horizontal_radius * math.tan(
        math.radians(elevation_degrees)
    )
    focal_length_mm = rng.choice((50.0, 52.0, 54.0))
    return {
        "seed": int(seed),
        "mode": f"bounded_orbit_{yaw_degrees:+d}deg",
        "position_m": [
            round(horizontal_radius * math.cos(angle), 6),
            round(horizontal_radius * math.sin(angle), 6),
            round(height_m, 6),
        ],
        "target_m": [round(target_x, 6), round(target_y, 6), target_z],
        "focal_length_mm": focal_length_mm,
        "sensor_width_mm": 36.0,
        "elevation_degrees": elevation_degrees,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 360])
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--scene-id")
    parser.add_argument("--support-id")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--semantic-rules", type=Path, required=True)
    parser.add_argument("--composition-rules", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--visual-rules", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_FAMILIES),
        default="three_ball_collision",
    )
    return parser.parse_args()


def initial_states(
    profile: str, bed_z: float, rules: dict[str, Any]
) -> list[dict[str, Any]]:
    try:
        records = rules["initial_states"][profile]
    except KeyError as exc:
        raise ValueError(f"unsupported billiards profile: {profile}") from exc
    center_z = bed_z + float(rules["ball_radius_m"]) + 0.001
    return [
        {
            "object_id": str(record["object_id"]),
            "position_m": [
                float(record["position_xy_m"][0]),
                float(record["position_xy_m"][1]),
                center_z,
            ],
            "velocity_m_s": [float(value) for value in record["velocity_m_s"]],
        }
        for record in records
    ]


def simulate(
    root: Path,
    static_support_binding: dict[str, Any],
    duration_s: float,
    output_fps: int,
    profile: str,
    backend: dict[str, Any],
    initial: list[dict[str, Any]] | None = None,
    object_materials: list[dict[str, float]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    rules = backend["billiards_rules"]
    engine_rules = rules["engine"]
    quality = rules["quality"]
    ball_radius_m = float(rules["ball_radius_m"])
    simulation_hz = simulation_hz_for_geometry(
        backend["engine"], [2.0 * ball_radius_m] * 3
    )
    steps_per_frame = simulation_hz // output_fps
    frame_count = int(round(duration_s * output_fps)) + 1
    bed_z = float(
        static_support_binding["target_support_frame"]["safe_surface"]["z_m"]
    )
    initial = initial or initial_states(profile, bed_z, rules)
    if object_materials is not None and len(object_materials) != len(initial):
        raise ValueError("billiards object material count differs from object count")
    client = pb.connect(pb.DIRECT)
    if client < 0:
        raise RuntimeError("PyBullet DIRECT connection failed")
    try:
        pb.resetSimulation()
        pb.setGravity(0.0, 0.0, -9.81)
        pb.setTimeStep(1.0 / simulation_hz)
        pb.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / simulation_hz,
            numSolverIterations=int(engine_rules["solver_iterations"]),
            deterministicOverlappingPairs=1,
            restitutionVelocityThreshold=float(
                engine_rules["restitution_velocity_threshold_m_s"]
            ),
            enableConeFriction=int(bool(engine_rules["enable_cone_friction"])),
            useSplitImpulse=int(bool(engine_rules["use_split_impulse"])),
        )
        support_body = create_pybullet_static_support(
            pb,
            root,
            static_support_binding,
        )
        pb.changeDynamics(
            support_body,
            -1,
            lateralFriction=float(rules["support_dynamics"]["lateral_friction"]),
            restitution=float(rules["support_dynamics"]["restitution"]),
        )
        sphere = pb.createCollisionShape(pb.GEOM_SPHERE, radius=ball_radius_m)
        bodies = []
        runtime_materials = []
        runtime_inertias = []
        for index, record in enumerate(initial):
            ball_dynamics = rules["ball_dynamics"]
            material = (
                object_materials[index]
                if object_materials is not None
                else {
                    "mass_kg": float(rules["ball_mass_kg"]),
                    "contact_friction": float(ball_dynamics["lateral_friction"]),
                    "contact_restitution": float(ball_dynamics["restitution"]),
                }
            )
            body = pb.createMultiBody(
                baseMass=float(material["mass_kg"]),
                baseCollisionShapeIndex=sphere,
                basePosition=record["position_m"],
            )
            pb.changeDynamics(
                body,
                -1,
                lateralFriction=float(material["contact_friction"]),
                restitution=float(material["contact_restitution"]),
                linearDamping=float(ball_dynamics["linear_damping"]),
                angularDamping=float(ball_dynamics["angular_damping"]),
                rollingFriction=float(ball_dynamics["rolling_friction"]),
                spinningFriction=float(ball_dynamics["spinning_friction"]),
                contactProcessingThreshold=float(
                    ball_dynamics["contact_processing_threshold_m"]
                ),
            )
            pb.resetBaseVelocity(body, linearVelocity=record["velocity_m_s"])
            bodies.append(body)
            info = pb.getDynamicsInfo(body, -1)
            runtime_materials.append([float(info[0]), float(info[1]), float(info[5])])
            runtime_inertias.append([float(value) for value in info[2]])

        positions = np.zeros((frame_count, len(bodies), 3), dtype=np.float64)
        quaternions = np.zeros((frame_count, len(bodies), 4), dtype=np.float64)
        velocities = np.zeros((frame_count, len(bodies), 3), dtype=np.float64)
        angular_velocities = np.zeros((frame_count, len(bodies), 3), dtype=np.float64)
        contact_counts = np.zeros((frame_count, len(bodies)), dtype=np.int32)
        ball_contact_frames = 0
        rail_contact_frames = 0
        ball_contact_indices: list[int] = []
        rail_contact_indices: list[int] = []
        min_z = math.inf
        for frame in range(frame_count):
            for index, body in enumerate(bodies):
                position, orientation = pb.getBasePositionAndOrientation(body)
                linear, angular = pb.getBaseVelocity(body)
                positions[frame, index] = position
                quaternions[frame, index] = orientation
                velocities[frame, index] = linear
                angular_velocities[frame, index] = angular
                contact_counts[frame, index] = len(pb.getContactPoints(bodyA=body))
                min_z = min(min_z, float(position[2]))
            if frame == frame_count - 1:
                break
            ball_contact = False
            rail_contact = False
            for _ in range(steps_per_frame):
                pb.stepSimulation()
                ball_contact |= any(
                    pb.getContactPoints(bodyA=bodies[i], bodyB=bodies[j])
                    for i in range(len(bodies))
                    for j in range(i + 1, len(bodies))
                )
                for body in bodies:
                    rail_contact |= any(
                        int(contact[2]) == support_body
                        and float(contact[6][2])
                        > bed_z
                        + float(
                            quality["minimum_rail_contact_height_above_bed_m"]
                        )
                        for contact in pb.getContactPoints(bodyA=body, bodyB=support_body)
                    )
            ball_contact_frames += int(ball_contact)
            rail_contact_frames += int(rail_contact)
            if ball_contact:
                ball_contact_indices.append(frame + 1)
            if rail_contact:
                rail_contact_indices.append(frame + 1)
    finally:
        pb.disconnect(client)

    arrays = {
        "time_s": np.arange(frame_count, dtype=np.float64) / float(output_fps),
        "position_m": positions,
        "quaternion_xyzw": quaternions,
        "linear_velocity_m_s": velocities,
        "angular_velocity_rad_s": angular_velocities,
        "contact_count": contact_counts,
        "runtime_material": np.asarray(runtime_materials, dtype=np.float64),
        "runtime_inertia_diagonal_kg_m2": np.asarray(
            runtime_inertias, dtype=np.float64
        ),
    }
    displacement = np.linalg.norm(positions - positions[0:1], axis=2)
    planar_speed = np.linalg.norm(velocities[:, :, :2], axis=2)
    vertical_speed = np.abs(velocities[:, :, 2])
    initial_max_planar_speed = float(planar_speed[0].max())
    initial_clearance_m = max(
        0.0,
        max(float(record["position_m"][2]) for record in initial)
        - bed_z
        - ball_radius_m,
    )
    vertical_speed_limit = (
        1.35 * math.sqrt(2.0 * 9.81 * initial_clearance_m) + 0.03
    )
    checks: dict[str, bool] = {
        "finite_trajectory": all(np.isfinite(value).all() for value in arrays.values()),
        "visible_motion": float(displacement.max())
        > float(quality["minimum_visible_motion_m"]),
        "balls_remain_on_or_above_bed": min_z
        >= bed_z
        + ball_radius_m
        - float(quality["maximum_bed_penetration_m"]),
        "balls_remain_inside_rails": bool(
            np.max(np.abs(positions[:, :, 0]))
            < float(quality["maximum_abs_x_m"])
            and np.max(np.abs(positions[:, :, 1]))
            < float(quality["maximum_abs_y_m"])
        ),
        "no_unforced_speed_gain": float(planar_speed.max())
        <= max(
            0.02,
            initial_max_planar_speed
            * float(quality["maximum_unforced_speed_gain_ratio"]),
        ),
        "vertical_motion_matches_initial_contact_clearance": float(
            vertical_speed.max()
        )
        <= vertical_speed_limit,
    }
    if profile == "three_ball_collision":
        checks.update(
            {
                "three_dynamic_balls": len(initial) == 3,
                "ball_ball_collision_observed": ball_contact_frames > 0,
                "central_collision_avoids_rails": rail_contact_frames == 0,
            }
        )
    elif profile == "single_ball_free_roll":
        direction = np.asarray(initial[0]["velocity_m_s"][:2], dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), 1.0e-12)
        projected_velocity = velocities[:, 0, :2] @ direction
        checks.update(
            {
                "one_dynamic_ball": len(initial) == 1,
                "free_roll_avoids_rails": rail_contact_frames == 0,
                "free_roll_does_not_reverse": float(projected_velocity.min()) >= -0.02,
            }
        )
    elif profile == "single_ball_rail_rebound":
        first_rail_contact = rail_contact_indices[0] if rail_contact_indices else None
        reversal: dict[str, Any] | None = None
        if first_rail_contact is not None and first_rail_contact > 0:
            before = velocities[first_rail_contact - 1, 0, :2]
            after = velocities[first_rail_contact, 0, :2]
            candidates = [
                axis
                for axis in range(2)
                if abs(float(before[axis]))
                >= float(quality["minimum_pre_rebound_normal_speed_m_s"])
                and float(before[axis] * after[axis]) < 0.0
                and abs(float(after[axis]))
                >= float(quality["minimum_post_rebound_normal_speed_m_s"])
            ]
            if candidates:
                axis = max(candidates, key=lambda value: abs(float(before[value])))
                ratio = abs(float(after[axis] / before[axis]))
                reversal = {
                    "axis": "xy"[axis],
                    "before_m_s": round(float(before[axis]), 6),
                    "after_m_s": round(float(after[axis]), 6),
                    "speed_ratio": round(ratio, 6),
                }
        checks.update(
            {
                "one_dynamic_ball": len(initial) == 1,
                "rail_contact_observed": rail_contact_frames > 0,
                "rail_normal_velocity_reverses": reversal is not None,
                "rail_rebound_energy_is_bounded": reversal is not None
                and float(quality["rail_rebound_speed_ratio"][0])
                <= float(reversal["speed_ratio"])
                <= float(quality["rail_rebound_speed_ratio"][1]),
            }
        )
    else:
        raise ValueError(f"unsupported billiards profile: {profile}")
    audit = {
        "audit_version": BILLIARDS_AUDIT_VERSION,
        "passed": all(checks.values()),
        "profile": profile,
        "checks": checks,
        "ball_contact_frames": ball_contact_frames,
        "rail_contact_frames": rail_contact_frames,
        "first_rail_contact_frame": (
            rail_contact_indices[0] if rail_contact_indices else None
        ),
        "rail_reversal": reversal if profile == "single_ball_rail_rebound" else None,
        "maximum_displacement_m": round(float(displacement.max()), 6),
        "minimum_center_z_m": round(min_z, 6),
        "initial_max_planar_speed_m_s": round(initial_max_planar_speed, 6),
        "maximum_planar_speed_m_s": round(float(planar_speed.max()), 6),
        "maximum_vertical_speed_m_s": round(float(vertical_speed.max()), 6),
        "vertical_speed_limit_m_s": round(vertical_speed_limit, 6),
        "simulation_hz": simulation_hz,
        "collision_authority": "exact_static_proxy",
        "support_binding_sha256": static_support_binding["binding_sha256"],
    }
    return arrays, audit, initial


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    registry_path = project_path(root, args.registry)
    semantic_rules_path = project_path(root, args.semantic_rules)
    composition_rules_path = project_path(root, args.composition_rules)
    backend_path = project_path(root, args.backend)
    visual_rules_path = project_path(root, args.visual_rules)
    catalog_path = project_path(root, args.catalog)
    registry = load_json(registry_path)
    proxy_manifest, proxy_records = load_catalog(
        root, catalog_path, require_runtime_validation=True
    )
    physical_by_id = records_by_id(proxy_records)
    semantic_rules = load_json(semantic_rules_path)
    scene_family = PROFILE_FAMILIES[args.profile]
    billiards_rules = semantic_rules["specialized_scene_families"][scene_family]
    if args.profile not in billiards_rules["profiles"]:
        raise ValueError(
            f"profile {args.profile} is not admitted by semantic family {scene_family}"
        )
    composition_rules = load_json(composition_rules_path)
    backend = load_json(backend_path)
    visual_rules = load_json(visual_rules_path)
    compositions = {
        record["asset_id"]: record for record in composition_rules["records"]
    }
    candidates = [
        record
        for record in registry["records"]
        if record["semantic_category"] == billiards_rules["support_category"]
        and record["admission"].get("sampling_enabled", False)
        and record["asset_id"] in compositions
        and scene_family in compositions[record["asset_id"]]["scene_fit"]["allowed"]
    ]
    if args.support_id:
        candidates = [
            record
            for record in candidates
            if record["asset_id"] == args.support_id
        ]
    if len(candidates) != 1:
        raise ValueError(
            f"expected one admitted support for {scene_family}"
            f" (requested={args.support_id!r}), found {len(candidates)}"
        )
    support = candidates[0]
    composition = compositions[support["asset_id"]]
    if not support["admission"].get("sampling_enabled", False):
        raise ValueError("pool-table support is not admitted")
    if support["semantic_category"] != billiards_rules["support_category"]:
        raise ValueError("pool-table support violates the billiards semantic rule")
    if composition["sampling_status"] != "ready_specialized":
        raise ValueError("pool-table component composition is not ready for specialized sampling")
    support_proxy_record = physical_by_id[support["asset_id"]]
    if not support_proxy_record["admission"]["sampling_ready"]:
        raise ValueError("pool-table exact support proxy is not sampling-ready")
    support_binding = compile_static_support_binding(
        support_proxy_record,
        usage_id="curated_support",
    )
    bed_z = float(
        support_binding["target_support_frame"]["safe_surface"]["z_m"]
    )
    initial = initial_states(args.profile, bed_z, backend["billiards_rules"])
    if len(initial) != int(billiards_rules["dynamic_object_count"]):
        raise ValueError("billiards initial state violates declared object count")
    if [record["object_id"] for record in initial] != list(
        billiards_rules["dynamic_semantics"]
    ):
        raise ValueError("billiards initial state violates declared object semantics")
    output.mkdir(parents=True, exist_ok=True)
    trajectory_path = output / "trajectory.npz"
    audit_path = output / "audit.json"
    simulation_record_path = output / "simulation_record.json"
    rng = random.Random(args.seed)
    hdri_records = load_json(root / "assets/manifests/hdri_admission.json")["records"]
    environment = choose_environment(
        root, support, hdri_records, visual_rules, rng
    )
    metadata_path = output / "metadata.json"
    scene_id = args.scene_id or f"billiards_{args.profile}_v1"
    simulation_hz = simulation_hz_for_geometry(
        backend["engine"],
        [2.0 * float(backend["billiards_rules"]["ball_radius_m"])] * 3,
    )
    frame_count = int(round(args.duration * args.fps)) + 1
    metadata = {
        "schema_version": "physweep_billiards_scene_v4",
        "scene_id": scene_id,
        "seed": args.seed,
        "semantics": {
            "scene_family": scene_family,
            "profile": args.profile,
            "description": PROFILE_DESCRIPTIONS[args.profile],
            "dynamic_object_count": len(initial),
        },
        "registry": {"path": str(registry_path.relative_to(root)), "sha256": sha256(registry_path)},
        "semantic_rules": {
            "path": str(semantic_rules_path.relative_to(root)),
            "sha256": sha256(semantic_rules_path),
        },
        "composition_rules": {
            "path": str(composition_rules_path.relative_to(root)),
            "sha256": sha256(composition_rules_path),
        },
        "visual_rules": {
            "path": str(visual_rules_path.relative_to(root)),
            "sha256": sha256(visual_rules_path),
        },
        "physical_proxy_catalog": {
            "path": str(catalog_path.relative_to(root)),
            "sha256": sha256(catalog_path),
            "records_sha256": proxy_manifest["records_sha256"],
        },
        "assets": {"support_asset_id": support["asset_id"]},
        "physics": {
            "backend": "pybullet_exact_static_support_v1",
            "backend_config": {
                "path": str(backend_path.relative_to(root)),
                "sha256": sha256(backend_path),
            },
            "profile": args.profile,
            "duration_s": args.duration,
            "output_fps": args.fps,
            "simulation_hz": simulation_hz,
            "frame_count": frame_count,
            "ball_radius_m": float(backend["billiards_rules"]["ball_radius_m"]),
            "ball_mass_kg": float(backend["billiards_rules"]["ball_mass_kg"]),
            "initial_states": initial,
            "static_support_binding": support_binding,
            "trajectory_path": str(trajectory_path.relative_to(root)),
            "audit_path": str(audit_path.relative_to(root)),
            "simulation_record_path": str(
                simulation_record_path.relative_to(root)
            ),
        },
        "camera": {
            **billiards_camera(
                args.seed,
                args.profile,
                visual_rules["specialized_camera_views"],
            ),
        },
        "render": {
            "evidence_contract": "physweep_specialized_render_evidence_v2",
            "engine": "BLENDER_EEVEE",
            "resolution": args.resolution,
            "samples": args.samples,
            "environment": environment,
            "inspection_frame_dir": str((output / "inspection_frames").relative_to(root)),
            "video_path": str((output / f"{scene_id}.mp4").relative_to(root)),
        },
        "implementation": {
            "generator": file_binding(root, Path(__file__)),
            "renderer": file_binding(root, root / "tools/rendering/render_billiards_scene.py"),
            "render_evidence": file_binding(
                root, root / "tools/rendering/specialized_render_evidence.py"
            ),
        },
    }
    attach_object_identity(
        metadata,
        trajectory_path=str(trajectory_path.relative_to(root)),
        mask_path=str((output / "masks" / scene_id).relative_to(root)),
    )
    metadata = freeze_metadata(metadata_path, metadata)
    physics = metadata["physics"]
    arrays, audit, simulated_initial = simulate(
        root,
        physics["static_support_binding"],
        float(physics["duration_s"]),
        int(physics["output_fps"]),
        str(physics["profile"]),
        backend,
        initial=physics["initial_states"],
    )
    if simulated_initial != physics["initial_states"]:
        raise RuntimeError("simulation initial state differs from frozen metadata")
    if int(audit["simulation_hz"]) != int(physics["simulation_hz"]):
        raise RuntimeError("simulation frequency differs from frozen metadata")
    if int(arrays["position_m"].shape[0]) != int(physics["frame_count"]):
        raise RuntimeError("trajectory length differs from frozen metadata")
    if not audit["passed"]:
        raise RuntimeError(f"billiards physics audit failed: {audit}")
    np.savez_compressed(trajectory_path, **arrays)
    write_json(audit_path, audit)
    write_simulation_record(
        root=root,
        metadata_path=metadata_path,
        metadata=metadata,
        trajectory_path=trajectory_path,
        audit_path=audit_path,
        record_path=simulation_record_path,
    )
    print(json.dumps({"metadata": str(metadata_path), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
