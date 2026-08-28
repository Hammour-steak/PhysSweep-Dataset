#!/usr/bin/env python3
"""Generate and audit a formal one-ball passive-pinball scene."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json_atomic as write_json
from tools.core.paths import resolve_project_path as project_path
from tools.dataset_contract.immutable_scene_contract import freeze_metadata, write_simulation_record
from tools.dataset_contract.object_identity_contract import attach_object_identity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("configs/passive_pinball_backend.json")
SPECIALIZED_REGISTRY = Path("configs/specialized_scene_backends.json")
RENDERER = Path("tools/rendering/render_passive_pinball_scene.py")
SCHEMA_VERSION = "physweep_passive_pinball_scene_v1"
AUDIT_VERSION = "physweep_passive_pinball_audit_v1"


def project_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"passive-pinball path is outside the project: {path}") from exc


def fixture_frame(fixture: dict[str, Any]) -> dict[str, list[float]]:
    tilt = math.radians(float(fixture["tilt_degrees_from_vertical"]))
    rotation_x = tilt - math.pi / 2.0
    right = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    down = np.asarray([0.0, math.sin(tilt), -math.cos(tilt)], dtype=np.float64)
    normal = np.asarray([0.0, math.cos(tilt), math.sin(tilt)], dtype=np.float64)
    matrix = np.column_stack((right, down, normal))
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-12):
        raise ValueError("passive-pinball fixture frame is not orthonormal")
    return {
        "right": right.tolist(),
        "down": down.tolist(),
        "normal": normal.tolist(),
        "orientation_quaternion_xyzw": [
            math.sin(rotation_x / 2.0),
            0.0,
            0.0,
            math.cos(rotation_x / 2.0),
        ],
    }


def _world_position(
    fixture: dict[str, Any], frame: dict[str, list[float]], x: float, down: float, normal: float
) -> list[float]:
    top = np.asarray(fixture["top_center_m"], dtype=np.float64)
    value = (
        top
        + x * np.asarray(frame["right"], dtype=np.float64)
        + down * np.asarray(frame["down"], dtype=np.float64)
        + normal * np.asarray(frame["normal"], dtype=np.float64)
    )
    return [round(float(item), 12) for item in value]


def build_fixture(config: dict[str, Any]) -> dict[str, Any]:
    fixture = config["fixture"]
    frame = fixture_frame(fixture)
    orientation = frame["orientation_quaternion_xyzw"]
    width = float(fixture["board_width_m"])
    height = float(fixture["board_height_m"])
    thickness = float(fixture["board_thickness_m"])
    rail_width = float(fixture["rail_width_m"])
    rail_height = float(fixture["rail_height_m"])
    peg_radius = float(fixture["peg_radius_m"])
    peg_length = float(fixture["peg_length_m"])
    colors = fixture["colors"]
    colliders: list[dict[str, Any]] = [
        {
            "id": "board",
            "role": "board",
            "shape": "box",
            "half_extents_m": [width / 2.0, height / 2.0, thickness / 2.0],
            "position_m": _world_position(fixture, frame, 0.0, height / 2.0, 0.0),
            "orientation_quaternion_xyzw": orientation,
            "color_rgba": colors["board_rgba"],
        }
    ]
    for side, sign in (("left", -1.0), ("right", 1.0)):
        colliders.append(
            {
                "id": f"rail_{side}",
                "role": "rail",
                "shape": "box",
                "half_extents_m": [rail_width / 2.0, height / 2.0, rail_height / 2.0],
                "position_m": _world_position(
                    fixture,
                    frame,
                    sign * (width + rail_width) / 2.0,
                    height / 2.0,
                    thickness / 2.0 + rail_height / 2.0,
                ),
                "orientation_quaternion_xyzw": orientation,
                "color_rgba": colors["rail_rgba"],
            }
        )
    row_count = int(fixture["peg_rows"])
    spacing = float(fixture["peg_column_spacing_m"])
    horizontal_limit = width / 2.0 - float(fixture["peg_horizontal_margin_m"])
    for row in range(row_count):
        row_offset = 0.0 if row % 2 == 0 else spacing / 2.0
        column = math.floor((-horizontal_limit - row_offset) / spacing)
        positions = []
        while True:
            x = row_offset + column * spacing
            if x > horizontal_limit + 1.0e-12:
                break
            if x >= -horizontal_limit - 1.0e-12:
                positions.append(x)
            column += 1
        down = float(fixture["peg_row_start_m"]) + row * float(
            fixture["peg_row_spacing_m"]
        )
        for index, x in enumerate(positions):
            colliders.append(
                {
                    "id": f"peg_{row:02d}_{index:02d}",
                    "role": "peg",
                    "shape": "cylinder",
                    "radius_m": peg_radius,
                    "length_m": peg_length,
                    "position_m": _world_position(
                        fixture,
                        frame,
                        x,
                        down,
                        thickness / 2.0 + peg_length / 2.0,
                    ),
                    "orientation_quaternion_xyzw": orientation,
                    "color_rgba": (
                        colors["peg_even_rgba"] if row % 2 == 0 else colors["peg_odd_rgba"]
                    ),
                }
            )
    catch_start = float(fixture["catch_start_m"])
    divider_height = float(fixture["catch_divider_height_m"])
    for index, x in enumerate(fixture["catch_divider_x_m"]):
        colliders.append(
            {
                "id": f"catch_divider_{index:02d}",
                "role": "catch_divider",
                "shape": "box",
                "half_extents_m": [rail_width / 2.0, divider_height / 2.0, rail_height / 2.0],
                "position_m": _world_position(
                    fixture,
                    frame,
                    float(x),
                    catch_start + divider_height / 2.0,
                    thickness / 2.0 + rail_height / 2.0,
                ),
                "orientation_quaternion_xyzw": orientation,
                "color_rgba": colors["catch_rgba"],
            }
        )
    colliders.append(
        {
            "id": "catch_bottom",
            "role": "catch_bottom",
            "shape": "box",
            "half_extents_m": [
                (width + 2.0 * rail_width) / 2.0,
                rail_width / 2.0,
                rail_height / 2.0,
            ],
            "position_m": _world_position(
                fixture,
                frame,
                0.0,
                height + rail_width / 2.0,
                thickness / 2.0 + rail_height / 2.0,
            ),
            "orientation_quaternion_xyzw": orientation,
            "color_rgba": colors["catch_rgba"],
        }
    )
    ids = [str(record["id"]) for record in colliders]
    if len(ids) != len(set(ids)):
        raise ValueError("passive-pinball fixture contains duplicate collider ids")
    return {
        "classification": "static",
        "representation": fixture["representation"],
        "frame": frame,
        "material": copy.deepcopy(fixture["material"]),
        "dimensions": {
            "width_m": width,
            "height_m": height,
            "board_thickness_m": thickness,
            "catch_start_m": catch_start,
        },
        "colliders": colliders,
    }


def validate_profile_offsets(
    config: dict[str, Any], fixture: dict[str, Any]
) -> None:
    source = config["fixture"]
    top = np.asarray(source["top_center_m"], dtype=np.float64)
    right = np.asarray(fixture["frame"]["right"], dtype=np.float64)
    peg_x = [
        float((np.asarray(record["position_m"], dtype=np.float64) - top) @ right)
        for record in fixture["colliders"]
        if record["role"] == "peg"
    ]
    for profile, rules in config["profiles"].items():
        offsets = [float(value) for value in rules["initial_x_offsets_m"]]
        if not offsets or len(offsets) != len(set(offsets)):
            raise ValueError(f"passive-pinball profile has invalid offsets: {profile}")
        aligned = [
            value
            for value in offsets
            if any(abs(value - peg) <= 1.0e-9 for peg in peg_x)
        ]
        if aligned:
            raise ValueError(
                f"passive-pinball profile has symmetry-degenerate offsets: "
                f"{profile}={aligned}"
            )


def initial_state(seed: int, profile: str, config: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    try:
        offsets = config["profiles"][profile]["initial_x_offsets_m"]
    except KeyError as exc:
        raise ValueError(f"unsupported passive-pinball profile: {profile}") from exc
    rng = random.Random(f"passive-pinball-initial:{profile}:{int(seed)}")
    x = float(rng.choice(offsets))
    frame = fixture["frame"]
    source = config["fixture"]
    radius = float(config["dynamic_object"]["radius_m"])
    normal = float(source["board_thickness_m"]) / 2.0 + radius + 0.0005
    return {
        "position_m": _world_position(source, frame, x, 0.055, normal),
        "orientation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "linear_velocity_m_s": [0.0, 0.0, 0.0],
        "angular_velocity_rad_s": [0.0, 0.0, 0.0],
    }


def passive_pinball_camera(seed: int, profile: str, config: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    rules = config["camera"]
    rng = random.Random(f"passive-pinball-camera:{profile}:{int(seed)}")
    frame = fixture["frame"]
    source = config["fixture"]
    target = np.asarray(
        _world_position(
            source,
            frame,
            float(rng.choice(rules["horizontal_offsets_m"])),
            float(rules["target_down_m"]),
            float(rules["target_normal_offset_m"]),
        ),
        dtype=np.float64,
    )
    normal = np.asarray(frame["normal"], dtype=np.float64)
    down = np.asarray(frame["down"], dtype=np.float64)
    position = target + float(rules["distance_m"]) * normal + float(
        rng.choice(rules["vertical_offsets_m"])
    ) * down
    return {
        "seed": int(seed),
        "mode": "passive_pinball_front_oblique",
        "position_m": [round(float(value), 9) for value in position],
        "target_m": [round(float(value), 9) for value in target],
        "focal_length_mm": float(rules["focal_length_mm"]),
        "sensor_width_mm": float(rules["sensor_width_mm"]),
    }


def _validate_metadata_files(root: Path, metadata: dict[str, Any]) -> None:
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported passive-pinball metadata")
    for label, binding in (
        ("backend config", metadata["physics"]["backend_config"]),
        ("generator", metadata["implementation"]["generator"]),
        ("renderer", metadata["implementation"]["renderer"]),
        ("specialized registry", metadata["implementation"]["specialized_registry"]),
    ):
        path = project_path(root, binding["path"])
        if not path.is_file() or sha256(path) != str(binding["sha256"]):
            raise ValueError(f"passive-pinball {label} binding changed")


def _local_coordinates(position: np.ndarray, top: np.ndarray, frame: dict[str, list[float]]) -> np.ndarray:
    relative = position - top
    return np.asarray(
        [
            float(relative @ np.asarray(frame["right"], dtype=np.float64)),
            float(relative @ np.asarray(frame["down"], dtype=np.float64)),
            float(relative @ np.asarray(frame["normal"], dtype=np.float64)),
        ],
        dtype=np.float64,
    )


def simulate(
    root: Path,
    metadata: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        import pybullet as pb  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError("pybullet is required for passive-pinball simulation") from exc

    _validate_metadata_files(root, metadata)
    simulation = metadata["simulation"]
    physics = metadata["physics"]
    fixture = physics["fixture"]
    dynamic = simulation["objects"][0]
    material = dynamic["material"]
    initial = dynamic["initial_state"]
    frame_count = int(simulation["time"]["frame_count"])
    output_fps = int(simulation["time"]["output_fps"])
    simulation_hz = int(simulation["time"]["simulation_hz"])
    steps_per_frame = simulation_hz // output_fps
    if simulation_hz % output_fps:
        raise ValueError("passive-pinball simulation frequency is not frame aligned")
    client = pb.connect(pb.DIRECT)
    if client < 0:
        raise RuntimeError("PyBullet DIRECT connection failed")
    try:
        pb.resetSimulation()
        pb.setGravity(*[float(value) for value in simulation["world"]["gravity_m_s2"]])
        pb.setTimeStep(1.0 / simulation_hz)
        engine = physics["engine"]
        pb.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / simulation_hz,
            numSolverIterations=int(engine["solver_iterations"]),
            deterministicOverlappingPairs=int(bool(engine["deterministic_overlapping_pairs"])),
            restitutionVelocityThreshold=float(engine["restitution_velocity_threshold_m_s"]),
            enableConeFriction=int(bool(engine["enable_cone_friction"])),
            useSplitImpulse=int(bool(engine["use_split_impulse"])),
        )
        shape_cache: dict[tuple[Any, ...], int] = {}
        body_to_collider: dict[int, str] = {}
        role_by_id: dict[str, str] = {}
        fixture_material = fixture["material"]
        for collider in fixture["colliders"]:
            if collider["shape"] == "box":
                key = ("box", *[float(value) for value in collider["half_extents_m"]])
                if key not in shape_cache:
                    shape_cache[key] = int(
                        pb.createCollisionShape(pb.GEOM_BOX, halfExtents=list(key[1:]))
                    )
            elif collider["shape"] == "cylinder":
                key = ("cylinder", float(collider["radius_m"]), float(collider["length_m"]))
                if key not in shape_cache:
                    shape_cache[key] = int(
                        pb.createCollisionShape(
                            pb.GEOM_CYLINDER,
                            radius=key[1],
                            height=key[2],
                        )
                    )
            else:
                raise ValueError(f"unsupported passive-pinball collider: {collider['shape']}")
            body = int(
                pb.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=shape_cache[key],
                    basePosition=[float(value) for value in collider["position_m"]],
                    baseOrientation=[
                        float(value) for value in collider["orientation_quaternion_xyzw"]
                    ],
                )
            )
            collider_id = str(collider["id"])
            body_to_collider[body] = collider_id
            role_by_id[collider_id] = str(collider["role"])
            pb.changeDynamics(
                body,
                -1,
                lateralFriction=float(fixture_material["contact_friction"]),
                restitution=float(fixture_material["contact_restitution"]),
            )
        radius = float(dynamic["collision_proxy"]["radius_m"])
        sphere = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius)
        ball = int(
            pb.createMultiBody(
                baseMass=float(material["mass_kg"]),
                baseCollisionShapeIndex=sphere,
                basePosition=[float(value) for value in initial["position_m"]],
                baseOrientation=[
                    float(value) for value in initial["orientation_quaternion_xyzw"]
                ],
            )
        )
        pb.resetBaseVelocity(
            ball,
            linearVelocity=[float(value) for value in initial["linear_velocity_m_s"]],
            angularVelocity=[float(value) for value in initial["angular_velocity_rad_s"]],
        )
        pb.changeDynamics(
            ball,
            -1,
            lateralFriction=float(material["contact_friction"]),
            restitution=float(material["contact_restitution"]),
            rollingFriction=float(material["rolling_friction"]),
            spinningFriction=float(material["spinning_friction"]),
            linearDamping=float(material["linear_damping"]),
            angularDamping=float(material["angular_damping"]),
            contactProcessingThreshold=0.0,
        )
        info = pb.getDynamicsInfo(ball, -1)
        runtime_material = np.asarray(
            [[float(info[0]), float(info[1]), float(info[5])]], dtype=np.float64
        )
        runtime_inertia = np.asarray([[float(value) for value in info[2]]], dtype=np.float64)
        positions = np.zeros((frame_count, 1, 3), dtype=np.float64)
        orientations = np.zeros((frame_count, 1, 4), dtype=np.float64)
        linear_velocities = np.zeros((frame_count, 1, 3), dtype=np.float64)
        angular_velocities = np.zeros((frame_count, 1, 3), dtype=np.float64)
        contact_counts = np.zeros((frame_count, 1), dtype=np.int32)
        contact_steps = {str(record["id"]): 0 for record in fixture["colliders"]}
        distinct_pegs: set[str] = set()
        peg_contact_steps = 0
        minimum_contact_distance = 0.0
        path_length = 0.0
        maximum_speed = 0.0
        mass = float(material["mass_kg"])
        inertia_scalar = 0.4 * mass * radius * radius
        gravity_z = abs(float(simulation["world"]["gravity_m_s2"][2]))
        initial_position = np.asarray(initial["position_m"], dtype=np.float64)
        previous_position = initial_position.copy()
        maximum_energy = gravity_z * float(initial_position[2])
        top = np.asarray(physics["fixture_source"]["top_center_m"], dtype=np.float64)
        minimum_local = np.asarray([math.inf, math.inf, math.inf], dtype=np.float64)
        maximum_local = np.asarray([-math.inf, -math.inf, -math.inf], dtype=np.float64)
        catch_entered = False

        def observe(frame: int) -> None:
            position, orientation = pb.getBasePositionAndOrientation(ball)
            linear, angular = pb.getBaseVelocity(ball)
            positions[frame, 0] = position
            orientations[frame, 0] = orientation
            linear_velocities[frame, 0] = linear
            angular_velocities[frame, 0] = angular
            contact_counts[frame, 0] = len(pb.getContactPoints(bodyA=ball))

        observe(0)
        total_steps = (frame_count - 1) * steps_per_frame
        for step in range(1, total_steps + 1):
            pb.stepSimulation()
            position = np.asarray(pb.getBasePositionAndOrientation(ball)[0], dtype=np.float64)
            linear, angular = pb.getBaseVelocity(ball)
            linear_array = np.asarray(linear, dtype=np.float64)
            angular_array = np.asarray(angular, dtype=np.float64)
            path_length += float(np.linalg.norm(position - previous_position))
            previous_position = position
            maximum_speed = max(maximum_speed, float(np.linalg.norm(linear_array)))
            energy = (
                0.5 * float(linear_array @ linear_array)
                + 0.5 * inertia_scalar / mass * float(angular_array @ angular_array)
                + gravity_z * float(position[2])
            )
            maximum_energy = max(maximum_energy, energy)
            local = _local_coordinates(position, top, fixture["frame"])
            minimum_local = np.minimum(minimum_local, local)
            maximum_local = np.maximum(maximum_local, local)
            if float(local[1]) >= float(fixture["dimensions"]["catch_start_m"]):
                catch_entered = True
            contacts = pb.getContactPoints(bodyA=ball)
            step_touched: set[str] = set()
            for contact in contacts:
                collider_id = body_to_collider.get(int(contact[2]))
                if collider_id is None:
                    continue
                step_touched.add(collider_id)
                minimum_contact_distance = min(minimum_contact_distance, float(contact[8]))
            for collider_id in step_touched:
                contact_steps[collider_id] += 1
                if role_by_id[collider_id] == "peg":
                    distinct_pegs.add(collider_id)
                    peg_contact_steps += 1
            if step % steps_per_frame == 0:
                observe(step // steps_per_frame)
    finally:
        pb.disconnect(client)

    arrays = {
        "time_s": np.arange(frame_count, dtype=np.float64) / float(output_fps),
        "position_m": positions,
        "quaternion_xyzw": orientations,
        "linear_velocity_m_s": linear_velocities,
        "angular_velocity_rad_s": angular_velocities,
        "contact_count": contact_counts,
        "runtime_material": runtime_material,
        "runtime_inertia_diagonal_kg_m2": runtime_inertia,
        "pinball__position_m": positions[:, 0],
        "pinball__quaternion_wxyz": orientations[:, 0, [3, 0, 1, 2]],
        "pinball__linear_velocity_m_s": linear_velocities[:, 0],
        "pinball__angular_velocity_rad_s": angular_velocities[:, 0],
    }
    quality = physics["quality"]
    initial_energy = gravity_z * float(initial_position[2])
    checks = {
        "finite_trajectory": all(np.isfinite(value).all() for value in arrays.values()),
        "frame_count": positions.shape == (frame_count, 1, 3),
        "minimum_path_length": path_length >= float(quality["minimum_path_length_m"]),
        "minimum_distinct_peg_contacts": len(distinct_pegs)
        >= int(quality["minimum_distinct_peg_contacts"]),
        "minimum_peg_contact_steps": peg_contact_steps
        >= int(quality["minimum_peg_contact_steps"]),
        "maximum_penetration": -minimum_contact_distance
        <= float(quality["maximum_penetration_m"]),
        "maximum_linear_speed": maximum_speed
        <= float(quality["maximum_linear_speed_m_s"]),
        "maximum_energy_gain": maximum_energy - initial_energy
        <= float(quality["maximum_energy_gain_j_per_kg"]),
        "local_x_bounds": max(abs(float(minimum_local[0])), abs(float(maximum_local[0])))
        <= float(quality["maximum_abs_local_x_m"]),
        "local_down_bounds": float(minimum_local[1])
        >= float(quality["minimum_local_down_m"])
        and float(maximum_local[1]) <= float(quality["maximum_local_down_m"]),
        "catch_entry": catch_entered or not bool(quality["catch_entry_required"]),
    }
    audit = {
        "schema_version": AUDIT_VERSION,
        "passed": all(checks.values()),
        "profile": metadata["semantics"]["profile"],
        "checks": checks,
        "metrics": {
            "path_length_m": round(path_length, 9),
            "distinct_peg_contacts": sorted(distinct_pegs),
            "peg_contact_steps": peg_contact_steps,
            "maximum_penetration_m": round(-minimum_contact_distance, 9),
            "maximum_linear_speed_m_s": round(maximum_speed, 9),
            "maximum_energy_gain_j_per_kg": round(maximum_energy - initial_energy, 9),
            "minimum_local_coordinates_m": [round(float(value), 9) for value in minimum_local],
            "maximum_local_coordinates_m": [round(float(value), 9) for value in maximum_local],
            "catch_entered": catch_entered,
            "contact_steps": contact_steps,
        },
    }
    return arrays, audit


def build_metadata(
    root: Path,
    output: Path,
    config_path: Path,
    config: dict[str, Any],
    seed: int,
    profile: str,
    scene_id: str,
    resolution: list[int] | None = None,
    samples: int | None = None,
) -> dict[str, Any]:
    if profile not in config["profiles"]:
        raise ValueError(f"unsupported passive-pinball profile: {profile}")
    fixture = build_fixture(config)
    validate_profile_offsets(config, fixture)
    dynamic = copy.deepcopy(config["dynamic_object"])
    initial = initial_state(seed, profile, config, fixture)
    dynamic_record = {
        "object_id": dynamic["object_id"],
        "semantic_type": dynamic["semantic_type"],
        "body_model": dynamic["body_model"],
        "is_dynamic": True,
        "collision_proxy": {"type": "sphere", "radius_m": dynamic["radius_m"]},
        "material": dynamic["material"],
        "initial_state": initial,
        "visual": {
            "shape": "sphere",
            "radius_m": dynamic["radius_m"],
            "color_rgba": dynamic["color_rgba"],
        },
    }
    physics_rules = config["physics"]
    frame_count = int(round(float(physics_rules["duration_s"]) * int(physics_rules["output_fps"]))) + 1
    trajectory_path = output / "trajectory.npz"
    audit_path = output / "audit.json"
    simulation_record_path = output / "simulation_record.json"
    generator_path = Path(__file__).resolve()
    renderer_path = project_path(root, RENDERER)
    registry_path = project_path(root, SPECIALIZED_REGISTRY)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "seed": int(seed),
        "dataset_stage": "passive_pinball_formal_base_candidate",
        "semantics": {
            "scene_family": config["scene_family"],
            "profile": profile,
            "description": config["profiles"][profile]["description"],
            "dynamic_object_count": 1,
        },
        "implementation": {
            "generator": {
                "path": project_relative(root, generator_path),
                "sha256": sha256(generator_path),
            },
            "renderer": {
                "path": project_relative(root, renderer_path),
                "sha256": sha256(renderer_path),
            },
            "specialized_registry": {
                "path": project_relative(root, registry_path),
                "sha256": sha256(registry_path),
            },
        },
        "simulation": {
            "time": {
                "duration_s": float(physics_rules["duration_s"]),
                "output_fps": int(physics_rules["output_fps"]),
                "simulation_hz": int(physics_rules["simulation_hz"]),
                "frame_count": frame_count,
            },
            "world": {"gravity_m_s2": copy.deepcopy(physics_rules["gravity_m_s2"])},
            "objects": [dynamic_record],
        },
        "physics": {
            "backend": "pybullet_passive_pinball_v1",
            "backend_config": {
                "path": project_relative(root, config_path),
                "sha256": sha256(config_path),
            },
            "profile": profile,
            "engine": {
                key: copy.deepcopy(physics_rules[key])
                for key in (
                    "solver_iterations",
                    "deterministic_overlapping_pairs",
                    "restitution_velocity_threshold_m_s",
                    "enable_cone_friction",
                    "use_split_impulse",
                )
            },
            "fixture_source": copy.deepcopy(config["fixture"]),
            "fixture": fixture,
            "quality": copy.deepcopy(config["quality"]),
            "trajectory_path": project_relative(root, trajectory_path),
            "audit_path": project_relative(root, audit_path),
            "simulation_record_path": project_relative(root, simulation_record_path),
        },
        "camera": passive_pinball_camera(seed, profile, config, fixture),
        "render": {
            "engine": config["render"]["engine"],
            "resolution": list(resolution or config["render"]["resolution"]),
            "samples": int(samples if samples is not None else config["render"]["samples"]),
            "world_color_rgb": copy.deepcopy(config["render"]["world_color_rgb"]),
            "lights": copy.deepcopy(config["render"]["lights"]),
            "inspection_frame_dir": project_relative(root, output / "inspection_frames"),
            "video_path": project_relative(root, output / f"{scene_id}.mp4"),
        },
    }
    attach_object_identity(
        metadata,
        trajectory_path=project_relative(root, trajectory_path),
        mask_path=project_relative(root, output / "masks" / scene_id),
    )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--profile", default="dense_pinfield_descent")
    parser.add_argument("--scene-id")
    parser.add_argument("--resolution", nargs=2, type=int)
    parser.add_argument("--samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = project_path(root, args.output)
    config_path = project_path(root, args.config)
    if output == root or root not in output.parents:
        raise ValueError("passive-pinball output must be a project subdirectory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"passive-pinball output is not empty: {output}")
    config = load_json(config_path)
    if config.get("schema_version") != "physweep_passive_pinball_backend_v1":
        raise ValueError("unsupported passive-pinball backend config")
    scene_id = args.scene_id or f"passive_pinball_{args.profile}_{int(args.seed):08d}"
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    metadata = build_metadata(
        root,
        output,
        config_path,
        config,
        int(args.seed),
        str(args.profile),
        scene_id,
        args.resolution,
        args.samples,
    )
    metadata = freeze_metadata(metadata_path, metadata)
    arrays, audit = simulate(root, metadata)
    if not audit["passed"]:
        raise RuntimeError(f"passive-pinball physics audit failed: {audit}")
    trajectory_path = output / "trajectory.npz"
    temporary = trajectory_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(trajectory_path)
    audit_path = output / "audit.json"
    write_json(audit_path, audit)
    write_simulation_record(
        root=root,
        metadata_path=metadata_path,
        metadata=metadata,
        trajectory_path=trajectory_path,
        audit_path=audit_path,
        record_path=output / "simulation_record.json",
    )
    print(json.dumps({"metadata": project_relative(root, metadata_path), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
