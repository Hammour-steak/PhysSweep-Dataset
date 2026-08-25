#!/usr/bin/env python3
"""Build and numerically audit the isolated marble-run release candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

try:
    from derive_physics_sweep import _round_value, _sweep_values
    from physics_invariants import quaternion_matrix_wxyz
except ModuleNotFoundError:  # imported as tools.* in tests
    from tools.derive_physics_sweep import _round_value, _sweep_values
    from tools.physics_invariants import quaternion_matrix_wxyz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path("configs/candidates/marble_run_v1.json")
OUTPUT_PATH = Path("outputs/specialized_scene_review/marble_run_v1/base")
SWEEP_CONFIG_PATH = Path("configs/physics_sweep.json")
SWEEP_OUTPUT_PATH = Path("outputs/specialized_scene_review/marble_run_v1/sweep")
RENDERER_PATH = Path("tools/render_marble_run_candidate.py")
METADATA_VERSION = "physweep_static_fixture_candidate_metadata_v1"
AUDIT_VERSION = "physweep_static_fixture_candidate_audit_v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def project_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"candidate path is outside the project: {path}") from exc


def _finite_vector(value: Any, length: int, label: str) -> list[float]:
    result = [float(item) for item in value]
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"invalid {label}: {value}")
    return result


def _positive(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"invalid {label}: {value}")
    return result


def _unit_quaternion(value: Any, label: str) -> list[float]:
    quaternion = _finite_vector(value, 4, label)
    if not math.isclose(
        sum(component * component for component in quaternion),
        1.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError(f"{label} is not normalized")
    return quaternion


def validate_config(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    if config.get("schema_version") != "physweep_static_fixture_candidate_config_v1":
        raise ValueError("unsupported marble-run candidate config")
    admission = config["admission"]
    if admission.get("status") != "candidate_only" or admission.get(
        "release_enabled"
    ) is not False:
        raise ValueError("unreviewed fixture must remain candidate-only")
    semantics = config["semantics"]
    if semantics.get("scene_family") != "marble_run":
        raise ValueError("candidate has the wrong scene family")
    if int(semantics.get("dynamic_object_count", -1)) != 1:
        raise ValueError("marble-run candidate must have one dynamic object")

    source = config["source"]
    source_root = project_path(root, source["local_root"])
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"fetch {source['repository_url']} at {source['commit']} into {source_root}"
        )
    revision = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != str(source["commit"]):
        raise ValueError(f"source revision changed: {revision}")
    dirty = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise ValueError("marble-run source checkout is dirty")

    source_paths: dict[str, Path] = {}
    for record in source["files"]:
        relative = str(record["path"])
        path = (source_root / relative).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise ValueError(f"source file escapes checkout: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != str(record["sha256"]):
            raise ValueError(f"source hash mismatch: {relative}")
        source_paths[relative] = path

    fixture = config["fixture"]
    if fixture.get("classification") != "static":
        raise ValueError("fixture must be static")
    if fixture.get("representation") != (
        "compound_exact_static_triangle_mesh_and_analytic_boxes"
    ):
        raise ValueError("fixture representation is not exact and explicit")
    _positive(fixture["uniform_scale_m_per_source_unit"], "fixture scale")
    matrix = np.asarray(fixture["source_to_world_rotation_matrix"], dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("invalid source-to-world rotation")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-9):
        raise ValueError("source-to-world transform is not orthonormal")
    if not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1.0e-9):
        raise ValueError("source-to-world transform is not a proper rotation")
    quaternion = _unit_quaternion(
        fixture["source_to_world_orientation_quaternion_xyzw"],
        "fixture orientation",
    )
    x, y, z, w = quaternion
    if not np.allclose(
        matrix,
        quaternion_matrix_wxyz(np.asarray([w, x, y, z], dtype=float)),
        atol=1.0e-9,
    ):
        raise ValueError("fixture rotation matrix and quaternion disagree")

    ids: list[str] = []
    for component in fixture["mesh_components"]:
        ids.append(str(component["id"]))
        if str(component["source_file"]) not in source_paths:
            raise ValueError(f"undeclared component source: {component['source_file']}")
        _finite_vector(component["base_position_m"], 3, "component position")
        _finite_vector(component["color_rgba"], 4, "component color")
    for collider in fixture["analytic_colliders"]:
        ids.append(str(collider["id"]))
        if collider.get("shape") != "box":
            raise ValueError(f"unsupported analytic collider: {collider['id']}")
        half = _finite_vector(collider["half_extents_m"], 3, "box half extents")
        if min(half) <= 0.0:
            raise ValueError(f"non-positive box: {collider['id']}")
        _finite_vector(collider["position_m"], 3, "box position")
        _finite_vector(collider["color_rgba"], 4, "box color")
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate fixture component id")

    dynamic = config["dynamic_object"]
    if dynamic.get("object_id") != "marble" or dynamic.get("shape") != "sphere":
        raise ValueError("candidate dynamic identity must be one marble sphere")
    _positive(dynamic["radius_m"], "marble radius")
    _finite_vector(dynamic["initial_state"]["position_m"], 3, "initial position")
    _unit_quaternion(
        dynamic["initial_state"]["orientation_quaternion_xyzw"],
        "initial orientation",
    )
    _finite_vector(dynamic["initial_state"]["linear_velocity_m_s"], 3, "initial velocity")
    _finite_vector(
        dynamic["initial_state"]["angular_velocity_rad_s"],
        3,
        "initial angular velocity",
    )
    material = dynamic["material"]
    _positive(material["mass_kg"], "marble mass")
    mass_range = _finite_vector(material["mass_range_kg"], 2, "marble mass range")
    if not mass_range[0] <= float(material["mass_kg"]) <= mass_range[1]:
        raise ValueError("base marble mass is outside its declared range")

    physics = config["physics"]
    gravity = _finite_vector(physics["gravity_m_s2"], 3, "gravity")
    if not np.allclose(gravity[:2], [0.0, 0.0], atol=1.0e-12) or gravity[2] >= 0.0:
        raise ValueError("candidate audit requires gravity along negative world z")
    duration = _positive(physics["duration_s"], "duration")
    output_fps = int(_positive(physics["output_fps"], "output fps"))
    simulation_hz = int(_positive(physics["simulation_hz"], "simulation hz"))
    if simulation_hz % output_fps:
        raise ValueError("simulation frequency must be an output-frame multiple")
    if not math.isclose(duration * output_fps, round(duration * output_fps)):
        raise ValueError("duration does not produce an integral frame count")

    render = config["render"]
    resolution = [int(value) for value in render["resolution"]]
    if len(resolution) != 2 or min(resolution) <= 0:
        raise ValueError("invalid candidate render resolution")
    if int(render["fps"]) != output_fps:
        raise ValueError("render and physics output fps differ")
    camera = render["camera"]
    _finite_vector(camera["position_m"], 3, "camera position")
    _finite_vector(camera["target_m"], 3, "camera target")
    _positive(camera["focal_length_mm"], "camera focal length")
    backboard = render["context"]["backboard"]
    if backboard.get("physics_role") != "render_only_context":
        raise ValueError("candidate backboard must remain render-only")
    if min(_finite_vector(backboard["half_extents_m"], 3, "backboard size")) <= 0:
        raise ValueError("candidate backboard has non-positive size")
    _finite_vector(backboard["position_m"], 3, "backboard position")
    light_ids = []
    for light in render["lights"]:
        light_ids.append(str(light["id"]))
        if light.get("type") != "AREA":
            raise ValueError(f"unsupported candidate light: {light['id']}")
        _finite_vector(light["position_m"], 3, "light position")
        _finite_vector(light["color_rgb"], 3, "light color")
        _positive(light["energy_w"], "light energy")
        _positive(light["size_m"], "light size")
    if len(light_ids) != len(set(light_ids)):
        raise ValueError("duplicate candidate light id")
    return source_paths


def materialize_collision_meshes(
    root: Path,
    output: Path,
    config: dict[str, Any],
    source_paths: dict[str, Path],
) -> dict[str, dict[str, Any]]:
    try:
        import trimesh  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError("trimesh is required to materialize candidate meshes") from exc

    results: dict[str, dict[str, Any]] = {}
    source_records = {
        str(record["path"]): record for record in config["source"]["files"]
    }
    used = sorted(
        {str(item["source_file"]) for item in config["fixture"]["mesh_components"]}
    )
    for relative in used:
        source = source_paths[relative]
        source_mesh = trimesh.load_mesh(source, process=False)
        if not isinstance(source_mesh, trimesh.Trimesh):
            raise ValueError(f"source is not one triangle mesh: {relative}")
        mesh = source_mesh.copy()
        mesh.merge_vertices()
        expected = source_records[relative].get("topology", {})
        observed = {
            "source_vertex_count": int(len(source_mesh.vertices)),
            "canonical_vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.faces)),
            "source_watertight": bool(source_mesh.is_watertight),
            "watertight_after_vertex_merge": bool(mesh.is_watertight),
        }
        if expected and observed != expected:
            raise ValueError(
                f"source topology mismatch for {relative}: {observed} != {expected}"
            )
        if not observed["watertight_after_vertex_merge"]:
            raise ValueError(f"canonical collision mesh is not watertight: {relative}")
        collision_path = output / "collision" / f"{source.stem}.obj"
        collision_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = collision_path.with_suffix(".tmp.obj")
        source_mesh.export(
            temporary, file_type="obj", include_normals=False, include_color=False
        )
        temporary.replace(collision_path)
        results[relative] = {
            "path": project_relative(root, collision_path),
            "sha256": sha256(collision_path),
            **observed,
        }
    return results


def build_metadata(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    collision: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    fixture = config["fixture"]
    mesh_components = []
    source_hashes = {
        str(record["path"]): str(record["sha256"])
        for record in config["source"]["files"]
    }
    for item in fixture["mesh_components"]:
        source_file = str(item["source_file"])
        mesh_components.append(
            {
                "id": str(item["id"]),
                "classification": "static",
                "representation": "exact_static_triangle_mesh",
                "source_path": (
                    Path(config["source"]["local_root"]) / source_file
                ).as_posix(),
                "source_sha256": source_hashes[source_file],
                "collision": copy.deepcopy(collision[source_file]),
                "mesh_scale": [
                    float(fixture["uniform_scale_m_per_source_unit"])
                ]
                * 3,
                "base_position_m": copy.deepcopy(item["base_position_m"]),
                "base_orientation_quaternion_xyzw": copy.deepcopy(
                    fixture["source_to_world_orientation_quaternion_xyzw"]
                ),
                "pybullet_flags": ["GEOM_FORCE_CONCAVE_TRIMESH"],
                "color_rgba": copy.deepcopy(item["color_rgba"]),
            }
        )
    dynamic = copy.deepcopy(config["dynamic_object"])
    frame_count = int(
        round(float(config["physics"]["duration_s"]) * int(config["physics"]["output_fps"]))
    ) + 1
    render = copy.deepcopy(config["render"])
    renderer_path = project_path(root, RENDERER_PATH)
    if not renderer_path.is_file():
        raise FileNotFoundError(renderer_path)
    render["implementation"] = {
        "path": project_relative(root, renderer_path),
        "sha256": sha256(renderer_path),
    }
    metadata = {
        "schema_version": METADATA_VERSION,
        "scene_id": str(config["candidate_id"]),
        "admission": copy.deepcopy(config["admission"]),
        "semantics": copy.deepcopy(config["semantics"]),
        "candidate_config": {
            "path": project_relative(root, config_path),
            "sha256": sha256(config_path),
        },
        "implementation": {
            "path": project_relative(root, Path(__file__)),
            "sha256": sha256(Path(__file__)),
        },
        "source": copy.deepcopy(config["source"]),
        "fixture": {
            "classification": "static",
            "representation": fixture["representation"],
            "collision_authority": fixture["collision_authority"],
            "mesh_components": mesh_components,
            "analytic_colliders": copy.deepcopy(fixture["analytic_colliders"]),
            "mesh_material": copy.deepcopy(fixture["material"]),
            "analytic_material": copy.deepcopy(fixture["catch_material"]),
        },
        "object_identity": {
            "object_count": 1,
            "objects": [
                {
                    "object_id": dynamic["object_id"],
                    "object_index": 0,
                    "role": "dynamic",
                    "semantic": "marble",
                }
            ],
        },
        "physics": {
            **copy.deepcopy(config["physics"]),
            "frame_count": frame_count,
            "objects": [dynamic],
        },
        "quality": copy.deepcopy(config["quality"]),
        "render": render,
    }
    return metadata


def _validate_metadata_files(root: Path, metadata: dict[str, Any]) -> None:
    if metadata.get("schema_version") != METADATA_VERSION:
        raise ValueError("unsupported marble-run candidate metadata")
    if metadata.get("admission", {}).get("release_enabled") is not False:
        raise ValueError("candidate metadata unexpectedly enables release")
    config_record = metadata["candidate_config"]
    config_path = project_path(root, config_record["path"])
    if sha256(config_path) != str(config_record["sha256"]):
        raise ValueError("candidate config hash mismatch")
    implementation = metadata["implementation"]
    implementation_path = project_path(root, implementation["path"])
    if sha256(implementation_path) != str(implementation["sha256"]):
        raise ValueError("candidate implementation hash mismatch")
    renderer = metadata["render"]["implementation"]
    renderer_path = project_path(root, renderer["path"])
    if sha256(renderer_path) != str(renderer["sha256"]):
        raise ValueError("candidate renderer hash mismatch")
    for component in metadata["fixture"]["mesh_components"]:
        source = project_path(root, component["source_path"])
        collision = project_path(root, component["collision"]["path"])
        if sha256(source) != str(component["source_sha256"]):
            raise ValueError(f"component source hash mismatch: {component['id']}")
        if sha256(collision) != str(component["collision"]["sha256"]):
            raise ValueError(f"component collision hash mismatch: {component['id']}")


def simulate(root: Path, metadata: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        import pybullet as pb  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise RuntimeError("pybullet is required to simulate the candidate") from exc

    _validate_metadata_files(root, metadata)
    physics = metadata["physics"]
    dynamic = physics["objects"][0]
    initial = dynamic["initial_state"]
    simulation_hz = int(physics["simulation_hz"])
    output_fps = int(physics["output_fps"])
    frame_count = int(physics["frame_count"])
    steps_per_frame = simulation_hz // output_fps
    client = pb.connect(pb.DIRECT)
    if client < 0:
        raise RuntimeError("PyBullet DIRECT connection failed")
    try:
        pb.resetSimulation()
        pb.setGravity(*[float(value) for value in physics["gravity_m_s2"]])
        pb.setTimeStep(1.0 / simulation_hz)
        pb.setPhysicsEngineParameter(
            numSolverIterations=int(physics["solver_iterations"]),
            deterministicOverlappingPairs=int(
                bool(physics["deterministic_overlapping_pairs"])
            ),
            restitutionVelocityThreshold=float(
                physics["restitution_velocity_threshold_m_s"]
            ),
            enableConeFriction=int(bool(physics["enable_cone_friction"])),
            useSplitImpulse=int(bool(physics["use_split_impulse"])),
        )

        body_ids: dict[str, int] = {}
        mesh_material = metadata["fixture"]["mesh_material"]
        collision_shapes: dict[tuple[str, str, tuple[float, ...]], int] = {}
        for component in metadata["fixture"]["mesh_components"]:
            path = project_path(root, component["collision"]["path"])
            scale = tuple(float(value) for value in component["mesh_scale"])
            shape_key = (str(path), str(component["collision"]["sha256"]), scale)
            if shape_key not in collision_shapes:
                shape = int(
                    pb.createCollisionShape(
                        pb.GEOM_MESH,
                        fileName=str(path),
                        meshScale=list(scale),
                        flags=pb.GEOM_FORCE_CONCAVE_TRIMESH,
                    )
                )
                if shape < 0:
                    raise RuntimeError(f"failed to create fixture mesh: {component['id']}")
                collision_shapes[shape_key] = shape
            shape = collision_shapes[shape_key]
            body = int(
                pb.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=shape,
                    basePosition=[float(value) for value in component["base_position_m"]],
                    baseOrientation=[
                        float(value)
                        for value in component["base_orientation_quaternion_xyzw"]
                    ],
                )
            )
            body_ids[str(component["id"])] = body
            pb.changeDynamics(
                body,
                -1,
                lateralFriction=float(mesh_material["contact_friction"]),
                restitution=float(mesh_material["contact_restitution"]),
            )

        analytic_material = metadata["fixture"]["analytic_material"]
        analytic_shapes: dict[tuple[float, float, float], int] = {}
        for collider in metadata["fixture"]["analytic_colliders"]:
            half_extents = tuple(
                float(value) for value in collider["half_extents_m"]
            )
            if half_extents not in analytic_shapes:
                analytic_shapes[half_extents] = int(
                    pb.createCollisionShape(
                        pb.GEOM_BOX,
                        halfExtents=list(half_extents),
                    )
                )
            shape = analytic_shapes[half_extents]
            body = int(
                pb.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=shape,
                    basePosition=[float(value) for value in collider["position_m"]],
                )
            )
            body_ids[str(collider["id"])] = body
            pb.changeDynamics(
                body,
                -1,
                lateralFriction=float(analytic_material["contact_friction"]),
                restitution=float(analytic_material["contact_restitution"]),
            )

        radius = float(dynamic["radius_m"])
        material = dynamic["material"]
        ball_shape = pb.createCollisionShape(pb.GEOM_SPHERE, radius=radius)
        ball = int(
            pb.createMultiBody(
                baseMass=float(material["mass_kg"]),
                baseCollisionShapeIndex=ball_shape,
                basePosition=[float(value) for value in initial["position_m"]],
                baseOrientation=[
                    float(value)
                    for value in initial["orientation_quaternion_xyzw"]
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

        positions: list[list[float]] = []
        orientations: list[list[float]] = []
        linear_velocities: list[list[float]] = []
        angular_velocities: list[list[float]] = []
        touched: set[str] = set()
        contact_steps: dict[str, int] = {key: 0 for key in body_ids}
        minimum_contact_distance = 0.0
        path_length = 0.0
        previous_position = np.asarray(initial["position_m"], dtype=float)
        maximum_speed = 0.0
        maximum_energy = -math.inf
        mass = float(material["mass_kg"])
        inertia = 0.4 * mass * radius * radius
        gravity_z = abs(float(physics["gravity_m_s2"][2]))

        def observe() -> None:
            position, orientation = pb.getBasePositionAndOrientation(ball)
            linear, angular = pb.getBaseVelocity(ball)
            positions.append([float(value) for value in position])
            orientations.append([float(value) for value in orientation])
            linear_velocities.append([float(value) for value in linear])
            angular_velocities.append([float(value) for value in angular])

        observe()
        initial_linear = np.asarray(initial["linear_velocity_m_s"], dtype=float)
        initial_angular = np.asarray(initial["angular_velocity_rad_s"], dtype=float)
        initial_energy = (
            0.5 * float(np.dot(initial_linear, initial_linear))
            + 0.5 * inertia / mass * float(np.dot(initial_angular, initial_angular))
            + gravity_z * float(initial["position_m"][2])
        )
        maximum_energy = initial_energy
        for step in range(1, (frame_count - 1) * steps_per_frame + 1):
            pb.stepSimulation()
            position = np.asarray(pb.getBasePositionAndOrientation(ball)[0], dtype=float)
            linear, angular = pb.getBaseVelocity(ball)
            linear_array = np.asarray(linear, dtype=float)
            angular_array = np.asarray(angular, dtype=float)
            path_length += float(np.linalg.norm(position - previous_position))
            previous_position = position
            maximum_speed = max(maximum_speed, float(np.linalg.norm(linear_array)))
            energy = (
                0.5 * float(np.dot(linear_array, linear_array))
                + 0.5 * inertia / mass * float(np.dot(angular_array, angular_array))
                + gravity_z * float(position[2])
            )
            maximum_energy = max(maximum_energy, energy)
            for component_id, body in body_ids.items():
                contacts = pb.getContactPoints(ball, body)
                if contacts:
                    touched.add(component_id)
                    contact_steps[component_id] += 1
                    minimum_contact_distance = min(
                        minimum_contact_distance,
                        min(float(contact[8]) for contact in contacts),
                    )
            if step % steps_per_frame == 0:
                observe()
    finally:
        pb.disconnect()

    arrays = {
        "positions": np.asarray(positions, dtype=np.float64),
        "orientations_xyzw": np.asarray(orientations, dtype=np.float64),
        "linear_velocities": np.asarray(linear_velocities, dtype=np.float64),
        "angular_velocities": np.asarray(angular_velocities, dtype=np.float64),
        "times": np.arange(frame_count, dtype=np.float64) / output_fps,
        "object_ids": np.asarray([str(dynamic["object_id"])]),
    }
    quality = metadata["quality"]
    mesh_ids = {
        str(component["id"])
        for component in metadata["fixture"]["mesh_components"]
    }
    analytic_ids = {
        str(component["id"])
        for component in metadata["fixture"]["analytic_colliders"]
    }
    minimum_analytic_contacts = int(quality["minimum_analytic_contact_count"])
    if metadata.get("sweep", {}).get("kind") == "sweep":
        minimum_analytic_contacts = int(
            quality["sweep_minimum_analytic_contact_count"]
        )
    checks = {
        "finite_trajectory": all(np.isfinite(value).all() for value in arrays.values() if value.dtype.kind in "fc"),
        "frame_count": len(arrays["times"]) == frame_count,
        "minimum_path_length": path_length >= float(quality["minimum_path_length_m"]),
        "maximum_linear_speed": maximum_speed <= float(quality["maximum_linear_speed_m_s"]),
        "maximum_penetration": -minimum_contact_distance <= float(quality["maximum_penetration_m"]),
        "minimum_world_z": float(arrays["positions"][:, 2].min()) >= float(quality["minimum_world_z_m"]),
        "maximum_wall_normal_drift": float(np.abs(arrays["positions"][:, 1]).max()) <= float(quality["maximum_abs_wall_normal_y_m"]),
        "maximum_energy_gain": maximum_energy - initial_energy <= float(quality["maximum_energy_gain_j_per_kg"]),
        "required_mesh_contacts": set(quality["required_mesh_contact_ids"]) <= touched,
        "minimum_analytic_contacts": len(touched & analytic_ids) >= minimum_analytic_contacts,
    }
    audit = {
        "schema_version": AUDIT_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": {
            "path_length_m": round(path_length, 9),
            "maximum_linear_speed_m_s": round(maximum_speed, 9),
            "maximum_penetration_m": round(-minimum_contact_distance, 9),
            "minimum_world_z_m": round(float(arrays["positions"][:, 2].min()), 9),
            "maximum_abs_wall_normal_y_m": round(float(np.abs(arrays["positions"][:, 1]).max()), 9),
            "maximum_energy_gain_j_per_kg": round(maximum_energy - initial_energy, 9),
            "mesh_components_contacted": sorted(touched & mesh_ids),
            "analytic_components_contacted": sorted(touched & analytic_ids),
            "contact_steps": contact_steps,
        },
    }
    return arrays, audit


def derive_sweep_metadata(
    base: dict[str, Any],
    root: Path,
    sweep_config_path: Path,
    base_metadata_path: Path,
) -> list[dict[str, Any]]:
    """Derive the candidate with the release one-factor endpoint algorithm."""
    rules = load_json(sweep_config_path)
    axes = list(rules["axes"])
    if axes != ["mass_kg", "contact_friction", "contact_restitution"]:
        raise ValueError("candidate requires the three active release sweep axes")
    objects = base["physics"]["objects"]
    if len(objects) != 1 or objects[0]["object_id"] != "marble":
        raise ValueError("candidate sweep requires exactly one marble")
    if not base_metadata_path.is_file():
        raise FileNotFoundError(base_metadata_path)
    parent_binding = {
        "parent_scene_id": str(base["scene_id"]),
        "parent_metadata_path": project_relative(root, base_metadata_path),
        "parent_metadata_sha256": sha256(base_metadata_path),
    }
    base_material = objects[0]["material"]
    variants: list[dict[str, Any]] = []

    canonical = copy.deepcopy(base)
    canonical["scene_id"] = f"{base['scene_id']}__base"
    canonical["dataset_stage"] = "static_fixture_candidate_canonical_base"
    canonical["sweep"] = {
        "schema_version": rules["version"],
        "kind": "base",
        "mode": "one_factor_reference",
        "target_object_id": None,
        "target_object_index": None,
        "parameter": None,
        "axis": None,
        "value": None,
        **parent_binding,
        "source_schema_version": base["schema_version"],
        "resolved_object_physics": [
            {
                "object_id": "marble",
                "object_index": 0,
                "material": copy.deepcopy(base_material),
            }
        ],
        "config_path": project_relative(root, sweep_config_path),
        "config_sha256": sha256(sweep_config_path),
        "initial_state_policy": "copied_from_base_unchanged",
        "visual_policy": "copied_from_base_unchanged",
    }
    variants.append(canonical)

    for axis in axes:
        axis_rules = rules["axes"][axis]
        base_value = float(base_material[axis])
        mass_bounds = (
            [float(value) for value in base_material["mass_range_kg"]]
            if axis == "mass_kg"
            else None
        )
        schema_domain = axis_rules.get("schema_domains", {}).get(
            base["schema_version"]
        )
        values = _sweep_values(
            base_value,
            axis_rules,
            mass_bounds,
            axis,
            domain_override=schema_domain,
            endpoint_policy=rules["endpoint_policy"],
        )
        rounded_base = _round_value(base_value)
        base_level_index = values.index(rounded_base)
        if base_level_index != len(values) // 2:
            raise ValueError(f"{axis} does not preserve the base at the middle level")
        for level_index, value in enumerate(values):
            if level_index == base_level_index:
                continue
            derived = copy.deepcopy(base)
            derived["scene_id"] = (
                f"{base['scene_id']}__sweep_marble_{axis}_{level_index:02d}"
            )
            derived["dataset_stage"] = "static_fixture_candidate_physics_sweep"
            material = derived["physics"]["objects"][0]["material"]
            material[axis] = value
            derived["sweep"] = {
                "schema_version": rules["version"],
                "kind": "sweep",
                "mode": "one_factor",
                "target_object_id": "marble",
                "target_object_index": 0,
                "parameter": axis,
                "axis": axis,
                "level_index": level_index,
                "level_count": len(values),
                "value": value,
                **parent_binding,
                "base_value": rounded_base,
                "base_level_index": base_level_index,
                "object_field": axis_rules["object_field"],
                "overridden_field": f"physics.objects[0].material.{axis}",
                "source_schema_version": base["schema_version"],
                "resolved_object_physics": [
                    {
                        "object_id": "marble",
                        "object_index": 0,
                        "material": copy.deepcopy(material),
                    }
                ],
                "config_path": project_relative(root, sweep_config_path),
                "config_sha256": sha256(sweep_config_path),
                "initial_state_policy": "copied_from_base_unchanged",
                "visual_policy": "copied_from_base_unchanged",
                "endpoint_policy": copy.deepcopy(rules["endpoint_policy"]),
            }
            variants.append(derived)
    if len(variants) != 13:
        raise RuntimeError(f"candidate sweep must contain 13 records, got {len(variants)}")
    return variants


def _without_variant_fields(metadata: dict[str, Any], axis: str | None) -> dict[str, Any]:
    value = copy.deepcopy(metadata)
    value.pop("scene_id", None)
    value.pop("dataset_stage", None)
    value.pop("sweep", None)
    if axis is not None:
        value["physics"]["objects"][0]["material"].pop(axis)
    return value


def validate_one_factor_variants(
    base: dict[str, Any], variants: list[dict[str, Any]]
) -> None:
    canonical = variants[0]
    if canonical["sweep"]["kind"] != "base":
        raise ValueError("first candidate sweep record is not the canonical base")
    if _without_variant_fields(canonical, None) != _without_variant_fields(base, None):
        raise ValueError("canonical sweep base changed non-sweep metadata")
    seen: set[tuple[str, int]] = set()
    for derived in variants[1:]:
        sweep = derived["sweep"]
        axis = str(sweep["axis"])
        key = (axis, int(sweep["level_index"]))
        if key in seen:
            raise ValueError(f"duplicate candidate sweep level: {key}")
        seen.add(key)
        if _without_variant_fields(derived, axis) != _without_variant_fields(base, axis):
            raise ValueError(f"candidate sweep changed more than {axis}")
        changed = derived["physics"]["objects"][0]["material"][axis]
        if changed != sweep["value"] or changed == base["physics"]["objects"][0]["material"][axis]:
            raise ValueError(f"candidate sweep did not change exactly {axis}")


def write_trajectory(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def generate_sweep(
    root: Path,
    output: Path,
    base: dict[str, Any],
    base_metadata_path: Path,
    sweep_config_path: Path,
) -> dict[str, Any]:
    variants = derive_sweep_metadata(
        base, root, sweep_config_path, base_metadata_path
    )
    validate_one_factor_variants(base, variants)
    records = []
    for metadata in variants:
        sweep = metadata["sweep"]
        label = "base" if sweep["kind"] == "base" else (
            f"{sweep['axis']}_{int(sweep['level_index']):02d}"
        )
        variant_output = output / label
        metadata_path = variant_output / "metadata.json"
        trajectory_path = variant_output / "trajectory.npz"
        audit_path = variant_output / "audit.json"
        write_json(metadata_path, metadata)
        arrays, audit = simulate(root, metadata)
        write_trajectory(trajectory_path, arrays)
        write_json(audit_path, audit)
        if not audit["passed"]:
            raise RuntimeError(f"candidate sweep audit failed for {label}: {audit}")
        records.append(
            {
                "scene_id": metadata["scene_id"],
                "kind": sweep["kind"],
                "axis": sweep.get("axis"),
                "level_index": sweep.get("level_index"),
                "value": sweep.get("value"),
                "metadata_path": project_relative(root, metadata_path),
                "metadata_sha256": sha256(metadata_path),
                "trajectory_path": project_relative(root, trajectory_path),
                "trajectory_sha256": sha256(trajectory_path),
                "audit_path": project_relative(root, audit_path),
                "audit_sha256": sha256(audit_path),
            }
        )
    manifest = {
        "schema_version": "physweep_static_fixture_candidate_sweep_manifest_v1",
        "candidate_id": base["scene_id"],
        "admission": copy.deepcopy(base["admission"]),
        "source_base": {
            "path": project_relative(root, base_metadata_path),
            "sha256": sha256(base_metadata_path),
        },
        "sweep_config": {
            "path": project_relative(root, sweep_config_path),
            "sha256": sha256(sweep_config_path),
        },
        "sample_count": len(records),
        "success_count": len(records),
        "failure_count": 0,
        "records": records,
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--generate-sweep", action="store_true")
    parser.add_argument("--sweep-config", type=Path, default=SWEEP_CONFIG_PATH)
    parser.add_argument("--sweep-output", type=Path, default=SWEEP_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config_path = project_path(root, args.config)
    output = project_path(root, args.output)
    config = load_json(config_path)
    source_paths = validate_config(root, config)
    collision = materialize_collision_meshes(root, output, config, source_paths)
    metadata = build_metadata(root, config_path, config, collision)
    base_metadata_path = output / "metadata.json"
    write_json(base_metadata_path, metadata)
    arrays, audit = simulate(root, metadata)
    trajectory_path = output / "trajectory.npz"
    write_trajectory(trajectory_path, arrays)
    write_json(output / "audit.json", audit)
    record = {
        "schema_version": "physweep_static_fixture_candidate_record_v1",
        "metadata": {
            "path": project_relative(root, output / "metadata.json"),
            "sha256": sha256(output / "metadata.json"),
        },
        "trajectory": {
            "path": project_relative(root, trajectory_path),
            "sha256": sha256(trajectory_path),
        },
        "audit": {
            "path": project_relative(root, output / "audit.json"),
            "sha256": sha256(output / "audit.json"),
            "passed": bool(audit["passed"]),
        },
    }
    write_json(output / "simulation_record.json", record)
    if not audit["passed"]:
        raise RuntimeError(f"marble-run candidate audit failed: {audit['checks']}")
    sweep_manifest = None
    if args.generate_sweep:
        sweep_manifest = generate_sweep(
            root,
            project_path(root, args.sweep_output),
            metadata,
            base_metadata_path,
            project_path(root, args.sweep_config),
        )
    print(
        json.dumps(
            {
                "output": project_relative(root, output),
                "sweep_sample_count": (
                    sweep_manifest["sample_count"] if sweep_manifest else 0
                ),
                **audit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
