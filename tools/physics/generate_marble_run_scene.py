#!/usr/bin/env python3
"""Generate and audit a formal one-marble passive track scene."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from tools.physics.generate_marble_run_candidate import (
    build_metadata as build_candidate_metadata,
    materialize_collision_meshes,
    simulate_marble_run_physics,
    validate_config as validate_candidate_config,
)
from tools.dataset_contract.immutable_scene_contract import freeze_metadata, write_simulation_record
from tools.dataset_contract.object_identity_contract import attach_object_identity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("configs/marble_run_backend.json")
SPECIALIZED_REGISTRY = Path("configs/specialized_scene_backends.json")
RENDERER = Path("tools/rendering/render_marble_run_scene.py")
CANDIDATE_PHYSICS = Path("tools/physics/generate_marble_run_candidate.py")
SCHEMA_VERSION = "physweep_marble_run_scene_v1"
AUDIT_VERSION = "physweep_marble_run_audit_v1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
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
        raise ValueError(f"marble-run path is outside the project: {path}") from exc


def load_backend(
    root: Path, config_path: Path
) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Path]]:
    backend = load_json(config_path)
    if backend.get("schema_version") != "physweep_marble_run_backend_v1":
        raise ValueError("unsupported marble-run backend config")
    admission = backend.get("admission", {})
    if (
        admission.get("status") != "approved_specialized"
        or admission.get("release_enabled") is not True
        or int(admission.get("dynamic_object_count", -1)) != 1
        or admission.get("active_mechanisms_supported") is not False
    ):
        raise ValueError("marble-run backend admission is not one-object passive")

    candidate_binding = backend["candidate_config"]
    candidate_path = project_path(root, candidate_binding["path"])
    candidate_path.relative_to(root)
    if sha256(candidate_path) != str(candidate_binding["sha256"]):
        raise ValueError("marble-run candidate config binding changed")
    candidate = load_json(candidate_path)
    source_paths = validate_candidate_config(root, candidate)

    offsets: list[float] = []
    profiles = backend.get("profiles", {})
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("marble-run backend contains no profiles")
    for profile, rules in profiles.items():
        values = [float(value) for value in rules["initial_track_offsets_m"]]
        if not values or len(values) != len(set(values)):
            raise ValueError(f"marble-run profile has invalid offsets: {profile}")
        if any(abs(value) > 0.015 + 1.0e-12 or abs(value) < 0.003 - 1.0e-12 for value in values):
            raise ValueError(f"marble-run profile leaves the audited offset set: {profile}")
        offsets.extend(values)
    if len(offsets) != len(set(offsets)):
        raise ValueError("marble-run release offsets belong to multiple profiles")

    inherited_physics = set(backend["physics"]["candidate_fields_inherited"])
    required_physics = {
        "output_fps",
        "simulation_hz",
        "gravity_m_s2",
        "solver_iterations",
        "deterministic_overlapping_pairs",
        "restitution_velocity_threshold_m_s",
        "enable_cone_friction",
        "use_split_impulse",
    }
    if inherited_physics != required_physics:
        raise ValueError("marble-run formal physics inheritance is incomplete")
    if float(backend["physics"]["duration_s"]) != 4.0:
        raise ValueError("marble-run formal duration must remain four seconds")
    inherited_visuals = set(backend["render"]["visual_fields_inherited"])
    if inherited_visuals != {"world_color_rgb", "context", "lights"}:
        raise ValueError("marble-run visual inheritance is incomplete")
    if backend["render"]["engine"] != "BLENDER_EEVEE":
        raise ValueError("marble-run release renderer must use Eevee")
    if [int(value) for value in backend["render"]["resolution"]] != [1280, 720]:
        raise ValueError("marble-run release resolution changed")
    if int(backend["render"]["samples"]) != 16:
        raise ValueError("marble-run release sample count changed")
    return backend, candidate_path, candidate, source_paths


def initial_state(
    seed: int,
    profile: str,
    backend: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], float]:
    try:
        offsets = backend["profiles"][profile]["initial_track_offsets_m"]
    except KeyError as exc:
        raise ValueError(f"unsupported marble-run profile: {profile}") from exc
    rng = random.Random(f"marble-run-initial:{profile}:{int(seed)}")
    offset = float(rng.choice(offsets))
    state = copy.deepcopy(candidate["dynamic_object"]["initial_state"])
    state["position_m"][0] = round(float(state["position_m"][0]) + offset, 12)
    return state, offset


def marble_run_camera(
    seed: int,
    profile: str,
    backend: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    rules = backend["camera"]
    source = candidate["render"]["camera"]
    rng = random.Random(f"marble-run-camera:{profile}:{int(seed)}")
    horizontal = float(rng.choice(rules["horizontal_offsets_m"]))
    vertical = float(rng.choice(rules["vertical_offsets_m"]))
    position = [float(value) for value in source["position_m"]]
    target = [float(value) for value in source["target_m"]]
    position[0] += horizontal
    target[0] += horizontal
    position[2] += vertical
    target[2] += vertical
    return {
        "seed": int(seed),
        "mode": "marble_run_front_oblique",
        "position_m": [round(value, 9) for value in position],
        "target_m": [round(value, 9) for value in target],
        "focal_length_mm": float(source["focal_length_mm"]),
        "sensor_width_mm": float(rules["sensor_width_mm"]),
    }


def _binding(root: Path, path: Path) -> dict[str, str]:
    resolved = project_path(root, path)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": project_relative(root, resolved), "sha256": sha256(resolved)}


def build_metadata(
    root: Path,
    output: Path,
    config_path: Path,
    backend: dict[str, Any],
    candidate_path: Path,
    candidate: dict[str, Any],
    collision: dict[str, dict[str, Any]],
    seed: int,
    profile: str,
    scene_id: str,
    resolution: list[int] | None = None,
    samples: int | None = None,
) -> dict[str, Any]:
    if profile not in backend["profiles"]:
        raise ValueError(f"unsupported marble-run profile: {profile}")
    candidate_metadata = build_candidate_metadata(
        root, candidate_path, candidate, collision
    )
    dynamic = copy.deepcopy(candidate["dynamic_object"])
    state, release_offset = initial_state(seed, profile, backend, candidate)
    dynamic_record = {
        "object_id": str(dynamic["object_id"]),
        "semantic_type": "marble",
        "body_model": "rigid_body",
        "is_dynamic": True,
        "collision_proxy": {
            "type": "sphere",
            "radius_m": float(dynamic["radius_m"]),
        },
        "material": copy.deepcopy(dynamic["material"]),
        "initial_state": state,
        "visual": {
            "shape": "sphere",
            "radius_m": float(dynamic["radius_m"]),
            "color_rgba": copy.deepcopy(dynamic["color_rgba"]),
        },
    }
    candidate_physics = candidate["physics"]
    duration = float(backend["physics"]["duration_s"])
    output_fps = int(candidate_physics["output_fps"])
    frame_count = int(round(duration * output_fps)) + 1
    trajectory_path = output / "trajectory.npz"
    audit_path = output / "audit.json"
    simulation_record_path = output / "simulation_record.json"
    render_source = candidate["render"]
    render_rules = backend["render"]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "scene_id": scene_id,
        "seed": int(seed),
        "dataset_stage": "marble_run_formal_base_candidate",
        "admission": copy.deepcopy(backend["admission"]),
        "semantics": {
            "scene_family": "marble_run",
            "profile": profile,
            "description": backend["profiles"][profile]["description"],
            "dynamic_object_count": 1,
            "active_mechanism_count": 0,
        },
        "source": copy.deepcopy(candidate["source"]),
        "attribution": copy.deepcopy(backend["attribution"]),
        "implementation": {
            "generator": _binding(root, Path(__file__)),
            "renderer": _binding(root, RENDERER),
            "candidate_physics": _binding(root, CANDIDATE_PHYSICS),
            "specialized_registry": _binding(root, SPECIALIZED_REGISTRY),
        },
        "simulation": {
            "time": {
                "duration_s": duration,
                "output_fps": output_fps,
                "simulation_hz": int(candidate_physics["simulation_hz"]),
                "frame_count": frame_count,
            },
            "world": {
                "gravity_m_s2": copy.deepcopy(candidate_physics["gravity_m_s2"])
            },
            "objects": [dynamic_record],
        },
        "physics": {
            "backend": "pybullet_marble_run_v1",
            "backend_config": _binding(root, config_path),
            "candidate_config": _binding(root, candidate_path),
            "profile": profile,
            "initial_track_offset_m": release_offset,
            "engine": {
                key: copy.deepcopy(candidate_physics[key])
                for key in backend["physics"]["candidate_fields_inherited"]
                if key not in {"output_fps", "simulation_hz", "gravity_m_s2"}
            },
            "fixture": copy.deepcopy(candidate_metadata["fixture"]),
            "quality": copy.deepcopy(candidate["quality"]),
            "trajectory_path": project_relative(root, trajectory_path),
            "audit_path": project_relative(root, audit_path),
            "simulation_record_path": project_relative(
                root, simulation_record_path
            ),
        },
        "camera": marble_run_camera(seed, profile, backend, candidate),
        "render": {
            "engine": str(render_rules["engine"]),
            "resolution": [
                int(value)
                for value in (resolution or render_rules["resolution"])
            ],
            "samples": int(samples if samples is not None else render_rules["samples"]),
            "world_color_rgb": copy.deepcopy(render_source["world_color_rgb"]),
            "context": copy.deepcopy(render_source["context"]),
            "lights": copy.deepcopy(render_source["lights"]),
            "inspection_frame_dir": project_relative(
                root, output / "inspection_frames"
            ),
            "video_path": project_relative(root, output / f"{scene_id}.mp4"),
        },
    }
    attach_object_identity(
        metadata,
        trajectory_path=project_relative(root, trajectory_path),
        mask_path=project_relative(root, output / "masks" / scene_id),
    )
    return metadata


def _validate_metadata_files(root: Path, metadata: dict[str, Any]) -> None:
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported marble-run metadata")
    if metadata.get("admission", {}).get("release_enabled") is not True:
        raise ValueError("marble-run formal metadata is not release enabled")
    for label, binding in (
        ("backend config", metadata["physics"]["backend_config"]),
        ("candidate config", metadata["physics"]["candidate_config"]),
        ("generator", metadata["implementation"]["generator"]),
        ("renderer", metadata["implementation"]["renderer"]),
        ("candidate physics", metadata["implementation"]["candidate_physics"]),
        ("specialized registry", metadata["implementation"]["specialized_registry"]),
    ):
        path = project_path(root, binding["path"])
        path.relative_to(root)
        if not path.is_file() or sha256(path) != str(binding["sha256"]):
            raise ValueError(f"marble-run {label} binding changed")
    backend = load_json(project_path(root, metadata["physics"]["backend_config"]["path"]))
    if backend.get("schema_version") != "physweep_marble_run_backend_v1":
        raise ValueError("marble-run metadata binds the wrong backend")
    candidate_binding = metadata["physics"]["candidate_config"]
    if candidate_binding != backend["candidate_config"]:
        raise ValueError("marble-run candidate binding differs from the backend")
    if str(metadata["physics"]["profile"]) not in backend["profiles"]:
        raise ValueError("marble-run metadata uses an undeclared profile")
    for component in metadata["physics"]["fixture"]["mesh_components"]:
        source = project_path(root, component["source_path"])
        collision = project_path(root, component["collision"]["path"])
        if sha256(source) != str(component["source_sha256"]):
            raise ValueError(f"marble-run source hash changed: {component['id']}")
        if sha256(collision) != str(component["collision"]["sha256"]):
            raise ValueError(f"marble-run collision hash changed: {component['id']}")


def _physics_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    time_binding = metadata["simulation"]["time"]
    world = metadata["simulation"]["world"]
    source = metadata["simulation"]["objects"][0]
    dynamic = {
        "object_id": str(source["object_id"]),
        "shape": "sphere",
        "radius_m": float(source["collision_proxy"]["radius_m"]),
        "material": copy.deepcopy(source["material"]),
        "initial_state": copy.deepcopy(source["initial_state"]),
        "color_rgba": copy.deepcopy(source["visual"]["color_rgba"]),
    }
    engine = metadata["physics"]["engine"]
    payload = {
        "fixture": copy.deepcopy(metadata["physics"]["fixture"]),
        "physics": {
            "duration_s": float(time_binding["duration_s"]),
            "output_fps": int(time_binding["output_fps"]),
            "simulation_hz": int(time_binding["simulation_hz"]),
            "frame_count": int(time_binding["frame_count"]),
            "gravity_m_s2": copy.deepcopy(world["gravity_m_s2"]),
            **copy.deepcopy(engine),
            "objects": [dynamic],
        },
        "quality": copy.deepcopy(metadata["physics"]["quality"]),
    }
    if "sweep" in metadata:
        payload["sweep"] = copy.deepcopy(metadata["sweep"])
    return payload


def simulate(
    root: Path, metadata: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _validate_metadata_files(root, metadata)
    raw, audit = simulate_marble_run_physics(root, _physics_payload(metadata))
    positions = raw["positions"][:, None, :]
    orientations = raw["orientations_xyzw"][:, None, :]
    linear = raw["linear_velocities"][:, None, :]
    angular = raw["angular_velocities"][:, None, :]
    arrays = {
        "time_s": raw["times"],
        "position_m": positions,
        "quaternion_xyzw": orientations,
        "linear_velocity_m_s": linear,
        "angular_velocity_rad_s": angular,
        "contact_count": raw["contact_count"],
        "runtime_material": raw["runtime_material"],
        "runtime_inertia_diagonal_kg_m2": raw[
            "runtime_inertia_diagonal_kg_m2"
        ],
        "marble__position_m": positions[:, 0],
        "marble__quaternion_wxyz": orientations[:, 0, [3, 0, 1, 2]],
        "marble__linear_velocity_m_s": linear[:, 0],
        "marble__angular_velocity_rad_s": angular[:, 0],
    }
    audit = copy.deepcopy(audit)
    audit["schema_version"] = AUDIT_VERSION
    audit["profile"] = str(metadata["semantics"]["profile"])
    audit["initial_track_offset_m"] = float(
        metadata["physics"]["initial_track_offset_m"]
    )
    return arrays, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--profile", default="early_release_chain")
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
        raise ValueError("marble-run output must be a project subdirectory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"marble-run output is not empty: {output}")
    backend, candidate_path, candidate, source_paths = load_backend(
        root, config_path
    )
    scene_id = args.scene_id or f"marble_run_{args.profile}_{int(args.seed):08d}"
    output.mkdir(parents=True, exist_ok=True)
    collision = materialize_collision_meshes(
        root, output, candidate, source_paths
    )
    metadata_path = output / "metadata.json"
    metadata = build_metadata(
        root,
        output,
        config_path,
        backend,
        candidate_path,
        candidate,
        collision,
        int(args.seed),
        str(args.profile),
        scene_id,
        args.resolution,
        args.samples,
    )
    metadata = freeze_metadata(metadata_path, metadata)
    arrays, audit = simulate(root, metadata)
    if not audit["passed"]:
        raise RuntimeError(f"marble-run physics audit failed: {audit}")
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
    print(
        json.dumps(
            {"metadata": project_relative(root, metadata_path), "audit": audit},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
