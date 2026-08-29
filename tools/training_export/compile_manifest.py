#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from tools.core.hashing import sha256_file as _sha256
from tools.core.paths import project_relative_path
from tools.core.sweep_values import SWEEP_LEVEL_COUNT, sweep_group_size
from tools.training_export.gt_scene_input import (
    DEFAULT_ENVIRONMENT_POINTS,
    DEFAULT_OBJECT_POINTS,
    MODEL_SCENE_SCHEMA,
    compile_model_scene_condition,
    inspect_model_scene_condition,
)
from tools.training_export.one_object_prompt_contract import (
    PROMPT_TEMPLATE_VERSION,
    build_training_prompt,
)
from tools.training_export.coordinate_frames import transform_world_vector_to_camera
from tools.dataset_contract.schema import (
    MANIFEST_SCHEMA,
    SAMPLE_SCHEMA,
    SWEEP_AXES,
    validate_manifest,
)


def _split_for_scene(scene_id: str) -> str:
    bucket = int(hashlib.sha256(scene_id.encode("utf-8")).hexdigest()[:8], 16) % 1000
    if bucket < 900:
        return "train"
    if bucket < 950:
        return "validation"
    return "test"


def _scene_id_from_parent(parent: str) -> str:
    path = Path(parent)
    if path.name != "metadata.json":
        raise ValueError(f"unexpected parent metadata path: {parent}")
    return path.parent.name


def _find_object(metadata: dict, object_id: str) -> dict:
    for item in metadata["simulation"]["objects"]:
        if item.get("object_id") == object_id:
            return item
    raise ValueError(f"object {object_id} is missing from {metadata['scene_id']}")


def deduplicate_base_records(
    records: list[dict],
    base_level_indices: dict[str, int],
    preferred_axis: str = "mass_kg",
) -> list[tuple[dict, bool]]:
    if set(base_level_indices) != set(SWEEP_AXES):
        raise ValueError("every sweep axis must declare a base level")
    base_records = [
        record
        for record in records
        if int(record["level_index"]) == int(base_level_indices[record["axis"]])
    ]
    base_axes = {record["axis"] for record in base_records}
    if base_axes not in ({preferred_axis}, set(SWEEP_AXES)):
        raise ValueError(
            "sweep records must contain one canonical base or one base per axis"
        )
    if len(base_records) != len(base_axes):
        raise ValueError("a sweep axis contains duplicate base records")
    canonical = next(record for record in base_records if record["axis"] == preferred_axis)
    nonbase = [
        record
        for record in records
        if int(record["level_index"]) != int(base_level_indices[record["axis"]])
    ]
    ordered = sorted(
        nonbase,
        key=lambda item: (SWEEP_AXES.index(item["axis"]), int(item["level_index"])),
    )
    return [(canonical, True), *[(record, False) for record in ordered]]


def _base_level_indices(records: list[dict], root: Path) -> dict[str, int]:
    result = {}
    for axis in SWEEP_AXES:
        source = next(record for record in records if record["axis"] == axis)
        metadata = json.loads((root / source["path"]).read_text(encoding="utf-8"))
        sweep = metadata["sweep"]
        if sweep["axis"] != axis:
            raise ValueError(f"sweep metadata axis mismatch: {source['path']}")
        base_level = int(sweep["base_level_index"])
        if not 0 <= base_level < int(sweep["level_count"]):
            raise ValueError(f"invalid base level index: {source['path']}")
        result[axis] = base_level
    return result


def _camera_rotation(camera: dict) -> np.ndarray:
    basis = np.eye(3, dtype=np.float64)
    return np.column_stack(
        [transform_world_vector_to_camera(vector, camera) for vector in basis]
    )


def _quaternion_matrix_wxyz(quaternion) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("object orientation quaternion is invalid")
    w, x, y, z = value / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _assert_vector_close(actual, expected, context: str, atol: float = 1e-7) -> None:
    if not np.allclose(
        np.asarray(actual, dtype=np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=1e-7,
        atol=atol,
    ):
        raise ValueError(f"{context} differs between metadata and trajectory")


def _assert_shared_t0(base_metadata: dict, target_metadata: dict, object_id: str) -> None:
    base_object = _find_object(base_metadata, object_id)
    target_object = _find_object(target_metadata, object_id)
    for key in (
        "position_m",
        "orientation_quaternion_wxyz",
        "linear_velocity_m_s",
        "angular_velocity_rad_s",
    ):
        _assert_vector_close(
            target_object["initial_state"][key],
            base_object["initial_state"][key],
            f"shared t0 {key}",
        )
    base_camera = base_metadata["visualization"]["camera"]
    target_camera = target_metadata["visualization"]["camera"]
    for key in ("position_m", "target_m"):
        _assert_vector_close(
            target_camera[key], base_camera[key], f"shared camera {key}"
        )
    for key in ("focal_length_mm", "sensor_width_mm"):
        if not np.isclose(
            float(target_camera[key]),
            float(base_camera[key]),
            rtol=1e-9,
            atol=1e-9,
        ):
            raise ValueError(f"shared camera {key} changes inside a sweep group")


def _runtime_physics(
    metadata: dict,
    project_root: Path,
    object_id: str,
) -> tuple[dict, Path, str]:
    trajectory_record = metadata["trajectory"]
    trajectory_path = (project_root / trajectory_record["path"]).resolve()
    if (
        not trajectory_path.is_relative_to(project_root.resolve())
        or not trajectory_path.is_file()
    ):
        raise FileNotFoundError(f"invalid target trajectory: {trajectory_path}")
    trajectory_sha256 = _sha256(trajectory_path)
    expected_sha256 = str(trajectory_record.get("sha256", ""))
    if expected_sha256 and trajectory_sha256 != expected_sha256:
        raise ValueError(f"trajectory hash mismatch: {trajectory_path}")

    prefix = f"{object_id}__"
    required = {
        f"{prefix}runtime_dynamics",
        f"{prefix}runtime_inertia_diagonal_kg_m2",
        f"{prefix}position_m",
        f"{prefix}quaternion_wxyz",
        f"{prefix}linear_velocity_m_s",
        f"{prefix}angular_velocity_rad_s",
    }
    with np.load(trajectory_path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(
                f"trajectory is missing runtime fields: {', '.join(missing)}"
            )
        dynamics = archive[f"{prefix}runtime_dynamics"].astype(np.float64)
        inertia_diagonal = archive[
            f"{prefix}runtime_inertia_diagonal_kg_m2"
        ].astype(np.float64)
        position = archive[f"{prefix}position_m"][0].astype(np.float64)
        quaternion = archive[f"{prefix}quaternion_wxyz"][0].astype(np.float64)
        linear_velocity = archive[f"{prefix}linear_velocity_m_s"][0].astype(
            np.float64
        )
        angular_velocity = archive[f"{prefix}angular_velocity_rad_s"][0].astype(
            np.float64
        )
    if dynamics.shape != (5,) or inertia_diagonal.shape != (3,):
        raise ValueError("trajectory runtime dynamics have unexpected shapes")
    if not np.isfinite(dynamics).all() or not np.isfinite(inertia_diagonal).all():
        raise ValueError("trajectory runtime dynamics contain non-finite values")
    if np.any(inertia_diagonal <= 0.0):
        raise ValueError("trajectory runtime inertia must be positive")

    dynamic_object = _find_object(metadata, object_id)
    material = dynamic_object["material"]
    expected_dynamics = np.asarray(
        [
            material["mass_kg"],
            material["contact_friction"],
            material["contact_restitution"],
            material.get("rolling_friction", 0.0),
            material.get("spinning_friction", 0.0),
        ],
        dtype=np.float64,
    )
    _assert_vector_close(dynamics, expected_dynamics, "runtime dynamics")
    initial_state = dynamic_object["initial_state"]
    _assert_vector_close(position, initial_state["position_m"], "initial position")
    quaternion_error = min(
        float(np.linalg.norm(quaternion - np.asarray(initial_state["orientation_quaternion_wxyz"]))),
        float(np.linalg.norm(quaternion + np.asarray(initial_state["orientation_quaternion_wxyz"]))),
    )
    if quaternion_error > 1e-7:
        raise ValueError("initial orientation differs between metadata and trajectory")
    _assert_vector_close(
        linear_velocity,
        initial_state["linear_velocity_m_s"],
        "initial linear velocity",
    )
    _assert_vector_close(
        angular_velocity,
        initial_state["angular_velocity_rad_s"],
        "initial angular velocity",
    )

    camera = metadata["visualization"]["camera"]
    camera_rotation = _camera_rotation(camera)
    object_rotation = _quaternion_matrix_wxyz(quaternion)
    camera_from_object = camera_rotation @ object_rotation
    inertia_camera = (
        camera_from_object
        @ np.diag(inertia_diagonal)
        @ camera_from_object.T
    )
    inertia_camera = (inertia_camera + inertia_camera.T) * 0.5
    linear_damping = float(material.get("linear_damping", 0.0))
    angular_damping = float(material.get("angular_damping", 0.0))
    if linear_damping < 0.0 or angular_damping < 0.0:
        raise ValueError("object damping must be non-negative")

    physics = {
        "source": "simulation_gt",
        "coordinate_frame": "camera_right_up_forward",
        "object": {
            "object_id": object_id,
            "mass_kg": float(dynamics[0]),
            "inertia_tensor_camera_kg_m2": inertia_camera.tolist(),
            "friction": float(dynamics[1]),
            "restitution": float(dynamics[2]),
            "rolling_friction": float(dynamics[3]),
            "spinning_friction": float(dynamics[4]),
            "linear_damping": linear_damping,
            "angular_damping": angular_damping,
            "initial_state": {
                "linear_velocity_camera_m_s": [
                    float(value)
                    for value in transform_world_vector_to_camera(
                        linear_velocity, camera
                    )
                ],
                "angular_velocity_camera_rad_s": [
                    float(value)
                    for value in transform_world_vector_to_camera(
                        angular_velocity, camera, axial=True
                    )
                ],
            },
        },
        "world": {
            "gravity_camera_m_s2": [
                float(value)
                for value in transform_world_vector_to_camera(
                    metadata["simulation"]["world"]["gravity_m_s2"], camera
                )
            ]
        },
    }
    return physics, trajectory_path, trajectory_sha256


def _record(
    source: dict,
    canonical_base: bool,
    base_bound_metadata: dict,
    base_bound_metadata_path: Path,
    target_bound_metadata: dict,
    target_bound_metadata_path: Path,
    source_sweep_metadata_path: Path,
    first_frame: Path,
    scene_condition: Path,
    target_video: Path,
    project_root: Path,
    base_scene_id: str,
) -> dict:
    object_id = str(source["target_object_id"])
    axis = None if canonical_base else str(source["axis"])
    _assert_shared_t0(base_bound_metadata, target_bound_metadata, object_id)
    physics, trajectory_path, trajectory_sha256 = _runtime_physics(
        target_bound_metadata,
        project_root,
        object_id,
    )
    caption = build_training_prompt(base_bound_metadata, object_id)
    time = target_bound_metadata["simulation"]["time"]
    return {
        "schema": SAMPLE_SCHEMA,
        "sample_id": str(source["scene_id"]),
        "base_scene_id": base_scene_id,
        "split": _split_for_scene(base_scene_id),
        "conditioning": {
            "first_frame": project_relative_path(project_root, first_frame),
            "scene": project_relative_path(project_root, scene_condition),
            "scene_source": "simulation_gt",
            "text": caption,
            "physics": physics,
        },
        "target": {
            "video": project_relative_path(project_root, target_video),
            "metadata": project_relative_path(
                project_root, target_bound_metadata_path
            ),
            "duration_s": float(time["duration_s"]),
            "fps": int(time["output_fps"]),
        },
        "sweep": {
            "mode": "base" if canonical_base else "one_factor",
            "axis": axis,
            "level_index": int(source["level_index"]),
            "level_count": SWEEP_LEVEL_COUNT,
            "base_level_index": int(source["base_level_index"]),
            "base_level_indices": {
                key: int(value)
                for key, value in source["base_level_indices"].items()
            },
            "source_axis": str(source["axis"]),
            "source_value": float(source["value"]),
        },
        "provenance": {
            "base_bound_metadata": project_relative_path(
                project_root, base_bound_metadata_path
            ),
            "base_source_metadata": str(source["parent"]),
            "sweep_source_metadata": project_relative_path(
                project_root, source_sweep_metadata_path
            ),
            "trajectory": project_relative_path(project_root, trajectory_path),
            "trajectory_sha256": trajectory_sha256,
        },
    }


def compile_manifest(args) -> dict:
    root = args.project_root.resolve()
    sweep_manifest_path = (root / args.sweep_manifest).resolve()
    base_render = (root / args.base_render_root).resolve()
    sweep_render = (root / args.sweep_render_root).resolve()
    output_root = (root / args.output_root).resolve()
    scene_source_root = (root / args.scene_source_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scene_root = output_root / "scenes"
    scene_root.mkdir(parents=True, exist_ok=True)

    sweep_manifest = json.loads(sweep_manifest_path.read_text(encoding="utf-8"))
    config_path = (root / sweep_manifest["config"]["path"]).resolve()
    sweep_config = json.loads(config_path.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for source in sweep_manifest["records"]:
        grouped[source["parent"]].append(source)
    parents = sorted(grouped)
    if args.limit_base_scenes is not None:
        parents = parents[: args.limit_base_scenes]

    manifest_path = output_root / "manifest.jsonl"
    temporary_path = output_root / "manifest.jsonl.tmp"
    sample_count = 0
    split_counts = Counter()
    mode_counts = Counter()
    base_level_counts = {axis: Counter() for axis in SWEEP_AXES}
    scene_reports = []
    with temporary_path.open("w", encoding="utf-8") as handle:
        for parent in parents:
            scene_id = _scene_id_from_parent(parent)
            base_level_indices = _base_level_indices(grouped[parent], root)
            for axis, level_index in base_level_indices.items():
                base_level_counts[axis][level_index] += 1
            annotated_records = []
            for item in grouped[parent]:
                source = dict(item)
                source["base_level_index"] = base_level_indices[source["axis"]]
                source["base_level_indices"] = dict(base_level_indices)
                annotated_records.append(source)
            unique_records = deduplicate_base_records(
                annotated_records, base_level_indices, args.canonical_base_axis
            )
            canonical_source = next(
                source for source, canonical in unique_records if canonical
            )
            bound_metadata_path = base_render / "generic" / "metadata" / f"{scene_id}.json"
            first_frame = base_render / "generic" / "frames" / scene_id / "frame_0001.png"
            if not bound_metadata_path.is_file() or not first_frame.is_file():
                canonical_scene_id = canonical_source["scene_id"]
                bound_metadata_path = (
                    sweep_render / "bound" / "metadata" / f"{canonical_scene_id}.json"
                )
                first_frame = (
                    sweep_render
                    / "bound"
                    / "frames"
                    / canonical_scene_id
                    / "frame_0001.png"
                )
            bound_metadata = json.loads(bound_metadata_path.read_text(encoding="utf-8"))
            scene_condition = scene_root / f"{scene_id}.npz"
            source_scene = scene_source_root / f"{scene_id}.npz"
            if not source_scene.is_file():
                raise FileNotFoundError(
                    f"missing exact GT source scene: {source_scene}; run "
                    "build_gt_training_scenes.py first"
                )
            rebuild_scene = args.force_scenes or not scene_condition.is_file()
            if not rebuild_scene:
                try:
                    scene_report = inspect_model_scene_condition(scene_condition)
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    rebuild_scene = True
            if rebuild_scene:
                compile_model_scene_condition(
                    source_scene,
                    scene_condition,
                    seed=args.seed,
                    object_points=args.object_points,
                    environment_points=args.environment_points,
                    bound_metadata=bound_metadata,
                )
                scene_report = inspect_model_scene_condition(scene_condition)
            if scene_report["scene_id"] != scene_id:
                raise ValueError(
                    f"scene id mismatch for {scene_condition}: "
                    f"{scene_report['scene_id']} != {scene_id}"
                )
            if scene_report["object_point_count"] != args.object_points:
                raise ValueError(f"object point quota mismatch: {scene_id}")
            if scene_report["environment_point_count"] != args.environment_points:
                raise ValueError(f"environment point quota mismatch: {scene_id}")
            scene_reports.append(
                {
                    "scene_id": scene_id,
                    "path": project_relative_path(root, scene_condition),
                    "sha256": _sha256(scene_condition),
                    "source_metadata": project_relative_path(
                        root, bound_metadata_path
                    ),
                    **scene_report,
                }
            )

            for source, canonical_base in unique_records:
                source_sweep_metadata_path = (root / source["path"]).resolve()
                target_bound_metadata_path = (
                    sweep_render
                    / "bound"
                    / "metadata"
                    / f"{source['scene_id']}.json"
                )
                target_video = (
                    sweep_render / "bound" / "videos" / f"{source['scene_id']}.mp4"
                )
                if not source_sweep_metadata_path.is_file():
                    raise FileNotFoundError(source_sweep_metadata_path)
                if not target_bound_metadata_path.is_file():
                    raise FileNotFoundError(target_bound_metadata_path)
                target_bound_metadata = json.loads(
                    target_bound_metadata_path.read_text(encoding="utf-8")
                )
                if target_bound_metadata.get("scene_id") != source["scene_id"]:
                    raise ValueError(
                        f"target bound scene id mismatch: {source['scene_id']}"
                    )
                record = _record(
                    source,
                    canonical_base,
                    bound_metadata,
                    bound_metadata_path,
                    target_bound_metadata,
                    target_bound_metadata_path,
                    source_sweep_metadata_path,
                    first_frame,
                    scene_condition,
                    target_video,
                    root,
                    scene_id,
                )
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                sample_count += 1
                split_counts[record["split"]] += 1
                sweep = record["sweep"]
                mode_counts["base" if sweep["mode"] == "base" else sweep["axis"]] += 1
    temporary_path.replace(manifest_path)

    validation = validate_manifest(manifest_path, root, check_files=True)
    scene_index_path = output_root / "scenes.json"
    scene_index = {
        "schema": "physweep.model_scene_condition_index.v1",
        "scene_count": len(scene_reports),
        "records": scene_reports,
    }
    scene_index_path.write_text(json.dumps(scene_index, indent=2), encoding="utf-8")
    summary = {
        "schema": MANIFEST_SCHEMA,
        "path_base": "physweep_project_root",
        "manifest": project_relative_path(root, manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "source_sweep_manifest": project_relative_path(root, sweep_manifest_path),
        "source_sweep_manifest_sha256": _sha256(sweep_manifest_path),
        "sweep_config": project_relative_path(root, config_path),
        "sweep_config_sha256": _sha256(config_path),
        "base_scene_count": len(parents),
        "sample_count": sample_count,
        "samples_per_base": sample_count / max(len(parents), 1),
        "split_counts": dict(sorted(split_counts.items())),
        "mode_counts": dict(mode_counts),
        "base_deduplication": {
            "input_records_per_base": 15,
            "output_records_per_base": sweep_group_size(1),
            "canonical_base_axis": args.canonical_base_axis,
            "base_level_source": "per_axis_sweep_metadata",
            "base_level_distribution": {
                axis: {str(level): count for level, count in sorted(counts.items())}
                for axis, counts in base_level_counts.items()
            },
        },
        "scene_condition": {
            "schema": MODEL_SCENE_SCHEMA,
            "point_count": args.object_points + args.environment_points,
            "object_points": args.object_points,
            "environment_points": args.environment_points,
            "shared_once_per_base": True,
            "source": "simulation_gt_complete_object_complete_ground_plus_camera_first_hit_visible_non_ground_at_t0",
            "source_root": project_relative_path(root, scene_source_root),
            "index": project_relative_path(root, scene_index_path),
            "index_sha256": _sha256(scene_index_path),
            "approximation_count": 0,
            "approximations": [],
        },
        "text_conditioning": {
            "template_version": PROMPT_TEMPLATE_VERSION,
            "source": "base_bound_metadata_semantics",
            "shared_once_per_base": True,
            "sweep_parameters_in_text": False,
        },
        "validation": validation,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile unique PhysSweep video-conditioning samples"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sweep-manifest", type=Path, required=True)
    parser.add_argument("--base-render-root", type=Path, required=True)
    parser.add_argument("--sweep-render-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-source-root", type=Path, required=True)
    parser.add_argument("--canonical-base-axis", choices=SWEEP_AXES, default="mass_kg")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--object-points", type=int, default=DEFAULT_OBJECT_POINTS)
    parser.add_argument(
        "--environment-points", type=int, default=DEFAULT_ENVIRONMENT_POINTS
    )
    parser.add_argument("--limit-base-scenes", type=int)
    parser.add_argument("--force-scenes", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = compile_manifest(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
