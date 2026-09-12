#!/usr/bin/env python3
"""Prepare schema-specific render manifests for complete sweep groups."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import tempfile
from contextlib import contextmanager
from collections import Counter
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json_atomic as write_json
from tools.core.paths import resolve_project_path as project_path
from tools.core.sweep_values import sweep_group_size, sweep_target_indices
from tools.release.sweep_validation import validate_groups
from tools.dataset_contract.object_identity_contract import attach_object_identity
from tools.dataset_contract.camera_contract import camera_clipping_range
from tools.physics.specialized_backend_registry import specialized_by_schema

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def schema_branches(root: Path) -> dict[str, str]:
    return {
        "physweep_pybullet_rigid_metadata_v1": "generic",
        **{
            schema: record["sweep_branch"]
            for schema, record in specialized_by_schema(root).items()
        },
    }


def relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root))


def load_accepted_base_cameras(
    root: Path, manifest_path: Path, expected_hash: str, *,
    base_physics_by_parent: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Load explicit reviewed cameras without mutating physics metadata."""
    if sha256(manifest_path) != expected_hash:
        raise ValueError("accepted base camera manifest hash mismatch")
    manifest = load_json(manifest_path)
    rows = manifest["records"]
    if (manifest.get("schema_version") != "physweep_accepted_base_cameras_v1"
            or manifest.get("status") != "accepted"
            or manifest.get("sample_count") != len(rows)):
        raise ValueError("invalid accepted base camera manifest")
    by_id = {}
    for row in rows:
        parent = str(row["parent_scene_id"])
        if not parent or parent in by_id or row.get("camera_review_passed") is not True:
            raise ValueError("accepted cameras contain duplicate or unreviewed bases")
        for field in ("source_metadata", "trajectory"):
            binding = row[field]
            if sha256(project_path(root, binding["path"])) != binding["sha256"]:
                raise ValueError(f"accepted base camera {field} hash mismatch")
        source = load_json(project_path(root, row["source_metadata"]["path"]))
        if str(source.get("scene_id")) != parent:
            raise ValueError("accepted camera source belongs to different parent")
        trajectory = project_path(root, row["trajectory"]["path"])
        reviewed = load_json(trajectory.with_name("simulation_record.json"))
        verify_camera_base_physics(root, row, reviewed)
        if project_path(root, reviewed["trajectory_path"]) != trajectory:
            raise ValueError("accepted camera trajectory differs from its simulation record")
        current = base_physics_by_parent.get(parent)
        if current is None:
            raise ValueError(f"current parent base physics is missing: {parent}")
        verify_camera_base_physics(root, row, current)
        if row["trajectory"]["sha256"] != current["trajectory_sha256"]:
            raise ValueError("accepted camera trajectory differs from current parent base physics")
        camera = row["camera"]
        camera_clipping_range(camera)
        for field in ("position_m", "target_m"):
            if len(camera[field]) != 3 or not all(math.isfinite(float(x)) for x in camera[field]):
                raise ValueError("accepted camera vectors must be finite 3D")
        if any(not math.isfinite(float(camera[k])) or float(camera[k]) <= 0 for k in ("focal_length_mm", "sensor_width_mm")):
            raise ValueError("accepted camera intrinsics are invalid")
        current_record = project_path(root, current["trajectory_path"]).with_name("simulation_record.json")
        by_id[parent] = {
            **row,
            "current_base_physics": {"path": relative(root, current_record), "sha256": sha256(current_record)},
        }
    return by_id


def verify_camera_base_physics(root: Path, row: dict[str, Any], physics: dict[str, Any]) -> None:
    """Bind a reviewed or current base trajectory through its simulation metadata."""
    dispatched_paths(root, physics)
    if not physics.get("audit_passed") or physics.get("failed_checks"):
        raise ValueError("accepted camera requires passed parent base physics")
    metadata_path = project_path(root, physics["metadata_path"])
    if sha256(metadata_path) != physics["metadata_sha256"]:
        raise ValueError("camera base physics metadata hash mismatch")
    metadata = load_json(metadata_path)
    if metadata.get("scene_id") != physics["scene_id"]:
        raise ValueError("camera base physics scene identity mismatch")
    parent = row["parent_scene_id"]
    source = row["source_metadata"]
    if metadata.get("scene_id") == parent and "sweep" not in metadata:
        path, digest = metadata_path, physics["metadata_sha256"]
    else:
        sweep = metadata.get("sweep", {})
        if sweep.get("kind") != "base" or sweep.get("parent_scene_id") != parent:
            raise ValueError("camera trajectory belongs to different parent base")
        path = project_path(root, sweep["parent_metadata_path"])
        digest = sweep["parent_metadata_sha256"]
    if path != project_path(root, source["path"]) or digest != source["sha256"]:
        raise ValueError("camera trajectory belongs to different parent metadata")


def apply_accepted_base_camera(
    root: Path, bound: dict[str, Any], cameras: dict[str, dict[str, Any]],
    manifest_binding: dict[str, str],
) -> None:
    sweep = bound["sweep"]
    parent = str(sweep["parent_scene_id"])
    if parent not in cameras:
        raise ValueError(f"accepted base camera is missing: {parent}")
    row = cameras[parent]
    source = row["source_metadata"]
    if (project_path(root, source["path"]) != project_path(root, sweep["parent_metadata_path"])
            or source["sha256"] != sweep["parent_metadata_sha256"]):
        raise ValueError("accepted camera belongs to different parent metadata")
    bound["camera"] = copy.deepcopy(row["camera"])
    if "context" in row:
        bound.setdefault("render", {})["context"] = copy.deepcopy(row["context"])
    bound["camera_inheritance"] = {
        "policy": "copied_from_accepted_parent_base",
        "parent_scene_id": parent,
        "accepted_base_cameras": dict(manifest_binding),
        "parent_trajectory": copy.deepcopy(row["trajectory"]),
        "current_base_physics": copy.deepcopy(row["current_base_physics"]),
    }


def select_complete_groups(
    metadata_records: list[dict[str, Any]],
    selected_parents: set[str],
    *,
    object_count: int = 1,
    target_object_indices: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    selected = [
        record for record in metadata_records if str(record["parent"]) in selected_parents
    ]
    targets = sweep_target_indices(object_count, target_object_indices)
    counts = Counter(str(record["parent"]) for record in selected)
    if set(counts) != selected_parents or any(
        value != sweep_group_size(len(targets)) for value in counts.values()
    ):
        raise ValueError(
            "selected sweep records do not form complete "
            f"{sweep_group_size(len(targets))}-sample groups"
        )
    validate_groups(selected, expected_target_indices=targets)
    return selected


def dispatched_paths(root: Path, physics: dict[str, Any]) -> dict[str, str]:
    trajectory = project_path(root, physics["trajectory_path"])
    audit = project_path(root, physics["audit_path"])
    record = trajectory.with_name("simulation_record.json")
    for path in (trajectory, audit, record):
        path.relative_to(root)
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(trajectory) != str(physics["trajectory_sha256"]):
        raise ValueError("trajectory hash does not match the physics manifest")
    if sha256(audit) != str(physics["audit_sha256"]):
        raise ValueError("audit hash does not match the physics manifest")
    simulation = load_json(record)
    if str(simulation["scene_id"]) != str(physics["scene_id"]):
        raise ValueError("simulation record scene id does not match physics manifest")
    for path_key, hash_key, expected_path in (
        ("metadata_path", "metadata_sha256", project_path(root, physics["metadata_path"])),
        ("trajectory_path", "trajectory_sha256", trajectory),
        ("audit_path", "audit_sha256", audit),
    ):
        if (
            project_path(root, simulation[path_key]) != expected_path
            or str(simulation[hash_key]) != str(physics[hash_key])
        ):
            raise ValueError(f"simulation record provenance mismatch: {path_key}")
    return {
        "trajectory_path": relative(root, trajectory),
        "audit_path": relative(root, audit),
        "simulation_record_path": relative(root, record),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--registry-root", type=Path, help="Explicit source tree containing the reviewed backend registry.")
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--staged-base-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--accepted-base-cameras", type=Path)
    parser.add_argument("--accepted-base-cameras-sha256")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


@contextmanager
def preparation_directory(output_root: Path, *, overwrite: bool):
    """Stage all preparation files before publishing the completion manifest.

    Legacy interrupted directories contain only inputs. Refuse automatic
    recovery if any render artifact or an unrecognized file is present.
    """
    output_root = output_root.resolve()
    def check_existing():
        if output_root.exists() and not overwrite:
            allowed = {"physics_manifest.json", "render_input_manifest.json",
                       "base_render_input_manifest.json", "derived_render_input_manifest.json"}
            if (output_root / "manifest.json").exists():
                raise FileExistsError(f"completed output exists: {output_root}")
            for path in output_root.rglob("*"):
                relative_path = path.relative_to(output_root)
                if path.is_symlink() or (path.is_file() and not (
                    (len(relative_path.parts) == 2 and path.name in allowed)
                    or (len(relative_path.parts) == 3 and relative_path.parts[1] == "metadata"
                        and path.suffix == ".json")
                )):
                    raise FileExistsError(f"cannot recover output containing render artifacts: {path}")
    check_existing()
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.preparing-", dir=output_root.parent))
    backup = stage.with_name(stage.name + ".previous")
    try:
        yield stage
        if not (stage / "manifest.json").is_file():
            raise RuntimeError("render preparation has no completion manifest")
        check_existing()
        if output_root.exists():
            output_root.rename(backup)
        try:
            stage.rename(output_root)
        except BaseException:
            if backup.exists():
                backup.rename(output_root)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    release_path = project_path(root, args.release_manifest)
    staged_path = project_path(root, args.staged_base_manifest)
    output_root = project_path(root, args.output_root)
    release_path.relative_to(root / "datasets")
    # Base selection can come from a frozen datasets/ manifest or a staged outputs/ view.
    staged_path.relative_to(root)
    if (root / "outputs").resolve() not in output_root.parents:
        raise ValueError("sweep render plan output must remain under root/outputs")

    release = load_json(release_path)
    camera_path = getattr(args, "accepted_base_cameras", None)
    camera_hash = getattr(args, "accepted_base_cameras_sha256", None)
    if (camera_path is None) != (camera_hash is None):
        raise ValueError("accepted camera manifest and hash must be provided together")
    camera_binding = None
    cameras = None
    if camera_path is not None:
        camera_path = project_path(root, camera_path)
        camera_binding = {"path": relative(root, camera_path), "sha256": camera_hash}
    object_count = int(release.get("object_count", 1))
    targets = sweep_target_indices(object_count, release.get("sweep_target_object_indices"))
    if object_count < 1:
        raise ValueError("source release object count must be positive")
    metadata_path = project_path(root, release["metadata_manifest"])
    physics_path = project_path(root, release["physics_manifest"])
    if sha256(metadata_path) != str(release["metadata_manifest_sha256"]):
        raise ValueError("release metadata manifest hash mismatch")
    if sha256(physics_path) != str(release["physics_manifest_sha256"]):
        raise ValueError("release physics manifest hash mismatch")
    metadata_manifest = load_json(metadata_path)
    physics_manifest = load_json(physics_path)
    if int(metadata_manifest["sample_count"]) != len(metadata_manifest["records"]):
        raise ValueError("release metadata sample count is inconsistent")
    if int(physics_manifest["sample_count"]) != len(physics_manifest["records"]):
        raise ValueError("release physics sample count is inconsistent")
    staged = load_json(staged_path)
    selected_parents = {
        str(record.get("metadata_path") or record.get("parent"))
        for record in staged["records"]
        if record.get("metadata_path") or record.get("parent")
    }
    if not selected_parents:
        raise ValueError("staged manifest contains no base metadata paths")
    selected = select_complete_groups(
        metadata_manifest["records"],
        selected_parents,
        object_count=object_count,
        target_object_indices=targets,
    )
    selected_scene_ids = [str(record["scene_id"]) for record in selected]
    if len(selected_scene_ids) != len(set(selected_scene_ids)):
        raise ValueError("selected sweep records contain duplicate scene ids")
    physics_by_scene = {
        str(record["scene_id"]): record for record in physics_manifest["records"]
    }
    if len(physics_by_scene) != len(physics_manifest["records"]):
        raise ValueError("release physics manifest contains duplicate scene ids")
    if camera_binding is not None:
        base_physics_by_parent = {}
        for record in metadata_manifest["records"]:
            if record.get("kind") != "base":
                continue
            source_path = project_path(root, record["path"])
            if sha256(source_path) != record["metadata_sha256"]:
                raise ValueError("camera parent base metadata hash mismatch")
            source = load_json(source_path)
            parent = source["sweep"]["parent_scene_id"]
            if parent in base_physics_by_parent:
                raise ValueError("camera parent has duplicate base physics")
            base_physics_by_parent[parent] = physics_by_scene[str(record["scene_id"])]
        cameras = load_accepted_base_cameras(
            root, camera_path, camera_hash, base_physics_by_parent=base_physics_by_parent,
        )
    branch_by_schema = schema_branches(getattr(args, 'registry_root', None) or root)

    branches: dict[
        str, list[tuple[dict[str, Any], dict[str, Any], dict[str, str]]]
    ] = {}
    for record in selected:
        scene_id = str(record["scene_id"])
        physics = physics_by_scene.get(scene_id)
        if physics is None:
            raise ValueError(f"selected physics record is missing: {scene_id}")
        if (
            not physics.get("ok")
            or not physics.get("audit_passed")
            or physics.get("failed_checks")
        ):
            raise ValueError(f"selected physics record did not pass: {record['scene_id']}")
        if str(physics["source_schema_version"]) != str(
            record["source_schema_version"]
        ):
            raise ValueError(f"selected source schema mismatch: {scene_id}")
        source_path = project_path(root, record["path"])
        if (
            project_path(root, physics["metadata_path"]) != source_path
            or sha256(source_path) != str(record["metadata_sha256"])
            or str(physics["metadata_sha256"]) != str(record["metadata_sha256"])
        ):
            raise ValueError(f"selected metadata provenance mismatch: {scene_id}")
        branch = branch_by_schema.get(str(record["source_schema_version"]))
        if branch is None:
            raise ValueError(f"unsupported render schema: {record['source_schema_version']}")
        branches.setdefault(branch, []).append(
            (record, physics, dispatched_paths(root, physics))
        )

    with preparation_directory(output_root, overwrite=args.overwrite) as stage:
        generic_records = []
        for _record, physics, paths in branches.get("generic", []):
            generic_records.append({**physics, **paths})
        if generic_records:
            generic_manifest = {
                "schema_version": "physweep_pybullet_batch_record_v1",
                "dataset_id": f"{release['dataset_id']}__sweep_generic",
                "sample_count": len(generic_records),
                "passed_count": len(generic_records),
                "rejected_count": 0,
                "error_count": 0,
                "records": generic_records,
            }
            write_json(
                stage / "generic" / "physics_manifest.json",
                generic_manifest,
            )

        for branch in sorted(set(branches) - {"generic"}):
            render_records = []
            partitioned_records: dict[str, list[dict[str, Any]]] = {
                "base": [],
                "derived": [],
            }
            for record, physics, paths in branches[branch]:
                source_path = project_path(root, physics["metadata_path"])
                bound = copy.deepcopy(load_json(source_path))
                bound["source_metadata"] = {
                    "path": relative(root, source_path),
                    "sha256": sha256(source_path),
                }
                bound.setdefault("physics", {}).update(paths)
                if cameras is not None:
                    apply_accepted_base_camera(root, bound, cameras, camera_binding)
                scene_id = str(record["scene_id"])
                frame_dir = output_root / branch / "frames" / scene_id
                video_path = output_root / branch / "videos" / f"{scene_id}.mp4"
                mask_path = output_root / branch / "masks" / scene_id
                bound.setdefault("render", {})["inspection_frame_dir"] = relative(
                    root, frame_dir
                )
                bound["render"]["video_path"] = relative(root, video_path)
                if bound.get('schema_version') in {'physweep_billiards_three_object_scene_v1','physweep_passive_pinball_three_object_scene_v1','physweep_marble_run_three_object_scene_v1'}:
                    from tools.rendering.three_object_specialized_binding import bind_render_implementation
                    bind_render_implementation(root, bound)
                attach_object_identity(
                    bound,
                    trajectory_path=paths["trajectory_path"],
                    mask_path=relative(root, mask_path),
                )
                bound_path = output_root / branch / "metadata" / f"{scene_id}.json"
                write_json(stage / bound_path.relative_to(output_root), bound)
                render_record = {
                    "scene_id": scene_id,
                    "metadata_path": relative(root, bound_path),
                    "metadata_sha256": sha256(stage / bound_path.relative_to(output_root)),
                    "render_output": {
                        "video_path": relative(root, video_path),
                        "inspection_frame_dir": relative(root, frame_dir),
                    },
                }
                render_records.append(render_record)
                sweep_kind = str(bound["sweep"]["kind"])
                if sweep_kind not in {"base", "sweep"}:
                    raise ValueError(f"unsupported sweep kind: {sweep_kind}")
                partitioned_records[
                    "base" if sweep_kind == "base" else "derived"
                ].append(render_record)
            manifest = {
                "schema_version": f"physweep_sweep_{branch}_render_manifest_v1",
                "dataset_id": f"{release['dataset_id']}__sweep_{branch}",
                "output_root": relative(root, output_root / branch),
                "sample_count": len(render_records),
                "records": render_records,
            }
            write_json(stage / branch / "render_input_manifest.json", manifest)
            for partition, records in partitioned_records.items():
                partition_manifest = {
                    **manifest,
                    "dataset_id": (
                        f"{release['dataset_id']}__sweep_{branch}_{partition}"
                    ),
                    "sample_count": len(records),
                    "records": records,
                }
                write_json(
                    stage / branch / f"{partition}_render_input_manifest.json",
                    partition_manifest,
                )

        summary = {
            "schema_version": "physweep_sweep_render_plan_v1",
            "source_release": relative(root, release_path),
            "source_release_sha256": sha256(release_path),
            "source_staged_base_manifest": relative(root, staged_path),
            "source_staged_base_manifest_sha256": sha256(staged_path),
            "group_count": len(selected_parents),
            "object_count": object_count,
            "sweep_target_object_indices": list(targets),
            "sample_count": len(selected),
            "branch_counts": {name: len(records) for name, records in branches.items()},
        }
        if camera_binding is not None:
            summary["accepted_base_cameras"] = camera_binding
        write_json(stage / "manifest.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
