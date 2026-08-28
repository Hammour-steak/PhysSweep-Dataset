#!/usr/bin/env python3
"""Prepare schema-specific render manifests for complete sweep groups."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root))


def project_path(root: Path, value: Path | str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def select_complete_groups(
    metadata_records: list[dict[str, Any]], selected_parents: set[str]
) -> list[dict[str, Any]]:
    selected = [
        record for record in metadata_records if str(record["parent"]) in selected_parents
    ]
    counts = Counter(str(record["parent"]) for record in selected)
    if set(counts) != selected_parents or any(value != 13 for value in counts.values()):
        raise ValueError("selected sweep records do not form complete 13-sample groups")
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
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--staged-base-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    release_path = project_path(root, args.release_manifest)
    staged_path = project_path(root, args.staged_base_manifest)
    output_root = project_path(root, args.output_root)
    release_path.relative_to(root / "datasets")
    staged_path.relative_to(root / "outputs")
    if (root / "outputs").resolve() not in output_root.parents:
        raise ValueError("sweep render plan output must remain under root/outputs")

    release = load_json(release_path)
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
    selected = select_complete_groups(metadata_manifest["records"], selected_parents)
    selected_scene_ids = [str(record["scene_id"]) for record in selected]
    if len(selected_scene_ids) != len(set(selected_scene_ids)):
        raise ValueError("selected sweep records contain duplicate scene ids")
    physics_by_scene = {
        str(record["scene_id"]): record for record in physics_manifest["records"]
    }
    if len(physics_by_scene) != len(physics_manifest["records"]):
        raise ValueError("release physics manifest contains duplicate scene ids")
    branch_by_schema = schema_branches(root)

    branches: dict[
        str, list[tuple[dict[str, Any], dict[str, Any], dict[str, str]]]
    ] = {
        name: [] for name in branch_by_schema.values()
    }
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
        branches[branch].append((record, physics, dispatched_paths(root, physics)))

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    generic_records = []
    for _record, physics, paths in branches["generic"]:
        generic_records.append({**physics, **paths})
    generic_manifest = {
        "schema_version": "physweep_pybullet_batch_record_v1",
        "dataset_id": "one_object_sweep_review_generic",
        "sample_count": len(generic_records),
        "passed_count": len(generic_records),
        "rejected_count": 0,
        "error_count": 0,
        "records": generic_records,
    }
    write_json(output_root / "generic" / "physics_manifest.json", generic_manifest)

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
            scene_id = str(record["scene_id"])
            bound_path = output_root / branch / "metadata" / f"{scene_id}.json"
            write_json(bound_path, bound)
            render_record = {
                "scene_id": scene_id,
                "metadata_path": relative(root, bound_path),
                "metadata_sha256": sha256(bound_path),
                "render_output": {
                    "video_path": relative(
                        root, output_root / branch / "videos" / f"{scene_id}.mp4"
                    ),
                    "inspection_frame_dir": relative(
                        root, output_root / branch / "frames" / scene_id
                    ),
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
            "dataset_id": f"one_object_sweep_review_{branch}",
            "output_root": relative(root, output_root / branch),
            "sample_count": len(render_records),
            "records": render_records,
        }
        write_json(output_root / branch / "render_input_manifest.json", manifest)
        for partition, records in partitioned_records.items():
            partition_manifest = {
                **manifest,
                "dataset_id": f"one_object_sweep_review_{branch}_{partition}",
                "sample_count": len(records),
                "records": records,
            }
            write_json(
                output_root / branch / f"{partition}_render_input_manifest.json",
                partition_manifest,
            )

    summary = {
        "schema_version": "physweep_sweep_render_plan_v1",
        "source_release": relative(root, release_path),
        "source_release_sha256": sha256(release_path),
        "source_staged_base_manifest": relative(root, staged_path),
        "source_staged_base_manifest_sha256": sha256(staged_path),
        "group_count": len(selected_parents),
        "sample_count": len(selected),
        "branch_counts": {name: len(records) for name, records in branches.items()},
    }
    write_json(output_root / "manifest.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
