"""Publish a fresh audited one-object source release without historical deltas."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json_atomic_sorted as write_json
from tools.core.paths import (
    project_relative_path,
    resolve_project_path_within_root,
)
from tools.core.sweep_values import SWEEP_GROUP_SIZE
from tools.dataset_contract.object_identity_contract import validate_object_identity
from tools.release.sweep_validation import validate_groups, validate_source_artifacts


RELEASE_SCHEMA = "physweep_one_object_source_release_v1"
METADATA_SCHEMA = "physweep_release_metadata_manifest_v1"
PHYSICS_SCHEMA = "physweep_release_physics_manifest_v1"
DATASET_ID = "physweep_one_object"


def _path(root: Path, value: Path) -> Path:
    return resolve_project_path_within_root(root, value)


def _relative(root: Path, value: Path) -> str:
    return project_relative_path(root, value)


def _records(document: dict[str, Any], label: str) -> list[dict[str, Any]]:
    records = document.get("records")
    if not isinstance(records, list) or int(document.get("sample_count", -1)) != len(
        records
    ):
        raise ValueError(f"{label} sample count differs")
    return records


def _verified_metadata(root: Path, record: dict[str, Any], label: str) -> Path:
    value = record.get("metadata_path") or record.get("path")
    if not value:
        raise ValueError(f"{label} has no metadata path")
    path = _path(root, Path(str(value)))
    if not path.is_file() or sha256(path) != str(record.get("metadata_sha256")):
        raise ValueError(f"{label} metadata hash differs: {path}")
    return path


def publish_source_release(
    *,
    root: Path,
    base_manifest_path: Path,
    sweep_metadata_manifest_path: Path,
    sweep_physics_manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    root = root.resolve()
    base_manifest_path = _path(root, base_manifest_path)
    sweep_metadata_manifest_path = _path(root, sweep_metadata_manifest_path)
    sweep_physics_manifest_path = _path(root, sweep_physics_manifest_path)
    output = _path(root, output)
    if (root / "datasets").resolve() not in output.parents:
        raise ValueError("source release must remain below root/datasets")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"source release already exists: {output}")

    base = load_json(base_manifest_path)
    metadata = load_json(sweep_metadata_manifest_path)
    physics = load_json(sweep_physics_manifest_path)
    base_records = _records(base, "base manifest")
    metadata_records = _records(metadata, "sweep metadata manifest")
    physics_records = _records(physics, "sweep physics manifest")
    group_count = validate_groups(metadata_records)
    parent_paths = {str(record["parent"]) for record in metadata_records}
    base_paths = [str(record.get("metadata_path", "")) for record in base_records]
    if not base_paths or len(base_paths) != len(set(base_paths)):
        raise ValueError("base manifest contains missing or duplicate metadata paths")
    if parent_paths != set(base_paths):
        raise ValueError("base manifest and sweep groups select different parents")

    for index, record in enumerate(base_records):
        path = _verified_metadata(root, record, f"base record {index}")
        identity = validate_object_identity(load_json(path))
        if int(identity["dynamic_object_count"]) != 1:
            raise ValueError(f"base record is not one-object: {path}")

    metadata_by_scene = {
        str(record["scene_id"]): record for record in metadata_records
    }
    physics_by_scene = {str(record["scene_id"]): record for record in physics_records}
    if (
        len(metadata_by_scene) != len(metadata_records)
        or len(physics_by_scene) != len(physics_records)
        or set(metadata_by_scene) != set(physics_by_scene)
    ):
        raise ValueError("sweep metadata and physics scene identities differ")
    validate_source_artifacts(root, metadata_records, physics_records)
    for scene_id, record in metadata_by_scene.items():
        metadata_path = _verified_metadata(root, record, f"sweep record {scene_id}")
        identity = validate_object_identity(load_json(metadata_path))
        if int(identity["dynamic_object_count"]) != 1:
            raise ValueError(f"sweep record is not one-object: {scene_id}")
        physical = physics_by_scene[scene_id]
        if (
            not physical.get("ok")
            or not physical.get("audit_passed")
            or physical.get("failed_checks")
            or _path(root, Path(str(physical["metadata_path"]))) != metadata_path
            or str(physical["metadata_sha256"]) != str(record["metadata_sha256"])
        ):
            raise ValueError(f"sweep physics did not pass exactly: {scene_id}")

    if (
        len(base_records) != group_count
        or len(metadata_records) != group_count * SWEEP_GROUP_SIZE
    ):
        raise ValueError("one-object release group totals differ")
    base_count = group_count
    derived_count = len(metadata_records) - base_count
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as tmp:
        work = Path(tmp) / output.name
        work.mkdir()
        base_output = work / "base_manifest.json"
        metadata_output = work / "metadata_manifest.json"
        physics_output = work / "physics_manifest.json"
        published_base = {**base, "dataset_id": DATASET_ID}
        write_json(base_output, published_base)
        source_binding = {
            "metadata_manifest": _relative(root, sweep_metadata_manifest_path),
            "metadata_manifest_sha256": sha256(sweep_metadata_manifest_path),
            "physics_manifest": _relative(root, sweep_physics_manifest_path),
            "physics_manifest_sha256": sha256(sweep_physics_manifest_path),
            "sample_count": len(metadata_records),
        }
        write_json(
            metadata_output,
            {
                "schema_version": METADATA_SCHEMA,
                "dataset_id": DATASET_ID,
                "sample_count": len(metadata_records),
                "group_count": group_count,
                "group_size": SWEEP_GROUP_SIZE,
                "sources": [source_binding],
                "records": metadata_records,
            },
        )
        write_json(
            physics_output,
            {
                "schema_version": PHYSICS_SCHEMA,
                "dataset_id": DATASET_ID,
                "sample_count": len(physics_records),
                "passed_count": len(physics_records),
                "rejected_count": 0,
                "error_count": 0,
                "pass_rate": 1.0,
                "group_count": group_count,
                "group_size": SWEEP_GROUP_SIZE,
                "sources": [source_binding],
                "records": physics_records,
            },
        )

        def published(path: Path) -> str:
            return _relative(root, output / path.name)

        release = {
            "schema_version": RELEASE_SCHEMA,
            "dataset_id": DATASET_ID,
            "sample_count": len(metadata_records),
            "base_count": base_count,
            "derived_count": derived_count,
            "base_manifest": published(base_output),
            "base_manifest_sha256": sha256(base_output),
            "metadata_manifest": published(metadata_output),
            "metadata_manifest_sha256": sha256(metadata_output),
            "physics_manifest": published(physics_output),
            "physics_manifest_sha256": sha256(physics_output),
        }
        write_json(work / "manifest.json", release)
        work.replace(output)
    return release
