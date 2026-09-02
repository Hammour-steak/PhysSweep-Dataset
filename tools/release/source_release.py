"""Publish audited object-count-aware source releases."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file
from tools.core.json_io import read_json, write_json_atomic_sorted
from tools.core.paths import project_relative_path, resolve_project_path_within_root
from tools.core.sweep_values import sweep_group_size
from tools.dataset_contract.object_identity_contract import validate_object_identity
from tools.release.sweep_validation import validate_groups, validate_source_artifacts


METADATA_SCHEMA = "physweep_release_metadata_manifest_v1"
PHYSICS_SCHEMA = "physweep_release_physics_manifest_v1"


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
    path = resolve_project_path_within_root(root, Path(str(value)))
    if not path.is_file() or sha256_file(path) != str(record.get("metadata_sha256")):
        raise ValueError(f"{label} metadata hash differs: {path}")
    return path


def _validate_object_count(
    root: Path,
    records: list[dict[str, Any]],
    *,
    expected: int,
    label: str,
) -> list[tuple[Path, dict[str, Any], tuple[str, ...]]]:
    validated = []
    for index, record in enumerate(records):
        path = _verified_metadata(root, record, f"{label} {index}")
        document = read_json(path)
        identity = validate_object_identity(document)
        if int(identity["dynamic_object_count"]) != expected:
            raise ValueError(
                f"{label} does not contain exactly {expected} dynamic objects: {path}"
            )
        validated.append(
            (path, document, tuple(str(value) for value in identity["dynamic_object_ids"]))
        )
    return validated


def _validate_sweep_record_bindings(
    records: list[dict[str, Any]],
    validated: list[tuple[Path, dict[str, Any], tuple[str, ...]]],
) -> None:
    """Bind every manifest coordinate to its authoritative metadata record."""

    for record, (path, document, dynamic_ids) in zip(records, validated, strict=True):
        sweep = document.get("sweep")
        if not isinstance(sweep, dict):
            raise ValueError(f"sweep metadata contract is missing: {path}")
        manifest_binding = {
            "scene_id": record.get("scene_id"),
            "parent": record.get("parent"),
            "kind": record.get("kind"),
            "axis": record.get("axis"),
            "level_index": record.get("level_index"),
            "value": record.get("value"),
            "target_object_id": record.get("target_object_id"),
            "target_object_index": record.get("target_object_index"),
            "source_schema_version": record.get("source_schema_version"),
        }
        metadata_binding = {
            "scene_id": document.get("scene_id"),
            "parent": sweep.get("parent_metadata_path"),
            "kind": sweep.get("kind"),
            "axis": sweep.get("axis"),
            "level_index": sweep.get("level_index"),
            "value": sweep.get("value"),
            "target_object_id": sweep.get("target_object_id"),
            "target_object_index": sweep.get("target_object_index"),
            "source_schema_version": sweep.get("source_schema_version"),
        }
        if manifest_binding != metadata_binding:
            raise ValueError(f"sweep manifest differs from metadata: {path}")
        if manifest_binding["kind"] == "base":
            continue
        if manifest_binding["kind"] != "sweep":
            raise ValueError(f"sweep record has invalid kind: {path}")
        target_index = manifest_binding["target_object_index"]
        target_id = manifest_binding["target_object_id"]
        if (
            isinstance(target_index, bool)
            or not isinstance(target_index, int)
            or target_index < 0
            or target_index >= len(dynamic_ids)
            or not isinstance(target_id, str)
            or not target_id
            or dynamic_ids[target_index] != target_id
        ):
            raise ValueError(f"sweep target axis differs from object identity: {path}")


def publish_source_release(
    *,
    root: Path,
    base_manifest_path: Path,
    sweep_metadata_manifest_path: Path,
    sweep_physics_manifest_path: Path,
    output: Path,
    object_count: int,
    dataset_id: str,
    release_schema: str,
) -> dict[str, Any]:
    """Publish immutable source manifests after full object/sweep validation."""

    if isinstance(object_count, bool) or not isinstance(object_count, int):
        raise TypeError("source release object_count must be an integer")
    if object_count < 1:
        raise ValueError("source release object_count must be positive")
    if not dataset_id or not release_schema:
        raise ValueError("source release identity must be nonempty")

    root = root.resolve()
    base_manifest_path = resolve_project_path_within_root(root, base_manifest_path)
    sweep_metadata_manifest_path = resolve_project_path_within_root(
        root, sweep_metadata_manifest_path
    )
    sweep_physics_manifest_path = resolve_project_path_within_root(
        root, sweep_physics_manifest_path
    )
    output = resolve_project_path_within_root(root, output)
    if (root / "datasets").resolve() not in output.parents:
        raise ValueError("source release must remain below root/datasets")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"source release already exists: {output}")

    base = read_json(base_manifest_path)
    metadata = read_json(sweep_metadata_manifest_path)
    physics = read_json(sweep_physics_manifest_path)
    base_records = _records(base, "base manifest")
    metadata_records = _records(metadata, "sweep metadata manifest")
    physics_records = _records(physics, "sweep physics manifest")
    _validate_object_count(
        root, base_records, expected=object_count, label="base record"
    )
    validated_sweeps = _validate_object_count(
        root, metadata_records, expected=object_count, label="sweep record"
    )
    _validate_sweep_record_bindings(metadata_records, validated_sweeps)
    group_summary = validate_groups(
        metadata_records,
        expected_target_indices=tuple(range(object_count)),
    )

    parent_paths = {str(record["parent"]) for record in metadata_records}
    base_paths = [str(record.get("metadata_path", "")) for record in base_records]
    if not base_paths or len(base_paths) != len(set(base_paths)):
        raise ValueError("base manifest contains missing or duplicate metadata paths")
    if parent_paths != set(base_paths):
        raise ValueError("base manifest and sweep groups select different parents")

    metadata_by_scene = {
        str(record["scene_id"]): record for record in metadata_records
    }
    physics_by_scene = {
        str(record["scene_id"]): record for record in physics_records
    }
    if (
        len(metadata_by_scene) != len(metadata_records)
        or len(physics_by_scene) != len(physics_records)
        or set(metadata_by_scene) != set(physics_by_scene)
    ):
        raise ValueError("sweep metadata and physics scene identities differ")
    validate_source_artifacts(root, metadata_records, physics_records)
    for scene_id, record in metadata_by_scene.items():
        metadata_path = _verified_metadata(root, record, f"sweep record {scene_id}")
        physical = physics_by_scene[scene_id]
        if (
            not physical.get("ok")
            or not physical.get("audit_passed")
            or physical.get("failed_checks")
            or resolve_project_path_within_root(
                root, Path(str(physical["metadata_path"]))
            )
            != metadata_path
            or str(physical["metadata_sha256"])
            != str(record["metadata_sha256"])
        ):
            raise ValueError(f"sweep physics did not pass exactly: {scene_id}")

    if (
        len(base_records) != group_summary.base_count
        or len(metadata_records)
        != group_summary.base_count * sweep_group_size(object_count)
        or group_summary.derived_count
        != len(metadata_records) - len(base_records)
    ):
        raise ValueError("source release group totals differ")

    group_count = group_summary.base_count
    group_size = sweep_group_size(object_count)
    base_count = len(base_records)
    derived_count = group_summary.derived_count
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as tmp:
        work = Path(tmp) / output.name
        work.mkdir()
        base_output = work / "base_manifest.json"
        metadata_output = work / "metadata_manifest.json"
        physics_output = work / "physics_manifest.json"
        write_json_atomic_sorted(base_output, {**base, "dataset_id": dataset_id})
        source_binding = {
            "metadata_manifest": project_relative_path(
                root, sweep_metadata_manifest_path
            ),
            "metadata_manifest_sha256": sha256_file(sweep_metadata_manifest_path),
            "physics_manifest": project_relative_path(root, sweep_physics_manifest_path),
            "physics_manifest_sha256": sha256_file(sweep_physics_manifest_path),
            "sample_count": len(metadata_records),
        }
        write_json_atomic_sorted(
            metadata_output,
            {
                "schema_version": METADATA_SCHEMA,
                "dataset_id": dataset_id,
                "sample_count": len(metadata_records),
                "group_count": group_count,
                "group_size": group_size,
                "sources": [source_binding],
                "records": metadata_records,
            },
        )
        write_json_atomic_sorted(
            physics_output,
            {
                "schema_version": PHYSICS_SCHEMA,
                "dataset_id": dataset_id,
                "sample_count": len(physics_records),
                "passed_count": len(physics_records),
                "rejected_count": 0,
                "error_count": 0,
                "pass_rate": 1.0,
                "group_count": group_count,
                "group_size": group_size,
                "sources": [source_binding],
                "records": physics_records,
            },
        )

        def published(path: Path) -> str:
            return project_relative_path(root, output / path.name)

        release = {
            "schema_version": release_schema,
            "dataset_id": dataset_id,
            "object_count": object_count,
            "sample_count": len(metadata_records),
            "base_count": base_count,
            "derived_count": derived_count,
            "base_manifest": published(base_output),
            "base_manifest_sha256": sha256_file(base_output),
            "metadata_manifest": published(metadata_output),
            "metadata_manifest_sha256": sha256_file(metadata_output),
            "physics_manifest": published(physics_output),
            "physics_manifest_sha256": sha256_file(physics_output),
        }
        write_json_atomic_sorted(work / "manifest.json", release)
        work.replace(output)
    return release
