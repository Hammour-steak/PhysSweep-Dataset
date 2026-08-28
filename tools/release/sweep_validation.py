"""Shared validation for immutable one-object sweep releases."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.paths import resolve_project_path

AXES = ("mass_kg", "contact_friction", "contact_restitution")
SUPPORTED_TARGET_OBJECT_INDICES = (0,)


def validate_groups(records: list[dict[str, Any]]) -> int:
    """Validate the frozen 13-record, one-object sweep group contract."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["parent"])].append(record)
    for parent, group in groups.items():
        if len(group) != 13:
            raise ValueError(f"group does not contain 13 records: {parent}")
        if sum(record["kind"] == "base" for record in group) != 1:
            raise ValueError(f"group does not contain one canonical base: {parent}")
        axis_counts = Counter(
            record.get("axis") for record in group if record["kind"] == "sweep"
        )
        if axis_counts != Counter({axis: 4 for axis in AXES}):
            raise ValueError(f"group has an invalid axis layout: {parent}")
        derived = [record for record in group if record["kind"] == "sweep"]
        for axis in AXES:
            levels = {
                int(record["level_index"])
                for record in derived
                if record["axis"] == axis
            }
            if levels != {0, 1, 3, 4}:
                raise ValueError(f"group has invalid sweep levels: {parent}/{axis}")
        targets = {
            (str(record["target_object_id"]), int(record["target_object_index"]))
            for record in derived
        }
        if (
            len(targets) != 1
            or next(iter(targets))[1] not in SUPPORTED_TARGET_OBJECT_INDICES
        ):
            raise ValueError(f"group does not target one object: {parent}")
    return len(groups)


def validate_source_artifacts(
    root: Path,
    metadata_records: list[dict[str, Any]],
    physics_records: list[dict[str, Any]],
) -> None:
    """Verify that release records bind to the declared immutable artifacts."""

    root = root.resolve()
    physics_by_id = {str(record["scene_id"]): record for record in physics_records}
    if len(physics_by_id) != len(physics_records):
        raise ValueError("physics manifest contains duplicate scene ids")
    for metadata in metadata_records:
        scene_id = str(metadata["scene_id"])
        physics = physics_by_id.get(scene_id)
        if physics is None:
            raise ValueError(f"physics record is missing: {scene_id}")
        metadata_path = resolve_project_path(root, metadata["path"])
        metadata_path.relative_to(root)
        metadata_hash = sha256(metadata_path)
        if (
            resolve_project_path(root, physics["metadata_path"]) != metadata_path
            or str(metadata["metadata_sha256"]) != metadata_hash
            or str(physics["metadata_sha256"]) != metadata_hash
        ):
            raise ValueError(f"metadata provenance mismatch: {scene_id}")
        if str(metadata["source_schema_version"]) != str(
            physics["source_schema_version"]
        ):
            raise ValueError(f"source schema mismatch: {scene_id}")
        for path_key, hash_key in (
            ("resolved_scene_path", "resolved_scene_sha256"),
            ("trajectory_path", "trajectory_sha256"),
            ("audit_path", "audit_sha256"),
        ):
            artifact = resolve_project_path(root, physics[path_key])
            artifact.relative_to(root)
            if sha256(artifact) != str(physics[hash_key]):
                raise ValueError(
                    f"physics artifact hash mismatch: {scene_id}/{path_key}"
                )
