"""Shared validation for immutable one-factor sweep source releases."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.paths import resolve_project_path
from tools.core.sweep_values import (
    SWEEP_AXES,
    SWEEP_DERIVED_LEVELS,
    SWEEP_VARIANTS_PER_TARGET,
    sweep_group_size,
)


@dataclass(frozen=True)
class SweepGroupSummary:
    """Counts established by validating complete base/target sweep groups."""

    base_count: int
    target_groups_per_base: int
    derived_count: int


def validate_target_sweep_grid(
    records: list[dict[str, Any]], *, label: str
) -> None:
    """Require the exact 12 unique axis/level coordinates for one target."""

    expected = {
        (axis, level_index)
        for axis in SWEEP_AXES
        for level_index in SWEEP_DERIVED_LEVELS
    }
    observed = []
    for record in records:
        axis = record.get("axis")
        level_index = record.get("level_index")
        if (
            not isinstance(axis, str)
            or axis not in SWEEP_AXES
            or isinstance(level_index, bool)
            or not isinstance(level_index, int)
        ):
            raise ValueError(f"target sweep coordinate is invalid: {label}")
        observed.append((axis, level_index))
    if len(records) != SWEEP_VARIANTS_PER_TARGET or set(observed) != expected:
        raise ValueError(f"target sweep grid differs: {label}")
    if len(observed) != len(set(observed)):
        raise ValueError(f"target sweep grid contains duplicates: {label}")


def validate_groups(
    records: list[dict[str, Any]],
    *,
    expected_target_indices: tuple[int, ...],
) -> SweepGroupSummary:
    """Validate one base and one complete 12-variant group per target object."""

    if (
        not expected_target_indices
        or len(set(expected_target_indices)) != len(expected_target_indices)
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in expected_target_indices
        )
    ):
        raise ValueError("expected sweep target indices must be unique and nonnegative")
    expected_targets = set(expected_target_indices)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["parent"])].append(record)
    for parent, group in groups.items():
        if len(group) != sweep_group_size(len(expected_target_indices)):
            raise ValueError(
                "group does not contain one base plus complete target sweeps: "
                f"{parent}"
            )
        if sum(record["kind"] == "base" for record in group) != 1:
            raise ValueError(f"group does not contain one canonical base: {parent}")
        derived = [record for record in group if record["kind"] == "sweep"]
        target_ids_by_index: dict[int, str] = {}
        by_target: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in derived:
            raw_target_index = record.get("target_object_index")
            if isinstance(raw_target_index, bool) or not isinstance(
                raw_target_index, int
            ):
                raise ValueError(f"target object index is invalid: {parent}")
            target_index = raw_target_index
            target_id = record.get("target_object_id")
            if not isinstance(target_id, str) or not target_id:
                raise ValueError(f"target object id is invalid: {parent}")
            previous = target_ids_by_index.setdefault(target_index, target_id)
            if previous != target_id:
                raise ValueError(
                    f"target object identity differs within group: {parent}/{target_index}"
                )
            by_target[target_index].append(record)
        if set(by_target) != expected_targets:
            raise ValueError(f"group target coverage differs: {parent}")
        if len(set(target_ids_by_index.values())) != len(target_ids_by_index):
            raise ValueError(f"group target object ids are not unique: {parent}")
        for target_index, target_records in by_target.items():
            validate_target_sweep_grid(
                target_records, label=f"{parent}/{target_index}"
            )
    return SweepGroupSummary(
        base_count=len(groups),
        target_groups_per_base=len(expected_target_indices),
        derived_count=len(groups)
        * len(expected_target_indices)
        * SWEEP_VARIANTS_PER_TARGET,
    )


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
