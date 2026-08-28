#!/usr/bin/env python3
"""Shared contracts for deterministic specialized release extensions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SPEC_SCHEMA = "physweep_specialized_release_extension_v1"


def project_root_reference(root: Path, project_root: Path) -> str:
    try:
        relative = project_root.resolve().relative_to(root.resolve())
    except ValueError:
        return str(project_root.resolve())
    return relative.as_posix() or "."


def load_extension_spec(root: Path, path: Path) -> dict[str, Any]:
    root = root.resolve()
    declared = path if path.is_absolute() else root / path
    declared.absolute().relative_to(root)
    resolved = declared.resolve()
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if document.get("schema_version") != SPEC_SCHEMA:
        raise ValueError("unsupported specialized release extension spec")
    source = document["source_release"]
    target = document["target_release"]
    group = document["group_contract"]
    replacement = document["replacement"]
    source_counts = Counter(
        {str(key): int(value) for key, value in source["pipeline_group_counts"].items()}
    )
    target_counts = Counter(
        {str(key): int(value) for key, value in target["pipeline_group_counts"].items()}
    )
    group_count = int(group["base_group_count"])
    replacement_count = int(group["replacement_group_count"])
    if (
        sum(source_counts.values()) != group_count
        or sum(target_counts.values()) != group_count
        or int(group["samples_per_group"]) != 13
        or int(group["sample_count"]) != 13 * group_count
        or replacement_count <= 0
    ):
        raise ValueError("specialized release group counts are inconsistent")
    expected_target = source_counts.copy()
    expected_target[str(replacement["source_pipeline"])] -= replacement_count
    expected_target[str(replacement["pipeline"])] += replacement_count
    if expected_target != target_counts or min(target_counts.values()) < 0:
        raise ValueError("target pipeline counts do not match the replacement")
    profiles = [str(value) for value in replacement["profiles"]]
    if (
        not profiles
        or len(profiles) != len(set(profiles))
        or replacement_count % len(profiles)
    ):
        raise ValueError("replacement profiles cannot be balanced exactly")
    preserved = [str(value) for value in group["preserved_slot_fields"]]
    if preserved != ["index", "motion_intent"]:
        raise ValueError("specialized replacement must preserve index and motion intent")
    for key in (
        "pipeline",
        "generator",
        "environment_id",
        "scene_schema_version",
        "backend_config",
        "generator_script",
        "scene_id_template",
        "selection_namespace",
        "candidate_seed_namespace",
    ):
        if not str(replacement.get(key, "")).strip():
            raise ValueError(f"specialized replacement lacks {key}")
    return document


def stable_seed(namespace: str) -> int:
    value = int.from_bytes(
        hashlib.sha256(namespace.encode("utf-8")).digest()[:8],
        byteorder="big",
    )
    return value % (2**31 - 2) + 1


def select_replacement_slots(
    records: list[dict[str, Any]],
    count: int,
    seed: int,
    *,
    namespace: str,
    source_pipeline: str,
    motion_intent: str,
) -> list[dict[str, Any]]:
    candidates = [
        record
        for record in records
        if record.get("pipeline") == source_pipeline
        and record.get("motion_intent") == motion_intent
    ]
    if len(candidates) < count:
        raise ValueError(
            f"replacement needs {count} eligible slots, got {len(candidates)}"
        )

    def rank(record: dict[str, Any]) -> str:
        payload = ":".join(
            (
                namespace,
                str(seed),
                str(record["index"]),
                str(record["scene_id"]),
                str(record["metadata_sha256"]),
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    chosen = sorted(
        candidates,
        key=lambda record: (rank(record), int(record["index"])),
    )[:count]
    return sorted(chosen, key=lambda record: int(record["index"]))


def index_replacements(
    source_base: dict[str, Any],
    replacement_manifest: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[dict[int, dict[str, Any]], set[str], set[str]]:
    replacement = spec["replacement"]
    group = spec["group_contract"]
    expected_count = int(group["replacement_group_count"])
    source_by_index = {
        int(record["index"]): record for record in source_base["records"]
    }
    if len(source_by_index) != len(source_base["records"]):
        raise ValueError("source base manifest contains duplicate indices")
    records = list(replacement_manifest["records"])
    if (
        int(replacement_manifest["sample_count"]) != len(records)
        or len(records) != expected_count
    ):
        raise ValueError(f"extension requires exactly {expected_count} replacements")
    indexed: dict[int, dict[str, Any]] = {}
    old_parents: set[str] = set()
    new_parents: set[str] = set()
    for record in records:
        index = int(record["index"])
        original = source_by_index.get(index)
        if original is None or index in indexed:
            raise ValueError(f"invalid or duplicate replacement index: {index}")
        if (
            original.get("pipeline") != replacement["source_pipeline"]
            or original.get("motion_intent")
            != replacement["source_motion_intent"]
            or record.get("pipeline") != replacement["pipeline"]
            or record.get("motion_intent") != original.get("motion_intent")
            or record.get("replaces_scene_id") != original.get("scene_id")
            or record.get("replaces_metadata_path") != original.get("metadata_path")
            or record.get("replaces_metadata_sha256")
            != original.get("metadata_sha256")
            or record.get("replaces_pipeline") != original.get("pipeline")
        ):
            raise ValueError(f"replacement changes its preserved slot: {index}")
        indexed[index] = record
        old_parents.add(str(original["metadata_path"]))
        new_parents.add(str(record["metadata_path"]))
    if len(old_parents) != expected_count or len(new_parents) != expected_count:
        raise ValueError("replacement parent metadata paths are not unique")
    return indexed, old_parents, new_parents
