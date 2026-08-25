#!/usr/bin/env python3
"""Bind immutable scene metadata to simulation artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SIMULATION_RECORD_VERSION = "physweep_simulation_record_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def project_relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def freeze_metadata(path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    """Write metadata once, then return the exact bytes that downstream reads."""

    write_json(path, metadata)
    frozen = load_json(path)
    if frozen != metadata:
        raise RuntimeError(f"metadata changed while being frozen: {path}")
    return frozen


def write_simulation_record(
    *,
    root: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    trajectory_path: Path,
    audit_path: Path,
    record_path: Path,
) -> dict[str, Any]:
    """Seal one simulation against its frozen metadata and output files."""

    if load_json(metadata_path) != metadata:
        raise ValueError("simulation metadata differs from the frozen source")
    record = {
        "schema_version": SIMULATION_RECORD_VERSION,
        "scene_id": str(metadata["scene_id"]),
        "metadata": {
            "path": project_relative(root, metadata_path),
            "sha256": sha256(metadata_path),
        },
        "trajectory": {
            "path": project_relative(root, trajectory_path),
            "sha256": sha256(trajectory_path),
        },
        "audit": {
            "path": project_relative(root, audit_path),
            "sha256": sha256(audit_path),
        },
    }
    write_json(record_path, record)
    return record


def validate_simulation_record(
    *,
    root: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], Path, Path]:
    """Verify metadata and both simulation artifacts before rendering."""

    source_binding = metadata.get("source_metadata")
    source_metadata_path = metadata_path
    if source_binding is not None:
        source_metadata_path = root / str(source_binding["path"])
        if sha256(source_metadata_path) != str(source_binding["sha256"]):
            raise ValueError("source metadata changed after visual binding")

    record_path = root / str(metadata["physics"]["simulation_record_path"])
    record = load_json(record_path)
    version = record.get("schema_version")
    if version not in {
        SIMULATION_RECORD_VERSION,
        "physweep_dispatched_simulation_record_v1",
    }:
        raise ValueError("unsupported simulation record version")
    if record.get("scene_id") != metadata.get("scene_id"):
        raise ValueError("simulation record scene id mismatch")
    expected_metadata_path = project_relative(root, source_metadata_path)
    record_metadata_path = (
        record["metadata"]["path"]
        if version == SIMULATION_RECORD_VERSION
        else project_relative(root, Path(str(record["metadata_path"])))
    )
    record_metadata_sha256 = (
        record["metadata"]["sha256"]
        if version == SIMULATION_RECORD_VERSION
        else str(record["metadata_sha256"])
    )
    if record_metadata_path != expected_metadata_path:
        raise ValueError("simulation record metadata path mismatch")
    if sha256(source_metadata_path) != record_metadata_sha256:
        raise ValueError("metadata changed after simulation")

    record_trajectory_path = (
        record["trajectory"]["path"]
        if version == SIMULATION_RECORD_VERSION
        else project_relative(root, Path(str(record["trajectory_path"])))
    )
    record_trajectory_sha256 = (
        record["trajectory"]["sha256"]
        if version == SIMULATION_RECORD_VERSION
        else str(record["trajectory_sha256"])
    )
    record_audit_path = (
        record["audit"]["path"]
        if version == SIMULATION_RECORD_VERSION
        else project_relative(root, Path(str(record["audit_path"])))
    )
    record_audit_sha256 = (
        record["audit"]["sha256"]
        if version == SIMULATION_RECORD_VERSION
        else str(record["audit_sha256"])
    )
    trajectory_path = root / record_trajectory_path
    audit_path = root / record_audit_path
    expected_trajectory_path = str(metadata["physics"]["trajectory_path"])
    expected_audit_path = str(metadata["physics"]["audit_path"])
    if record_trajectory_path != expected_trajectory_path:
        raise ValueError("trajectory path differs from frozen metadata")
    if record_audit_path != expected_audit_path:
        raise ValueError("audit path differs from frozen metadata")
    for label, path, expected_sha256 in (
        ("trajectory", trajectory_path, record_trajectory_sha256),
        ("audit", audit_path, record_audit_sha256),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != expected_sha256:
            raise ValueError(f"{label} changed after simulation")
    normalized_record = dict(record)
    normalized_record.setdefault(
        "metadata",
        {"path": record_metadata_path, "sha256": record_metadata_sha256},
    )
    normalized_record.setdefault(
        "trajectory",
        {"path": record_trajectory_path, "sha256": record_trajectory_sha256},
    )
    normalized_record.setdefault(
        "audit",
        {"path": record_audit_path, "sha256": record_audit_sha256},
    )
    return normalized_record, trajectory_path, audit_path
