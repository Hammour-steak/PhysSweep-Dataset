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

    record_path = root / str(metadata["physics"]["simulation_record_path"])
    record = load_json(record_path)
    if record.get("schema_version") != SIMULATION_RECORD_VERSION:
        raise ValueError("unsupported simulation record version")
    if record.get("scene_id") != metadata.get("scene_id"):
        raise ValueError("simulation record scene id mismatch")
    expected_metadata_path = project_relative(root, metadata_path)
    if record["metadata"]["path"] != expected_metadata_path:
        raise ValueError("simulation record metadata path mismatch")
    if sha256(metadata_path) != record["metadata"]["sha256"]:
        raise ValueError("metadata changed after simulation")

    trajectory_path = root / str(record["trajectory"]["path"])
    audit_path = root / str(record["audit"]["path"])
    expected_trajectory_path = str(metadata["physics"]["trajectory_path"])
    expected_audit_path = str(metadata["physics"]["audit_path"])
    if record["trajectory"]["path"] != expected_trajectory_path:
        raise ValueError("trajectory path differs from frozen metadata")
    if record["audit"]["path"] != expected_audit_path:
        raise ValueError("audit path differs from frozen metadata")
    for label, path in (("trajectory", trajectory_path), ("audit", audit_path)):
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != record[label]["sha256"]:
            raise ValueError(f"{label} changed after simulation")
    return record, trajectory_path, audit_path
