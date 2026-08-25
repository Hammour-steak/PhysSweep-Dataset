#!/usr/bin/env python3
"""Load the single declarative registry for specialized scene backends."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = Path("configs/specialized_scene_backends.json")
REQUIRED_FIELDS = (
    "pipeline",
    "source_schema_version",
    "sweep_branch",
    "renderer_id",
    "renderer_script",
    "render_manifest_schema",
    "render_manifest_name",
)


def load_specialized_backends(
    root: Path = PROJECT_ROOT,
    path: Path | str = DEFAULT_REGISTRY,
) -> list[dict[str, Any]]:
    registry_path = Path(path)
    if not registry_path.is_absolute():
        registry_path = root / registry_path
    document = json.loads(registry_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "physweep_specialized_scene_backends_v1":
        raise ValueError("unsupported specialized backend registry")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("specialized backend registry contains no records")
    result: list[dict[str, Any]] = []
    for index, value in enumerate(records):
        source = value if isinstance(value, dict) else {}
        missing = [field for field in REQUIRED_FIELDS if field not in source]
        if missing:
            raise ValueError(f"specialized backend record {index} lacks: {missing}")
        record = {field: str(value[field]) for field in REQUIRED_FIELDS}
        renderer = root / record["renderer_script"]
        if not renderer.is_file():
            raise FileNotFoundError(renderer)
        result.append(record)
    for field in ("pipeline", "source_schema_version", "sweep_branch", "renderer_id"):
        values = [record[field] for record in result]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate specialized backend {field}")
    return result


def specialized_by_pipeline(root: Path = PROJECT_ROOT) -> dict[str, dict[str, Any]]:
    return {record["pipeline"]: record for record in load_specialized_backends(root)}


def specialized_by_schema(root: Path = PROJECT_ROOT) -> dict[str, dict[str, Any]]:
    return {
        record["source_schema_version"]: record
        for record in load_specialized_backends(root)
    }
