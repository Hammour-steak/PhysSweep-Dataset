"""Deterministic JSON file I/O for dataset-generation tools."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    """Read one UTF-8 JSON value."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(
    path: Path,
    value: Any,
    *,
    atomic: bool = False,
    sort_keys: bool = False,
) -> None:
    """Write stable UTF-8 JSON, optionally replacing the destination atomically."""

    path.parent.mkdir(parents=True, exist_ok=True)
    contents = (
        json.dumps(
            value,
            indent=2,
            ensure_ascii=True,
            sort_keys=sort_keys,
        )
        + "\n"
    )
    if not atomic:
        path.write_text(contents, encoding="utf-8")
        return

    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_text(contents, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON with atomic replacement while preserving key order."""

    write_json(path, value, atomic=True)


def write_json_sorted(path: Path, value: Any) -> None:
    """Write JSON with deterministic lexical key ordering."""

    write_json(path, value, sort_keys=True)


def write_json_atomic_sorted(path: Path, value: Any) -> None:
    """Write sorted JSON with atomic replacement."""

    write_json(path, value, atomic=True, sort_keys=True)
