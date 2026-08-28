"""Streaming file hashes used by manifests and provenance checks."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    """Hash the canonical compact JSON encoding used by immutable bindings."""

    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_json_without_field(value: dict[str, Any], field: str) -> str:
    """Hash a JSON record after omitting one self-referential field."""

    payload = copy.deepcopy(value)
    payload.pop(field, None)
    return sha256_json(payload)


def sha256_json_binding(value: dict[str, Any]) -> str:
    """Hash an immutable binding without its self-referential digest."""

    return sha256_json_without_field(value, "binding_sha256")


def relative_file_binding(root: Path, path: Path) -> dict[str, str]:
    """Bind a project-local file by relative path and content hash."""

    root = root.resolve()
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": sha256_file(resolved),
    }
