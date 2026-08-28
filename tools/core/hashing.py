"""Streaming file hashes used by manifests and provenance checks."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_file_binding(root: Path, path: Path) -> dict[str, str]:
    """Bind a project-local file by relative path and content hash."""

    root = root.resolve()
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": sha256_file(resolved),
    }
