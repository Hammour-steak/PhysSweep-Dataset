"""Project-root path resolution with explicit external-path semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def join_project_path(root: Path, value: str | Path) -> Path:
    """Join a project-relative path without canonicalizing either component."""

    path = Path(value)
    return path if path.is_absolute() else root / path


def resolve_project_path(root: Path, value: str | Path) -> Path:
    """Resolve an absolute path or a path relative to ``root``."""

    return join_project_path(root, value).resolve()


def resolve_project_path_within_root(root: Path, value: str | Path) -> Path:
    """Resolve a path and reject values that escape ``root``."""

    root = root.resolve()
    resolved = resolve_project_path(root, value)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path is outside project root: {resolved}") from exc
    return resolved


def project_relative_path(root: Path, path: str | Path) -> str:
    """Return a portable project-relative path and reject external paths."""

    root = root.resolve()
    resolved = resolve_project_path(root, path)
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside project root: {resolved}") from exc


def safe_scene_id(value: Any) -> str:
    """Return a scene identifier that is safe as one path component."""

    scene_id = str(value)
    if not scene_id or Path(scene_id).name != scene_id or scene_id in {".", ".."}:
        raise ValueError(f"invalid scene id: {scene_id!r}")
    return scene_id
