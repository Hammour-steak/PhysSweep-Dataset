"""Project-root path resolution with explicit external-path semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_project_path(root: Path, value: str | Path) -> Path:
    """Resolve an absolute path or a path relative to ``root``."""

    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def resolve_project_path_within_root(root: Path, value: str | Path) -> Path:
    """Resolve a path and reject values that escape ``root``."""

    root = root.resolve()
    resolved = resolve_project_path(root, value)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path is outside project root: {resolved}") from exc
    return resolved


def safe_scene_id(value: Any) -> str:
    """Return a scene identifier that is safe as one path component."""

    scene_id = str(value)
    if not scene_id or Path(scene_id).name != scene_id or scene_id in {".", ".."}:
        raise ValueError(f"invalid scene id: {scene_id!r}")
    return scene_id
