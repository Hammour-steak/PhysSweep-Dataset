"""Project-root path resolution with explicit external-path semantics."""

from __future__ import annotations

from pathlib import Path


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
