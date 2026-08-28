"""Canonical dataset directory names and sibling release roots."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_COUNT_WORDS = {1: "one", 2: "two", 3: "three"}


def dataset_directory_name(object_count: int) -> str:
    if isinstance(object_count, bool) or not isinstance(object_count, int):
        raise TypeError("dataset object_count must be an integer")
    if object_count < 1:
        raise ValueError("dataset object_count must be positive")
    prefix = _COUNT_WORDS.get(object_count, str(object_count))
    return f"{prefix}_object"


def release_roots(
    release_root: Path,
    *,
    object_count: int,
    project_root: Path = PROJECT_ROOT,
) -> tuple[Path, Path]:
    release_root = Path(release_root)
    if not release_root.is_absolute():
        release_root = project_root / release_root
    release_root = release_root.resolve(strict=False)
    expected_name = dataset_directory_name(object_count)
    if release_root.name != expected_name:
        raise ValueError(f"release root must be named {expected_name}")
    return release_root / "base", release_root / "sweep"
