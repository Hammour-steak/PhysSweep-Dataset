"""Shared filesystem and subprocess primitives for object-count generators."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.core.json_io import read_json, write_json_atomic_sorted
from tools.core.paths import safe_scene_id
from tools.release.layout import dataset_directory_name


@dataclass(frozen=True)
class Layout:
    base_dataset: Path
    base_manifest: Path
    sweep_metadata: Path
    sweep_physics: Path
    source_release: Path
    base_render: Path
    sweep_render: Path
    canonical_release: Path


def generation_layout(
    root: Path,
    work_id: str,
    release_root: Path,
    *,
    object_count: int,
) -> Layout:
    work_id = safe_scene_id(work_id)
    root = root.resolve()
    canonical = release_root if release_root.is_absolute() else root / release_root
    canonical = canonical.resolve()
    expected_name = dataset_directory_name(object_count)
    if (
        canonical.name != expected_name
        or (root / "outputs").resolve() not in canonical.parents
    ):
        raise ValueError(
            f"canonical release must be an outputs/.../{expected_name} directory"
        )
    base_dataset = root / "datasets" / work_id / "base"
    return Layout(
        base_dataset=base_dataset,
        base_manifest=base_dataset / "manifest.json",
        sweep_metadata=root / "datasets" / work_id / "sweep" / "metadata",
        sweep_physics=root / "datasets" / work_id / "sweep" / "physics",
        source_release=root / "datasets" / work_id / "release",
        base_render=root / "outputs" / work_id / "base",
        sweep_render=root / "outputs" / work_id / "sweep",
        canonical_release=canonical,
    )


def bind_generation_plan(path: Path, plan: dict[str, Any], resume: bool) -> None:
    if path.is_file():
        if not resume:
            raise FileExistsError(f"generation plan already exists: {path}")
        if read_json(path) != plan:
            raise ValueError("resume request differs from the frozen generation plan")
        return
    write_json_atomic_sorted(path, plan)


def run(command: list[str], root: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def run_once(
    command: list[str],
    *,
    root: Path,
    completion: Path,
    resume: bool,
) -> None:
    if completion.is_file():
        if not resume:
            raise FileExistsError(f"stage already exists; pass --resume: {completion}")
        return
    run(command, root)
    if not completion.is_file():
        raise RuntimeError(f"stage did not create its completion artifact: {completion}")


def verify_render_manifest(path: Path, expected_count: int) -> None:
    manifest = read_json(path)
    if (
        int(manifest.get("sample_count", -1)) != expected_count
        or int(manifest.get("failure_count", -1)) != 0
        or int(manifest.get("success_count", -1)) != expected_count
    ):
        raise ValueError(f"render manifest is incomplete: {path}")
