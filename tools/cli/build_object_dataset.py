#!/usr/bin/env python3
"""Shared publisher for canonical object-count-aware dataset views."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from tools.core.hashing import sha256_file
from tools.core.paths import resolve_project_path_within_root
from tools.release.base_release_view import (
    PipelineSpec,
    build_view as build_base_view,
    verify_view as verify_base_view,
)
from tools.release.layout import dataset_directory_name, release_roots
from tools.release.sweep_release_view import (
    build_view as build_sweep_view,
    verify_view as verify_sweep_view,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_SCHEMA = "physweep_dataset_build_v2"
EXPECTED_CONFIG_KEYS = {"schema_version", "release_root", "object_count"}


def load_config(path: Path, *, expected_object_count: int) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != EXPECTED_CONFIG_KEYS:
        raise ValueError("dataset config fields differ")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported dataset build config")
    expected_root = f"outputs/{dataset_directory_name(expected_object_count)}"
    if config.get("release_root") != expected_root:
        raise ValueError(f"dataset release_root must be {expected_root}")
    if config.get("object_count") != expected_object_count:
        raise ValueError(
            f"dataset must contain exactly {expected_object_count} objects per sample"
        )
    return config


def pipeline_specs(
    values: Iterable[tuple[str, str, str, str]],
) -> list[PipelineSpec]:
    return [
        PipelineSpec(name, schema, Path(project), Path(render))
        for name, schema, project, render in values
    ]


def source_release_binding(
    release_project_root: Path, release_manifest: Path
) -> dict[str, str]:
    root = release_project_root.resolve()
    manifest_path = resolve_project_path_within_root(root, release_manifest)
    release = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_path = resolve_project_path_within_root(
        root, release["metadata_manifest"]
    )
    return {
        "metadata_sha256": sha256_file(metadata_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def require_source_binding(
    release_root: Path, expected: dict[str, str]
) -> None:
    base = json.loads(
        (release_root / "base" / "manifest.json").read_text(encoding="utf-8")
    )
    base_provenance = base.get("provenance", {})
    observed_base = {
        "metadata_sha256": base_provenance.get(
            "source_generation_release_metadata", {}
        ).get("manifest_sha256"),
        "manifest_sha256": base_provenance.get("source_sweep_release", {}).get(
            "manifest_sha256"
        ),
    }
    if observed_base != expected:
        raise ValueError("existing base release belongs to a different source release")
    sweep_path = release_root / "sweep" / "manifest.json"
    if not sweep_path.is_file():
        return
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    sweep_provenance = sweep.get("provenance", {})
    observed_sweep = {
        "metadata_sha256": sweep_provenance.get(
            "source_generation_release_metadata_sha256"
        ),
        "manifest_sha256": sweep_provenance.get(
            "source_sweep_release_manifest_sha256"
        ),
    }
    if observed_sweep != expected:
        raise ValueError("existing sweep release belongs to a different source release")


def publish_dataset(
    *,
    release_project_root: Path,
    release_manifest: Path,
    release_root: Path,
    pipeline_specs: Iterable[PipelineSpec],
    workers: int,
    resume: bool,
    expected_object_count: int,
    build_base_view_fn=build_base_view,
    verify_base_view_fn=verify_base_view,
    build_sweep_view_fn=build_sweep_view,
    verify_sweep_view_fn=verify_sweep_view,
) -> dict[str, Any]:
    base_root, sweep_root = release_roots(
        release_root, object_count=expected_object_count
    )
    base_exists = base_root.exists() or base_root.is_symlink()
    sweep_exists = sweep_root.exists() or sweep_root.is_symlink()
    if sweep_exists and not base_exists:
        raise ValueError("sweep release exists without its sibling base release")
    if (base_exists or sweep_exists) and not resume:
        raise FileExistsError(f"canonical release already exists: {release_root}")
    specs = list(pipeline_specs)
    if base_exists:
        require_source_binding(
            base_root.parent,
            source_release_binding(release_project_root, release_manifest),
        )
        base = verify_base_view_fn(
            base_root, expected_object_count=expected_object_count
        )
    else:
        base = build_base_view_fn(
            release_project_root=release_project_root,
            release_manifest=release_manifest,
            output=base_root,
            pipeline_specs=specs,
            expected_object_count=expected_object_count,
        )
    if sweep_exists:
        sweep = verify_sweep_view_fn(
            sweep_root,
            base_root=base_root,
            expected_object_count=expected_object_count,
        )
    else:
        sweep = build_sweep_view_fn(
            release_project_root=release_project_root,
            release_manifest=release_manifest,
            base_root=base_root,
            output=sweep_root,
            pipeline_specs=specs,
            workers=workers,
            resume=resume,
            expected_object_count=expected_object_count,
        )
    return {"release_root": str(release_root), "base": base, "sweep": sweep}


def verify_dataset(
    release_root: Path,
    *,
    expected_object_count: int,
    verify_base_view_fn=verify_base_view,
    verify_sweep_view_fn=verify_sweep_view,
) -> dict[str, Any]:
    base_root, sweep_root = release_roots(
        release_root, object_count=expected_object_count
    )
    return {
        "release_root": str(release_root),
        "base": verify_base_view_fn(
            base_root, expected_object_count=expected_object_count
        ),
        "sweep": verify_sweep_view_fn(
            sweep_root,
            base_root=base_root,
            expected_object_count=expected_object_count,
        ),
    }
