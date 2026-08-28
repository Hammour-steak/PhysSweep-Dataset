#!/usr/bin/env python3
"""Publish and verify the canonical one-object dataset from a frozen release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from tools.release.base_release_view import (
    PipelineSpec,
    build_view as build_base_view,
    one_object_release_roots,
    verify_view as verify_base_view,
)
from tools.release.sweep_release_view import (
    build_view as build_sweep_view,
    verify_view as verify_sweep_view,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("configs/datasets/one_object.json")
CONFIG_SCHEMA = "physweep_dataset_build_v2"
EXPECTED_OBJECT_COUNT = 1
EXPECTED_CONFIG_KEYS = {"schema_version", "release_root", "object_count"}
FORBIDDEN_KEYS = {
    "cache_root",
    "checkpoint",
    "lora_rank",
    "scene_tokens",
    "training_steps",
    "wan_repo",
}


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(
            *(_walk_keys(item) for item in value.values()), set()
        )
    if isinstance(value, list):
        return set().union(*(_walk_keys(item) for item in value), set())
    return set()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != EXPECTED_CONFIG_KEYS:
        raise ValueError("one-object dataset config fields differ")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported dataset build config")
    if config.get("release_root") != "outputs/one_object":
        raise ValueError("one-object release_root must be outputs/one_object")
    if config.get("object_count") != EXPECTED_OBJECT_COUNT:
        raise ValueError("one-object dataset must contain exactly one object per sample")
    forbidden = sorted(_walk_keys(config) & FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError(f"model-only keys in dataset config: {', '.join(forbidden)}")
    return config


def pipeline_specs(
    values: Iterable[tuple[str, str, str, str, str]],
) -> list[PipelineSpec]:
    return [
        PipelineSpec(name, schema, Path(project), Path(render), Path(masks))
        for name, schema, project, render, masks in values
    ]


def publish_dataset(
    *,
    release_project_root: Path,
    release_manifest: Path,
    release_root: Path,
    base_specs: Iterable[PipelineSpec],
    sweep_specs: Iterable[PipelineSpec],
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    base_root, sweep_root = one_object_release_roots(release_root)
    base_specs = list(base_specs)
    sweep_specs = list(sweep_specs)
    if base_root.exists():
        base = verify_base_view(base_root, expected_object_count=EXPECTED_OBJECT_COUNT)
    else:
        base = build_base_view(
            release_project_root=release_project_root,
            release_manifest=release_manifest,
            output=base_root,
            pipeline_specs=base_specs,
            expected_object_count=EXPECTED_OBJECT_COUNT,
        )
    if sweep_root.exists():
        sweep = verify_sweep_view(
            sweep_root,
            base_root=base_root,
            expected_object_count=EXPECTED_OBJECT_COUNT,
        )
    else:
        sweep = build_sweep_view(
            release_project_root=release_project_root,
            release_manifest=release_manifest,
            base_root=base_root,
            output=sweep_root,
            pipeline_specs=sweep_specs,
            workers=workers,
            resume=resume,
            expected_object_count=EXPECTED_OBJECT_COUNT,
        )
    return {"release_root": str(release_root), "base": base, "sweep": sweep}


def verify_dataset(release_root: Path) -> dict[str, Any]:
    base_root, sweep_root = one_object_release_roots(release_root)
    return {
        "release_root": str(release_root),
        "base": verify_base_view(
            base_root, expected_object_count=EXPECTED_OBJECT_COUNT
        ),
        "sweep": verify_sweep_view(
            sweep_root,
            base_root=base_root,
            expected_object_count=EXPECTED_OBJECT_COUNT,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release-project-root", type=Path)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument(
        "--base-pipeline",
        nargs=5,
        action="append",
        metavar=("NAME", "SOURCE_SCHEMA", "PROJECT_ROOT", "RENDER_ROOT", "MASK_ROOT"),
    )
    parser.add_argument(
        "--sweep-pipeline",
        nargs=5,
        action="append",
        metavar=("NAME", "SOURCE_SCHEMA", "PROJECT_ROOT", "RENDER_ROOT", "MASK_ROOT"),
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_config(config_path)
    release_root = root / config["release_root"]
    if args.verify_only:
        result = verify_dataset(release_root)
    else:
        if args.release_project_root is None or args.release_manifest is None:
            raise SystemExit(
                "--release-project-root and --release-manifest are required when publishing"
            )
        if not args.base_pipeline or not args.sweep_pipeline:
            raise SystemExit(
                "at least one --base-pipeline and --sweep-pipeline are required"
            )
        result = publish_dataset(
            release_project_root=args.release_project_root,
            release_manifest=args.release_manifest,
            release_root=release_root,
            base_specs=pipeline_specs(args.base_pipeline),
            sweep_specs=pipeline_specs(args.sweep_pipeline),
            workers=args.workers,
            resume=args.resume,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
