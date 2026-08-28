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
from tools.release.build_sweep_release_view import (
    build_view as build_sweep_view,
    verify_view as verify_sweep_view,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("configs/datasets/one_object.json")
CONFIG_SCHEMA = "physweep_dataset_build_v2"
EXPECTED_VIEWS = ["base", "sweep"]
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
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported dataset build config")
    if config.get("dataset_id") != "physweep_one_object":
        raise ValueError("unexpected one-object dataset id")
    if config.get("release_root") != "outputs/one_object":
        raise ValueError("one-object release_root must be outputs/one_object")
    if config.get("object_count") != 1 or config.get("views") != EXPECTED_VIEWS:
        raise ValueError("one-object dataset must publish base and sweep views")
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
    specs: Iterable[PipelineSpec],
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    base_root, sweep_root = one_object_release_roots(release_root)
    specs = list(specs)
    if base_root.exists():
        base = verify_base_view(base_root)
    else:
        base = build_base_view(
            release_project_root=release_project_root,
            release_manifest=release_manifest,
            output=base_root,
            pipeline_specs=specs,
        )
    if sweep_root.exists():
        sweep = verify_sweep_view(sweep_root, base_root=base_root)
    else:
        sweep = build_sweep_view(
            release_project_root=release_project_root,
            release_manifest=release_manifest,
            base_root=base_root,
            output=sweep_root,
            pipeline_specs=specs,
            workers=workers,
            resume=resume,
        )
    return {"release_root": str(release_root), "base": base, "sweep": sweep}


def verify_dataset(release_root: Path) -> dict[str, Any]:
    base_root, sweep_root = one_object_release_roots(release_root)
    return {
        "release_root": str(release_root),
        "base": verify_base_view(base_root),
        "sweep": verify_sweep_view(sweep_root, base_root=base_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release-project-root", type=Path)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument(
        "--pipeline",
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
        if not args.pipeline:
            raise SystemExit("at least one --pipeline is required when publishing")
        result = publish_dataset(
            release_project_root=args.release_project_root,
            release_manifest=args.release_manifest,
            release_root=release_root,
            specs=pipeline_specs(args.pipeline),
            workers=args.workers,
            resume=args.resume,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
