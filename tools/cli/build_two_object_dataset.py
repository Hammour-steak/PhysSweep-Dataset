#!/usr/bin/env python3
"""Publish or verify the canonical two-object dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.cli.build_object_dataset import (
    load_config as load_object_config,
    pipeline_specs,
    publish_dataset as publish_object_dataset,
    verify_dataset as verify_object_dataset,
)
from tools.release.base_release_view import (
    build_view as build_base_view,
    verify_view as verify_base_view,
)
from tools.release.sweep_release_view import (
    build_view as build_sweep_view,
    verify_view as verify_sweep_view,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("configs/datasets/two_object.json")
EXPECTED_OBJECT_COUNT = 2


def load_config(path: Path):
    return load_object_config(path, expected_object_count=EXPECTED_OBJECT_COUNT)


def publish_dataset(**kwargs):
    return publish_object_dataset(
        **kwargs,
        expected_object_count=EXPECTED_OBJECT_COUNT,
        build_base_view_fn=build_base_view,
        verify_base_view_fn=verify_base_view,
        build_sweep_view_fn=build_sweep_view,
        verify_sweep_view_fn=verify_sweep_view,
    )


def verify_dataset(release_root: Path):
    return verify_object_dataset(
        release_root,
        expected_object_count=EXPECTED_OBJECT_COUNT,
        verify_base_view_fn=verify_base_view,
        verify_sweep_view_fn=verify_sweep_view,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--release-project-root", type=Path)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument(
        "--pipeline",
        nargs=4,
        action="append",
        metavar=("NAME", "SOURCE_SCHEMA", "PROJECT_ROOT", "RENDER_ROOT"),
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
            raise SystemExit("at least one --pipeline is required")
        result = publish_dataset(
            release_project_root=args.release_project_root,
            release_manifest=args.release_manifest,
            release_root=release_root,
            pipeline_specs=pipeline_specs(args.pipeline),
            workers=args.workers,
            resume=args.resume,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
