#!/usr/bin/env python3
"""Publish or verify the canonical two-object dataset."""

from __future__ import annotations

from pathlib import Path

from tools.cli.build_object_dataset import (
    load_config as load_object_config,
    pipeline_specs,
    publish_dataset as publish_object_dataset,
    run_cli,
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


def main() -> None:
    run_cli(
        description=__doc__ or "",
        default_config=DEFAULT_CONFIG,
        project_root=PROJECT_ROOT,
        load_config_fn=load_config,
        publish_dataset_fn=publish_dataset,
        verify_dataset_fn=verify_dataset,
    )


if __name__ == "__main__":
    main()
