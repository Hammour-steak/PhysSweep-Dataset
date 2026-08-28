#!/usr/bin/env python3
"""Run the configured one-object dataset pipeline without model dependencies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("configs/datasets/one_object.json")
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
        return set(value) | set().union(*(_walk_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_walk_keys(item) for item in value), set())
    return set()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "physweep_dataset_build_v1":
        raise ValueError("unsupported dataset build config")
    forbidden = sorted(_walk_keys(config) & FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError(f"model-only keys in dataset config: {', '.join(forbidden)}")
    if not config.get("stages"):
        raise ValueError("dataset config contains no stages")
    return config


def _resolve_command(root: Path, stage: dict[str, Any]) -> list[str]:
    variables = {
        "root": str(root),
        "python": sys.executable,
        **stage.get("variables", {}),
    }
    return [str(token).format_map(variables) for token in stage["command"]]


def _selected_stages(
    stages: list[dict[str, Any]], from_stage: str | None, until_stage: str | None
) -> list[dict[str, Any]]:
    names = [stage["name"] for stage in stages]
    start = names.index(from_stage) if from_stage else 0
    stop = names.index(until_stage) + 1 if until_stage else len(stages)
    if start >= stop:
        raise ValueError("--from-stage must not follow --until-stage")
    return stages[start:stop]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--from-stage")
    parser.add_argument("--until-stage")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_config(config_path)
    stages = _selected_stages(config["stages"], args.from_stage, args.until_stage)
    print(f"dataset_id: {config['dataset_id']}")
    print(f"release_root: {config['release_root']}")
    for stage in stages:
        command = _resolve_command(root, stage)
        print(f"[{stage['name']}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
