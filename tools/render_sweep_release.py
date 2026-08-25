#!/usr/bin/env python3
"""Run one explicit rendering stage for a published one-object sweep release."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_NAMES = (
    "prepare_base_render_plan",
    "prepare_sweep_render_plan",
    "bind_generic_base_cameras",
    "bind_generic_sweep_visuals",
    "render_asset_bases",
    "freeze_asset_sweep_cameras",
    "render_asset_sweeps",
    "render_billiards_sweeps",
    "render_generic_sweeps",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--release-manifest",
        type=Path,
        default=Path("datasets/one_object_sweep/release/manifest.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/one_object_sweep_release"),
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--gpus", default="4,5,7")
    parser.add_argument("--stage", choices=STAGE_NAMES, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    root = args.root.resolve()
    release_path = (root / args.release_manifest).resolve()
    release_path.relative_to(root / "datasets")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    output = (root / args.output_root).resolve()
    if (root / "outputs").resolve() not in output.parents:
        raise ValueError("render output must remain under root/outputs")
    status_path = output.parent / f"{output.name}_{args.stage}_status.json"
    python = sys.executable
    stage_commands = dict([
        (
            "prepare_base_render_plan",
            [python, "tools/prepare_formal_render_manifests.py", "--root", str(root), "--manifest", str(root / release["base_manifest"]), "--output-root", str(output), "--selection", "all"],
        ),
        (
            "prepare_sweep_render_plan",
            [python, "tools/prepare_sweep_render_manifests.py", "--root", str(root), "--release-manifest", str(release_path), "--staged-base-manifest", str(output / "staged_manifest.json"), "--output-root", str(output / "sweep")],
        ),
        (
            "bind_generic_base_cameras",
            [python, "tools/bind_pybullet_visuals.py", "--root", str(root), "--manifest", str(output / "manifests/generic_source_manifest.json"), "--output-root", str(output / "generic"), "--workers", str(args.workers)],
        ),
        (
            "bind_generic_sweep_visuals",
            [python, "tools/bind_physics_sweep_visuals.py", "--root", str(root), "--sweep-manifest", str(output / "sweep/generic/physics_manifest.json"), "--base-bound-manifest", str(output / "generic/bound_manifest.json"), "--output-root", str(output / "sweep/generic/bound")],
        ),
        (
            "render_asset_bases",
            [python, "tools/render_asset_proxy_manifest.py", "--root", str(root), "--manifest", str(output / "sweep/asset/base_render_input_manifest.json"), "--workers", str(args.workers), "--gpus", args.gpus, "--resume", "--result-manifest", str(output / "sweep/asset/base_render_manifest.json")],
        ),
        (
            "freeze_asset_sweep_cameras",
            [python, "tools/freeze_asset_sweep_cameras.py", "--root", str(root), "--manifest", str(output / "sweep/asset/render_input_manifest.json"), "--base-manifest", str(output / "sweep/asset/base_render_input_manifest.json")],
        ),
        (
            "render_asset_sweeps",
            [python, "tools/render_asset_proxy_manifest.py", "--root", str(root), "--manifest", str(output / "sweep/asset/derived_render_input_manifest.json"), "--workers", str(args.workers), "--gpus", args.gpus, "--resume", "--result-manifest", str(output / "sweep/asset/derived_render_manifest.json")],
        ),
        (
            "render_billiards_sweeps",
            [python, "tools/render_billiards_manifest.py", "--root", str(root), "--manifest", str(output / "sweep/billiards/render_input_manifest.json"), "--workers", str(args.workers), "--gpus", args.gpus, "--resume"],
        ),
        (
            "render_generic_sweeps",
            [python, "tools/render_pybullet_manifest.py", "--root", str(root), "--manifest", str(output / "sweep/generic/bound/bound_manifest.json"), "--workers", str(args.workers), "--gpus", args.gpus, "--resume"],
        ),
    ])
    command = stage_commands[args.stage]
    status: dict[str, Any] = {
        "schema_version": "physweep_release_render_status_v1",
        "release_manifest": str(release_path.relative_to(root)),
        "output_root": str(output.relative_to(root)),
        "gpus": args.gpus,
        "workers": args.workers,
        "stage": args.stage,
        "state": "running",
    }
    started = time.time()
    try:
        write_json(status_path, status)
        subprocess.run(command, cwd=root, check=True)
        status["state"] = "complete"
    except Exception as error:
        status["state"] = "failed"
        status["error"] = str(error)
        raise
    finally:
        status["wall_time_s"] = round(time.time() - started, 3)
        write_json(status_path, status)


if __name__ == "__main__":
    main()
