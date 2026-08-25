#!/usr/bin/env python3
"""Render a published one-object sweep release through all visual backends."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    release_path = (root / args.release_manifest).resolve()
    release = json.loads(release_path.read_text(encoding="utf-8"))
    output = (root / args.output_root).resolve()
    status_path = output.parent / f"{output.name}_status.json"
    python = sys.executable
    stages = [
        (
            "prepare_base_render_plan",
            [python, "tools/prepare_formal_render_manifests.py", "--root", str(root), "--manifest", str(root / release["base_manifest"]), "--output-root", str(output), "--selection", "all", "--overwrite"],
        ),
        (
            "prepare_sweep_render_plan",
            [python, "tools/prepare_sweep_render_manifests.py", "--root", str(root), "--release-manifest", str(release_path), "--staged-base-manifest", str(output / "staged_manifest.json"), "--output-root", str(output / "sweep"), "--overwrite"],
        ),
        (
            "bind_generic_base_cameras",
            [python, "tools/bind_pybullet_visuals.py", "--root", str(root), "--manifest", str(output / "manifests/generic_source_manifest.json"), "--output-root", str(output / "generic"), "--workers", str(args.workers)],
        ),
        (
            "bind_generic_sweep_visuals",
            [python, "tools/bind_physics_sweep_visuals.py", "--root", str(root), "--sweep-manifest", str(output / "sweep/generic/physics_manifest.json"), "--base-bound-manifest", str(output / "generic/bound_manifest.json"), "--output-root", str(output / "sweep/generic/bound"), "--overwrite"],
        ),
        (
            "render_asset_bases",
            [python, "tools/render_asset_proxy_manifest.py", "--root", str(root), "--manifest", str(output / "sweep/asset/base_render_input_manifest.json"), "--workers", str(args.workers), "--gpus", args.gpus],
        ),
        (
            "freeze_asset_sweep_cameras",
            [python, "tools/freeze_asset_sweep_cameras.py", "--root", str(root), "--manifest", str(output / "sweep/asset/render_input_manifest.json"), "--base-manifest", str(output / "sweep/asset/base_render_input_manifest.json")],
        ),
        (
            "render_asset_sweeps",
            [python, "tools/render_asset_proxy_manifest.py", "--root", str(root), "--manifest", str(output / "sweep/asset/render_input_manifest.json"), "--workers", str(args.workers), "--gpus", args.gpus],
        ),
        (
            "render_billiards_sweeps",
            [python, "tools/render_billiards_manifest.py", "--root", str(root), "--manifest", str(output / "sweep/billiards/render_input_manifest.json"), "--workers", str(args.workers), "--gpus", args.gpus],
        ),
        (
            "render_generic_sweeps",
            [python, "tools/render_pybullet_manifest.py", "--root", str(root), "--manifest", str(output / "sweep/generic/bound/bound_manifest.json"), "--workers", str(args.workers), "--gpus", args.gpus],
        ),
    ]
    status: dict[str, Any] = {
        "schema_version": "physweep_release_render_status_v1",
        "release_manifest": str(release_path.relative_to(root)),
        "output_root": str(output.relative_to(root)),
        "gpus": args.gpus,
        "workers": args.workers,
        "state": "running",
        "completed_stages": [],
        "current_stage": None,
    }
    started = time.time()
    try:
        for name, command in stages:
            status["current_stage"] = name
            write_json(status_path, status)
            subprocess.run(command, cwd=root, check=True)
            status["completed_stages"].append(name)
        status["state"] = "complete"
        status["current_stage"] = None
    except Exception as error:
        status["state"] = "failed"
        status["error"] = str(error)
        raise
    finally:
        status["wall_time_s"] = round(time.time() - started, 3)
        write_json(status_path, status)


if __name__ == "__main__":
    main()
