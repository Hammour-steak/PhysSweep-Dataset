#!/usr/bin/env python3
"""Run one explicit rendering stage for a published one-object sweep release."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from specialized_backend_registry import specialized_by_pipeline
except ModuleNotFoundError:
    from tools.specialized_backend_registry import specialized_by_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def load_release(root: Path, value: str | Path) -> tuple[Path, dict[str, Any], Path]:
    root = root.resolve()
    release_path = project_path(root, value)
    release_path.relative_to(root / "datasets")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("schema_version") not in {
        "physweep_one_object_sweep_release_v1",
        "physweep_one_object_sweep_release_v2",
        "physweep_one_object_sweep_release_v3",
    }:
        raise ValueError("render runner requires a v1, v2, or v3 one-object sweep release")
    base_path = project_path(root, release["base_manifest"])
    base_path.relative_to(root / "datasets")
    if sha256(base_path) != str(release["base_manifest_sha256"]):
        raise ValueError("release base manifest hash mismatch")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    group_count = int(release["base_group_count"])
    sample_count = int(release["sample_count"])
    if (
        int(base["sample_count"]) != len(base["records"])
        or len(base["records"]) != group_count
        or sample_count != 13 * group_count
    ):
        raise ValueError("release and base manifest counts are inconsistent")
    return release_path, release, base_path


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
    parser.add_argument(
        "--pipeline",
        help="Prepare only this pipeline when staging a versioned release delta.",
    )
    parser.add_argument("--stage", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    root = args.root.resolve()
    release_path, release, base_path = load_release(root, args.release_manifest)
    output = project_path(root, args.output_root)
    if (root / "outputs").resolve() not in output.parents:
        raise ValueError("render output must remain under root/outputs")
    status_path = output.parent / f"{output.name}_{args.stage}_status.json"
    python = sys.executable
    prepare_base_command = [
        python,
        "tools/prepare_formal_render_manifests.py",
        "--root",
        str(root),
        "--manifest",
        str(base_path),
        "--output-root",
        str(output),
        "--selection",
        "all",
    ]
    if args.pipeline is not None:
        prepare_base_command.extend(["--pipeline", args.pipeline])
    stage_commands = {
        "prepare_base_render_plan": prepare_base_command,
        "prepare_sweep_render_plan": [
            python,
            "tools/prepare_sweep_render_manifests.py",
            "--root",
            str(root),
            "--release-manifest",
            str(release_path),
            "--staged-base-manifest",
            str(output / "staged_manifest.json"),
            "--output-root",
            str(output / "sweep"),
        ],
        "bind_generic_base_cameras": [
            python,
            "tools/bind_pybullet_visuals.py",
            "--root",
            str(root),
            "--manifest",
            str(output / "manifests/generic_source_manifest.json"),
            "--output-root",
            str(output / "generic"),
            "--workers",
            str(args.workers),
        ],
        "bind_generic_sweep_visuals": [
            python,
            "tools/bind_physics_sweep_visuals.py",
            "--root",
            str(root),
            "--sweep-manifest",
            str(output / "sweep/generic/physics_manifest.json"),
            "--base-bound-manifest",
            str(output / "generic/bound_manifest.json"),
            "--output-root",
            str(output / "sweep/generic/bound"),
        ],
        "render_asset_bases": [
            python,
            "tools/render_asset_proxy_manifest.py",
            "--root",
            str(root),
            "--manifest",
            str(output / "sweep/asset/base_render_input_manifest.json"),
            "--workers",
            str(args.workers),
            "--gpus",
            args.gpus,
            "--resume",
            "--result-manifest",
            str(output / "sweep/asset/base_render_manifest.json"),
        ],
        "freeze_asset_sweep_cameras": [
            python,
            "tools/freeze_asset_sweep_cameras.py",
            "--root",
            str(root),
            "--manifest",
            str(output / "sweep/asset/render_input_manifest.json"),
            "--base-manifest",
            str(output / "sweep/asset/base_render_input_manifest.json"),
        ],
        "render_asset_sweeps": [
            python,
            "tools/render_asset_proxy_manifest.py",
            "--root",
            str(root),
            "--manifest",
            str(output / "sweep/asset/derived_render_input_manifest.json"),
            "--workers",
            str(args.workers),
            "--gpus",
            args.gpus,
            "--resume",
            "--result-manifest",
            str(output / "sweep/asset/derived_render_manifest.json"),
        ],
        "render_generic_sweeps": [
            python,
            "tools/render_pybullet_manifest.py",
            "--root",
            str(root),
            "--manifest",
            str(output / "sweep/generic/bound/bound_manifest.json"),
            "--workers",
            str(args.workers),
            "--gpus",
            args.gpus,
            "--sweep-kind",
            "sweep",
            "--resume",
            "--result-manifest",
            str(output / "sweep/generic/bound/derived_render_manifest.json"),
        ],
        "render_generic_bases": [
            python,
            "tools/render_pybullet_manifest.py",
            "--root",
            str(root),
            "--manifest",
            str(output / "sweep/generic/bound/bound_manifest.json"),
            "--workers",
            str(args.workers),
            "--gpus",
            args.gpus,
            "--sweep-kind",
            "base",
            "--resume",
            "--result-manifest",
            str(output / "sweep/generic/bound/base_render_manifest.json"),
        ],
    }
    for record in specialized_by_pipeline(root).values():
        branch = str(record["sweep_branch"])
        renderer = str(record["renderer_id"])
        if renderer == "asset":
            continue
        for kind, stage_suffix in (("base", "bases"), ("derived", "sweeps")):
            stage_commands[f"render_{branch}_{stage_suffix}"] = [
                python,
                "tools/render_asset_proxy_manifest.py",
                "--renderer",
                renderer,
                "--root",
                str(root),
                "--manifest",
                str(
                    output
                    / f"sweep/{branch}/{kind}_render_input_manifest.json"
                ),
                "--workers",
                str(args.workers),
                "--gpus",
                args.gpus,
                "--resume",
                "--result-manifest",
                str(output / f"sweep/{branch}/{kind}_render_manifest.json"),
            ]
    if args.stage not in stage_commands:
        raise ValueError(f"unknown render stage: {args.stage}")
    command = stage_commands[args.stage]
    status: dict[str, Any] = {
        "schema_version": "physweep_release_render_status_v1",
        "release_manifest": str(release_path.relative_to(root)),
        "output_root": str(output.relative_to(root)),
        "gpus": args.gpus,
        "workers": args.workers,
        "pipeline_filter": args.pipeline,
        "stage": args.stage,
        "state": "running",
    }
    started = time.time()
    try:
        write_json(status_path, status)
        subprocess.run(command, cwd=root, check=True)
        status["state"] = "complete"
    except KeyboardInterrupt:
        status["state"] = "interrupted"
        status["error"] = "keyboard interrupt"
        raise
    except Exception as error:
        status["state"] = "failed"
        status["error"] = str(error)
        raise
    finally:
        status["wall_time_s"] = round(time.time() - started, 3)
        write_json(status_path, status)


if __name__ == "__main__":
    main()
