#!/usr/bin/env python3
"""Run one explicit rendering stage for a published one-object sweep release."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from tools.physics.specialized_backend_registry import specialized_by_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERIC_SOURCE_SCHEMA = "physweep_pybullet_rigid_metadata_v1"


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
    base_records = base.get("records")
    pipeline_counts = release.get("pipeline_group_counts")
    if not isinstance(base_records, list):
        raise ValueError("release and base manifest structure is inconsistent")
    actual_pipelines = Counter(
        str(record.get("pipeline", "")) for record in base_records
    )
    if pipeline_counts is None:
        if release["schema_version"] != "physweep_one_object_sweep_release_v1":
            raise ValueError("release pipeline distribution is missing")
        expected_pipelines = actual_pipelines
    elif isinstance(pipeline_counts, dict):
        expected_pipelines = Counter(
            {
                str(pipeline): int(count)
                for pipeline, count in pipeline_counts.items()
            }
        )
    else:
        raise ValueError("release pipeline distribution is invalid")
    base_ids = [str(record.get("scene_id", "")) for record in base_records]
    if (
        int(base["sample_count"]) != len(base_records)
        or len(base_records) != group_count
        or sample_count != 13 * group_count
        or sum(expected_pipelines.values()) != group_count
        or "" in expected_pipelines
        or any(count <= 0 for count in expected_pipelines.values())
        or actual_pipelines != expected_pipelines
        or "" in base_ids
        or len(base_ids) != len(set(base_ids))
    ):
        raise ValueError("release and base manifest contract is inconsistent")
    return release_path, release, base_path


def release_metadata_selection(
    root: Path, release: dict[str, Any]
) -> tuple[Path, str]:
    """Verify and return the exact metadata manifest used for render selection."""
    metadata_path = project_path(root, release["metadata_manifest"])
    metadata_path.relative_to(root / "datasets")
    expected_hash = str(release["metadata_manifest_sha256"])
    if not metadata_path.is_file() or sha256(metadata_path) != expected_hash:
        raise ValueError("release metadata manifest hash mismatch")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = metadata.get("records")
    if (
        not isinstance(records, list)
        or int(metadata.get("sample_count", -1)) != len(records)
        or len(records) != int(release["sample_count"])
    ):
        raise ValueError("release metadata manifest count is inconsistent")
    scene_ids = [str(record.get("scene_id", "")) for record in records]
    if "" in scene_ids or len(scene_ids) != len(set(scene_ids)):
        raise ValueError("release metadata manifest scene ids are invalid")
    pipeline_counts = release.get("pipeline_group_counts")
    if (
        pipeline_counts is None
        and release["schema_version"] == "physweep_one_object_sweep_release_v1"
    ):
        base_path = project_path(root, release["base_manifest"])
        base_path.relative_to(root / "datasets")
        if sha256(base_path) != str(release["base_manifest_sha256"]):
            raise ValueError("release base manifest hash mismatch")
        base_records = json.loads(base_path.read_text(encoding="utf-8"))["records"]
        generic_group_count = sum(
            record.get("pipeline") == "generic_pybullet" for record in base_records
        )
    elif isinstance(pipeline_counts, dict):
        generic_group_count = int(pipeline_counts.get("generic_pybullet", -1))
    else:
        generic_group_count = -1
    generic_records = [
        record
        for record in records
        if record.get("source_schema_version") == GENERIC_SOURCE_SCHEMA
    ]
    parent_counts: dict[str, int] = {}
    base_parent_counts: dict[str, int] = {}
    sweep_count = 0
    for record in generic_records:
        parent = str(record.get("parent", ""))
        parent_counts[parent] = parent_counts.get(parent, 0) + 1
        kind = record.get("kind")
        if kind == "base":
            base_parent_counts[parent] = base_parent_counts.get(parent, 0) + 1
        elif kind == "sweep":
            sweep_count += 1
    if (
        generic_group_count < 0
        or len(generic_records) != generic_group_count * 13
        or sweep_count != generic_group_count * 12
        or "" in parent_counts
        or len(parent_counts) != generic_group_count
        or any(count != 13 for count in parent_counts.values())
        or set(base_parent_counts) != set(parent_counts)
        or any(count != 1 for count in base_parent_counts.values())
    ):
        raise ValueError("release generic metadata groups are inconsistent")
    return metadata_path, expected_hash


def generic_render_command(
    python: str,
    root: Path,
    output: Path,
    workers: int,
    gpus: str,
    sweep_kind: str,
    selection_path: Path,
    selection_sha256: str,
) -> list[str]:
    if sweep_kind not in {"base", "sweep"}:
        raise ValueError(f"unsupported generic render kind: {sweep_kind}")
    result_name = (
        "base_render_manifest.json"
        if sweep_kind == "base"
        else "derived_render_manifest.json"
    )
    bound_root = output / "sweep/generic/bound"
    return [
        python,
        "-m",
        "tools.rendering.render_pybullet_manifest",
        "--root",
        str(root),
        "--manifest",
        str(bound_root / "bound_manifest.json"),
        "--workers",
        str(workers),
        "--gpus",
        gpus,
        "--sweep-kind",
        sweep_kind,
        "--selection-manifest",
        str(selection_path),
        "--selection-manifest-sha256",
        selection_sha256,
        "--selection-source-schema",
        GENERIC_SOURCE_SCHEMA,
        "--resume",
        "--result-manifest",
        str(bound_root / result_name),
    ]


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
    selection_path, selection_sha256 = release_metadata_selection(root, release)
    output = project_path(root, args.output_root)
    if (root / "outputs").resolve() not in output.parents:
        raise ValueError("render output must remain under root/outputs")
    status_path = output.parent / f"{output.name}_{args.stage}_status.json"
    python = sys.executable
    prepare_base_command = [
        python,
        "-m",
        "tools.rendering.prepare_formal_render_manifests",
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
            "-m",
            "tools.rendering.prepare_sweep_render_manifests",
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
            "-m",
            "tools.rendering.bind_pybullet_visuals",
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
            "-m",
            "tools.rendering.bind_physics_sweep_visuals",
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
            "-m",
            "tools.rendering.render_asset_proxy_manifest",
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
            "-m",
            "tools.rendering.freeze_asset_sweep_cameras",
            "--root",
            str(root),
            "--manifest",
            str(output / "sweep/asset/render_input_manifest.json"),
            "--base-manifest",
            str(output / "sweep/asset/base_render_input_manifest.json"),
        ],
        "render_asset_sweeps": [
            python,
            "-m",
            "tools.rendering.render_asset_proxy_manifest",
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
        "render_generic_sweeps": generic_render_command(
            python,
            root,
            output,
            args.workers,
            args.gpus,
            "sweep",
            selection_path,
            selection_sha256,
        ),
        "render_generic_bases": generic_render_command(
            python,
            root,
            output,
            args.workers,
            args.gpus,
            "base",
            selection_path,
            selection_sha256,
        ),
    }
    for record in specialized_by_pipeline(root).values():
        branch = str(record["sweep_branch"])
        renderer = str(record["renderer_id"])
        if renderer == "asset":
            continue
        for kind, stage_suffix in (("base", "bases"), ("derived", "sweeps")):
            stage_commands[f"render_{branch}_{stage_suffix}"] = [
                python,
                "-m",
                "tools.rendering.render_asset_proxy_manifest",
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
