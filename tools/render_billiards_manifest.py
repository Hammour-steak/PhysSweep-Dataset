#!/usr/bin/env python3
"""Render all billiards records from a scene-family manifest in parallel."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from blender_worker_environment import (
    build_egl_device_selector,
    isolated_blender_environment,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def worker(
    root: Path,
    blender: Path,
    script: Path,
    source_record: dict[str, Any],
    output: Path,
    gpu: int,
    selector_path: Path,
) -> dict[str, Any]:
    metadata_path = root / str(source_record["metadata_path"])
    metadata = load_json(metadata_path)
    render_output = source_record.get("render_output", metadata["render"])
    frame_dir = Path(render_output["inspection_frame_dir"])
    if not frame_dir.is_absolute():
        frame_dir = root / frame_dir
    video_path = Path(render_output["video_path"])
    if not video_path.is_absolute():
        video_path = root / video_path
    record_path = frame_dir / "render_record.json"
    record_path.unlink(missing_ok=True)
    video_path.unlink(missing_ok=True)
    started = time.perf_counter()
    with isolated_blender_environment(gpu, selector_path) as (
        environment,
        selector_marker,
    ):
        completed = subprocess.run(
            [
                str(blender),
                "-b",
                "--python",
                str(script),
                "--",
                "--metadata",
                str(metadata_path),
                "--video-path",
                str(video_path),
                "--inspection-frame-dir",
                str(frame_dir),
            ],
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    log_path = output / "logs" / f"{metadata['scene_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    render_record = (
        load_json(record_path)
        if completed.returncode == 0 and record_path.is_file()
        else None
    )
    ok = (
        completed.returncode == 0
        and render_record is not None
        and video_path.is_file()
        and video_path.stat().st_size > 0
        and selector_marker in completed.stdout
    )
    return {
        "scene_id": metadata["scene_id"],
        "ok": ok,
        "returncode": completed.returncode,
        "gpu": gpu,
        "egl_device_verified": selector_marker in completed.stdout,
        "wall_time_s": round(time.perf_counter() - started, 6),
        "log_path": str(log_path),
        "render_record": render_record,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--blender",
        type=Path,
        default=PROJECT_ROOT / "runtime/blender-3.4.0-linux-x64/blender",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    source_records = manifest.get("records")
    if source_records is None:
        source_records = [
            {"metadata_path": str(value)}
            for value in manifest["billiards_metadata_paths"]
        ]
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one id")
    output = manifest_path.parent
    script = root / "tools/render_billiards_scene.py"
    selector = build_egl_device_selector(root)
    selector_path = root / str(selector["binary_path"])
    started = time.perf_counter()
    jobs = [
        (
            root,
            args.blender.resolve(),
            script,
            source_record,
            output,
            gpus[index % len(gpus)],
            selector_path,
        )
        for index, source_record in enumerate(source_records)
    ]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, args.workers)
    ) as executor:
        records = list(executor.map(lambda values: worker(*values), jobs))
    failures = [record for record in records if not record["ok"]]
    summary = {
        "schema_version": "physweep_billiards_render_manifest_v1",
        "source_manifest": str(manifest_path),
        "sample_count": len(records),
        "success_count": len(records) - len(failures),
        "failure_count": len(failures),
        "wall_time_s": round(time.perf_counter() - started, 6),
        "egl_device_selector": selector,
        "records": records,
    }
    write_json(output / "billiards_render_manifest.json", summary)
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "records"},
            indent=2,
        )
    )
    if failures:
        print(json.dumps(failures[:5], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
