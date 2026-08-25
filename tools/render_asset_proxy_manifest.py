#!/usr/bin/env python3
"""Render an asset-only proxy manifest with parallel Blender workers."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
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
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def worker(
    root: Path,
    blender: Path,
    script: Path,
    record: dict[str, Any],
    output: Path,
    gpu: int,
    selector_path: Path,
    resume: bool,
) -> dict[str, Any]:
    metadata_path = root / record["metadata_path"]
    metadata = load_json(metadata_path)
    render_output = record.get("render_output", metadata["render"])
    frame_dir = Path(render_output["inspection_frame_dir"])
    if not frame_dir.is_absolute():
        frame_dir = root / frame_dir
    video_path = Path(render_output["video_path"])
    if not video_path.is_absolute():
        video_path = root / video_path
    render_record_path = frame_dir / "render_record.json"
    if resume and render_record_path.is_file() and video_path.is_file():
        render_record = load_json(render_record_path)
        inspection_frames = sorted(frame_dir.glob("frame_*.png"))
        source_metadata = metadata.get("source_metadata")
        expected_metadata_sha256 = sha256(metadata_path)
        if isinstance(source_metadata, dict):
            source_path = root / str(source_metadata["path"])
            expected_metadata_sha256 = sha256(source_path)
            if expected_metadata_sha256 != str(source_metadata["sha256"]):
                raise ValueError(f"source metadata hash mismatch: {source_path}")
        reusable = (
            str(render_record.get("scene_id")) == str(record["scene_id"])
            and str(render_record.get("metadata_sha256"))
            == expected_metadata_sha256
            and str(render_record.get("video_sha256")) == sha256(video_path)
            and video_path.stat().st_size > 0
            and len(inspection_frames) == 3
            and all(frame.stat().st_size > 0 for frame in inspection_frames)
        )
        if reusable:
            return {
                "scene_id": record["scene_id"],
                "ok": True,
                "returncode": 0,
                "gpu": None,
                "egl_device_verified": bool(
                    render_record.get("egl_device_verified", True)
                ),
                "wall_time_s": 0.0,
                "log_path": None,
                "render_record": render_record,
                "reused": True,
            }
    render_record_path.unlink(missing_ok=True)
    video_path.unlink(missing_ok=True)
    command = [
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
    ]
    started = time.perf_counter()
    with isolated_blender_environment(gpu, selector_path) as (
        environment,
        selector_marker,
    ):
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    log_path = output / "logs" / f"{record['scene_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    render_record = (
        load_json(render_record_path)
        if completed.returncode == 0 and render_record_path.is_file()
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
        "scene_id": record["scene_id"],
        "ok": ok,
        "returncode": completed.returncode,
        "gpu": gpu,
        "egl_device_verified": selector_marker in completed.stdout,
        "wall_time_s": round(time.perf_counter() - started, 6),
        "log_path": str(log_path),
        "render_record": render_record,
        "reused": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--blender", type=Path, default=PROJECT_ROOT / "runtime/blender-3.4.0-linux-x64/blender")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a video only when its scene id and metadata/video hashes verify.",
    )
    parser.add_argument(
        "--result-manifest",
        type=Path,
        help="Explicit summary path; defaults to OUTPUT_ROOT/render_manifest.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest = load_json(args.manifest.resolve())
    output = Path(manifest["output_root"])
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one id")
    script = root / "tools/render_asset_proxy_scene.py"
    selector = build_egl_device_selector(root)
    selector_path = root / str(selector["binary_path"])
    started = time.perf_counter()
    jobs = [
        (
            root,
            args.blender.resolve(),
            script,
            record,
            output,
            gpus[index % len(gpus)],
            selector_path,
            args.resume,
        )
        for index, record in enumerate(manifest["records"])
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        records = list(executor.map(lambda values: worker(*values), jobs))
    failures = [record for record in records if not record["ok"]]
    summary = {
        "schema_version": "physweep_asset_proxy_render_manifest_v1",
        "source_manifest": str(args.manifest.resolve().relative_to(root)),
        "source_manifest_sha256": sha256(args.manifest.resolve()),
        "sample_count": len(records),
        "success_count": len(records) - len(failures),
        "failure_count": len(failures),
        "reused_count": sum(record.get("reused", False) for record in records),
        "wall_time_s": round(time.perf_counter() - started, 6),
        "egl_device_selector": selector,
        "records": records,
    }
    result_manifest = (
        args.result_manifest.resolve()
        if args.result_manifest is not None
        else output / "render_manifest.json"
    )
    result_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_json(result_manifest, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    if failures:
        print(json.dumps(failures[:5], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
