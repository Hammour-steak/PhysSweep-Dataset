#!/usr/bin/env python3
"""Render a bound PhysSweep PyBullet manifest with parallel Blender workers."""

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
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def worker(
    root: str,
    blender: str,
    script: str,
    output_root: str,
    sample: dict[str, Any],
    gpu: int,
    first_frame_only: bool,
    selector_path: str,
) -> dict[str, Any]:
    project_root = Path(root)
    metadata_path = project_root / str(sample["metadata_path"])
    record_path = Path(output_root) / "frames" / str(sample["scene_id"]) / "render_record.json"
    record_path.unlink(missing_ok=True)
    command = [blender, "-b", "--python", script, "--", "--metadata", str(metadata_path)]
    if first_frame_only:
        command.append("--first-frame-only")
    started = time.perf_counter()
    with isolated_blender_environment(gpu, Path(selector_path)) as (
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
    log_path = Path(output_root) / "logs" / f"{sample['scene_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    record = load_json(record_path) if completed.returncode == 0 and record_path.exists() else None
    selector_verified = selector_marker in completed.stdout
    return {
        "scene_id": str(sample["scene_id"]),
        "ok": completed.returncode == 0 and record is not None and selector_verified,
        "returncode": completed.returncode,
        "gpu": gpu,
        "egl_device_verified": selector_verified,
        "wall_time_s": round(time.perf_counter() - started, 6),
        "log_path": str(log_path),
        "render_record": record,
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
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument("--first-frame-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--profiles",
        nargs="+",
        help="Render only samples bound to these scene-visual profile ids.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    samples = list(manifest["samples"])
    if args.profiles:
        requested_profiles = {str(value) for value in args.profiles}
        selected_samples = []
        selected_profiles = set()
        for sample in samples:
            metadata_path = root / str(sample["metadata_path"])
            metadata = load_json(metadata_path)
            profile_id = str(metadata["visualization"]["environment"]["profile_id"])
            if profile_id in requested_profiles:
                selected_samples.append(sample)
                selected_profiles.add(profile_id)
        missing_profiles = requested_profiles - selected_profiles
        if missing_profiles:
            raise ValueError(
                f"bound manifest lacks requested profiles: {sorted(missing_profiles)}"
            )
        samples = selected_samples
    if args.limit is not None:
        samples = samples[: args.limit]
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one id")
    output_root_value = Path(str(manifest["output_root"]))
    output_root = (
        output_root_value
        if output_root_value.is_absolute()
        else root / output_root_value
    ).resolve()
    selector = build_egl_device_selector(root)
    selector_path = root / str(selector["binary_path"])
    script = root / "tools/render_pybullet_rigid.py"
    started = time.perf_counter()
    jobs = [
        (
            str(root),
            str(args.blender.resolve()),
            str(script),
            str(output_root),
            sample,
            gpus[index % len(gpus)],
            bool(args.first_frame_only),
            str(selector_path),
        )
        for index, sample in enumerate(samples)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        records = list(executor.map(lambda values: worker(*values), jobs))
    records.sort(key=lambda record: record["scene_id"])
    failures = [record for record in records if not record["ok"]]
    summary = {
        "schema_version": "physweep_pybullet_render_manifest_v1",
        "source_manifest": str(manifest_path),
        "render_scope": "first_frame_only" if args.first_frame_only else "full_animation",
        "sample_count": len(records),
        "success_count": len(records) - len(failures),
        "failure_count": len(failures),
        "wall_time_s": round(time.perf_counter() - started, 6),
        "egl_device_selector": selector,
        "records": records,
    }
    output_path = output_root / "render_manifest.json"
    write_json(output_path, summary)
    print(f"render manifest: {output_path}")
    print(
        f"success={summary['success_count']} failures={summary['failure_count']} "
        f"wall_time_s={summary['wall_time_s']:.3f}"
    )
    if failures:
        print(json.dumps(failures[:5], indent=2, ensure_ascii=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
