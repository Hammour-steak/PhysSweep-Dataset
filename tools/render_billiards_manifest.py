#!/usr/bin/env python3
"""Render all billiards records from a scene-family manifest in parallel."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any

from blender_worker_environment import build_egl_device_selector
from render_asset_proxy_manifest import output_path, project_path, sha256, worker


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


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
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    root = args.root.resolve()
    manifest_path = project_path(root, args.manifest)
    manifest_path.relative_to(root)
    manifest = load_json(manifest_path)
    source_records = manifest.get("records")
    if source_records is None:
        source_records = [
            {"metadata_path": str(value)}
            for value in manifest["billiards_metadata_paths"]
        ]
    source_records = [dict(record) for record in source_records]
    for record in source_records:
        if "scene_id" not in record:
            metadata_path = project_path(root, str(record["metadata_path"]))
            metadata_path.relative_to(root)
            metadata = load_json(metadata_path)
            record["scene_id"] = str(metadata["scene_id"])
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one id")
    output = output_path(root, manifest_path.parent)
    script = root / "tools/render_billiards_scene.py"
    blender = project_path(root, args.blender)
    blender.relative_to(root)
    selector = build_egl_device_selector(root)
    selector_path = root / str(selector["binary_path"])
    started = time.perf_counter()
    jobs = [
        (
            root,
            blender,
            script,
            source_record,
            output,
            gpus[index % len(gpus)],
            selector_path,
            args.resume,
        )
        for index, source_record in enumerate(source_records)
    ]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        records = list(executor.map(lambda values: worker(*values), jobs))
    failures = [record for record in records if not record["ok"]]
    summary = {
        "schema_version": "physweep_billiards_render_manifest_v1",
        "source_manifest": str(manifest_path.relative_to(root)),
        "source_manifest_sha256": sha256(manifest_path),
        "sample_count": len(records),
        "success_count": len(records) - len(failures),
        "failure_count": len(failures),
        "reused_count": sum(record.get("reused", False) for record in records),
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
