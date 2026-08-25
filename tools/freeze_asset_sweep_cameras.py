#!/usr/bin/env python3
"""Freeze each asset sweep group to the camera solved from its base sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest = load_json(args.manifest.resolve())
    base_manifest = load_json(args.base_manifest.resolve())
    cameras: dict[str, dict[str, Any]] = {}
    for record in base_manifest["records"]:
        metadata_path = root / str(record["metadata_path"])
        metadata = load_json(metadata_path)
        parent = str(metadata["sweep"]["parent_scene_id"])
        frame_dir = root / str(record["render_output"]["inspection_frame_dir"])
        render_record = load_json(frame_dir / "render_record.json")
        camera = render_record["camera"]
        cameras[parent] = {
            "solver_version": "frozen_from_one_factor_base_v1",
            "source_base_scene_id": str(metadata["scene_id"]),
            "position_m": camera["position_m"],
            "target_m": camera["target_m"],
            "focal_length_mm": camera["focal_length_mm"],
            "focus_span_m": camera.get("focus_span_m"),
        }
    if len(cameras) != len(base_manifest["records"]):
        raise ValueError("base manifest contains duplicate sweep parents")

    counts: dict[str, int] = {parent: 0 for parent in cameras}
    for record in manifest["records"]:
        metadata_path = root / str(record["metadata_path"])
        metadata = load_json(metadata_path)
        parent = str(metadata["sweep"]["parent_scene_id"])
        if parent not in cameras:
            raise ValueError(f"missing rendered base camera for {parent}")
        metadata["camera_binding"] = cameras[parent]
        write_json(metadata_path, metadata)
        counts[parent] += 1
    if any(count != 13 for count in counts.values()):
        raise ValueError("camera binding did not cover complete 13-sample groups")
    print(json.dumps({"group_count": len(cameras), "sample_count": sum(counts.values())}, indent=2))


if __name__ == "__main__":
    main()
