#!/usr/bin/env python3
"""Validate and index matching Eevee and Cycles PhysSweep videos."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": str(stream["r_frame_rate"]),
        "frame_count": int(stream["nb_frames"]),
        "duration_s": round(float(stream["duration"]), 6),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eevee-videos", type=Path, required=True)
    parser.add_argument("--cycles-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    eevee_root = args.eevee_videos.resolve()
    cycles_root = args.cycles_root.resolve()
    merged_root = cycles_root / "videos"
    reference_root = cycles_root / "eevee_reference" / "videos"
    merged_root.mkdir(parents=True, exist_ok=True)
    reference_root.mkdir(parents=True, exist_ok=True)

    source_roots = [
        cycles_root / "generic" / "videos",
        cycles_root / "assets" / "videos",
        cycles_root / "billiards" / "videos",
    ]
    source_paths = sorted(
        path for root in source_roots for path in root.glob("*.mp4")
    )
    if len(source_paths) != 40:
        raise ValueError(f"expected 40 Cycles videos, found {len(source_paths)}")
    names = [path.name for path in source_paths]
    if len(set(names)) != len(names):
        raise ValueError("Cycles category outputs contain duplicate filenames")

    for source in source_paths:
        shutil.copy2(source, merged_root / source.name)

    source_eevee_paths = {path.name: path for path in eevee_root.glob("*.mp4")}
    cycles_paths = {path.name: path for path in merged_root.glob("*.mp4")}
    if set(source_eevee_paths) != set(cycles_paths):
        missing_cycles = sorted(set(source_eevee_paths) - set(cycles_paths))
        missing_eevee = sorted(set(cycles_paths) - set(source_eevee_paths))
        raise ValueError(
            f"filename mismatch: missing_cycles={missing_cycles}, "
            f"missing_eevee={missing_eevee}"
        )
    for source in source_eevee_paths.values():
        shutil.copy2(source, reference_root / source.name)
    eevee_paths = {path.name: path for path in reference_root.glob("*.mp4")}

    records = []
    for name in sorted(cycles_paths):
        eevee_path = eevee_paths[name]
        cycles_path = cycles_paths[name]
        eevee_probe = probe(eevee_path)
        cycles_probe = probe(cycles_path)
        if eevee_probe != cycles_probe:
            raise ValueError(
                f"video specification mismatch for {name}: "
                f"eevee={eevee_probe}, cycles={cycles_probe}"
            )
        records.append(
            {
                "filename": name,
                "eevee": {
                    "path": str(eevee_path),
                    "sha256": sha256(eevee_path),
                },
                "cycles": {
                    "path": str(cycles_path),
                    "sha256": sha256(cycles_path),
                },
                "video": cycles_probe,
            }
        )

    index = {
        "schema_version": "physweep_render_engine_comparison_index_v1",
        "video_count": len(records),
        "engines": {
            "reference": "BLENDER_EEVEE",
            "candidate": "CYCLES",
        },
        "matched_filenames": True,
        "matched_video_specifications": True,
        "records": records,
    }
    write_json(cycles_root / "comparison_index.json", index)
    print(
        json.dumps(
            {
                "video_count": index["video_count"],
                "matched_filenames": index["matched_filenames"],
                "matched_video_specifications": index[
                    "matched_video_specifications"
                ],
                "cycles_videos": str(merged_root),
                "eevee_videos": str(reference_root),
                "index": str(cycles_root / "comparison_index.json"),
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
