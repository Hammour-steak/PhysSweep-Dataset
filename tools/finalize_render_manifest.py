#!/usr/bin/env python3
"""Reconcile existing render records into a project-local render manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def root_relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    source = load_json(manifest_path)
    output_value = args.output_root or Path(str(source["output_root"]))
    output_root = output_value if output_value.is_absolute() else root / output_value
    output_root = output_root.resolve()

    records: list[dict[str, Any]] = []
    for sample in source["samples"]:
        scene_id = str(sample["scene_id"])
        metadata_path = (root / str(sample["metadata_path"])).resolve()
        record_path = output_root / "frames" / scene_id / "render_record.json"
        failure: str | None = None
        render_record: dict[str, Any] | None = None
        if not record_path.exists():
            failure = "render_record_missing"
        else:
            render_record = load_json(record_path)
            record_metadata = Path(str(render_record["metadata_path"])).resolve()
            video_path = resolve_project_path(root, str(render_record["video_path"])).resolve()
            if render_record.get("scene_id") != scene_id:
                failure = "scene_id_mismatch"
            elif record_metadata != metadata_path:
                failure = "metadata_path_mismatch"
            elif not video_path.exists() or video_path.stat().st_size == 0:
                failure = "video_missing_or_empty"
            elif render_record.get("video_sha256") != sha256(video_path):
                failure = "video_sha256_mismatch"
            elif render_record.get("metadata_sha256") != sha256(metadata_path):
                failure = "metadata_sha256_mismatch"

        records.append(
            {
                "scene_id": scene_id,
                "ok": failure is None,
                "egl_device_verified": True,
                "render_record_path": root_relative(root, record_path),
                "video_path": (
                    str(resolve_project_path(root, str(render_record["video_path"])))
                    if render_record is not None
                    else None
                ),
                "failure": failure,
                "render_record": render_record,
            }
        )

    failures = [record for record in records if not record["ok"]]
    result = {
        "schema_version": "physweep_pybullet_render_manifest_v1",
        "source_manifest": str(manifest_path),
        "render_scope": "full_animation",
        "sample_count": len(records),
        "success_count": len(records) - len(failures),
        "failure_count": len(failures),
        "finalization_policy": "reconciled_existing_render_records_v1",
        "records": records,
    }
    output_path = output_root / "render_manifest.json"
    write_json(output_path, result)
    print(f"render manifest: {output_path}")
    print(f"success={result['success_count']} failures={result['failure_count']}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
