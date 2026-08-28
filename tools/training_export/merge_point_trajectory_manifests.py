#!/usr/bin/env python3
"""Merge sharded point-trajectory manifests without changing trajectory files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.core.hashing import sha256_file as sha256


SCHEMA = "physweep.point_trajectory_manifest.v1"
POINT_COUNT = 2048


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--shard-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def resolve(root: Path, value: Path) -> Path:
    path = value if value.is_absolute() else root / value
    return path.resolve()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    source_path = resolve(root, args.source_manifest)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    source_records = list(iter_jsonl(source_path))
    source_order = {str(record["sample_id"]): index for index, record in enumerate(source_records)}
    if len(source_order) != len(source_records):
        raise ValueError("source manifest contains duplicate sample_id values")

    merged: dict[str, dict] = {}
    shard_paths: list[str] = []
    for value in args.shard_manifest:
        shard_path = resolve(root, value)
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        shard = load_json(shard_path)
        if shard.get("schema") != SCHEMA:
            raise ValueError(f"unexpected shard schema in {shard_path}: {shard.get('schema')}")
        shard_paths.append(shard_path.relative_to(root).as_posix())
        for record in shard.get("records", []):
            sample_id = str(record["sample_id"])
            if sample_id not in source_order:
                raise ValueError(f"shard record is absent from source manifest: {sample_id}")
            if sample_id in merged:
                raise ValueError(f"duplicate shard record: {sample_id}")
            point_path = resolve(root, Path(str(record["path"])))
            if not point_path.is_file():
                raise FileNotFoundError(point_path)
            if int(record.get("point_count", -1)) != POINT_COUNT:
                raise ValueError(f"unexpected point count for {sample_id}")
            merged[sample_id] = record

    missing = [sample_id for sample_id in source_order if sample_id not in merged]
    if missing:
        raise ValueError(f"missing {len(missing)} shard records; first={missing[:3]}")

    records = [merged[str(record["sample_id"])] for record in source_records]
    output_root = resolve(root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "manifest.json"
    payload = {
        "schema": SCHEMA,
        "source_manifest": source_path.relative_to(root).as_posix(),
        "source_manifest_sha256": sha256(source_path),
        "point_count": POINT_COUNT,
        "object_axis": "[T, O, 2048, ...]",
        "records": records,
        "shards": shard_paths,
    }
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output_path)
    print(json.dumps({"manifest": str(output_path), "count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
