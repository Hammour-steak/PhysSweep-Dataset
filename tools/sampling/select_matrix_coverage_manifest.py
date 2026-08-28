#!/usr/bin/env python3
"""Select one accepted matrix record for every requested object and support."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generic_dynamic_id(root: Path, record: dict[str, Any]) -> str | None:
    if str(record["environment_id"]) != "generic_matrix":
        value = record.get("dynamic_asset_id")
        return None if value is None else str(value)
    metadata = load_json(root / str(record["metadata_path"]))
    return str(
        metadata["semantic_sampling"]["five_dimensions"]["foreground_object"][
            "visual_asset_id"
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dynamic-asset-ids", nargs="+", required=True)
    parser.add_argument("--support-asset-ids", nargs="+", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    output_path = args.output if args.output.is_absolute() else root / args.output
    source = load_json(manifest_path)
    records = source["records"]
    if any(str(record.get("status")) != "simulated_accepted" for record in records):
        raise ValueError("coverage selection requires accepted source records")
    dynamic_by_id: dict[str, dict[str, Any]] = {}
    support_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        dynamic_id = generic_dynamic_id(root, record)
        if dynamic_id is not None:
            dynamic_by_id.setdefault(dynamic_id, record)
        support_id = record.get("support_asset_id")
        if support_id is not None:
            support_by_id.setdefault(str(support_id), record)
    missing_dynamic = sorted(set(args.dynamic_asset_ids) - set(dynamic_by_id))
    missing_support = sorted(set(args.support_asset_ids) - set(support_by_id))
    if missing_dynamic or missing_support:
        raise ValueError(
            f"coverage records are missing: dynamic={missing_dynamic}, support={missing_support}"
        )
    selected = []
    selected_scene_ids: set[str] = set()
    for asset_id in args.dynamic_asset_ids:
        record = dynamic_by_id[str(asset_id)]
        if str(record["scene_id"]) not in selected_scene_ids:
            selected.append(copy.deepcopy(record))
            selected_scene_ids.add(str(record["scene_id"]))
    for asset_id in args.support_asset_ids:
        record = support_by_id[str(asset_id)]
        if str(record["scene_id"]) not in selected_scene_ids:
            selected.append(copy.deepcopy(record))
            selected_scene_ids.add(str(record["scene_id"]))
    result = copy.deepcopy(source)
    result["dataset_id"] = f"{source['dataset_id']}_coverage_review"
    result["sample_count"] = len(selected)
    result["motion_counts"] = dict(
        Counter(str(record["motion_intent"]) for record in selected)
    )
    result["environment_counts"] = dict(
        Counter(str(record["environment_id"]) for record in selected)
    )
    result["profile_counts"] = dict(
        Counter(str(record["profile"]) for record in selected)
    )
    result["records"] = selected
    result["coverage_selection"] = {
        "source_manifest": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": sha256(manifest_path),
        },
        "dynamic_asset_ids": [str(value) for value in args.dynamic_asset_ids],
        "support_asset_ids": [str(value) for value in args.support_asset_ids],
        "selected_records": len(selected),
    }
    write_json(output_path, result)
    print(
        json.dumps(
            {
                "output": output_path.relative_to(root).as_posix(),
                "selected": len(selected),
                "motions": result["motion_counts"],
                "environments": result["environment_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
