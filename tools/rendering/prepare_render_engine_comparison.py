#!/usr/bin/env python3
"""Clone one immutable PhysSweep batch for a render-engine comparison."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def clone_metadata(
    source_path: Path,
    destination_path: Path,
    output_root: Path,
    engine: str,
    samples: int | None,
    generic: bool,
) -> dict[str, Any]:
    metadata = copy.deepcopy(load_json(source_path))
    render = metadata["visualization"]["render"] if generic else metadata["render"]
    render["engine"] = engine
    if samples is not None:
        render["samples"] = samples
    scene_id = str(metadata["scene_id"])
    render["video_path"] = relative(output_root / "videos" / f"{scene_id}.mp4")
    render["inspection_frame_dir"] = relative(output_root / "frames" / scene_id)
    write_json(destination_path, metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--generic-bound-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--engine", default="CYCLES")
    parser.add_argument("--samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    generic_source = load_json(args.generic_bound_manifest.resolve())
    generic_root = output_root / "generic"
    generic_records = []
    for record in generic_source["samples"]:
        source_path = PROJECT_ROOT / record["metadata_path"]
        destination = generic_root / "metadata" / f"{record['scene_id']}.json"
        clone_metadata(
            source_path,
            destination,
            generic_root,
            args.engine,
            args.samples,
            generic=True,
        )
        cloned_record = copy.deepcopy(record)
        cloned_record["metadata_path"] = relative(destination)
        cloned_record["metadata_sha256"] = sha256(destination)
        generic_records.append(cloned_record)
    generic_manifest = copy.deepcopy(generic_source)
    generic_manifest["output_root"] = str(generic_root)
    generic_manifest["samples"] = generic_records
    write_json(generic_root / "bound_manifest.json", generic_manifest)

    asset_source = load_json(dataset_root / "asset_proxy_manifest.json")
    asset_root = output_root / "assets"
    asset_records = []
    for record in asset_source["records"]:
        source_path = PROJECT_ROOT / record["metadata_path"]
        destination = asset_root / "metadata" / f"{record['scene_id']}.json"
        clone_metadata(
            source_path,
            destination,
            asset_root,
            args.engine,
            args.samples,
            generic=False,
        )
        cloned_record = copy.deepcopy(record)
        cloned_record["metadata_path"] = relative(destination)
        asset_records.append(cloned_record)
    asset_manifest = copy.deepcopy(asset_source)
    asset_manifest["output_root"] = str(asset_root)
    asset_manifest["records"] = asset_records
    write_json(asset_root / "manifest.json", asset_manifest)

    dataset_manifest = load_json(dataset_root / "manifest.json")
    billiards_root = output_root / "billiards"
    billiards_records = []
    for value in dataset_manifest["billiards_metadata_paths"]:
        source_path = PROJECT_ROOT / value
        source_metadata = load_json(source_path)
        scene_id = str(source_metadata["scene_id"])
        destination = billiards_root / "metadata" / f"{scene_id}.json"
        clone_metadata(
            source_path,
            destination,
            billiards_root,
            args.engine,
            args.samples,
            generic=False,
        )
        billiards_records.append(
            {"scene_id": scene_id, "metadata_path": relative(destination)}
        )
    billiards_manifest = {
        "schema_version": "physweep_billiards_render_manifest_v1",
        "output_root": str(billiards_root),
        "records": billiards_records,
    }
    write_json(billiards_root / "manifest.json", billiards_manifest)

    first_generic_metadata = load_json(
        PROJECT_ROOT / generic_records[0]["metadata_path"]
    )
    effective_samples = int(
        first_generic_metadata["visualization"]["render"]["samples"]
    )
    comparison = {
        "schema_version": "physweep_render_engine_comparison_v1",
        "source_dataset": str(dataset_root),
        "engine": args.engine,
        "samples": effective_samples,
        "output_root": str(output_root),
        "counts": {
            "generic": len(generic_records),
            "assets": len(asset_records),
            "billiards": len(billiards_records),
            "total": len(generic_records) + len(asset_records) + len(billiards_records),
        },
        "manifests": {
            "generic": str(generic_root / "bound_manifest.json"),
            "assets": str(asset_root / "manifest.json"),
            "billiards": str(billiards_root / "manifest.json"),
        },
    }
    write_json(output_root / "comparison_manifest.json", comparison)
    print(json.dumps(comparison, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
