#!/usr/bin/env python3
"""Export dense identity-preserving point trajectories from a PhysSweep manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.core.hashing import sha256_file as sha256

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from tools.dataset_contract.point_trajectory import (
    POINT_COUNT,
    POINT_TRAJECTORY_SCHEMA,
    build_point_trajectory,
    save_point_trajectory,
)


DEFAULT_MANIFEST = Path("datasets/physweep_training/manifest.jsonl")
DEFAULT_OUTPUT = Path("datasets/physweep_training/point_trajectories")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def project_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"dataset path must be project-relative: {value}")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"dataset path escapes the project root: {value}")
    if not resolved.is_file():
        raise FileNotFoundError(f"cannot resolve project file: {value}")
    return resolved


def select_records(records: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.sample_id:
        requested = set(args.sample_id)
        selected = [record for record in records if record["sample_id"] in requested]
        missing = sorted(requested - {record["sample_id"] for record in selected})
        if missing:
            raise ValueError(f"unknown sample ids: {missing}")
    else:
        selected = records
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("no records selected")
    return selected


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    manifest_path = (root / args.manifest).resolve()
    output_root = (root / args.output_root).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest does not exist: {manifest_path}")
    records = select_records(list(iter_jsonl(manifest_path)), args)
    output_records = []
    for index, record in enumerate(records, 1):
        sample_id = str(record["sample_id"])
        scene_path = project_path(root, record["conditioning"]["scene"])
        trajectory_value = record.get("provenance", {}).get("trajectory")
        if not trajectory_value:
            raise ValueError(f"record has no simulation trajectory: {sample_id}")
        trajectory_path = project_path(root, trajectory_value)
        output_path = output_root / sample_id / "point_trajectory.npz"
        if output_path.is_file() and not args.overwrite:
            print(f"skip {index}/{len(records)} {sample_id}", flush=True)
        else:
            payload = build_point_trajectory(scene_path, trajectory_path)
            save_point_trajectory(output_path, payload)
            print(f"export {index}/{len(records)} {sample_id}", flush=True)
        with np.load(output_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            shape = list(archive["points_world_m"].shape)
            object_ids = [str(value) for value in archive["object_ids"]]
        expected_object_id = str(
            record["conditioning"]["physics"]["object"]["object_id"]
        )
        if object_ids != [expected_object_id]:
            raise ValueError(
                f"point trajectory object binding mismatch for {sample_id}: "
                f"{object_ids} != {[expected_object_id]}"
            )
        output_records.append(
            {
                "sample_id": sample_id,
                "path": output_path.relative_to(root).as_posix(),
                "sha256": sha256(output_path),
                "schema": POINT_TRAJECTORY_SCHEMA,
                "point_count": POINT_COUNT,
                "object_count": len(object_ids),
                "object_ids": object_ids,
                "shape": shape,
                "source_scene": scene_path.relative_to(root).as_posix(),
                "source_trajectory": trajectory_path.relative_to(root).as_posix(),
                "initial_alignment_error_m": metadata["initial_alignment_error_m"],
            }
        )
    manifest = {
        "schema": "physweep.point_trajectory_manifest.v1",
        "source_manifest": manifest_path.relative_to(root).as_posix(),
        "source_manifest_sha256": sha256(manifest_path),
        "point_count": POINT_COUNT,
        "object_axis": "[T, O, 2048, ...]",
        "records": output_records,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path_out = output_root / "manifest.json"
    temporary = manifest_path_out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(manifest_path_out)
    print(json.dumps({"manifest": str(manifest_path_out), "count": len(output_records)}, indent=2))


if __name__ == "__main__":
    main()
