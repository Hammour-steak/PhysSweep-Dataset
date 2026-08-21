#!/usr/bin/env python3
"""Simulate and audit every scene in a PhysSweep base or sweep manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from simulate_pybullet_rigid import load_json, run, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def manifest_samples(manifest: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    schema = str(manifest.get("schema_version", ""))
    if schema == "physweep_pybullet_base_manifest_v1":
        samples = list(manifest["samples"])
        dataset_id = str(manifest["dataset_id"])
    elif schema == "physweep_physics_sweep_manifest_v1":
        if int(manifest.get("error_count", -1)) != 0:
            raise ValueError("cannot simulate a sweep manifest with derivation errors")
        samples = [
            {"scene_id": record["scene_id"], "metadata_path": record["path"]}
            for record in manifest["records"]
        ]
        dataset_id = str(manifest["dataset_id"])
    else:
        raise ValueError(f"unsupported PyBullet batch manifest: {schema!r}")
    if int(manifest.get("sample_count", len(samples))) != len(samples):
        raise ValueError("manifest sample count does not match its records")
    if not samples:
        raise ValueError("manifest contains no samples")
    return dataset_id, samples


def worker(root: str, sample: dict[str, Any]) -> dict[str, Any]:
    project_root = Path(root)
    metadata_path = project_root / str(sample["metadata_path"])
    output_dir = metadata_path.parent / "physics"
    try:
        return {"ok": True, **run(metadata_path, output_dir)}
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {
            "ok": False,
            "scene_id": str(sample["scene_id"]),
            "metadata_path": str(metadata_path),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    dataset_id, samples = manifest_samples(manifest)
    started = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        records = list(executor.map(worker, [str(root)] * len(samples), samples))
    records.sort(key=lambda record: str(record["scene_id"]))
    passed = [record for record in records if record.get("ok") and record.get("audit_passed")]
    rejected = [record for record in records if record.get("ok") and not record.get("audit_passed")]
    errors = [record for record in records if not record.get("ok")]
    failed_checks = Counter(
        check
        for record in rejected
        for check in record.get("failed_checks", [])
    )
    summary = {
        "schema_version": "physweep_pybullet_batch_record_v1",
        "dataset_id": dataset_id,
        "source_manifest": str(manifest_path),
        "sample_count": len(samples),
        "passed_count": len(passed),
        "rejected_count": len(rejected),
        "error_count": len(errors),
        "pass_rate": round(len(passed) / max(1, len(samples)), 6),
        "failed_check_counts": dict(failed_checks),
        "wall_time_s": round(time.perf_counter() - started, 6),
        "workers": max(1, args.workers),
        "records": records,
    }
    output_path = manifest_path.parent / "simulation_manifest.json"
    write_json(output_path, summary)
    print(f"simulation manifest: {output_path}")
    print(
        f"passed={len(passed)} rejected={len(rejected)} errors={len(errors)} "
        f"pass_rate={summary['pass_rate']:.3f} wall_time_s={summary['wall_time_s']:.3f}"
    )
    if failed_checks:
        print(json.dumps(dict(failed_checks), indent=2, ensure_ascii=True))
    if errors:
        print(json.dumps(errors[:5], indent=2, ensure_ascii=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
