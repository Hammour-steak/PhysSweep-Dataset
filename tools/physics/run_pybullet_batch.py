#!/usr/bin/env python3
"""Simulate and audit every scene in a PhysSweep base or sweep manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.physics.pybullet_backend_dispatcher import dispatch_simulation, load_json, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAFE_SCENE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def worker_context() -> multiprocessing.context.BaseContext:
    """Start every PyBullet worker with a fresh native-library state."""
    return multiprocessing.get_context("spawn")


def group_samples_by_schema(
    root: Path, samples: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Keep distinct PyBullet adapters out of the same long-lived worker."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        metadata_path = (root / str(sample["metadata_path"])).resolve()
        try:
            metadata_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"metadata path escapes project root: {metadata_path}") from exc
        metadata = load_json(metadata_path)
        schema = str(metadata.get("schema_version", ""))
        if not schema:
            raise ValueError(f"metadata schema is missing for {sample['scene_id']}")
        declared_schema = sample.get("source_schema_version")
        if declared_schema is not None and str(declared_schema) != schema:
            raise ValueError(f"manifest schema differs from metadata: {sample['scene_id']}")
        if str(metadata.get("scene_id")) != str(sample["scene_id"]):
            raise ValueError(f"manifest scene id differs from metadata: {sample['scene_id']}")
        groups[str(schema)].append(sample)
    return dict(groups)


def manifest_samples(manifest: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    schema = str(manifest.get("schema_version", ""))
    if schema == "physweep_pybullet_base_manifest_v1":
        samples = list(manifest["samples"])
        dataset_id = str(manifest["dataset_id"])
    elif schema in {
        "physweep_physics_sweep_manifest_v1",
        "physweep_physics_sweep_manifest_v2",
    }:
        if int(manifest.get("error_count", -1)) != 0:
            raise ValueError("cannot simulate a sweep manifest with derivation errors")
        samples = [
            {
                "scene_id": record["scene_id"],
                "metadata_path": record["path"],
                "source_schema_version": record.get("source_schema_version"),
            }
            for record in manifest["records"]
        ]
        dataset_id = str(manifest["dataset_id"])
    else:
        raise ValueError(f"unsupported PyBullet batch manifest: {schema!r}")
    if int(manifest.get("sample_count", len(samples))) != len(samples):
        raise ValueError("manifest sample count does not match its records")
    if not samples:
        raise ValueError("manifest contains no samples")
    scene_ids = [str(sample["scene_id"]) for sample in samples]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("manifest contains duplicate scene ids")
    invalid_ids = [
        scene_id for scene_id in scene_ids if not SAFE_SCENE_ID.fullmatch(scene_id)
    ]
    if invalid_ids:
        raise ValueError(f"manifest contains unsafe scene ids: {invalid_ids[:3]}")
    return dataset_id, samples


def prepare_output_root(output_root: Path, allow_existing: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()) and not allow_existing:
        raise ValueError(
            f"output root is not empty: {output_root}; choose a new path or pass "
            "--allow-existing-output"
        )
    output_root.mkdir(parents=True, exist_ok=True)


def batch_failed(
    *, rejected_count: int, error_count: int, allow_audit_rejections: bool
) -> bool:
    """Return whether the batch result must fail the command."""
    return error_count > 0 or (rejected_count > 0 and not allow_audit_rejections)


def worker(
    root: str, output_root: str, sample: dict[str, Any]
) -> dict[str, Any]:
    project_root = Path(root)
    metadata_path = project_root / str(sample["metadata_path"])
    output_dir = Path(output_root) / str(sample["scene_id"])
    try:
        return {
            "ok": True,
            **dispatch_simulation(metadata_path, output_dir, project_root),
        }
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
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Result directory; defaults to <dataset>/physics.",
    )
    parser.add_argument("--workers", type=int, default=min(24, os.cpu_count() or 1))
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Explicitly permit replacing per-scene files in a non-empty output tree.",
    )
    parser.add_argument(
        "--allow-audit-rejections",
        action="store_true",
        help="Return success for audited rejections so a caller can resample them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    dataset_id, samples = manifest_samples(manifest)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        samples = samples[: args.max_samples]
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else manifest_path.parent.parent / "physics"
    )
    prepare_output_root(output_root, args.allow_existing_output)
    started = time.perf_counter()
    records = []
    schema_groups = group_samples_by_schema(root, samples)
    for schema in sorted(schema_groups):
        group = schema_groups[schema]
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(max(1, args.workers), len(group)),
            mp_context=worker_context(),
        ) as executor:
            records.extend(
                executor.map(
                    worker,
                    [str(root)] * len(group),
                    [str(output_root)] * len(group),
                    group,
                )
            )
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
        "worker_isolation": "source_schema_process_pool",
        "source_schema_counts": {
            schema: len(group) for schema, group in sorted(schema_groups.items())
        },
        "records": records,
    }
    output_path = output_root / "manifest.json"
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
    if batch_failed(
        rejected_count=len(rejected),
        error_count=len(errors),
        allow_audit_rejections=args.allow_audit_rejections,
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
