#!/usr/bin/env python3
"""Audit every generic camera candidate without stopping at the first failure."""

from __future__ import annotations

import argparse
import json
import signal
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from bind_pybullet_visuals import (
    bind_scene,
    load_json,
    manifest_rules_path,
    sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def audit_job(
    sample: dict[str, Any],
    job: tuple[Any, ...],
    timeout_seconds: int | None,
) -> dict[str, Any]:
    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(
            f"camera audit exceeded {timeout_seconds} seconds"
        )

    if timeout_seconds is not None:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
    try:
        return {"ok": True, "sample": bind_scene(*job)}
    except Exception as error:  # pylint: disable=broad-exception-caught
        return {
            "ok": False,
            "scene_id": str(sample["scene_id"]),
            "metadata_path": str(sample["metadata_path"]),
            "error_type": type(error).__name__,
            "error": str(error),
        }
    finally:
        if timeout_seconds is not None:
            signal.alarm(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--job-timeout-seconds", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    rules_path = manifest_rules_path(root, manifest)
    rules = load_json(rules_path)
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.job_timeout_seconds is not None and args.job_timeout_seconds <= 0:
        raise ValueError("--job-timeout-seconds must be positive")

    def sample_path(sample: dict[str, Any], key: str, fallback: Path) -> Path:
        value = sample.get(key)
        if value is None:
            return fallback
        path = Path(str(value))
        return path if path.is_absolute() else root / path

    samples = list(manifest["samples"])
    jobs = []
    job_samples = []
    for sample in samples:
        metadata_path = root / str(sample["metadata_path"])
        job_samples.append(sample)
        jobs.append(
            (
                root,
                metadata_path,
                sample_path(
                    sample,
                    "simulation_record_path",
                    metadata_path.parent / "physics" / "simulation_record.json",
                ),
                sample_path(
                    sample,
                    "trajectory_path",
                    metadata_path.parent / "physics" / "trajectory.npz",
                ),
                output_root,
                rules,
                None,
                None,
            )
        )
    if args.workers == 1:
        results = [
            audit_job(sample, job, args.job_timeout_seconds)
            for sample, job in zip(job_samples, jobs)
        ]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(
                executor.map(
                    audit_job,
                    job_samples,
                    jobs,
                    [args.job_timeout_seconds] * len(jobs),
                )
            )
    failures = [result for result in results if not result["ok"]]
    successes = [result["sample"] for result in results if result["ok"]]
    if len(successes) + len(failures) != len(samples):
        raise RuntimeError("camera audit did not account for every source sample")
    audit = {
        "schema_version": "physweep_generic_camera_audit_v1",
        "source_manifest": str(manifest_path.relative_to(root)),
        "source_manifest_sha256": sha256(manifest_path),
        "output_root": str(output_root.relative_to(root)),
        "visual_binder": {
            "path": str(Path(bind_scene.__code__.co_filename).resolve().relative_to(root)),
            "sha256": sha256(Path(bind_scene.__code__.co_filename).resolve()),
        },
        "auditor": {
            "path": str(Path(__file__).resolve().relative_to(root)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "camera_rules": {
            "path": str(rules_path.relative_to(root)),
            "sha256": sha256(rules_path),
        },
        "sample_count": len(successes) + len(failures),
        "success_count": len(successes),
        "failure_count": len(failures),
        "failures": failures,
        "samples": successes,
    }
    write_json(output_root / "camera_audit_manifest.json", audit)
    if not failures:
        write_json(
            output_root / "bound_manifest.json",
            {
                "schema_version": "physweep_pybullet_bound_manifest_v2",
                "source_manifest": str(manifest_path),
                "output_root": str(output_root),
                "implementation": audit["visual_binder"],
                "camera_rules": audit["camera_rules"],
                "sample_count": len(successes),
                "samples": successes,
                "camera_audit_manifest": str(
                    (output_root / "camera_audit_manifest.json").relative_to(root)
                ),
            },
        )
    print(
        json.dumps(
            {
                "sample_count": len(successes) + len(failures),
                "success_count": len(successes),
                "failure_count": len(failures),
                "failures": failures,
            },
            indent=2,
            ensure_ascii=True,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
