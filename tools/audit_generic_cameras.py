#!/usr/bin/env python3
"""Audit every generic camera candidate without stopping at the first failure."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

try:
    from bind_pybullet_visuals import (
        bind_scene,
        load_json,
        manifest_rules_path,
        sha256,
    )
except ModuleNotFoundError:
    from tools.bind_pybullet_visuals import (
        bind_scene,
        load_json,
        manifest_rules_path,
        sha256,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_input_path(
    root: Path,
    value: str | Path | None,
    fallback: Path,
) -> Path:
    """Validate and preserve a project path while allowing storage symlinks."""
    if value is None:
        if fallback == Path():
            raise ValueError("camera source path is missing")
        path = fallback
    else:
        path = Path(str(value))
    declared = path if path.is_absolute() else root / path
    # ``abspath`` collapses ``..`` without dereferencing a project symlink.
    # The binder must retain this lexical path so it can publish a path relative
    # to the current release root; ordinary file reads still follow the symlink.
    declared_absolute = Path(os.path.abspath(declared))
    declared_absolute.relative_to(Path(os.path.abspath(root)))
    return declared_absolute


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
    if args.job_timeout_seconds is not None and args.job_timeout_seconds <= 0:
        raise ValueError("--job-timeout-seconds must be positive")
    root = args.root.resolve()
    manifest_path = (
        args.manifest if args.manifest.is_absolute() else root / args.manifest
    ).resolve()
    manifest_path.relative_to(root)
    manifest = load_json(manifest_path)
    rules_path = manifest_rules_path(root, manifest)
    rules = load_json(rules_path)
    output_root = (
        args.output_root
        if args.output_root.is_absolute()
        else root / args.output_root
    ).resolve()
    outputs_root = (root / "outputs").resolve()
    if outputs_root not in output_root.parents:
        raise ValueError("camera audit output must remain under outputs")
    samples = list(manifest["samples"])
    scene_ids = [str(sample["scene_id"]) for sample in samples]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("camera source manifest contains duplicate scene ids")
    jobs = []
    job_samples = []
    for sample in samples:
        metadata_path = project_input_path(
            root, sample.get("metadata_path"), Path()
        )
        if sha256(metadata_path) != str(sample["metadata_sha256"]):
            raise ValueError(f"camera source metadata hash mismatch: {metadata_path}")
        job_samples.append(sample)
        jobs.append(
            (
                root,
                metadata_path,
                project_input_path(
                    root,
                    sample.get("simulation_record_path"),
                    metadata_path.parent / "physics" / "simulation_record.json",
                ),
                project_input_path(
                    root,
                    sample.get("trajectory_path"),
                    metadata_path.parent / "physics" / "trajectory.npz",
                ),
                output_root,
                rules,
                None,
                None,
            )
        )
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
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
                "source_manifest": str(manifest_path.relative_to(root)),
                "source_manifest_sha256": sha256(manifest_path),
                "output_root": str(output_root.relative_to(root)),
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
