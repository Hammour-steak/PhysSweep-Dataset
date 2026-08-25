#!/usr/bin/env python3
"""Deterministically resample generic base slots rejected by camera audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from sample_one_object_scene_matrix import (
    generic_retry_seed,
    load_json,
    sample_generic_candidate_batch,
    sha256,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--camera-audit-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--camera-timeout-seconds", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_attempts < 2:
        raise ValueError("--max-attempts must be at least two")
    if args.camera_timeout_seconds <= 0:
        raise ValueError("--camera-timeout-seconds must be positive")
    root = args.root.resolve()
    base_path = resolve(root, args.base_manifest)
    audit_path = resolve(root, args.camera_audit_manifest)
    output_path = resolve(root, args.output_manifest)
    if output_path.exists():
        raise FileExistsError(f"output exists: {output_path}")
    datasets_root = (root / "datasets").resolve()
    if datasets_root not in output_path.parents:
        raise ValueError("output manifest must remain under datasets")
    base = load_json(base_path)
    audit = load_json(audit_path)
    if int(audit["failure_count"]) != len(audit["failures"]):
        raise ValueError("camera audit failure count is inconsistent")
    if sha256(root / str(audit["source_manifest"])) != str(
        audit["source_manifest_sha256"]
    ):
        raise ValueError("camera audit source manifest hash mismatch")
    by_metadata_path = {
        str(record["metadata_path"]): record for record in base["records"]
    }
    originals = []
    for failure in audit["failures"]:
        metadata_path = str(failure["metadata_path"])
        if metadata_path not in by_metadata_path:
            raise ValueError(f"camera failure is absent from base manifest: {metadata_path}")
        original = by_metadata_path[metadata_path]
        if original["pipeline"] != "generic_pybullet":
            raise ValueError("camera resampling accepts generic slots only")
        originals.append(original)
    originals.sort(key=lambda record: int(record["index"]))

    generic_manifest = load_json(root / str(base["generic_manifest_path"]))
    bundle_path = root / str(generic_manifest["sampling_bundle_path"])
    production = generic_manifest["production_spec"]
    attempts: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for original in originals:
        slot_index = int(original["index"])
        for attempt in range(2, args.max_attempts + 1):
            seed = generic_retry_seed(args.base_seed, slot_index, attempt)
            dataset_id = (
                f"one_object_camera_replacements/slot_{slot_index:06d}_"
                f"attempt_{attempt:02d}"
            )
            (
                candidate_manifest_path,
                candidate_manifest,
                physics_manifest_path,
                physics_manifest,
            ) = sample_generic_candidate_batch(
                root=root,
                bundle_path=bundle_path,
                output_dataset=dataset_id,
                motions=[str(original["motion_intent"])],
                seed=seed,
                duration_s=float(production["duration_s"]),
                output_fps=int(production["output_fps"]),
                resolution=[int(value) for value in production["resolution"]],
                render_samples=int(production["samples"]),
                physics_workers=1,
            )
            sample = candidate_manifest["samples"][0]
            physics_record = physics_manifest["records"][0]
            attempt_record: dict[str, Any] = {
                "slot_index": slot_index,
                "replaces_scene_id": str(original["scene_id"]),
                "replaces_metadata_path": str(original["metadata_path"]),
                "motion_intent": str(original["motion_intent"]),
                "attempt": attempt,
                "seed": seed,
                "candidate_scene_id": str(sample["scene_id"]),
                "candidate_manifest": str(candidate_manifest_path.relative_to(root)),
                "candidate_manifest_sha256": sha256(candidate_manifest_path),
                "candidate_physics_manifest": str(physics_manifest_path.relative_to(root)),
                "candidate_physics_manifest_sha256": sha256(physics_manifest_path),
                "physics_accepted": bool(
                    physics_record.get("ok") and physics_record.get("audit_passed")
                ),
                "failed_checks": list(physics_record.get("failed_checks", [])),
            }
            if not attempt_record["physics_accepted"]:
                attempt_record["camera_accepted"] = False
                attempt_record["camera_error"] = "physics candidate was not admitted"
                attempts.append(attempt_record)
                continue
            trajectory_path = Path(str(physics_record["trajectory_path"]))
            if not trajectory_path.is_absolute():
                trajectory_path = root / trajectory_path
            source = {
                **candidate_manifest,
                "samples": [
                    {
                        **sample,
                        "simulation_record_path": str(
                            trajectory_path.with_name(
                                "simulation_record.json"
                            ).relative_to(root)
                        ),
                        "trajectory_path": str(trajectory_path.relative_to(root)),
                    }
                ],
            }
            camera_source_path = candidate_manifest_path.parent / "camera_source_manifest.json"
            write_json(camera_source_path, source)
            camera_output = (
                root
                / "outputs"
                / "one_object_camera_replacement_checks"
                / f"slot_{slot_index:06d}_attempt_{attempt:02d}"
            )
            command = [
                sys.executable,
                str(root / "tools/audit_generic_cameras.py"),
                "--root",
                str(root),
                "--manifest",
                str(camera_source_path),
                "--output-root",
                str(camera_output),
                "--workers",
                "1",
                "--overwrite",
                "--job-timeout-seconds",
                str(args.camera_timeout_seconds),
            ]
            completed = subprocess.run(
                command,
                cwd=root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            camera_stdout = completed.stdout
            camera_returncode = completed.returncode
            bound_manifest_path = camera_output / "bound_manifest.json"
            audit_manifest_path = camera_output / "camera_audit_manifest.json"
            camera_accepted = camera_returncode == 0 and bound_manifest_path.is_file()
            attempt_record.update(
                {
                    "camera_accepted": camera_accepted,
                    "camera_source_manifest": str(camera_source_path.relative_to(root)),
                    "camera_source_manifest_sha256": sha256(camera_source_path),
                    "camera_audit_manifest": (
                        str(audit_manifest_path.relative_to(root))
                        if audit_manifest_path.is_file()
                        else None
                    ),
                    "camera_audit_manifest_sha256": (
                        sha256(audit_manifest_path)
                        if audit_manifest_path.is_file()
                        else None
                    ),
                    "camera_bound_manifest": (
                        str(bound_manifest_path.relative_to(root))
                        if bound_manifest_path.is_file()
                        else None
                    ),
                    "camera_bound_manifest_sha256": (
                        sha256(bound_manifest_path)
                        if bound_manifest_path.is_file()
                        else None
                    ),
                    "camera_log_tail": "\n".join(camera_stdout.splitlines()[-20:]),
                }
            )
            attempts.append(attempt_record)
            if camera_accepted:
                accepted.append(attempt_record)
                break
        else:
            raise RuntimeError(f"camera replacement attempts exhausted for slot {slot_index}")

    manifest = {
        "schema_version": "physweep_generic_camera_resampling_v1",
        "base_manifest": str(base_path.relative_to(root)),
        "base_manifest_sha256": sha256(base_path),
        "camera_audit_manifest": str(audit_path.relative_to(root)),
        "camera_audit_manifest_sha256": sha256(audit_path),
        "base_seed": args.base_seed,
        "failed_slot_count": len(originals),
        "attempt_count": len(attempts),
        "accepted_count": len(accepted),
        "attempts": attempts,
        "accepted": accepted,
    }
    write_json(output_path, manifest)
    print(
        json.dumps(
            {
                "failed_slot_count": len(originals),
                "attempt_count": len(attempts),
                "accepted_count": len(accepted),
                "accepted": accepted,
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
