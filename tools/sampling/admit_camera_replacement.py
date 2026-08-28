#!/usr/bin/env python3
"""Admit one audited generic camera replacement into a frozen base slot."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json_atomic as write_json
from tools.core.paths import (
    project_relative_path as root_relative,
    resolve_project_path as resolve,
)
from tools.sampling.sample_one_object_scene_matrix import generic_retry_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def require_single_record(document: dict[str, Any], key: str) -> dict[str, Any]:
    records = document.get(key)
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError(f"manifest must contain exactly one {key} record")
    if int(document.get("sample_count", len(records))) != 1:
        raise ValueError("manifest sample_count must be one")
    return records[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--existing-replacement-manifest", type=Path, required=True)
    parser.add_argument("--slot-index", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-physics-manifest", type=Path, required=True)
    parser.add_argument("--camera-bound-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.attempt < 2:
        raise ValueError("camera replacement attempts begin at two")
    root = args.root.resolve()
    base_path = resolve(root, args.base_manifest)
    existing_path = resolve(root, args.existing_replacement_manifest)
    candidate_path = resolve(root, args.candidate_manifest)
    physics_path = resolve(root, args.candidate_physics_manifest)
    camera_path = resolve(root, args.camera_bound_manifest)
    for source_path in (
        base_path,
        existing_path,
        candidate_path,
        physics_path,
        camera_path,
    ):
        source_path.relative_to(root)
    output_path = resolve(root, args.output)
    if output_path.exists():
        raise FileExistsError(f"replacement output already exists: {output_path}")
    datasets_root = (root / "datasets").resolve()
    if output_path.parent != datasets_root and datasets_root not in output_path.parents:
        raise ValueError("replacement manifest must remain under datasets")

    base = load_json(base_path)
    base_records = [
        record
        for record in base["records"]
        if int(record["index"]) == args.slot_index
    ]
    if len(base_records) != 1:
        raise ValueError(f"base slot is not unique: {args.slot_index}")
    original = base_records[0]
    if original["pipeline"] != "generic_pybullet":
        raise ValueError("camera replacement is only valid for a generic base slot")
    original_metadata_path = resolve(root, original["metadata_path"])
    if sha256(original_metadata_path) != str(original["metadata_sha256"]):
        raise ValueError("original base metadata hash mismatch")

    source_generic_path = resolve(root, base["generic_manifest_path"])
    source_generic = load_json(source_generic_path)
    candidate = load_json(candidate_path)
    sample = require_single_record(candidate, "samples")
    expected_seed = generic_retry_seed(
        int(base["seed"]), args.slot_index, args.attempt
    )
    if int(candidate["seed"]) != expected_seed:
        raise ValueError("candidate seed is not the deterministic slot retry seed")
    metadata_path = resolve(root, sample["metadata_path"])
    if sha256(metadata_path) != str(sample["metadata_sha256"]):
        raise ValueError("candidate metadata hash mismatch")
    candidate_metadata = load_json(metadata_path)
    if str(candidate_metadata["scene_id"]) != str(sample["scene_id"]):
        raise ValueError("candidate metadata scene id does not match its manifest")
    if int(candidate_metadata["seed"]) != expected_seed:
        raise ValueError("candidate metadata seed does not match its manifest")
    generated_motion = str(
        candidate_metadata["simulation"]["objects"][0]["expected_motion"][
            "motion_family"
        ]
    )
    if generated_motion != str(original["motion_intent"]):
        raise ValueError("candidate motion does not preserve the base slot")
    frozen_fields = (
        "production_spec",
        "sampling_bundle_path",
        "sampling_bundle_sha256",
        "rules_path",
        "rules_sha256",
        "compiled_from",
        "implementation",
        "backend_path",
        "backend_sha256",
    )
    for field in frozen_fields:
        if candidate.get(field) != source_generic.get(field):
            raise ValueError(f"candidate changes frozen generic provenance: {field}")

    physics = load_json(physics_path)
    physics_record = require_single_record(physics, "records")
    if (
        not physics_record.get("ok")
        or not physics_record.get("audit_passed")
        or physics_record.get("failed_checks")
    ):
        raise ValueError("candidate physics audit did not pass")
    if str(physics_record["scene_id"]) != str(sample["scene_id"]):
        raise ValueError("candidate physics scene id does not match metadata")
    if resolve(root, physics_record["metadata_path"]) != metadata_path:
        raise ValueError("candidate physics metadata path does not match metadata")
    if str(physics_record["metadata_sha256"]) != str(sample["metadata_sha256"]):
        raise ValueError("candidate physics metadata hash does not match metadata")
    if (
        int(physics.get("passed_count", 0)) != 1
        or int(physics.get("rejected_count", 0)) != 0
        or int(physics.get("error_count", 0)) != 0
    ):
        raise ValueError("candidate physics manifest does not declare one clean pass")
    trajectory_path = resolve(root, physics_record["trajectory_path"])
    simulation_record_path = (
        resolve(root, sample["simulation_record_path"])
        if sample.get("simulation_record_path")
        else trajectory_path.with_name("simulation_record.json")
    )
    audit_path = resolve(root, physics_record["audit_path"])
    resolved_scene_path = resolve(root, physics_record["resolved_scene_path"])
    for artifact in (
        simulation_record_path,
        resolved_scene_path,
        trajectory_path,
        audit_path,
    ):
        if not artifact.is_file():
            raise FileNotFoundError(f"candidate physics artifact is missing: {artifact}")
    if sha256(trajectory_path) != str(physics_record["trajectory_sha256"]):
        raise ValueError("candidate trajectory hash mismatch")
    if sha256(audit_path) != str(physics_record["audit_sha256"]):
        raise ValueError("candidate physics audit hash mismatch")
    if sha256(resolved_scene_path) != str(physics_record["resolved_scene_sha256"]):
        raise ValueError("candidate resolved scene hash mismatch")
    simulation_record = load_json(simulation_record_path)
    simulation_bindings = (
        ("metadata", metadata_path, sample["metadata_sha256"]),
        (
            "resolved_scene",
            resolved_scene_path,
            physics_record["resolved_scene_sha256"],
        ),
        ("trajectory", trajectory_path, physics_record["trajectory_sha256"]),
        ("audit", audit_path, physics_record["audit_sha256"]),
    )
    if str(simulation_record["scene_id"]) != str(sample["scene_id"]):
        raise ValueError("candidate simulation record scene id mismatch")
    for label, expected_path, expected_hash in simulation_bindings:
        if resolve(root, simulation_record[f"{label}_path"]) != expected_path:
            raise ValueError(f"candidate simulation {label} path mismatch")
        if str(simulation_record[f"{label}_sha256"]) != str(expected_hash):
            raise ValueError(f"candidate simulation {label} hash mismatch")

    camera = load_json(camera_path)
    camera_sample = require_single_record(camera, "samples")
    if str(camera_sample["scene_id"]) != str(sample["scene_id"]):
        raise ValueError("camera audit scene id does not match candidate")
    if str(camera["camera_rules"]["sha256"]) != str(candidate["rules_sha256"]):
        raise ValueError("camera audit used different camera rules")
    if str(camera["implementation"]["sha256"]) != str(
        candidate["implementation"]["visual_binder"]["sha256"]
    ):
        raise ValueError("camera audit used a different visual binder")
    camera_rules_path = resolve(root, camera["camera_rules"]["path"])
    binder_path = resolve(root, camera["implementation"]["path"])
    camera_rules_path.relative_to(root)
    binder_path.relative_to(root)
    if camera_rules_path != resolve(root, candidate["rules_path"]):
        raise ValueError("camera audit rules path does not match the candidate")
    if sha256(camera_rules_path) != str(camera["camera_rules"]["sha256"]):
        raise ValueError("camera audit rules file hash mismatch")
    if sha256(binder_path) != str(camera["implementation"]["sha256"]):
        raise ValueError("camera audit binder file hash mismatch")
    camera_metadata_path = resolve(root, camera_sample["metadata_path"])
    if sha256(camera_metadata_path) != str(camera_sample["metadata_sha256"]):
        raise ValueError("bound camera metadata hash mismatch")
    diagnostics = camera_sample.get("camera_diagnostics", {})
    if not isinstance(diagnostics, dict) or not diagnostics:
        raise ValueError("camera audit diagnostics are missing")
    if not math.isfinite(float(diagnostics["score"])):
        raise ValueError("camera audit score is not finite")
    if resolve(root, camera_sample["trajectory_path"]) != trajectory_path:
        raise ValueError("camera audit trajectory does not match candidate physics")
    bound_metadata = load_json(camera_metadata_path)
    if str(bound_metadata["scene_id"]) != str(sample["scene_id"]):
        raise ValueError("bound camera metadata scene id does not match candidate")
    if bound_metadata["visualization"]["camera"]["diagnostics"] != diagnostics:
        raise ValueError("camera diagnostics do not match bound metadata")
    bound_artifacts = (
        ("source metadata", bound_metadata["source_metadata"], metadata_path),
        ("trajectory", bound_metadata["trajectory"], trajectory_path),
        (
            "simulation record",
            bound_metadata["simulation_record"],
            simulation_record_path,
        ),
    )
    for label, binding, expected_path in bound_artifacts:
        if resolve(root, binding["path"]) != expected_path:
            raise ValueError(f"bound camera {label} path does not match candidate")
        if sha256(expected_path) != str(binding["sha256"]):
            raise ValueError(f"bound camera {label} hash mismatch")

    existing = load_json(existing_path)
    existing_indices = [int(record["index"]) for record in existing["records"]]
    if int(existing.get("sample_count", -1)) != len(existing_indices):
        raise ValueError("existing replacement sample count is inconsistent")
    if len(existing_indices) != len(set(existing_indices)):
        raise ValueError("existing replacement manifest contains duplicate indices")
    if any(int(record["index"]) == args.slot_index for record in existing["records"]):
        raise ValueError("replacement manifest already contains this slot")
    replacement_scene_id = (
        f"{original['scene_id']}__camera_replacement_{args.attempt:02d}"
    )
    replacement_record = {
        **original,
        "scene_id": replacement_scene_id,
        "seed": int(candidate["seed"]),
        "candidate_scene_id": str(sample["scene_id"]),
        "metadata_path": root_relative(root, metadata_path),
        "metadata_sha256": str(sample["metadata_sha256"]),
        "simulation_record_path": root_relative(root, simulation_record_path),
        "trajectory_path": root_relative(root, trajectory_path),
        "status": "simulated_camera_accepted",
        "replaces_scene_id": str(original["scene_id"]),
        "replaces_metadata_path": str(original["metadata_path"]),
        "replacement_seed": int(candidate["seed"]),
        "camera_replacement": {
            "attempt": args.attempt,
            "candidate_manifest": root_relative(root, candidate_path),
            "candidate_manifest_sha256": sha256(candidate_path),
            "candidate_physics_manifest": root_relative(root, physics_path),
            "candidate_physics_manifest_sha256": sha256(physics_path),
            "trajectory_audit": root_relative(root, audit_path),
            "trajectory_audit_sha256": sha256(audit_path),
            "camera_bound_manifest": root_relative(root, camera_path),
            "camera_bound_manifest_sha256": sha256(camera_path),
            "camera_rules_sha256": str(camera["camera_rules"]["sha256"]),
            "visual_binder_sha256": str(camera["implementation"]["sha256"]),
        },
    }
    records = [*existing["records"], replacement_record]
    records.sort(key=lambda record: int(record["index"]))
    output = {
        **existing,
        "schema_version": "physweep_one_object_replacement_manifest_v2",
        "created_at": str(candidate["created_at"]),
        "sample_count": len(records),
        "records": records,
        "release_sources": {
            "base_manifest": root_relative(root, base_path),
            "base_manifest_sha256": sha256(base_path),
            "existing_replacement_manifest": root_relative(root, existing_path),
            "existing_replacement_manifest_sha256": sha256(existing_path),
            "camera_candidate_manifest": root_relative(root, candidate_path),
            "camera_candidate_manifest_sha256": sha256(candidate_path),
            "camera_physics_manifest": root_relative(root, physics_path),
            "camera_physics_manifest_sha256": sha256(physics_path),
            "camera_bound_manifest": root_relative(root, camera_path),
            "camera_bound_manifest_sha256": sha256(camera_path),
        },
    }
    write_json(output_path, output)
    print(f"replacement manifest: {output_path}")
    print(f"replacements={len(records)} admitted_slot={args.slot_index}")


if __name__ == "__main__":
    main()
