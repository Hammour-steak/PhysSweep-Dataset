#!/usr/bin/env python3
"""Admit one audited generic camera replacement into a frozen base slot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(root: Path, path: Path | str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def root_relative(root: Path, path: Path | str) -> str:
    return str(resolve(root, path).relative_to(root))


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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    base_path = resolve(root, args.base_manifest)
    existing_path = resolve(root, args.existing_replacement_manifest)
    candidate_path = resolve(root, args.candidate_manifest)
    physics_path = resolve(root, args.candidate_physics_manifest)
    camera_path = resolve(root, args.camera_bound_manifest)
    output_path = resolve(root, args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output_path}")
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
    metadata_path = resolve(root, sample["metadata_path"])
    if sha256(metadata_path) != str(sample["metadata_sha256"]):
        raise ValueError("candidate metadata hash mismatch")
    candidate_metadata = load_json(metadata_path)
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
    if int(physics.get("passed_count", 0)) != 1:
        raise ValueError("candidate physics manifest does not declare one pass")
    trajectory_path = resolve(root, physics_record["trajectory_path"])
    simulation_record_path = (
        resolve(root, sample["simulation_record_path"])
        if sample.get("simulation_record_path")
        else trajectory_path.with_name("simulation_record.json")
    )
    audit_path = resolve(root, physics_record["audit_path"])
    for artifact in (simulation_record_path, trajectory_path, audit_path):
        if not artifact.is_file():
            raise FileNotFoundError(f"candidate physics artifact is missing: {artifact}")
    if sha256(audit_path) != str(physics_record["audit_sha256"]):
        raise ValueError("candidate physics audit hash mismatch")

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
    camera_metadata_path = resolve(root, camera_sample["metadata_path"])
    if sha256(camera_metadata_path) != str(camera_sample["metadata_sha256"]):
        raise ValueError("bound camera metadata hash mismatch")
    diagnostics = camera_sample.get("camera_diagnostics", {})
    if not isinstance(diagnostics, dict) or not diagnostics:
        raise ValueError("camera audit diagnostics are missing")
    if not math.isfinite(float(diagnostics["score"])):
        raise ValueError("camera audit score is not finite")

    existing = load_json(existing_path)
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
