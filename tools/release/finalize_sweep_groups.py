#!/usr/bin/env python3
"""Publish only complete, fully audited one-factor sweep groups."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AXES = ("mass_kg", "contact_friction", "contact_restitution")
DERIVED_LEVELS = (0, 1, 3, 4)


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


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path is outside project root: {resolved}") from exc
    return resolved


def relative_path(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root))


def validate_source_groups(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["parent"])].append(record)
    expected_levels = {(axis, level) for axis in AXES for level in DERIVED_LEVELS}
    for parent, group in groups.items():
        base = [record for record in group if record["kind"] == "base"]
        sweeps = [record for record in group if record["kind"] == "sweep"]
        levels = {(str(record["axis"]), int(record["level_index"])) for record in sweeps}
        if len(group) != 13 or len(base) != 1 or len(sweeps) != 12:
            raise ValueError(f"invalid sweep group cardinality: {parent}")
        if levels != expected_levels:
            raise ValueError(f"invalid one-factor levels: {parent}")
    return dict(groups)


def finalize(
    root: Path,
    source_manifest_path: Path,
    physics_manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = load_json(source_manifest_path)
    physics = load_json(physics_manifest_path)
    if source.get("schema_version") != "physweep_physics_sweep_manifest_v2":
        raise ValueError("unsupported sweep metadata manifest")
    if physics.get("schema_version") != "physweep_pybullet_batch_record_v1":
        raise ValueError("unsupported physics manifest")
    source_records = list(source["records"])
    physics_records = list(physics["records"])
    if int(source.get("sample_count", -1)) != len(source_records):
        raise ValueError("source sample count is inconsistent")
    if int(physics.get("sample_count", -1)) != len(physics_records):
        raise ValueError("physics sample count is inconsistent")

    source_by_scene = {str(record["scene_id"]): record for record in source_records}
    physics_by_scene = {str(record["scene_id"]): record for record in physics_records}
    if len(source_by_scene) != len(source_records) or len(physics_by_scene) != len(physics_records):
        raise ValueError("duplicate scene id")
    if set(source_by_scene) != set(physics_by_scene):
        raise ValueError("physics records do not exactly cover source metadata")

    groups = validate_source_groups(source_records)
    accepted_parents: list[str] = []
    rejected_groups: list[dict[str, Any]] = []
    failed_checks: Counter[str] = Counter()
    for parent, group in sorted(groups.items()):
        failed = []
        for source_record in group:
            record = physics_by_scene[str(source_record["scene_id"])]
            if not record.get("ok") or not record.get("audit_passed"):
                failed.append(
                    {
                        "scene_id": str(record["scene_id"]),
                        "failed_checks": list(record.get("failed_checks", [])),
                        "error": record.get("error"),
                    }
                )
                failed_checks.update(record.get("failed_checks", []))
        if failed:
            rejected_groups.append(
                {
                    "parent": parent,
                    "group_size": len(group),
                    "failed_sample_count": len(failed),
                    "failed_samples": failed,
                }
            )
        else:
            accepted_parents.append(parent)

    accepted_parent_set = set(accepted_parents)
    accepted_scene_ids = {
        str(record["scene_id"])
        for parent, group in groups.items()
        if parent in accepted_parent_set
        for record in group
    }
    accepted_records = [
        record
        for record in physics_records
        if str(record["scene_id"]) in accepted_scene_ids
    ]
    accepted = {
        **physics,
        "dataset_id": f"{physics['dataset_id']}_accepted_complete_groups",
        "source_sample_count": len(physics_records),
        "sample_count": len(accepted_records),
        "passed_count": len(accepted_records),
        "rejected_count": 0,
        "error_count": 0,
        "pass_rate": 1.0,
        "group_policy": "complete_one_factor_group_only",
        "source_group_count": len(groups),
        "accepted_group_count": len(accepted_parents),
        "rejected_group_count": len(rejected_groups),
        "records": accepted_records,
    }
    report = {
        "schema_version": "physweep_rejected_sweep_groups_v1",
        "source_metadata_manifest": relative_path(root, source_manifest_path),
        "source_metadata_manifest_sha256": sha256(source_manifest_path),
        "physics_manifest": relative_path(root, physics_manifest_path),
        "physics_manifest_sha256": sha256(physics_manifest_path),
        "source_group_count": len(groups),
        "accepted_group_count": len(accepted_parents),
        "rejected_group_count": len(rejected_groups),
        "accepted_sample_count": len(accepted_records),
        "failed_check_counts": dict(failed_checks),
        "groups": rejected_groups,
    }
    return accepted, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--physics-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected-output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source_path = project_path(root, args.source_manifest)
    physics_path = project_path(root, args.physics_manifest)
    output_path = project_path(root, args.output)
    rejected_path = project_path(root, args.rejected_output)
    accepted, rejected = finalize(root, source_path, physics_path)
    write_json(output_path, accepted)
    write_json(rejected_path, rejected)
    print(f"accepted manifest: {output_path}")
    print(f"rejected groups: {rejected_path}")
    print(
        f"accepted_groups={accepted['accepted_group_count']} "
        f"rejected_groups={accepted['rejected_group_count']} "
        f"accepted_samples={accepted['sample_count']}"
    )


if __name__ == "__main__":
    main()
