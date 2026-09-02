#!/usr/bin/env python3
"""Stage audited generic and specialized 2obj bases for production rendering."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Any

from tools.core.hashing import relative_file_binding, sha256_file
from tools.core.json_io import read_json, write_json_atomic_sorted
from tools.core.paths import resolve_project_path_within_root
from tools.dataset_contract.object_identity_contract import attach_object_identity
from tools.rendering.prepare_sweep_render_manifests import dispatched_paths
from tools.sampling.assemble_two_object_base import SPECIALIZED_FAMILIES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "physweep_two_object_base_render_plan_v1"
GENERIC_SCHEMA = "physweep_pybullet_rigid_metadata_v1"


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _records(document: dict[str, Any], label: str) -> list[dict[str, Any]]:
    records = document.get("records")
    if not isinstance(records, list) or int(document.get("sample_count", -1)) != len(records):
        raise ValueError(f"{label} count differs")
    return records


def prepare_base_render_plan(
    root: Path,
    base_manifest_path: Path,
    physics_manifest_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    root = root.resolve()
    base_manifest_path = resolve_project_path_within_root(root, base_manifest_path)
    physics_manifest_path = resolve_project_path_within_root(root, physics_manifest_path)
    output_root = resolve_project_path_within_root(root, output_root)
    if (root / "outputs").resolve() not in output_root.parents:
        raise ValueError("two-object base render plan must remain below root/outputs")
    if output_root.exists():
        raise FileExistsError(output_root)

    base = read_json(base_manifest_path)
    physics = read_json(physics_manifest_path)
    if (
        base.get("schema_version") != "physweep_two_object_base_manifest_v1"
        or base.get("dataset_id") != "physweep_two_object"
        or int(base.get("object_count", -1)) != 2
    ):
        raise ValueError("unsupported two-object base manifest")
    base_records = _records(base, "two-object base manifest")
    physics_records = _records(physics, "two-object base physics manifest")
    if (
        int(physics.get("passed_count", -1)) != len(physics_records)
        or int(physics.get("rejected_count", -1)) != 0
        or int(physics.get("error_count", -1)) != 0
    ):
        raise ValueError("two-object base physics did not pass completely")
    physics_by_scene = {str(record.get("scene_id", "")): record for record in physics_records}
    if "" in physics_by_scene or len(physics_by_scene) != len(physics_records):
        raise ValueError("two-object base physics contains duplicate scene ids")
    if set(physics_by_scene) != {str(record["scene_id"]) for record in base_records}:
        raise ValueError("two-object base and physics scene ids differ")

    branch_by_schema = {GENERIC_SCHEMA: "generic", **{schema: family for family, schema in SPECIALIZED_FAMILIES.items()}}
    branch_records: dict[str, list[dict[str, Any]]] = {
        branch: [] for branch in branch_by_schema.values()
    }
    generic_samples: list[dict[str, Any]] = []
    generic_physics: list[dict[str, Any]] = []
    for base_record in base_records:
        scene_id = str(base_record["scene_id"])
        source_path = resolve_project_path_within_root(
            root, Path(str(base_record["metadata_path"]))
        )
        source_digest = sha256_file(source_path)
        source = read_json(source_path)
        schema = str(base_record["source_schema_version"])
        physics_record = physics_by_scene[scene_id]
        if (
            str(source.get("scene_id", "")) != scene_id
            or str(source.get("schema_version", "")) != schema
            or source_digest != str(base_record["metadata_sha256"])
            or str(physics_record.get("metadata_sha256", "")) != source_digest
            or str(physics_record.get("source_schema_version", "")) != schema
            or not physics_record.get("ok")
            or not physics_record.get("audit_passed")
            or physics_record.get("failed_checks")
        ):
            raise ValueError(f"two-object base provenance differs: {scene_id}")
        branch = branch_by_schema.get(schema)
        if branch is None or str(base_record.get("family")) != branch:
            raise ValueError(f"two-object base family/schema differs: {scene_id}")
        if schema == GENERIC_SCHEMA:
            generic_samples.append(
                {
                    "scene_id": scene_id,
                    "metadata_path": _relative(root, source_path),
                    "metadata_sha256": source_digest,
                }
            )
            generic_physics.append(copy.deepcopy(physics_record))
            continue

        paths = dispatched_paths(root, physics_record)
        bound = copy.deepcopy(source)
        bound["source_metadata"] = {
            "path": _relative(root, source_path),
            "sha256": source_digest,
        }
        bound.setdefault("physics", {}).update(paths)
        frame_dir = output_root / branch / "frames" / scene_id
        video_path = output_root / branch / "videos" / f"{scene_id}.mp4"
        mask_path = output_root / branch / "masks" / scene_id
        bound.setdefault("render", {})["inspection_frame_dir"] = _relative(root, frame_dir)
        bound["render"]["video_path"] = _relative(root, video_path)
        attach_object_identity(
            bound,
            trajectory_path=paths["trajectory_path"],
            mask_path=_relative(root, mask_path),
        )
        bound_path = output_root / branch / "metadata" / f"{scene_id}.json"
        write_json_atomic_sorted(bound_path, bound)
        branch_records[branch].append(
            {
                "scene_id": scene_id,
                "metadata_path": _relative(root, bound_path),
                "metadata_sha256": sha256_file(bound_path),
                "render_output": {
                    "inspection_frame_dir": _relative(root, frame_dir),
                    "video_path": _relative(root, video_path),
                },
            }
        )

    generic_root = output_root / "generic"
    source_binding = base.get("sources", {}).get("generic", {})
    generic_source_document = read_json(
        resolve_project_path_within_root(root, Path(str(source_binding.get("path", ""))))
    )
    generic_source = {
        "schema_version": "physweep_pybullet_base_manifest_v1",
        "dataset_id": "physweep_two_object",
        "sample_count": len(generic_samples),
        "scene_rules": copy.deepcopy(generic_source_document.get("scene_rules")),
        "samples": generic_samples,
    }
    generic_source_path = generic_root / "source_manifest.json"
    write_json_atomic_sorted(generic_source_path, generic_source)
    generic_physics_manifest = {
        **{
            key: value
            for key, value in physics.items()
            if key not in {"source_manifest", "sample_count", "passed_count", "records", "source_schema_counts"}
        },
        "source_manifest": _relative(root, generic_source_path),
        "sample_count": len(generic_physics),
        "passed_count": len(generic_physics),
        "source_schema_counts": {GENERIC_SCHEMA: len(generic_physics)},
        "records": generic_physics,
    }
    generic_physics_path = generic_root / "physics_manifest.json"
    write_json_atomic_sorted(generic_physics_path, generic_physics_manifest)

    for branch in SPECIALIZED_FAMILIES:
        records = branch_records[branch]
        write_json_atomic_sorted(
            output_root / branch / "render_input_manifest.json",
            {
                "schema_version": "physweep_two_object_specialized_render_input_v1",
                "dataset_id": "physweep_two_object",
                "output_root": _relative(root, output_root / branch),
                "sample_count": len(records),
                "records": records,
            },
        )
    counts = Counter(str(record["family"]) for record in base_records)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "physweep_two_object",
        "object_count": 2,
        "source_base_manifest": relative_file_binding(root, base_manifest_path),
        "source_physics_manifest": relative_file_binding(root, physics_manifest_path),
        "sample_count": len(base_records),
        "branch_counts": {
            branch: int(counts.get(branch, 0))
            for branch in ("generic", *SPECIALIZED_FAMILIES)
        },
        "generic_source_manifest": _relative(root, generic_source_path),
        "generic_physics_manifest": _relative(root, generic_physics_path),
        "generic_bound_manifest": _relative(root, generic_root / "bound_manifest.json"),
        "generic_render_manifest": _relative(root, generic_root / "render_manifest.json"),
        "specialized_renderer": "two_object_specialized",
    }
    write_json_atomic_sorted(output_root / "render_plan.json", plan)
    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--physics-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = prepare_base_render_plan(
        args.root, args.base_manifest, args.physics_manifest, args.output_root
    )
    print(plan)


if __name__ == "__main__":
    main()
