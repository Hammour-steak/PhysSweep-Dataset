#!/usr/bin/env python3
"""Prepare schema-specific render manifests for complete sweep groups."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_BRANCH = {
    "physweep_pybullet_rigid_metadata_v1": "generic",
    "physweep_asset_proxy_scene_v3": "asset",
    "physweep_billiards_scene_v4": "billiards",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root))


def select_complete_groups(
    metadata_records: list[dict[str, Any]], selected_parents: set[str]
) -> list[dict[str, Any]]:
    selected = [
        record for record in metadata_records if str(record["parent"]) in selected_parents
    ]
    counts = Counter(str(record["parent"]) for record in selected)
    if set(counts) != selected_parents or any(value != 13 for value in counts.values()):
        raise ValueError("selected sweep records do not form complete 13-sample groups")
    return selected


def dispatched_paths(root: Path, physics: dict[str, Any]) -> dict[str, str]:
    trajectory = Path(str(physics["trajectory_path"])).resolve()
    audit = Path(str(physics["audit_path"])).resolve()
    record = trajectory.with_name("simulation_record.json")
    for path in (trajectory, audit, record):
        path.relative_to(root)
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "trajectory_path": relative(root, trajectory),
        "audit_path": relative(root, audit),
        "simulation_record_path": relative(root, record),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--staged-base-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    release_path = args.release_manifest.resolve()
    staged_path = args.staged_base_manifest.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    release = load_json(release_path)
    metadata_manifest = load_json(root / str(release["metadata_manifest"]))
    physics_manifest = load_json(root / str(release["physics_manifest"]))
    staged = load_json(staged_path)
    selected_parents = {str(record["metadata_path"]) for record in staged["records"]}
    selected = select_complete_groups(metadata_manifest["records"], selected_parents)
    physics_by_scene = {
        str(record["scene_id"]): record for record in physics_manifest["records"]
    }

    branches: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {
        name: [] for name in SCHEMA_BRANCH.values()
    }
    for record in selected:
        physics = physics_by_scene[str(record["scene_id"])]
        if not physics.get("ok") or not physics.get("audit_passed"):
            raise ValueError(f"selected physics record did not pass: {record['scene_id']}")
        branch = SCHEMA_BRANCH.get(str(record["source_schema_version"]))
        if branch is None:
            raise ValueError(f"unsupported render schema: {record['source_schema_version']}")
        branches[branch].append((record, physics))

    generic_records = []
    for record, physics in branches["generic"]:
        generic_records.append({**physics, **dispatched_paths(root, physics)})
    generic_manifest = {
        "schema_version": "physweep_pybullet_batch_record_v1",
        "dataset_id": "one_object_sweep_review_generic",
        "sample_count": len(generic_records),
        "passed_count": len(generic_records),
        "rejected_count": 0,
        "error_count": 0,
        "records": generic_records,
    }
    write_json(output_root / "generic" / "physics_manifest.json", generic_manifest)

    for branch in ("asset", "billiards"):
        render_records = []
        for record, physics in branches[branch]:
            source_path = Path(str(physics["metadata_path"])).resolve()
            source_path.relative_to(root)
            bound = copy.deepcopy(load_json(source_path))
            paths = dispatched_paths(root, physics)
            bound["source_metadata"] = {
                "path": relative(root, source_path),
                "sha256": sha256(source_path),
            }
            bound.setdefault("physics", {}).update(paths)
            scene_id = str(record["scene_id"])
            bound_path = output_root / branch / "metadata" / f"{scene_id}.json"
            write_json(bound_path, bound)
            render_records.append(
                {
                    "scene_id": scene_id,
                    "metadata_path": relative(root, bound_path),
                    "render_output": {
                        "video_path": relative(
                            root, output_root / branch / "videos" / f"{scene_id}.mp4"
                        ),
                        "inspection_frame_dir": relative(
                            root, output_root / branch / "frames" / scene_id
                        ),
                    },
                }
            )
        manifest = {
            "schema_version": f"physweep_sweep_{branch}_render_manifest_v1",
            "dataset_id": f"one_object_sweep_review_{branch}",
            "output_root": relative(root, output_root / branch),
            "sample_count": len(render_records),
            "records": render_records,
        }
        write_json(output_root / branch / "render_input_manifest.json", manifest)
        base_records = [
            record
            for record in render_records
            if load_json(root / record["metadata_path"])["sweep"]["kind"] == "base"
        ]
        base_manifest = dict(manifest)
        base_manifest["dataset_id"] = f"one_object_sweep_review_{branch}_base"
        base_manifest["sample_count"] = len(base_records)
        base_manifest["records"] = base_records
        write_json(
            output_root / branch / "base_render_input_manifest.json",
            base_manifest,
        )
        derived_records = [
            record
            for record in render_records
            if load_json(root / record["metadata_path"])["sweep"]["kind"]
            == "sweep"
        ]
        derived_manifest = dict(manifest)
        derived_manifest["dataset_id"] = (
            f"one_object_sweep_review_{branch}_derived"
        )
        derived_manifest["sample_count"] = len(derived_records)
        derived_manifest["records"] = derived_records
        write_json(
            output_root / branch / "derived_render_input_manifest.json",
            derived_manifest,
        )

    summary = {
        "schema_version": "physweep_sweep_render_plan_v1",
        "source_release": relative(root, release_path),
        "source_staged_base_manifest": relative(root, staged_path),
        "group_count": len(selected_parents),
        "sample_count": len(selected),
        "branch_counts": {name: len(records) for name, records in branches.items()},
    }
    write_json(output_root / "manifest.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
