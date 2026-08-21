#!/usr/bin/env python3
"""Bind sweep trajectories to the frozen visual binding of each base scene."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def root_relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root))


def project_path(root: Path, value: str) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path is outside project root: {resolved}") from exc
    return resolved


def validated_sweep_samples(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != "physweep_pybullet_batch_record_v1":
        raise ValueError("sweep binding requires a PyBullet simulation manifest")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("simulation manifest has no records")
    sample_count = int(manifest.get("sample_count", -1))
    if sample_count != len(records):
        raise ValueError("simulation manifest sample count is inconsistent")
    if int(manifest.get("rejected_count", -1)) != 0:
        raise ValueError("simulation manifest contains rejected samples")
    if int(manifest.get("error_count", -1)) != 0:
        raise ValueError("simulation manifest contains worker errors")
    if int(manifest.get("passed_count", -1)) != sample_count:
        raise ValueError("not every sweep sample passed simulation audit")

    samples: list[dict[str, Any]] = []
    for record in records:
        if not record.get("ok") or not record.get("audit_passed"):
            raise ValueError(
                f"sweep sample did not pass simulation audit: {record.get('scene_id')}"
            )
        samples.append(
            {
                "scene_id": str(record["scene_id"]),
                "metadata_path": str(record["metadata_path"]),
            }
        )
    return samples


def bind_one(
    root: Path,
    sweep_sample: dict[str, Any],
    base_bound_by_scene: dict[str, dict[str, Any]],
    output_root: Path,
) -> dict[str, Any]:
    scene_id = str(sweep_sample["scene_id"])
    sweep_path = project_path(root, str(sweep_sample["metadata_path"]))
    sweep = load_json(sweep_path)
    sweep_binding = sweep.get("sweep")
    if not isinstance(sweep_binding, dict):
        raise ValueError(f"sweep metadata lacks sweep binding: {sweep_path}")
    parent_scene_id = str(sweep_binding["parent_scene_id"])
    parent_bound = base_bound_by_scene.get(parent_scene_id)
    if parent_bound is None:
        raise ValueError(f"parent base bound metadata is missing: {parent_scene_id}")
    parent_bound_path = project_path(root, str(parent_bound["metadata_path"]))
    parent_metadata = load_json(parent_bound_path)
    if parent_metadata.get("schema_version") != (
        "physweep_pybullet_rigid_bound_metadata_v1"
    ):
        raise ValueError(f"parent is not bound metadata: {parent_bound_path}")

    trajectory_path = sweep_path.parent / "physics" / "trajectory.npz"
    simulation_record_path = sweep_path.parent / "physics" / "simulation_record.json"
    audit_path = sweep_path.parent / "physics" / "trajectory_audit.json"
    simulation_record = load_json(simulation_record_path)
    if simulation_record.get("scene_id") != scene_id:
        raise ValueError(f"simulation scene mismatch: {scene_id}")
    if simulation_record.get("metadata_path") != str(sweep_path):
        raise ValueError(f"simulation metadata path mismatch: {scene_id}")
    if sha256(sweep_path) != str(simulation_record["metadata_sha256"]):
        raise ValueError(f"sweep metadata changed after simulation: {scene_id}")
    if sha256(trajectory_path) != str(simulation_record["trajectory_sha256"]):
        raise ValueError(f"sweep trajectory changed after simulation: {scene_id}")
    if sha256(audit_path) != str(simulation_record["audit_sha256"]):
        raise ValueError(f"sweep audit changed after simulation: {scene_id}")

    bound = copy.deepcopy(sweep)
    bound["schema_version"] = "physweep_pybullet_rigid_bound_metadata_v1"
    bound["source_metadata"] = {
        "path": root_relative(root, sweep_path),
        "sha256": sha256(sweep_path),
    }
    bound["trajectory"] = {
        "path": root_relative(root, trajectory_path),
        "sha256": sha256(trajectory_path),
    }
    bound["simulation_record"] = {
        "path": root_relative(root, simulation_record_path),
        "sha256": sha256(simulation_record_path),
    }
    bound["visualization"] = copy.deepcopy(parent_metadata["visualization"])
    bound["visualization"]["binding_version"] = (
        "physweep_pybullet_sweep_visual_binding_v1"
    )
    bound["visualization"]["camera_inheritance"] = {
        "policy": "copied_from_parent_base",
        "parent_scene_id": parent_scene_id,
        "parent_bound_metadata_path": root_relative(root, parent_bound_path),
        "parent_bound_metadata_sha256": sha256(parent_bound_path),
    }
    render = bound["visualization"]["render"]
    render["video_path"] = root_relative(
        root, output_root / "videos" / f"{scene_id}.mp4"
    )
    render["inspection_frame_dir"] = root_relative(
        root, output_root / "frames" / scene_id
    )
    output_path = output_root / "metadata" / f"{scene_id}.json"
    write_json(output_path, bound)
    return {
        "scene_id": scene_id,
        "parent_scene_id": parent_scene_id,
        "mode": sweep_binding.get("mode", "one_factor"),
        "target_object_id": sweep_binding.get("target_object_id"),
        "target_object_index": sweep_binding.get("target_object_index"),
        "parameter": sweep_binding.get("parameter", sweep_binding["axis"]),
        "axis": sweep_binding["axis"],
        "level_index": sweep_binding["level_index"],
        "metadata_path": root_relative(root, output_path),
        "metadata_sha256": sha256(output_path),
        "camera_policy": "copied_from_parent_base",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--sweep-manifest", type=Path, required=True)
    parser.add_argument("--base-bound-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    sweep_manifest_path = args.sweep_manifest.resolve()
    base_bound_manifest_path = args.base_bound_manifest.resolve()
    sweep_manifest = load_json(sweep_manifest_path)
    base_bound_manifest = load_json(base_bound_manifest_path)
    sweep_samples = validated_sweep_samples(sweep_manifest)
    if base_bound_manifest.get("schema_version") != (
        "physweep_pybullet_bound_manifest_v2"
    ):
        raise ValueError("base binding manifest has an unsupported schema")
    base_bound_by_scene = {
        str(record["scene_id"]): record
        for record in base_bound_manifest["samples"]
    }
    records = [
        bind_one(root, sample, base_bound_by_scene, output_root)
        for sample in sweep_samples
    ]
    manifest = {
        "schema_version": "physweep_pybullet_sweep_bound_manifest_v1",
        "dataset_id": str(sweep_manifest["dataset_id"]),
        "source_manifest": root_relative(root, sweep_manifest_path),
        "base_bound_manifest": root_relative(root, base_bound_manifest_path),
        "output_root": root_relative(root, output_root),
        "sample_count": len(records),
        "camera_policy": "parent_base_binding_is_frozen_for_each_one_factor_group",
        "samples": records,
    }
    write_json(output_root / "bound_manifest.json", manifest)
    print(f"bound manifest: {output_root / 'bound_manifest.json'}")
    print(f"samples: {len(records)}")


if __name__ == "__main__":
    main()
