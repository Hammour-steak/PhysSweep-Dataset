#!/usr/bin/env python3
"""Bind sweep trajectories to the frozen visual binding of each base scene."""

from __future__ import annotations

import argparse
import copy
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json_atomic as write_json
from tools.core.paths import (
    project_relative_path as root_relative,
    resolve_project_path_within_root as project_path,
)
from tools.dataset_contract.trajectory_contract import object_trajectory_view
from tools.rendering.camera_solver import audit_two_object_camera


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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
                "trajectory_path": str(record["trajectory_path"]),
                "audit_path": str(record["audit_path"]),
                "simulation_record_path": str(
                    PurePosixPath(str(record["trajectory_path"])).with_name(
                        "simulation_record.json"
                    )
                ),
            }
        )
    scene_ids = [sample["scene_id"] for sample in samples]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("simulation manifest contains duplicate scene ids")
    return samples


def bind_one(
    root: Path,
    sweep_sample: dict[str, Any],
    base_bound_by_scene: dict[str, dict[str, Any]],
    output_root: Path,
    published_root: Path | None = None,
) -> dict[str, Any]:
    published_root = published_root or output_root
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

    trajectory_path = project_path(root, str(sweep_sample["trajectory_path"]))
    simulation_record_path = project_path(
        root, str(sweep_sample["simulation_record_path"])
    )
    audit_path = project_path(root, str(sweep_sample["audit_path"]))
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
    objects = sweep.get("simulation", {}).get("objects", [])
    if isinstance(objects, list) and len(objects) == 2:
        inherited_camera = bound["visualization"]["camera"]
        if inherited_camera.get("solver_version") != (
            "joint_full_motion_envelope_group_camera_v5"
        ):
            raise ValueError(
                f"two-object sweep lacks a group-envelope camera: {scene_id}"
            )
        with np.load(trajectory_path) as source:
            trajectory = {key: source[key] for key in source.files}
        trajectory = object_trajectory_view(sweep, trajectory)
        # Validate every member again against its immutable trajectory, but do
        # not write member-specific diagnostics into the shared camera record.
        # The complete group was already audited when this camera was solved.
        audit_two_object_camera(sweep, trajectory, inherited_camera)
    bound["visualization"]["binding_version"] = (
        "physweep_pybullet_sweep_visual_binding_v1"
    )
    camera_inheritance = {
        "policy": "copied_from_parent_base",
        "parent_scene_id": parent_scene_id,
        "parent_bound_metadata_path": root_relative(root, parent_bound_path),
        "parent_bound_metadata_sha256": sha256(parent_bound_path),
    }
    if isinstance(objects, list) and len(objects) == 2:
        camera_inheritance["derived_trajectory_camera_audit"] = (
            "joint_full_motion_envelope_camera_v5"
        )
    bound["visualization"]["camera_inheritance"] = camera_inheritance
    render = bound["visualization"]["render"]
    render["video_path"] = root_relative(
        root, published_root / "videos" / f"{scene_id}.mp4"
    )
    render["inspection_frame_dir"] = root_relative(
        root, published_root / "frames" / scene_id
    )
    render["instance_mask_dir"] = root_relative(
        root, published_root / "masks" / scene_id
    )
    output_path = output_root / "metadata" / f"{scene_id}.json"
    write_json(output_path, bound)
    return {
        "scene_id": scene_id,
        "parent_scene_id": parent_scene_id,
        "kind": sweep_binding["kind"],
        "mode": sweep_binding.get("mode", "one_factor"),
        "target_object_id": sweep_binding.get("target_object_id"),
        "target_object_index": sweep_binding.get("target_object_index"),
        "parameter": sweep_binding.get("parameter") or sweep_binding.get("axis"),
        "axis": sweep_binding.get("axis"),
        "level_index": sweep_binding.get("level_index"),
        "metadata_path": root_relative(
            root, published_root / "metadata" / f"{scene_id}.json"
        ),
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
    output_root = project_path(root, str(args.output_root))
    if (root / "outputs").resolve() not in output_root.parents:
        raise ValueError("sweep visual output must remain under root/outputs")
    sweep_manifest_path = project_path(root, str(args.sweep_manifest))
    base_bound_manifest_path = project_path(root, str(args.base_bound_manifest))
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
    if (
        int(base_bound_manifest["sample_count"]) != len(base_bound_manifest["samples"])
        or len(base_bound_by_scene) != len(base_bound_manifest["samples"])
    ):
        raise ValueError("base bound manifest contains duplicate or missing samples")
    if output_root.exists() and not args.overwrite:
        raise SystemExit(f"output exists; pass --overwrite: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.", dir=output_root.parent
    )
    staging = Path(temporary_output.name)
    try:
        records = [
            bind_one(root, sample, base_bound_by_scene, staging, output_root)
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
        write_json(staging / "bound_manifest.json", manifest)
        if output_root.exists():
            shutil.rmtree(output_root)
        staging.replace(output_root)
    finally:
        temporary_output.cleanup()
    print(f"bound manifest: {output_root / 'bound_manifest.json'}")
    print(f"samples: {len(records)}")


if __name__ == "__main__":
    main()
