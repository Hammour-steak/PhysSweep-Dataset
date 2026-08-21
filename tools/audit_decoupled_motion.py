#!/usr/bin/env python3
"""Summarize motion and contact coverage from a decoupled dataset manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from rigid_trajectory import active_motion_duration_s


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def trajectory_metrics(path: Path, fps: float) -> dict[str, Any]:
    with np.load(path) as source:
        arrays = {key: source[key] for key in source.files}

    def first_key(exact: str, suffix: str) -> str:
        if exact in arrays:
            return exact
        matches = sorted(key for key in arrays if key.endswith(suffix))
        if not matches:
            raise KeyError(f"{path} has no {exact} or *{suffix}")
        return matches[0]

    position = np.asarray(
        arrays[first_key("position_m", "__position_m")], dtype=np.float64
    )
    if position.ndim == 3:
        position = position[:, 0, :]
    if position.ndim != 2 or position.shape[1] != 3:
        raise ValueError(f"{path} has an unsupported position shape")
    displacement = position - position[0]
    planar = np.linalg.norm(displacement[:, :2], axis=1)
    spatial = np.linalg.norm(displacement, axis=1)
    segment = np.linalg.norm(np.diff(position, axis=0), axis=1)
    velocity_keys = sorted(
        key
        for key in arrays
        if key == "linear_velocity_m_s"
        or key.endswith("__linear_velocity_m_s")
    )
    if velocity_keys:
        velocity = np.asarray(arrays[velocity_keys[0]], dtype=np.float64)
        if velocity.ndim == 3:
            velocity = velocity[:, 0, :]
        if velocity.ndim != 2 or velocity.shape[1] != 3:
            raise ValueError(f"{path} has an unsupported velocity shape")
        speed = np.linalg.norm(velocity, axis=1)
    else:
        speed = np.concatenate(([0.0], segment * fps))
    active_duration_s = active_motion_duration_s(
        np.column_stack((speed, np.zeros((speed.shape[0], 2), dtype=np.float64))),
        np.arange(speed.shape[0], dtype=np.float64) / float(fps),
        0.03,
    )
    support = arrays.get("support_contact")
    if support is None:
        keys = sorted(
            key
            for key in arrays
            if key.endswith("__primary_support_contact_count")
        )
        support = arrays[keys[0]] > 0 if keys else np.asarray([], dtype=np.int8)
    ground = arrays.get("ground_contact")
    if ground is None:
        keys = sorted(
            key
            for key in arrays
            if key.endswith("__collider_contact_count__environment_floor")
        )
        ground = arrays[keys[0]] > 0 if keys else np.asarray([], dtype=np.int8)
    prop = arrays.get("prop_contact", np.asarray([], dtype=np.int8))
    return {
        "planar_displacement_m": round(float(planar.max()), 6),
        "spatial_displacement_m": round(float(spatial.max()), 6),
        "path_length_m": round(float(segment.sum()), 6),
        "active_until_s": round(active_duration_s, 4),
        "initial_speed_m_s": round(float(speed[0]), 6),
        "maximum_speed_m_s": round(float(speed.max()), 6),
        "support_contact_frames": int(np.asarray(support, dtype=np.int8).sum()),
        "ground_contact_frames": int(np.asarray(ground, dtype=np.int8).sum()),
        "prop_contact_frames": int(np.asarray(prop, dtype=np.int8).sum()),
    }


def summarize(manifest_path: Path, project_root: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "physweep_one_object_decoupled_manifest_v3":
        raise ValueError("motion audit requires a v3 decoupled manifest")

    records: list[dict[str, Any]] = []
    for outer in manifest["records"]:
        metadata_path = project_root / str(outer["metadata_path"])
        metadata = load_json(metadata_path)
        pipeline = str(outer["pipeline"])
        if pipeline == "generic_pybullet":
            simulation = metadata["simulation"]
            object_record = simulation["objects"][0]
            trajectory_path = metadata_path.parent / "physics" / "trajectory.npz"
            fps = float(simulation["time"]["output_fps"])
            effective_friction = object_record["material"].get("contact_friction")
            prop_id = None
        elif pipeline in {"asset_proxy", "billiards"}:
            physics = metadata["physics"]
            trajectory_path = project_root / str(physics["trajectory_path"])
            fps = float(physics["output_fps"])
            calculation = physics.get("initial_state", {}).get("calculation", {})
            effective_friction = calculation.get("effective_friction")
            prop_id = metadata.get("assets", {}).get("static_prop_asset_id")
        else:
            raise ValueError(f"unknown pipeline in outer manifest: {pipeline}")
        records.append(
            {
                "scene_id": str(outer["scene_id"]),
                "motion_intent": str(outer["motion_intent"]),
                "environment_id": str(outer["environment_id"]),
                "profile": str(outer["profile"]),
                "pipeline": pipeline,
                "dynamic_asset_id": outer.get("dynamic_asset_id"),
                "static_prop_asset_id": prop_id,
                "effective_friction": effective_friction,
                **trajectory_metrics(trajectory_path, fps),
            }
        )

    planar = np.asarray(
        [record["planar_displacement_m"] for record in records], dtype=np.float64
    )
    active = np.asarray(
        [record["active_until_s"] for record in records], dtype=np.float64
    )
    prop_records = [
        record for record in records if record["static_prop_asset_id"] is not None
    ]
    return {
        "schema_version": "physweep_decoupled_motion_audit_v1",
        "source_manifest": str(manifest_path),
        "scene_count": len(records),
        "motions": dict(Counter(record["motion_intent"] for record in records)),
        "environments": dict(
            Counter(record["environment_id"] for record in records)
        ),
        "profiles": dict(Counter(record["profile"] for record in records)),
        "planar_displacement_m": {
            "minimum": round(float(planar.min()), 6),
            "median": round(float(np.median(planar)), 6),
            "maximum": round(float(planar.max()), 6),
            "below_0_20_m": int((planar < 0.20).sum()),
            "below_0_30_m": int((planar < 0.30).sum()),
        },
        "active_until_s": {
            "minimum": round(float(active.min()), 4),
            "median": round(float(np.median(active)), 4),
            "maximum": round(float(active.max()), 4),
            "below_1_s": int((active < 1.0).sum()),
            "below_1_5_s": int((active < 1.5).sum()),
        },
        "static_prop_scenes": len(prop_records),
        "static_prop_contact_scenes": sum(
            record["prop_contact_frames"] > 0 for record in prop_records
        ),
        "records": sorted(
            records, key=lambda record: record["planar_displacement_m"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.manifest.resolve(), args.project_root.resolve())
    text = json.dumps(result, indent=2, ensure_ascii=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
