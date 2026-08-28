#!/usr/bin/env python3
"""Audit bound scene cameras and rendered first-frame exposure."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from tools.rendering.appearance_adaptation import (
    LOW_CONTRAST_MEAN_CEILING,
    MIN_LUMA_STD,
    MIN_MEAN_LUMA,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--minimum-mean-luma", type=float, default=MIN_MEAN_LUMA)
    parser.add_argument("--minimum-luma-std", type=float, default=MIN_LUMA_STD)
    parser.add_argument("--maximum-dark-fraction", type=float, default=0.25)
    return parser.parse_args()


def nested(mapping: dict, *keys: str, default=None):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def image_metrics(path: Path) -> dict[str, float]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot decode {path}")
    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return {
        "mean_luma": round(float(luma.mean()), 4),
        "luma_std": round(float(luma.std()), 4),
        "dark_fraction": round(float((luma < 20.0).mean()), 6),
        "bright_fraction": round(float((luma > 235.0).mean()), 6),
    }


def simulation_metrics(metadata: dict, repository_root: Path) -> dict:
    record_path = repository_root / metadata["simulation_record"]["path"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    audit_path = Path(record["audit_path"])
    if not audit_path.is_absolute():
        audit_path = repository_root / audit_path
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    metrics = audit["metrics"]
    return {
        "physics_audit_passed": bool(audit["passed"]),
        "physics_failed_check_count": sum(
            not bool(check["passed"]) for check in audit["checks"]
        ),
        "physics_advisory_count": len(audit.get("advisories", [])),
        "horizontal_displacement_m": metrics["horizontal_displacement_m"],
        "path_length_3d_m": metrics["path_length_3d_m"],
        "vertical_range_m": metrics["vertical_range_m"],
        "maximum_linear_speed_m_s": metrics["maximum_linear_speed_m_s"],
        "maximum_angular_speed_rad_s": metrics["maximum_angular_speed_rad_s"],
        "active_motion_duration_s": metrics["active_motion_duration_s"],
        "primary_support_contact_fraction": metrics[
            "primary_support_contact_fraction"
        ],
        "maximum_penetration_m": metrics["maximum_penetration_m"],
        "mechanical_energy_gain_j": metrics["mechanical_energy_gain_j"],
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    metadata_dir = output_root / "metadata"
    frame_dir = output_root / "frames"
    repository_root = Path.cwd().resolve()
    rows: list[dict] = []

    for metadata_path in sorted(metadata_dir.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        scene_id = metadata["scene_id"]
        frame_path = frame_dir / scene_id / "frame_0001.png"
        diagnostics = nested(
            metadata, "visualization", "camera", "diagnostics", default={}
        )
        dimensions = nested(
            metadata,
            "semantic_sampling",
            "five_dimensions",
            default={},
        )
        support = dimensions.get("support_interaction", {})
        motion = dimensions.get("motion", {})
        metrics = image_metrics(frame_path)
        physics = simulation_metrics(metadata, repository_root)
        row = {
            "scene_id": scene_id,
            "support_type": support.get("support_type"),
            "scene_visual_profile": support.get("scene_visual_profile"),
            "motion_family": motion.get("family"),
            "motion_subtype": motion.get("subtype"),
            "camera_distance_m": diagnostics.get("distance_m"),
            "initial_object_span_ndc": diagnostics.get("initial_object_span_ndc"),
            "median_object_span_ndc": diagnostics.get(
                "median_primary_object_span_ndc"
            ),
            "initial_object_visible_fraction": diagnostics.get(
                "initial_object_visible_fraction"
            ),
            "primary_center_visible_fraction": diagnostics.get(
                "primary_center_visible_fraction"
            ),
            "full_center_visible_fraction": diagnostics.get(
                "full_trajectory_center_visible_fraction"
            ),
            "support_context_visible_fraction": diagnostics.get(
                "support_context_visible_fraction"
            ),
            "required_anchor_visible_fraction": diagnostics.get(
                "required_structure_anchor_visible_fraction"
            ),
            "trajectory_unoccluded_fraction": diagnostics.get(
                "primary_trajectory_unoccluded_fraction"
            ),
            **physics,
            **metrics,
        }
        flags = []
        if not row["physics_audit_passed"]:
            flags.append("physics_audit_failed")
        if row["initial_object_visible_fraction"] != 1.0:
            flags.append("initial_object_clipped")
        if row["mean_luma"] < args.minimum_mean_luma:
            flags.append("dark_mean")
        if (
            row["luma_std"] < args.minimum_luma_std
            and row["mean_luma"] < LOW_CONTRAST_MEAN_CEILING
        ):
            flags.append("low_contrast")
        if row["dark_fraction"] > args.maximum_dark_fraction:
            flags.append("large_dark_region")
        row["flags"] = ";".join(flags)
        rows.append(row)

    if not rows:
        raise SystemExit(f"no metadata found below {metadata_dir}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    flagged = [row for row in rows if row["flags"]]
    summary = {
        "sample_count": len(rows),
        "flagged_count": len(flagged),
        "thresholds": {
            "minimum_mean_luma": args.minimum_mean_luma,
            "minimum_luma_std": args.minimum_luma_std,
            "maximum_dark_fraction": args.maximum_dark_fraction,
        },
        "ranges": {
            key: {
                "minimum": min(float(row[key]) for row in rows),
                "maximum": max(float(row[key]) for row in rows),
            }
            for key in (
                "camera_distance_m",
                "initial_object_span_ndc",
                "median_object_span_ndc",
                "initial_object_visible_fraction",
                "primary_center_visible_fraction",
                "full_center_visible_fraction",
                "support_context_visible_fraction",
                "required_anchor_visible_fraction",
                "trajectory_unoccluded_fraction",
                "horizontal_displacement_m",
                "path_length_3d_m",
                "vertical_range_m",
                "maximum_linear_speed_m_s",
                "maximum_angular_speed_rad_s",
                "active_motion_duration_s",
                "primary_support_contact_fraction",
                "maximum_penetration_m",
                "mechanical_energy_gain_j",
                "mean_luma",
                "luma_std",
                "dark_fraction",
                "bright_fraction",
            )
        },
        "flagged_samples": [
            {
                "scene_id": row["scene_id"],
                "support_type": row["support_type"],
                "motion_family": row["motion_family"],
                "flags": row["flags"].split(";"),
            }
            for row in flagged
        ],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"samples={len(rows)} flagged={len(flagged)} "
        f"csv={args.csv.resolve()} json={args.json.resolve()}"
    )


if __name__ == "__main__":
    main()
