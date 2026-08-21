#!/usr/bin/env python3
"""Aggregate physics and render QA for an asset-only validation batch."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    output = Path(manifest["output_root"])
    records = []
    for source in manifest["records"]:
        metadata = load_json(root / source["metadata_path"])
        audit = load_json(root / metadata["physics"]["audit_path"])
        frame_dir = root / metadata["render"]["inspection_frame_dir"]
        render_record_path = frame_dir / "render_record.json"
        render_record = load_json(render_record_path) if render_record_path.exists() else None
        video_path = root / metadata["render"]["video_path"]
        records.append(
            {
                "scene_id": source["scene_id"],
                "dynamic_asset_id": source["dynamic_asset_id"],
                "support_asset_id": source["support_asset_id"],
                "static_prop_asset_id": source["static_prop_asset_id"],
                "motion_profile": source["motion_profile"],
                "physics_passed": bool(audit["passed"]),
                "maximum_displacement_m": float(audit["maximum_displacement_m"]),
                "maximum_linear_speed_m_s": float(audit["maximum_linear_speed_m_s"]),
                "maximum_angular_speed_rad_s": float(audit["maximum_angular_speed_rad_s"]),
                "minimum_contact_distance_m": float(audit["minimum_contact_distance_m"]),
                "support_contact_frames": int(audit["support_contact_frames"]),
                "ground_contact_frames": int(audit["ground_contact_frames"]),
                "prop_contact_frames": int(audit["prop_contact_frames"]),
                "rendered": render_record is not None and video_path.exists(),
                "video_bytes": video_path.stat().st_size if video_path.exists() else 0,
            }
        )
    displacements = [record["maximum_displacement_m"] for record in records]
    summary = {
        "schema_version": "physweep_asset_proxy_batch_qa_v1",
        "source_manifest": str(manifest_path),
        "sample_count": len(records),
        "physics_passed_count": sum(record["physics_passed"] for record in records),
        "rendered_count": sum(record["rendered"] for record in records),
        "dynamic_asset_count": len({record["dynamic_asset_id"] for record in records}),
        "support_asset_count": len({record["support_asset_id"] for record in records}),
        "static_prop_asset_count": len(
            {record["static_prop_asset_id"] for record in records if record["static_prop_asset_id"]}
        ),
        "displacement_m": {
            "minimum": round(min(displacements), 6),
            "median": round(statistics.median(displacements), 6),
            "maximum": round(max(displacements), 6),
        },
        "worst_penetration_m": round(min(record["minimum_contact_distance_m"] for record in records), 7),
        "maximum_linear_speed_m_s": round(max(record["maximum_linear_speed_m_s"] for record in records), 6),
        "maximum_angular_speed_rad_s": round(max(record["maximum_angular_speed_rad_s"] for record in records), 6),
        "scenes_with_static_prop_contact": sum(record["prop_contact_frames"] > 0 for record in records),
        "records": records,
    }
    output_path = output / "qa_summary.json"
    write_json(output_path, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
