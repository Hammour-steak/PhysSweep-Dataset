#!/usr/bin/env python3
"""Select one deterministic review scene for every mesh environment profile."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def review_score(metadata: dict[str, Any]) -> tuple[int, int, int, int, int]:
    dimensions = metadata["semantic_sampling"]["five_dimensions"]
    support = dimensions["support_interaction"]
    camera = dimensions["camera_observation"]
    motion = dimensions["motion"]
    support_class_priority = {
        "ground_flat": 0,
        "raised_flat": 1,
        "ground_feature": 2,
        "raised_feature": 3,
    }
    camera_priority = {
        "front_left_oblique": 0,
        "front_right_oblique": 0,
        "left_oblique": 1,
        "right_oblique": 1,
        "rear_oblique": 2,
        "top_oblique": 3,
    }
    motion_priority = {
        "roll_or_slide_1obj": 0,
        "slide_push_1obj": 0,
        "bounce_1obj": 1,
        "drop_fall_1obj": 1,
        "edge_fall_1obj": 2,
        "wall_impact_1obj": 2,
        "projectile_1obj": 3,
        "arc_projectile_1obj": 3,
        "ramp_to_flat_1obj": 4,
        "slope_slide_down_1obj": 4,
        "slope_slide_up_1obj": 4,
    }
    motion_family = str(motion["family"])
    return (
        motion_priority[motion_family],
        support_class_priority[str(support["scene_class"])],
        0 if str(support["support_visual_type"]) == "procedural_proxy" else 1,
        camera_priority[str(camera["camera_profile"])],
        int(metadata["sample_index"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--profile-ids",
        nargs="+",
        help="Optionally select only this reviewed subset of profile ids.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_manifest = load_json(args.input_manifest)
    profiles_document = load_json(args.profiles)
    profiles = {
        str(record["id"]): record for record in profiles_document["profiles"]
    }
    if args.profile_ids:
        requested = {str(value) for value in args.profile_ids}
        missing = requested - set(profiles)
        if missing:
            raise ValueError(f"unknown requested profiles: {sorted(missing)}")
        profiles = {
            profile_id: profile
            for profile_id, profile in profiles.items()
            if profile_id in requested
        }
    candidates: dict[str, list[tuple[tuple[int, int, int, int, int], dict[str, Any]]]] = {
        profile_id: [] for profile_id in profiles
    }
    for sample in source_manifest["samples"]:
        metadata = load_json(project_path(root, str(sample["metadata_path"])))
        scene_visual = metadata["appearance"]["scene_visual"]
        if str(scene_visual.get("visual_type")) != "mesh_backdrop":
            continue
        profile_id = str(scene_visual["id"])
        if profile_id not in profiles:
            if args.profile_ids:
                continue
            raise ValueError(f"sample references unknown mesh profile: {profile_id}")
        support_id = str(
            metadata["semantic_sampling"]["five_dimensions"][
                "support_interaction"
            ]["support_type"]
        )
        if support_id not in {str(value) for value in profiles[profile_id]["support_ids"]}:
            raise ValueError(
                f"profile-support compatibility violation: {profile_id} {support_id}"
            )
        candidates[profile_id].append((review_score(metadata), sample))

    missing = [profile_id for profile_id, records in candidates.items() if not records]
    if missing:
        raise ValueError(f"source manifest lacks profile coverage: {missing}")

    selected_samples: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for profile_id in profiles:
        score, sample = min(
            candidates[profile_id], key=lambda value: (value[0], value[1]["scene_id"])
        )
        metadata = load_json(project_path(root, str(sample["metadata_path"])))
        dimensions = metadata["semantic_sampling"]["five_dimensions"]
        selected_samples.append(copy.deepcopy(sample))
        coverage.append(
            {
                "profile_id": profile_id,
                "scene_id": str(sample["scene_id"]),
                "support_id": str(dimensions["support_interaction"]["support_type"]),
                "scene_class": str(dimensions["support_interaction"]["scene_class"]),
                "camera_profile": str(dimensions["camera_observation"]["camera_profile"]),
                "motion": str(dimensions["motion"]["family"]),
                "score": list(score),
            }
        )

    output = copy.deepcopy(source_manifest)
    output["dataset_id"] = args.dataset_id
    output["sample_count"] = len(selected_samples)
    output["samples"] = selected_samples
    output["status"] = "environment_profile_coverage_pending_simulation"
    output["source_manifest"] = str(args.input_manifest)
    output["selection_policy"] = {
        "version": "physweep_scene_visual_profile_coverage_selection_v3",
        "one_scene_per_profile": True,
        "preference_order": [
            "low_vertical_extent_motion_before_projectile_and_ramp_motion",
            "flat_before_feature_support",
            "procedural_before_mesh_support_visual",
            "oblique_before_top_camera",
            "simple_before_ramp_motion",
        ],
    }
    output["environment_profile_coverage"] = coverage
    write_json(args.output, output)
    print("profiles", len(profiles))
    print("selected", len(selected_samples))
    for record in coverage:
        print(
            record["profile_id"],
            record["support_id"],
            record["camera_profile"],
            record["motion"],
        )


if __name__ == "__main__":
    main()
