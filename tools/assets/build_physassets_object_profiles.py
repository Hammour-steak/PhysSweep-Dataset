#!/usr/bin/env python3
"""Compile reviewed PhysAssets core records into mesh-only object profiles."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_jsonl
from tools.assets.physassets_alignment import best_axis_alignment


PHYSICS_BY_CATEGORY = {
    "sports_ball": {"friction": 0.72, "restitution": 0.62, "mass_range_kg": [0.18, 0.75]},
    "barrel": {"friction": 0.55, "restitution": 0.16, "mass_range_kg": [2.0, 24.0]},
    "can": {"friction": 0.45, "restitution": 0.22, "mass_range_kg": [0.18, 0.75]},
    "book": {"friction": 0.68, "restitution": 0.06, "mass_range_kg": [0.20, 1.80]},
    "box_package": {"friction": 0.58, "restitution": 0.08, "mass_range_kg": [0.10, 2.50]},
    "block_die": {"friction": 0.52, "restitution": 0.20, "mass_range_kg": [0.08, 0.80]},
    "other": {"friction": 0.50, "restitution": 0.12, "mass_range_kg": [0.05, 0.60]},
    "rod": {"friction": 0.48, "restitution": 0.12, "mass_range_kg": [0.08, 0.80]},
}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "asset"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--core-index", type=Path, required=True)
    parser.add_argument("--blender-bounds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--version",
        default="physweep_physassets_core_object_profiles_v2",
    )
    parser.add_argument("--base-curated-profiles", type=Path)
    parser.add_argument("--candidate-output", type=Path)
    parser.add_argument(
        "--candidate-version",
        default="physweep_physassets_core_object_profiles_candidate_v4",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    rows = read_jsonl(args.core_index.resolve())
    blender_bounds = {str(row["sample_id"]): row for row in read_jsonl(args.blender_bounds.resolve())}
    profiles = []
    for row in rows:
        proxy_path = Path(row["proxy_json"]).resolve()
        record = json.loads(proxy_path.read_text(encoding="utf-8"))
        colliders = record["proxy"]["colliders"]
        if len(colliders) != 1:
            raise ValueError(f"core asset {row['sample_id']} is not a single primitive")
        collider = colliders[0]
        shape = {"box": "cuboid", "sphere": "sphere", "cylinder": "cylinder"}[collider["shape"]]
        size = [float(value) for value in collider["size_m"]]
        imported_extent = [float(value) for value in blender_bounds[str(row["sample_id"])]["blender_import_extent"]]
        alignment, canonical_extent, alignment_error = best_axis_alignment(imported_extent, size)
        if alignment_error > 0.06:
            raise ValueError(f"asset {row['sample_id']} has no axis alignment within 6%: {alignment_error:.6f}")
        source_glb = Path(record["source_glb"]).resolve()
        try:
            relative_glb = source_glb.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"asset source is outside project root: {source_glb}") from exc
        category = str(row["category"])
        physics = PHYSICS_BY_CATEGORY[category]
        asset_id = str(record["asset_id"])
        profiles.append({
            "id": f"{asset_id}_{slug(str(row['name']))}",
            "semantic_category": f"physassets_{category}",
            "pose_profile": "support_normal",
            "visual_variants": [{
                "id": asset_id,
                "type": "mesh",
                "asset_id": asset_id,
                "path": relative_glb.as_posix(),
                "sha256": sha256(source_glb),
                "alignment_euler_degrees": alignment,
                "alignment_coordinate_frame": "blender_imported_z_up",
                "canonical_extent_m": canonical_extent,
                "material_hint": category,
                "color": [0.55, 0.55, 0.55, 1.0],
                "license": {
                    "label": "Objaverse source metadata",
                    "source_uid": str(record["objaverse_uid"]),
                    "verification_required_before_redistribution": True
                }
            }],
            "collision": {"type": shape, "dimensions_m": size},
            "physics": physics,
            "source_review": {
                "sample_id": str(row["sample_id"]),
                "quality_score": float(row["quality_score"]),
                "proxy_method": str(row["method"]),
                "proxy_json": proxy_path.relative_to(root).as_posix(),
                "blender_import_extent": imported_extent,
                "alignment_relative_error": alignment_error
            }
        })
    output = {
        "version": str(args.version),
        "visual_sampling": {"target_mesh_fraction": 1.0, "require_exact_target_when_feasible": True},
        "policy": {
            "asset_only": True,
            "primitive_visual_fallback_forbidden": True,
            "visual_and_collision_are_separate": True,
            "visual_mesh_never_defines_runtime_collision": True,
            "all_meshes_require_audited_alignment_and_hash": True,
            "mesh_alignment_coordinate_frame": "blender_imported_z_up",
            "collision_motion_exclusions": {
                "sphere": ["slide_push_1obj"]
            },
        },
        "source_core_index": args.core_index.resolve().relative_to(root).as_posix(),
        "profile_count": len(profiles),
        "profiles": profiles
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    summary = {"output": str(args.output), "profiles": len(profiles)}
    if bool(args.base_curated_profiles) != bool(args.candidate_output):
        raise ValueError(
            "--base-curated-profiles and --candidate-output must be provided together"
        )
    if args.base_curated_profiles is not None:
        curated = json.loads(
            args.base_curated_profiles.resolve().read_text(encoding="utf-8")
        )
        raw_by_id = {str(profile["id"]): profile for profile in profiles}
        curated_by_id = {
            str(profile["id"]): profile for profile in curated["profiles"]
        }
        missing = sorted(set(curated_by_id) - set(raw_by_id))
        if missing:
            raise ValueError(f"curated base profiles are absent from expansion: {missing}")
        for profile_id, curated_profile in curated_by_id.items():
            raw_profile = raw_by_id[profile_id]
            if (
                curated_profile["collision"] != raw_profile["collision"]
                or curated_profile["source_review"]["sample_id"]
                != raw_profile["source_review"]["sample_id"]
                or curated_profile["source_review"]["proxy_json"]
                != raw_profile["source_review"]["proxy_json"]
            ):
                raise ValueError(
                    f"curated base physics binding changed for {profile_id}"
                )
        candidate = dict(output)
        candidate["version"] = str(args.candidate_version)
        candidate["profiles"] = [
            curated_by_id.get(str(profile["id"]), profile) for profile in profiles
        ]
        candidate["profile_count"] = len(candidate["profiles"])
        candidate.setdefault("policy", {})["candidate_visual_source"] = {
            "base_curated_profiles": args.base_curated_profiles.resolve()
            .relative_to(root)
            .as_posix(),
            "new_profiles_use_reviewed_raw_sources": True,
        }
        args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
        args.candidate_output.write_text(
            json.dumps(candidate, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        summary["candidate_output"] = str(args.candidate_output)
        summary["base_curated_profiles"] = len(curated_by_id)
        summary["new_profiles"] = len(profiles) - len(curated_by_id)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
