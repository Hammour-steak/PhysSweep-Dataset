#!/usr/bin/env python3
"""Finalize the compact core, reserve, and proxy-refinement PhysAssets pools."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from tools.assets.physassets_alignment import best_axis_alignment


CORE_CAPS = {
    "sports_ball": 20,
    "barrel": 12,
    "can": 12,
    "box_package": 18,
    "book": 12,
    "crate": 6,
    "bowl_pot_jar": 3,
    "cylinder_roll": 3,
    "block_die": 2,
    "drinkware": 2,
    "other": 2,
    "rod": 2,
    "bottle": 0,
}


def shape_reasons(row: dict, shape: dict, visual: dict) -> list[str]:
    method = str(row["method"])
    ratio = float(visual["proxy_to_visual_hull_volume_ratio"])
    fill = float(visual["median_silhouette_fill"])
    reasons = []
    if method == "semantic_sphere":
        if float(shape.get("sphere_radius_cv", 99.0)) > 0.05:
            reasons.append("non_spherical_radius_variation")
        if float(shape.get("sphere_radius_p10_over_p90", 0.0)) < 0.90:
            reasons.append("non_spherical_outliers")
        return reasons
    if method == "semantic_box":
        if ratio > 1.15:
            reasons.append("box_overfill")
        if fill < 0.72:
            reasons.append("sparse_box_silhouette")
        return reasons
    if method not in {"semantic_upright_cylinder", "semantic_axis_cylinder"}:
        return ["unsupported_core_proxy_method"]
    if float(shape.get("radial_extent_min_over_max", 0.0)) < 0.90:
        reasons.append("non_circular_cross_section")
    minimum_fill = 0.80 if method == "semantic_upright_cylinder" else 0.55
    if fill < minimum_fill:
        reasons.append("sparse_cylinder_silhouette")
    if bool(shape.get("radial_profile_valid")):
        if float(shape.get("radial_profile_min_over_max", 0.0)) < 0.75:
            reasons.append("varying_axial_radius")
        if float(shape.get("radial_profile_cv", 99.0)) > 0.10:
            reasons.append("irregular_axial_profile")
    elif ratio > 1.15:
        reasons.append("cylinder_overfill_without_profile")
    return reasons


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--shape-fit", type=Path, required=True)
    parser.add_argument("--visual-scores", type=Path, required=True)
    parser.add_argument("--blender-bounds", type=Path, required=True)
    parser.add_argument("--manual-review-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    selected = [json.loads(line) for line in args.selection.read_text(encoding="utf-8").splitlines() if line.strip()]
    shapes = {str(row["sample_id"]): row for row in (json.loads(line) for line in args.shape_fit.read_text(encoding="utf-8").splitlines() if line.strip())}
    visuals = {str(row["sample_id"]): row for row in (json.loads(line) for line in args.visual_scores.read_text(encoding="utf-8").splitlines() if line.strip())}
    blender_bounds = {str(row["sample_id"]): row for row in (json.loads(line) for line in args.blender_bounds.read_text(encoding="utf-8").splitlines() if line.strip())}
    manual_review = {
        "category_proxy_refinement": {},
        "asset_proxy_refinement": {},
    }
    if args.manual_review_config:
        manual_review.update(json.loads(args.manual_review_config.read_text(encoding="utf-8")))

    shape_passed = defaultdict(list)
    refinement = []
    for row in selected:
        sid = str(row["sample_id"])
        reasons = shape_reasons(row, shapes[sid], visuals[sid])
        proxy = json.loads(Path(row["proxy_json"]).read_text(encoding="utf-8"))
        colliders = proxy["proxy"]["colliders"]
        if len(colliders) != 1:
            alignment = {"passed": False, "reason": "multiple_colliders"}
            reasons.append("blender_import_proxy_alignment:multiple_colliders")
        else:
            source_extent = [float(value) for value in blender_bounds[sid]["blender_import_extent"]]
            target_extent = [float(value) for value in colliders[0]["size_m"]]
            euler, predicted, error = best_axis_alignment(source_extent, target_extent)
            alignment = {
                "passed": error <= 0.06,
                "euler_degrees": euler,
                "predicted_extent_m": predicted,
                "maximum_relative_error": error,
            }
            if error > 0.06:
                reasons.append("blender_import_proxy_alignment:relative_error_above_0.06")
        category_reason = manual_review["category_proxy_refinement"].get(str(row["category"]))
        asset_reason = manual_review["asset_proxy_refinement"].get(sid)
        if category_reason:
            reasons.append(f"manual_category_review:{category_reason}")
        if asset_reason:
            reasons.append(f"manual_asset_review:{asset_reason}")
        enriched = {**row, "local_shape_fit": shapes[sid], "blender_alignment_fit": alignment, "final_review_reasons": reasons}
        if reasons:
            enriched["final_status"] = "proxy_refinement"
            refinement.append(enriched)
        else:
            shape_passed[str(row["category"])].append(enriched)

    core, reserve = [], []
    for group, rows in shape_passed.items():
        rows.sort(key=lambda row: (-float(row["quality_score"]), int(row["sample_id"])))
        cap = CORE_CAPS[group]
        for index, row in enumerate(rows):
            if index < cap:
                core.append({**row, "final_status": "core", "final_review_reasons": []})
            else:
                reserve.append({**row, "final_status": "quality_reserve", "final_review_reasons": ["category_cap"]})

    core.sort(key=lambda row: (str(row["category"]), -float(row["quality_score"]), int(row["sample_id"])))
    reserve.sort(key=lambda row: (str(row["category"]), -float(row["quality_score"]), int(row["sample_id"])))
    refinement.sort(key=lambda row: (str(row["category"]), -float(row["quality_score"]), int(row["sample_id"])))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "core_assets.jsonl", core)
    write_jsonl(args.output_dir / "quality_reserve.jsonl", reserve)
    write_jsonl(args.output_dir / "proxy_refinement.jsonl", refinement)
    summary = {
        "reviewed_candidates": len(selected),
        "core_assets": len(core),
        "quality_reserve": len(reserve),
        "proxy_refinement": len(refinement),
        "core_by_category": dict(sorted(Counter(row["category"] for row in core).items())),
        "refinement_by_category": dict(sorted(Counter(row["category"] for row in refinement).items())),
        "refinement_reasons": dict(sorted(Counter(reason for row in refinement for reason in row["final_review_reasons"]).items())),
        "core_caps": CORE_CAPS,
        "manual_review_config": str(args.manual_review_config) if args.manual_review_config else None,
    }
    (args.output_dir / "final_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
