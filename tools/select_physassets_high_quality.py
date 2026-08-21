#!/usr/bin/env python3
"""Select a compact, balanced, high-quality PhysAssets proxy library."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


FIT_LIMITS = {
    "semantic_sphere": 1.15,
    "semantic_box": 1.35,
    "semantic_upright_cylinder": 1.40,
    "semantic_axis_cylinder": 1.35,
}
QUOTAS = {
    "sports_ball": 30,
    "barrel": 20,
    "bottle": 25,
    "can": 20,
    "drinkware": 15,
    "bowl_pot_jar": 20,
    "box_package": 25,
    "book": 20,
    "crate": 10,
    "block_die": 8,
    "rod": 8,
    "cylinder_roll": 8,
    "other": 4,
}


def category(name: str) -> str:
    words = set(name.lower().replace("_", " ").replace("-", " ").split())
    if words & {"ball", "football", "basketball"}:
        return "sports_ball"
    if "barrel" in words:
        return "barrel"
    if "bottle" in words:
        return "bottle"
    if "can" in words:
        return "can"
    if words & {"cup", "glass", "goblet"}:
        return "drinkware"
    if words & {"bowl", "pot", "jar", "vase"}:
        return "bowl_pot_jar"
    if "book" in words or "notebook" in words:
        return "book"
    if "crate" in words:
        return "crate"
    if words & {"die", "block"}:
        return "block_die"
    if words & {"pencil", "pen", "rod", "stick", "screw", "flashlight", "log", "bat"}:
        return "rod"
    if words & {"box", "carton", "pack", "case", "kit", "wallet", "phone"}:
        return "box_package"
    if words & {"cylinder", "roll", "trash"}:
        return "cylinder_roll"
    return "other"


def percentile_scale(rows: list[dict], key: str, use_log: bool = False) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    if use_log:
        values = np.log1p(np.maximum(values, 0.0))
    low, high = np.quantile(values, [0.05, 0.95])
    if high <= low:
        return {str(row["sample_id"]): 0.5 for row in rows}
    return {
        str(row["sample_id"]): float(np.clip((value - low) / (high - low), 0.0, 1.0))
        for row, value in zip(rows, values)
    }


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def hard_reasons(row: dict) -> list[str]:
    reasons = []
    if float(row["proxy_to_visual_hull_volume_ratio"]) > FIT_LIMITS[str(row["method"])]:
        reasons.append("proxy_fit")
    if not 0.06 <= float(row["median_occupancy"]) <= 0.75:
        reasons.append("framing")
    if int(row["clipped_view_count"]) > 1:
        reasons.append("multi_view_clipping")
    if int(row["max_significant_components"]) > 2:
        reasons.append("fragmented_silhouette")
    if int(row["vertex_count"]) < 200:
        reasons.append("low_mesh_detail")
    if int(row["texture_count"]) < 1:
        reasons.append("no_texture")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.scores.read_text(encoding="utf-8").splitlines() if line.strip()]
    eligible, rejected = [], []
    for row in rows:
        row["category"] = category(str(row["name"]))
        reasons = hard_reasons(row)
        if reasons:
            rejected.append({**row, "quality_status": "rejected", "quality_reasons": reasons})
        else:
            eligible.append(row)

    scales = {
        "sharpness": percentile_scale(eligible, "median_sharpness", use_log=True),
        "detail": percentile_scale(eligible, "median_texture_detail", use_log=True),
        "entropy": percentile_scale(eligible, "median_luminance_entropy"),
        "color": percentile_scale(eligible, "median_colorfulness"),
        "mesh": percentile_scale(eligible, "vertex_count", use_log=True),
        "textures": percentile_scale(eligible, "texture_count", use_log=True),
    }
    for row in eligible:
        sid = str(row["sample_id"])
        ratio = float(row["proxy_to_visual_hull_volume_ratio"])
        fit_limit = FIT_LIMITS[str(row["method"])]
        fit = float(np.clip((fit_limit - ratio) / max(1e-9, fit_limit - 1.0), 0.0, 1.0))
        occupancy = float(row["median_occupancy"])
        framing = float(np.clip(1.0 - abs(occupancy - 0.33) / 0.33, 0.0, 1.0))
        row["quality_score"] = round(
            0.30 * fit
            + 0.18 * scales["detail"][sid]
            + 0.15 * scales["sharpness"][sid]
            + 0.10 * scales["entropy"][sid]
            + 0.08 * scales["color"][sid]
            + 0.08 * scales["mesh"][sid]
            + 0.06 * scales["textures"][sid]
            + 0.05 * framing,
            6,
        )

    grouped = defaultdict(list)
    for row in eligible:
        grouped[str(row["category"])].append(row)
    selected = []
    for group, candidates in grouped.items():
        candidates.sort(key=lambda row: (-float(row["quality_score"]), int(row["sample_id"])))
        hashes = []
        for row in candidates:
            signature = str(row["front_view_dhash"])
            if signature and any(hamming(signature, previous) <= 3 for previous in hashes):
                rejected.append({**row, "quality_status": "rejected", "quality_reasons": ["near_duplicate"]})
                continue
            if len(hashes) >= QUOTAS[group]:
                rejected.append({**row, "quality_status": "rejected", "quality_reasons": ["category_quota"]})
                continue
            hashes.append(signature)
            selected.append({
                **row,
                "quality_status": "selected_for_overlay_review",
                "representative_image": str(args.extracted_root / str(row["sample_id"]) / "000.png"),
                "decision": group,
                "object_name": str(row["name"]),
                "material": f"score={float(row['quality_score']):.3f}",
                "confidence": float(row["quality_score"]),
            })

    selected.sort(key=lambda row: (str(row["category"]), -float(row["quality_score"]), int(row["sample_id"])))
    rejected.sort(key=lambda row: int(row["sample_id"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selected.jsonl").write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in selected), encoding="utf-8")
    (args.output_dir / "rejected.jsonl").write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rejected), encoding="utf-8")
    summary = {
        "input_passed_proxies": len(rows),
        "hard_gate_eligible": len(eligible),
        "selected_for_overlay_review": len(selected),
        "selected_by_category": dict(sorted(Counter(row["category"] for row in selected).items())),
        "rejection_reasons": dict(sorted(Counter(reason for row in rejected for reason in row["quality_reasons"]).items())),
        "quotas": QUOTAS,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
