#!/usr/bin/env python3
"""Turn raw visual classifications into conservative PhysSweep asset pools."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DIRECT_FAMILIES = {"container", "sports", "tool", "household", "natural"}
SEMANTIC_EXCLUDES = {
    "food",
    "plant",
    "clothing",
    "character",
    "architecture",
}
VALID_STRUCTURES = {"single", "simple_compound"}


def choose_bucket(row: dict) -> tuple[str, str]:
    if row.get("classifier_status") != "ok":
        return "review", "classifier_error"
    if not row.get("is_complete_mesh", False):
        return "exclude", "incomplete_mesh"
    if row.get("has_visible_base_or_ground", False):
        return "exclude", "baked_ground_or_support"
    if row.get("has_flexible_parts", False):
        return "exclude", "flexible_parts"
    family = str(row.get("semantic_family", "other"))
    if family in SEMANTIC_EXCLUDES:
        return "exclude", f"excluded_semantic_family:{family}"

    decision = str(row.get("decision", "uncertain"))
    confidence = float(row.get("confidence", 0))
    structure = str(row.get("structure", "unknown"))
    if decision == "direct":
        if (
            confidence >= 0.9
            and family in DIRECT_FAMILIES
            and structure in VALID_STRUCTURES
            and not row.get("has_moving_parts", False)
        ):
            return "direct", "conservative_direct_gate"
        return "specialized", "direct_requires_specialized_review"
    if decision == "specialized":
        return "specialized", "model_specialized"
    if decision == "support":
        return "support", "model_support"
    if decision == "exclude":
        return "exclude", "model_exclude"
    return "review", "uncertain"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows_by_id: dict[str, dict] = {}
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows_by_id[str(row["sample_id"])] = row

    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        bucket: (args.output_dir / f"{bucket}.jsonl").open("w", encoding="utf-8")
        for bucket in ("direct", "specialized", "support", "exclude", "review")
    }
    counts: Counter[str] = Counter()
    material_counts: dict[str, Counter[str]] = {
        bucket: Counter() for bucket in handles
    }
    try:
        for sample_id in sorted(rows_by_id, key=int):
            row = rows_by_id[sample_id]
            bucket, rule = choose_bucket(row)
            row["final_bucket"] = bucket
            row["final_rule"] = rule
            handles[bucket].write(json.dumps(row, ensure_ascii=True) + "\n")
            counts[bucket] += 1
            material_counts[bucket][str(row.get("material", "<missing>"))] += 1
    finally:
        for handle in handles.values():
            handle.close()

    summary = {
        "input_sample_count": len(rows_by_id),
        "bucket_counts": dict(counts),
        "material_counts": {
            bucket: dict(counter.most_common())
            for bucket, counter in material_counts.items()
        },
        "direct_gate": {
            "semantic_families": sorted(DIRECT_FAMILIES),
            "structures": sorted(VALID_STRUCTURES),
            "minimum_confidence": 0.9,
            "requires_complete_mesh": True,
            "rejects_visible_base_or_ground": True,
            "rejects_flexible_or_moving_parts": True,
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
