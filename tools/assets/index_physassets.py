#!/usr/bin/env python3
"""Build a compact PhysAssets index without modifying the extracted dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


SOFT_MATERIALS = {
    "cloth",
    "fabric",
    "foam",
    "fur",
    "hair",
    "leather",
    "paper",
    "rubber",
    "skin",
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def sample_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name), path.name
    except ValueError:
        return 2**63 - 1, path.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    samples = sorted(
        (path.parent for path in args.root.glob("*/pose.json")),
        key=sample_sort_key,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    material_counts: Counter[str] = Counter()
    prefilter_counts: Counter[str] = Counter()

    fields = [
        "sample_id",
        "objaverse_uid",
        "material",
        "youngs_modulus_pa",
        "poisson_ratio",
        "representative_image",
        "material_prefilter",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            pose = load_json(sample / "pose.json")
            physical = load_json(sample / "physical.json")
            material = str(physical.get("material", "")).strip()
            normalized = material.casefold()
            prefilter = "soft_or_deformable" if normalized in SOFT_MATERIALS else "visual_review"
            material_counts[material or "<missing>"] += 1
            prefilter_counts[prefilter] += 1
            writer.writerow(
                {
                    "sample_id": sample.name,
                    "objaverse_uid": pose.get("scene_name", ""),
                    "material": material,
                    "youngs_modulus_pa": physical.get("E", ""),
                    "poisson_ratio": physical.get("nu", ""),
                    "representative_image": str(sample / "000.png"),
                    "material_prefilter": prefilter,
                }
            )

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "sample_count": len(samples),
                "material_counts": dict(material_counts.most_common()),
                "material_prefilter_counts": dict(prefilter_counts),
                "soft_material_labels": sorted(SOFT_MATERIALS),
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json.loads(summary_path.read_text()), indent=2))


if __name__ == "__main__":
    main()
