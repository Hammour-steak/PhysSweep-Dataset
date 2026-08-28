#!/usr/bin/env python3
"""Promote reviewed reserve assets into a versioned PhysAssets core index."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json
from tools.core.json_io import read_jsonl
from tools.core.json_io import write_json_sorted as write_json
from tools.core.paths import join_project_path as resolved


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = read_json(config_path)
    base_path = resolved(root, str(config["base_core_index"]))
    reserve_path = resolved(root, str(config["quality_reserve_index"]))
    output_path = resolved(root, str(config["output_core_index"]))
    report_path = resolved(root, str(config["output_report"]))
    base = read_jsonl(base_path)
    reserve = read_jsonl(reserve_path)
    selected_ids = [str(value) for value in config["selected_sample_ids"]]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("selected_sample_ids contains duplicates")
    base_ids = {str(row["sample_id"]) for row in base}
    reserve_by_id = {str(row["sample_id"]): row for row in reserve}
    missing = sorted(set(selected_ids) - set(reserve_by_id))
    overlap = sorted(set(selected_ids) & base_ids)
    if missing or overlap:
        raise ValueError(f"invalid promotion set: missing={missing}, overlap={overlap}")
    maximum_fit = float(
        config["policy"]["promotion_requires_proxy_fit_ratio_at_most"]
    )
    promoted = []
    for sample_id in selected_ids:
        source = reserve_by_id[sample_id]
        if source.get("final_status") != "quality_reserve":
            raise ValueError(f"candidate is not in quality reserve: {sample_id}")
        if config["policy"]["reserve_reason_must_only_be_category_cap"] and source.get(
            "final_review_reasons"
        ) != ["category_cap"]:
            raise ValueError(f"candidate has a substantive rejection reason: {sample_id}")
        alignment = source.get("blender_alignment_fit", {})
        if not bool(alignment.get("passed")):
            raise ValueError(f"candidate alignment failed: {sample_id}")
        if float(source["proxy_to_visual_hull_volume_ratio"]) > maximum_fit:
            raise ValueError(f"candidate proxy fit exceeds policy: {sample_id}")
        overlay_root = base_path.parent / "proxy_overlays"
        for view in ("front", "side", "top"):
            matches = list(overlay_root.glob(f"{sample_id}_*_{view}.png"))
            if len(matches) != 1:
                raise ValueError(f"candidate overlay is missing or ambiguous: {sample_id}:{view}")
        proxy_path = Path(str(source["proxy_json"]))
        visual_path = Path(str(source["source_glb"]))
        if not proxy_path.is_file() or not visual_path.is_file():
            raise FileNotFoundError(f"candidate source is incomplete: {sample_id}")
        record = copy.deepcopy(source)
        record["final_status"] = "core"
        record["final_review_reasons"] = []
        record["expansion_admission"] = {
            "version": str(config["version"]),
            "reason": "manually reviewed category-cap reserve promotion",
            "proxy_overlay_views": ["front", "side", "top"],
        }
        promoted.append(record)
    expanded = base + promoted
    expected = config["expected"]
    if (
        len(base) != int(expected["base_profiles"])
        or len(promoted) != int(expected["added_profiles"])
        or len(expanded) != int(expected["expanded_profiles"])
    ):
        raise ValueError("foreground expansion counts differ from the declared contract")
    write_jsonl(output_path, expanded)
    report = {
        "version": str(config["version"]),
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": sha256(config_path),
        },
        "sources": {
            "base_core_index": {
                "path": base_path.relative_to(root).as_posix(),
                "sha256": sha256(base_path),
            },
            "quality_reserve_index": {
                "path": reserve_path.relative_to(root).as_posix(),
                "sha256": sha256(reserve_path),
            },
        },
        "output": {
            "path": output_path.relative_to(root).as_posix(),
            "sha256": sha256(output_path),
        },
        "counts": {
            "base": len(base),
            "promoted": len(promoted),
            "expanded": len(expanded),
        },
        "promoted_sample_ids": selected_ids,
    }
    write_json(report_path, report)
    print(json.dumps(report["counts"], indent=2))


if __name__ == "__main__":
    main()
