#!/usr/bin/env python3
"""Check the integrity of the finalized high-quality PhysAssets pools."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from tools.core.json_io import read_jsonl

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--proxy-root", type=Path, required=True)
    args = parser.parse_args()

    names = ("core_assets", "quality_reserve", "proxy_refinement")
    pools = {name: read_jsonl(args.selection_dir / f"{name}.jsonl") for name in names}
    selected = read_jsonl(args.selection_dir / "selected.jsonl")
    all_ids = [str(row["sample_id"]) for rows in pools.values() for row in rows]
    selected_ids = {str(row["sample_id"]) for row in selected}
    core = pools["core_assets"]

    missing_proxies = [
        str(row["sample_id"])
        for row in core
        if not (args.proxy_root / str(row["sample_id"]) / "proxy.json").is_file()
    ]
    missing_overlays = [
        f"{row['sample_id']}_{view}"
        for row in core
        for view in ("front", "side", "top")
        if not any((args.selection_dir / "proxy_overlays").glob(f"{row['sample_id']}_*_{view}.png"))
    ]
    errors = []
    if len(all_ids) != len(set(all_ids)):
        errors.append("final pools contain duplicate sample IDs")
    if set(all_ids) != selected_ids:
        errors.append("final pools do not partition selected.jsonl")
    if missing_proxies:
        errors.append("core assets are missing proxy.json files")
    if missing_overlays:
        errors.append("core assets are missing review overlays")

    report = {
        "status": "pass" if not errors else "fail",
        "selected": len(selected),
        "pool_counts": {name: len(rows) for name, rows in pools.items()},
        "unique_final_ids": len(set(all_ids)),
        "core_by_category": dict(sorted(Counter(row["category"] for row in core).items())),
        "core_by_proxy_method": dict(sorted(Counter(row["method"] for row in core).items())),
        "missing_proxies": missing_proxies,
        "missing_overlays": missing_overlays,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
