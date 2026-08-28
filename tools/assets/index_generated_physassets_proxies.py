#!/usr/bin/env python3
"""Build lightweight indexes for a generated PhysAssets proxy pool."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for path in sorted(args.root.glob("*/proxy.json"), key=lambda item: int(item.parent.name)):
        record = json.loads(path.read_text(encoding="utf-8"))
        records.append((path, record))

    pools = {"passed": [], "needs_review": []}
    methods = Counter()
    admissions = Counter()
    for path, record in records:
        admission = str(record["admission"])
        method = str(record["proxy"]["method"])
        admissions[admission] += 1
        methods[f"{method}:{admission}"] += 1
        pools[admission].append({
            "sample_id": str(record["sample_id"]),
            "objaverse_uid": str(record["objaverse_uid"]),
            "name": str(record["name"]),
            "method": method,
            "source_glb": str(record["source_glb"]),
            "proxy_json": str(path),
            "proxy_to_visual_hull_volume_ratio": record["fit_quality"]["proxy_to_visual_hull_volume_ratio"],
        })

    for name, values in pools.items():
        target = args.root / f"{name}.jsonl"
        target.write_text("".join(json.dumps(item, ensure_ascii=True) + "\n" for item in values), encoding="utf-8")
    summary = {
        "records": len(records),
        "admission_counts": dict(sorted(admissions.items())),
        "method_admission_counts": dict(sorted(methods.items())),
        "physics_validation_failures": sum(not record["validation"]["passed"] for _, record in records),
    }
    (args.root / "index_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
