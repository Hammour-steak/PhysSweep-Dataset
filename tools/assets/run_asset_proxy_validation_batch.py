#!/usr/bin/env python3
"""Generate repeatable multi-profile validation scenes for asset proxies."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json
from tools.core.process import run_checked as run


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUPPORT_ID = "sketchfab_bg_8a5b41d6445c4f1fbefb2e4abfeebb0d"
DEFAULT_PROFILES = ["vertical_drop", "resting_push", "diagonal_push", "edge_exit"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-id", action="append", dest="asset_ids")
    parser.add_argument("--support-id", default=DEFAULT_SUPPORT_ID)
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 360])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    registry_path = (
        args.registry.resolve()
        if args.registry
        else root / "configs/asset_proxy_registry.json"
    )
    output = args.output.resolve()
    if root not in output.parents:
        raise ValueError("validation output must remain under the PhysSweep project")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    registry = load_json(registry_path)
    enabled = {
        str(record["asset_id"]): record
        for record in registry["records"]
        if bool(record["admission"].get("sampling_enabled", False))
        and record["proxy"]["kind"] == "dynamic_rigid"
    }
    selected_ids = list(args.asset_ids or sorted(enabled))
    unknown = sorted(set(selected_ids) - set(enabled))
    if unknown:
        raise ValueError(f"unknown enabled dynamic assets: {unknown}")

    records = []
    for index, asset_id in enumerate(selected_ids):
        record = enabled[asset_id]
        slug = re.sub(r"[^a-z0-9]+", "_", str(record["name"]).lower()).strip("_")
        child = output / "assets" / asset_id
        run(
            [
                sys.executable,
                "-m",
                "tools.sampling.sample_asset_proxy_scenes",
                "--root",
                str(root),
                "--output",
                str(child),
                "--count",
                str(len(args.profiles)),
                "--seed",
                str(args.seed + index),
                "--support-id",
                str(args.support_id),
                "--dynamic-id",
                asset_id,
                "--profiles",
                *[str(value) for value in args.profiles],
                "--no-static-props",
                "--scene-id-prefix",
                f"proxyval_{slug[:28]}",
                "--duration",
                str(args.duration),
                "--fps",
                str(args.fps),
                "--resolution",
                *[str(value) for value in args.resolution],
                "--samples",
                str(args.samples),
            ],
            root,
        )
        child_manifest = load_json(child / "manifest.json")
        if int(child_manifest["passed_count"]) != len(args.profiles):
            raise RuntimeError(f"asset proxy validation failed: {asset_id}")
        records.extend(child_manifest["records"])

    manifest = {
        "schema_version": "physweep_asset_proxy_validation_batch_v1",
        "registry": {
            "path": str(registry_path.relative_to(root)),
            "sha256": sha256(registry_path),
        },
        "support_asset_id": str(args.support_id),
        "profiles": [str(value) for value in args.profiles],
        "output_root": str(output),
        "asset_count": len(selected_ids),
        "sample_count": len(records),
        "passed_count": sum(bool(record["audit_passed"]) for record in records),
        "asset_ids": selected_ids,
        "records": records,
    }
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "output": str(output),
                "assets": manifest["asset_count"],
                "samples": manifest["sample_count"],
                "passed": manifest["passed_count"],
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
