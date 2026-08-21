#!/usr/bin/env python3
"""Validate every reviewed static prop in an approved support pairing."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = PROJECT_ROOT / "configs/one_object_sampling_matrix.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], root: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise RuntimeError(f"command failed: {' '.join(command)}\n{tail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--output", type=Path, required=True)
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
    matrix_path = (
        args.matrix.resolve()
        if args.matrix
        else root / DEFAULT_MATRIX.relative_to(PROJECT_ROOT)
    )
    registry_path = root / "configs/asset_proxy_registry.json"
    output = args.output.resolve()
    if root not in output.parents:
        raise ValueError("validation output must remain under the PhysSweep project")
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    matrix = load_json(matrix_path)
    registry = load_json(registry_path)
    registry_by_id = {
        str(record["asset_id"]): record for record in registry["records"]
    }
    prop_environment = next(
        environment
        for environment in matrix["environments"]
        if environment["id"] == "curated_support_with_prop"
    )

    records: list[dict[str, Any]] = []
    profiles = sorted(
        {
            str(profile)
            for values in prop_environment["motion_bindings"].values()
            for profile in values
        }
    )
    job_index = 0
    for pair in prop_environment["support_prop_pairs"]:
        prop_id = str(pair["static_prop_asset_id"])
        support_id = str(pair["support_asset_id"])
        pool_id = str(pair["dynamic_pool_id"])
        dynamic_ids = [
            str(value) for value in prop_environment["dynamic_pools"][pool_id]
        ]
        prop = registry_by_id[prop_id]
        prop_slug = re.sub(
            r"[^a-z0-9]+", "_", str(prop["name"]).lower()
        ).strip("_")
        for dynamic_id in dynamic_ids:
            dynamic_slug = re.sub(
                r"[^a-z0-9]+",
                "_",
                str(registry_by_id[dynamic_id]["name"]).lower(),
            ).strip("_")
            for profile in profiles:
                child = (
                    output
                    / "props"
                    / prop_id
                    / dynamic_id
                    / profile
                )
                run(
                    [
                        sys.executable,
                        str(root / "tools/sample_asset_proxy_scenes.py"),
                        "--root",
                        str(root),
                        "--output",
                        str(child),
                        "--count",
                        "1",
                        "--seed",
                        str(args.seed + job_index),
                        "--support-id",
                        support_id,
                        "--dynamic-id",
                        dynamic_id,
                        "--static-prop-id",
                        prop_id,
                        "--profiles",
                        profile,
                        "--scene-id-prefix",
                        f"propval_{prop_slug[:16]}_{dynamic_slug[:16]}",
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
                if int(child_manifest["passed_count"]) != 1:
                    raise RuntimeError(
                        "static prop validation failed: "
                        f"{prop_id} {dynamic_id} {profile}"
                    )
                child_record = child_manifest["records"][0]
                records.append(
                    {
                        **child_record,
                        "validation_pair": {
                            "support_asset_id": support_id,
                            "static_prop_asset_id": prop_id,
                            "dynamic_pool_id": pool_id,
                            "dynamic_asset_id": dynamic_id,
                            "profile": profile,
                        },
                    }
                )
                job_index += 1

    manifest = {
        "schema_version": "physweep_static_prop_validation_batch_v2",
        "matrix": {
            "path": str(matrix_path.relative_to(root)),
            "sha256": sha256(matrix_path),
        },
        "registry": {
            "path": str(registry_path.relative_to(root)),
            "sha256": sha256(registry_path),
        },
        "output_root": str(output),
        "asset_count": len(
            {record["static_prop_asset_id"] for record in records}
        ),
        "combination_count": len(records),
        "sample_count": len(records),
        "passed_count": sum(bool(record["audit_passed"]) for record in records),
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
