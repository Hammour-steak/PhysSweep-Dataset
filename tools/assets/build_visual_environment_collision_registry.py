#!/usr/bin/env python3
"""Promote admitted environments in the unified asset registry."""

from __future__ import annotations

import argparse
import collections
import copy
from pathlib import Path
from typing import Any

from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json


def counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "by_asset_role": dict(
            sorted(collections.Counter(str(r["asset_role"]) for r in records).items())
        ),
        "by_proxy_kind": dict(
            sorted(
                collections.Counter(str(r["proxy"]["kind"]) for r in records).items()
            )
        ),
        "sampling_enabled": sum(
            bool(r["admission"].get("sampling_enabled", False)) for r in records
        ),
        "total": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-registry", type=Path, required=True)
    parser.add_argument("--collision-proxies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", default="physweep_asset_proxy_registry_v7")
    args = parser.parse_args()

    registry = copy.deepcopy(load_json(args.base_registry))
    proxy_document = load_json(args.collision_proxies)
    proxies = {str(record["asset_id"]): record for record in proxy_document["records"]}
    records = {str(record["asset_id"]): record for record in registry["records"]}
    if not set(proxies) <= set(records):
        raise ValueError("environment collision proxy is absent from the registry")
    registry["version"] = str(args.version)
    registry["base_registry"] = str(args.base_registry)
    registry["policy"].update(
        {
            "visual_mesh_never_becomes_collision_geometry": True,
            "every_admitted_environment_has_a_separate_static_proxy": True,
            "environment_proxy_is_always_loaded": True,
        }
    )
    for asset_id, proxy_record in proxies.items():
        record = records[asset_id]
        if record["asset_role"] != "render_only_context":
            raise ValueError(f"environment has an unexpected prior role: {asset_id}")
        proxy = proxy_record["proxy"]
        record["asset_role"] = "static_environment"
        record["admission"].update(
            {
                "status": "physics_ready",
                "sampling_enabled": False,
                "visual_sampling_enabled": True,
            }
        )
        record["proxy"] = {
            "kind": "static_environment_mesh",
            "colliders": [
                {
                    "id": "environment_mesh",
                    "shape": "static_concave_mesh",
                    "mesh_path": str(proxy["path"]),
                    "mesh_sha256": str(proxy["sha256"]),
                    "world_pose": "frozen_per_scene_metadata",
                }
            ],
        }
        record["review"]["physics_status"] = "static_concave_proxy_validated"
    registry["counts"] = counts(registry["records"])
    write_json(args.output, registry)
    print(f"promoted {len(proxies)}")


if __name__ == "__main__":
    main()
