#!/usr/bin/env python3
"""Attach reviewed static collision proxies to visual-environment profiles."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--collision-proxies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--version", default="physweep_scene_mesh_profiles_v7"
    )
    args = parser.parse_args()

    document = load_json(args.profiles)
    proxy_document = load_json(args.collision_proxies)
    proxies = {
        str(record["asset_id"]): record for record in proxy_document["records"]
    }
    if len(proxies) != len(proxy_document["records"]):
        raise ValueError("duplicate visual-environment collision proxy")
    profile_assets = {
        str(profile["asset"]["asset_id"]) for profile in document["profiles"]
    }
    if set(proxies) != profile_assets:
        raise ValueError("collision proxy set does not match admitted environments")

    output = copy.deepcopy(document)
    output["version"] = str(args.version)
    output["policy"].pop("visual_only", None)
    output["policy"].pop("never_changes_collision_or_trajectory", None)
    output["policy"].update(
        {
            "visual_and_collision_are_paired": True,
            "collision_authority": "frozen_static_environment_proxy",
            "environment_proxy_is_always_loaded": True,
            "visual_and_collision_world_pose_is_identical": True,
            "raw_visual_glb_is_never_used_as_runtime_collision": True,
            "global_environment_floor_remains_authoritative": True,
        }
    )
    output["sources"]["collision_proxies"] = str(args.collision_proxies)
    for profile in output["profiles"]:
        asset = profile["asset"]
        record = proxies[str(asset["asset_id"])]
        if str(record["profile_id"]) != str(profile["id"]):
            raise ValueError(f"proxy/profile mismatch: {profile['id']}")
        if str(record["source"]["visual_sha256"]) != str(asset["sha256"]):
            raise ValueError(f"proxy source hash mismatch: {profile['id']}")
        proxy = record["proxy"]
        asset["collision_proxy"] = {
            "schema_version": str(record["schema_version"]),
            "representation": str(proxy["representation"]),
            "method": str(proxy["method"]),
            "path": str(proxy["path"]),
            "sha256": str(proxy["sha256"]),
            "vertex_count": int(proxy["vertex_count"]),
            "face_count": int(proxy["face_count"]),
            "bounds_min_m": [float(value) for value in proxy["bounds_min_m"]],
            "bounds_max_m": [float(value) for value in proxy["bounds_max_m"]],
            "extents_m": [float(value) for value in proxy["extents_m"]],
            "flags": [str(value) for value in proxy["flags"]],
            "transform_contract": copy.deepcopy(record["transform_contract"]),
        }
    write_json(args.output, output)
    print(f"profiles {len(output['profiles'])}")


if __name__ == "__main__":
    main()
