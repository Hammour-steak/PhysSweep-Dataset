#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path


def glb_features(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        magic, version, _ = struct.unpack("<4sII", handle.read(12))
        if magic != b"glTF" or version != 2:
            raise ValueError(f"unsupported GLB: {path}")
        chunk_length, chunk_type = struct.unpack("<II", handle.read(8))
        if chunk_type != 0x4E4F534A:
            raise ValueError(f"first GLB chunk is not JSON: {path}")
        document = json.loads(handle.read(chunk_length).decode("utf-8"))
    return len(document.get("animations", [])), len(document.get("skins", []))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/asset_proxy_registry.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    registry_path = args.registry if args.registry.is_absolute() else root / args.registry
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_by_id = {record["asset_id"]: record for record in registry["records"]}
    assets: dict[str, str] = {}
    for record in manifest["records"]:
        if record["pipeline"] == "generic_pybullet":
            metadata = json.loads(
                (root / record["metadata_path"]).read_text(encoding="utf-8")
            )
            profile = metadata["simulation"]["objects"][0]["visual_profile"]
            if profile["type"] == "mesh":
                assets[str(profile["asset_id"])] = str(profile["path"])
        elif record["pipeline"] == "asset_proxy":
            asset_id = str(record["dynamic_asset_id"])
            visual = registry_by_id[asset_id]["visual"]
            assets[asset_id] = str(visual["path"])

    records = []
    for asset_id, relative_path in sorted(assets.items()):
        animations, skins = glb_features(root / relative_path)
        records.append(
            {
                "asset_id": asset_id,
                "path": relative_path,
                "animations": animations,
                "skins": skins,
            }
        )
    animated = [record for record in records if record["animations"] or record["skins"]]
    print(
        json.dumps(
            {
                "pipelines": dict(Counter(record["pipeline"] for record in manifest["records"])),
                "dynamic_glb_count": len(records),
                "animated_or_skinned_count": len(animated),
                "animated_or_skinned": animated,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
