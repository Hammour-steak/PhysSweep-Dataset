"""Verify the portable collision assets shared by base and sweep releases."""

from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

from tools.release.base_release_schema import FIXTURE_SCHEMA, verified_file


def fixture_asset_bindings(value: Any, context: tuple[str, ...] = ()) -> list[tuple[str, str]]:
    bindings = []
    if isinstance(value, list):
        for item in value:
            bindings.extend(fixture_asset_bindings(item, context))
    elif isinstance(value, dict):
        for path_key, digest_key in (("path", "sha256"), ("mesh_path", "mesh_sha256")):
            path = value.get(path_key)
            collision = (
                path_key == "mesh_path"
                or any(part in {"mesh", "collision"} for part in context)
                or isinstance(path, str) and path.startswith("fixture_assets/")
            )
            if path_key in value and collision and "visual" not in context:
                digest = value.get(digest_key)
                if not isinstance(path, str) or not isinstance(digest, str):
                    raise ValueError("fixture collision asset binding is incomplete")
                bindings.append((path, digest))
        for key, item in value.items():
            bindings.extend(fixture_asset_bindings(item, context + (str(key),)))
    return bindings


def verify_fixture_catalog_files(output: Path, records: list[dict[str, Any]]) -> dict[str, int]:
    usage = {}
    assets = set()
    fixture_root, asset_root = output / "fixtures", output / "fixture_assets"
    if fixture_root.is_symlink() or asset_root.is_symlink():
        raise ValueError("fixture directories must be materialized")
    for record in records:
        if set(record) != {"sha256", "usage_count"} or int(record["usage_count"]) <= 0:
            raise ValueError("fixture catalog record differs")
        digest = str(record["sha256"])
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None or digest in usage:
            raise ValueError("invalid or duplicate fixture hash")
        path = verified_file(fixture_root / f"{digest}.json", digest, "fixture")
        if path.is_symlink():
            raise ValueError("fixture must be materialized")
        fixture = json.loads(path.read_text(encoding="utf-8"))
        if fixture.get("schema_version") != FIXTURE_SCHEMA:
            raise ValueError("fixture schema differs")
        for relative, asset_digest in fixture_asset_bindings(fixture):
            parts = PurePosixPath(relative).parts
            if (
                len(parts) != 2 or parts[0] != "fixture_assets"
                or "\\" in relative or relative != "/".join(parts)
                or re.fullmatch(r"[0-9a-f]{64}", asset_digest) is None
            ):
                raise ValueError("fixture asset path or hash differs")
            asset = output / relative
            if asset.is_symlink():
                raise ValueError("fixture asset must be materialized")
            verified_file(asset, asset_digest, "fixture collision asset")
            assets.add(parts[1])
        usage[digest] = int(record["usage_count"])
    if {path.name for path in fixture_root.iterdir()} != {
        "manifest.json", *(f"{digest}.json" for digest in usage)
    }:
        raise ValueError("unexpected fixture files")
    if {path.name for path in asset_root.iterdir()} != assets:
        raise ValueError("unexpected fixture assets")
    return usage
