#!/usr/bin/env python3
"""Reachability checks between reviewed asset proxies and the scene matrix."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class AssetMatrixReachabilityTests(unittest.TestCase):
    def test_all_enabled_dynamic_assets_are_reachable(self) -> None:
        registry = load_json(PROJECT_ROOT / "configs/asset_proxy_registry.json")
        matrix = load_json(
            PROJECT_ROOT / "configs/one_object_sampling_matrix.json"
        )

        enabled_dynamic = {
            str(record["asset_id"])
            for record in registry["records"]
            if bool(record["admission"].get("sampling_enabled", False))
            and record["proxy"]["kind"] == "dynamic_rigid"
        }
        reachable_dynamic = set()
        for environment in matrix["environments"]:
            if environment["generator"] != "asset_proxy":
                continue
            reachable_dynamic.update(
                str(asset_id)
                for asset_id in environment.get("dynamic_asset_ids", [])
            )
            for values in environment.get("dynamic_pools", {}).values():
                reachable_dynamic.update(str(asset_id) for asset_id in values)

        self.assertEqual(reachable_dynamic, enabled_dynamic)

    def test_all_enabled_static_props_are_reachable(self) -> None:
        registry = load_json(PROJECT_ROOT / "configs/asset_proxy_registry.json")
        matrix = load_json(
            PROJECT_ROOT / "configs/one_object_sampling_matrix.json"
        )

        enabled_props = {
            str(record["asset_id"])
            for record in registry["records"]
            if bool(record["admission"].get("sampling_enabled", False))
            and record["proxy"]["kind"] == "static_compound"
        }
        reachable_props = {
            str(pair["static_prop_asset_id"])
            for environment in matrix["environments"]
            for pair in environment.get("support_prop_pairs", [])
        }

        self.assertEqual(reachable_props, enabled_props)


if __name__ == "__main__":
    unittest.main()
