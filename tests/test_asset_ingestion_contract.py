from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from audit_asset_ingestion import (  # noqa: E402
    CONTRACT_PATH,
    audit_active_references,
    audit_contract,
)
from publish_asset_catalog import POLICY_PATH, build_manifest  # noqa: E402


class AssetIngestionContractTests(unittest.TestCase):
    def test_asset_library_manifests_have_canonical_locations(self) -> None:
        library = json.loads((ROOT / "assets/ASSET_LIBRARY.json").read_text(encoding="utf-8"))
        for relative_path in library["manifests"].values():
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)

        relative_path = library["manifests"]["foreground"]
        manifest = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
        self.assertNotIn("source_manifest", manifest)
        self.assertFalse(
            (ROOT / "assets/foreground_curation/real_object_admission_v1.json").exists()
        )

        material_curation = json.loads(
            (ROOT / library["manifests"]["material_curation"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("source_raw_manifest", material_curation)

    def test_active_asset_release_is_complete_and_sampling_ready(self) -> None:
        result = audit_contract(ROOT, ROOT / CONTRACT_PATH)
        self.assertEqual(result["status"], "clean")
        self.assertEqual(result["local_sketchfab_assets"], 122)
        self.assertEqual(result["foreground_profiles"], 84)
        self.assertEqual(result["enabled_registry_assets"], 46)

    def test_catalog_can_be_republished_from_current_release_inputs(self) -> None:
        manifest = build_manifest(ROOT, ROOT / POLICY_PATH)
        self.assertEqual(manifest["counts"]["total"], 2920)
        self.assertEqual(manifest["validation"]["counts"]["failed"], 0)
        self.assertNotIn("source_catalog", manifest)

    def test_active_tree_has_no_deleted_history_references(self) -> None:
        contract = json.loads((ROOT / CONTRACT_PATH).read_text(encoding="utf-8"))
        excluded = {
            ROOT / value["path"]
            for value in contract["immutable_provenance"].values()
        }
        audit_active_references(ROOT, excluded)


if __name__ == "__main__":
    unittest.main()
