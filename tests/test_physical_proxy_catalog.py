from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from tools.assets.physical_proxy_catalog import load_catalog, records_by_id  # noqa: E402
from tools.sampling.sample_pybullet_base import BUNDLE_PATH  # noqa: E402
from tools.assets.static_support_proxy import compile_static_support_binding  # noqa: E402


class PhysicalProxyCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
        cls.manifest, cls.records = load_catalog(
            ROOT,
            ROOT / bundle["physical_proxy_catalog"],
            require_runtime_validation=True,
        )
        cls.by_id = records_by_id(cls.records)
        cls.policy = json.loads(
            (ROOT / cls.manifest["policy_path"]).read_text(
                encoding="utf-8"
            )
        )

    def test_every_generated_physassets_proxy_is_catalogued(self) -> None:
        source_summary = json.loads(
            (
                ROOT
                / "external/physassets/generated_proxies/current/index_summary.json"
            ).read_text(encoding="utf-8")
        )
        records = [
            record
            for record in self.records
            if record["source"]["collection"] == "physassets"
        ]
        self.assertEqual(len(records), int(source_summary["records"]))
        self.assertEqual(
            sum(record["admission"]["proxy_ready"] for record in records),
            int(source_summary["admission_counts"]["passed"]),
        )

    def test_every_local_sketchfab_glb_has_one_disposition(self) -> None:
        glbs = list((ROOT / "assets/library/sketchfab").glob("**/model.glb"))
        records = [
            record
            for record in self.records
            if record["source"]["collection"] == "sketchfab"
        ]
        self.assertEqual(len(records), len(glbs))
        self.assertEqual(
            {record["asset_id"] for record in records},
            {path.parent.name for path in glbs},
        )

    def test_current_physassets_selection_is_catalog_backed(self) -> None:
        profiles = json.loads(
            (ROOT / "configs/physassets_core_object_profiles.json").read_text(
                encoding="utf-8"
            )
        )
        expected = {
            visual["asset_id"]
            for profile in profiles["profiles"]
            for visual in profile["visual_variants"]
        }
        actual = {
            record["asset_id"]
            for record in self.records
            if record["source"]["collection"] == "physassets"
            and record["admission"]["active_matrix_selected"]
        }
        self.assertEqual(actual, expected)
        for asset_id in expected:
            self.assertTrue(self.by_id[asset_id]["admission"]["sampling_ready"])
            self.assertEqual(self.by_id[asset_id]["qa"]["grade"], "A")

    def test_static_triangle_mesh_is_never_dynamic(self) -> None:
        static_meshes = [
            record
            for record in self.records
            if record["proxy"]["representation"] == "static_concave_mesh"
        ]
        self.assertTrue(static_meshes)
        for record in static_meshes:
            self.assertEqual(record["classification"]["body_type"], "static")
            self.assertIn("static_collision", record["capabilities"])
            self.assertTrue(record["admission"]["sampling_ready"])

    def test_static_supports_use_blender_evaluated_meshes(self) -> None:
        static_meshes = [
            record
            for record in self.records
            if record["proxy"]["representation"] == "static_concave_mesh"
        ]
        for record in static_meshes:
            self.assertEqual(
                record["proxy"]["method"],
                "blender_evaluated_exact_triangle_mesh",
            )
            extraction = record["qa"]["geometry"]["extraction"]
            self.assertEqual(
                extraction["method"], "blender_evaluated_reference_frame"
            )
            self.assertTrue((ROOT / extraction["sidecar_path"]).is_file())

    def test_static_support_physical_validation_is_complete(self) -> None:
        physical = json.loads(
            (ROOT / self.manifest["validation"]["path"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(physical["counts"]["failed"], 0)
        self.assertEqual(physical["catalog_records_sha256"], self.manifest["records_sha256"])
        self.assertNotIn("visual_validation", self.manifest)

    def test_kitchen_sink_geometry_is_preserved_by_proxy_policy(self) -> None:
        kitchen = self.by_id[
            "sketchfab_bg_e46deaab889548948a31e4264de61e5a"
        ]
        self.assertEqual(
            kitchen["proxy"]["representation"], "static_concave_mesh"
        )
        self.assertIn("holes_preserved", kitchen["capabilities"])
        self.assertEqual(
            kitchen["proxy"]["mesh"]["scale_binding"]["mode"],
            "support_surface_frame_to_metadata_v1",
        )

    def test_pool_collision_mesh_excludes_balls_and_cues(self) -> None:
        pool = self.by_id[
            "sketchfab_bg_0f2ae181a2dd4b00a6ec25073692037f"
        ]
        names = set(pool["qa"]["geometry"]["selected_node_names"])
        self.assertEqual(
            names,
            {
                "Pool Table_02 - Default_0",
                "Pool Table_03 - Default_0",
                "Pool Table_07 - Default_0",
                "Pool Table_08 - Default_0",
            },
        )
        self.assertFalse(
            any(name.startswith(("Ball", "Cue stick")) for name in names)
        )

    def test_every_static_support_usage_compiles_to_one_exact_transform(self) -> None:
        usages = 0
        for record in self.records:
            if record["proxy"]["representation"] != "static_concave_mesh":
                continue
            for usage in record["proxy"]["usages"]:
                binding = compile_static_support_binding(
                    record,
                    target_size_xy_m=usage["target_size_xy_m"],
                    target_center_xy_m=usage["target_center_xy_m"],
                    target_support_plane_z_m=usage[
                        "target_support_plane_z_m"
                    ],
                    usage_id=usage["id"],
                    maximum_axis_scale_ratio=usage[
                        "maximum_axis_scale_ratio"
                    ],
                )
                source = binding["source_support_frame"]
                target = binding["target_support_frame"]
                scale = binding["mesh"]["scale"]
                position = binding["mesh"]["base_position_m"]
                for axis in range(2):
                    mapped_center = (
                        float(source["center_xy"][axis]) * float(scale[axis])
                        + float(position[axis])
                    )
                    self.assertAlmostEqual(
                        mapped_center, float(target["center_xy_m"][axis]), places=8
                    )
                mapped_plane = (
                    float(source["plane_z"]) * float(scale[2])
                    + float(position[2])
                )
                self.assertAlmostEqual(
                    mapped_plane, float(target["plane_z_m"]), places=8
                )
                self.assertEqual(len(binding["binding_sha256"]), 64)
                self.assertEqual(
                    binding["visual"]["path"], record["source"]["visual_path"]
                )
                self.assertEqual(
                    binding["usage_contract"]["source"], usage["source"]
                )
                usages += 1
        self.assertGreaterEqual(usages, 9)

    def test_none_proxies_cannot_enter_sampling(self) -> None:
        for record in self.records:
            if record["proxy"]["representation"] != "none":
                continue
            self.assertFalse(record["admission"]["proxy_ready"])
            self.assertFalse(record["admission"]["sampling_ready"])

    def test_rules_are_declared_capability_driven(self) -> None:
        policy = self.policy["policy"]
        self.assertTrue(policy["asset_ids_are_forbidden_in_motion_code"])
        self.assertTrue(policy["rules_dispatch_on_capabilities_not_asset_ids"])
        self.assertTrue(policy["concave_triangle_mesh_is_static_only"])
        self.assertTrue(policy["coacd_is_reserved_for_complex_dynamic_assets"])
        self.assertTrue(
            policy["static_support_requires_visual_collision_surface_parity"]
        )
        self.assertTrue(policy["trimesh_static_extraction_is_provisional_only"])


if __name__ == "__main__":
    unittest.main()
