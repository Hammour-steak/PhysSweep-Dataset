import copy
import unittest
from collections import Counter
from pathlib import Path

from tests.three_object_fixtures import source
from tools.sampling.three_object_coverage import build_plan, bound_object_pool
from tools.sampling.three_object_sampling_request import load_pilot_rules, validate_rules
from tools.sampling.three_object_sources import _source_family_schemas


ASSETS = [
    "sketchfab_4ae035ea89ea40bbaa82403b9c36afab",
    "sketchfab_640b0c8287274629a7f4ff3ce74a5999",
    "sketchfab_76fa80950b1f48a8ad6a8441fe443241",
    "sketchfab_b0902ff83cf946c984f9f7e0197ddb7f",
    "sketchfab_dae37cc4869a4155b6999efe26df710c",
    "sketchfab_ebb7c4102dc94ef7ba14d9c5df43c448",
    "sketchfab_f33ed1f09bef4d9bb757c25c1304ec96",
    "sketchfab_f4d11e48cc52479c802ec2cf0d629f68",
]


def fixture_pool():
    objects = []
    for index in range(3):
        row = source(index)
        row["source"]["source_family"] = "generic"
        row["metadata"]["semantic_sampling"]["five_dimensions"]["foreground_object"]["scale_bin"] = "medium"
        objects.append(row)
    variants = [
        ("cuboid", "large"), ("cylinder", "small"),
        ("cuboid", "large"), ("cuboid", "medium"),
        ("cuboid", "large"), ("cylinder", "large"),
        ("sphere", "small"), ("cuboid", "small"),
    ]
    for index, (asset, variant) in enumerate(zip(ASSETS, variants), 10):
        row = source(index)
        row["source"]["source_family"] = "asset"
        obj = row["metadata"]["simulation"]["objects"][0]
        obj["geometry"]["type"] = variant[0]
        obj["collision_profile"] = copy.deepcopy(obj["geometry"])
        obj["visual_profile"]["id"] = asset
        row["metadata"]["semantic_sampling"]["five_dimensions"]["foreground_object"]["scale_bin"] = variant[1]
        objects.append(row)
    hosts = []
    for host in ("ground_flat", "raised_flat"):
        for category in ("home_office", "lab_studio"):
            row = source()
            row["source"]["source_family"] = "generic"
            row["metadata"]["simulation"]["support"]["scene_class"] = host
            row["metadata"]["appearance"]["scene_visual"]["environment_category"] = category
            hosts.append(row)
    return {"objects": objects, "hosts": hosts, "summary": {}}


class PrimitiveAssetCoverageTests(unittest.TestCase):
    def test_frozen_assets_have_two_cells_and_actual_shape_scale(self):
        rules = load_pilot_rules(matrix_path=Path("three_object_d5g_sampling_matrix.json"))
        plan = build_plan(fixture_pool(), rules)
        self.assertEqual(plan["required_asset_ids"], ASSETS)
        counts = Counter(
            (cell["required_visual_asset_by_role"]["R"], cell["template"])
            for cell in plan["cells"]
        )
        self.assertEqual(len(counts), 16)
        self.assertEqual(set(counts.values()), {1})
        self.assertEqual(
            {cell["required_source_family_by_role"]["R"] for cell in plan["cells"]},
            {"asset"},
        )
        changed = fixture_pool()
        changed["objects"].reverse()
        changed["hosts"].reverse()
        self.assertEqual(plan, build_plan(changed, rules))

    def test_role_family_binding_prevents_cross_family_substitution(self):
        rows = fixture_pool()["objects"]
        pools = {("sphere", "medium"): rows[:3]}
        impostor = copy.deepcopy(rows[0])
        impostor["source"]["source_family"] = "asset"
        pools[("sphere", "medium")].append(impostor)
        cell = {
            "shape_by_role": {"P": "sphere"},
            "scale_by_role": {"P": "medium"},
            "required_source_family_by_role": {"P": "generic"},
        }
        self.assertEqual(bound_object_pool(pools, cell, "P"), rows[:3])
        cell["required_source_family_by_role"]["P"] = "typo"
        with self.assertRaisesRegex(ValueError, "source-family"):
            bound_object_pool(pools, cell, "P")

    def test_source_schema_mapping_and_scope_are_explicit(self):
        rules = load_pilot_rules(matrix_path=Path("three_object_d5g_sampling_matrix.json"))
        source_rules = rules["matrix"]["source"]
        self.assertEqual(
            _source_family_schemas(source_rules),
            {
                "physweep_pybullet_rigid_metadata_v1": "generic",
                "physweep_asset_proxy_scene_v3": "asset",
            },
        )
        changed = copy.deepcopy(rules["matrix"])
        changed["required_asset_ids"] = changed["required_asset_ids"][:-1]
        with self.assertRaisesRegex(ValueError, "eight explicit assets"):
            validate_rules(
                changed,
                rules["motion"],
                rules["scene"],
                rules["sweep"],
                rules["dataset"],
            )


if __name__ == "__main__":
    unittest.main()
