import copy
import unittest
from pathlib import Path

from tools.sampling.three_object_rule_matrix import (
    load_and_validate_rule_matrix,
    validate_rule_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


def family(matrix, family_id):
    return next(row for row in matrix["families"] if row["id"] == family_id)


def subcase(matrix, subcase_id):
    return next(
        row for row in family(matrix, "generic")["subcases"]
        if row["id"] == subcase_id
    )


def template(matrix, template_id):
    return next(
        row for row in subcase(matrix, "flat")["motion_templates"]
        if row["id"] == template_id
    )


class ThreeObjectRuleMatrixTests(unittest.TestCase):
    def test_compact_four_family_matrix_is_valid(self):
        matrix, report = load_and_validate_rule_matrix(ROOT)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["schema_version"], "physweep_three_object_rule_matrix_v3")
        self.assertEqual(report["family_count"], 4)
        self.assertEqual(report["generic"]["subcase_count"], 3)
        self.assertEqual(report["generic"]["motion_template_count"], 4)
        self.assertEqual(report["asset_scope"]["generic_visual_count"], 84)
        self.assertEqual(report["asset_scope"]["role_safe_nonball_count"], 10)
        self.assertEqual(report["asset_scope"]["exact_support_count"], 18)
        self.assertFalse(report["production_quota_present"])
        self.assertEqual([row["id"] for row in matrix["families"]], [
            "generic", "billiards", "pinball", "marble_run"
        ])

    def test_connected_template_cannot_be_widened_to_nonball_driver(self):
        matrix, _ = load_and_validate_rule_matrix(ROOT)
        changed = copy.deepcopy(matrix)
        chain = template(changed, "chain_transfer")
        chain["variants"][0]["shape_by_role"]["P"].append("cuboid")
        with self.assertRaisesRegex(ValueError, "shape capability"):
            validate_rule_matrix(ROOT, changed)

    def test_multi_mesh_pair_rotates_assets_but_limits_one_per_scene(self):
        matrix, _ = load_and_validate_rule_matrix(ROOT)
        pair = template(matrix, "pair_control")
        multi = next(row for row in pair["variants"] if row["id"] == "axis_aligned_multi_mesh")
        self.assertEqual(multi["true_asset_roles"], ["P", "Q", "R"])
        self.assertEqual(multi["maximum_true_assets_per_scene"], 1)
        changed = copy.deepcopy(matrix)
        changed_pair = template(changed, "pair_control")
        changed_multi = next(row for row in changed_pair["variants"] if row["id"] == "axis_aligned_multi_mesh")
        changed_multi["maximum_true_assets_per_scene"] = 2
        with self.assertRaisesRegex(ValueError, "multi-mesh pair"):
            validate_rule_matrix(ROOT, changed)

    def test_generic_camera_cannot_become_overhead_heavy(self):
        matrix, _ = load_and_validate_rule_matrix(ROOT)
        changed = copy.deepcopy(matrix)
        subcase(changed, "flat")["camera"]["view_families"][0]["elevation_degrees"] = 65
        with self.assertRaisesRegex(ValueError, "camera angle"):
            validate_rule_matrix(ROOT, changed)

    def test_camera_quota_counts_the_selected_frozen_family(self):
        matrix, _ = load_and_validate_rule_matrix(ROOT)
        self.assertEqual(
            matrix["common_sampling_contract"]["camera_diversity_accounting"],
            "count_selected_frozen_family_after_base_admission",
        )
        changed = copy.deepcopy(matrix)
        changed["common_sampling_contract"]["camera_diversity_accounting"] = "count_requested_family"
        with self.assertRaisesRegex(ValueError, "common production contract"):
            validate_rule_matrix(ROOT, changed)

    def test_inclined_subcase_cannot_inherit_crossing_motion(self):
        matrix, _ = load_and_validate_rule_matrix(ROOT)
        changed = copy.deepcopy(matrix)
        inclined = subcase(changed, "inclined")
        inclined["motion_templates"][0]["variants"].append({
            "id": "crossing_control_left",
            "shape_by_role": {role: ["sphere"] for role in ("P", "Q", "R")},
        })
        with self.assertRaisesRegex(ValueError, "inclined variant"):
            validate_rule_matrix(ROOT, changed)

    def test_changed_asset_scope_reference_is_rejected(self):
        matrix, _ = load_and_validate_rule_matrix(ROOT)
        changed = copy.deepcopy(matrix)
        changed["asset_scope"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "changed or missing rule reference"):
            validate_rule_matrix(ROOT, changed)

    def test_special_fixture_capacity_cannot_be_widened(self):
        matrix, _ = load_and_validate_rule_matrix(ROOT)
        changed = copy.deepcopy(matrix)
        family(changed, "marble_run")["finite_initial_state_capacity"] = 25
        with self.assertRaisesRegex(ValueError, "finite initial-state capacity"):
            validate_rule_matrix(ROOT, changed)

    def test_quota_cannot_be_smuggled_into_rule_matrix(self):
        matrix, _ = load_and_validate_rule_matrix(ROOT)
        changed = copy.deepcopy(matrix)
        changed["base_count"] = 1538
        with self.assertRaisesRegex(ValueError, "production quota"):
            validate_rule_matrix(ROOT, changed)


if __name__ == "__main__":
    unittest.main()
