import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.derive_physics_sweep import derive_one, load_json, _sweep_values


class PhysicsSweepTests(unittest.TestCase):
    def test_one_factor_derivation_preserves_base(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs/physics_sweep.json"
        config = load_json(config_path)
        base = {
            "schema_version": "physweep_pybullet_rigid_metadata_v1",
            "scene_id": "base_scene",
            "dataset_stage": "one_object_base_candidate",
            "simulation": {
                "objects": [
                    {
                        "object_id": "object_a",
                        "body_model": "rigid_body",
                        "semantic_type": "unknown_object",
                        "material": {
                            "mass_kg": 1.0,
                            "contact_friction": 0.4,
                            "contact_restitution": 0.2,
                        },
                        "initial_state": {"linear_velocity_m_s": [1.0, 0.0, 0.0]},
                    }
                ]
            },
            "camera_request": {"fixed": True},
            "render": {"resolution": [1280, 720]},
        }
        original = copy.deepcopy(base)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            base_path = Path(temp) / "base.json"
            base_path.write_text(json.dumps(base), encoding="utf-8")
            derived = derive_one(
                base,
                base_path,
                root,
                config,
                config_path,
                "contact_friction",
                0,
                {},
                {},
            )
            self.assertEqual(derived["sweep"]["axis"], "contact_friction")
        self.assertEqual(base, original)

    def test_multi_object_binding_changes_only_target_object(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs/physics_sweep.json"
        config = load_json(config_path)
        base = {
            "schema_version": "physweep_pybullet_rigid_metadata_v1",
            "scene_id": "two_object_scene",
            "dataset_stage": "two_object_base_candidate",
            "simulation": {
                "objects": [
                    {
                        "object_id": "obj_0",
                        "body_model": "rigid_body",
                        "material": {
                            "mass_kg": 1.0,
                            "contact_friction": 0.4,
                            "contact_restitution": 0.2,
                        },
                        "initial_state": {},
                    },
                    {
                        "object_id": "obj_1",
                        "body_model": "rigid_body",
                        "material": {
                            "mass_kg": 2.0,
                            "contact_friction": 0.6,
                            "contact_restitution": 0.3,
                        },
                        "initial_state": {},
                    },
                ]
            },
        }
        with tempfile.TemporaryDirectory(dir=root) as temp:
            base_path = Path(temp) / "base.json"
            base_path.write_text(json.dumps(base), encoding="utf-8")
            derived = derive_one(
                base,
                base_path,
                root,
                config,
                config_path,
                "mass_kg",
                0,
                {},
                {},
                target_object_index=1,
            )

        self.assertEqual(derived["sweep"]["target_object_id"], "obj_1")
        self.assertEqual(derived["sweep"]["target_object_index"], 1)
        self.assertEqual(
            derived["simulation"]["objects"][0]["material"],
            base["simulation"]["objects"][0]["material"],
        )
        self.assertNotEqual(
            derived["simulation"]["objects"][1]["material"]["mass_kg"],
            base["simulation"]["objects"][1]["material"]["mass_kg"],
        )
        resolved = derived["sweep"]["resolved_object_physics"]
        self.assertEqual([item["object_id"] for item in resolved], ["obj_0", "obj_1"])

    def test_each_supported_axis_changes_only_its_runtime_field(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs/physics_sweep.json"
        config = load_json(config_path)
        base = {
            "schema_version": "physweep_pybullet_rigid_metadata_v1",
            "scene_id": "base_scene",
            "dataset_stage": "one_object_base_candidate",
            "simulation": {
                "objects": [
                    {
                        "object_id": "object_a",
                        "body_model": "rigid_body",
                        "semantic_type": "unknown_object",
                        "material": {
                            "mass_kg": 1.0,
                            "contact_friction": 0.4,
                            "contact_restitution": 0.2,
                        },
                        "initial_state": {"linear_velocity_m_s": [1.0, 0.0, 0.0]},
                    }
                ]
            },
        }
        with tempfile.TemporaryDirectory(dir=root) as temp:
            base_path = Path(temp) / "base.json"
            base_path.write_text(json.dumps(base), encoding="utf-8")
            for axis in config["axes"]:
                derived = derive_one(
                    base,
                    base_path,
                    root,
                    config,
                    config_path,
                    axis,
                    0,
                    {},
                    {},
                )
                self.assertEqual(derived["sweep"]["axis"], axis)
                self.assertEqual(
                    derived["simulation"]["objects"][0]["initial_state"],
                    base["simulation"]["objects"][0]["initial_state"],
                )
                for field in config["axes"]:
                    actual = derived["simulation"]["objects"][0]["material"][field]
                    expected = base["simulation"]["objects"][0]["material"][field]
                    if field == axis:
                        self.assertNotEqual(actual, expected)
                    else:
                        self.assertEqual(actual, expected)

    def test_ranges_are_resolved_from_the_base_value(self):
        root = Path(__file__).resolve().parents[1]
        config = load_json(root / "configs/physics_sweep.json")
        base_values = {
            "mass_kg": 1.0,
            "contact_friction": 0.4,
            "contact_restitution": 0.2,
        }
        expected_ranges = {
            "mass_kg": (0.5, 2.0),
            "contact_friction": (0.1, 1.0),
            "contact_restitution": (0.0, 0.8),
        }
        for axis, base_value in base_values.items():
            values = _sweep_values(
                base_value,
                config["axes"][axis],
                None,
                axis,
            )
            low, high = expected_ranges[axis]
            self.assertAlmostEqual(min(values), low, places=6)
            self.assertAlmostEqual(max(values), high, places=6)
            self.assertIn(base_value, values)

    def test_middle_policy_keeps_base_as_third_level(self):
        axis_rules = {
            "level_count": 5,
            "level_positions": [0.0, 0.25, 0.5, 0.75, 1.0],
            "range_policy": {
                "mode": "relative_multipliers",
                "lower_multiplier": 0.25,
                "upper_multiplier": 3.5,
            },
            "domain": [0.02, 1.0],
            "scale": "linear",
        }
        values = _sweep_values(
            0.4,
            axis_rules,
            None,
            "contact_friction",
            endpoint_policy={
                "level_count": 5,
                "normalized_positions": [0.0, 0.25, 0.5, 0.75, 1.0],
                "base_value_policy": "preserve_exactly_at_middle_position",
                "edge_policy": "reject_if_middle_impossible",
            },
        )
        self.assertEqual(len(values), 5)
        self.assertEqual(values[2], 0.4)
        self.assertEqual(values, [0.1, 0.25, 0.4, 0.7, 1.0])

    def test_middle_policy_expands_high_friction_domain_instead_of_shifting_base(self):
        axis_rules = {
            "level_count": 5,
            "level_positions": [0.0, 0.25, 0.5, 0.75, 1.0],
            "range_policy": {
                "mode": "relative_multipliers",
                "lower_multiplier": 0.25,
            "upper_multiplier": 3.5,
            },
            "domain": [0.02, 1.0],
            "scale": "linear",
        }
        values = _sweep_values(
            0.68,
            axis_rules,
            None,
            "contact_friction",
            endpoint_policy={
                "level_count": 5,
                "normalized_positions": [0.0, 0.25, 0.5, 0.75, 1.0],
                "base_value_policy": "preserve_exactly_at_middle_position",
                "edge_policy": "reject_if_middle_impossible",
            },
        )
        self.assertEqual(values[2], 0.68)
        self.assertEqual(values, [0.17, 0.425, 0.68, 0.84, 1.0])


if __name__ == "__main__":
    unittest.main()
