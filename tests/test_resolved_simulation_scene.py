import copy
import unittest
from pathlib import Path

from tools.physics.resolved_simulation_scene import compile_resolved_scene, sha256


ROOT = Path(__file__).resolve().parents[1]


class ResolvedSimulationSceneTests(unittest.TestCase):
    def marble_run_metadata(self):
        backend = ROOT / "configs/marble_run_backend.json"
        return {
            "schema_version": "physweep_marble_run_scene_v1",
            "scene_id": "marble_run",
            "semantics": {"profile": "early_release_chain"},
            "simulation": {
                "time": {
                    "duration_s": 4.0,
                    "output_fps": 24,
                    "simulation_hz": 240,
                    "frame_count": 97,
                },
                "world": {"gravity_m_s2": [0.0, 0.0, -9.81]},
                "objects": [
                    {
                        "object_id": "marble",
                        "collision_proxy": {"type": "sphere", "radius_m": 0.011},
                        "material": {
                            "mass_kg": 0.120208,
                            "contact_friction": 0.16,
                            "contact_restitution": 0.55,
                        },
                        "initial_state": {
                            "position_m": [0.0, 0.0, 1.5],
                            "orientation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                            "linear_velocity_m_s": [0.0, 0.0, 0.0],
                            "angular_velocity_rad_s": [0.0, 0.0, 0.0],
                        },
                    }
                ],
            },
            "physics": {
                "backend_config": {
                    "path": str(backend.relative_to(ROOT)),
                    "sha256": sha256(backend),
                },
                "profile": "early_release_chain",
                "fixture": {"mesh_components": [], "analytic_colliders": []},
                "quality": {},
            },
            "object_identity": {
                "objects": [{"object_id": "marble", "role": "dynamic"}]
            },
        }

    def test_marble_run_adapter_remains_callable(self):
        scene = compile_resolved_scene(self.marble_run_metadata(), ROOT)
        self.assertEqual(scene["backend_binding"]["adapter_id"], "marble_run_v1")
        self.assertEqual([record["object_id"] for record in scene["objects"]], ["marble"])

    def test_marble_run_two_object_adapter_is_explicit(self):
        metadata = self.marble_run_metadata()
        second = copy.deepcopy(metadata["simulation"]["objects"][0])
        metadata["simulation"]["objects"][0]["object_id"] = "object_a"
        second["object_id"] = "object_b"
        second["initial_state"]["position_m"][0] = 0.08
        metadata["simulation"]["objects"].append(second)
        metadata["object_identity"]["objects"] = [
            {"object_id": "object_a", "role": "dynamic"},
            {"object_id": "object_b", "role": "dynamic"},
        ]
        metadata["physics"]["two_object_quality"] = {
            "maximum_first_pair_contact_time_s": 0.5,
            "maximum_penetration_m": 0.001,
        }
        scene = compile_resolved_scene(metadata, ROOT)
        self.assertEqual(
            scene["backend_binding"]["adapter_id"],
            "marble_run_two_object_v1",
        )
        self.assertEqual(scene["backend_binding"]["supported_dynamic_object_counts"], [2])

    def billiards_metadata(self):
        backend = ROOT / "configs/pybullet_backend.json"
        return {
            "schema_version": "physweep_billiards_scene_v4",
            "scene_id": "billiards",
            "physics": {
                "duration_s": 1.0,
                "output_fps": 24,
                "simulation_hz": 240,
                "frame_count": 25,
                "backend_config": {
                    "path": str(backend.relative_to(ROOT)),
                    "sha256": sha256(backend),
                },
                "ball_radius_m": 0.028575,
                "ball_mass_kg": 0.17,
                "profile": "single_ball_free_roll",
                "static_support_binding": {"support_asset_id": "pool_table"},
                "initial_states": [
                    {
                        "object_id": "ball_0",
                        "position_m": [0.0, 0.0, 1.0],
                        "velocity_m_s": [0.5, 0.0, 0.0],
                    }
                ],
            },
            "object_identity": {
                "objects": [{"object_id": "ball_0", "role": "dynamic"}]
            },
        }

    def test_billiards_adapter_remains_callable(self):
        scene = compile_resolved_scene(self.billiards_metadata(), ROOT)
        self.assertEqual(scene["backend_binding"]["adapter_id"], "billiards_v4")
        self.assertEqual([record["object_id"] for record in scene["objects"]], ["ball_0"])

    def test_billiards_two_object_adapter_is_explicit(self):
        metadata = self.billiards_metadata()
        second = copy.deepcopy(metadata["physics"]["initial_states"][0])
        second["object_id"] = "ball_1"
        second["position_m"][0] = 0.2
        metadata["physics"]["initial_states"].append(second)
        metadata["physics"]["two_object_quality"] = {
            "maximum_first_pair_contact_time_s": 1.0,
            "maximum_penetration_m": 0.001,
        }
        metadata["object_identity"]["objects"].append(
            {"object_id": "ball_1", "role": "dynamic"}
        )
        scene = compile_resolved_scene(metadata, ROOT)
        self.assertEqual(
            scene["backend_binding"]["adapter_id"],
            "billiards_two_object_v1",
        )
        self.assertEqual(scene["backend_binding"]["supported_dynamic_object_counts"], [2])

    def generic_metadata(self):
        return {
            "schema_version": "physweep_pybullet_rigid_metadata_v1",
            "scene_id": "generic",
            "simulation": {
                "time": {
                    "duration_s": 1.0,
                    "output_fps": 24,
                    "simulation_hz": 240,
                    "frame_count": 25,
                },
                "world": {"gravity_m_s2": [0.0, 0.0, -9.81]},
                "objects": [
                    {
                        "object_id": "object_a",
                        "geometry": {"type": "sphere", "size_m": [0.2, 0.2, 0.2]},
                        "material": {
                            "mass_kg": 1.0,
                            "contact_friction": 0.4,
                            "contact_restitution": 0.2,
                        },
                        "initial_state": {
                            "position_m": [0.0, 0.0, 1.0],
                            "orientation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            "linear_velocity_m_s": [0.0, 0.0, 0.0],
                            "angular_velocity_rad_s": [0.0, 0.0, 0.0],
                        },
                    }
                ],
            },
            "object_identity": {
                "objects": [{"object_id": "object_a", "role": "dynamic"}]
            },
        }

    def test_generic_base_compiles_without_target(self):
        scene = compile_resolved_scene(self.generic_metadata(), Path("."))
        self.assertEqual(scene["variant"]["kind"], "base")
        self.assertIsNone(scene["variant"]["target_object_id"])
        self.assertEqual(scene["objects"][0]["material"]["mass_kg"], 1.0)

    def test_resolved_sweep_material_is_authoritative(self):
        metadata = self.generic_metadata()
        metadata["sweep"] = {
            "kind": "sweep",
            "target_object_id": "object_a",
            "target_object_index": 0,
            "parameter": "mass_kg",
            "value": 2.0,
            "resolved_object_physics": [
                {
                    "object_id": "object_a",
                    "object_index": 0,
                    "material": {
                        "mass_kg": 2.0,
                        "contact_friction": 0.4,
                        "contact_restitution": 0.2,
                    },
                }
            ],
        }
        scene = compile_resolved_scene(metadata, Path("."))
        self.assertEqual(scene["objects"][0]["material"]["mass_kg"], 2.0)
        self.assertEqual(
            scene["objects"][0]["inertia_policy"],
            "pybullet_from_collision_proxy_and_mass",
        )

    def test_canonical_base_rejects_hidden_target(self):
        metadata = self.generic_metadata()
        metadata["sweep"] = {
            "kind": "base",
            "target_object_id": "object_a",
            "parameter": None,
            "value": None,
        }
        with self.assertRaisesRegex(ValueError, "must not bind"):
            compile_resolved_scene(metadata, Path("."))

    def test_canonical_base_rejects_hidden_axis(self):
        metadata = self.generic_metadata()
        metadata["sweep"] = {
            "kind": "base",
            "target_object_id": None,
            "target_object_index": None,
            "parameter": None,
            "axis": "mass_kg",
            "value": None,
            "level_index": None,
        }
        with self.assertRaisesRegex(ValueError, "must not bind"):
            compile_resolved_scene(metadata, Path("."))

    def test_input_metadata_is_not_mutated(self):
        metadata = self.generic_metadata()
        original = copy.deepcopy(metadata)
        compile_resolved_scene(metadata, Path("."))
        self.assertEqual(metadata, original)

    def test_resolved_materials_must_cover_every_object(self):
        metadata = self.generic_metadata()
        metadata["sweep"] = {
            "kind": "sweep",
            "target_object_id": "object_a",
            "target_object_index": 0,
            "parameter": "mass_kg",
            "value": 2.0,
            "resolved_object_physics": [],
        }
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            compile_resolved_scene(metadata, Path("."))

    def test_sweep_value_must_match_resolved_material(self):
        metadata = self.generic_metadata()
        metadata["sweep"] = {
            "kind": "sweep",
            "target_object_id": "object_a",
            "target_object_index": 0,
            "parameter": "mass_kg",
            "value": 2.0,
            "resolved_object_physics": [
                {
                    "object_id": "object_a",
                    "object_index": 0,
                    "material": {
                        "mass_kg": 3.0,
                        "contact_friction": 0.4,
                        "contact_restitution": 0.2,
                    },
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "differs from resolved"):
            compile_resolved_scene(metadata, Path("."))

    def test_resolved_object_index_must_match_order(self):
        metadata = self.generic_metadata()
        metadata["sweep"] = {
            "kind": "sweep",
            "target_object_id": "object_a",
            "target_object_index": 0,
            "parameter": "mass_kg",
            "value": 2.0,
            "resolved_object_physics": [
                {
                    "object_id": "object_a",
                    "object_index": 1,
                    "material": {
                        "mass_kg": 2.0,
                        "contact_friction": 0.4,
                        "contact_restitution": 0.2,
                    },
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "order and indices"):
            compile_resolved_scene(metadata, Path("."))

    def test_time_contract_is_checked_before_simulation(self):
        metadata = self.generic_metadata()
        metadata["simulation"]["time"]["frame_count"] = 24
        with self.assertRaisesRegex(ValueError, "frame count"):
            compile_resolved_scene(metadata, Path("."))

    def test_generic_adapter_declares_two_object_limit(self):
        metadata = self.generic_metadata()
        second = copy.deepcopy(metadata["simulation"]["objects"][0])
        second["object_id"] = "object_b"
        metadata["simulation"]["objects"].append(second)
        metadata["object_identity"]["objects"].append(
            {"object_id": "object_b", "role": "dynamic"}
        )
        scene = compile_resolved_scene(metadata, Path("."))
        self.assertEqual(
            scene["backend_binding"]["supported_dynamic_object_counts"],
            [1, 2],
        )
        self.assertEqual(
            [record["object_id"] for record in scene["objects"]],
            ["object_a", "object_b"],
        )

        third = copy.deepcopy(second)
        third["object_id"] = "object_c"
        metadata["simulation"]["objects"].append(third)
        metadata["object_identity"]["objects"].append(
            {"object_id": "object_c", "role": "dynamic"}
        )
        with self.assertRaisesRegex(ValueError, "does not support 3"):
            compile_resolved_scene(metadata, Path("."))

    def test_non_finite_material_is_rejected(self):
        metadata = self.generic_metadata()
        metadata["simulation"]["objects"][0]["material"]["mass_kg"] = float("nan")
        with self.assertRaisesRegex(ValueError, "must be finite"):
            compile_resolved_scene(metadata, Path("."))

    def test_non_unit_orientation_is_rejected(self):
        metadata = self.generic_metadata()
        metadata["simulation"]["objects"][0]["initial_state"][
            "orientation_quaternion_wxyz"
        ] = [2.0, 0.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "unit quaternion"):
            compile_resolved_scene(metadata, Path("."))


if __name__ == "__main__":
    unittest.main()
