import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.core.sweep_values import sweep_values
from tools.sampling.derive_physics_sweep import (
    collect_inputs,
    derive_one,
    load_json,
    resolve_prior_provenance,
    validate_output_dir,
    _friction_domain,
    _mass_bounds,
    normalize_canonical_base,
)


class PhysicsSweepTests(unittest.TestCase):
    def test_canonical_base_has_no_sweep_target(self):
        derived = {
            "scene_id": "scene__sweep_object_a_mass_kg_02",
            "dataset_stage": "object_physics_sweep_candidate",
            "semantic_sampling": {
                "five_dimensions": {
                    "motion": {"family": "roll_or_slide_1obj"}
                }
            },
            "sweep": {
                "schema_version": "physweep_object_bound_sweep_v2",
                "kind": "sweep",
                "mode": "one_factor",
                "parent_scene_id": "scene",
                "parent_metadata_path": "base/metadata.json",
                "parent_metadata_sha256": "abc",
                "target_object_id": "object_a",
                "target_object_index": 0,
                "parameter": "mass_kg",
                "value": 1.0,
                "base_value": 1.0,
                "source_schema_version": "physweep_pybullet_rigid_metadata_v1",
                "resolved_object_physics": [
                    {
                        "object_id": "object_a",
                        "object_index": 0,
                        "material": {
                            "mass_kg": 1.0,
                            "contact_friction": 0.4,
                            "contact_restitution": 0.2,
                        },
                    }
                ],
                "resolved_state_policy": "all_dynamic_objects_serialized_in_metadata",
                "config_path": "configs/physics_sweep.json",
                "config_sha256": "def",
                "initial_state_policy": "copied_from_base_unchanged",
                "visual_policy": "copied_from_base_unchanged",
            },
            "schema_version": "physweep_pybullet_rigid_metadata_v1",
            "simulation": {
                "objects": [
                    {
                        "object_id": "object_a",
                        "body_model": "rigid_body",
                        "semantic_type": "ball",
                        "material": {},
                    }
                ]
            },
        }
        canonical = normalize_canonical_base(derived)
        self.assertEqual(canonical["scene_id"], "scene__base")
        self.assertEqual(canonical["sweep"]["kind"], "base")
        self.assertIsNone(canonical["sweep"]["target_object_id"])
        self.assertIsNone(canonical["sweep"]["parameter"])
        self.assertNotIn("sweep_target", canonical["object_identity"])

    def test_mass_bounds_use_stable_asset_id_priority(self):
        metadata = {
            "schema_version": "physweep_pybullet_rigid_metadata_v1",
            "simulation": {
                "objects": [
                    {
                        "object_id": "object_a",
                        "asset_id": "asset_primary",
                        "semantic_type": "semantic_fallback",
                        "visual_profile": {
                            "asset_id": "asset_visual",
                            "id": "visual_fallback",
                        },
                    }
                ]
            },
        }
        profiles = {
            "asset_primary": {"physics": {"mass_range_kg": [1.0, 2.0]}},
            "asset_visual": {"physics": {"mass_range_kg": [3.0, 4.0]}},
            "semantic_fallback": {"physics": {"mass_range_kg": [5.0, 6.0]}},
        }
        self.assertEqual(_mass_bounds(metadata, profiles, {}, 0), [1.0, 2.0])

    def test_relevant_friction_motion_requires_analytic_contract(self):
        metadata = {
            "schema_version": "physweep_pybullet_rigid_metadata_v1",
            "simulation": {
                "objects": [
                    {
                        "expected_motion": {
                            "motion_family": "slide_push",
                            "minimum_displacement_m": 0.2,
                        },
                        "initial_state": {"linear_velocity_m_s": [1.0, 0.0, 0.0]},
                    }
                ]
            },
        }
        with self.assertRaisesRegex(ValueError, "analytic support-frame contract"):
            _friction_domain(
                metadata,
                {"domain": [0.02, 1.0], "transition_margin": 1.25},
                0.3,
                "contact_friction",
                0,
            )

    def test_prior_provenance_rejects_changed_frozen_dependency(self):
        root = Path(__file__).resolve().parents[1]
        config = {
            "prior_sources": {
                "registry": "configs/asset_proxy_registry.json"
            }
        }
        with self.assertRaisesRegex(ValueError, "dependency changed"):
            resolve_prior_provenance(
                root,
                config,
                {
                    "dependencies": {
                        "registry": {
                            "path": "configs/asset_proxy_registry.json",
                            "sha256": "0" * 64,
                        }
                    }
                },
            )

    def test_nonempty_output_is_rejected(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as temp:
            output = Path(temp) / "dataset" / "metadata"
            output.mkdir(parents=True)
            marker = output / "old.json"
            marker.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "clean output"):
                validate_output_dir(root, output)
            self.assertTrue(marker.exists())

    def test_manifest_input_uses_exact_declared_records(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as temp:
            temp_path = Path(temp)
            first = temp_path / "first" / "metadata.json"
            second = temp_path / "second" / "metadata.json"
            extra = temp_path / "extra" / "metadata.json"
            for path in (first, second, extra):
                path.parent.mkdir(parents=True)
                path.write_text("{}", encoding="utf-8")
            manifest = temp_path / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_count": 2,
                        "records": [
                            {"metadata_path": str(first.relative_to(root))},
                            {"metadata_path": str(second.relative_to(root))},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            inputs = collect_inputs(root, None, None, manifest)
        self.assertEqual(inputs, sorted([first.resolve(), second.resolve()]))

    def test_sampler_base_manifest_uses_declared_samples(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as temp:
            temp_path = Path(temp)
            metadata = temp_path / "scene" / "metadata.json"
            metadata.parent.mkdir(parents=True)
            metadata.write_text("{}", encoding="utf-8")
            manifest = temp_path / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "physweep_pybullet_base_manifest_v1",
                        "sample_count": 1,
                        "samples": [
                            {"metadata_path": str(metadata.relative_to(root))}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            inputs = collect_inputs(root, None, None, manifest)
        self.assertEqual(inputs, [metadata.resolve()])

    def test_unified_two_object_manifest_uses_declared_records(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=root) as temp:
            temp_path = Path(temp)
            metadata = temp_path / "scene" / "metadata.json"
            metadata.parent.mkdir(parents=True)
            metadata.write_text("{}", encoding="utf-8")
            manifest = temp_path / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "physweep_two_object_base_manifest_v1",
                        "sample_count": 1,
                        "records": [
                            {"metadata_path": str(metadata.relative_to(root))}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            inputs = collect_inputs(root, None, None, manifest)
        self.assertEqual(inputs, [metadata.resolve()])

    def test_manifest_input_cannot_be_mixed_with_directory_scanning(self):
        root = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(ValueError, "not both"):
            collect_inputs(
                root,
                None,
                Path("datasets/one_object_base"),
                Path("datasets/one_object_base/manifest.json"),
            )

    def test_asset_proxy_schema_resolves_registry_material(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs/physics_sweep.json"
        config = load_json(config_path)
        base = {
            "schema_version": "physweep_asset_proxy_scene_v3",
            "scene_id": "asset_scene",
            "dynamic_asset_name": "Bottle",
            "assets": {"dynamic_asset_id": "asset_a"},
            "physics": {
                "motion_profile": "vertical_drop",
                "mass_kg": 0.5,
                "trajectory_path": "base/trajectory.npz",
                "audit_path": "base/audit.json",
            },
            "render": {"video_path": "base/video.mp4"},
            "object_identity": {
                "objects": [
                    {
                        "object_id": "object_a",
                        "object_index": 0,
                        "role": "dynamic",
                        "asset_id": "asset_a",
                    }
                ]
            },
        }
        registry = {
            "asset_a": {
                "proxy": {
                    "mass_range_kg": [0.25, 1.0],
                    "material": {"friction": 0.3, "restitution": 0.12},
                }
            }
        }
        with tempfile.TemporaryDirectory(dir=root) as temp:
            base_path = Path(temp) / "metadata.json"
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
                registry,
                target_object_index=0,
            )
        self.assertEqual(
            derived["sweep"]["source_schema_version"],
            "physweep_asset_proxy_scene_v3",
        )
        self.assertEqual(derived["sweep"]["base_value"], 0.3)
        self.assertEqual(derived["physics"]["runtime_material"]["mass_kg"], 0.5)
        self.assertEqual(
            derived["physics"]["runtime_material"]["contact_restitution"],
            0.12,
        )
        self.assertNotEqual(
            derived["physics"]["runtime_material"]["contact_friction"], 0.3
        )
        self.assertNotIn("trajectory_path", derived["physics"])
        self.assertNotIn("audit_path", derived["physics"])
        self.assertNotIn("video_path", derived["render"])

    def test_billiards_uses_its_reviewed_restitution_domain(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs/physics_sweep.json"
        config = load_json(config_path)
        base = {
            "schema_version": "physweep_billiards_scene_v4",
            "scene_id": "billiards_scene",
            "semantics": {"profile": "single_ball_free_roll"},
            "physics": {
                "ball_mass_kg": 0.17,
                "backend_config": {"path": "configs/pybullet_backend.json"},
                "initial_states": [
                    {
                        "object_id": "cue_ball",
                        "position_m": [0.0, 0.0, 0.8],
                        "velocity_m_s": [0.4, 0.0, 0.0],
                    }
                ],
            },
            "object_identity": {
                "objects": [
                    {
                        "object_id": "cue_ball",
                        "object_index": 0,
                        "role": "dynamic",
                    }
                ]
            },
        }
        with tempfile.TemporaryDirectory(dir=root) as temp:
            base_path = Path(temp) / "metadata.json"
            base_path.write_text(json.dumps(base), encoding="utf-8")
            derived = derive_one(
                base,
                base_path,
                root,
                config,
                config_path,
                "contact_restitution",
                4,
                {},
                {},
                target_object_index=0,
            )
        self.assertEqual(derived["sweep"]["base_value"], 0.92)
        self.assertEqual(derived["sweep"]["allowed_domain"], [0.3, 1.0])
        self.assertEqual(
            derived["physics"]["runtime_material"]["contact_restitution"], 1.0
        )

    def test_single_ball_billiards_derives_exactly_13_one_factor_records(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs/physics_sweep.json"
        config = load_json(config_path)
        base = {
            "schema_version": "physweep_billiards_scene_v4",
            "scene_id": "billiards_single_ball",
            "semantics": {"profile": "single_ball_free_roll"},
            "physics": {
                "ball_mass_kg": 0.17,
                "backend_config": {"path": "configs/pybullet_backend.json"},
                "initial_states": [
                    {
                        "object_id": "cue_ball",
                        "position_m": [-0.58, -0.16, 0.809575],
                        "velocity_m_s": [0.48, 0.08, 0.0],
                    }
                ],
            },
            "render": {
                "resolution": [640, 360],
                "camera": {"seed": 123, "mode": "bounded_orbit_-15deg"},
            },
            "object_identity": {
                "objects": [
                    {
                        "object_id": "cue_ball",
                        "object_index": 0,
                        "role": "dynamic",
                    }
                ]
            },
        }
        base_material = {
            "mass_kg": 0.17,
            "contact_friction": 0.16,
            "contact_restitution": 0.92,
        }
        records = []
        canonical_emitted = False
        with tempfile.TemporaryDirectory(dir=root) as temp:
            base_path = Path(temp) / "metadata.json"
            base_path.write_text(json.dumps(base), encoding="utf-8")
            for axis in config["axes"]:
                for level_index in range(config["axes"][axis]["level_count"]):
                    derived = derive_one(
                        base,
                        base_path,
                        root,
                        config,
                        config_path,
                        axis,
                        level_index,
                        {},
                        {},
                        target_object_index=0,
                    )
                    if level_index == derived["sweep"]["base_level_index"]:
                        if axis != config["canonical_base_axis"]:
                            continue
                        derived = normalize_canonical_base(derived)
                        canonical_emitted = True
                    records.append(derived)

        self.assertTrue(canonical_emitted)
        self.assertEqual(len(records), 13)
        self.assertEqual(
            [record["sweep"]["kind"] for record in records].count("base"), 1
        )
        for axis in config["axes"]:
            axis_records = [
                record
                for record in records
                if record["sweep"].get("axis") == axis
            ]
            self.assertEqual(len(axis_records), 4)
            self.assertEqual(
                {record["sweep"]["level_index"] for record in axis_records},
                {0, 1, 3, 4},
            )
            for record in axis_records:
                expected_material = dict(base_material)
                expected_material[axis] = record["sweep"]["value"]
                self.assertEqual(
                    record["physics"]["runtime_material"], expected_material
                )
                self.assertEqual(
                    record["physics"]["initial_states"],
                    base["physics"]["initial_states"],
                )
                self.assertEqual(record["render"], base["render"])

    def test_marble_run_derives_exactly_13_one_factor_records(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs/physics_sweep.json"
        config = load_json(config_path)
        material = {
            "mass_kg": 0.120208,
            "mass_range_kg": [0.085, 0.34],
            "contact_friction": 0.16,
            "contact_restitution": 0.55,
        }
        initial_state = {
            "position_m": [0.0, 0.0, 1.5],
            "orientation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "linear_velocity_m_s": [0.0, 0.0, 0.0],
            "angular_velocity_rad_s": [0.0, 0.0, 0.0],
        }
        base = {
            "schema_version": "physweep_marble_run_scene_v1",
            "scene_id": "marble_run",
            "semantics": {"profile": "early_release_chain"},
            "simulation": {
                "objects": [
                    {
                        "object_id": "marble",
                        "body_model": "rigid_body",
                        "material": material,
                        "initial_state": initial_state,
                    }
                ]
            },
            "physics": {"trajectory_path": "parent/trajectory.npz"},
            "render": {"video_path": "parent/video.mp4"},
            "object_identity": {
                "objects": [{"object_id": "marble", "role": "dynamic"}]
            },
        }
        records = []
        with tempfile.TemporaryDirectory(dir=root) as temp:
            base_path = Path(temp) / "metadata.json"
            base_path.write_text(json.dumps(base), encoding="utf-8")
            for axis in config["axes"]:
                for level_index in range(config["axes"][axis]["level_count"]):
                    derived = derive_one(
                        base,
                        base_path,
                        root,
                        config,
                        config_path,
                        axis,
                        level_index,
                        {},
                        {},
                        target_object_index=0,
                    )
                    if level_index == derived["sweep"]["base_level_index"]:
                        if axis != config["canonical_base_axis"]:
                            continue
                        derived = normalize_canonical_base(derived)
                    records.append(derived)

        self.assertEqual(len(records), 13)
        self.assertEqual(
            [record["sweep"]["kind"] for record in records].count("base"), 1
        )
        for record in records:
            self.assertEqual(
                record["simulation"]["objects"][0]["initial_state"], initial_state
            )
            self.assertNotIn("trajectory_path", record["physics"])
            self.assertNotIn("video_path", record["render"])
            if record["sweep"]["kind"] == "sweep":
                changed = record["sweep"]["axis"]
                expected = dict(material)
                expected[changed] = record["sweep"]["value"]
                actual = record["simulation"]["objects"][0]["material"]
                self.assertEqual(actual, expected)

    def test_one_factor_derivation_preserves_base(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs/physics_sweep.json"
        config = load_json(config_path)
        base = {
            "schema_version": "physweep_pybullet_rigid_metadata_v1",
            "scene_id": "base_scene",
            "dataset_stage": "one_object_base_candidate",
            "semantic_sampling": {
                "five_dimensions": {
                    "motion": {"family": "roll_or_slide_1obj"}
                }
            },
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
                target_object_index=0,
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
            "semantic_sampling": {
                "five_dimensions": {
                    "motion": {"family": "surface_dual_independent_2obj"}
                }
            },
            "simulation": {
                "support": {"semantic_type": "wood_floor"},
                "interaction": {
                    "motion_pattern": "surface_dual_independent_2obj"
                },
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

    def test_metadata_sweep_domain_overrides_the_global_friction_domain(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs/physics_sweep.json"
        config = load_json(config_path)
        base = {
            "schema_version": "physweep_marble_run_scene_v1",
            "scene_id": "two_marble_scene",
            "semantics": {
                "motion": {"family": "two_marble_catch_up_collision"}
            },
            "physics": {"sweep_domains": {"contact_friction": [0.05, 1.0]}},
            "simulation": {
                "objects": [
                    {
                        "object_id": object_id,
                        "body_model": "rigid_body",
                        "material": {
                            "mass_kg": 0.1,
                            "contact_friction": 0.1,
                            "contact_restitution": 0.35,
                        },
                        "initial_state": {},
                    }
                    for object_id in ("object_a", "object_b")
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
                "contact_friction",
                0,
                {},
                {},
                target_object_index=0,
            )
        self.assertEqual(derived["sweep"]["allowed_domain"], [0.05, 1.0])
        self.assertEqual(derived["sweep"]["value"], 0.05)

    def test_each_supported_axis_changes_only_its_runtime_field(self):
        root = Path(__file__).resolve().parents[1]
        config_path = root / "configs/physics_sweep.json"
        config = load_json(config_path)
        base = {
            "schema_version": "physweep_pybullet_rigid_metadata_v1",
            "scene_id": "base_scene",
            "dataset_stage": "one_object_base_candidate",
            "semantic_sampling": {
                "five_dimensions": {
                    "motion": {"family": "roll_or_slide_1obj"}
                }
            },
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
                    target_object_index=0,
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
            values = sweep_values(
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
        values = sweep_values(
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

    def test_middle_policy_rejects_base_on_hard_boundary(self):
        axis_rules = {
            "level_count": 5,
            "level_positions": [0.0, 0.25, 0.5, 0.75, 1.0],
            "range_policy": {"mode": "global"},
            "domain": [0.0, 0.8],
            "scale": "linear",
        }
        with self.assertRaisesRegex(ValueError, "cannot occupy the middle level"):
            sweep_values(
                0.0,
                axis_rules,
                None,
                "contact_restitution",
                endpoint_policy={
                    "normalized_positions": [0.0, 0.25, 0.5, 0.75, 1.0],
                    "base_value_policy": "preserve_exactly_at_middle_position",
                    "edge_policy": "reject_if_middle_impossible",
                },
            )

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
        values = sweep_values(
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
