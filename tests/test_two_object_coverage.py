from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from tools.core.hashing import sha256_file
from tools.scene_rules.two_object import (
    allowed_camera_view_families,
    resolve_scene_rule,
    validate_two_object_scene_rules,
)
from tools.sampling.object_collection import compile_object_collection_scene
from tools.sampling.sample_two_object_base import _validated_intents
from tools.sampling.sample_two_object_coverage import (
    _axis_counts,
    _dynamics_profiles_eligible,
    _source_dynamics_profile,
    _source_meets_rule_camera_extent,
    _validate_complete_scene_coverage,
    coverage_cells,
    released_source_pool,
    select_coverage_sources,
)
from tools.sampling.two_object_sources import _asset_object_template


ROOT = Path(__file__).resolve().parents[1]


def load_matrix() -> dict:
    return json.loads(
        (ROOT / "configs/two_object_sampling_matrix.json").read_text(
            encoding="utf-8"
        )
    )


def load_scene_rules() -> dict:
    return json.loads(
        (ROOT / "configs/two_object_scene_rules.json").read_text(
            encoding="utf-8"
        )
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class TwoObjectCoverageTests(unittest.TestCase):
    def test_interacting_pair_dynamics_are_bounded_without_restricting_independent_motion(
        self,
    ) -> None:
        matrix = load_matrix()

        def source(mass: float, size: list[float] | None = None) -> dict:
            return {
                "metadata": {
                    "simulation": {
                        "objects": [
                            {
                                "material": {"mass_kg": mass},
                                "geometry": {
                                    "size_m": size or [0.1, 0.1, 0.1]
                                },
                            }
                        ]
                    }
                }
            }

        light, heavy = source(0.08), source(13.6)
        extreme_mass_profiles = [
            _source_dynamics_profile(value) for value in (light, heavy)
        ]
        self.assertFalse(
            _dynamics_profiles_eligible(
                extreme_mass_profiles,
                matrix,
                {"interaction_class": "interacting"},
            )
        )
        self.assertTrue(
            _dynamics_profiles_eligible(
                extreme_mass_profiles,
                matrix,
                {"interaction_class": "independent"},
            )
        )
        thin = source(1.0, [0.2, 0.05, 0.02])
        extreme_shape_profiles = [
            _source_dynamics_profile(value) for value in (thin, source(1.0))
        ]
        self.assertFalse(
            _dynamics_profiles_eligible(
                extreme_shape_profiles,
                matrix,
                {"interaction_class": "interacting"},
            )
        )
        self.assertTrue(
            _dynamics_profiles_eligible(
                extreme_shape_profiles,
                matrix,
                {"interaction_class": "independent"},
            )
        )

    def test_scene_rules_are_separate_unambiguous_and_regime_aware(self) -> None:
        matrix = load_matrix()
        self.assertNotIn("scene_compatibility", matrix)
        self.assertNotIn("visual_environment_coverage", matrix)
        self.assertNotIn("host_eligibility", matrix["candidate_pool"])
        rules = load_scene_rules()
        self.assertEqual(
            rules["schema_version"], "physweep_two_object_scene_rules_v4"
        )
        validate_two_object_scene_rules(rules)
        support = {
            "scene_class": "raised_flat",
            "support_shape": "rectangular_slab",
            "structure_family": "long_lab_bench",
            "surface_frame": {"slope_angle_degrees": 0.0},
        }
        self.assertEqual(
            resolve_scene_rule(rules, support, "airborne_supported")["id"],
            "raised_long_flat",
        )
        inclined_support = {
            "scene_class": "ground_feature",
            "support_shape": "inclined_ramp",
            "structure_family": "straight_long_shallow",
            "surface_frame": {"slope_angle_degrees": 9.7},
        }
        self.assertEqual(
            resolve_scene_rule(
                rules, inclined_support, "supported_supported"
            )["id"],
            "inclined_flat_slab",
        )
        self.assertIsNone(
            resolve_scene_rule(
                rules,
                inclined_support,
                "supported_supported",
                "interacting",
                "front_left_low",
                "surface_glancing_opposed_2obj",
            )
        )
        self.assertIsNone(
            resolve_scene_rule(
                rules,
                inclined_support,
                "supported_supported",
                "interacting",
                "front_left_low",
                "surface_glancing_hit_rest_2obj",
            )
        )
        standard_incline = copy.deepcopy(inclined_support)
        standard_incline["structure_family"] = "straight_standard"
        standard_incline["surface_frame"]["slope_angle_degrees"] = 14.036
        self.assertEqual(
            resolve_scene_rule(
                rules, standard_incline, "supported_supported"
            )["id"],
            "inclined_flat_slab",
        )
        self.assertIsNone(
            resolve_scene_rule(
                rules,
                inclined_support,
                "supported_supported",
                "interacting",
                "side_left_mid",
                "surface_crossing_2obj",
            )
        )
        self.assertEqual(
            resolve_scene_rule(
                rules,
                inclined_support,
                "supported_supported",
                "interacting",
                "front_left_low",
                "surface_crossing_2obj",
            )["id"],
            "inclined_flat_slab",
        )
        self.assertIsNone(
            resolve_scene_rule(
                rules, inclined_support, "airborne_supported"
            )
        )
        self.assertIsNone(
            resolve_scene_rule(
                rules,
                inclined_support,
                "supported_supported",
                "independent",
            )
        )
        self.assertIsNone(
            resolve_scene_rule(
                rules,
                inclined_support,
                "supported_supported",
                "interacting",
                "front_left_low",
                "surface_hit_rest_2obj",
            )
        )
        self.assertEqual(
            resolve_scene_rule(
                rules,
                inclined_support,
                "supported_supported",
                "interacting",
                "side_left_mid",
            )["id"],
            "inclined_flat_slab",
        )
        limited = copy.deepcopy(rules)
        limited["physical_rules"][0]["allowed_kinematic_regimes"] = [
            "supported_supported"
        ]
        validate_two_object_scene_rules(limited)
        overlapping = copy.deepcopy(rules)
        overlapping["physical_rules"][1]["allowed_structure_families"] = [
            "floor_patch"
        ]
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_two_object_scene_rules(overlapping)
        unknown_regime = copy.deepcopy(rules)
        unknown_regime["physical_rules"][0][
            "allowed_kinematic_regimes"
        ].append("unknown")
        with self.assertRaisesRegex(ValueError, "physical scene rule"):
            validate_two_object_scene_rules(unknown_regime)
        unknown_interaction = copy.deepcopy(rules)
        unknown_interaction["physical_rules"][0][
            "allowed_interaction_classes"
        ].append("unknown")
        with self.assertRaisesRegex(ValueError, "physical scene rule"):
            validate_two_object_scene_rules(unknown_interaction)
        invalid_camera = copy.deepcopy(rules)
        invalid_camera["physical_rules"][0][
            "allowed_camera_view_families"
        ].append("")
        with self.assertRaisesRegex(ValueError, "physical scene rule"):
            validate_two_object_scene_rules(invalid_camera)
        invalid_override = copy.deepcopy(rules)
        invalid_override["physical_rules"][-1][
            "camera_view_family_overrides"
        ]["surface_crossing_2obj"] = ["unknown_camera_family"]
        with self.assertRaisesRegex(ValueError, "physical scene rule"):
            validate_two_object_scene_rules(invalid_override)
        missing_camera_extent = copy.deepcopy(rules)
        missing_camera_extent["physical_rules"][-1].pop(
            "object_camera_plane_readability_by_view_family"
        )
        with self.assertRaisesRegex(ValueError, "physical scene rule"):
            validate_two_object_scene_rules(missing_camera_extent)
        incomplete_camera_extent = copy.deepcopy(rules)
        incomplete_camera_extent["physical_rules"][-1][
            "object_camera_plane_readability_by_view_family"
        ].pop("rear_right_high")
        with self.assertRaisesRegex(ValueError, "physical scene rule"):
            validate_two_object_scene_rules(incomplete_camera_extent)
        missing_pair_geometry = copy.deepcopy(rules)
        missing_pair_geometry["physical_rules"][-1].pop(
            "pair_camera_geometry_view_families"
        )
        with self.assertRaisesRegex(ValueError, "physical scene rule"):
            validate_two_object_scene_rules(missing_pair_geometry)
        incomplete_pair_geometry = copy.deepcopy(rules)
        incomplete_pair_geometry["physical_rules"][-1][
            "pair_camera_geometry_view_families"
        ].pop()
        with self.assertRaisesRegex(ValueError, "physical scene rule"):
            validate_two_object_scene_rules(incomplete_pair_geometry)
        flat_camera_extent = copy.deepcopy(rules)
        flat_camera_extent["physical_rules"][0][
            "object_camera_plane_readability_by_view_family"
        ] = {
            "side_left_mid": {
                "geometry_size_axes": [1, 2],
                "minimum_extent_m": 0.195,
            }
        }
        with self.assertRaisesRegex(ValueError, "physical scene rule"):
            validate_two_object_scene_rules(flat_camera_extent)
        unknown_extent_family = copy.deepcopy(rules)
        unknown_extent_family["physical_rules"][-1][
            "object_camera_plane_readability_by_view_family"
        ] = {
            "unknown": {
                "geometry_size_axes": [1, 2],
                "minimum_extent_m": 0.195,
            }
        }
        with self.assertRaisesRegex(ValueError, "physical scene rule"):
            validate_two_object_scene_rules(unknown_extent_family)
        string_slope = copy.deepcopy(rules)
        string_slope["physical_rules"][0][
            "maximum_abs_slope_degrees"
        ] = "0.0"
        with self.assertRaisesRegex(ValueError, "physical scene rule"):
            validate_two_object_scene_rules(string_slope)
        numeric_id = copy.deepcopy(rules)
        numeric_id["physical_rules"][0]["id"] = 1
        with self.assertRaisesRegex(ValueError, "ids must be unique"):
            validate_two_object_scene_rules(numeric_id)

    def test_inclined_camera_source_gate_rejects_thin_objects(self) -> None:
        rule = load_scene_rules()["physical_rules"][-1]
        side_cell = {"camera_view_family_id": "side_left_mid"}
        crossing_cell = {"camera_view_family_id": "front_left_low"}

        def source(size: list[float]) -> dict:
            return {
                "metadata": {
                    "simulation": {
                        "objects": [{"geometry": {"size_m": size}}]
                    }
                }
            }

        self.assertTrue(
            _source_meets_rule_camera_extent(
                source([0.205863, 0.205863, 0.205863]), rule, side_cell
            )
        )
        self.assertFalse(
            _source_meets_rule_camera_extent(
                source([0.115425, 0.115425, 0.192375]), rule, side_cell
            )
        )
        self.assertTrue(
            _source_meets_rule_camera_extent(
                source([0.115425, 0.115425, 0.192375]),
                rule,
                crossing_cell,
            )
        )
        self.assertFalse(
            _source_meets_rule_camera_extent(
                source([0.145816, 0.028575, 0.114854]),
                rule,
                crossing_cell,
            )
        )
        self.assertFalse(
            _source_meets_rule_camera_extent(
                source([0.165004, 0.22, 0.059764]),
                rule,
                crossing_cell,
            )
        )

    def test_complete_scene_coverage_rejects_declared_but_unselected_rules(
        self,
    ) -> None:
        matrix = load_matrix()
        rules = load_scene_rules()
        selections = []
        for intent in matrix["motion_intents"]:
            motion_id = str(intent["id"])
            for rule in rules["physical_rules"]:
                if (
                    intent["kinematic_regime"]
                    not in rule["allowed_kinematic_regimes"]
                    or intent["interaction_class"]
                    not in rule["allowed_interaction_classes"]
                ):
                    continue
                families = allowed_camera_view_families(rule, motion_id)
                for family in families:
                    for category in rules["visual_environment_coverage"][
                        "categories"
                    ]:
                        if rule["id"] not in category["allowed_scene_rules"]:
                            continue
                        selections.append(
                            {
                                "cell": {
                                    "motion_id": motion_id,
                                    "scene_rule_id": str(rule["id"]),
                                    "camera_view_family_id": family,
                                    "visual_environment_category": str(
                                        category["id"]
                                    ),
                                }
                            }
                        )
        _validate_complete_scene_coverage(matrix, rules, selections)
        missing_rule = [
            selection
            for selection in selections
            if selection["cell"]["scene_rule_id"] != "ground_open_hardscape"
        ]
        with self.assertRaisesRegex(ValueError, "misses physical scene rules"):
            _validate_complete_scene_coverage(matrix, rules, missing_rule)
        missing_environment = [
            selection
            for selection in selections
            if not (
                selection["cell"]["scene_rule_id"] == "ground_patch_flat"
                and selection["cell"]["visual_environment_category"]
                == "outdoor_courtyard"
            )
        ]
        with self.assertRaisesRegex(ValueError, "misses scene/environment pairs"):
            _validate_complete_scene_coverage(matrix, rules, missing_environment)
        missing_motion_rule = [
            selection
            for selection in selections
            if not (
                selection["cell"]["motion_id"] == "air_air_collision_2obj"
                and selection["cell"]["scene_rule_id"]
                == "ground_open_hardscape"
            )
        ]
        with self.assertRaisesRegex(ValueError, "misses motion/scene-rule pairs"):
            _validate_complete_scene_coverage(matrix, rules, missing_motion_rule)

    def test_asset_source_adapter_accepts_strict_centered_proxies(self) -> None:
        matrix = load_matrix()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            registry_path = root / "registry.json"
            visual_path = root / "assets/phone.glb"
            visual_path.parent.mkdir(parents=True)
            visual_path.write_bytes(b"unit asset mesh")
            asset_id = "asset_phone"
            registry = {
                "records": [
                    {
                        "asset_id": asset_id,
                        "proxy": {
                            "colliders": [
                                {
                                    "shape": "box",
                                    "size_m": [0.16, 0.08, 0.01],
                                    "position_m": [0.0, 0.0, 0.0],
                                    "rotation_euler_degrees": [0.0, 0.0, 0.0],
                                }
                            ]
                        },
                        "visual": {
                            "path": "assets/phone.glb",
                            "sha256": sha256_file(visual_path),
                            "canonical_extent_m": [0.16, 0.08, 0.01],
                            "alignment_euler_degrees": [0.0, 0.0, 0.0],
                        },
                    }
                ]
            }
            write_json(registry_path, registry)
            generation = {
                "scene_id": "asset_source",
                "registry": {
                    "path": registry_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(registry_path),
                },
                "assets": {"dynamic_asset_id": asset_id},
            }
            material = {
                "mass_kg": 0.2,
                "contact_friction": 0.3,
                "contact_restitution": 0.1,
                "linear_damping": 0.02,
                "angular_damping": 0.03,
                "rolling_friction": 0.0,
                "spinning_friction": 0.0,
            }
            released = {
                "physics": {"objects": [{"asset_id": asset_id, "material": material}]},
                "semantics": {
                    "objects": [
                        {"object_id": "object_a", "semantic_label": "smartphone"}
                    ]
                },
            }
            template, reason = _asset_object_template(
                source_root=root,
                runtime_root=root,
                generation_metadata=generation,
                release_metadata=released,
                eligibility=matrix["candidate_pool"]["object_eligibility"],
                registry_cache={},
                visual_hash_cache={},
            )
            self.assertEqual(reason, "eligible")
            obj = template["simulation"]["objects"][0]
            self.assertEqual(obj["geometry"]["type"], "cuboid")
            self.assertEqual(obj["visual_profile"]["type"], "mesh")
            self.assertEqual(
                template["semantic_sampling"]["five_dimensions"][
                    "foreground_object"
                ]["scale_bin"],
                "small",
            )
            visual_path.write_bytes(b"tampered unit asset mesh")
            with self.assertRaisesRegex(ValueError, "visual hash mismatch"):
                _asset_object_template(
                    source_root=root,
                    runtime_root=root,
                    generation_metadata=generation,
                    release_metadata=released,
                    eligibility=matrix["candidate_pool"]["object_eligibility"],
                    registry_cache={},
                    visual_hash_cache={},
                )
            visual_path.write_bytes(b"unit asset mesh")
            registry["records"][0]["proxy"]["colliders"] = [
                {
                    "shape": "cylinder",
                    "size_m": [0.08, 0.08, 0.06],
                    "position_m": [0.0, 0.0, -0.03],
                    "rotation_euler_degrees": [0.0, 0.0, 0.0],
                },
                {
                    "shape": "cylinder",
                    "size_m": [0.10, 0.10, 0.06],
                    "position_m": [0.0, 0.0, 0.03],
                    "rotation_euler_degrees": [0.0, 0.0, 0.0],
                },
            ]
            registry["records"][0]["visual"]["canonical_extent_m"] = [
                0.10,
                0.10,
                0.12,
            ]
            write_json(registry_path, registry)
            generation["registry"]["sha256"] = sha256_file(registry_path)
            compound, reason = _asset_object_template(
                source_root=root,
                runtime_root=root,
                generation_metadata=generation,
                release_metadata=released,
                eligibility=matrix["candidate_pool"]["object_eligibility"],
                registry_cache={},
                visual_hash_cache={},
            )
            self.assertEqual(reason, "eligible")
            compound_obj = compound["simulation"]["objects"][0]
            self.assertEqual(
                compound_obj["geometry"],
                {"type": "cylinder", "size_m": [0.10, 0.10, 0.12]},
            )
            self.assertEqual(compound_obj["collision_profile"]["type"], "compound")
            self.assertEqual(len(compound_obj["collision_profile"]["colliders"]), 2)

            registry["records"][0]["proxy"]["colliders"][1]["position_m"][0] = 0.01
            write_json(registry_path, registry)
            generation["registry"]["sha256"] = sha256_file(registry_path)
            rejected, reason = _asset_object_template(
                source_root=root,
                runtime_root=root,
                generation_metadata=generation,
                release_metadata=released,
                eligibility=matrix["candidate_pool"]["object_eligibility"],
                registry_cache={},
                visual_hash_cache={},
            )
            self.assertIsNone(rejected)
            self.assertEqual(reason, "off_axis_compound_proxy")

            registry["records"][0]["proxy"]["colliders"][0]["position_m"] = [
                0.0,
                0.0,
                -0.025,
            ]
            registry["records"][0]["proxy"]["colliders"][1]["position_m"] = [
                0.0,
                0.0,
                0.035,
            ]
            write_json(registry_path, registry)
            generation["registry"]["sha256"] = sha256_file(registry_path)
            rejected, reason = _asset_object_template(
                source_root=root,
                runtime_root=root,
                generation_metadata=generation,
                release_metadata=released,
                eligibility=matrix["candidate_pool"]["object_eligibility"],
                registry_cache={},
                visual_hash_cache={},
            )
            self.assertIsNone(rejected)
            self.assertEqual(reason, "off_center_compound_proxy")

    def test_matrix_declares_complete_ordered_cartesian_coverage(self) -> None:
        matrix = load_matrix()
        self.assertEqual(
            matrix["schema_version"], "physweep_two_object_sampling_matrix_v14"
        )
        scene_rules = load_scene_rules()
        validate_two_object_scene_rules(scene_rules)
        self.assertEqual(
            [rule["id"] for rule in scene_rules["physical_rules"]],
            [
                "ground_patch_flat",
                "ground_long_flat",
                "ground_open_hardscape",
                "raised_wide_flat",
                "raised_long_flat",
                "inclined_flat_slab",
            ],
        )
        intents = {record["id"]: record for record in matrix["motion_intents"]}
        self.assertEqual(
            intents["surface_head_on_2obj"]["linear_velocity_m_s"],
            [[0.60, 0.0, 0.0], [-0.24, 0.0, 0.0]],
        )
        self.assertEqual(
            intents["surface_crossing_2obj"]["impact_offset_ratio"], 0.10
        )
        self.assertEqual(
            intents["surface_glancing_hit_rest_2obj"]["linear_velocity_m_s"],
            [[0.68, 0.24, 0.0], [0.0, 0.0, 0.0]],
        )
        self.assertEqual(
            intents["surface_glancing_hit_rest_2obj"]["impact_offset_ratio"],
            0.28,
        )
        self.assertEqual(
            intents["surface_glancing_opposed_2obj"]["linear_velocity_m_s"],
            [[0.62, 0.0, 0.0], [-0.36, -0.36, 0.0]],
        )
        self.assertEqual(
            intents["surface_glancing_opposed_2obj"]["impact_offset_ratio"],
            0.22,
        )
        self.assertEqual(
            intents["surface_catch_up_2obj"]["contact_time_s"], 0.35
        )
        self.assertEqual(
            intents["surface_catch_up_2obj"]["linear_velocity_m_s"],
            [[0.97, 0.0, 0.0], [0.39, 0.0, 0.0]],
        )
        self.assertEqual(
            intents["air_drop_hit_supported_2obj"]["linear_velocity_m_s"],
            [[0.12, 0.0, 0.0], [0.0, 0.0, 0.0]],
        )
        self.assertEqual(
            matrix["pair_observation"]["maximum_camera_distance_m"], 6.0
        )
        self.assertEqual(
            matrix["pair_observation"][
                "maximum_camera_distance_above_minimum_m"
            ],
            5.0,
        )
        self.assertEqual(
            matrix["pair_observation"]["preferred_full_motion_envelope_span_ndc"],
            0.55,
        )
        cells, full_count = coverage_cells(matrix)
        self.assertEqual(
            matrix["shape_motion_compatibility"]["pair_sets"][
                "independent_control_pairs"
            ],
            ["sphere_to_sphere", "cuboid_to_cuboid", "cylinder_to_cylinder"],
        )
        self.assertEqual(full_count, 1317)
        self.assertEqual(len(cells), 1317)
        interaction_counts = Counter(cell["interaction_class"] for cell in cells)
        self.assertEqual(
            interaction_counts,
            {"interacting": 1155, "independent": 162},
        )
        self.assertGreater(interaction_counts["interacting"] / full_count, 0.80)
        counts = _axis_counts(cells)
        self.assertEqual(
            counts["interaction_class"],
            {"independent": 162, "interacting": 1155},
        )
        self.assertEqual(
            set(counts["motion"].values()), {18, 54, 66, 162, 198, 243}
        )
        self.assertEqual(
            set(counts["ordered_scale_pair"].values()), {137, 158}
        )
        self.assertEqual(
            counts["scene_class"],
            {"ground_feature": 165, "ground_flat": 576, "raised_flat": 576},
        )
        glancing_rest_cells = [
            cell
            for cell in cells
            if cell["motion_id"] == "surface_glancing_hit_rest_2obj"
        ]
        self.assertEqual(len(glancing_rest_cells), 162)
        self.assertEqual(
            {cell["scene_class"] for cell in glancing_rest_cells},
            {"ground_flat", "raised_flat"},
        )
        self.assertEqual(
            {cell["shape_pair_id"] for cell in glancing_rest_cells},
            {
                record["id"]
                for record in matrix["coverage_plan"][
                    "role_ordered_shape_pairs"
                ]
            },
        )
        self.assertEqual(
            {cell["scale_pair_id"] for cell in glancing_rest_cells},
            {
                record["id"]
                for record in matrix["coverage_plan"][
                    "role_ordered_scale_pairs"
                ]
            },
        )
        self.assertEqual(
            {cell["camera_view_family_id"] for cell in glancing_rest_cells},
            set(counts["camera_view_family"]),
        )
        self.assertLessEqual(
            max(counts["camera_view_family"].values())
            - min(counts["camera_view_family"].values()),
            1,
        )
        incline_cells = [
            cell for cell in cells if cell["scene_class"] == "ground_feature"
        ]
        large_incline_scale_pairs = {
            "medium_to_medium",
            "medium_to_large",
            "large_to_medium",
            "large_to_large",
        }
        unconstrained_incline_motions = {"surface_crossing_2obj"}
        for cell in incline_cells:
            if cell["motion_id"] not in unconstrained_incline_motions:
                self.assertIn(cell["scale_pair_id"], large_incline_scale_pairs)
        all_scale_pairs = {
            record["id"]
            for record in matrix["coverage_plan"]["role_ordered_scale_pairs"]
        }
        for motion_id in sorted(unconstrained_incline_motions):
            self.assertEqual(
                {
                    cell["scale_pair_id"]
                    for cell in incline_cells
                    if cell["motion_id"] == motion_id
                },
                all_scale_pairs,
            )
        family_ids = set(counts["camera_view_family"])
        scene_camera_families = {
            "ground_feature": {
                "side_left_mid",
                "side_right_mid",
                "front_left_low",
                "rear_right_high",
            },
            "ground_flat": family_ids,
            "raised_flat": family_ids,
        }
        for axis in (
            "motion_id",
            "shape_pair_id",
            "scale_pair_id",
            "scene_class",
        ):
            conditional = defaultdict(Counter)
            for cell in cells:
                conditional[str(cell[axis])][
                    str(cell["camera_view_family_id"])
                ] += 1
            for axis_value, axis_counts in conditional.items():
                expected_families = (
                    scene_camera_families[axis_value]
                    if axis == "scene_class"
                    else family_ids
                )
                self.assertEqual(set(axis_counts), expected_families, axis_value)

    def test_balanced_smoke_prefix_covers_every_axis(self) -> None:
        cells, full_count = coverage_cells(load_matrix(), 72)
        self.assertEqual(full_count, 1317)
        interaction_counts = Counter(cell["interaction_class"] for cell in cells)
        self.assertEqual(interaction_counts, {"interacting": 64, "independent": 8})
        self.assertGreater(interaction_counts["interacting"] / len(cells), 0.80)
        counts = _axis_counts(cells)
        intents = {
            record["id"]: record["interaction_class"]
            for record in load_matrix()["motion_intents"]
        }
        for interaction_class in ("interacting", "independent"):
            motion_counts = [
                count
                for motion_id, count in counts["motion"].items()
                if intents[motion_id] == interaction_class
            ]
            self.assertLessEqual(max(motion_counts) - min(motion_counts), 2)
        self.assertEqual(len(counts["ordered_shape_pair"]), 9)
        self.assertGreaterEqual(min(counts["ordered_shape_pair"].values()), 2)
        self.assertLessEqual(
            max(counts["ordered_scale_pair"].values())
            - min(counts["ordered_scale_pair"].values()),
            2,
        )
        self.assertEqual(set(counts["scene_class"]), {
            "ground_feature", "ground_flat", "raised_flat"
        })
        self.assertLessEqual(
            max(counts["scene_class"].values())
            - min(counts["scene_class"].values()),
            7,
        )
        self.assertEqual(len(counts["camera_view_family"]), 6)
        self.assertLessEqual(
            max(counts["camera_view_family"].values())
            - min(counts["camera_view_family"].values()),
            6,
        )

    def test_matrix_rejects_missing_scale_cell_and_weakened_uniqueness(self) -> None:
        matrix = load_matrix()
        missing = copy.deepcopy(matrix)
        missing["coverage_plan"]["role_ordered_scale_pairs"].pop()
        with self.assertRaisesRegex(ValueError, "full product"):
            _validated_intents(missing)
        missing_shape = copy.deepcopy(matrix)
        missing_shape["coverage_plan"]["role_ordered_shape_pairs"].pop()
        with self.assertRaisesRegex(ValueError, "full product"):
            _validated_intents(missing_shape)
        invalid_scale_compatibility = copy.deepcopy(matrix)
        invalid_scale_compatibility["coverage_plan"][
            "scene_motion_scale_compatibility"
        ][0]["allowed_scale_pair_ids"] = []
        with self.assertRaisesRegex(ValueError, "scene-motion-scale"):
            _validated_intents(invalid_scale_compatibility)
        overlapping_scale_compatibility = copy.deepcopy(matrix)
        overlapping_scale_compatibility["coverage_plan"][
            "scene_motion_scale_compatibility"
        ].append(
            copy.deepcopy(
                overlapping_scale_compatibility["coverage_plan"][
                    "scene_motion_scale_compatibility"
                ][0]
            )
        )
        with self.assertRaisesRegex(ValueError, "overlaps"):
            _validated_intents(overlapping_scale_compatibility)
        unknown_shape_pair = copy.deepcopy(matrix)
        unknown_shape_pair["shape_motion_compatibility"]["pair_sets"][
            "all_shape_pairs"
        ].append("unknown_to_sphere")
        with self.assertRaisesRegex(ValueError, "shape-pair sets"):
            _validated_intents(unknown_shape_pair)
        scene_rules = load_scene_rules()
        unsupported_host = copy.deepcopy(scene_rules)
        unsupported_host["host_eligibility"][
            "allowed_visual_types"
        ].append("mesh_environment")
        with self.assertRaisesRegex(ValueError, "motion-neutral"):
            validate_two_object_scene_rules(unsupported_host)
        weakened = copy.deepcopy(matrix)
        weakened["coverage_plan"]["selection_policy"][
            "source_pair_uniqueness"
        ] = "allow_reuse"
        with self.assertRaisesRegex(ValueError, "may not be weakened"):
            _validated_intents(weakened)
        excessive_reuse = copy.deepcopy(matrix)
        excessive_reuse["coverage_plan"]["selection_policy"][
            "maximum_host_source_reuse"
        ] = 3
        with self.assertRaisesRegex(ValueError, "may not be weakened"):
            _validated_intents(excessive_reuse)
        active_host = copy.deepcopy(scene_rules)
        active_host["host_eligibility"][
            "allowed_collider_roles"
        ].append("impact_wall")
        with self.assertRaisesRegex(ValueError, "motion-neutral"):
            validate_two_object_scene_rules(active_host)
        bounded_host = copy.deepcopy(scene_rules)
        bounded_host["host_eligibility"][
            "camera_envelope_policy"
        ] = "allow_bounded"
        with self.assertRaisesRegex(ValueError, "motion-neutral"):
            validate_two_object_scene_rules(bounded_host)
        impossible_interaction_mix = copy.deepcopy(matrix)
        impossible_interaction_mix["coverage_plan"][
            "minimum_interacting_fraction"
        ] = 0.90
        with self.assertRaisesRegex(ValueError, "interaction mix"):
            coverage_cells(impossible_interaction_mix)
        exact_interaction_mix = copy.deepcopy(matrix)
        cells, _ = coverage_cells(exact_interaction_mix)
        exact_interaction_mix["coverage_plan"][
            "minimum_interacting_fraction"
        ] = sum(
            cell["interaction_class"] == "interacting" for cell in cells
        ) / len(cells)
        self.assertEqual(len(coverage_cells(exact_interaction_mix)[0]), len(cells))
        missing_environment = copy.deepcopy(scene_rules)
        missing_environment["visual_environment_coverage"]["categories"].pop()
        missing_environment["visual_environment_coverage"]["categories"][0][
            "allowed_scene_rules"
        ] = ["unknown_scene_rule"]
        with self.assertRaisesRegex(ValueError, "visual-environment categories"):
            validate_two_object_scene_rules(missing_environment)

    def test_source_selection_obeys_scale_and_uniqueness(self) -> None:
        matrix = load_matrix()
        cells, _ = coverage_cells(matrix, 18)
        objects = []
        for source_family in ("generic", "asset"):
            for shape in ("sphere", "cuboid", "cylinder"):
                for scale in ("small", "medium", "large"):
                    for profile_index in range(3):
                        for source_index in range(4):
                            source_id = (
                                f"object_{source_family}_{shape}_{scale}_"
                                f"{profile_index}_{source_index}"
                            )
                            objects.append(
                                {
                                    "metadata": {},
                                    "source": {"scene_id": source_id},
                                    "source_family": source_family,
                                    "shape_family_id": shape,
                                    "scale_bin": scale,
                                    "visual_profile_id": (
                                        f"{source_family}_{shape}_profile_"
                                        f"{profile_index}"
                                    ),
                                }
                            )
        hosts = []
        scene_rules = load_scene_rules()
        environments_by_rule = {
            str(rule["id"]): [
                str(category["id"])
                for category in scene_rules["visual_environment_coverage"][
                    "categories"
                ]
                if str(rule["id"]) in category["allowed_scene_rules"]
            ]
            for rule in scene_rules["physical_rules"]
        }
        for rule in scene_rules["physical_rules"]:
            scene_class = str(rule["scene_class"])
            scene_rule_id = str(rule["id"])
            environment_categories = environments_by_rule[scene_rule_id]
            for profile_index in range(3):
                for source_index in range(3):
                    source_id = (
                        f"host_{scene_rule_id}_{profile_index}_{source_index}"
                    )
                    hosts.append(
                        {
                            "metadata": {},
                            "source": {"scene_id": source_id},
                            "scene_rule_id": scene_rule_id,
                            "scene_class": scene_class,
                            "visual_profile_id": f"host_profile_{profile_index}",
                            "visual_type": "procedural_room",
                            "environment_category": environment_categories[
                                (profile_index + source_index)
                                % len(environment_categories)
                            ],
                        }
                    )
        selected = select_coverage_sources(cells, objects, hosts, matrix)
        self.assertEqual(len(selected), 18)
        self.assertTrue(
            all(
                selection["cell"]["cell_id"].startswith(
                    f"{cell['cell_id']}__"
                )
                for cell, selection in zip(cells, selected, strict=True)
            )
        )
        selected_object_ids = [
            str(record["source"]["scene_id"])
            for selection in selected
            for record in selection["objects"]
        ]
        selected_pairs = {
            tuple(
                sorted(
                    str(record["source"]["scene_id"])
                    for record in selection["objects"]
                )
            )
            for selection in selected
        }
        selected_host_ids = [
            str(selection["host"]["source"]["scene_id"])
            for selection in selected
        ]
        self.assertEqual(len(selected_pairs), 18)
        self.assertEqual(len(set(selected_host_ids)), 18)
        self.assertLessEqual(max(Counter(selected_object_ids).values()), 2)
        self.assertLessEqual(max(Counter(selected_host_ids).values()), 2)
        source_pair_counts = Counter(
            str(selection["cell"]["source_family_pair_id"])
            for selection in selected
        )
        self.assertEqual(
            set(source_pair_counts),
            {
                "generic_to_generic",
                "generic_to_asset",
                "asset_to_generic",
                "asset_to_asset",
            },
        )
        self.assertLessEqual(
            max(source_pair_counts.values()) - min(source_pair_counts.values()), 1
        )
        camera_source_pair_counts = defaultdict(Counter)
        for selection in selected:
            cell = selection["cell"]
            camera_source_pair_counts[cell["camera_view_family_id"]][
                cell["source_family_pair_id"]
            ] += 1
        self.assertTrue(
            all(
                max(counts.values()) - min(counts.values()) <= 1
                for counts in camera_source_pair_counts.values()
            ),
            camera_source_pair_counts,
        )
        for selection in selected:
            cell = selection["cell"]
            left, right = selection["objects"]
            self.assertEqual(left["scale_bin"], cell["object_a_scale_bin"])
            self.assertEqual(right["scale_bin"], cell["object_b_scale_bin"])
            self.assertEqual(
                left["shape_family_id"], cell["object_a_shape"]
            )
            self.assertEqual(
                right["shape_family_id"], cell["object_b_shape"]
            )
            self.assertEqual(
                selection["host"]["scene_class"], cell["scene_class"]
            )
            self.assertEqual(
                selection["host"]["scene_rule_id"], cell["scene_rule_id"]
            )
            self.assertEqual(
                selection["host"]["environment_category"],
                cell["visual_environment_category"],
            )
        per_scene_environment = defaultdict(Counter)
        for selection in selected:
            cell = selection["cell"]
            per_scene_environment[cell["scene_class"]][
                cell["visual_environment_category"]
            ] += 1
        self.assertTrue(
            all(
                max(counts.values()) - min(counts.values()) <= 1
                for counts in per_scene_environment.values()
            ),
            per_scene_environment,
        )
        camera_scene_environment = defaultdict(lambda: defaultdict(Counter))
        for selection in selected:
            cell = selection["cell"]
            camera_scene_environment[cell["camera_view_family_id"]][
                cell["scene_class"]
            ][cell["visual_environment_category"]] += 1
        for scene_counts in camera_scene_environment.values():
            for counts in scene_counts.values():
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 2)
        self.assertEqual(
            {selection["cell"]["scene_rule_id"] for selection in selected},
            {str(rule["id"]) for rule in scene_rules["physical_rules"]},
        )

    def test_source_selection_allows_matching_size_and_appearance(self) -> None:
        cell = {
            "cell_id": "same_size_and_appearance",
            "motion_id": "surface_single_independent_2obj",
            "object_a_shape": "sphere",
            "object_b_shape": "sphere",
            "object_a_scale_bin": "small",
            "object_b_scale_bin": "small",
            "scene_class": "ground_flat",
            "camera_view_family_id": "side_left_mid",
        }
        objects = [
            {
                "metadata": {},
                "source": {"scene_id": f"object_{index}"},
                "source_family": "generic",
                "shape_family_id": "sphere",
                "scale_bin": "small",
                "visual_profile_id": "shared_profile",
            }
            for index in range(2)
        ]
        hosts = [
            {
                "metadata": {},
                "source": {"scene_id": "host"},
                "scene_rule_id": "ground_patch_flat",
                "scene_class": "ground_flat",
                "visual_profile_id": "host_profile",
                "visual_type": "procedural_room",
                "environment_category": "minimal",
            },
            {
                "metadata": {},
                "source": {"scene_id": "unused_prefix_host"},
                "scene_rule_id": "ground_patch_flat",
                "scene_class": "ground_flat",
                "visual_profile_id": "unused_prefix_profile",
                "visual_type": "procedural_room",
                "environment_category": "minimal",
            },
        ]
        selected = select_coverage_sources(
            [cell],
            objects,
            hosts,
            load_matrix(),
            require_all_profiles=False,
        )
        left, right = selected[0]["objects"]
        self.assertEqual(left["scale_bin"], right["scale_bin"])
        self.assertEqual(
            left["visual_profile_id"], right["visual_profile_id"]
        )

    def test_object_collection_preserves_appearance_for_three_objects(self) -> None:
        def candidate(index: int) -> dict:
            material = {
                "record": {
                    "asset_source": "unit",
                    "asset_id": f"material_{index}",
                    "path": f"materials/{index}",
                },
                "texture_scale": 1.0 + index,
                "semantic_color_mix": 0.05 * index,
            }
            return {
                "schema_version": "physweep_pybullet_rigid_metadata_v1",
                "scene_id": f"source_{index}",
                "simulation": {
                    "objects": [
                        {
                            "object_id": "object_a",
                            "body_model": "rigid_body",
                            "semantic_type": f"source object {index}",
                            "visual_profile": {
                                "id": f"profile_{index}",
                                "material_policy": "source_or_bound_fallback",
                            },
                        }
                    ]
                },
                "semantic_sampling": {
                    "five_dimensions": {
                        "foreground_object": {
                            "semantic_category": "sphere",
                            "scale_bin": "medium",
                            "uniform_scale": 1.0,
                        }
                    }
                },
                "appearance": {"materials": {"dynamic_object": material}},
            }

        sources = [candidate(index) for index in range(3)]
        original_sources = copy.deepcopy(sources)
        host = copy.deepcopy(sources[0])
        roles = [{"object_id": f"object_{index}"} for index in range(3)]
        scene = compile_object_collection_scene(host, sources, roles)
        self.assertEqual(sources, original_sources)
        for index, role in enumerate(roles):
            object_id = role["object_id"]
            self.assertEqual(
                scene["simulation"]["objects"][index]["visual_profile"],
                sources[index]["simulation"]["objects"][0]["visual_profile"],
            )
            self.assertEqual(
                scene["appearance"]["materials"]["dynamic_objects"][object_id],
                sources[index]["appearance"]["materials"]["dynamic_object"],
            )

        missing_generic_material = copy.deepcopy(sources[1])
        missing_generic_material["appearance"]["materials"] = {}
        with self.assertRaisesRegex(
            ValueError, "lacks a dynamic appearance material"
        ):
            compile_object_collection_scene(
                host,
                [sources[0], missing_generic_material],
                roles[:2],
            )

        asset_adapter = copy.deepcopy(missing_generic_material)
        asset_adapter["simulation"]["objects"][0]["visual_profile"].update(
            {"type": "mesh", "material_policy": "source_or_bound_fallback"}
        )
        fallback_scene = compile_object_collection_scene(
            host,
            [sources[0], asset_adapter],
            roles[:2],
        )
        self.assertEqual(
            fallback_scene["appearance"]["materials"]["dynamic_objects"][
                "object_1"
            ],
            host["appearance"]["materials"]["dynamic_object"],
        )

    def test_released_base_manifest_pins_generation_manifest(self) -> None:
        matrix = load_matrix()
        scene_rules = load_scene_rules()
        ground_rule = copy.deepcopy(scene_rules["physical_rules"][0])
        second_ground_rule = copy.deepcopy(ground_rule)
        ground_rule["id"] = "ground_floor_patch"
        ground_rule["allowed_structure_families"] = ["floor_patch"]
        second_ground_rule["id"] = "ground_open_hardscape"
        second_ground_rule["allowed_structure_families"] = ["open_hardscape"]
        scene_rules["physical_rules"] = [ground_rule, second_ground_rule]
        scene_rules["visual_environment_coverage"]["categories"] = [
            {
                "id": "minimal",
                "allowed_scene_rules": ["ground_floor_patch"],
            },
            {
                "id": "home_office",
                "allowed_scene_rules": ["ground_open_hardscape"],
            },
        ]
        validate_two_object_scene_rules(scene_rules)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source_root = root / "source"
            records = []
            for index, scale in enumerate(("small", "large")):
                scene_id = f"source_{index}"
                metadata_path = source_root / f"metadata/{scene_id}.json"
                metadata = {
                    "schema_version": "physweep_pybullet_rigid_metadata_v1",
                    "scene_id": scene_id,
                    "simulation": {
                        "objects": [
                            {
                                "body_model": "rigid_body",
                                "geometry": {
                                    "type": "sphere",
                                    "size_m": [0.2, 0.2, 0.2],
                                },
                                "collision_profile": {
                                    "type": "sphere",
                                    "size_m": [0.2, 0.2, 0.2],
                                },
                                "initial_state": {
                                    "pose_profile": "support_normal"
                                },
                                "visual_profile": {"id": f"profile_{index}"},
                            }
                        ],
                        "support": {
                            "scene_class": "ground_flat",
                            "support_shape": "rectangular_slab",
                            "structure_family": (
                                "floor_patch" if index == 0 else "open_hardscape"
                            ),
                            "surface_frame": {"slope_angle_degrees": 0.0},
                            "colliders": [
                                {"id": "support", "role": "primary_support"}
                            ],
                        },
                    },
                    "semantic_sampling": {
                        "five_dimensions": {
                            "foreground_object": {"scale_bin": scale}
                        }
                    },
                    "appearance": {
                        "scene_visual": {
                            "id": f"host_{index}",
                            "visual_type": "procedural_room",
                            "environment_category": (
                                "minimal" if index == 0 else "home_office"
                            ),
                        }
                    },
                }
                write_json(metadata_path, metadata)
                records.append(
                    {
                        "scene_id": scene_id,
                        "path": metadata_path.relative_to(source_root).as_posix(),
                        "metadata_sha256": sha256_file(metadata_path),
                        "source_schema_version": metadata["schema_version"],
                        "kind": "base",
                    }
                )
            source_manifest_path = source_root / "release/metadata_manifest.json"
            write_json(
                source_manifest_path,
                {
                    "schema_version": "physweep_release_metadata_manifest_v2",
                    "dataset_id": "unit_one_object",
                    "sample_count": 2,
                    "group_count": 2,
                    "records": records,
                },
            )
            released_path = root / "outputs/one_object/base/manifest.json"
            pipelines = {}
            for family, family_records in (
                ("generic", records),
                ("asset", []),
            ):
                branch_path = released_path.parent / family / "manifest.json"
                compact_records = []
                for record in family_records:
                    scene_id = record["scene_id"]
                    compact_path = branch_path.parent / scene_id / "metadata.json"
                    compact = {
                        "schema_version": "physweep_base_sample_v11",
                        "scene_id": scene_id,
                        "family": family,
                        "lineage": {
                            "source_generation_metadata_sha256": record[
                                "metadata_sha256"
                            ]
                        },
                    }
                    write_json(compact_path, compact)
                    compact_records.append(
                        {
                            "scene_id": scene_id,
                            "metadata_sha256": sha256_file(compact_path),
                        }
                    )
                write_json(
                    branch_path,
                    {
                        "schema_version": "physweep_base_pipeline_view_v12",
                        "pipeline": family,
                        "sample_count": len(compact_records),
                        "records": compact_records,
                    },
                )
                pipelines[family] = {
                    "manifest": branch_path.relative_to(
                        released_path.parent
                    ).as_posix(),
                    "manifest_sha256": sha256_file(branch_path),
                }
            write_json(
                released_path,
                {
                    "schema_version": "physweep_base_release_view_v14",
                    "dataset_id": "unit_one_object",
                    "sample_count": 2,
                    "pipelines": pipelines,
                    "provenance": {
                        "source_generation_release_metadata": {
                            "schema_version": (
                                "physweep_release_metadata_manifest_v2"
                            ),
                            "manifest_sha256": sha256_file(
                                source_manifest_path
                            ),
                        }
                    },
                },
            )
            objects, hosts = released_source_pool(
                root=root,
                released_base_manifest_path=released_path,
                source_root=source_root,
                source_manifest_path=source_manifest_path,
                matrix=matrix,
                scene_rules=scene_rules,
            )
            self.assertEqual(len(objects), 2)
            self.assertEqual(len(hosts), 2)
            uncovered_rule = copy.deepcopy(scene_rules)
            missing = copy.deepcopy(uncovered_rule["physical_rules"][0])
            missing["id"] = "ground_unrepresented_flat"
            missing["allowed_structure_families"] = ["unrepresented_flat"]
            uncovered_rule["physical_rules"].append(missing)
            uncovered_rule["visual_environment_coverage"]["categories"][0][
                "allowed_scene_rules"
            ].append("ground_unrepresented_flat")
            validate_two_object_scene_rules(uncovered_rule)
            with self.assertRaisesRegex(
                ValueError, "do not cover declared physical scene rules"
            ):
                released_source_pool(
                    root=root,
                    released_base_manifest_path=released_path,
                    source_root=source_root,
                    source_manifest_path=source_manifest_path,
                    matrix=matrix,
                    scene_rules=uncovered_rule,
                )
            tampered = json.loads(
                source_manifest_path.read_text(encoding="utf-8")
            )
            tampered["dataset_id"] = "changed"
            write_json(source_manifest_path, tampered)
            with self.assertRaisesRegex(ValueError, "does not name"):
                released_source_pool(
                    root=root,
                    released_base_manifest_path=released_path,
                    source_root=source_root,
                    source_manifest_path=source_manifest_path,
                    matrix=matrix,
                    scene_rules=scene_rules,
                )


if __name__ == "__main__":
    unittest.main()
