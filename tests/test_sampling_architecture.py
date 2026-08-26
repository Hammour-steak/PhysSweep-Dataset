from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from rigid_geometry import SUPPORT_BUILDERS  # noqa: E402
from sample_pybullet_base import (  # noqa: E402
    BUNDLE_PATH,
    compile_camera_observation,
    constrained_trajectory_extent,
    load_active_rules,
    scene_visual_profile_admits_camera,
    support_allowed,
)
from scene_kit_compiler import (  # noqa: E402
    compile_object_profile,
    compile_scene_kit,
    validate_generic_capabilities,
    validate_object_visual_curation,
    validate_registry_counts,
)


class SamplingArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = load_active_rules(ROOT)
        cls.objects = json.loads(
            (ROOT / "configs/physassets_core_object_profiles.json").read_text(encoding="utf-8")
        )["profiles"]
        cls.kits = json.loads(
            (ROOT / "configs/scene_kits.json").read_text(encoding="utf-8")
        )["kits"]
        cls.scene_visuals = json.loads(
            (ROOT / "configs/scene_visual_profiles.json").read_text(
                encoding="utf-8"
            )
        )
        cls.scene_mesh_visuals = json.loads(
            (ROOT / "configs/scene_mesh_profiles.json").read_text(
                encoding="utf-8"
            )
        )
        cls.support_mesh_visuals = json.loads(
            (ROOT / "configs/support_mesh_profiles.json").read_text(
                encoding="utf-8"
            )
        )
        cls.asset_proxy_registry = json.loads(
            (ROOT / "configs/asset_proxy_registry.json").read_text(
                encoding="utf-8"
            )
        )
        cls.asset_proxy_annotations = json.loads(
            (
                ROOT
                / "configs/source_annotations/background_asset_proxy_annotations_v1.json"
            ).read_text(encoding="utf-8")
        )
        cls.asset_semantic_rules = json.loads(
            (ROOT / "configs/asset_semantic_scene_rules.json").read_text(
                encoding="utf-8"
            )
        )
        cls.asset_scene_composition = json.loads(
            (ROOT / "configs/asset_scene_composition.json").read_text(
                encoding="utf-8"
            )
        )
        cls.backend = json.loads(
            (ROOT / "configs/pybullet_backend.json").read_text(encoding="utf-8")
        )
        cls.passive_pinball_backend = json.loads(
            (ROOT / "configs/passive_pinball_backend.json").read_text(
                encoding="utf-8"
            )
        )
        cls.marble_run_backend = json.loads(
            (ROOT / "configs/marble_run_backend.json").read_text(
                encoding="utf-8"
            )
        )
        cls.base_rules = json.loads(
            (ROOT / "configs/one_object_sampling_rules.json").read_text(
                encoding="utf-8"
            )
        )
        cls.visual_rules = json.loads(
            (ROOT / "configs/visual_sampling.json").read_text(encoding="utf-8")
        )
        cls.backend_capabilities = json.loads(
            (ROOT / "configs/backend_capabilities.json").read_text(
                encoding="utf-8"
            )
        )
        cls.bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
        cls.object_visual_curation = json.loads(
            (ROOT / cls.bundle["object_visual_curation"]).read_text(encoding="utf-8")
        )
        cls.object_visual_preflight = json.loads(
            (ROOT / cls.bundle["object_visual_preflight_report"]).read_text(
                encoding="utf-8"
            )
        )
        cls.object_profiles_path = ROOT / cls.bundle["object_profiles"]
        cls.object_profiles = json.loads(
            cls.object_profiles_path.read_text(encoding="utf-8")
        )
        cls.preflight_policy_path = ROOT / cls.bundle["object_visual_preflight"]
        cls.preflight_policy = json.loads(
            cls.preflight_policy_path.read_text(encoding="utf-8")
        )
        cls.preflight_report_path = (
            ROOT / cls.bundle["object_visual_preflight_report"]
        )
        cls.material_manifest = json.loads(
            (ROOT / cls.bundle["material_manifest"]).read_text(encoding="utf-8")
        )

    def test_scene_visual_camera_range_must_overlap_motion_range(self) -> None:
        camera_rules = {
            "motion_intents": {
                "projectile_1obj": {"elevation_range_degrees": [22.0, 44.0]}
            }
        }
        incompatible = {
            "visual_type": "mesh_backdrop",
            "camera_context": {"maximum_elevation_degrees": 20.0},
        }
        compatible = {
            "visual_type": "mesh_backdrop",
            "camera_context": {"maximum_elevation_degrees": 28.0},
        }
        self.assertFalse(
            scene_visual_profile_admits_camera(
                incompatible,
                "projectile_1obj",
                "low_fast_horizontal",
                "short",
                {"label": "concrete_floor_mat"},
                camera_rules,
            )
        )
        self.assertTrue(
            scene_visual_profile_admits_camera(
                compatible,
                "projectile_1obj",
                "low_fast_horizontal",
                "short",
                {"label": "concrete_floor_mat"},
                camera_rules,
            )
        )

    def test_ground_flat_camera_caps_high_elevation_without_changing_tables(self) -> None:
        camera_rules = self.base_rules["camera_observation"]
        expected_motion = {}
        ground = {"scene_class": "ground_flat"}
        raised = {"scene_class": "raised_flat"}

        ground_observation = compile_camera_observation(
            camera_rules, "projectile_1obj", expected_motion, ground
        )
        raised_observation = compile_camera_observation(
            camera_rules, "projectile_1obj", expected_motion, raised
        )

        self.assertEqual(
            ground_observation["elevation_range_degrees"], [22.0, 28.0]
        )
        self.assertEqual(
            raised_observation["elevation_range_degrees"], [22.0, 44.0]
        )

    def test_reviewed_environment_rejects_unreadable_ramp_camera_axis(self) -> None:
        camera_rules = {
            "minimum_inclined_surface_side_readability": 0.65,
            "motion_intents": {
                "slope_slide_down_1obj": {
                    "structure_context": "inclined_surface",
                    "elevation_range_degrees": [18.0, 42.0],
                }
            },
        }
        profile = {
            "visual_type": "mesh_backdrop",
            "camera_context": {
                "minimum_elevation_degrees": 18.0,
                "maximum_elevation_degrees": 30.0,
            },
            "composition": {
                "review_status": "approved",
                "camera": {"maximum_local_azimuth_deviation_degrees": 12.0},
                "bindings": [
                    {
                        "support_ids": ["ground_ramp_short_steep"],
                        "motion_families": ["slope_slide_down_1obj"],
                    }
                ],
            },
        }
        support = {"label": "ground_ramp_short_steep"}
        top_oblique = {
            "overrides": {"view_rule": {"azimuth_degrees": -75.0}}
        }
        left_oblique = {
            "overrides": {"view_rule": {"azimuth_degrees": 145.0}}
        }
        self.assertFalse(
            scene_visual_profile_admits_camera(
                profile,
                "slope_slide_down_1obj",
                "slow_down",
                "long",
                support,
                camera_rules,
                camera_profile=top_oblique,
            )
        )
        self.assertTrue(
            scene_visual_profile_admits_camera(
                profile,
                "slope_slide_down_1obj",
                "slow_down",
                "long",
                support,
                camera_rules,
                camera_profile=left_oblique,
            )
        )

    def test_partial_exit_policy_still_keeps_most_primary_motion_visible(self) -> None:
        backend = json.loads((ROOT / self.bundle["backend"]).read_text(encoding="utf-8"))
        quality = backend["quality"]
        self.assertTrue(quality["allow_exit_after_primary_motion"])
        self.assertEqual(
            quality["minimum_primary_trajectory_center_visible_fraction"], 0.75
        )
        self.assertEqual(
            quality["minimum_full_trajectory_center_visible_fraction"], 0.50
        )
        self.assertEqual(quality["minimum_initial_object_visible_fraction"], 1.0)

    def test_active_bundle_declares_every_sampling_dependency(self) -> None:
        required = {
            "base_rules",
            "object_profiles",
            "object_visual_preflight",
            "object_visual_preflight_report",
            "object_visual_curation",
            "object_visual_repairs",
            "scene_kits",
            "scene_visual_profiles",
            "scene_mesh_profiles",
            "environment_composition",
            "support_mesh_profiles",
            "asset_proxy_registry",
            "physical_proxy_catalog",
            "compatibility",
            "visual_sampling",
            "backend",
            "backend_capabilities",
            "material_manifest",
            "hdri_manifest",
            "environment_collision_proxies",
        }
        self.assertTrue(required <= set(self.bundle))
        for key in required:
            self.assertTrue((ROOT / self.bundle[key]).is_file(), key)
        required_implementation = {
            "sampler",
            "compiler",
            "geometry",
            "inclined_support",
            "support_structure",
            "time_step",
            "simulator",
            "trajectory_audit",
            "proxy_catalog",
            "static_support_binding",
            "physics_invariants",
            "batch_runner",
            "visual_binder",
            "rigid_renderer",
            "visual_preflight",
            "visual_repair",
            "visual_curation",
            "camera_geometry",
            "environment_collision",
            "motion_rule_package",
            "motion_rule_contracts",
            "motion_rule_common",
            "motion_rule_registry",
            "motion_rule_planar",
            "motion_rule_ballistic",
            "motion_rule_incline",
            "motion_rule_transition",
        }
        self.assertEqual(set(self.bundle["implementation"]), required_implementation)
        for path in self.bundle["implementation"].values():
            self.assertTrue((ROOT / path).is_file(), path)

    def test_active_bundle_preserves_matrix_cardinality(self) -> None:
        self.assertEqual(len(self.rules["axes"]["motion_axis"]), 11)
        active_profile_count = int(
            self.object_visual_curation["counts"]["active_profiles"]
        )
        self.assertEqual(len(self.rules["axes"]["object_axis"]), active_profile_count)
        self.assertTrue(
            all(
                variant["type"] == "mesh"
                for obj in self.rules["axes"]["object_axis"]
                for variant in obj["visual_variants"]
            )
        )

    def test_every_visual_profile_is_curated_and_compiled(self) -> None:
        curated = {
            str(record["profile_id"])
            for record in self.object_visual_curation["records"]
        }
        active = {str(record["label"]) for record in self.rules["axes"]["object_axis"]}
        self.assertEqual(curated, active)
        self.assertEqual(len(active), self.object_visual_curation["counts"]["active_profiles"])
        preflight_assets = {
            str(record["visual_asset_id"])
            for record in self.object_visual_preflight["records"]
        }
        curated_assets = {
            str(visual["visual_asset_id"])
            for record in self.object_visual_curation["records"]
            for visual in record["visuals"]
        }
        self.assertEqual(preflight_assets, curated_assets)
        self.assertTrue(self.object_visual_preflight["complete_profile_set"])
        self.assertEqual(
            len(preflight_assets),
            self.object_visual_preflight["counts"]["visuals"],
        )
        for record in self.object_visual_preflight["records"]:
            self.assertEqual(len(record["review_views"]), 4)
            for view in record["review_views"]:
                self.assertTrue((ROOT / view["path"]).is_file())
            proxy_evidence = record["proxy_evidence"]
            self.assertEqual(proxy_evidence["status"], "verified")
            self.assertEqual(
                {view["view"] for view in proxy_evidence["overlay_views"]},
                {"front", "side", "top"},
            )
            self.assertTrue(
                (ROOT / proxy_evidence["source_proxy"]["path"]).is_file()
            )
            for view in proxy_evidence["overlay_views"]:
                self.assertTrue((ROOT / view["path"]).is_file())
        repaired = [
            visual
            for record in self.object_visual_curation["records"]
            for visual in record["visuals"]
            if visual["status"] == "repaired_verified"
        ]
        self.assertEqual(len(repaired), 4)
        for visual in repaired:
            self.assertNotEqual(visual["source_visual"], visual["admitted_visual"])
            self.assertTrue((ROOT / visual["admitted_visual"]["path"]).is_file())
            self.assertEqual(
                visual["verification"]["source_preflight"]["disposition"],
                "repair_required",
            )
        source_verified = [
            visual
            for record in self.object_visual_curation["records"]
            for visual in record["visuals"]
            if visual["status"] == "source_verified"
        ]
        self.assertTrue(source_verified)
        self.assertTrue(
            all(
                visual["verification"]["source_preflight"]["disposition"]
                == "source_verified"
                and not visual["verification"]["source_preflight"]["issues"]
                and not visual["verification"]["source_preflight"][
                    "visibility_issues"
                ]
                for visual in source_verified
            )
        )
        self.assertEqual(
            self.object_visual_preflight["counts"]["reviewed_transparency"], 3
        )
        self.assertEqual(
            self.object_visual_preflight["counts"]["manual_review_required"], 0
        )
        self.assertEqual(
            self.object_visual_preflight["counts"]["proxy_evidence_verified"],
            len(preflight_assets),
        )
        self.assertEqual(
            self.preflight_policy["policy"]["minimum_review_subject_fraction"],
            0.02,
        )

    def test_visual_curation_rejects_an_unresolved_preflight_finding(self) -> None:
        report = copy.deepcopy(self.object_visual_preflight)
        curation = copy.deepcopy(self.object_visual_curation)
        target = next(
            record
            for record in report["records"]
            if record["disposition"] == "source_verified"
            and record["reviewed_transparency"] is None
        )
        target["disposition"] = "manual_review_required"
        target["visibility_issues"] = ["subject_not_visible:front"]
        curated = next(
            visual
            for record in curation["records"]
            for visual in record["visuals"]
            if visual["visual_asset_id"] == target["visual_asset_id"]
        )
        curated["verification"]["source_preflight"]["disposition"] = (
            "manual_review_required"
        )
        curated["verification"]["source_preflight"]["visibility_issues"] = [
            "subject_not_visible:front"
        ]
        with self.assertRaisesRegex(ValueError, "unrepaired finding"):
            validate_object_visual_curation(
                ROOT,
                self.object_profiles_path,
                self.object_profiles,
                self.preflight_policy_path,
                self.preflight_report_path,
                report,
                curation,
            )

    def test_visual_curation_rejects_tampered_proxy_runtime_evidence(self) -> None:
        report = copy.deepcopy(self.object_visual_preflight)
        curation = copy.deepcopy(self.object_visual_curation)
        target = report["records"][0]
        target["proxy_evidence"]["runtime_probe"]["record_sha256"] = "0" * 64
        curated = next(
            visual
            for record in curation["records"]
            for visual in record["visuals"]
            if visual["visual_asset_id"] == target["visual_asset_id"]
        )
        curated["verification"]["source_preflight"]["proxy_evidence"] = copy.deepcopy(
            target["proxy_evidence"]
        )
        with self.assertRaisesRegex(ValueError, "runtime probe"):
            validate_object_visual_curation(
                ROOT,
                self.object_profiles_path,
                self.object_profiles,
                self.preflight_policy_path,
                self.preflight_report_path,
                report,
                curation,
            )

    def test_every_generic_motion_has_a_useful_duration_contract(self) -> None:
        configured = set(
            self.backend["quality"]["minimum_active_duration_s_by_motion"]
        )
        motions = set(self.base_rules["axes"]["motion_axis"])
        self.assertEqual(configured, motions)
        self.assertTrue(
            all(
                float(value) > 0.0
                for value in self.backend["quality"][
                    "minimum_active_duration_s_by_motion"
                ].values()
            )
        )

    def test_base_rules_do_not_duplicate_compiled_profile_axes(self) -> None:
        forbidden = {
            "object_axis",
            "support_axis",
            "visual_surface_material_by_family",
            "wall_material_by_family",
            "surface_value_by_family",
            "object_material_by_value",
            "object_material_by_hint",
            "support_material_by_type",
            "wall_material_axis",
        }
        self.assertTrue(forbidden.isdisjoint(self.base_rules["axes"]))
        self.assertNotIn("sampling_rules", self.base_rules)

    def test_visual_material_rules_have_one_owner(self) -> None:
        self.assertTrue(self.visual_rules["surface_material_pools_by_family"])
        self.assertTrue(self.visual_rules["fallback_object_material_pools_by_value"])
        self.assertTrue(self.visual_rules["wall_fallback_pool"])

    def test_wall_material_exclusions_are_absent_from_every_wall_pool(self) -> None:
        excluded = set(self.visual_rules["wall_material_exclusions"])
        pools = [self.visual_rules["wall_fallback_pool"]]
        pools.extend(self.visual_rules["wall_pools_by_environment"].values())
        pools.extend(self.visual_rules["wall_primary_pools_by_theme"].values())
        pools.extend(self.visual_rules["wall_accent_pools_by_theme"].values())
        self.assertTrue(excluded)
        self.assertTrue(all(excluded.isdisjoint(pool) for pool in pools))

    def test_dynamic_object_material_pools_use_only_object_admitted_grades(self) -> None:
        assets = {
            str(record["asset_id"]): record
            for record in self.material_manifest["assets"]
        }
        allowed = set(
            self.visual_rules["material_role_admission"]["dynamic_object"][
                "allowed_object_grades"
            ]
        )
        pools = [*self.visual_rules["object_material_pools"].values()]
        pools.extend(self.visual_rules["fallback_object_material_pools_by_value"].values())
        for pool in pools:
            for asset_id in pool:
                with self.subTest(asset_id=asset_id):
                    self.assertIn(assets[asset_id]["object_grade"], allowed)

    def test_generic_capabilities_are_runtime_validated(self) -> None:
        validate_generic_capabilities(
            self.rules, self.backend, self.backend_capabilities
        )
        invalid = copy.deepcopy(self.backend_capabilities)
        invalid["generic_base_scope"]["motions"].pop()
        with self.assertRaises(ValueError):
            validate_generic_capabilities(self.rules, self.backend, invalid)

    def test_extent_limit_is_owned_by_compatibility_rules(self) -> None:
        axes = self.rules["axes"]
        long_extent = next(
            record
            for record in axes["trajectory_extent_axis"]
            if record["label"] == "long"
        )
        limited = constrained_trajectory_extent(
            "slope_slide_up_1obj", long_extent, axes, self.rules
        )
        unchanged = constrained_trajectory_extent(
            "slide_push_1obj", long_extent, axes, self.rules
        )
        self.assertEqual(limited["label"], "medium")
        self.assertEqual(unchanged["label"], "long")

    def test_backend_contains_only_executable_parameter_rules(self) -> None:
        self.assertEqual(
            self.backend["scope"],
            "generic_single_body_and_specialized_rigid_scenes",
        )
        self.assertNotIn("simulation_hz", self.backend["engine"])
        self.assertGreaterEqual(
            self.backend["engine"]["minimum_simulation_hz"],
            self.backend["engine"]["output_fps"],
        )
        pseudo_rules = {
            "mass_sampling",
            "friction_sampling",
            "slope_friction_limit",
            "slide_speed_rule",
            "rolling_speed_rule",
            "projectile_speed_rule",
            "slope_up_speed_rule",
        }
        self.assertTrue(
            pseudo_rules.isdisjoint(self.backend["base_parameter_rules"])
        )

    def test_angular_speed_is_below_video_sampling_nyquist(self) -> None:
        output_fps = float(self.backend["engine"]["output_fps"])
        conservative_limit = 0.8 * math.pi * output_fps
        self.assertLessEqual(
            float(self.backend["quality"]["maximum_angular_speed_rad_s"]),
            conservative_limit,
        )
        self.assertLessEqual(
            float(
                self.backend["asset_proxy_rules"]["quality"][
                    "maximum_angular_speed_rad_s"
                ]
            ),
            conservative_limit,
        )

    def test_object_contract_separates_visual_and_collision(self) -> None:
        for profile in self.objects:
            with self.subTest(profile=profile["id"]):
                self.assertIn("visual_variants", profile)
                self.assertIn("collision", profile)
                self.assertIn("physics", profile)
                compiled = compile_object_profile(profile)
                self.assertEqual(compiled["shape"], profile["collision"]["type"])
                self.assertTrue(compiled["visual_variants"])
                for visual in compiled["visual_variants"]:
                    if visual["type"] == "mesh":
                        self.assertIn(
                            visual["alignment_coordinate_frame"],
                            {"raw_gltf_z_up", "blender_imported_z_up"},
                        )

    def test_real_profiles_share_one_proxy_across_visual_variants(self) -> None:
        mesh_profiles = [
            profile
            for profile in self.objects
            if any(variant["type"] == "mesh" for variant in profile["visual_variants"])
        ]
        self.assertTrue(mesh_profiles)
        for profile in mesh_profiles:
            self.assertIn("collision", profile)
            self.assertTrue(
                any(
                    variant["type"] == "mesh"
                    for variant in profile["visual_variants"]
                )
            )
            self.assertTrue(
                all("collision" not in variant for variant in profile["visual_variants"])
            )

    def test_scene_topologies_are_registered(self) -> None:
        for kit in self.kits:
            with self.subTest(kit=kit["id"]):
                self.assertIn(kit["topology"], SUPPORT_BUILDERS)
                if kit.get("sampling_enabled", True):
                    self.assertIn("topology", compile_scene_kit(kit))

    def test_long_shallow_ramp_geometry_variants_compile_to_one_semantic_support(self) -> None:
        kit = next(
            item for item in self.kits if item["id"] == "ground_ramp_long_shallow"
        )
        compiled = compile_scene_kit(kit)
        variants = compiled["geometry_variants"]
        self.assertEqual(
            {variant["id"] for variant in variants},
            {"standard", "extended_landing", "wide_gentle"},
        )
        self.assertTrue(all(variant["size"][1] >= 1.75 for variant in variants))
        self.assertTrue(
            all(
                variant["overrides"]["placement"]["anchor_low_edge_to_floor"]
                for variant in variants
            )
        )

    def test_each_motion_has_admitted_scene_kits(self) -> None:
        supports = self.rules["axes"]["support_axis"]
        for motion in self.rules["axes"]["motion_axis"]:
            admitted = [
                support
                for support in supports
                if support_allowed(support, motion, self.rules)
            ]
            self.assertTrue(admitted, motion)

    def test_pool_table_is_catalogued_but_not_silently_sampled(self) -> None:
        pool = next(kit for kit in self.kits if kit["id"] == "pool_table_standard")
        self.assertEqual(pool["visual"]["type"], "mesh")
        self.assertFalse(pool["sampling_enabled"])
        active = {item["label"] for item in self.rules["axes"]["support_axis"]}
        self.assertNotIn(pool["id"], active)

    def test_long_ground_scene_kits_compile_explicit_constraints(self) -> None:
        expected = {
            "indoor_long_floor": None,
            "open_hardscape": None,
            "long_corridor": "x",
        }
        for kit_id, motion_axis in expected.items():
            kit = next(item for item in self.kits if item["id"] == kit_id)
            compiled = compile_scene_kit(kit)
            placement = compiled["overrides"]["placement"]
            with self.subTest(kit=kit_id):
                self.assertTrue(placement["ground_surface"])
                self.assertGreater(
                    placement["maximum_planar_trajectory_distance_m"], 1.65
                )
                self.assertTrue(compiled["allowed_motions"])
                self.assertTrue(compiled["environment_categories"])
                self.assertEqual(placement.get("motion_axis"), motion_axis)

    def test_long_table_scene_kits_compile_explicit_constraints(self) -> None:
        expected = {
            "long_wood_table": 2.25,
            "long_lab_bench": 2.2,
            "long_kitchen_counter": 2.1,
        }
        for kit_id, maximum_distance in expected.items():
            kit = next(item for item in self.kits if item["id"] == kit_id)
            compiled = compile_scene_kit(kit)
            placement = compiled["overrides"]["placement"]
            with self.subTest(kit=kit_id):
                self.assertFalse(placement["ground_surface"])
                self.assertEqual(placement["motion_axis"], "x")
                self.assertEqual(
                    placement["maximum_planar_trajectory_distance_m"],
                    maximum_distance,
                )
                self.assertIn("edge_fall_1obj", compiled["allowed_motions"])
                self.assertTrue(compiled["environment_categories"])

    def test_edge_fall_admits_a_real_low_platform_but_not_a_ramp(self) -> None:
        supports = {
            item["label"]: item for item in self.rules["axes"]["support_axis"]
        }
        self.assertTrue(
            support_allowed(
                supports["low_pedestal"], "edge_fall_1obj", self.rules
            )
        )
        self.assertFalse(
            support_allowed(
                supports["raised_ramp_standard"],
                "edge_fall_1obj",
                self.rules,
            )
        )

    def test_scene_visual_profiles_pair_structure_and_reach_themes(self) -> None:
        self.assertEqual(
            self.scene_visuals["policy"]["role"],
            "procedural_environment_template",
        )
        self.assertTrue(
            self.scene_visuals["policy"]["visual_and_collision_are_paired"]
        )
        self.assertTrue(
            self.scene_visuals["policy"]["never_changes_declared_motion"]
        )
        active_themes = {
            str(kit["theme"])
            for kit in self.kits
            if bool(kit.get("sampling_enabled", True))
        }
        admitted_themes = {
            str(theme)
            for profile in self.scene_visuals["profiles"]
            for theme in profile["themes"]
        }
        self.assertTrue(active_themes <= admitted_themes)
        for profile in self.scene_visuals["profiles"]:
            self.assertIn("decor", profile)
            for decor in profile["decor"]:
                self.assertEqual(len(decor["size_m"]), 3)
                self.assertEqual(len(decor["offset_lateral_depth_z"]), 3)
            for piece in profile.get("set_pieces", []):
                self.assertEqual(len(piece["size_m"]), 3)
                self.assertEqual(len(piece["offset_lateral_outward_z"]), 3)
                self.assertLess(piece["offset_lateral_outward_z"][1], 0.0)

    def test_large_environment_profiles_keep_a_clear_motion_lane(self) -> None:
        expected = {
            "warehouse_long_bay",
            "garage_service_lane",
            "office_long_room",
            "outdoor_low_wall",
        }
        profiles = {
            str(profile["id"]): profile
            for profile in self.scene_visuals["profiles"]
        }
        self.assertTrue(expected <= set(profiles))
        contextual_profiles = {
            "warehouse_long_bay",
            "garage_service_lane",
            "office_long_room",
        }
        for profile_id in expected:
            profile = profiles[profile_id]
            clear_lane = float(profile["clear_lane_half_width_m"])
            with self.subTest(profile=profile_id):
                self.assertEqual(profile["layout_family"], "large_clear_lane")
                self.assertGreaterEqual(profile["back_wall_distance_m"], 3.4)
                for piece in profile["set_pieces"]:
                    lateral = abs(piece["offset_lateral_outward_z"][0])
                    half_width = piece["size_m"][0] / 2.0
                    self.assertGreaterEqual(lateral - half_width, clear_lane)
                if profile_id in contextual_profiles:
                    context = profile["camera_context"]
                    self.assertGreaterEqual(context["depth_offset_m"], 0.4)
                    self.assertLessEqual(context["focal_length_cap_mm"], 38.0)
                    self.assertLess(
                        context["minimum_elevation_degrees"],
                        context["maximum_elevation_degrees"],
                    )

    def test_scene_mesh_profiles_are_audited_render_only_assets(self) -> None:
        self.assertEqual(
            self.scene_mesh_visuals["sampling"]["target_mesh_fraction"], 0.4
        )
        policy = self.scene_mesh_visuals["policy"]
        self.assertTrue(policy["visual_and_collision_are_paired"])
        self.assertTrue(policy["environment_proxy_is_always_loaded"])
        self.assertEqual(
            policy["collision_authority"], "frozen_static_environment_proxy"
        )
        active_themes = {
            str(kit["theme"])
            for kit in self.kits
            if bool(kit.get("sampling_enabled", True))
        }
        mesh_themes = {
            str(theme)
            for profile in self.scene_mesh_visuals["profiles"]
            for theme in profile["themes"]
        }
        self.assertTrue(active_themes <= mesh_themes)
        for profile in self.scene_mesh_visuals["profiles"]:
            self.assertEqual(profile["visual_type"], "mesh_backdrop")
            self.assertTrue(profile["scene_classes"])
            asset = profile["asset"]
            self.assertEqual(len(asset["sha256"]), 64)
            self.assertEqual(len(asset["source_bbox_size"]), 3)
            self.assertGreater(asset["target_extent_m"], 0.0)
            self.assertIn(asset["normalization_axis"], {"x", "y", "z"})

    def test_support_mesh_profiles_are_calibrated_exact_supports(self) -> None:
        policy = self.support_mesh_visuals["policy"]
        self.assertEqual(
            policy["collision_authority"],
            "exact_static_proxy_when_selected",
        )
        self.assertTrue(policy["immutable_binding_required"])
        self.assertEqual(
            policy["fallback_phase"], "before_metadata_freeze_only"
        )
        self.assertLessEqual(policy["maximum_axis_scale_ratio"], 1.9)
        active_supports = {
            str(item["label"]): item for item in self.rules["axes"]["support_axis"]
        }
        admitted = set()
        for profile in self.support_mesh_visuals["profiles"]:
            self.assertEqual(profile["visual_type"], "mesh_support")
            self.assertEqual(len(profile["sha256"]), 64)
            self.assertEqual(len(profile["source_bbox_size"]), 3)
            self.assertGreater(profile["source_support_plane_z_from_bottom"], 0.0)
            self.assertIn(
                profile["material_policy"],
                {"embedded_pbr", "support_surface_pbr_override"},
            )
            for support_id in profile["support_ids"]:
                admitted.add(support_id)
                self.assertEqual(
                    active_supports[support_id]["topology"], "flat_surface"
                )
        self.assertEqual(
            admitted, {"wood_tabletop", "lab_bench", "kitchen_counter"}
        )
        rejected = {
            record["asset_id"]
            for record in self.support_mesh_visuals["rejected_candidates"]
        }
        self.assertIn(
            "sketchfab_bg_9634d6a81a494925a909512c8048f446", rejected
        )

    def test_asset_proxy_registry_covers_every_curated_sketchfab_asset(self) -> None:
        records = self.asset_proxy_registry["records"]
        self.assertEqual(
            len(records), self.asset_proxy_registry["counts"]["total"]
        )
        self.assertEqual(len({record["asset_id"] for record in records}), len(records))

    def test_every_admitted_asset_has_an_explicit_physical_proxy(self) -> None:
        admitted = [
            record
            for record in self.asset_proxy_registry["records"]
            if record["admission"].get("sampling_enabled", False)
        ]
        self.assertEqual(
            len(admitted), self.asset_proxy_registry["counts"]["sampling_enabled"]
        )
        for record in admitted:
            with self.subTest(asset_id=record["asset_id"]):
                self.assertNotEqual(record["proxy"]["kind"], "none")
                self.assertTrue(record["proxy"]["colliders"])

    def test_rejected_support_assets_cannot_leak_into_sampling(self) -> None:
        for record in self.asset_proxy_registry["records"]:
            if record["asset_role"] != "interactive_support":
                continue
            if record["admission"]["status"] == "rejected":
                self.assertFalse(record["admission"]["sampling_enabled"])
                self.assertEqual(record["proxy"]["kind"], "none")

    def test_registry_summary_is_runtime_validated(self) -> None:
        validate_registry_counts(self.asset_proxy_registry)
        invalid = copy.deepcopy(self.asset_proxy_registry)
        invalid["counts"]["total"] += 1
        with self.assertRaises(ValueError):
            validate_registry_counts(invalid)

    def test_game_table_requires_specialized_billiards_semantics(self) -> None:
        self.assertEqual(
            set(self.asset_semantic_rules),
            {"schema_version", "generic_one_object", "specialized_scene_families"},
        )
        generic = self.asset_semantic_rules["generic_one_object"]
        self.assertEqual(set(generic), {"excluded_support_categories"})
        self.assertIn("support_game_table", generic["excluded_support_categories"])
        single = self.asset_semantic_rules["specialized_scene_families"][
            "billiards_single_ball"
        ]
        self.assertEqual(single["support_category"], "support_game_table")
        self.assertEqual(single["dynamic_object_count"], 1)
        self.assertEqual(single["dynamic_semantics"], ["cue_ball"])
        self.assertEqual(
            set(single["profiles"]),
            {"single_ball_free_roll", "single_ball_rail_rebound"},
        )
        self.assertFalse(single["pocket_sink_supported"])
        billiards = self.asset_semantic_rules["specialized_scene_families"][
            "billiards_collision"
        ]
        self.assertEqual(billiards["support_category"], "support_game_table")
        self.assertEqual(billiards["dynamic_object_count"], 3)
        self.assertEqual(
            billiards["dynamic_semantics"],
            ["cue_ball", "object_ball_1", "object_ball_2"],
        )
        self.assertFalse(billiards["pocket_sink_supported"])
        self.assertEqual(billiards["profiles"], ["three_ball_collision"])

    def test_specialized_physics_values_have_one_config_owner(self) -> None:
        asset_profiles = set(self.backend["asset_proxy_rules"]["motion_profiles"])
        self.assertEqual(
            asset_profiles,
            {
                "vertical_drop",
                "resting_push",
                "diagonal_push",
                "edge_exit",
                "workbench_clear_zone_drop",
                "workbench_long_axis_push",
            },
        )
        specialized_families = self.asset_semantic_rules[
            "specialized_scene_families"
        ]
        semantic_billiards_profiles = {
            profile
            for family_id in ("billiards_single_ball", "billiards_collision")
            for profile in specialized_families[family_id]["profiles"]
        }
        semantic_pinball_profiles = set(
            specialized_families["passive_pinball_single_ball"]["profiles"]
        )
        semantic_marble_profiles = set(
            specialized_families["marble_run_single_ball"]["profiles"]
        )
        configured_pinball_profiles = set(
            self.passive_pinball_backend["profiles"]
        )
        configured_marble_profiles = set(self.marble_run_backend["profiles"])
        self.assertTrue(
            semantic_billiards_profiles.isdisjoint(semantic_pinball_profiles)
        )
        self.assertEqual(
            set(self.backend["billiards_rules"]["initial_states"]),
            semantic_billiards_profiles,
        )
        self.assertEqual(
            configured_pinball_profiles,
            semantic_pinball_profiles,
        )
        self.assertTrue(
            semantic_billiards_profiles.isdisjoint(semantic_marble_profiles)
        )
        self.assertTrue(
            semantic_pinball_profiles.isdisjoint(semantic_marble_profiles)
        )
        self.assertEqual(configured_marble_profiles, semantic_marble_profiles)

    def test_every_non_dynamic_asset_has_one_composition_decision(self) -> None:
        expected = {
            record["asset_id"]
            for record in self.asset_proxy_registry["records"]
            if record["asset_role"]
            not in {"dynamic_object", "static_environment", "rejected"}
        }
        records = self.asset_scene_composition["records"]
        actual = [record["asset_id"] for record in records]
        self.assertEqual(len(actual), len(set(actual)))
        self.assertEqual(set(actual), expected)

    def test_generic_asset_sampling_uses_only_reviewed_compositions(self) -> None:
        decisions = {
            record["asset_id"]: record
            for record in self.asset_scene_composition["records"]
        }
        for record in self.asset_proxy_registry["records"]:
            if record["asset_role"] in {
                "dynamic_object",
                "static_environment",
                "rejected",
            }:
                continue
            decision = decisions.get(record["asset_id"])
            self.assertIsNotNone(decision, record["asset_id"])
            if decision["sampling_status"] == "ready_generic":
                self.assertEqual(record["proxy"]["kind"], "support_compound")
                self.assertTrue(record["admission"]["sampling_enabled"])
            if decision["sampling_status"] == "ready_static":
                self.assertEqual(record["proxy"]["kind"], "static_compound")
                self.assertTrue(record["admission"]["sampling_enabled"])

    def test_ready_split_assets_render_only_declared_components(self) -> None:
        decisions = {
            record["asset_id"]: record
            for record in self.asset_scene_composition["records"]
        }
        for record in self.asset_proxy_registry["records"]:
            decision = decisions.get(record["asset_id"])
            if not decision or decision["sampling_status"] not in {
                "ready_generic",
                "ready_specialized",
            }:
                continue
            policy = decision["component_policy"]
            if policy.get("mode") != "exact_partition":
                continue
            rendered = policy.get("rendered_objects", [])
            excluded = policy.get("excluded_objects", [])
            self.assertTrue(rendered)
            self.assertFalse(set(rendered) & set(excluded))
            self.assertEqual(record["visual"].get("include_object_names"), rendered)
            self.assertEqual(
                set(rendered) | set(excluded),
                set(policy["support_objects"]) | set(policy["separate_context_objects"]),
            )

    def test_component_variant_props_bind_one_declared_mesh(self) -> None:
        decisions = {
            record["asset_id"]: record
            for record in self.asset_scene_composition["records"]
        }
        for record in self.asset_proxy_registry["records"]:
            variants = record["visual"].get("variant_object_names", [])
            if not variants:
                continue
            decision = decisions[record["asset_id"]]
            policy = decision["component_policy"]
            self.assertEqual(record["proxy"]["kind"], "static_compound")
            self.assertTrue(record["admission"]["sampling_enabled"])
            self.assertEqual(decision["sampling_status"], "ready_static")
            self.assertEqual(policy["mode"], "variant_select_one")
            self.assertEqual(policy["rendered_object_count"], 1)
            self.assertEqual(policy["variant_objects"], variants)

    def test_every_non_dynamic_asset_has_a_final_v1_disposition(self) -> None:
        registry = {
            record["asset_id"]: record for record in self.asset_proxy_registry["records"]
        }
        expected_roles = {
            "ready_generic": "interactive_support",
            "ready_specialized": "interactive_support",
            "ready_static": "static_prop",
            "ready_context": "render_only_context",
            "ready_context_only": "render_only_context",
        }
        for decision in self.asset_scene_composition["records"]:
            status = decision["sampling_status"]
            self.assertFalse(status.startswith("pending"), decision["asset_id"])
            if status in expected_roles:
                self.assertEqual(
                    registry[decision["asset_id"]]["asset_role"],
                    expected_roles[status],
                )

    def test_every_enabled_proxy_has_passed_physics_review(self) -> None:
        for record in self.asset_proxy_registry["records"]:
            if not record["admission"]["sampling_enabled"]:
                continue
            self.assertRegex(
                record["review"]["physics_status"],
                r"^(?:(?:exact_mesh_)?probe_passed_\d{4}_\d{2}_\d{2}|exact_static_mesh_probe_required_v2)$",
                record["asset_id"],
            )

    def test_active_support_calibration_matches_source_annotations(self) -> None:
        annotations = {
            record["asset_id"]: record
            for record in self.asset_proxy_annotations["records"]
        }
        visual_keys = (
            "alignment_euler_degrees",
            "include_object_names",
            "source_bbox_size",
            "source_support_bounds_xy",
            "source_support_plane_z_from_bottom",
            "target_support_size_xy_m",
        )
        for record in self.asset_proxy_registry["records"]:
            if not record["admission"]["sampling_enabled"]:
                continue
            if record["proxy"]["kind"] != "support_compound":
                continue
            source = annotations.get(record["asset_id"])
            if source is None:
                self.assertEqual(
                    record["review"]["physics_status"],
                    "exact_static_mesh_probe_required_v2",
                )
                continue
            for key in visual_keys:
                self.assertEqual(
                    record["visual"].get(key),
                    source["visual"].get(key),
                    f"{record['asset_id']} visual.{key}",
                )
            self.assertEqual(
                record["proxy"]["usable_surfaces"],
                source["proxy"]["usable_surfaces"],
                f"{record['asset_id']} usable_surfaces",
            )


if __name__ == "__main__":
    unittest.main()
