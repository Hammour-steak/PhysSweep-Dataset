from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"

from tools.sampling.sample_pybullet_base import (  # noqa: E402
    balanced_objects_for_motions,
    bounce_observation_contract,
    coverage_cycle_by_group,
    build_batch,
    derive_initial_condition,
    direction_for_support,
    load_active_rules,
    load_json,
    manifest_counts,
    object_supports_motion,
    restitution_for_motion,
    support_mesh_scale_ratio,
    support_allowed,
)
from tools.rendering.bind_pybullet_visuals import frozen_environment_binding  # noqa: E402
from tools.core.rigid_geometry import build_support_geometry  # noqa: E402


class PyBulletSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = load_active_rules(ROOT)
        cls.backend = load_json(ROOT / "configs/pybullet_backend.json")
        cls.visual = load_json(ROOT / "configs/visual_sampling.json")
        material_manifest = load_json(
            ROOT / "assets/manifests/polyhaven_render_library.json"
        )
        hdri_manifest = load_json(ROOT / "assets/manifests/hdri_admission.json")
        cls.materials = {
            str(record["asset_id"]): record for record in material_manifest["assets"]
        }
        cls.hdri_records = list(hdri_manifest["records"])
        cls.coverage_count = (
            len(cls.rules["axes"]["motion_axis"])
            * len(cls.rules["axes"]["object_axis"])
        )
        cls.scenes = build_batch(
            cls.rules,
            cls.backend,
            cls.materials,
            cls.hdri_records,
            cls.visual,
            20260725,
            cls.coverage_count,
        )

    def test_seeded_sampling_is_reproducible(self) -> None:
        repeat = build_batch(
            self.rules,
            self.backend,
            self.materials,
            self.hdri_records,
            self.visual,
            20260725,
            self.coverage_count,
        )
        self.assertEqual(
            json.dumps(self.scenes, sort_keys=True),
            json.dumps(repeat, sort_keys=True),
        )

    def test_long_shallow_ramp_variants_are_balanced_and_frozen(self) -> None:
        count = 6
        scenes = build_batch(
            self.rules,
            self.backend,
            self.materials,
            self.hdri_records,
            self.visual,
            20260823,
            count,
            motion_sequence=["ramp_to_flat_1obj"] * count,
            support_sequence=["ground_ramp_long_shallow"] * count,
        )
        variant_ids = [
            scene["simulation"]["support"]["geometry_variant_id"]
            for scene in scenes
        ]
        self.assertEqual(
            set(variant_ids), {"standard", "extended_landing", "wide_gentle"}
        )
        self.assertTrue(all(variant_ids.count(value) == 2 for value in set(variant_ids)))
        for scene in scenes:
            semantic_id = scene["semantic_sampling"]["five_dimensions"][
                "support_interaction"
            ]["geometry_variant_id"]
            self.assertEqual(
                semantic_id,
                scene["simulation"]["support"]["geometry_variant_id"],
            )

    def test_long_corridor_constrains_direction_and_motion(self) -> None:
        corridor = next(
            support
            for support in self.rules["axes"]["support_axis"]
            if support["label"] == "long_corridor"
        )
        positive = direction_for_support(
            {"label": "diagonal", "angle_degrees": 70.0}, corridor
        )
        negative = direction_for_support(
            {"label": "diagonal", "angle_degrees": 130.0}, corridor
        )
        self.assertEqual(positive["angle_degrees"], 0.0)
        self.assertEqual(negative["angle_degrees"], 180.0)
        self.assertTrue(support_allowed(corridor, "slide_push_1obj", self.rules))
        self.assertFalse(support_allowed(corridor, "drop_fall_1obj", self.rules))

    def test_long_corridor_distance_reaches_declared_cap(self) -> None:
        corridor = next(
            support
            for support in self.rules["axes"]["support_axis"]
            if support["label"] == "long_corridor"
        )
        obj = next(
            record
            for record in self.rules["axes"]["object_axis"]
            if record["shape"] == "cuboid"
            and object_supports_motion(record, "slide_push_1obj")
        )
        subtype = self.rules["axes"]["motion_subtype_axis"]["slide_push_1obj"][
            -1
        ]
        extent = next(
            record
            for record in self.rules["axes"]["trajectory_extent_axis"]
            if record["label"] == "long"
        )
        zone = next(
            record
            for record in self.rules["axes"]["initial_position_zone_axis"]
            if record["label"] == "center_start"
        )
        direction = {"label": "support_axis_x_positive", "angle_degrees": 0.0}
        geometry = build_support_geometry(
            corridor, "slide_push_1obj", subtype, [1.0, 0.0, 0.0]
        )
        initial = derive_initial_condition(
            random.Random(11),
            self.backend,
            "slide_push_1obj",
            subtype,
            extent,
            zone,
            direction,
            obj["shape"],
            obj["size"],
            obj["pose_profile"],
            geometry,
            0.35,
            0.08,
        )
        self.assertEqual(
            initial["expected_motion"]["target_displacement_m"], 2.8
        )
        expected_friction = 2.45**2 / (2.0 * 9.81 * 2.8)
        self.assertAlmostEqual(
            initial["effective_contact_friction"], expected_friction, places=6
        )

    def test_long_support_does_not_expand_ballistic_distance(self) -> None:
        corridor = next(
            support
            for support in self.rules["axes"]["support_axis"]
            if support["label"] == "long_corridor"
        )
        obj = next(
            record
            for record in self.rules["axes"]["object_axis"]
            if record["shape"] == "cylinder"
        )
        subtype = self.rules["axes"]["motion_subtype_axis"]["projectile_1obj"][
            0
        ]
        extent = next(
            record
            for record in self.rules["axes"]["trajectory_extent_axis"]
            if record["label"] == "long"
        )
        zone = next(
            record
            for record in self.rules["axes"]["initial_position_zone_axis"]
            if record["label"] == "center_start"
        )
        direction = {"label": "support_axis_x_positive", "angle_degrees": 0.0}
        geometry = build_support_geometry(
            corridor, "projectile_1obj", subtype, [1.0, 0.0, 0.0]
        )
        initial = derive_initial_condition(
            random.Random(12),
            self.backend,
            "projectile_1obj",
            subtype,
            extent,
            zone,
            direction,
            obj["shape"],
            obj["size"],
            obj["pose_profile"],
            geometry,
            0.35,
            0.08,
        )
        self.assertEqual(
            initial["expected_motion"]["minimum_horizontal_displacement_m"],
            round(1.65 * 0.35, 6),
        )

    def test_long_table_constrains_motion_to_its_long_axis(self) -> None:
        support = next(
            record
            for record in self.rules["axes"]["support_axis"]
            if record["label"] == "long_wood_table"
        )
        direction = direction_for_support(
            {"label": "diagonal", "angle_degrees": 65.0}, support
        )
        self.assertEqual(direction["angle_degrees"], 0.0)
        self.assertTrue(support_allowed(support, "edge_fall_1obj", self.rules))
        self.assertFalse(
            support_allowed(support, "slope_slide_down_1obj", self.rules)
        )

    def test_object_motion_exclusions_are_enforced_for_explicit_sequences(self) -> None:
        objects = [
            {"label": "flat_only", "excluded_motion_families": ["slope"]},
            {"label": "all_motion"},
        ]
        selected = balanced_objects_for_motions(
            ["slope", "flat", "slope", "flat"], objects, random.Random(7)
        )
        self.assertEqual(selected[0]["label"], "all_motion")
        self.assertEqual(selected[2]["label"], "all_motion")
        self.assertIn("flat_only", {selected[1]["label"], selected[3]["label"]})

    def test_long_ramp_rejects_objects_that_cannot_remain_readable(self) -> None:
        remote = next(
            record
            for record in self.rules["axes"]["object_axis"]
            if record["label"] == "physassets_19920_remote"
        )
        self.assertLess(
            sorted(remote["size"], reverse=True)[1],
            self.rules["architecture"]["compatibility"]["motions"]
            ["ramp_to_flat_1obj"]["minimum_object_characteristic_extent_m"],
        )
        self.assertFalse(
            object_supports_motion(remote, "ramp_to_flat_1obj", self.rules)
        )
        self.assertTrue(object_supports_motion(remote, "slide_push_1obj", self.rules))

    def test_spherical_proxies_use_rolling_instead_of_slide_push(self) -> None:
        spheres = [
            obj for obj in self.rules["axes"]["object_axis"]
            if obj["shape"] == "sphere"
        ]
        self.assertGreater(len(spheres), 0)
        for obj in spheres:
            self.assertFalse(object_supports_motion(obj, "slide_push_1obj"))
            self.assertTrue(object_supports_motion(obj, "roll_or_slide_1obj"))

    def test_bounce_observation_scales_with_shape_and_object_height(self) -> None:
        short_box = bounce_observation_contract(
            self.backend, "cuboid", [0.10, 0.08, 0.10]
        )
        tall_box = bounce_observation_contract(
            self.backend, "cuboid", [0.10, 0.08, 0.25]
        )
        sphere = bounce_observation_contract(
            self.backend, "sphere", [0.12, 0.12, 0.12]
        )
        self.assertEqual(short_box["minimum_rebound_height_m"], 0.025)
        self.assertEqual(tall_box["minimum_rebound_height_m"], 0.04)
        self.assertLess(
            short_box["restitution_observation_ratio_range"][0],
            sphere["restitution_observation_ratio_range"][0],
        )

    def test_grouped_coverage_cycle_visits_every_value_per_group(self) -> None:
        groups = ["a"] * 6 + ["b"] * 6
        values = ["left", "right", "top"]
        selected = coverage_cycle_by_group(groups, values, random.Random(11))
        for group in {"a", "b"}:
            used = {
                value
                for current_group, value in zip(groups, selected)
                if current_group == group
            }
            self.assertEqual(used, set(values))

    def test_explicit_support_sequence_enables_structure_level_validation(self) -> None:
        supports = [
            str(record["label"])
            for record in self.rules["axes"]["support_axis"]
            if record.get("overrides", {})
            .get("placement", {})
            .get("support_shape") == "inclined_ramp"
        ]
        motions = ["slope_slide_down_1obj"] * len(supports)
        scenes = build_batch(
            self.rules,
            self.backend,
            self.materials,
            self.hdri_records,
            self.visual,
            20260726,
            len(supports),
            motion_sequence=motions,
            support_sequence=supports,
        )
        sampled = [
            scene["semantic_sampling"]["five_dimensions"]["support_interaction"][
                "support_type"
            ]
            for scene in scenes
        ]
        self.assertEqual(sampled, supports)

    def test_wall_impact_restitution_preserves_material_response(self) -> None:
        steel = restitution_for_motion(
            random.Random(7),
            self.backend,
            "wall_impact_1obj",
            0.72,
        )
        soft = restitution_for_motion(
            random.Random(7),
            self.backend,
            "wall_impact_1obj",
            0.08,
        )
        lower, upper = self.backend["base_parameter_rules"][
            "restitution_by_motion"
        ]["wall_impact_1obj"]
        self.assertGreaterEqual(steel, lower)
        self.assertLessEqual(steel, upper)
        self.assertEqual(soft, 0.08)

    def test_seeded_sampling_is_reproducible_across_hash_seeds(self) -> None:
        script = """
import hashlib, json
from pathlib import Path
from tools.sampling.sample_pybullet_base import build_batch, load_active_rules, load_json
root = Path.cwd()
rules = load_active_rules(root)
backend = load_json(root / 'configs/pybullet_backend.json')
visual = load_json(root / 'configs/visual_sampling.json')
materials = {str(x['asset_id']): x for x in load_json(root / 'assets/manifests/polyhaven_render_library.json')['assets']}
hdri = list(load_json(root / 'assets/manifests/hdri_admission.json')['records'])
payload = json.dumps(build_batch(rules, backend, materials, hdri, visual, 20260725, 20), sort_keys=True).encode()
print(hashlib.sha256(payload).hexdigest())
"""
        hashes = []
        for hash_seed in ("1", "2"):
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = hash_seed
            environment["PYTHONPATH"] = str(ROOT)
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            hashes.append(result.stdout.strip())
        self.assertEqual(hashes[0], hashes[1])

    def test_coverage_batch_visits_every_declared_major_axis_value(self) -> None:
        coverage = manifest_counts(self.scenes)
        axes = self.rules["axes"]
        self.assertEqual(set(coverage["motion"]), set(axes["motion_axis"]))
        self.assertEqual(
            set(coverage["object"]),
            {str(record["label"]) for record in axes["object_axis"]},
        )
        self.assertEqual(
            set(coverage["support"]),
            {str(record["label"]) for record in axes["support_axis"]},
        )
        self.assertEqual(
            set(coverage["scene_class"]),
            {str(record["label"]) for record in axes["scene_class_axis"]},
        )
        self.assertEqual(
            set(coverage["camera"]),
            {str(record["label"]) for record in axes["camera_axis"]},
        )
        self.assertEqual(
            set(coverage["surface_family"]),
            set(self.visual["surface_material_pools_by_family"]),
        )
        self.assertEqual(
            set(coverage["environment_category"]),
            set(self.visual["environment_categories"]),
        )
        category_counts = list(coverage["environment_category"].values())
        allowed_category_delta = max(2, math.ceil(0.02 * self.coverage_count))
        self.assertLessEqual(
            max(category_counts) - min(category_counts),
            allowed_category_delta,
        )
        self.assertEqual(coverage["visual_type"], {"mesh": self.coverage_count})
        mesh_backdrops = round(
            self.coverage_count
            * float(
                self.rules["architecture"]["scene_visual_profiles"]["sampling"][
                    "target_mesh_fraction"
                ]
            )
        )
        actual_mesh_backdrops = coverage["scene_visual_type"].get(
            "mesh_backdrop", 0
        )
        self.assertGreater(actual_mesh_backdrops, 0)
        self.assertLessEqual(actual_mesh_backdrops, mesh_backdrops)
        self.assertEqual(
            sum(coverage["scene_visual_type"].values()), self.coverage_count
        )
        approved_mesh_profiles = {
            str(profile["id"])
            for profile in self.rules["architecture"]["scene_visual_profiles"][
                "profiles"
            ]
            if profile["visual_type"] == "mesh_backdrop"
            and profile["composition"]["review_status"] == "approved"
        }
        sampled_mesh_profiles = {
            str(scene["appearance"]["scene_visual"]["id"])
            for scene in self.scenes
            if scene["appearance"]["scene_visual"]["visual_type"]
            == "mesh_backdrop"
        }
        self.assertTrue(sampled_mesh_profiles)
        self.assertTrue(sampled_mesh_profiles <= approved_mesh_profiles)
        self.assertEqual(
            set(coverage["support_visual_type"]),
            {"mesh_support", "procedural_proxy"},
        )
        self.assertEqual(
            set(coverage["support_visual"]),
            {
                "procedural_support_proxy",
                *{
                    str(profile["id"])
                    for profile in self.rules["architecture"][
                        "support_mesh_profiles"
                    ]["profiles"]
                },
            },
        )

    def test_environment_category_controls_background_materials_and_hdri(self) -> None:
        for scene in self.scenes:
            appearance = scene["appearance"]
            category = str(appearance["environment_category"])
            with self.subTest(scene=scene["scene_id"], category=category):
                self.assertEqual(
                    category,
                    scene["appearance"]["scene_visual"][
                        "environment_category"
                    ],
                )
                self.assertIn(
                    appearance["materials"]["room_floor"]["record"]["asset_id"],
                    self.visual["room_floor_pools_by_environment"][category],
                )
                self.assertIn(
                    appearance["materials"]["back_wall"]["record"]["asset_id"],
                    self.visual["wall_pools_by_environment"][category],
                )
                self.assertIn(
                    appearance["hdri"]["role"],
                    self.visual["hdri_roles_by_environment"][category],
                )

    def test_dynamic_object_materials_use_object_admitted_grades(self) -> None:
        allowed = set(
            self.visual["material_role_admission"]["dynamic_object"][
                "allowed_object_grades"
            ]
        )
        for scene in self.scenes:
            grade = scene["appearance"]["materials"]["dynamic_object"]["record"][
                "curation_grade"
            ]["object"]
            with self.subTest(scene=scene["scene_id"], grade=grade):
                self.assertIn(grade, allowed)

    def test_set_pieces_remain_outside_the_motion_core(self) -> None:
        for profile in self.rules["architecture"]["scene_visual_profiles"][
            "profiles"
        ]:
            if profile["visual_type"] != "procedural_room":
                continue
            for piece in profile.get("set_pieces", []):
                lateral = abs(float(piece["offset_lateral_outward_z"][0]))
                half_width = float(piece["size_m"][0]) / 2.0
                with self.subTest(profile=profile["id"], piece=piece["id"]):
                    self.assertGreaterEqual(lateral - half_width, 1.05)

    def test_outdoor_profiles_use_explicit_ground_or_raised_supports(self) -> None:
        for profile in self.rules["architecture"]["scene_visual_profiles"][
            "profiles"
        ]:
            if profile["environment_category"] != "outdoor_courtyard":
                continue
            with self.subTest(profile=profile["id"]):
                scene_classes = set(profile["scene_classes"])
                self.assertTrue(
                    scene_classes <= {
                        "ground_flat",
                        "ground_feature",
                        "raised_feature",
                    }
                )
                self.assertTrue(
                    {"ground_flat", "ground_feature"} <= scene_classes
                )
                if profile["visual_type"] == "procedural_room":
                    self.assertNotIn("raised_feature", scene_classes)

    def test_wall_free_profiles_only_admit_ground_supports(self) -> None:
        for profile in self.rules["architecture"]["scene_visual_profiles"][
            "profiles"
        ]:
            if profile["visual_type"] != "procedural_room":
                continue
            if profile.get("wall_enabled", True):
                continue
            with self.subTest(profile=profile["id"]):
                self.assertEqual(
                    set(profile["scene_classes"]),
                    {"ground_flat", "ground_feature"},
                )

    def test_semantic_support_structure_material_overrides_environment(self) -> None:
        wood_tray_pool = set(
            self.visual["structure_material_pools_by_support"]["wood_tray"]
        )
        for scene in self.scenes:
            if scene["simulation"]["support"]["semantic_type"] != "wood_tray":
                continue
            material_id = scene["appearance"]["materials"]["support_structure"][
                "record"
            ]["asset_id"]
            self.assertIn(material_id, wood_tray_pool, scene["scene_id"])

    def test_camera_profiles_have_distinct_elevation_targets(self) -> None:
        elevations = [
            float(record["overrides"]["view_rule"]["elevation_degrees"])
            for record in self.rules["axes"]["camera_axis"]
        ]
        self.assertEqual(len(elevations), len(set(elevations)))
        self.assertGreaterEqual(max(elevations) - min(elevations), 24.0)

    def test_lab_structure_materials_are_clean_neutral_surfaces(self) -> None:
        self.assertEqual(
            set(
                self.visual["structure_material_pools_by_environment"][
                    "lab_studio"
                ]
            ),
            {
                "white_plaster_02_4k",
                "painted_plaster_wall_4k",
                "garage_floor_4k",
            },
        )
        self.assertEqual(
            set(self.visual["wall_pools_by_environment"]["lab_studio"]),
            {"white_plaster_02_4k", "painted_plaster_wall_4k"},
        )
        self.assertEqual(
            set(self.visual["room_floor_pools_by_environment"]["lab_studio"]),
            {"rubber_tiles_4k", "garage_floor_4k"},
        )

    def test_only_key_light_casts_environment_shadows(self) -> None:
        environment = frozen_environment_binding(
            self.scenes[0],
            {
                "position_m": [2.0, -2.0, 2.0],
                "target_m": [0.0, 0.0, 0.4],
            },
        )
        self.assertTrue(environment["key_light"]["cast_shadow"])
        self.assertFalse(environment["fill_light"]["cast_shadow"])
        self.assertTrue(environment["key_light"]["contact_shadow"])
        self.assertFalse(environment["fill_light"]["contact_shadow"])

    def test_edge_fall_clearance_reaches_a_full_support_exit(self) -> None:
        for scene in self.scenes:
            obj = scene["simulation"]["objects"][0]
            expected = obj["expected_motion"]
            if expected["motion_family"] != "edge_fall_1obj":
                continue
            velocity = [float(value) for value in obj["initial_state"]["linear_velocity_m_s"][:2]]
            speed = math.hypot(*velocity)
            direction = [value / speed for value in velocity]
            start = [float(value) for value in obj["initial_state"]["contact_point_m"][:2]]
            size = [float(value) for value in obj["geometry"]["size_m"]]
            radius = math.hypot(size[0], size[1]) / 2.0
            bounds = scene["simulation"]["support"]["safe_surface_bounds"]
            distances = []
            for axis, key in enumerate(("x", "y")):
                component = direction[axis]
                if abs(component) < 1.0e-8:
                    continue
                boundary = (
                    float(bounds[key][1]) + radius
                    if component > 0.0
                    else float(bounds[key][0]) - radius
                )
                distances.append((boundary - start[axis]) / component)
            with self.subTest(scene=scene["scene_id"]):
                self.assertGreaterEqual(
                    expected["calculated_clearance_distance_m"],
                    min(value for value in distances if value > 0.0),
                )

    def test_sphere_rolling_resistance_matches_ordinary_surfaces(self) -> None:
        rolling_friction = self.backend["contact"]["rolling_friction_by_shape"][
            "sphere"
        ]
        self.assertGreater(rolling_friction, 0.0)
        self.assertLessEqual(rolling_friction, 0.0025)
        self.assertAlmostEqual(
            self.backend["contact"]["rolling_deceleration_calibration_m_s2"],
            0.22,
            places=2,
        )

    def test_round_objects_on_slopes_start_with_surface_rotation(self) -> None:
        candidates = []
        for scene in self.scenes:
            obj = scene["simulation"]["objects"][0]
            motion = obj["expected_motion"]["motion_family"]
            if not motion.startswith("slope_"):
                continue
            if obj["geometry"]["type"] != "sphere":
                continue
            candidates.append(scene)
            angular_speed = math.sqrt(
                sum(
                    value * value
                    for value in obj["initial_state"]["angular_velocity_rad_s"]
                )
            )
            self.assertGreater(angular_speed, 0.1, scene["scene_id"])
            self.assertIn(
                "minimum_peak_angular_speed_rad_s", obj["expected_motion"]
            )
        self.assertTrue(candidates)

    def test_every_scene_is_a_complete_immutable_rigid_contract(self) -> None:
        for scene in self.scenes:
            with self.subTest(scene=scene["scene_id"]):
                self.assertEqual(
                    scene["schema_version"], "physweep_pybullet_rigid_metadata_v1"
                )
                self.assertEqual(
                    scene["semantic_sampling"]["sampling_bundle_version"],
                    self.rules["architecture"]["bundle_version"],
                )
                self.assertEqual(
                    scene["semantic_sampling"]["matrix_version"],
                    "physweep_one_object_sparse_tensor_rules_v5",
                )
                self.assertEqual(scene["simulation"]["backend"]["id"], "pybullet_rigid_v1")
                self.assertEqual(len(scene["simulation"]["objects"]), 1)
                obj = scene["simulation"]["objects"][0]
                self.assertEqual(obj["body_model"], "rigid_body")
                self.assertIn(obj["geometry"]["type"], {"cuboid", "sphere", "cylinder"})
                values = (
                    obj["initial_state"]["position_m"]
                    + obj["initial_state"]["linear_velocity_m_s"]
                    + obj["initial_state"]["angular_velocity_rad_s"]
                )
                self.assertTrue(all(math.isfinite(float(value)) for value in values))
                scale_readability = scene["semantic_sampling"]["five_dimensions"][
                    "foreground_object"
                ]["scale_readability"]
                self.assertIn(
                    scale_readability["characteristic_rule"],
                    {"diameter", "second_largest_extent", "largest_extent"},
                )
                self.assertGreaterEqual(
                    scale_readability["effective_scale"],
                    scale_readability["sampled_scale"],
                )
                self.assertLessEqual(
                    scale_readability["readability_floor_scale"],
                    self.backend["base_parameter_rules"][
                        "object_scale_readability"
                    ]["maximum_readability_uplift_scale"],
                )
                semantic_category = scene["semantic_sampling"]["five_dimensions"][
                    "foreground_object"
                ]["semantic_category"]
                if semantic_category == "physassets_book":
                    self.assertEqual(
                        scale_readability["policy_scope"],
                        "semantic_category:physassets_book",
                    )
                    self.assertEqual(
                        scale_readability["characteristic_rule"], "largest_extent"
                    )
                    self.assertGreaterEqual(
                        max(obj["geometry"]["size_m"]), 0.22 - 1.0e-6
                    )
                colliders = scene["simulation"]["support"]["colliders"]
                self.assertGreaterEqual(len(colliders), 1)
                self.assertEqual(len(colliders), len({item["id"] for item in colliders}))
                interaction = scene["semantic_sampling"]["five_dimensions"]["support_interaction"]
                self.assertEqual(
                    interaction["scene_visual_profile"],
                    scene["appearance"]["scene_visual"]["id"],
                )
                self.assertEqual(
                    interaction["scene_visual_type"],
                    scene["appearance"]["scene_visual"]["visual_type"],
                )
                self.assertEqual(
                    interaction["support_visual_profile"],
                    scene["appearance"]["support_visual"]["id"],
                )
                self.assertEqual(
                    interaction["support_visual_type"],
                    scene["appearance"]["support_visual"]["visual_type"],
                )
                support_visual = scene["appearance"]["support_visual"]
                if support_visual["visual_type"] == "mesh_support":
                    self.assertEqual(
                        scene["simulation"]["support"]["topology"],
                        "flat_surface",
                    )
                    self.assertIn(
                        scene["simulation"]["support"]["semantic_type"],
                        support_visual["support_ids"],
                    )
                    support = scene["simulation"]["support"]
                    self.assertEqual(
                        support["collision_authority"], "exact_static_proxy"
                    )
                    binding = support["exact_static_binding"]
                    self.assertEqual(binding["asset_id"], support_visual["asset_id"])
                    self.assertEqual(len(binding["binding_sha256"]), 64)
                    replaced = [
                        collider
                        for collider in support["colliders"]
                        if collider["role"] in {"primary_support", "support_structure"}
                    ]
                    self.assertTrue(replaced)
                    self.assertTrue(
                        all(not item["collision_enabled"] for item in replaced)
                    )
                self.assertEqual(
                    interaction["scene_class"],
                    scene["simulation"]["support"]["scene_class"],
                )
                materials = scene["appearance"]["materials"]
                material_ids = [
                    materials[role]["record"]["asset_id"]
                    for role in (
                        "dynamic_object",
                        "support_surface",
                        "room_floor",
                        "back_wall",
                    )
                ]
                self.assertEqual(len(material_ids), len(set(material_ids)))
                expected_limits = {
                        "maximum_trajectory_penetration_m": self.backend["quality"][
                            "maximum_trajectory_penetration_m"
                        ],
                        "maximum_trajectory_penetration_fraction_of_min_extent": self.backend[
                            "quality"
                        ]["maximum_trajectory_penetration_fraction_of_min_extent"],
                        "maximum_initial_penetration_m": self.backend["quality"][
                            "maximum_initial_penetration_m"
                        ],
                        "maximum_angular_speed_rad_s": self.backend["quality"][
                            "maximum_angular_speed_rad_s"
                        ],
                        "maximum_rotational_surface_speed_m_s": self.backend["quality"][
                            "maximum_rotational_surface_speed_m_s"
                        ],
                        "rolling_coupling_ratio_range": self.backend["quality"][
                            "rolling_coupling_ratio_range"
                        ],
                    }
                actual_limits = scene["qa"]["limits"]
                self.assertGreaterEqual(
                    actual_limits["maximum_linear_speed_m_s"],
                    self.backend["quality"]["maximum_linear_speed_m_s"],
                )
                self.assertEqual(
                    {
                        key: value
                        for key, value in actual_limits.items()
                        if key != "maximum_linear_speed_m_s"
                    },
                    expected_limits,
                )

    def test_small_batch_falls_back_from_incompatible_mesh_environments(self) -> None:
        scenes = build_batch(
            self.rules,
            self.backend,
            self.materials,
            self.hdri_records,
            self.visual,
            20260726,
            20,
        )
        counts = manifest_counts(scenes)["scene_visual_type"]
        self.assertEqual(sum(counts.values()), 20)
        self.assertLessEqual(counts.get("mesh_backdrop", 0), 8)
        for scene in scenes:
            scene_visual = scene["appearance"]["scene_visual"]
            if scene_visual["visual_type"] != "mesh_backdrop":
                continue
            self.assertEqual(
                scene_visual["composition"]["review_status"], "approved"
            )
            self.assertTrue(
                scene["simulation"]["support"]["scene_class"].startswith(
                    "ground_"
                )
            )
        eligible = [
            scene
            for scene in scenes
            if scene["simulation"]["support"]["semantic_type"]
            in {"wood_tabletop", "lab_bench", "kitchen_counter"}
        ]
        self.assertTrue(eligible)
        self.assertTrue(
            any(
                scene["appearance"]["support_visual"]["visual_type"]
                == "mesh_support"
                for scene in eligible
            )
        )

    def test_explicit_object_sequence_uses_curated_profiles(self) -> None:
        labels = [
            "physassets_11905_book",
            "physassets_13383_book",
            "physassets_13740_book",
            "physassets_5822_basketball",
        ]
        motions = [
            "slope_slide_up_1obj",
            "wall_impact_1obj",
            "edge_fall_1obj",
            "roll_or_slide_1obj",
        ]
        scenes = build_batch(
            self.rules,
            self.backend,
            self.materials,
            self.hdri_records,
            self.visual,
            20260803,
            len(labels),
            motion_sequence=motions,
            object_sequence=labels,
        )
        self.assertEqual(
            [scene["simulation"]["objects"][0]["semantic_type"] for scene in scenes],
            labels,
        )
        for scene in scenes:
            visual = scene["simulation"]["objects"][0]["visual_profile"]
            self.assertIn("assets/library/physassets/curated_visuals/", visual["path"])

    def test_sampled_support_meshes_satisfy_scale_policy(self) -> None:
        scenes = build_batch(
            self.rules,
            self.backend,
            self.materials,
            self.hdri_records,
            self.visual,
            2026072204,
            20,
        )
        for scene in scenes:
            profile = scene["appearance"]["support_visual"]
            if profile["visual_type"] != "mesh_support":
                continue
            ratio = support_mesh_scale_ratio(
                scene["simulation"]["support"], profile
            )
            self.assertLessEqual(
                ratio,
                float(profile["maximum_axis_scale_ratio"]),
                scene["scene_id"],
            )

    def test_visible_raised_support_geometry_blocks_camera_rays(self) -> None:
        for scene in self.scenes:
            support = scene["simulation"]["support"]
            if support["scene_class"].startswith("ground_"):
                continue
            blockers = [
                collider
                for collider in support["colliders"]
                if collider["visible"]
                and collider["role"]
                in {"primary_support", "support_structure", "landing_surface"}
            ]
            self.assertTrue(blockers, scene["scene_id"])
            self.assertTrue(
                all(collider["occludes_camera"] for collider in blockers),
                scene["scene_id"],
            )

    def test_ramps_are_used_only_by_ramp_motion(self) -> None:
        for scene in self.scenes:
            motion = scene["simulation"]["objects"][0]["expected_motion"]["motion_family"]
            support_shape = scene["simulation"]["support"]["support_shape"]
            self.assertEqual(
                motion.startswith("slope_") or motion == "ramp_to_flat_1obj",
                support_shape == "inclined_ramp",
            )

    def test_scene_class_split_prevents_raised_scene_dominance(self) -> None:
        coverage = manifest_counts(self.scenes)["scene_class"]
        ground_count = coverage["ground_flat"] + coverage["ground_feature"]
        raised_count = coverage["raised_flat"] + coverage["raised_feature"]
        self.assertGreaterEqual(ground_count, 0.45 * len(self.scenes))
        self.assertLessEqual(raised_count, 0.55 * len(self.scenes))

        for scene in self.scenes:
            support = scene["simulation"]["support"]
            motion = scene["simulation"]["objects"][0]["expected_motion"]["motion_family"]
            scene_class = support["scene_class"]
            with self.subTest(scene=scene["scene_id"]):
                if motion.startswith("slope_"):
                    self.assertTrue(scene_class.endswith("_feature"))
                elif motion in {"drop_fall_1obj", "projectile_1obj", "arc_projectile_1obj"}:
                    self.assertTrue(scene_class.endswith("_flat"))

    def test_coverage_batch_visits_every_allowed_motion_object_pair(self) -> None:
        pairs = [
            (
                scene["semantic_sampling"]["five_dimensions"]["motion"]["family"],
                scene["semantic_sampling"]["five_dimensions"]["foreground_object"]["object_type"],
            )
            for scene in self.scenes
        ]
        expected = {
            (motion, str(obj["label"]))
            for motion in self.rules["axes"]["motion_axis"]
            for obj in self.rules["axes"]["object_axis"]
            if object_supports_motion(obj, motion, self.rules)
        }
        self.assertEqual(len(pairs), self.coverage_count)
        self.assertTrue(expected <= set(pairs))
        self.assertTrue(set(pairs) <= expected)

    def test_camera_minimum_object_span_has_one_general_floor(self) -> None:
        base = float(self.backend["quality"]["minimum_initial_object_span_ndc"])
        for scene in self.scenes:
            with self.subTest(scene=scene["scene_id"]):
                request = scene["camera_request"]
                threshold = float(request["minimum_initial_object_span_ndc"])
                self.assertNotIn("object_span_scale", request)
                self.assertAlmostEqual(threshold, base, places=6)

    def test_camera_maximum_span_relaxes_only_for_large_objects(self) -> None:
        base = float(self.backend["quality"]["maximum_initial_object_span_ndc"])
        cap = float(self.backend["quality"]["maximum_initial_object_span_cap_ndc"])
        for scene in self.scenes:
            with self.subTest(scene=scene["scene_id"]):
                threshold = float(
                    scene["camera_request"]["maximum_initial_object_span_ndc"]
                )
                self.assertGreaterEqual(threshold, base)
                self.assertLessEqual(threshold, cap)

    def test_camera_semantics_use_motion_intent_and_structure_context(self) -> None:
        self.assertNotIn("framing_profile_axis", self.rules["axes"])
        declared = self.rules["camera_observation"]
        self.assertEqual(
            set(declared["motion_intents"]), set(self.rules["axes"]["motion_axis"])
        )
        for scene in self.scenes:
            request = scene["camera_request"]
            observation = request["observation"]
            self.assertIn(
                observation["structure_context"], declared["structure_contexts"]
            )
            self.assertNotIn("framing_profile", request)

    def test_camera_and_scale_readability_rules_have_general_floors(self) -> None:
        self.assertEqual(
            self.backend["quality"]["minimum_support_context_visible_fraction"],
            0.50,
        )
        self.assertGreaterEqual(
            self.backend["quality"]["minimum_initial_object_span_ndc"], 0.06
        )
        self.assertNotIn(
            "minimum_camera_object_span_scale", self.backend["quality"]
        )
        self.assertGreaterEqual(
            self.backend["quality"]["minimum_initial_object_visible_fraction"],
            0.65,
        )
        self.assertGreaterEqual(
            self.backend["quality"]["initial_object_center_margin_ndc"], 0.02
        )
        adjusted = [
            scene
            for scene in self.scenes
            if scene["semantic_sampling"]["five_dimensions"]["foreground_object"][
                "scale_readability"
            ]["adjusted"]
        ]
        self.assertTrue(adjusted)


if __name__ == "__main__":
    unittest.main()
