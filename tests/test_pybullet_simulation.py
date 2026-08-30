from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from tools.rendering.bind_pybullet_visuals import (  # noqa: E402
    binding_samples,
    resolve_render_request,
)
from tools.rendering.camera_solver import (  # noqa: E402
    _two_object_elevation_candidates,
    audit_two_object_camera,
    camera_inside_structural_envelope,
    camera_occlusion_colliders,
    camera_target_centers,
    image_center_visibility_mask,
    segment_intersects_box,
    segments_intersect_box,
    solve_camera,
    solve_two_object_camera_group,
    support_context_points,
    unoccluded_fraction,
)
from tools.rendering.appearance_adaptation import (  # noqa: E402
    choose_rendered_frame_exposure_adjustment,
    frame_statistics_within_fixed_limits,
)
from tools.assets.environment_collision import binding_sha256  # noqa: E402
from tools.physics.rigid_trajectory import audit_trajectory  # noqa: E402
from tools.sampling.sample_pybullet_base import (  # noqa: E402
    BUNDLE_PATH,
    build_batch,
    load_active_rules,
    load_json,
    sampling_manifest_rule_sources,
)
from tools.physics.simulate_pybullet_rigid import simulate  # noqa: E402
from tools.sampling.sample_two_object_base import (  # noqa: E402
    build_two_object_matrix,
    build_two_object_scene,
)


class PyBulletSimulationTests(unittest.TestCase):
    def test_visual_binding_consumes_audited_physics_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "datasets" / "batch" / "manifest.json"
            source_path.parent.mkdir(parents=True)
            metadata_path = source_path.parent / "scene.json"
            metadata_path.write_text("{}", encoding="utf-8")
            source = {
                "samples": [
                    {
                        "scene_id": "scene_a",
                        "metadata_path": str(metadata_path.relative_to(root)),
                        "metadata_sha256": "metadata-hash",
                    }
                ]
            }
            source_path.write_text(json.dumps(source), encoding="utf-8")
            trajectory = root / "datasets" / "batch" / "physics" / "scene_a" / "trajectory.npz"
            physics = {
                "schema_version": "physweep_pybullet_batch_record_v1",
                "source_manifest": str(source_path),
                "sample_count": 1,
                "records": [
                    {
                        "ok": True,
                        "audit_passed": True,
                        "scene_id": "scene_a",
                        "metadata_path": str(metadata_path),
                        "metadata_sha256": "metadata-hash",
                        "trajectory_path": str(trajectory),
                    }
                ],
            }
            resolved_source, samples = binding_samples(root, physics)
        self.assertEqual(resolved_source, source)
        self.assertEqual(samples[0]["trajectory_path"], str(trajectory))
        self.assertEqual(
            samples[0]["simulation_record_path"],
            str(trajectory.with_name("simulation_record.json")),
        )
    def test_visual_binding_inherits_frozen_render_request(self) -> None:
        metadata = {"render_request": {"resolution": [1280, 720], "samples": 16}}

        self.assertEqual(resolve_render_request(metadata, None, None), ((1280, 720), 16))
        self.assertEqual(
            resolve_render_request(metadata, (640, 360), 4), ((640, 360), 4)
        )

    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = load_active_rules(ROOT)
        backend = load_json(ROOT / "configs/pybullet_backend.json")
        visual = load_json(ROOT / "configs/visual_sampling.json")
        materials_manifest = load_json(
            ROOT / "assets/manifests/polyhaven_render_library.json"
        )
        hdri_manifest = load_json(ROOT / "assets/manifests/hdri_admission.json")
        materials = {
            str(record["asset_id"]): record
            for record in materials_manifest["assets"]
        }
        coverage_count = (
            len(cls.rules["axes"]["motion_axis"])
            * len(cls.rules["axes"]["object_axis"])
        )
        candidates = build_batch(
            cls.rules,
            backend,
            materials,
            list(hdri_manifest["records"]),
            visual,
            20260719,
            coverage_count,
        )
        cls.candidates = candidates
        cls.scene_by_motion = {}
        cls.scene_by_object = {}
        for scene in candidates:
            motion = scene["simulation"]["objects"][0]["expected_motion"]["motion_family"]
            cls.scene_by_motion.setdefault(motion, scene)
            object_type = scene["simulation"]["objects"][0]["semantic_type"]
            cls.scene_by_object.setdefault(object_type, scene)
        rolling_spheres = [
            scene
            for scene in candidates
            if scene["simulation"]["objects"][0]["expected_motion"]["motion_family"]
            == "roll_or_slide_1obj"
            and scene["simulation"]["objects"][0]["geometry"]["type"] == "sphere"
        ]
        cls.rolling_stress_scene = min(
            rolling_spheres,
            key=lambda scene: np.linalg.norm(
                scene["simulation"]["objects"][0]["initial_state"][
                    "linear_velocity_m_s"
                ][:2]
            ),
        )

    @staticmethod
    def without_incidental_environment(scene: dict) -> dict:
        isolated = copy.deepcopy(scene)
        binding = isolated["environment_binding"]
        binding["colliders"] = []
        binding["binding_sha256"] = binding_sha256(binding)
        return isolated

    def test_camera_uses_visible_support_and_environment_blockers(self) -> None:
        scene = next(
            candidate
            for candidate in self.candidates
            if any(
                collider.get("role") == "room_detail"
                for collider in candidate["environment_binding"]["colliders"]
            )
        )
        blockers = camera_occlusion_colliders(scene)
        expected_ids = {
            str(collider["id"])
            for collider in scene["simulation"]["support"]["colliders"]
            if bool(collider.get("visible", True))
            and bool(collider.get("occludes_camera", False))
            and collider.get("primitive") == "box"
        }
        expected_ids.update(
            str(collider["id"])
            for collider in scene["environment_binding"]["colliders"]
            if bool(collider.get("visible", True))
            and bool(collider.get("collision_enabled", True))
            and collider.get("primitive") == "box"
        )
        self.assertEqual({str(collider["id"]) for collider in blockers}, expected_ids)
        self.assertTrue(
            all(
                bool(collider["occludes_camera"])
                for collider in scene["environment_binding"]["colliders"]
                if collider.get("primitive") == "box"
                and bool(collider.get("visible", True))
            )
        )

        legacy_detail = copy.deepcopy(
            next(
                collider
                for collider in scene["environment_binding"]["colliders"]
                if collider.get("role") == "room_detail"
                and collider.get("primitive") == "box"
            )
        )
        legacy_detail["occludes_camera"] = False
        center = np.asarray(legacy_detail["position_m"], dtype=np.float64)
        camera = center + np.asarray([0.0, 0.0, 2.0], dtype=np.float64)
        self.assertEqual(
            unoccluded_fraction(camera, center[None, :], [legacy_detail]),
            0.0,
        )

    def test_every_motion_family_simulates_and_passes_semantic_qa(self) -> None:
        self.assertEqual(set(self.scene_by_motion), set(self.rules["axes"]["motion_axis"]))
        for motion, scene in self.scene_by_motion.items():
            with self.subTest(motion=motion):
                trajectory, audit = simulate(
                    self.without_incidental_environment(scene)
                )
                self.assertTrue(audit["passed"], audit)
                check_ids = {record["id"] for record in audit["checks"]}
                self.assertIn("useful_active_duration", check_ids)
                self.assertTrue(
                    {
                        "initial_position_matches_metadata",
                        "initial_orientation_matches_metadata",
                        "initial_linear_velocity_matches_metadata",
                        "initial_angular_velocity_matches_metadata",
                        "pybullet_dynamics_match_metadata",
                        "collision_proxy_count_matches_definition",
                        "collision_proxy_types_match_definition",
                        "collision_proxy_dimensions_match_definition",
                        "collision_proxy_positions_match_definition",
                        "collision_proxy_orientations_match_definition",
                        "runtime_inertia_is_finite_and_positive",
                        "primitive_inertia_matches_geometry",
                        "pybullet_support_dynamics_match_metadata",
                        "bounded_unforced_mechanical_energy_gain",
                        "coulomb_friction_force_within_limit",
                    }
                    <= check_ids
                )
                if motion in {
                    "drop_fall_1obj",
                    "projectile_1obj",
                    "arc_projectile_1obj",
                    "bounce_1obj",
                }:
                    self.assertIn("airborne_gravity_fit", check_ids)
                if motion in {"projectile_1obj", "arc_projectile_1obj"}:
                    self.assertIn("ballistic_theory_position_rmse", check_ids)
                    self.assertIn(
                        "ballistic_theory_maximum_position_error", check_ids
                    )

                    self.assertIn(
                        "airborne_horizontal_velocity_drift", check_ids
                    )
                if motion == "bounce_1obj":
                    self.assertIn(
                        "bounce_response_matches_restitution", check_ids
                    )
                required_collider = scene["simulation"]["objects"][0][
                    "expected_motion"
                ].get("required_collider_contact_id")
                if required_collider:
                    self.assertIn(
                        f"{scene['simulation']['objects'][0]['object_id']}__collider_contact_count__{required_collider}",
                        trajectory,
                    )
                if motion == "ramp_to_flat_1obj":
                    self.assertTrue(
                        {
                            "transition_contact_order",
                            "transition_destination_only_contact",
                            "transition_no_source_recontact",
                            "no_unplanned_environment_contact",
                            "minimum_post_transition_travel",
                        }
                        <= check_ids
                    )
                    self.assertIn(
                        "minimum_post_transition_travel_m",
                        scene["simulation"]["objects"][0]["expected_motion"],
                    )
                camera = solve_camera(scene, trajectory, self.rules)
                diagnostics = camera["diagnostics"]
                self.assertGreaterEqual(
                    diagnostics["primary_center_visible_fraction"], 0.80
                )
                self.assertGreaterEqual(
                    diagnostics["support_context_visible_fraction"], 0.50
                )
                self.assertGreaterEqual(
                    diagnostics["required_structure_anchor_visible_fraction"],
                    scene["camera_request"]["observation"][
                        "minimum_anchor_visible_fraction"
                    ],
                )
                self.assertGreaterEqual(
                    diagnostics["initial_object_visible_fraction"], 0.75
                )
                self.assertGreaterEqual(
                    diagnostics["primary_trajectory_unoccluded_fraction"],
                    scene["camera_request"][
                        "minimum_primary_trajectory_unoccluded_fraction"
                    ],
                )
                self.assertGreaterEqual(
                    diagnostics["full_trajectory_unoccluded_fraction"],
                    scene["camera_request"][
                        "minimum_full_trajectory_unoccluded_fraction"
                    ],
                )
                self.assertGreaterEqual(
                    diagnostics["target_unoccluded_fraction"], 0.0
                )
                self.assertLessEqual(
                    diagnostics["target_unoccluded_fraction"], 1.0
                )
                self.assertGreaterEqual(
                    diagnostics[
                        "required_structure_anchors_unoccluded_fraction"
                    ],
                    scene["camera_request"]["observation"][
                        "minimum_anchor_unoccluded_fraction"
                    ],
                )
                self.assertGreaterEqual(
                    diagnostics["distance_m"], diagnostics["minimum_distance_m"]
                )
                self.assertGreaterEqual(
                    diagnostics["full_trajectory_center_visible_fraction"], 0.30
                )
                self.assertLessEqual(
                    diagnostics["focus_span_ndc"],
                    scene["camera_request"]["maximum_focus_span_ndc"],
                )
                self.assertGreaterEqual(
                    diagnostics["median_primary_object_span_ndc"],
                    scene["camera_request"]["observation"][
                        "minimum_median_object_span_ndc"
                    ],
                )

    def test_useful_duration_rejects_an_artificial_early_stop(self) -> None:
        scene = copy.deepcopy(self.scene_by_motion["slide_push_1obj"])
        trajectory, _ = simulate(scene)
        object_id = scene["simulation"]["objects"][0]["object_id"]
        velocity_key = f"{object_id}__linear_velocity_m_s"
        trajectory[velocity_key] = trajectory[velocity_key].copy()
        trajectory[velocity_key][2:] = 0.0
        audit = audit_trajectory(scene, trajectory)
        duration_check = next(
            record
            for record in audit["checks"]
            if record["id"] == "useful_active_duration"
        )
        self.assertFalse(duration_check["passed"])

    def test_camera_full_target_and_visibility_use_complete_exact_image_data(self) -> None:
        focus_points = np.asarray(
            [[-0.2, 0.0, 0.0], [0.2, 1.0, 0.4]], dtype=np.float64
        )
        positions = np.asarray(
            [[0.0, 0.0, 0.2], [0.0, 1.0, 0.2], [0.0, 4.0, 0.2]],
            dtype=np.float64,
        )
        primary_target, full_target = camera_target_centers(focus_points, positions)
        np.testing.assert_allclose(primary_target, [0.0, 0.5, 0.2])
        np.testing.assert_allclose(full_target, [0.0, 2.0, 0.2])

        projected = np.asarray(
            [
                [0.0, 0.5, 1.0],
                [1.0, 0.5, 1.0],
                [-0.001, 0.5, 1.0],
                [1.001, 0.5, 1.0],
                [0.5, 0.5, 0.05],
            ],
            dtype=np.float64,
        )
        self.assertEqual(
            image_center_visibility_mask(projected).tolist(),
            [True, True, False, False, False],
        )

    def test_sampling_manifest_separates_bundle_from_compiled_camera_rules(self) -> None:
        bundle_path = ROOT / BUNDLE_PATH.relative_to(BUNDLE_PATH.parents[1])
        bundle = load_json(bundle_path)
        sources = sampling_manifest_rule_sources(ROOT, bundle_path, bundle)
        self.assertEqual(sources["sampling_bundle_path"], str(bundle_path.relative_to(ROOT)))
        self.assertEqual(sources["rules_path"], str(bundle["base_rules"]))
        self.assertNotEqual(sources["sampling_bundle_path"], sources["rules_path"])
        self.assertEqual(len(sources["sampling_bundle_sha256"]), 64)
        self.assertEqual(len(sources["rules_sha256"]), 64)

    def test_ramp_landing_uses_local_floor_anchor(self) -> None:
        metadata = {
            "simulation": {
                "support": {
                    "safe_surface_bounds": {
                        "x": [-1.0, 1.0],
                        "y": [-0.5, 0.5],
                    },
                    "surface_frame": {"slope_angle_degrees": 14.0},
                    "surface_center_z_m": 0.15,
                    "visual_geometry": {
                        "primitive": "solid_wedge",
                        "size_xy_m": [2.0, 1.0],
                        "base_z_m": 0.0,
                        "high_top_z_m": 0.30,
                        "slope_axis": "y",
                    },
                    "colliders": [
                        {
                            "id": "environment_floor",
                            "size_m": [20.0, 20.0, 0.1],
                            "position_m": [0.0, 0.0, -0.05],
                            "rotation_euler_degrees": [0.0, 0.0, 0.0],
                        }
                    ],
                }
            }
        }
        observation = {
            "structure_context": "ramp_and_landing",
            "focus_event": {"collider_id": "environment_floor"},
        }
        positions = np.asarray(
            [[0.0, 0.35, 0.30], [0.1, -0.80, 0.12]],
            dtype=np.float64,
        )
        points, required = support_context_points(
            metadata,
            azimuth_degrees=0.0,
            focus_xy=np.asarray([0.0, 0.0], dtype=np.float64),
            observation=observation,
            positions=positions,
        )
        anchors = points[required]
        self.assertEqual(len(anchors), 5)
        self.assertTrue(
            np.allclose(
                anchors[:4],
                [
                    [-1.0, -0.5, 0.01],
                    [1.0, -0.5, 0.01],
                    [-1.0, 0.5, 0.31],
                    [1.0, 0.5, 0.31],
                ],
            )
        )
        self.assertTrue(np.allclose(anchors[4], [0.1, -0.8, 0.01]))
        self.assertLessEqual(float(np.max(np.abs(anchors[:, 0]))), 1.0)

    def test_camera_policy_is_compiled_once_into_metadata(self) -> None:
        for scene in self.candidates:
            obj = scene["simulation"]["objects"][0]
            expected = obj["expected_motion"]
            motion = expected["motion_family"]
            request = scene["camera_request"]
            observation = request["observation"]
            declared = self.rules["camera_observation"]["motion_intents"][motion]
            self.assertEqual(
                observation["version"], self.rules["camera_observation"]["version"]
            )
            self.assertEqual(observation["intent"], declared["intent"])
            self.assertEqual(
                observation["structure_context"], declared["structure_context"]
            )
            self.assertNotEqual(
                observation["focus_event"]["type"], "required_motion_collider"
            )
            if declared["focus_event"]["type"] in {
                "required_motion_collider",
                "transition_destination_contact",
            }:
                self.assertEqual(
                    observation["focus_event"]["collider_id"],
                    expected["required_collider_contact_id"],
                )
            if declared["focus_event"]["type"] == "transition_destination_contact":
                contract = scene["simulation"]["support"]["transition_contract"]
                self.assertEqual(
                    observation["focus_event"]["collider_id"],
                    contract["destination_collider_id"],
                )
                self.assertEqual(
                    expected["transition_contract_version"], contract["version"]
                )
            self.assertEqual(
                request["minimum_camera_elevation_degrees"],
                observation["elevation_range_degrees"][0],
            )
            self.assertEqual(
                request["maximum_camera_elevation_degrees"],
                observation["elevation_range_degrees"][1],
            )
            if observation["structure_context"] in {
                "inclined_surface",
                "ramp_and_landing",
            }:
                self.assertEqual(
                    self.rules["camera_observation"][
                        "minimum_inclined_surface_side_readability"
                    ],
                    0.90,
                )
                self.assertEqual(
                    observation["minimum_anchor_visible_fraction"], 1.0
                )
                self.assertEqual(
                    request["minimum_support_context_visible_fraction"], 0.80
                )
                self.assertEqual(request["maximum_focus_span_ndc"], 1.00)
                self.assertEqual(
                    request["maximum_camera_distance_above_minimum_m"], 4.0
                )
                self.assertEqual(request["maximum_camera_distance_m"], 6.0)
            self.assertNotIn("framing_profile", request)
            self.assertFalse(
                any(key.startswith("camera_") for key in expected), expected
            )

    def test_long_raised_surface_camera_anchors_cover_both_axes(self) -> None:
        metadata = {
            "simulation": {
                "support": {
                    "scene_class": "raised_flat",
                    "safe_surface_bounds": {
                        "x": [-2.3, 2.3],
                        "y": [-0.62, 0.62],
                    },
                    "surface_frame": {"slope_angle_degrees": 0.0},
                    "surface_center_z_m": 0.82,
                    "motion_axis": "x",
                    "maximum_planar_trajectory_distance_m": 2.2,
                    "colliders": [],
                }
            }
        }
        observation = {
            "structure_context": "horizontal_surface",
            "focus_event": {"type": "fraction"},
        }
        points, required = support_context_points(
            metadata,
            azimuth_degrees=35.0,
            focus_xy=np.asarray([1.0, 0.0], dtype=np.float64),
            observation=observation,
            positions=np.asarray(
                [[0.0, 0.0, 0.9], [2.0, 0.0, 0.9]], dtype=np.float64
            ),
        )
        anchors = points[required]
        self.assertEqual(len(anchors), 5)
        self.assertAlmostEqual(float(np.ptp(anchors[:, 0])), 2.2, places=6)
        self.assertGreater(float(np.ptp(anchors[:, 1])), 0.7)
        self.assertTrue(np.allclose(anchors[:, 2], 0.83))

    def test_direct_simulation_is_repeatable(self) -> None:
        scene = self.scene_by_motion["slide_push_1obj"]
        first, first_audit = simulate(scene)
        second, second_audit = simulate(scene)
        self.assertTrue(first_audit["passed"])
        self.assertEqual(first_audit, second_audit)
        self.assertEqual(set(first), set(second))
        for key in first:
            self.assertTrue(np.array_equal(first[key], second[key]), key)

    def test_two_object_head_on_is_repeatable_and_object_complete(self) -> None:
        template = self.without_incidental_environment(self.rolling_stress_scene)
        scene = build_two_object_scene(
            template,
            load_json(ROOT / "configs/two_object_sampling_matrix.json"),
            "surface_head_on_2obj",
        )
        first, first_audit = simulate(scene)
        second, second_audit = simulate(scene)
        self.assertTrue(first_audit["passed"], first_audit)
        self.assertEqual(first_audit, second_audit)
        self.assertEqual(set(first), set(second))
        for key in first:
            self.assertTrue(np.array_equal(first[key], second[key]), key)
        object_ids = [
            str(record["object_id"])
            for record in scene["simulation"]["objects"]
        ]
        self.assertEqual(object_ids, ["object_a", "object_b"])
        foreground = scene["semantic_sampling"]["five_dimensions"][
            "foreground_objects"
        ]
        self.assertEqual(
            [set(record) for record in foreground],
            [
                {"object_id", "semantic_category", "scale_bin", "uniform_scale"},
                {"object_id", "semantic_category", "scale_bin", "uniform_scale"},
            ],
        )
        identity = scene["object_identity"]
        source_mention = (
            "the "
            + str(template["simulation"]["objects"][0]["semantic_type"])
            .replace("_", " ")
            .strip()
        )
        self.assertEqual(
            [record["text"] for record in identity["text"]["object_mentions"]],
            [source_mention, source_mention],
        )
        self.assertEqual(
            identity["text"]["caption"],
            (
                f"{source_mention} and {source_mention} collide while moving "
                "toward each other."
            ),
        )
        initial_midpoint_xy = np.mean(
            [
                record["initial_state"]["position_m"][:2]
                for record in scene["simulation"]["objects"]
            ],
            axis=0,
        )
        bounds = scene["simulation"]["support"]["safe_surface_bounds"]
        support_center_xy = np.asarray(
            [np.mean(bounds["x"]), np.mean(bounds["y"])], dtype=np.float64
        )
        self.assertTrue(
            np.allclose(
                initial_midpoint_xy,
                support_center_xy,
                atol=1.0e-6,
                rtol=0.0,
            )
        )
        for object_id, other_object_id in (
            ("object_a", "object_b"),
            ("object_b", "object_a"),
        ):
            self.assertEqual(
                first[f"{object_id}__position_m"].shape,
                (scene["simulation"]["time"]["frame_count"], 3),
            )
            self.assertGreater(
                int(
                    np.max(
                        first[
                            f"{object_id}__object_contact_count__{other_object_id}"
                        ]
                    )
                ),
                0,
            )
        camera = solve_camera(scene, first, self.rules)
        self.assertEqual(
            camera["solver_version"],
            "joint_full_motion_envelope_camera_v3",
        )
        diagnostics = camera["diagnostics"]
        self.assertEqual(diagnostics["object_count"], 2)
        self.assertEqual(
            diagnostics["joint_motion_envelope_visible_fraction"], 1.0
        )
        self.assertGreaterEqual(
            diagnostics[
                "pair_keyframe_projected_center_separation_to_radius_sum_ratio"
            ],
            scene["simulation"]["interaction"][
                "minimum_pair_keyframe_projected_center_separation_to_radius_sum_ratio"
            ],
        )
        self.assertEqual(diagnostics["pair_keyframe_kind"], "first_contact")
        self.assertEqual(
            list(diagnostics["per_object_visibility"]),
            ["object_a", "object_b"],
        )
        derived_camera_scene = copy.deepcopy(scene)
        derived_camera_scene["scene_id"] += "__derived_camera_test"
        derived_camera_scene["sweep"] = {
            "kind": "sweep",
            "parent_scene_id": scene["scene_id"],
            "parameter": "mass_kg",
        }
        self.assertEqual(
            solve_camera(derived_camera_scene, first, self.rules),
            camera,
        )
        inelastic_sweep = copy.deepcopy(scene)
        inelastic_sweep["scene_id"] += "__inelastic_sweep_test"
        inelastic_sweep["simulation"]["objects"][0]["material"][
            "contact_restitution"
        ] = 0.0
        inelastic_sweep["sweep"] = {
            "kind": "sweep",
            "parent_scene_id": scene["scene_id"],
            "parameter": "contact_restitution",
        }
        inelastic_trajectory, inelastic_audit = simulate(inelastic_sweep)
        self.assertTrue(inelastic_audit["passed"], inelastic_audit)
        self.assertEqual(inelastic_audit["advisories"], [])
        inherited_diagnostics = audit_two_object_camera(
            inelastic_sweep, inelastic_trajectory, camera
        )
        self.assertEqual(
            inherited_diagnostics["joint_motion_envelope_visible_fraction"],
            1.0,
        )
        group_camera = solve_two_object_camera_group(
            scene,
            [
                (derived_camera_scene, first),
                (inelastic_sweep, inelastic_trajectory),
            ],
            self.rules,
        )
        self.assertEqual(
            group_camera["solver_version"],
            "joint_full_motion_envelope_group_camera_v3",
        )
        self.assertEqual(
            group_camera["diagnostics"]["camera_group"]["member_count"], 2
        )
        for member, member_trajectory in (
            (derived_camera_scene, first),
            (inelastic_sweep, inelastic_trajectory),
        ):
            self.assertEqual(
                audit_two_object_camera(
                    member, member_trajectory, group_camera
                )["joint_motion_envelope_visible_fraction"],
                1.0,
            )

    def test_two_object_camera_supports_heterogeneous_sphere_scales(self) -> None:
        host = self.without_incidental_environment(self.rolling_stress_scene)
        matrix = load_json(ROOT / "configs/two_object_sampling_matrix.json")
        spheres = [
            scene
            for scene in self.candidates
            if scene["simulation"]["objects"][0]["geometry"]["type"]
            == "sphere"
        ]
        smallest = min(
            spheres,
            key=lambda scene: scene["simulation"]["objects"][0]["geometry"][
                "size_m"
            ][0],
        )
        largest = max(
            spheres,
            key=lambda scene: scene["simulation"]["objects"][0]["geometry"][
                "size_m"
            ][0],
        )
        for sources in ((smallest, largest), (largest, smallest)):
            with self.subTest(
                object_ids=[
                    source["simulation"]["objects"][0]["visual_profile"]["id"]
                    for source in sources
                ]
            ):
                scene = build_two_object_scene(
                    host, matrix, "surface_head_on_2obj", sources
                )
                trajectory, audit = simulate(scene)
                self.assertTrue(audit["passed"], audit)
                camera = solve_camera(scene, trajectory, self.rules)
                diagnostics = camera["diagnostics"]
                self.assertEqual(
                    diagnostics["joint_motion_envelope_visible_fraction"],
                    1.0,
                )
                self.assertGreaterEqual(
                    diagnostics[
                        "pair_keyframe_projected_center_separation_to_radius_sum_ratio"
                    ],
                    scene["simulation"]["interaction"][
                        "minimum_pair_keyframe_projected_center_separation_to_radius_sum_ratio"
                    ],
                )
                self.assertLessEqual(
                    abs(diagnostics["side_view_deviation_degrees"]),
                    scene["simulation"]["interaction"][
                        "maximum_camera_side_deviation_degrees"
                    ],
                )

    def test_two_object_camera_tries_both_sides_of_the_motion_axis(self) -> None:
        host = self.without_incidental_environment(self.rolling_stress_scene)
        matrix = load_json(ROOT / "configs/two_object_sampling_matrix.json")
        scene = build_two_object_scene(host, matrix, "surface_head_on_2obj")
        trajectory, audit = simulate(scene)
        self.assertTrue(audit["passed"], audit)
        preferred = solve_camera(scene, trajectory, self.rules)
        blocked = copy.deepcopy(scene)
        binding = blocked["environment_binding"]
        binding["visual_objects"] = []
        binding["colliders"] = []
        camera_position = np.asarray(preferred["position_m"], dtype=np.float64)
        camera_target = np.asarray(preferred["target_m"], dtype=np.float64)
        outward = camera_position[:2] - camera_target[:2]
        outward /= np.linalg.norm(outward)
        wall_center = camera_target[:2] + 0.5 * outward
        wall_yaw = math.degrees(math.atan2(outward[1], outward[0])) - 90.0
        wall = {
            "id": "preferred_side_camera_blocker",
            "primitive": "box",
            "role": "room_wall",
            "material_role": "back_wall",
            "size_m": [6.0, 0.10, 4.0],
            "position_m": [float(wall_center[0]), float(wall_center[1]), 2.0],
            "rotation_euler_degrees": [0.0, 0.0, wall_yaw],
            "visible": True,
            "collision_enabled": True,
            "occludes_camera": True,
        }
        binding["visual_objects"].append(copy.deepcopy(wall))
        binding["colliders"].append(wall)
        binding["binding_sha256"] = binding_sha256(binding)
        alternate = solve_camera(blocked, trajectory, self.rules)
        self.assertGreater(
            alternate["diagnostics"]["pair_camera_candidate_failure_count"],
            0,
        )
        preferred_azimuth = preferred["diagnostics"][
            "pair_selected_side_azimuth_degrees"
        ]
        alternate_azimuth = alternate["diagnostics"][
            "pair_selected_side_azimuth_degrees"
        ]
        self.assertAlmostEqual(
            abs(alternate_azimuth - preferred_azimuth), 180.0, places=6
        )

    def test_two_object_camera_uses_contract_interior_elevations(self) -> None:
        candidates = _two_object_elevation_candidates(
            {
                "minimum_elevation": 18.0,
                "preferred_elevation": 28.0,
                "maximum_elevation": 42.0,
            }
        )
        self.assertEqual(candidates, (28.0, 35.0, 23.0, 38.5, 20.5))
        self.assertTrue(all(18.0 < value < 42.0 for value in candidates))

    def test_two_object_motion_matrix_contact_contracts(self) -> None:
        host = self.without_incidental_environment(self.rolling_stress_scene)
        matrix = load_json(ROOT / "configs/two_object_sampling_matrix.json")
        scenes = build_two_object_matrix(host, matrix)
        self.assertEqual(len(scenes), 9)
        self.assertEqual(
            [
                scene["simulation"]["interaction"]["motion_pattern"]
                for scene in scenes
            ],
            [
                "surface_hit_rest_2obj",
                "surface_head_on_2obj",
                "surface_crossing_2obj",
                "surface_catch_up_2obj",
                "air_drop_hit_supported_2obj",
                "air_projectile_hit_supported_2obj",
                "surface_single_independent_2obj",
                "surface_dual_independent_2obj",
                "air_supported_independent_2obj",
            ],
        )
        self.assertEqual(
            [scene["sample_index"] for scene in scenes], list(range(1, 10))
        )
        classes = [
            scene["simulation"]["interaction"]["interaction_class"]
            for scene in scenes
        ]
        self.assertEqual(classes.count("interacting"), 6)
        self.assertEqual(classes.count("independent"), 3)
        semantics = {
            scene["simulation"]["interaction"]["motion_pattern"]: scene[
                "semantic_sampling"
            ]["five_dimensions"]["motion"]
            for scene in scenes
        }
        self.assertIsNone(
            semantics["surface_hit_rest_2obj"]["trajectory_angle_degrees"]
        )
        self.assertEqual(
            semantics["surface_head_on_2obj"]["trajectory_angle_degrees"],
            180.0,
        )
        self.assertEqual(
            semantics["surface_crossing_2obj"]["trajectory_angle_degrees"],
            90.0,
        )
        self.assertEqual(
            semantics["surface_catch_up_2obj"]["trajectory_angle_degrees"],
            0.0,
        )
        self.assertEqual(
            semantics["surface_crossing_2obj"]["impact_offset_ratio"], 0.30
        )
        drop_scene = next(
            scene
            for scene in scenes
            if scene["simulation"]["interaction"]["motion_pattern"]
            == "air_drop_hit_supported_2obj"
        )
        drop_object, supported_object = drop_scene["simulation"]["objects"]
        drop_position = drop_object["initial_state"]["position_m"]
        supported_position = supported_object["initial_state"]["position_m"]
        drop_velocity = drop_object["initial_state"]["linear_velocity_m_s"]
        drop_radius = 0.5 * float(drop_object["geometry"]["size_m"][2])
        support_height = float(
            drop_scene["simulation"]["support"]["surface_center_z_m"]
        )
        approach = drop_scene["simulation"]["interaction"][
            "approach_axis_xyz"
        ]
        self.assertGreater(drop_position[2] - support_height - drop_radius, 0.70)
        horizontal_separation = abs(
            drop_position[0] - supported_position[0]
        )
        self.assertGreater(horizontal_separation, 0.03)
        self.assertLess(horizontal_separation, 0.20)
        self.assertGreater(drop_velocity[0], 0.0)
        self.assertEqual(drop_velocity[2], 0.0)
        self.assertGreater(abs(approach[0]), 0.05)
        self.assertLess(abs(approach[0]), 0.20)
        self.assertLess(approach[2], -0.95)
        with self.assertRaisesRegex(ValueError, "may not be repeated"):
            build_two_object_matrix(
                host,
                matrix,
                motion_ids=["surface_head_on_2obj", "surface_head_on_2obj"],
            )
        for scene in scenes:
            with self.subTest(scene_id=scene["scene_id"]):
                trajectory, audit = simulate(scene)
                self.assertTrue(audit["passed"], audit)
                interaction = scene["simulation"]["interaction"]
                object_a, object_b = interaction["object_ids"]
                contacts = trajectory[
                    f"{object_a}__object_contact_count__{object_b}"
                ]
                contact_occurred = bool(np.any(contacts > 0))
                self.assertEqual(
                    contact_occurred,
                    interaction["interaction_class"] == "interacting",
                )
                self.assertEqual(
                    audit["metrics"]["motion_pattern"],
                    interaction["motion_pattern"],
                )
                camera = solve_camera(scene, trajectory, self.rules)
                diagnostics = camera["diagnostics"]
                self.assertEqual(
                    camera["solver_version"],
                    "joint_full_motion_envelope_camera_v3",
                )
                self.assertEqual(
                    diagnostics["joint_motion_envelope_visible_fraction"],
                    1.0,
                )
                expected_keyframe_kind = (
                    "first_contact"
                    if interaction["interaction_class"] == "interacting"
                    else "closest_approach"
                )
                self.assertEqual(
                    diagnostics["pair_keyframe_kind"], expected_keyframe_kind
                )
                expected_lower = np.min(
                    np.stack(
                        [
                            trajectory[f"{object_id}__aabb_min_m"]
                            for object_id in interaction["object_ids"]
                        ]
                    ),
                    axis=(0, 1),
                )
                expected_upper = np.max(
                    np.stack(
                        [
                            trajectory[f"{object_id}__aabb_max_m"]
                            for object_id in interaction["object_ids"]
                        ]
                    ),
                    axis=(0, 1),
                )
                self.assertTrue(
                    np.allclose(
                        diagnostics["joint_motion_envelope_world_bounds_m"][
                            "min"
                        ],
                        expected_lower,
                        atol=1.0e-6,
                        rtol=0.0,
                    )
                )
                self.assertTrue(
                    np.allclose(
                        diagnostics["joint_motion_envelope_world_bounds_m"][
                            "max"
                        ],
                        expected_upper,
                        atol=1.0e-6,
                        rtol=0.0,
                    )
                )
                for visibility in diagnostics["per_object_visibility"].values():
                    self.assertEqual(
                        visibility["full_motion_aabb_visible_fraction"], 1.0
                    )
                    self.assertGreaterEqual(
                        visibility["median_span_ndc"],
                        interaction["minimum_per_object_median_span_ndc"],
                    )

    def test_two_object_independent_contract_rejects_pair_contact(self) -> None:
        host = self.without_incidental_environment(self.rolling_stress_scene)
        matrix = load_json(ROOT / "configs/two_object_sampling_matrix.json")
        scene = build_two_object_scene(
            host, matrix, "surface_head_on_2obj"
        )
        trajectory, _ = simulate(scene)
        declared_independent = copy.deepcopy(scene)
        interaction = declared_independent["simulation"]["interaction"]
        interaction["type"] = "pairwise_independent"
        interaction["interaction_class"] = "independent"
        interaction["contact_requirement"] = "must_not_contact"
        declared_independent["semantic_sampling"]["five_dimensions"]["motion"][
            "interaction_class"
        ] = "independent"
        object_ids = interaction["object_ids"]
        for index, obj in enumerate(declared_independent["simulation"]["objects"]):
            expected = obj["expected_motion"]
            expected.pop("required_object_contact_id")
            expected["forbidden_object_contact_id"] = object_ids[1 - index]
        audit = audit_trajectory(declared_independent, trajectory)
        self.assertFalse(audit["passed"])
        failed = {
            record["id"] for record in audit["checks"] if not record["passed"]
        }
        self.assertIn("forbidden_pair_collision", failed)

    def test_two_object_audit_requires_explicit_pair_contract(self) -> None:
        host = self.without_incidental_environment(self.rolling_stress_scene)
        matrix = load_json(ROOT / "configs/two_object_sampling_matrix.json")
        scene = build_two_object_scene(
            host, matrix, "surface_head_on_2obj"
        )
        trajectory, _ = simulate(scene)
        self.assertNotIn(
            "approach_axis_xy", scene["simulation"]["interaction"]
        )
        invalid_axis = copy.deepcopy(scene)
        invalid_axis["simulation"]["interaction"]["approach_axis_xyz"] = [
            0.0,
            0.0,
            1.0,
        ]
        with self.assertRaisesRegex(ValueError, "horizontal projection"):
            audit_trajectory(invalid_axis, trajectory)
        for field in (
            "minimum_initial_clearance_m",
            "maximum_first_contact_time_s",
            "minimum_pre_contact_closing_speed_m_s",
        ):
            with self.subTest(field=field):
                incomplete = copy.deepcopy(scene)
                del incomplete["simulation"]["interaction"][field]
                with self.assertRaises(KeyError):
                    audit_trajectory(incomplete, trajectory)
        missing_expectation = copy.deepcopy(scene)
        del missing_expectation["simulation"]["objects"][0]["expected_motion"][
            "minimum_displacement_m"
        ]
        with self.assertRaises(KeyError):
            audit_trajectory(missing_expectation, trajectory)

    def test_two_object_scene_composes_independent_object_sources(self) -> None:
        host = self.without_incidental_environment(self.rolling_stress_scene)
        host_object = host["simulation"]["objects"][0]
        host_visual_id = str(host_object["visual_profile"]["id"])
        spheres = [
            scene
            for scene in self.candidates
            if scene["simulation"]["objects"][0]["geometry"]["type"]
            == "sphere"
            and str(
                scene["simulation"]["objects"][0]["visual_profile"]["id"]
            )
            != host_visual_id
        ]
        self.assertTrue(spheres)
        host_extent = float(host_object["geometry"]["size_m"][0])
        secondary = min(
            spheres,
            key=lambda scene: abs(
                float(scene["simulation"]["objects"][0]["geometry"]["size_m"][0])
                - host_extent
            ),
        )
        secondary_object = secondary["simulation"]["objects"][0]
        host["sweep"] = {
            "kind": "base",
            "resolved_object_physics": [
                {
                    "object_id": "object_a",
                    "object_index": 0,
                    "material": copy.deepcopy(host_object["material"]),
                }
            ],
        }
        host["scene_id"] += "__base"
        original_host = copy.deepcopy(host)
        original_secondary = copy.deepcopy(secondary)

        scene = build_two_object_scene(
            host,
            load_json(ROOT / "configs/two_object_sampling_matrix.json"),
            "surface_head_on_2obj",
            (secondary, host),
            sample_index=7,
        )

        objects = scene["simulation"]["objects"]
        self.assertEqual(scene["dataset_id"], "physweep_two_object")
        self.assertNotIn("__base__", scene["scene_id"])
        self.assertNotIn("sweep", scene)
        self.assertEqual(
            [obj["visual_profile"]["id"] for obj in objects],
            [
                secondary_object["visual_profile"]["id"],
                host_object["visual_profile"]["id"],
            ],
        )
        self.assertEqual(objects[0]["geometry"], secondary_object["geometry"])
        self.assertEqual(objects[1]["geometry"], host_object["geometry"])
        self.assertEqual(
            objects[0]["collision_profile"],
            secondary_object["collision_profile"],
        )
        self.assertEqual(
            objects[0]["material"]["mass_kg"],
            secondary_object["material"]["mass_kg"],
        )
        bounds = host["simulation"]["support"]["safe_surface_bounds"]
        anchor_x, anchor_y = np.mean(bounds["x"]), np.mean(bounds["y"])
        midpoint_x = 0.5 * sum(obj["initial_state"]["position_m"][0] for obj in objects)
        self.assertAlmostEqual(midpoint_x, anchor_x)
        self.assertAlmostEqual(objects[0]["initial_state"]["position_m"][1], anchor_y)
        self.assertEqual(scene["sample_index"], 7)
        self.assertNotEqual(
            scene["environment_binding"], host["environment_binding"]
        )
        self.assertEqual(
            scene["environment_binding"]["profile_id"],
            host["environment_binding"]["profile_id"],
        )
        self.assertEqual(
            scene["simulation"]["interaction"]["scene_compatibility"],
            {
                "schema_version": "physweep_two_object_scene_compatibility_v1",
                "scene_class": host["simulation"]["support"]["scene_class"],
                "environment_binding_policy": (
                    "recompiled_for_preferred_pair_side"
                ),
            },
        )
        self.assertEqual(scene["simulation"]["support"], host["simulation"]["support"])
        self.assertEqual(host, original_host)
        self.assertEqual(secondary, original_secondary)
        self.assertEqual(
            set(scene["appearance"]["materials"]["dynamic_objects"]),
            {"object_a", "object_b"},
        )
        with self.assertRaisesRegex(ValueError, "one role per candidate"):
            build_two_object_scene(
                host,
                load_json(ROOT / "configs/two_object_sampling_matrix.json"),
                "surface_head_on_2obj",
                (),
            )
        derived_source = copy.deepcopy(secondary)
        derived_source["sweep"] = {"kind": "sweep"}
        with self.assertRaisesRegex(ValueError, "unswept or canonical bases"):
            build_two_object_scene(
                host,
                load_json(ROOT / "configs/two_object_sampling_matrix.json"),
                "surface_head_on_2obj",
                (derived_source, host),
            )

    def test_runtime_proxy_dimension_tampering_is_rejected(self) -> None:
        scene = self.scene_by_motion["drop_fall_1obj"]
        trajectory, _ = simulate(scene)
        obj = scene["simulation"]["objects"][0]
        key = f"{obj['object_id']}__runtime_proxy_dimensions_m"
        trajectory[key] = trajectory[key].copy()
        trajectory[key][0, 0] += 0.01
        audit = audit_trajectory(scene, trajectory)
        failed = {record["id"] for record in audit["checks"] if not record["passed"]}
        self.assertIn("collision_proxy_dimensions_match_definition", failed)

    def test_runtime_support_parameter_tampering_is_rejected(self) -> None:
        scene = self.scene_by_motion["slide_push_1obj"]
        trajectory, _ = simulate(scene)
        obj = scene["simulation"]["objects"][0]
        key = f"{obj['object_id']}__runtime_support_dynamics"
        trajectory[key] = trajectory[key].copy()
        trajectory[key][0, 0] += 0.1
        audit = audit_trajectory(scene, trajectory)
        failed = {record["id"] for record in audit["checks"] if not record["passed"]}
        self.assertIn("pybullet_support_dynamics_match_metadata", failed)

    def test_nonballistic_airborne_path_is_rejected(self) -> None:
        scene = self.scene_by_motion["arc_projectile_1obj"]
        trajectory, _ = simulate(scene)
        obj = scene["simulation"]["objects"][0]
        position_key = f"{obj['object_id']}__position_m"
        contact_key = f"{obj['object_id']}__all_contact_count"
        contact_indices = np.flatnonzero(trajectory[contact_key] > 0)
        airborne_count = int(contact_indices[0]) if contact_indices.size else len(
            trajectory["time_s"]
        )
        trajectory[position_key] = trajectory[position_key].copy()
        trajectory[position_key][:airborne_count, 0] += 0.20 * np.linspace(
            0.0, 1.0, airborne_count
        ) ** 2
        audit = audit_trajectory(scene, trajectory)
        failed = {record["id"] for record in audit["checks"] if not record["passed"]}
        self.assertIn("ballistic_theory_position_rmse", failed)

    def test_wall_impact_without_rebound_requirement_tolerates_rest(self) -> None:
        scene = copy.deepcopy(self.scene_by_motion["wall_impact_1obj"])
        obj = scene["simulation"]["objects"][0]
        expected = obj["expected_motion"]
        expected["minimum_post_impact_rebound_speed_m_s"] = 0.0
        trajectory, _ = simulate(scene)
        contact_key = (
            f"{obj['object_id']}__collider_contact_count__"
            f"{expected['required_collider_contact_id']}"
        )
        contact_indices = np.flatnonzero(trajectory[contact_key] > 0)
        self.assertGreater(contact_indices.size, 0)
        contact_index = int(contact_indices[0])
        velocity_key = f"{obj['object_id']}__linear_velocity_m_s"
        trajectory[velocity_key][contact_index:, :2] = (
            np.asarray(expected["impact_normal_xy"], dtype=np.float64) * 1.0e-8
        )
        audit = audit_trajectory(scene, trajectory)
        self.assertTrue(audit["passed"], audit)
        self.assertNotIn(
            "wall_post_impact_rebound_speed",
            {check["id"] for check in audit["checks"]},
        )

    def test_object_and_support_visuals_do_not_change_physics(self) -> None:
        scene = next(
            candidate
            for candidate in build_batch(
                self.rules,
                load_json(ROOT / "configs/pybullet_backend.json"),
                {
                    str(record["asset_id"]): record
                    for record in load_json(
                        ROOT / "assets/manifests/polyhaven_render_library.json"
                    )["assets"]
                },
                list(
                    load_json(ROOT / "assets/manifests/hdri_admission.json")[
                        "records"
                    ]
                ),
                load_json(ROOT / "configs/visual_sampling.json"),
                20260725,
                198,
            )
            if candidate["simulation"]["objects"][0]["visual_profile"]["type"]
            == "mesh"
            and candidate["appearance"]["support_visual"]["visual_type"]
            == "mesh_support"
        )
        primitive_scene = copy.deepcopy(scene)
        primitive_scene["simulation"]["objects"][0]["visual_profile"] = {
            "id": "physics_invariance_primitive",
            "type": "primitive",
            "material_hint": "test",
            "color": [0.5, 0.5, 0.5, 1.0],
        }
        primitive_scene["appearance"]["support_visual"] = {
            "id": "procedural_support_proxy",
            "visual_type": "procedural_proxy",
            "support_ids": [
                primitive_scene["simulation"]["support"]["semantic_type"]
            ],
        }
        mesh_trajectory, mesh_audit = simulate(scene)
        primitive_trajectory, primitive_audit = simulate(primitive_scene)
        self.assertEqual(mesh_audit, primitive_audit)
        self.assertEqual(set(mesh_trajectory), set(primitive_trajectory))
        for key in mesh_trajectory:
            self.assertTrue(
                np.array_equal(mesh_trajectory[key], primitive_trajectory[key]), key
            )

    def test_every_object_profile_simulates_without_backend_changes(self) -> None:
        expected = {
            str(record["label"])
            for record in self.rules["axes"]["object_axis"]
        }
        self.assertEqual(set(self.scene_by_object), expected)
        for object_type, scene in self.scene_by_object.items():
            with self.subTest(object_type=object_type):
                isolated_scene = self.without_incidental_environment(scene)
                obj = isolated_scene["simulation"]["objects"][0]
                self.assertIn(obj["geometry"]["type"], {"cuboid", "sphere", "cylinder"})
                trajectory, audit = simulate(isolated_scene)
                self.assertTrue(audit["passed"], audit)
                self.assertTrue(
                    np.isfinite(trajectory[f"{obj['object_id']}__position_m"]).all()
                )

    def test_low_speed_rolling_boundary_remains_visible(self) -> None:
        trajectory, audit = simulate(self.rolling_stress_scene)
        self.assertTrue(audit["passed"], audit)
        self.assertGreater(
            audit["metrics"]["horizontal_path_length_m"],
            0.30,
        )

    def test_rolling_sphere_initial_contact_velocity_is_zero(self) -> None:
        obj = self.rolling_stress_scene["simulation"]["objects"][0]
        initial = obj["initial_state"]
        radius = float(obj["geometry"]["size_m"][0]) / 2.0
        velocity = np.asarray(initial["linear_velocity_m_s"], dtype=np.float64)
        angular = np.asarray(initial["angular_velocity_rad_s"], dtype=np.float64)
        contact_offset = np.asarray([0.0, 0.0, -radius], dtype=np.float64)
        contact_velocity = velocity + np.cross(angular, contact_offset)
        self.assertLess(np.linalg.norm(contact_velocity), 1.0e-5)

    def test_wrong_schema_is_rejected(self) -> None:
        scene = dict(self.scene_by_motion["drop_fall_1obj"])
        scene["schema_version"] = "unsupported_schema"
        with self.assertRaises(ValueError):
            simulate(scene)

    def test_oriented_box_blocks_only_crossing_view_segments(self) -> None:
        wall = {
            "position_m": [0.0, 0.0, 1.0],
            "size_m": [0.06, 1.0, 0.6],
            "rotation_euler_degrees": [0.0, 0.0, 30.0],
        }
        self.assertTrue(
            segment_intersects_box(
                np.asarray([-1.0, 0.0, 1.0]),
                np.asarray([1.0, 0.0, 1.0]),
                wall,
            )
        )
        self.assertFalse(
            segment_intersects_box(
                np.asarray([-1.0, 1.0, 1.0]),
                np.asarray([1.0, 1.0, 1.0]),
                wall,
            )
        )

    def test_vectorized_box_intersections_match_scalar_solver(self) -> None:
        wall = {
            "position_m": [0.2, -0.1, 0.8],
            "size_m": [0.12, 1.1, 0.7],
            "rotation_euler_degrees": [5.0, -8.0, 37.0],
        }
        start = np.asarray([-1.4, -0.6, 1.4])
        ends = np.asarray(
            [
                [1.2, -0.2, 0.9],
                [1.2, 1.5, 0.9],
                [-0.8, -0.4, 1.3],
                [0.2, -0.1, 0.8],
            ]
        )
        expected = np.asarray(
            [segment_intersects_box(start, end, wall) for end in ends]
        )
        np.testing.assert_array_equal(
            segments_intersect_box(start, ends, wall), expected
        )

    def test_rendered_frame_exposure_gate_uses_shared_audit_limits(self) -> None:
        passing = {
            "mean_luma": 60.0,
            "luma_std": 20.0,
            "mean_gradient": 5.6,
            "clipped_dark_fraction": 0.0,
            "clipped_light_fraction": 0.0,
        }
        self.assertTrue(frame_statistics_within_fixed_limits(passing))
        failing = dict(passing, luma_std=13.6)
        self.assertFalse(frame_statistics_within_fixed_limits(failing))
        adjustment = choose_rendered_frame_exposure_adjustment([failing])
        self.assertIn("low_luma_contrast", adjustment["reasons"])
        self.assertGreater(adjustment["applied_delta_ev"], 0.0)
        self.assertLessEqual(adjustment["applied_delta_ev"], 0.35)
        bright_uniform = dict(passing, mean_luma=100.0, luma_std=14.0)
        self.assertTrue(frame_statistics_within_fixed_limits(bright_uniform))
        self.assertNotIn(
            "low_luma_contrast",
            choose_rendered_frame_exposure_adjustment([bright_uniform])["reasons"],
        )
        pre_encode_borderline = dict(passing, mean_luma=46.0)
        self.assertTrue(frame_statistics_within_fixed_limits(pre_encode_borderline))
        self.assertIn(
            "low_mean_luma",
            choose_rendered_frame_exposure_adjustment([pre_encode_borderline])[
                "reasons"
            ],
        )

    def test_corridor_camera_must_remain_between_and_below_walls(self) -> None:
        metadata = {
            "simulation": {
                "support": {
                    "motion_axis": "x",
                    "camera_envelope": {
                        "type": "paired_parallel_walls",
                        "motion_axis": "x",
                        "collider_ids": ["side_wall_a", "side_wall_b"],
                        "clearance_m": 0.35,
                    },
                    "colliders": [
                        {
                            "id": "side_wall_a",
                            "position_m": [0.0, -1.3, 1.2],
                            "size_m": [7.0, 0.12, 2.4],
                        },
                        {
                            "id": "side_wall_b",
                            "position_m": [0.0, 1.3, 1.2],
                            "size_m": [7.0, 0.12, 2.4],
                        },
                    ],
                }
            }
        }
        self.assertTrue(
            camera_inside_structural_envelope(
                metadata, np.asarray([2.0, 0.4, 1.8])
            )
        )
        self.assertFalse(
            camera_inside_structural_envelope(
                metadata, np.asarray([0.0, -1.9, 1.8])
            )
        )
        self.assertFalse(
            camera_inside_structural_envelope(
                metadata, np.asarray([0.0, 1.0, 1.8])
            )
        )
        self.assertFalse(
            camera_inside_structural_envelope(
                metadata, np.asarray([0.0, 0.0, 2.5])
            )
        )

    def test_rendered_frame_exposure_gate_is_bounded_and_idempotent(self) -> None:
        healthy = {
            "mean_luma": 100.0,
            "luma_std": 24.0,
        }
        self.assertEqual(
            choose_rendered_frame_exposure_adjustment([healthy])["applied_delta_ev"],
            0.0,
        )
        very_dark = {"mean_luma": 1.0, "luma_std": 0.5}
        first = choose_rendered_frame_exposure_adjustment([very_dark])
        self.assertEqual(first["applied_delta_ev"], 0.35)
        second = choose_rendered_frame_exposure_adjustment([very_dark], 0.99)
        self.assertAlmostEqual(second["applied_delta_ev"], 0.01, places=6)


if __name__ == "__main__":
    unittest.main()
