from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from bind_pybullet_visuals import (  # noqa: E402
    camera_target_centers,
    image_center_visibility_mask,
    segment_intersects_box,
    segments_intersect_box,
    solve_camera,
    support_context_points,
)
from appearance_adaptation import (  # noqa: E402
    choose_rendered_frame_exposure_adjustment,
    frame_statistics_within_fixed_limits,
)
from rigid_trajectory import audit_trajectory  # noqa: E402
from sample_pybullet_base import (  # noqa: E402
    BUNDLE_PATH,
    build_batch,
    load_active_rules,
    load_json,
    sampling_manifest_rule_sources,
)
from simulate_pybullet_rigid import simulate  # noqa: E402


class PyBulletSimulationTests(unittest.TestCase):
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

    def test_every_motion_family_simulates_and_passes_semantic_qa(self) -> None:
        self.assertEqual(set(self.scene_by_motion), set(self.rules["axes"]["motion_axis"]))
        for motion, scene in self.scene_by_motion.items():
            with self.subTest(motion=motion):
                trajectory, audit = simulate(scene)
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
        self.assertEqual(len(anchors), 3)
        self.assertTrue(np.allclose(anchors[:2, 1], [-0.5, 0.5]))
        self.assertTrue(np.allclose(anchors[2], [0.1, -0.8, 0.01]))
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
            if declared["focus_event"]["type"] == "required_motion_collider":
                self.assertEqual(
                    observation["focus_event"]["collider_id"],
                    expected["required_collider_contact_id"],
                )
            self.assertEqual(
                request["minimum_camera_elevation_degrees"],
                declared["elevation_range_degrees"][0],
            )
            self.assertEqual(
                request["maximum_camera_elevation_degrees"],
                declared["elevation_range_degrees"][1],
            )
            self.assertNotIn("framing_profile", request)
            self.assertFalse(
                any(key.startswith("camera_") for key in expected), expected
            )

    def test_direct_simulation_is_repeatable(self) -> None:
        scene = self.scene_by_motion["slide_push_1obj"]
        first, first_audit = simulate(scene)
        second, second_audit = simulate(scene)
        self.assertTrue(first_audit["passed"])
        self.assertEqual(first_audit, second_audit)
        self.assertEqual(set(first), set(second))
        for key in first:
            self.assertTrue(np.array_equal(first[key], second[key]), key)

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
                obj = scene["simulation"]["objects"][0]
                self.assertIn(obj["geometry"]["type"], {"cuboid", "sphere", "cylinder"})
                trajectory, audit = simulate(scene)
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
            "mean_luma": 41.8,
            "luma_std": 8.1,
            "mean_gradient": 5.6,
            "clipped_dark_fraction": 0.0,
            "clipped_light_fraction": 0.0,
        }
        self.assertTrue(frame_statistics_within_fixed_limits(passing))
        failing = dict(passing, luma_std=7.1364)
        self.assertFalse(frame_statistics_within_fixed_limits(failing))
        adjustment = choose_rendered_frame_exposure_adjustment([failing])
        self.assertIn("low_luma_contrast", adjustment["reasons"])
        self.assertGreater(adjustment["applied_delta_ev"], 0.0)
        self.assertLessEqual(adjustment["applied_delta_ev"], 0.35)

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
        second = choose_rendered_frame_exposure_adjustment([very_dark], 0.69)
        self.assertAlmostEqual(second["applied_delta_ev"], 0.01, places=6)


if __name__ == "__main__":
    unittest.main()
