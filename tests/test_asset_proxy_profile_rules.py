from __future__ import annotations

import copy
import json
import random
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from sample_asset_proxy_scenes import (  # noqa: E402
    asset_motion_usefulness,
    asset_camera_observation,
    audit_asset_trajectory,
    motion_initial_state,
    proxy_volume_fill_ratio,
)


class AssetProxyProfileRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = json.loads(
            (ROOT / "configs/pybullet_backend.json").read_text(encoding="utf-8")
        )
        cls.visual = json.loads(
            (ROOT / "configs/visual_sampling.json").read_text(encoding="utf-8")
        )
        cls.camera = json.loads(
            (ROOT / "configs/one_object_sampling_rules.json").read_text(
                encoding="utf-8"
            )
        )["camera_observation"]

    def test_every_asset_motion_profile_compiles_camera_observation(self) -> None:
        profiles = set(self.backend["asset_proxy_rules"]["motion_profiles"])
        self.assertEqual(
            profiles, set(self.camera["asset_profile_motion_intents"])
        )
        for profile in profiles:
            observation = asset_camera_observation(self.camera, profile)
            motion = self.camera["asset_profile_motion_intents"][profile]
            declared = self.camera["motion_intents"][motion]
            self.assertEqual(observation["intent"], declared["intent"])
            self.assertEqual(
                observation["structure_context"], declared["structure_context"]
            )
            self.assertIn(
                observation["structure_context"], self.camera["structure_contexts"]
            )

    def test_edge_exit_camera_keeps_landing_and_resting_pose_visible(self) -> None:
        observation = asset_camera_observation(self.camera, "edge_exit")
        self.assertEqual(observation["structure_context"], "edge_and_landing")
        self.assertEqual(observation["focus_event"]["post_event_frames"], 18)
        self.assertTrue(observation["include_final_settled_pose"])
        self.assertGreaterEqual(
            observation["minimum_full_trajectory_center_visible_fraction"], 0.80
        )
        self.assertGreaterEqual(observation["minimum_transition_margin_ndc"], 0.075)
        self.assertEqual(observation["focal_length_candidates_mm"], [40.0, 36.0])
        self.assertEqual(
            self.camera["structure_contexts"]["edge_and_landing"][
                "minimum_anchor_visible_fraction"
            ],
            1.0,
        )

    def test_every_asset_motion_profile_has_a_useful_duration_contract(self) -> None:
        profiles = set(self.backend["asset_proxy_rules"]["motion_profiles"])
        configured = set(
            self.backend["asset_proxy_rules"]["quality"][
                "minimum_active_duration_s_by_profile"
            ]
        )
        self.assertEqual(configured, profiles)
        for profile in sorted(profiles):
            contract = asset_motion_usefulness(self.backend, profile)
            self.assertGreater(contract["minimum_active_duration_s"], 0.0)
            self.assertEqual(contract["active_speed_threshold_m_s"], 0.03)

    def test_coast_distance_scales_with_available_support_distance(self) -> None:
        bounds_low = np.asarray([-0.05, -0.05, -0.05], dtype=np.float64)
        bounds_high = np.asarray([0.05, 0.05, 0.05], dtype=np.float64)
        common = {
            "profile": "resting_push",
            "bounds_low": bounds_low,
            "bounds_high": bounds_high,
            "prop": None,
            "rules": self.backend["asset_proxy_rules"],
            "dynamic_friction": 0.40,
            "support_friction": 0.62,
            "gravity_m_s2": 9.81,
            "proxy_motion_class": "sliding",
        }
        small = motion_initial_state(
            surface={"size_xy_m": [1.0, 0.8], "center_xy_m": [0.0, 0.0], "z_m": 0.8},
            rng=random.Random(17),
            **common,
        )
        large = motion_initial_state(
            surface={"size_xy_m": [2.4, 1.2], "center_xy_m": [0.0, 0.0], "z_m": 0.8},
            rng=random.Random(17),
            **common,
        )
        small_calculation = small["calculation"]
        large_calculation = large["calculation"]
        self.assertGreater(
            large_calculation["target_coast_distance_m"],
            small_calculation["target_coast_distance_m"] + 0.10,
        )
        for calculation in (small_calculation, large_calculation):
            self.assertLessEqual(
                calculation["target_coast_distance_m"],
                calculation["available_safe_distance_m"] * 0.82 + 1.0e-7,
            )

    def test_trajectory_audit_does_not_mutate_velocity_samples(self) -> None:
        frame_count = 25
        arrays = {
            "time_s": np.arange(frame_count, dtype=np.float64) / 24.0,
            "position_m": np.column_stack(
                (
                    np.linspace(0.0, 0.30, frame_count),
                    np.zeros(frame_count),
                    np.full(frame_count, 0.85),
                )
            ),
            "linear_velocity_m_s": np.column_stack(
                (
                    np.linspace(0.80, 0.0, frame_count),
                    np.full(frame_count, 0.20),
                    np.zeros(frame_count),
                )
            ),
            "angular_velocity_rad_s": np.zeros((frame_count, 3)),
            "support_contact": np.ones(frame_count, dtype=np.int8),
            "ground_contact": np.zeros(frame_count, dtype=np.int8),
            "prop_contact": np.zeros(frame_count, dtype=np.int8),
        }
        before = copy.deepcopy(arrays["linear_velocity_m_s"])
        audit_asset_trajectory(
            "diagonal_push",
            arrays,
            minimum_contact_distance=0.0,
            initial_penetration_m=0.0,
            initial_prop_contacts=0,
            proxy_extent_m=[0.10, 0.10, 0.10],
            surface={"size_xy_m": [1.5, 0.9], "center_xy_m": [0.0, 0.0], "z_m": 0.8},
            quality=self.backend["quality"],
            asset_rules=self.backend["asset_proxy_rules"],
        )
        np.testing.assert_array_equal(arrays["linear_velocity_m_s"], before)

    def test_round_proxy_uses_duration_based_launch_speed(self) -> None:
        state = motion_initial_state(
            profile="resting_push",
            surface={"size_xy_m": [2.0, 0.8], "center_xy_m": [0.0, 0.0], "z_m": 0.8},
            bounds_low=np.asarray([-0.05, -0.05, -0.05], dtype=np.float64),
            bounds_high=np.asarray([0.05, 0.05, 0.05], dtype=np.float64),
            prop=None,
            rng=random.Random(19),
            rules=self.backend["asset_proxy_rules"],
            dynamic_friction=0.78,
            support_friction=0.62,
            gravity_m_s2=9.81,
            proxy_motion_class="rolling_round",
        )
        calculation = state["calculation"]
        self.assertEqual(calculation["method"], "rolling_travel_time_target")
        self.assertLessEqual(
            calculation["launch_speed_m_s"],
            self.backend["asset_proxy_rules"]["round_proxy_motion"][
                "maximum_launch_speed_m_s"
            ],
        )
        self.assertGreaterEqual(calculation["target_duration_s"], 2.4)

    def test_edge_exit_requires_a_reviewed_clear_direction(self) -> None:
        common = {
            "profile": "edge_exit",
            "surface": {
                "size_xy_m": [2.0, 0.8],
                "center_xy_m": [0.0, 0.0],
                "z_m": 0.8,
            },
            "bounds_low": np.asarray([-0.05, -0.05, -0.05], dtype=np.float64),
            "bounds_high": np.asarray([0.05, 0.05, 0.05], dtype=np.float64),
            "prop": None,
            "rng": random.Random(23),
            "rules": self.backend["asset_proxy_rules"],
            "dynamic_friction": 0.35,
            "support_friction": 0.62,
            "gravity_m_s2": 9.81,
            "proxy_motion_class": "sliding",
        }
        with self.assertRaisesRegex(ValueError, "clear_exit_directions_xy"):
            motion_initial_state(interaction_policy={}, **common)

        state = motion_initial_state(
            interaction_policy={"clear_exit_directions_xy": [[0.0, -1.0]]},
            **common,
        )
        self.assertEqual(state["calculation"]["clear_exit_direction_xy"], [0.0, -1.0])
        self.assertLess(state["linear_velocity_m_s"][1], 0.0)

    def test_edge_exit_uses_physical_boundary_and_nonround_tipping_energy(self) -> None:
        common = {
            "profile": "edge_exit",
            "surface": {
                "size_xy_m": [2.02, 0.82],
                "center_xy_m": [0.0, 0.0],
                "z_m": 0.82,
            },
            "bounds_low": np.asarray([-0.21, -0.147, -0.072], dtype=np.float64),
            "bounds_high": np.asarray([0.21, 0.147, 0.072], dtype=np.float64),
            "prop": None,
            "rules": self.backend["asset_proxy_rules"],
            "dynamic_friction": 0.55,
            "support_friction": 0.62,
            "gravity_m_s2": 9.81,
            "interaction_policy": {"clear_exit_directions_xy": [[0.0, -1.0]]},
        }
        safe_boundary = motion_initial_state(
            rng=random.Random(37),
            proxy_motion_class="sliding",
            **common,
        )
        physical_boundary = motion_initial_state(
            rng=random.Random(37),
            proxy_motion_class="sliding",
            physical_support_size_xy_m=[2.2, 0.995],
            **common,
        )
        calculation = physical_boundary["calculation"]
        self.assertGreater(
            calculation["coast_distance_m"],
            safe_boundary["calculation"]["coast_distance_m"],
        )
        self.assertGreater(calculation["tipping_barrier_height_m"], 0.0)
        self.assertGreater(
            calculation["launch_speed_m_s"],
            safe_boundary["calculation"]["launch_speed_m_s"],
        )
        self.assertAlmostEqual(
            calculation["projected_half_extent_m"], 0.147, places=6
        )

        rolling = motion_initial_state(
            rng=random.Random(37),
            proxy_motion_class="rolling_round",
            physical_support_size_xy_m=[2.2, 0.995],
            **common,
        )
        self.assertEqual(rolling["calculation"]["tipping_barrier_height_m"], 0.0)

    def test_edge_exit_compactness_rejects_sparse_compound_proxies(self) -> None:
        solid = {
            "asset_id": "solid",
            "proxy": {
                "colliders": [
                    {
                        "shape": "box",
                        "size_m": [0.42, 0.42, 0.05],
                        "position_m": [0.0, 0.0, 0.0],
                        "rotation_euler_degrees": [0.0, 0.0, 0.0],
                    }
                ]
            },
        }
        sparse = {
            "asset_id": "sparse",
            "proxy": {
                "colliders": [
                    {
                        "shape": "box",
                        "size_m": [0.42, 0.05, 0.05],
                        "position_m": [0.0, 0.0, 0.0],
                        "rotation_euler_degrees": [0.0, 0.0, 0.0],
                    },
                    {
                        "shape": "box",
                        "size_m": [0.05, 0.42, 0.05],
                        "position_m": [0.0, 0.0, 0.0],
                        "rotation_euler_degrees": [0.0, 0.0, 0.0],
                    },
                ]
            },
        }
        threshold = self.backend["asset_proxy_rules"]["motion_profiles"][
            "edge_exit"
        ]["minimum_proxy_volume_fill_ratio"]
        self.assertGreaterEqual(proxy_volume_fill_ratio(solid), threshold)
        self.assertLess(proxy_volume_fill_ratio(sparse), threshold)

    def test_unplanned_static_prop_contact_is_rejected(self) -> None:
        frame_count = 25
        arrays = {
            "time_s": np.arange(frame_count, dtype=np.float64) / 24.0,
            "position_m": np.column_stack(
                (
                    np.linspace(0.0, 0.35, frame_count),
                    np.zeros(frame_count),
                    np.full(frame_count, 0.85),
                )
            ),
            "linear_velocity_m_s": np.column_stack(
                (
                    np.linspace(0.9, 0.0, frame_count),
                    np.zeros(frame_count),
                    np.zeros(frame_count),
                )
            ),
            "angular_velocity_rad_s": np.zeros((frame_count, 3)),
            "support_contact": np.ones(frame_count, dtype=np.int8),
            "ground_contact": np.zeros(frame_count, dtype=np.int8),
            "prop_contact": np.zeros(frame_count, dtype=np.int8),
        }
        arrays["prop_contact"][8] = 1
        audit = audit_asset_trajectory(
            "resting_push",
            arrays,
            minimum_contact_distance=0.0,
            initial_penetration_m=0.0,
            initial_prop_contacts=0,
            proxy_extent_m=[0.10, 0.10, 0.10],
            surface={"size_xy_m": [1.5, 0.9], "center_xy_m": [0.0, 0.0], "z_m": 0.8},
            quality=self.backend["quality"],
            asset_rules=self.backend["asset_proxy_rules"],
        )
        self.assertFalse(audit["checks"]["no_unplanned_static_prop_contact"])
        self.assertFalse(audit["passed"])

    def test_prop_lane_clearance_is_geometry_derived(self) -> None:
        state = motion_initial_state(
            profile="resting_push",
            surface={
                "size_xy_m": [1.64, 0.74],
                "center_xy_m": [0.0, 0.0],
                "z_m": 0.78,
            },
            bounds_low=np.asarray([-0.12, -0.12, -0.04], dtype=np.float64),
            bounds_high=np.asarray([0.12, 0.12, 0.04], dtype=np.float64),
            prop={
                "position_m": [0.0, 0.135, 0.78],
                "world_aabb_extent_m": [0.44, 0.42, 0.07],
            },
            rng=random.Random(29),
            rules=self.backend["asset_proxy_rules"],
            dynamic_friction=0.44,
            support_friction=0.62,
            gravity_m_s2=9.81,
            proxy_motion_class="sliding",
        )
        self.assertLess(state["position_m"][1], 0.0)
        self.assertGreaterEqual(
            state["calculation"]["prop_lane_clearance_m"],
            self.backend["asset_proxy_rules"]["placement"][
                "prop_lane_minimum_clearance_m"
            ],
        )

    def test_prop_lane_uses_a_feasible_yaw_fallback(self) -> None:
        rules = copy.deepcopy(self.backend["asset_proxy_rules"])
        rules["placement"]["yaw_degrees"] = [28.0, 28.0]
        rules["placement"]["prop_lane_yaw_attempts"] = 1
        state = motion_initial_state(
            profile="resting_push",
            surface={
                "size_xy_m": [1.6, 0.8],
                "center_xy_m": [0.0, 0.0],
                "z_m": 0.78,
            },
            bounds_low=np.asarray([-0.14, -0.04, -0.04], dtype=np.float64),
            bounds_high=np.asarray([0.14, 0.04, 0.04], dtype=np.float64),
            prop={
                "position_m": [0.0, 0.0, 0.78],
                "world_aabb_center_xy_m": [0.0, 0.0],
                "world_aabb_extent_m": [0.4, 0.4, 0.07],
            },
            rng=random.Random(31),
            rules=rules,
            dynamic_friction=0.40,
            support_friction=0.62,
            gravity_m_s2=9.81,
            proxy_motion_class="sliding",
        )
        self.assertAlmostEqual(state["orientation_quaternion_xyzw"][2], 0.0)
        self.assertGreaterEqual(
            state["calculation"]["prop_lane_clearance_m"], 0.02
        )

    def test_asset_lighting_uses_one_dominant_key(self) -> None:
        rules = self.visual["asset_proxy_render"]
        lights = {record["name"]: record for record in rules["area_lights"]}
        self.assertGreater(lights["key"]["energy_w"], 4.0 * lights["fill"]["energy_w"])
        self.assertGreater(lights["key"]["energy_w"], 6.0 * lights["rim"]["energy_w"])
        self.assertLessEqual(max(rules["world_strength"]), 0.30)
        self.assertLessEqual(rules["light_scale"]["maximum"], 1.25)


if __name__ == "__main__":
    unittest.main()
