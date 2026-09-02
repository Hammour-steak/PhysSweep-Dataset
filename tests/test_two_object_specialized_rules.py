from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from tools.motion_rules.two_object.specialized import (
    family_index,
    load_two_object_specialized_rules,
    profile_index,
    resolve_billiards_initial_states,
    resolve_marble_run_initial_states,
    resolve_pinball_initial_states,
    resolve_specialized_camera_binding,
)
from tools.physics.generate_passive_pinball_scene import build_fixture


ROOT = Path(__file__).resolve().parents[1]


class TwoObjectSpecializedRulesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rules = load_two_object_specialized_rules(ROOT)
        cls.families = family_index(cls.rules)
        cls.profiles = profile_index(cls.rules)

    def test_three_frozen_fixture_families_have_three_profiles_and_views_each(self) -> None:
        self.assertEqual(
            set(self.families), {"billiards", "passive_pinball", "marble_run"}
        )
        self.assertEqual(len(self.profiles), 9)
        self.assertTrue(
            all(
                profile["interaction_class"] == "interacting"
                and profile["contact_requirement"] == "must_contact"
                for _, profile in self.profiles.values()
            )
        )
        for family in self.families.values():
            view_ids = {
                view["id"] for view in family["camera_contract"]["view_families"]
            }
            self.assertEqual(len(view_ids), 3)
            self.assertEqual(
                {profile["camera_view_family_id"] for profile in family["profiles"]},
                view_ids,
            )

    def test_billiards_profiles_resolve_two_separated_spheres(self) -> None:
        backend = json.loads(
            (ROOT / "configs/pybullet_backend.json").read_text(encoding="utf-8")
        )
        radius = float(backend["billiards_rules"]["ball_radius_m"])
        for profile_id in (
            "two_ball_direct_collision",
            "two_ball_glancing_collision",
            "two_ball_opposed_collision",
        ):
            _, profile = self.profiles[profile_id]
            states = resolve_billiards_initial_states(
                profile, bed_z_m=0.65, ball_radius_m=radius
            )
            self.assertEqual(
                [state["object_id"] for state in states],
                ["object_a", "object_b"],
            )
            distance = np.linalg.norm(
                np.asarray(states[0]["position_m"])
                - np.asarray(states[1]["position_m"])
            )
            self.assertGreater(distance, 2.0 * radius)

    def test_pinball_profiles_resolve_in_the_fixture_frame(self) -> None:
        config = json.loads(
            (ROOT / "configs/passive_pinball_backend.json").read_text(
                encoding="utf-8"
            )
        )
        fixture = build_fixture(config)
        radius = float(config["dynamic_object"]["radius_m"])
        normal = np.asarray(fixture["frame"]["normal"], dtype=np.float64)
        top = np.asarray(config["fixture"]["top_center_m"], dtype=np.float64)
        for profile_id in (
            "two_ball_top_collision",
            "two_ball_offset_collision",
            "two_ball_diagonal_catch_up_collision",
        ):
            _, profile = self.profiles[profile_id]
            states = resolve_pinball_initial_states(
                profile,
                fixture_source=config["fixture"],
                fixture_frame=fixture["frame"],
                ball_radius_m=radius,
            )
            normal_offsets = [
                float((np.asarray(state["position_m"]) - top) @ normal)
                for state in states
            ]
            expected = (
                float(config["fixture"]["board_thickness_m"]) / 2.0
                + radius
                + 0.0005
            )
            self.assertTrue(np.allclose(normal_offsets, expected, atol=1.0e-12))

    def test_marble_profiles_preserve_the_reviewed_track_pose(self) -> None:
        backend = json.loads(
            (ROOT / "configs/marble_run_backend.json").read_text(encoding="utf-8")
        )
        candidate = json.loads(
            (ROOT / backend["candidate_config"]["path"]).read_text(encoding="utf-8")
        )
        dynamic = candidate["dynamic_object"]
        base = np.asarray(dynamic["initial_state"]["position_m"], dtype=np.float64)
        radius = float(dynamic["radius_m"])
        for profile_id in (
            "two_marble_catch_up_collision",
            "two_marble_delayed_catch_up_collision",
            "two_marble_counterflow_collision",
        ):
            _, profile = self.profiles[profile_id]
            states = resolve_marble_run_initial_states(
                profile,
                base_initial_state=dynamic["initial_state"],
                ball_radius_m=radius,
            )
            for state in states:
                position = np.asarray(state["position_m"], dtype=np.float64)
                self.assertEqual(position[1], base[1])
                self.assertEqual(position[2], base[2])

    def test_specialized_camera_views_are_distinct_bounded_orbits(self) -> None:
        base = {
            "position_m": [0.5, 4.0, 1.8],
            "target_m": [0.0, 0.0, 1.0],
            "focal_length_mm": 50.0,
            "sensor_width_mm": 36.0,
        }
        for family in self.families.values():
            bindings = [
                resolve_specialized_camera_binding(family, profile, base)
                for profile in family["profiles"]
            ]
            positions = {
                tuple(round(value, 6) for value in binding["position_m"])
                for binding in bindings
            }
            self.assertEqual(len(positions), 3)
            self.assertTrue(
                all(binding["target_m"] == base["target_m"] for binding in bindings)
            )
            self.assertEqual(
                {binding["view_family_id"] for binding in bindings},
                {
                    view["id"]
                    for view in family["camera_contract"]["view_families"]
                },
            )


if __name__ == "__main__":
    unittest.main()
