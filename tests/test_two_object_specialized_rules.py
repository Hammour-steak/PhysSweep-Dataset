from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.assets.visual_environment_binding import (
    choose_specialized_environment,
    render_only_backdrop_objects,
)
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
            background_ids = set(family["background_contract"]["profile_ids"])
            self.assertEqual(len(background_ids), 3)
            self.assertEqual(
                {profile["background_profile_id"] for profile in family["profiles"]},
                background_ids,
            )
            self.assertFalse(family["background_contract"]["collision_enabled"])

    def test_marble_run_declares_its_audited_friction_domain(self) -> None:
        self.assertEqual(
            self.families["marble_run"]["sweep_domains"],
            {"contact_friction": [0.05, 1.0]},
        )
        self.assertNotIn("sweep_domains", self.families["billiards"])
        self.assertNotIn("sweep_domains", self.families["passive_pinball"])

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

    def test_specialized_backdrop_is_render_only_and_behind_the_fixture(self) -> None:
        scene_profiles = json.loads(
            (ROOT / "configs/scene_visual_profiles.json").read_text(encoding="utf-8")
        )["profiles"]
        by_id = {profile["id"]: profile for profile in scene_profiles}
        camera = {"position_m": [0.0, 4.0, 2.0]}
        for family in self.families.values():
            minimum = family["background_contract"][
                "minimum_back_wall_distance_m"
            ]
            for profile_id in family["background_contract"]["profile_ids"]:
                objects = render_only_backdrop_objects(
                    by_id[profile_id], camera, [0.0, 0.0, 0.0], minimum
                )
                self.assertGreaterEqual(len(objects), 2)
                self.assertTrue(
                    all(record["collision_enabled"] is False for record in objects)
                )
                wall = objects[0]
                self.assertLessEqual(wall["position_m"][1], -float(minimum))

    def test_specialized_environment_binding_is_complete_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for material_id in ("floor", "wall"):
                texture_dir = (
                    root
                    / "assets/library/polyhaven/materials"
                    / material_id
                    / "textures"
                )
                texture_dir.mkdir(parents=True)
                for suffix in ("diff", "rough", "nor_gl"):
                    (texture_dir / f"{material_id}_{suffix}_4k.png").write_bytes(
                        suffix.encode("ascii")
                    )
            rules = {
                "hdri_roles_by_environment": {"minimal": ["studio_soft"]},
                "hdri_strength_ranges_by_environment": {"minimal": [0.2, 0.3]},
                "room_floor_pools_by_environment": {"minimal": ["floor"]},
                "wall_pools_by_environment": {"minimal": ["wall"]},
                "asset_proxy_render": {
                    "color_management": {},
                    "light_scale": {},
                    "area_lights": [],
                },
            }
            profile = {
                "id": "minimal_wall",
                "environment_category": "minimal",
                "back_wall_distance_m": 2.35,
                "decor": [],
            }
            contract = {
                "physics_role": "render_only_context",
                "profile_ids": ["minimal_wall"],
                "minimum_back_wall_distance_m": 2.6,
                "collision_enabled": False,
            }
            hdri = [
                {
                    "name": "studio",
                    "source_path": "assets/studio.hdr",
                    "sha256": "0" * 64,
                    "role": "studio_soft",
                    "tier": "primary",
                    "sample_weight": 1.0,
                }
            ]
            arguments = {
                "family_id": "billiards",
                "background_contract": contract,
                "scene_profile": profile,
                "camera": {"position_m": [0.0, 4.0, 2.0]},
                "scene_anchor_m": [0.0, 0.0, 0.0],
                "hdri_records": hdri,
                "visual_rules": rules,
            }
            first = choose_specialized_environment(
                root, **arguments, rng=random.Random(41)
            )
            second = choose_specialized_environment(
                root, **arguments, rng=random.Random(41)
            )
            self.assertEqual(first, second)
            self.assertEqual(first["physics_role"], "render_only_context")
            self.assertFalse(first["collision_enabled"])
            self.assertEqual(first["room"]["wall_mode"], "profile_backdrop_only")
            self.assertTrue(first["backdrop_objects"])


if __name__ == "__main__":
    unittest.main()
