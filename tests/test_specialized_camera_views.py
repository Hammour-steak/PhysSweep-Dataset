from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from camera_geometry import (  # noqa: E402
    blocker_safe_seeded_view_order,
    seeded_view_order,
)


class SpecializedCameraViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.views = json.loads(
            (ROOT / "configs/visual_sampling.json").read_text(encoding="utf-8")
        )["specialized_camera_views"]

    def test_all_specialized_profiles_have_a_view_pool(self) -> None:
        self.assertEqual(
            set(self.views),
            {
                "resting_push",
                "diagonal_push",
                "vertical_drop",
                "edge_exit",
                "workbench_clear_zone_drop",
                "workbench_long_axis_push",
                "single_ball_free_roll",
                "single_ball_rail_rebound",
                "three_ball_collision",
            },
        )
        for profile, rule in self.views.items():
            horizontal_key = (
                "relative_azimuth_degrees"
                if profile == "edge_exit"
                else "yaw_offset_degrees"
                if profile.startswith("single_ball_") or profile == "three_ball_collision"
                else "azimuth_degrees"
            )
            self.assertGreaterEqual(len(set(rule[horizontal_key])), 5, profile)
            self.assertGreaterEqual(len(set(rule["elevation_degrees"])), 3, profile)

    def test_seeded_order_uses_the_whole_pool(self) -> None:
        values = [-70, -52, -34, -18, 18, 34, 52, 70]
        selected = {
            seeded_view_order(values, hashlib.sha256(str(seed).encode()).digest(), 0)[0]
            for seed in range(256)
        }
        self.assertEqual(selected, set(map(float, values)))

    def test_billiards_views_have_no_near_duplicate_yaws(self) -> None:
        expected = [-30, -15, 0, 15, 30]
        for profile in (
            "single_ball_free_roll",
            "single_ball_rail_rebound",
            "three_ball_collision",
        ):
            yaws = self.views[profile]["yaw_offset_degrees"]
            self.assertEqual(yaws, expected, profile)
            self.assertTrue(
                all(right - left >= 15 for left, right in zip(yaws, yaws[1:])),
                profile,
            )

    def test_blocker_filter_keeps_safe_views_diverse(self) -> None:
        values = [-70, -52, -34, -18, 18, 34, 52, 70]
        selected = {
            blocker_safe_seeded_view_order(
                values,
                hashlib.sha256(str(seed).encode()).digest(),
                0,
                (0.0, 1.0),
            )[0]
            for seed in range(256)
        }
        self.assertGreaterEqual(len(selected), 4)
        self.assertTrue(selected.issubset(set(map(float, values))))


if __name__ == "__main__":
    unittest.main()
