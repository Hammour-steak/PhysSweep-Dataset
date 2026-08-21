#!/usr/bin/env python3
"""Contracts for the grouped one-object motion-rule registry."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = PROJECT_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from motion_rules import (  # noqa: E402
    asset_motion_group,
    motion_group,
    registered_asset_motion_profiles,
    registered_motion_families,
)


EXPECTED_GROUPS = {
    "planar": {"slide_push_1obj", "roll_or_slide_1obj"},
    "ballistic": {
        "drop_fall_1obj",
        "projectile_1obj",
        "arc_projectile_1obj",
        "bounce_1obj",
    },
    "incline": {
        "slope_slide_down_1obj",
        "slope_slide_up_1obj",
        "ramp_to_flat_1obj",
    },
    "transition": {"wall_impact_1obj", "edge_fall_1obj"},
}

EXPECTED_ASSET_PROFILES = {
    "vertical_drop": "ballistic",
    "workbench_clear_zone_drop": "ballistic",
    "resting_push": "planar",
    "diagonal_push": "planar",
    "workbench_long_axis_push": "planar",
    "edge_exit": "transition",
}


class MotionRuleRegistryTests(unittest.TestCase):
    def test_generic_registry_has_the_complete_motion_matrix(self) -> None:
        expected = set().union(*EXPECTED_GROUPS.values())
        self.assertEqual(registered_motion_families(), expected)

    def test_each_generic_motion_has_one_expected_group(self) -> None:
        for group, motions in EXPECTED_GROUPS.items():
            for motion in motions:
                with self.subTest(motion=motion):
                    self.assertEqual(motion_group(motion), group)

    def test_curated_asset_profiles_use_the_same_physical_groups(self) -> None:
        self.assertEqual(
            registered_asset_motion_profiles(), set(EXPECTED_ASSET_PROFILES)
        )
        for profile, group in EXPECTED_ASSET_PROFILES.items():
            with self.subTest(profile=profile):
                self.assertEqual(asset_motion_group(profile), group)

    def test_unknown_motion_names_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported motion family"):
            motion_group("unclassified_motion")
        with self.assertRaisesRegex(
            ValueError, "unsupported curated-asset motion profile"
        ):
            asset_motion_group("unclassified_profile")


if __name__ == "__main__":
    unittest.main()
