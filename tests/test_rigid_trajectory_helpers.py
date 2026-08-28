from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from tools.motion_rules.one_object.common import (
    precontact_lateral_drift as _precontact_lateral_drift,
    sampled_extremum_tolerance as _sampled_extremum_tolerance,
)
from tools.physics.rigid_trajectory import (
    active_motion_duration_s,
    _distance_lower_bound,
    _distance_upper_bound,
)


class RigidTrajectoryHelperTests(unittest.TestCase):
    def test_precontact_drift_excludes_collision_impulse_frame(self) -> None:
        positions = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.8],
                [0.0, 0.0, 0.5],
                [0.04, 0.0, 0.55],
            ],
            dtype=np.float64,
        )
        self.assertEqual(_precontact_lateral_drift(positions, 3), 0.0)

    def test_distance_bounds_use_the_larger_tolerance(self) -> None:
        self.assertAlmostEqual(_distance_lower_bound(0.3, 0.0001, 0.02), 0.294)
        self.assertAlmostEqual(_distance_upper_bound(0.008, 0.0001, 0.02), 0.00816)

    def test_extremum_tolerance_scales_with_frame_interval_squared(self) -> None:
        at_24_fps = _sampled_extremum_tolerance(9.81, 24.0, 2.0)
        at_48_fps = _sampled_extremum_tolerance(9.81, 48.0, 2.0)
        self.assertAlmostEqual(at_24_fps, 4.0 * at_48_fps)

    def test_active_motion_duration_uses_the_last_useful_speed_sample(self) -> None:
        velocity = np.asarray(
            [
                [0.10, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [0.04, 0.0, 0.0],
                [0.00, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        time_s = np.arange(5, dtype=np.float64) / 4.0
        self.assertAlmostEqual(
            active_motion_duration_s(velocity, time_s, 0.03), 0.75
        )

    def test_active_motion_duration_rejects_mismatched_time_samples(self) -> None:
        with self.assertRaisesRegex(ValueError, "time samples"):
            active_motion_duration_s(
                np.zeros((3, 3), dtype=np.float64),
                np.zeros(2, dtype=np.float64),
                0.03,
            )


if __name__ == "__main__":
    unittest.main()
