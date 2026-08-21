from __future__ import annotations

import json
import random
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from sample_asset_proxy_scenes import audit_asset_trajectory  # noqa: E402
from sample_pybullet_base import (  # noqa: E402
    load_active_rules,
    weighted_scene_class_cycle,
)
from physics_time_step import simulation_hz_for_geometry  # noqa: E402


class PhysicsRuleInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = json.loads(
            (ROOT / "configs/pybullet_backend.json").read_text(encoding="utf-8")
        )
        cls.rules = load_active_rules(ROOT)

    @staticmethod
    def arrays(
        positions: list[list[float]],
        support: list[int],
        ground: list[int] | None = None,
    ) -> dict[str, np.ndarray]:
        count = len(positions)
        return {
            "time_s": np.arange(count, dtype=np.float64) / 24.0,
            "position_m": np.asarray(positions, dtype=np.float64),
            "quaternion_xyzw": np.tile(
                np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
                (count, 1),
            ),
            "linear_velocity_m_s": np.zeros((count, 3), dtype=np.float64),
            "angular_velocity_rad_s": np.zeros((count, 3), dtype=np.float64),
            "support_contact": np.asarray(support, dtype=np.int8),
            "ground_contact": np.asarray(
                ground if ground is not None else [0] * count, dtype=np.int8
            ),
            "prop_contact": np.zeros(count, dtype=np.int8),
        }

    def audit(
        self, profile: str, arrays: dict[str, np.ndarray], penetration: float = 0.0
    ) -> dict:
        return audit_asset_trajectory(
            profile,
            arrays,
            -penetration,
            0.0,
            0,
            [0.10, 0.10, 0.10],
            {"z_m": 0.78},
            self.backend["quality"],
            self.backend["asset_proxy_rules"],
            expected_motion={
                "minimum_active_duration_s": 0.0,
                "active_speed_threshold_m_s": 0.03,
            },
        )

    def test_thin_geometry_uses_more_internal_steps(self) -> None:
        thick_hz = simulation_hz_for_geometry(
            self.backend["engine"], [0.20, 0.20, 0.20]
        )
        thin_hz = simulation_hz_for_geometry(
            self.backend["engine"], [0.20, 0.10, 0.01]
        )
        self.assertGreaterEqual(
            thick_hz, self.backend["engine"]["minimum_simulation_hz"]
        )
        self.assertLess(thick_hz, thin_hz)
        self.assertEqual(
            thin_hz,
            self.backend["engine"]["adaptive_time_step"]["maximum_hz"],
        )
        self.assertEqual(thin_hz % self.backend["engine"]["output_fps"], 0)

    def test_relative_penetration_rejects_large_fraction(self) -> None:
        arrays = self.arrays(
            [[0.0, 0.0, 0.3], [0.0, 0.0, 0.15], [0.0, 0.0, 0.1], [0.0, 0.0, 0.1]],
            [0, 0, 1, 1],
        )
        audit = self.audit("vertical_drop", arrays, penetration=0.015)
        self.assertFalse(audit["checks"]["penetration_within_limit"])
        self.assertEqual(audit["penetration_limit_m"], 0.008)

    def test_edge_profile_requires_actual_exit_and_ground_contact(self) -> None:
        valid = self.arrays(
            [
                [0.0, 0.0, 0.82],
                [0.05, 0.0, 0.82],
                [0.10, 0.0, 0.82],
                [0.16, 0.0, 0.81],
                [0.22, 0.0, 0.70],
                [0.28, 0.0, 0.48],
                [0.34, 0.0, 0.20],
                [0.38, 0.0, 0.08],
            ],
            [1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        )
        valid_audit = self.audit("edge_exit", valid)
        self.assertTrue(valid_audit["passed"], valid_audit)

        stays_on_support = self.arrays(
            [[0.04 * index, 0.0, 0.82] for index in range(8)],
            [1] * 8,
        )
        invalid_audit = self.audit("edge_exit", stays_on_support)
        self.assertFalse(invalid_audit["checks"]["edge_exits_primary_support"])
        self.assertFalse(invalid_audit["checks"]["edge_contacts_ground_after_exit"])

    def test_small_weighted_cycles_are_seed_sensitive(self) -> None:
        values = set()
        for seed in range(100):
            values.update(
                weighted_scene_class_cycle(
                    self.rules["axes"]["scene_class_axis"],
                    self.rules["axes"]["support_axis"],
                    "bounce_1obj",
                    1,
                    random.Random(seed),
                    self.rules,
                )
            )
        self.assertEqual(values, {"ground_flat", "raised_flat", "raised_feature"})


if __name__ == "__main__":
    unittest.main()
