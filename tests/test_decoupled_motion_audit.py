from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from tools.physics.audit_decoupled_motion import trajectory_metrics  # noqa: E402


class DecoupledMotionAuditTests(unittest.TestCase):
    def test_multibody_arrays_use_the_primary_dynamic_body(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory.npz"
            position = np.zeros((5, 3, 3), dtype=np.float64)
            position[:, 0, 0] = np.linspace(0.0, 0.4, 5)
            position[:, 1, 0] = 100.0
            velocity = np.zeros((5, 3, 3), dtype=np.float64)
            velocity[:, 0, 0] = np.linspace(1.0, 0.0, 5)
            velocity[:, 1, 0] = 100.0
            np.savez_compressed(
                path,
                position_m=position,
                linear_velocity_m_s=velocity,
                support_contact=np.ones(5, dtype=np.int8),
                ground_contact=np.zeros(5, dtype=np.int8),
            )
            metrics = trajectory_metrics(path, 4.0)
        self.assertEqual(metrics["planar_displacement_m"], 0.4)
        self.assertEqual(metrics["initial_speed_m_s"], 1.0)


if __name__ == "__main__":
    unittest.main()
