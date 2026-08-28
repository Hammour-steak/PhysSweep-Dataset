from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from tools.dataset_contract.trajectory_contract import object_trajectory_view  # noqa: E402


class CameraTrajectoryContractTests(unittest.TestCase):
    def test_dispatch_layout_maps_objects_and_adapter_geometry(self) -> None:
        metadata = {"simulation": {"objects": [{"object_id": "object_a"}]}}
        values = np.zeros((3, 1, 3), dtype=np.float64)
        trajectory = {
            "object_ids": np.asarray(["object_a"]),
            "position_m": values,
            "quaternion_wxyz": np.zeros((3, 1, 4), dtype=np.float64),
            "linear_velocity_m_s": values,
            "angular_velocity_rad_s": values,
            "adapter__object_a__aabb_min_m": values[:, 0] - 0.1,
            "adapter__object_a__aabb_max_m": values[:, 0] + 0.1,
        }

        mapped = object_trajectory_view(metadata, trajectory)

        self.assertEqual(mapped["object_a__position_m"].shape, (3, 3))
        np.testing.assert_allclose(mapped["object_a__aabb_min_m"], -0.1)
        np.testing.assert_allclose(mapped["object_a__aabb_max_m"], 0.1)

    def test_dispatch_layout_rejects_object_order_mismatch(self) -> None:
        metadata = {"simulation": {"objects": [{"object_id": "object_a"}]}}
        with self.assertRaises(ValueError):
            object_trajectory_view(
                metadata, {"object_ids": np.asarray(["object_b"])}
            )


if __name__ == "__main__":
    unittest.main()
