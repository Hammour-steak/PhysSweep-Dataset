import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from tools.training_export.point_trajectory import (
    POINT_COUNT,
    build_point_trajectory,
    rigid_points_from_poses,
    validate_point_trajectory,
)


def _camera_files(root: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    camera_from_world = np.eye(4, dtype=np.float32)
    intrinsics = np.asarray(
        [[100.0, 0.0, 32.0], [0.0, 100.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    surface = root / "surface.npz"
    np.savez_compressed(
        surface,
        camera_from_world=camera_from_world,
        camera_intrinsics=intrinsics,
        image_size_px=np.asarray([64, 64], dtype=np.int32),
    )
    return surface, camera_from_world, intrinsics


class PointTrajectoryTest(unittest.TestCase):
    def test_rigid_points_preserve_identity_and_pose(self):
        initial = np.zeros((POINT_COUNT, 3), dtype=np.float32)
        initial[:, 2] = 2.0
        initial[0, :2] = [0.1, 0.0]
        positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        quaternions = np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2)
        world, camera, error = rigid_points_from_poses(
            initial,
            positions,
            quaternions,
            np.eye(4),
        )
        self.assertLess(error, 1.0e-6)
        np.testing.assert_allclose(camera[0], initial, atol=1.0e-6)
        np.testing.assert_allclose(camera[1, 0] - camera[0, 0], [1.0, 0.0, 0.0])

    def test_multibody_payload_preserves_fixed_object_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            surface, _, _ = _camera_files(root)
            points = np.zeros((POINT_COUNT, 3), dtype=np.float32)
            points[:, 2] = 2.0
            points[0, 0] = 0.1
            scene = root / "scene.npz"
            np.savez_compressed(
                scene,
                object_xyz_camera_m=np.stack([points, points + [0.0, 0.2, 0.0]]),
                object_ids=np.asarray(["obj_0", "obj_1"]),
                metadata_json=np.asarray(
                    json.dumps(
                        {
                            "source_surface": str(surface),
                            "object_ids": ["obj_0", "obj_1"],
                        }
                    )
                ),
            )
            trajectory = root / "trajectory.npz"
            np.savez_compressed(
                trajectory,
                time_s=np.asarray([0.0, 1.0], dtype=np.float32),
                obj_0__position_m=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
                obj_0__quaternion_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
                obj_1__position_m=np.asarray([[0.0, 0.2, 0.0], [0.0, 0.3, 0.0]]),
                obj_1__quaternion_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2),
            )
            payload = build_point_trajectory(scene, trajectory)
            validate_point_trajectory(payload)
            self.assertEqual(payload["points_world_m"].shape, (2, 2, POINT_COUNT, 3))
            self.assertEqual(payload["tracks_xy_px"].shape, (2, 2, POINT_COUNT, 2))
if __name__ == "__main__":
    unittest.main()
