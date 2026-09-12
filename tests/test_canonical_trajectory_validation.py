"""Reject hash-valid but numerically invalid public trajectories."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.release.base_release_schema import TRAJECTORY_SCHEMA, canonical_trajectory, write_deterministic_npz
from tools.release.base_release_view import validate_trajectory_artifact


def fixture():
    arrays = {'schema_version': np.asarray(TRAJECTORY_SCHEMA), 'object_ids': np.asarray(['a', 'b']),
              'time_s': np.arange(3) / 24, 'position_m': np.zeros((3, 2, 3)),
              'quaternion_wxyz': np.tile([1., 0, 0, 0], (3, 2, 1)),
              'linear_velocity_m_s': np.zeros((3, 2, 3)), 'angular_velocity_rad_s': np.zeros((3, 2, 3)),
              'contact_count': np.zeros((3, 2), dtype=np.int32)}
    objects = [{'object_id': name, 'initial_state': {'position_m': [0., 0, 0], 'quaternion_wxyz': [1., 0, 0, 0],
                'linear_velocity_m_s': [0., 0, 0], 'angular_velocity_rad_s': [0., 0, 0]}} for name in ('a', 'b')]
    return arrays, {'physics': {'time': {'duration_s': 2/24, 'output_fps': 24}, 'objects': objects}}


class CanonicalTrajectoryValidationTests(unittest.TestCase):
    def test_public_verification_rejects_bad_numerics_shapes_time_and_initial_states(self):
        mutations = {
            'nonfinite': lambda a: a['position_m'].__setitem__((1, 0, 0), float('nan')),
            'position_shape': lambda a: a.update(position_m=np.zeros((3, 2, 2))),
            'velocity_shape': lambda a: a.update(linear_velocity_m_s=np.zeros((3, 1, 3))),
            'time_shape': lambda a: a.update(time_s=np.zeros((3, 1))),
            'wrong_fps': lambda a: a.update(time_s=np.arange(3) * 10.),
            'initial_position': lambda a: a['position_m'].__setitem__((0, 0, 0), 100.),
            'initial_velocity': lambda a: a['linear_velocity_m_s'].__setitem__((0, 0, 0), 1.),
            'initial_spin': lambda a: a['angular_velocity_rad_s'].__setitem__((0, 0, 0), 1.),
            'quaternion_norm': lambda a: a['quaternion_wxyz'].__setitem__((1, 0, 0), 2.),
            'initial_orientation': lambda a: a['quaternion_wxyz'].__setitem__((0, 0), [0., 1., 0., 0.]),
            'fractional_contact': lambda a: a.update(contact_count=np.ones((3, 2)) * .5),
            'negative_contact': lambda a: a['contact_count'].__setitem__((1, 0), -1),
            'overflow_contact': lambda a: a.update(contact_count=np.ones((3, 2), dtype=np.int64) * 2**32),
            'object_shape': lambda a: a.update(object_ids=np.asarray([['a', 'b']])),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trajectory.npz'
            for name, mutate in mutations.items():
                arrays, metadata = fixture(); mutate(arrays); write_deterministic_npz(path, arrays)
                with self.subTest(name=name), self.assertRaises(ValueError):
                    validate_trajectory_artifact(path, metadata)

    def test_valid_large_sweep_motion_and_equivalent_quaternion_sign_are_allowed(self):
        arrays, metadata = fixture(); arrays['position_m'][1:] = 1000.
        arrays['linear_velocity_m_s'][1:] = 1.e4; arrays['quaternion_wxyz'] *= -1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trajectory.npz'; write_deterministic_npz(path, arrays)
            validate_trajectory_artifact(path, metadata)

    def test_build_does_not_silently_truncate_fractional_contacts(self):
        arrays, _ = fixture(); arrays['contact_count'] = np.ones((3, 2)) * .5
        arrays.update(runtime_material=np.ones((2, 3)), inertia_diagonal_kg_m2=np.ones((2, 3)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trajectory.npz'; np.savez_compressed(path, **arrays)
            with self.assertRaisesRegex(ValueError, 'contact_count'):
                canonical_trajectory(path)
