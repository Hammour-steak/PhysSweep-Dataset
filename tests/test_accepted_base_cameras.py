import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.core.hashing import sha256_file
from tools.rendering.prepare_sweep_render_manifests import (
    apply_accepted_base_camera, load_accepted_base_cameras,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return {'path': str(path), 'sha256': sha256_file(path)}


def physics(root, name, metadata, trajectory_bytes):
    directory = root / 'physics' / name
    directory.mkdir(parents=True, exist_ok=True)
    trajectory = directory / 'trajectory.npz'
    trajectory.write_bytes(trajectory_bytes)
    audit = directory / 'trajectory_audit.json'
    write(audit, {'passed': True})
    record = {'scene_id': json.loads(Path(metadata['path']).read_text())['scene_id'],
              'metadata_path': metadata['path'], 'metadata_sha256': metadata['sha256'],
              'trajectory_path': str(trajectory), 'trajectory_sha256': sha256_file(trajectory),
              'audit_path': str(audit), 'audit_sha256': sha256_file(audit),
              'audit_passed': True, 'failed_checks': []}
    write(directory / 'simulation_record.json', record)
    return record


def camera_fixture(root, parent='parent', trajectory_bytes=b'current'):
    source = write(root / f'{parent}.json', {'scene_id': parent})
    original = physics(root, parent, source, trajectory_bytes)
    sweep = {'kind': 'base', 'parent_scene_id': parent,
             'parent_metadata_path': source['path'], 'parent_metadata_sha256': source['sha256']}
    base = write(root / f'{parent}__base.json', {'scene_id': f'{parent}__base', 'sweep': sweep})
    current = physics(root, f'{parent}__base', base, trajectory_bytes)
    row = {'parent_scene_id': parent, 'camera_review_passed': True,
           'source_metadata': source,
           'trajectory': {'path': original['trajectory_path'], 'sha256': original['trajectory_sha256']},
           'camera': {'position_m': [0, -3, 2], 'target_m': [0, 0, 0],
                      'focal_length_mm': 40, 'sensor_width_mm': 36},
           'context': {'backboard_y': -2.4}}
    return row, current


def load(root, rows, current):
    path = root / 'cameras.json'
    write(path, {'schema_version': 'physweep_accepted_base_cameras_v1', 'status': 'accepted',
                 'sample_count': len(rows), 'records': rows})
    return load_accepted_base_cameras(root, path, sha256_file(path), base_physics_by_parent=current)


class AcceptedBaseCameraTests(unittest.TestCase):
    def test_verified_camera_inherits_without_changing_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            row, current = camera_fixture(root)
            cameras = load(root, [row], {'parent': current})
            source = row['source_metadata']
            bound = {'sweep': {'parent_scene_id': 'parent', 'parent_metadata_path': source['path'],
                              'parent_metadata_sha256': source['sha256']},
                     'simulation': {'unchanged': True}}
            original = copy.deepcopy(bound)
            apply_accepted_base_camera(root, bound, cameras, {'path': 'cameras.json', 'sha256': sha256_file(root / 'cameras.json')})
            self.assertEqual(bound['simulation'], original['simulation'])
            self.assertEqual(bound['camera'], row['camera'])
            self.assertEqual(bound['render']['context'], row['context'])
            self.assertEqual(bound['camera_inheritance']['current_base_physics']['sha256'],
                             sha256_file(Path(current['trajectory_path']).with_name('simulation_record.json')))
            bound['sweep']['parent_metadata_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'different parent'):
                apply_accepted_base_camera(root, bound, cameras, {})

    def test_another_parents_valid_hashed_trajectory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            row, current = camera_fixture(root)
            other, _ = camera_fixture(root, 'other', b'other trajectory')
            row['trajectory'] = other['trajectory']
            with self.assertRaisesRegex(ValueError, 'different parent'):
                load(root, [row], {'parent': current})

    def test_old_valid_trajectory_of_same_parent_is_rejected_after_resimulation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            row, current = camera_fixture(root)
            current = physics(root, 'parent__base',
                              {'path': current['metadata_path'], 'sha256': current['metadata_sha256']}, b'new simulation')
            with self.assertRaisesRegex(ValueError, 'current parent base physics'):
                load(root, [row], {'parent': current})

    def test_review_can_use_the_canonical_base_trajectory_directly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            row, current = camera_fixture(root)
            row['trajectory'] = {'path': current['trajectory_path'], 'sha256': current['trajectory_sha256']}
            self.assertIn('parent', load(root, [row], {'parent': current}))

    def test_unbound_or_tampered_physics_cannot_supply_camera_evidence(self):
        for mutation in ('missing_parent', 'missing_record', 'metadata_hash', 'trajectory_hash', 'manifest_hash'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                row, current = camera_fixture(root)
                index = {'parent': current}
                if mutation == 'missing_parent':
                    index = {}
                elif mutation == 'missing_record':
                    Path(row['trajectory']['path']).with_name('simulation_record.json').unlink()
                elif mutation == 'metadata_hash':
                    Path(current['metadata_path']).write_text('{}')
                elif mutation == 'trajectory_hash':
                    Path(row['trajectory']['path']).write_bytes(b'tampered')
                if mutation == 'manifest_hash':
                    load(root, [row], index)
                    with self.assertRaisesRegex(ValueError, 'manifest hash'):
                        load_accepted_base_cameras(root, root / 'cameras.json', 'wrong', base_physics_by_parent=index)
                else:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        load(root, [row], index)


if __name__ == '__main__':
    unittest.main()
