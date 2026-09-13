import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import generate
from tools.assets import prepare_runtime as setup
from tools.core.sampling_counts import allocate_counts


class PublicGenerationTests(unittest.TestCase):
    def test_two_object_request_does_not_load_three_object_pipeline(self):
        subprocess.run([sys.executable, '-c',
            "import sys, generate; "
            "request = generate.two_object_request(10, 31); "
            "assert sum(request['family_base_counts'].values()) == 10; "
            "assert not any('three_object' in name for name in sys.modules)"],
            cwd=generate.ROOT, check=True)

    def test_plan_needs_no_assets_or_processes(self):
        output = io.StringIO()
        with patch.object(generate.subprocess, 'run', side_effect=AssertionError('must not execute')), contextlib.redirect_stdout(output):
            generate.main(['--objects', 'all', '--count', '100', '--setup', '--plan-only'])
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['samples_per_object_count'], 1300)
        self.assertEqual(plan['objects'], [1, 2, 3])

    def test_two_object_quota_preserves_total_profiles_and_seed(self):
        for count in (10, 100, 3077):
            request = generate.two_object_request(count, 31)
            quotas = request['family_base_counts']
            self.assertEqual(sum(quotas.values()), count)
            self.assertTrue(all(3 <= quotas[f] <= 75 for f in ('billiards', 'passive_pinball', 'marble_run')))
            self.assertEqual(request['generic_coverage']['seed'], 31)
        with self.assertRaises(ValueError):
            generate.two_object_request(9, 31)

    def test_largest_remainder_covers_every_selected_mode(self):
        self.assertEqual(allocate_counts(10, {'a': 2, 'b': 1}, minimum=1), {'a': 6, 'b': 4})
        with self.assertRaises(ValueError):
            allocate_counts(1, {'a': 2, 'b': 1}, minimum=1)

    def test_download_publishes_only_matching_content(self):
        payload = b'complete asset'
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'model.glb'
            with patch.object(setup.urllib.request, 'urlopen', return_value=io.BytesIO(payload)) as network:
                setup.download('https://example.test/model', target, expected)
                setup.download('https://example.test/model', target, expected)
                self.assertEqual(network.call_count, 1)
            self.assertEqual(target.read_bytes(), payload)
            with self.assertRaises(ValueError):
                setup.download('https://example.test/model', target, '0'*64)
            self.assertEqual(target.read_bytes(), payload)

    def test_corrupt_download_leaves_no_partial_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'model.glb'
            with patch.object(setup.urllib.request, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(b'bad')), patch.object(setup.time, 'sleep'):
                with self.assertRaises(ValueError):
                    setup.download('https://example.test/model', target, '0'*64)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_archive_rejects_traversal_before_any_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root/'bad.tar'
            with tarfile.open(archive, 'w') as package:
                for name in ('good.txt', '../outside.txt'):
                    member = tarfile.TarInfo(name); member.size = 1
                    package.addfile(member, io.BytesIO(b'x'))
            destination = root/'extract'
            destination.mkdir()
            with self.assertRaises(ValueError):
                setup.extract_checked(archive, destination)
            self.assertEqual(list(destination.iterdir()), [])

    def test_archive_rejects_external_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); archive = root/'bad.tar'
            with tarfile.open(archive, 'w') as package:
                member = tarfile.TarInfo('link'); member.type = tarfile.SYMTYPE; member.linkname = '../../outside'
                package.addfile(member)
            with self.assertRaises(ValueError):
                setup.extract_checked(archive, root/'extract')

    def test_upstream_download_does_not_require_a_sketchfab_token(self):
        with patch.dict(setup.os.environ, {}, clear=True):
            self.assertEqual(setup.resource_url({'url': 'https://example.test/a'}), 'https://example.test/a')
            with self.assertRaisesRegex(RuntimeError, 'SKETCHFAB_API_TOKEN'):
                setup.resource_url({'sketchfab_uid': 'a'*32})

    def test_sketchfab_checks_policy_before_requesting_a_download(self):
        with patch.dict(setup.os.environ, {'SKETCHFAB_API_TOKEN': 'test-token'}), patch.object(setup, 'request_json', return_value={'license': {'slug': 'by-nc'}}) as network:
            with self.assertRaises(ValueError):
                setup.resource_url({'sketchfab_uid': 'a'*32})
            self.assertEqual(network.call_count, 1)

    def test_specialized_binding_satisfies_the_immutable_path_contract(self):
        from tools.rendering import prepare_three_object
        from tools.dataset_contract.immutable_scene_contract import validate_simulation_record
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root/'source.json'
            source.write_text(json.dumps({'scene_id': 'scene', 'render': {},
                'object_identity': {'trajectory': {}, 'instance_masks': {}}}))
            physical = {'schema_version': 'physweep_dispatched_simulation_record_v1', 'scene_id': 'scene',
                'metadata_path': str(source), 'metadata_sha256': setup.digest(source)}
            for field, filename in [('trajectory', 'trajectory.npz'), ('audit', 'audit.json')]:
                path = root/filename; path.write_bytes(b'immutable artifact')
                physical[field+'_path'] = str(path)
                physical[field+'_sha256'] = setup.digest(path)
            (root/'simulation_record.json').write_text(json.dumps(physical))
            group = {'family': 'billiards', 'camera': {'view': 'fixed'}, 'physics': physical}
            with patch.object(prepare_three_object, 'bind_render_implementation'):
                _, records = prepare_three_object.prepare_group(root, root/'render', group, [physical])
            path = root/records[0]['metadata_path']
            bound = json.loads(path.read_text())
            self.assertEqual(bound['physics']['trajectory_path'], 'trajectory.npz')
            self.assertEqual(bound['physics']['audit_path'], 'audit.json')
            _, trajectory, audit = validate_simulation_record(root=root, metadata_path=path, metadata=bound)
            self.assertEqual(trajectory, root/'trajectory.npz')
            self.assertEqual(audit, root/'audit.json')


if __name__ == '__main__':
    unittest.main()
