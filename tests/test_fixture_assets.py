"""Reject broken collision geometry in portable releases."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.core.hashing import sha256_file
from tools.release.base_release_schema import FIXTURE_SCHEMA
from tools.release.fixture_assets import verify_fixture_catalog_files


class FixtureAssetsTests(unittest.TestCase):
    def test_missing_corrupt_and_unbound_assets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'fixtures').mkdir(); (root / 'fixture_assets').mkdir()
            (root / 'fixtures/manifest.json').write_text('{}')
            asset = root / 'fixture_assets/track.obj'
            asset.write_text('v 0 0 0\n')
            fixture = {'schema_version': FIXTURE_SCHEMA, 'physical': {
                'mesh': {'path': 'fixture_assets/track.obj', 'sha256': sha256_file(asset)}}}
            temporary = root / 'fixture.json'; temporary.write_text(json.dumps(fixture))
            digest = sha256_file(temporary)
            temporary.rename(root / f'fixtures/{digest}.json')
            records = [{'sha256': digest, 'usage_count': 1}]
            self.assertEqual(verify_fixture_catalog_files(root, records), {digest: 1})
            original = asset.read_bytes(); asset.unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                verify_fixture_catalog_files(root, records)
            asset.write_text('wrong geometry')
            with self.assertRaises(ValueError):
                verify_fixture_catalog_files(root, records)
            asset.write_bytes(original)
            extra = root / 'fixture_assets/unreferenced.obj'; extra.write_text('extra')
            with self.assertRaisesRegex(ValueError, 'unexpected fixture assets'):
                verify_fixture_catalog_files(root, records)
            extra.unlink()
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                verify_fixture_catalog_files(root, records * 2)

    def test_external_or_noncanonical_collision_paths_are_rejected(self):
        for relative in ('../track.obj', '/tmp/track.obj', 'fixture_assets/../track.obj',
                         'fixture_assets//track.obj', 'fixture_assets\\track.obj'):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); (root / 'fixtures').mkdir(); (root / 'fixture_assets').mkdir()
                (root / 'fixtures/manifest.json').write_text('{}')
                fixture = {'schema_version': FIXTURE_SCHEMA, 'physical': {'mesh': {
                    'path': relative, 'sha256': 'a' * 64}}}
                temporary = root / 'fixture.json'; temporary.write_text(json.dumps(fixture))
                digest = sha256_file(temporary); temporary.rename(root / f'fixtures/{digest}.json')
                with self.assertRaisesRegex(ValueError, 'asset path'):
                    verify_fixture_catalog_files(root, [{'sha256': digest, 'usage_count': 1}])
