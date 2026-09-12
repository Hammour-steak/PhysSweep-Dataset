from pathlib import Path
import tempfile
import unittest

from tools.core.hashing import sha256_file
from tools.core.json_io import write_json
from tools.sampling.released_object_sources import verified_generation_records, RELEASE_SCHEMA_PAIRS, declared_within


class ReleasedObjectSourceTests(unittest.TestCase):
    def fixture(self, root, release_schema='physweep_base_release_view_v15', *, lineage=None, sample_schema=None):
        pipeline_schema, expected_sample_schema=RELEASE_SCHEMA_PAIRS[release_schema]
        raw={'schema_version':'physweep_pybullet_rigid_metadata_v1','scene_id':'source__base','simulation':{'objects':[{'object_id':'object_a'}]}}
        raw_path=root/'raw/metadata.json';raw_path.parent.mkdir(parents=True,exist_ok=True);write_json(raw_path,raw)
        generation={'schema_version':'physweep_release_metadata_manifest_v2','dataset_id':'fixture','sample_count':1,'group_count':1,
            'records':[{'scene_id':'source__base','kind':'base','source_schema_version':raw['schema_version'],
                        'path':'raw/metadata.json','metadata_sha256':sha256_file(raw_path)}]}
        gen_path=root/'generation.json';write_json(gen_path,generation)
        sample_path=root/'release/generic/source__base/metadata.json';sample_path.parent.mkdir(parents=True,exist_ok=True)
        write_json(sample_path,{'schema_version':sample_schema or expected_sample_schema,'scene_id':'source__base',
            'lineage':{'source_generation_metadata_sha256':lineage or sha256_file(raw_path)}})
        pipe_path=root/'release/generic/manifest.json'
        write_json(pipe_path,{'schema_version':pipeline_schema,'pipeline':'generic','sample_count':1,
            'records':[{'scene_id':'source__base','metadata_sha256':sha256_file(sample_path)}]})
        release_path=root/'release/manifest.json'
        write_json(release_path,{'schema_version':release_schema,'dataset_id':'fixture','sample_count':1,
            'pipelines':{'generic':{'manifest':'generic/manifest.json','manifest_sha256':sha256_file(pipe_path)}},
            'provenance':{'source_generation_release_metadata':{'schema_version':generation['schema_version'],'manifest_sha256':sha256_file(gen_path)}}})
        return dict(root=root,released_base_manifest_path=release_path,source_root=root,source_manifest_path=gen_path,
            source_contract={'released_base_manifest_schema_version':release_schema,'generation_manifest_schema_version':generation['schema_version'],'sample_kind':'base'},
            family_schemas={raw['schema_version']:'generic'})

    def test_current_and_frozen_schema_pairs_resolve_raw_metadata(self):
        for schema in RELEASE_SCHEMA_PAIRS:
            with self.subTest(schema=schema),tempfile.TemporaryDirectory() as tmp:
                rows=verified_generation_records(**self.fixture(Path(tmp),schema))
                self.assertEqual(len(rows),1)
                self.assertEqual(rows[0]['metadata']['schema_version'],'physweep_pybullet_rigid_metadata_v1')
                self.assertNotIn('admission',rows[0]['source'])

    def test_mixed_schema_versions_and_wrong_lineage_rejected(self):
        for mutation in ({'lineage':'0'*64},{'sample_schema':'physweep_base_sample_v11'}):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                args=self.fixture(Path(tmp),**mutation)
                with self.assertRaises(ValueError):verified_generation_records(**args)

    def test_modified_raw_or_pipeline_file_rejected(self):
        for path in ('raw/metadata.json','release/generic/manifest.json','generation.json'):
            with self.subTest(path=path),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);args=self.fixture(root)
                with (root/path).open('a') as f:f.write(' ')
                with self.assertRaises(ValueError):verified_generation_records(**args)

    def test_parent_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):declared_within(Path(tmp),Path('../outside.json'))
