from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from tools.cli import generate_two_object_dataset as generator
from tools.core.hashing import sha256_json
from tools.core.json_io import read_json
from tools.motion_rules.two_object.specialized import load_two_object_specialized_rules
from tools.sampling.two_object_sampling_request import validate_sampling_request, varied_profiles
from tools.cli.dataset_generation import generation_layout

ROOT = Path(__file__).resolve().parents[1]


class MetadataStageTests(unittest.TestCase):
    def setUp(self):
        self.rules = load_two_object_specialized_rules(ROOT)
        self.request = read_json(ROOT/'configs/two_object_production_sampling.json')

    def test_production_quota_and_effective_matrix_are_explicit(self):
        before = (ROOT/generator.TWO_OBJECT_MATRIX).read_bytes()
        _, matrix, counts, limit = generator.sampling_settings(ROOT, Path('configs/two_object_production_sampling.json'), None)
        self.assertEqual(counts, {'generic':2855,'billiards':75,'passive_pinball':75,'marble_run':72})
        self.assertEqual(limit,2855)
        self.assertEqual(sum(counts.values())*13,40001)
        self.assertEqual(matrix['coverage_plan']['replicates_per_cell'],3)
        self.assertEqual(matrix['coverage_plan']['selection_policy']['maximum_object_source_reuse'],5)
        self.assertEqual(matrix['coverage_plan']['selection_policy']['maximum_host_source_reuse'],3)
        self.assertEqual(before,(ROOT/generator.TWO_OBJECT_MATRIX).read_bytes())
        with self.assertRaisesRegex(ValueError,'conflicts'):
            generator.sampling_settings(ROOT,Path('configs/two_object_production_sampling.json'),31)
        _, _, old_counts, _ = generator.sampling_settings(ROOT,None,31)
        self.assertEqual(sum(old_counts.values()),9)

    def test_specialized_variation_has_exact_unique_reproducible_physical_states(self):
        total=0
        for family in self.rules['scene_families']:
            name=family['id']
            count=self.request['family_base_counts'][name]
            variation=self.request['specialized_variation'][name]
            rows=list(varied_profiles(family,count,variation,20260905))
            self.assertEqual(len(rows),count)
            self.assertEqual(len({sha256_json(row[0]['objects']) for row in rows}),count)
            reversed_family=copy.deepcopy(family)
            reversed_family['profiles'].reverse()
            reversed_grid={axis:list(reversed(values)) for axis,values in variation.items()}
            self.assertEqual(rows,list(varied_profiles(reversed_family,count,reversed_grid,20260905)))
            originals={profile['id']:profile for profile in family['profiles']}
            for profile,identity in rows:
                self.assertEqual(profile['quality'],originals[profile['id']]['quality'])
                self.assertEqual(profile['camera_view_family_id'],originals[profile['id']]['camera_view_family_id'])
                self.assertEqual(profile['contact_requirement'],'must_contact')
                self.assertEqual(identity['initial_profile_sha256'],sha256_json(profile['objects']))
            total+=count
        self.assertEqual(total,222)

    def test_saved_effective_matrix_round_trips_through_the_actual_sampler(self):
        _, matrix, _, _ = generator.sampling_settings(ROOT,Path('configs/two_object_production_sampling.json'),None)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sampling_matrix.json'
            generator.freeze_effective_matrix(path,matrix)
            cells,_=generator.coverage_cells(read_json(path),31,scene_rules=read_json(ROOT/generator.TWO_OBJECT_SCENE_RULES))
            self.assertEqual(len(cells),31)
            before=(path.read_bytes(),path.stat().st_mtime_ns)
            generator.freeze_effective_matrix(path,matrix)
            self.assertEqual(before,(path.read_bytes(),path.stat().st_mtime_ns))
            path.write_text(json.dumps(matrix,sort_keys=True))
            with self.assertRaisesRegex(ValueError,'matrix changed'):
                generator.freeze_effective_matrix(path,matrix)

    def test_invalid_or_duplicate_variation_fails_before_sampling(self):
        for value in (True,0,-1,1.5):
            invalid=copy.deepcopy(self.request)
            invalid['family_base_counts']['billiards']=value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_sampling_request(invalid,self.rules)
        for values in ([1,1],[float('nan')],[0.0],[]):
            invalid=copy.deepcopy(self.request)
            invalid['specialized_variation']['billiards']['speed_scales']=values
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_sampling_request(invalid,self.rules)
        invalid=copy.deepcopy(self.request)
        invalid['family_base_counts']['billiards']=76
        with self.assertRaisesRegex(ValueError,'capacity'):
            validate_sampling_request(invalid,self.rules)

    def test_metadata_checkpoint_stops_and_resumes_without_admission_or_release(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root=Path(directory)
            args=argparse.Namespace(root=root,work_id='metadata_test',release_root=Path('outputs/two_object'),
                released_base_manifest=root/'released.json',source_root=root,source_manifest=Path('source.json'),
                billiards_template=root/'b.json',passive_pinball_template=root/'p.json',marble_run_template=root/'m.json',
                generic_limit=1,specialized_seed=7,physics_workers=1,render_workers=1,gpus='0',
                max_admission_attempts=8,resume=False,plan_only=False,metadata_only=True,sampling_config=None)
            counts={'generic':1,'billiards':3,'passive_pinball':3,'marble_run':3}
            sampled={'object_count':2,'family_counts':counts,'sample_count':10,'records':[],
                     'status':'sampled_pending_simulation'}
            stack.enter_context(patch.object(generator,'parse_args',return_value=args))
            stack.enter_context(patch.object(generator,'generation_plan',return_value={'test':'same plan'}))
            stack.enter_context(patch.object(generator,'sampling_settings',return_value=(None,{},counts,1)))
            assembler=stack.enter_context(patch.object(generator,'assemble_base_manifest',return_value=sampled))
            def fake_sampler(command,*,completion,resume):
                self.assertIn(command[2],('tools.sampling.sample_two_object_coverage','tools.sampling.sample_two_object_specialized'))
                if not completion.exists():
                    completion.parent.mkdir(parents=True,exist_ok=True)
                    completion.write_text('{}')
            stack.enter_context(patch.object(generator,'run_once',side_effect=fake_sampler))
            admission=stack.enter_context(patch.object(generator,'admit_two_object_groups',side_effect=RuntimeError('reached admission')))
            publish=stack.enter_context(patch.object(generator,'publish_dataset'))
            generator.main()
            layout=generator.generation_layout(root,args.work_id,args.release_root)
            checkpoint=layout.base_dataset/'sampled_manifest.json'
            before=(checkpoint.read_bytes(),checkpoint.stat().st_mtime_ns)
            self.assertFalse(layout.base_manifest.exists())
            self.assertFalse(layout.sweep_metadata.exists())
            self.assertFalse(layout.canonical_release.exists())
            admission.assert_not_called(); publish.assert_not_called()
            args.resume=True
            generator.main()
            self.assertEqual(before,(checkpoint.read_bytes(),checkpoint.stat().st_mtime_ns))
            self.assertEqual(assembler.call_count,2)
            args.metadata_only=False
            with self.assertRaisesRegex(RuntimeError,'reached admission'):
                generator.main()
            admission.assert_called_once()
            checkpoint.write_text('{}')
            with self.assertRaisesRegex(ValueError,'frozen checkpoint'):
                generator.main()

    def test_one_and_two_object_directories_share_the_same_layout(self):
        root=ROOT
        one=generation_layout(root,'same_work',Path('outputs/one_object'),object_count=1)
        two=generation_layout(root,'same_work',Path('outputs/two_object'),object_count=2)
        for field in ('base_dataset','base_manifest','sweep_metadata','sweep_physics','source_release','base_render','sweep_render'):
            self.assertEqual(getattr(one,field),getattr(two,field))
        self.assertEqual(one.canonical_release.parent,two.canonical_release.parent)
        with self.assertRaises(ValueError):
            generator.generation_layout(root,'same_work',Path('outputs/one_object'))


if __name__=='__main__':
    unittest.main()
