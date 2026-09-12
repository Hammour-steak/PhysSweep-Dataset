import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.three_object_fixtures import scene
from tools.cli.dataset_generation import generation_layout
from tools.cli.three_object_pilot_physics import completed_groups,ready_candidate_schema_versions
from tools.core.hashing import sha256_file
from tools.core.json_io import write_json_atomic_sorted
from tools.motion_rules.three_object.motion import apply_motion
from tools.sampling.three_object_sampling_request import load_pilot_rules


class ThreeObjectPilotCandidateFilteringTests(unittest.TestCase):
    def test_unavailable_metadata_cell_is_skipped_during_schema_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'metadata.json'
            write_json_atomic_sorted(path,{'schema_version':'physweep_three_object_scene_v2'})
            records=[
                {'status':'metadata_unavailable_no_reallocation'},
                {'status':'metadata_ready','metadata_path':str(path)},
            ]
            self.assertEqual(
                ready_candidate_schema_versions(records),
                {'physweep_three_object_scene_v2'},
            )


@unittest.skipUnless(importlib.util.find_spec('pybullet'),'requires real PyBullet')
class ThreeObjectPilotExecutionTests(unittest.TestCase):
    def test_worker_count_preserves_rejected_inputs_arrays_order_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);records=[]
            for index in range(2):
                m=scene();m['scene_id']=f'bounded_pilot_{index}'
                apply_motion(m,load_pilot_rules(),role_order=['P','Q','R'],speed_m_s=.3,spacing_ratio=4.,template_id='chain_transfer')
                path=root/'inputs'/m['scene_id']/'metadata.json';write_json_atomic_sorted(path,m)
                records.append({'scene_id':m['scene_id'],'metadata_path':str(path),'metadata_sha256':sha256_file(path),
                                'cell':{'template':'chain_transfer','target_role':'P'}})
            results=[]
            for workers in (1,2):
                wid=f'workers_{workers}';layout=generation_layout(root,wid,Path(f'outputs/staged/{wid}/three_object'),object_count=3)
                jobs=[(root,layout,record,False) for record in records]
                completed=list(completed_groups(jobs,workers));results.append(completed)
                self.assertEqual([r[0]['scene_id'] for r in completed],[r['scene_id'] for r in records])
                self.assertTrue(all(r[0]['integrity_passed'] and not r[0]['audit_passed'] and r[1] is None for r in completed))
                before={str(p):sha256_file(p) for p in root.rglob('*') if p.is_file()}
                resumed=list(completed_groups([(r,l,m,True) for r,l,m,_ in jobs],3-workers))
                self.assertEqual(resumed,completed)
                self.assertEqual(before,{str(p):sha256_file(p) for p in root.rglob('*') if p.is_file()})
            for first,second in zip(*results):
                self.assertEqual(first[0]['semantic_failures'],second[0]['semantic_failures'])
                with np.load(first[0]['physics']['trajectory_path']) as a,np.load(second[0]['physics']['trajectory_path']) as b:
                    self.assertEqual(set(a.files),set(b.files))
                    for key in a.files:np.testing.assert_array_equal(a[key],b[key])

    def test_failed_wave_never_submits_later_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);layout=generation_layout(root,'failure_wave',Path('outputs/staged/three_object'),object_count=3)
            records=[{'metadata_path':str(root/f'missing_{i}.json'),'metadata_sha256':'incorrect','scene_id':f'missing_{i}'} for i in range(3)]
            with self.assertRaises(FileNotFoundError):list(completed_groups([(root,layout,r,False) for r in records],2))
            self.assertFalse(layout.base_dataset.exists())


if __name__=='__main__':unittest.main()
