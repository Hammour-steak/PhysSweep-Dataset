import copy
import tempfile
import unittest
from pathlib import Path
from tools.cli.three_object_visual_stage import unfinished_render
from tools.core.hashing import sha256_file


class ThreeObjectRenderRecoveryTests(unittest.TestCase):
    def test_only_a_bound_recorded_render_failure_is_retryable(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'bound.json';path.write_text('{}')
            samples=[{'scene_id':f'scene_{i}'} for i in range(13)]
            result={'schema_version':'physweep_pybullet_render_manifest_v1','render_scope':'full_animation',
                'source_manifest':'bound.json','source_manifest_sha256':sha256_file(path),'sample_count':13,'success_count':12,'failure_count':1,
                'records':[{'scene_id':r['scene_id'],'ok':i!=4} for i,r in enumerate(samples)]}
            self.assertTrue(unfinished_render(result,samples,path,root))
            complete=copy.deepcopy(result);complete['records'][4]['ok']=True;complete.update(success_count=13,failure_count=0)
            self.assertFalse(unfinished_render(complete,samples,path,root))
            for change in ('source','duplicates','counts','scope'):
                changed=copy.deepcopy(result)
                if change=='source':changed['source_manifest_sha256']='different'
                elif change=='duplicates':changed['records'][0]['scene_id']=changed['records'][1]['scene_id']
                elif change=='counts':changed['failure_count']=0
                else:changed['render_scope']='first_frame_only'
                with self.assertRaises(ValueError):unfinished_render(changed,samples,path,root)


if __name__=='__main__':unittest.main()
