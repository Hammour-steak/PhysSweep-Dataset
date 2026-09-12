import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.core.hashing import sha256_file


@unittest.skipUnless(importlib.util.find_spec('pybullet'),'requires physics runtime')
class ThreeObjectCheckpointTests(unittest.TestCase):
    def test_resume_rejects_same_content_at_a_different_source_path(self):
        from tools.cli.generate_three_object_dataset import simulation
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);output=root/'physics';output.mkdir()
            source=root/'metadata.json';source.write_text(json.dumps({'scene_id':'candidate'}))
            record={'scene_id':'candidate','audit_passed':True,'failed_checks':[]}
            for key in ('metadata','resolved_scene','trajectory','audit'):
                path=source if key=='metadata' else root/key
                if key!='metadata':path.write_bytes(b'unchanged artifact')
                record[key+'_path']=str(path);record[key+'_sha256']=sha256_file(path)
            (output/'simulation_record.json').write_text(json.dumps(record))
            with patch('tools.cli.generate_three_object_dataset.dispatch_simulation',side_effect=AssertionError('must reuse')):
                self.assertEqual(simulation(source,output,root,True),record)
                other=root/'another_metadata.json';other.write_bytes(source.read_bytes())
                with self.assertRaisesRegex(ValueError,'path changed'):simulation(other,output,root,True)
                (root/'audit').write_bytes(b'corrupted audit')
                with self.assertRaisesRegex(ValueError,'audit changed'):simulation(source,output,root,True)

    def test_group_visual_comparison_ignores_only_output_paths(self):
        from tools.cli.three_object_visual_stage import visual_payload
        base={'visualization':{'camera':{'position_m':[1,2,3]},'resource_binding':{'sha256':'bound'},
             'render':{'video_path':'base.mp4','inspection_frame_dir':'base','instance_mask_dir':'unused',
                       'color_management':{'exposure':0.0}},'binding_version':'base'}}
        sweep=copy.deepcopy(base);sweep['visualization']['render']['video_path']='sweep.mp4'
        sweep['visualization']['camera_inheritance']={'policy':'copied_from_parent_base'}
        self.assertEqual(visual_payload(base),visual_payload(sweep))
        sweep['visualization']['render']['color_management']['exposure']=.5
        self.assertNotEqual(visual_payload(base),visual_payload(sweep))

    def test_visual_camera_diagnostic_receives_the_resource_root(self):
        from tools.cli.three_object_visual_stage import _audit_bound_camera
        root=Path('/dataset_root')
        bound={'simulation': {'support': {'exact_static_binding': {'mesh': 'bound'}}}}
        trajectory={'time_s': [0.0]}
        camera={'profile': 'front_oblique'}
        expected={'passed': True}
        with patch(
            'tools.cli.three_object_visual_stage.audit_camera', return_value=expected
        ) as camera_audit:
            self.assertIs(
                _audit_bound_camera(root, bound, trajectory, camera),
                expected,
            )
        camera_audit.assert_called_once_with(
            bound, trajectory, camera, root=root
        )


if __name__=='__main__':unittest.main()
