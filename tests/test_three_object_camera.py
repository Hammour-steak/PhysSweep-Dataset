import copy
import unittest
import tempfile
from pathlib import Path
import numpy as np
from tests.three_object_fixtures import scene
from tools.rendering.camera_solver import solve_camera
from tools.rendering.three_object_camera import audit_camera,blocked_by_sphere,object_framing_passes
from tools.rendering.three_object_visuals import event_inspection_frames,frozen_lighting_report
from tools.assets.three_object_resources import resource_snapshot,validate_resources


def still_trajectory(metadata):
    result={}
    for obj in metadata['simulation']['objects']:
        key=obj['object_id'];radius=obj['geometry']['size_m'][0]/2
        positions=np.repeat([obj['initial_state']['position_m']],97,axis=0)
        result[key+'__position_m']=positions
        result[key+'__aabb_min_m']=positions-radius
        result[key+'__aabb_max_m']=positions+radius
    return result


class ThreeObjectCameraTest(unittest.TestCase):
    def test_texture_content_and_directory_changes_invalidate_binding(self):
        metadata=scene()
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'textures';folder.mkdir()
            (folder/'diffuse.png').write_bytes(b'original texture')
            metadata['appearance']['test_texture']={'path':'textures'}
            binding=resource_snapshot(root,metadata)
            validate_resources(root,metadata,binding)
            (folder/'diffuse.png').write_bytes(b'changed texture')
            with self.assertRaisesRegex(ValueError,'changed'):validate_resources(root,metadata,binding)
            (folder/'diffuse.png').write_bytes(b'original texture')
            (folder/'normal.png').write_bytes(b'new map')
            with self.assertRaisesRegex(ValueError,'changed'):validate_resources(root,metadata,binding)

    def test_all_objects_are_checked_and_camera_is_deterministic(self):
        metadata=scene();trajectory=still_trajectory(metadata)
        camera=solve_camera(metadata,trajectory,{})
        self.assertEqual(camera,solve_camera(metadata,trajectory,{}))
        self.assertTrue(camera['diagnostics']['admission']['passed'])
        self.assertEqual(len(camera['diagnostics']['admission']['objects']),3)
        changed=copy.deepcopy(trajectory)
        for key in changed:
            if key.startswith('object_c'):changed[key][-1,0]+=20
        audit=audit_camera(metadata,changed,camera)
        self.assertFalse(audit['objects']['object_c']['passed'])
        self.assertTrue(audit['objects']['object_a']['passed'])

    def test_selected_camera_obeys_two_sided_screen_fraction_limits(self):
        metadata=scene();trajectory=still_trajectory(metadata)
        camera=solve_camera(metadata,trajectory,{})
        self.assertEqual(camera['solver_version'],'three_object_base_camera_v2')
        rules=metadata['three_object']['camera_rules']
        for summary in camera['diagnostics']['admission']['objects'].values():
            self.assertGreaterEqual(summary['minimum_extent_fraction_of_short_side'],rules['minimum_object_extent_fraction_of_short_side'])
            self.assertLessEqual(summary['maximum_extent_fraction_of_short_side'],rules['maximum_object_extent_fraction_of_short_side'])
        self.assertIn('selection_score',camera['diagnostics'])

    def test_small_non_event_overflow_is_bounded(self):
        rules=scene()['three_object']['camera_rules']
        summary={
            'minimum_extent_fraction_of_short_side':.12,
            'maximum_extent_fraction_of_short_side':.20,
            'safe_frame_violation_fraction':.10,
            'out_of_frame_fraction':7/97,
            'maximum_frame_overflow_fraction':.02,
            'required_event_frames_inside':True,
            'minimum_unoccluded_proxy_sample_fraction':.9,
        }
        self.assertTrue(object_framing_passes(summary,rules))
        for key,value in (
            ('out_of_frame_fraction',8/97),
            ('maximum_frame_overflow_fraction',.031),
            ('required_event_frames_inside',False),
            ('minimum_extent_fraction_of_short_side',.049),
            ('maximum_extent_fraction_of_short_side',.251),
        ):
            changed=copy.deepcopy(summary);changed[key]=value
            self.assertFalse(object_framing_passes(changed,rules),key)

    def test_occlusion_is_segment_geometry_not_box_overlap(self):
        camera=np.array([0.,0.,0.]);points=np.array([[4.,0.,0.],[4.,2.,0.]])
        np.testing.assert_array_equal(blocked_by_sphere(camera,points,np.array([2.,0.,0.]),.3),[True,False])
        self.assertFalse(blocked_by_sphere(camera,points,np.array([6.,0.,0.]),.3).any())

    def test_static_wall_blocks_base_and_sweep_cannot_choose_camera(self):
        metadata=scene();trajectory=still_trajectory(metadata);camera=solve_camera(metadata,trajectory,{})
        metadata['environment_binding']['colliders']=[{'primitive':'box','size_m':[30,30,.2],
            'position_m':[0,0,.5],'rotation_euler_degrees':[0,0,0],'visible':True,'collision_enabled':True}]
        self.assertFalse(audit_camera(metadata,trajectory,camera)['passed'])
        metadata['sweep']={'kind':'sweep'}
        with self.assertRaisesRegex(ValueError,'reuse'):solve_camera(metadata,trajectory,{})

    def test_keyframes_bracket_subframe_contacts_and_limit_recontacts(self):
        evidence={'simulation_hz':1704,'events':[
            {'object_ids':['a','b'],'start_substep':480},
            {'object_ids':['b','c'],'start_substep':2361},
            {'object_ids':['a','b'],'start_substep':4000}]}
        frames=event_inspection_frames(evidence,{'frame_count':97,'output_fps':24})
        self.assertTrue({1,6,7,8,9,33,34,35,36,49,97}<=set(frames))
        self.assertLessEqual(len(frames),11)

    def test_fixed_exposure_requires_explicit_temporal_and_lighting_policy(self):
        config={'three_object_lighting_policy':'frozen_shared_preset_v1','use_motion_blur':False,
                'color_management':{'exposure':0.0}}
        self.assertEqual(frozen_lighting_report(config)['result_exposure_ev'],0.)
        config['use_motion_blur']=True
        with self.assertRaisesRegex(ValueError,'future motion'):frozen_lighting_report(config)


if __name__=='__main__':unittest.main()
