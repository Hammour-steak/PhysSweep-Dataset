import copy
import json
import unittest
import numpy as np
from tests.three_object_fixtures import source
from tools.sampling.sample_three_object_base import build_three_object_scene
from tools.sampling.three_object_sampling_request import load_pilot_rules
from tools.core.primitive_support import initial_extents
from tools.rendering.three_object_primitives import blocked_by_primitive,surface_samples
from tools.rendering.three_object_camera import solve_three_object_camera,audit_camera,required_event_frame_indices
from tools.motion_rules.three_object.motion import validate_motion_contract
from tools.scene_rules.three_object import validate_initial_layout
from pathlib import Path


def mixed_scene(shape='cuboid',template='chain_transfer',role_order=None):
    rules=load_pilot_rules(matrix_path=Path('three_object_d5a_sampling_matrix.json'))
    roles=role_order or ['P','Q','R'];objects=[source(i) for i in range(3)]
    obj=objects[roles.index('R')]['metadata']['simulation']['objects'][0]
    size=[.16,.16,.30] if shape=='cylinder' else [.18,.24,.30]
    obj['geometry']={'type':shape,'size_m':size};obj['collision_profile']={'type':shape,'dimensions_m':size}
    obj['semantic_type']=shape
    return build_three_object_scene(host_source=source(),object_sources=objects,rules=rules,scene_id='mixed_primitive',template_id=template,role_order=roles),rules


class MixedPrimitivesTest(unittest.TestCase):
    def test_true_half_height_and_role_compatibility(self):
        for shape in ('cuboid','cylinder'):
            for template in ('chain_transfer','pair_control'):
                for roles in (['P','Q','R'],['Q','R','P'],['R','P','Q']):
                    scene,rules=mixed_scene(shape,template,roles)
                    obj=scene['simulation']['objects'][roles.index('R')]
                    self.assertAlmostEqual(obj['initial_state']['position_m'][2],.15)
                    self.assertEqual(obj['material']['mass_kg'],.5)
                    bad=copy.deepcopy(scene);bad['simulation']['objects'][roles.index('R')]['initial_state']['position_m'][2]=.08
                    with self.assertRaisesRegex(ValueError,'rest on support'):validate_initial_layout(bad,rules['scene'],.001)
        with self.assertRaises(ValueError):mixed_scene(template='successive_hits')

    def test_upright_pose_and_primitive_profile_required(self):
        scene,_=mixed_scene();obj=scene['simulation']['objects'][2]
        obj['initial_state']['orientation_quaternion_wxyz']=[2**-.5,2**-.5,0,0]
        with self.assertRaisesRegex(ValueError,'upright'):initial_extents(obj)
        obj['collision_profile']['type']='sphere'
        with self.assertRaises(ValueError):initial_extents(obj)

    def test_ray_cylinder_does_not_use_bounding_box_empty_corners(self):
        camera=np.array([-3.,.9,.9]);ends=np.array([[3.,.9,.9],[3.,.9,2.]])
        args=(camera,ends,np.zeros(3),[1,0,0,0])
        np.testing.assert_array_equal(blocked_by_primitive(*args,'cylinder',[2,2,2]),[True,False])
        camera=np.array([.9,.9,3.]);ends=np.array([[.9,.9,-3.],[.9,.9,2.]])
        self.assertFalse(blocked_by_primitive(camera,ends,np.zeros(3),[1,0,0,0],'cylinder',[2,2,2]).any())
        self.assertTrue(blocked_by_primitive(camera,ends,np.zeros(3),[1,0,0,0],'cuboid',[2,2,2])[0])

    def test_surface_points_and_primitive_camera(self):
        for shape in ('cuboid','cylinder'):
            scene,_=mixed_scene(shape);trajectory={}
            for obj in scene['simulation']['objects']:
                oid=obj['object_id'];positions=np.repeat([obj['initial_state']['position_m']],97,axis=0);half=initial_extents(obj)
                trajectory[oid+'__position_m']=positions
                trajectory[oid+'__aabb_min_m']=positions-half;trajectory[oid+'__aabb_max_m']=positions+half
                trajectory[oid+'__quaternion_wxyz']=np.repeat([[1,0,0,0]],97,axis=0)
            camera=solve_three_object_camera(scene,trajectory)
            self.assertTrue(audit_camera(scene,trajectory,camera)['passed'])
            trajectory['object_c__position_m'][-1,0]+=100;trajectory['object_c__aabb_min_m'][-1,0]+=100;trajectory['object_c__aabb_max_m'][-1,0]+=100
            self.assertFalse(audit_camera(scene,trajectory,camera)['passed'])
            points,weights=surface_samples(shape,[2,2,2],np.zeros(3),[1,0,0,0],np.array([4.,4.,4.]))
            self.assertTrue(np.all(weights>0))
            if shape=='cylinder':self.assertTrue(np.all(np.isclose(np.linalg.norm(points[:,:2],axis=1),1)|np.isclose(abs(points[:,2]),1)))

    def test_mixed_shape_keyframes_prefer_substep_contact_evidence(self):
        scene,_=mixed_scene('cuboid')
        contract=validate_motion_contract(scene)
        roles=contract['roles']
        pairs=[tuple(sorted((roles[a],roles[b]))) for a,b in contract['template']['required_pairs']]
        trajectory={}
        for obj in scene['simulation']['objects']:
            oid=obj['object_id'];positions=np.repeat([obj['initial_state']['position_m']],97,axis=0);half=initial_extents(obj)
            trajectory[oid+'__position_m']=positions
            trajectory[oid+'__aabb_min_m']=positions-half;trajectory[oid+'__aabb_max_m']=positions+half
        events=[{'object_ids':list(pair),'start_substep':step} for pair,step in zip(pairs,(490,970))]
        trajectory['three_object_event_evidence_json']=np.asarray(json.dumps({
            'schema_version':'physweep_three_object_contact_events_v1',
            'simulation_hz':480,'events':events,
        }))
        # 490/480*24 = 24.5 and 970/480*24 = 48.5. Padding one
        # therefore retains both adjacent video frames and their neighbours.
        frames=required_event_frame_indices(scene,trajectory,contract)
        self.assertTrue({0,23,24,25,26,47,48,49,50}<=set(frames))


if __name__=='__main__':unittest.main()
