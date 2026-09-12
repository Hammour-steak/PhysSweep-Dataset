import copy,importlib.util,unittest
import numpy as np
from tests import test_three_object_pinball_metadata as fixtures
from tools.rendering.three_object_pinball_camera import audit_pinball_camera,camera_contract,fixture_framing_points
from tools.rendering.three_object_camera import sphere_trajectory_view,solve_camera_candidates
from tools.rendering.static_fixture_rays import static_fixture_rays


class PinballCameraTests(unittest.TestCase):
    def test_fixture_prism_contains_cylinder_and_box_is_not_world_aabb(self):
        m={'physics':{'fixture':{'colliders':[{'shape':'cylinder','radius_m':1.,'length_m':2.,'position_m':[0.,0.,0.],
           'orientation_quaternion_xyzw':[0.,0.,0.,1.]}]}}}
        points=fixture_framing_points(m);ring=points[:32,:2]
        dense=np.stack([np.cos(np.arange(1024)*2*np.pi/1024),np.sin(np.arange(1024)*2*np.pi/1024)],axis=1)
        for a,b in zip(ring,np.roll(ring,-1,axis=0)):
            edge=b-a;relative=dense-a
            self.assertTrue(np.all(edge[0]*relative[:,1]-edge[1]*relative[:,0]>=-1e-12))
        q=np.sin(np.pi/8);w=np.cos(np.pi/8)
        m['physics']['fixture']['colliders']=[{'shape':'box','half_extents_m':[1.,.1,.1],'position_m':[0.,0.,0.],
            'orientation_quaternion_xyzw':[0.,0.,q,w]}]
        points=fixture_framing_points(m)
        self.assertEqual(points.shape,(8,3))
        self.assertLess(max(abs(points[:,0]+points[:,1])),1.5)

    def inputs(self):
        m=fixtures.ThreeObjectPinballMetadataTests().candidate();objects=m['simulation']['objects']
        arrays={'object_ids':np.asarray([o['object_id'] for o in objects]),
            'position_m':np.repeat([[o['initial_state']['position_m'] for o in objects]],97,axis=0),
            'quaternion_wxyz':np.tile([1.,0.,0.,0.],(97,3,1)),
            'linear_velocity_m_s':np.zeros((97,3,3)),'angular_velocity_rad_s':np.zeros((97,3,3))}
        return m,arrays

    def test_whole_fixture_is_required_even_when_three_balls_fit(self):
        m,arrays=self.inputs();clear=lambda c,p:np.zeros(len(p),dtype=bool)
        points=np.array([[-.8,-.1,1.],[.8,.2,2.05]])
        contract,binding=camera_contract(m)
        self.assertEqual(contract['camera_rules']['reference'],'world_XYZ_front_positive_Y_up_positive_Z')
        self.assertEqual(m['three_object']['camera_rules']['reference'],'initial_P_direction_and_support_normal')
        self.assertEqual(contract['camera_rules']['focal_length_mm'],[120.,85.])
        self.assertEqual(m['three_object']['camera_rules']['focal_length_mm'],[35.,50.])
        for key in ('minimum_object_extent_px','maximum_occluded_fraction','frame_margin_fraction','maximum_candidate_count','whole_fixture_required'):
            self.assertEqual(contract['camera_rules'][key],m['three_object']['camera_rules'][key])
        camera=solve_camera_candidates(m,sphere_trajectory_view(m,arrays),contract=contract,resolution=m['render']['resolution'],inclined=False,
            framing_points=points,audit_fn=lambda m,_,c:audit_pinball_camera(m,arrays,c,static_occlusion=clear,fixture_points=points))
        audit=audit_pinball_camera(m,arrays,camera,static_occlusion=clear,fixture_points=np.concatenate([points,[[20.,0.,1.]]]))
        self.assertTrue(all(o['passed'] for o in audit['objects'].values()));self.assertFalse(audit['passed'])
        with self.assertRaises(ValueError):audit_pinball_camera(m,arrays,camera,static_occlusion=None,fixture_points=points)
        self.assertFalse(audit_pinball_camera(m,arrays,camera,static_occlusion=lambda c,p:np.ones(len(p),dtype=bool),fixture_points=points)['passed'])
        damaged=copy.deepcopy(arrays);damaged['position_m'][-1,2,0]+=20
        self.assertFalse(audit_pinball_camera(m,damaged,camera,static_occlusion=clear,fixture_points=points)['objects']['object_c']['passed'])
        m['sweep']={'kind':'sweep'}
        with self.assertRaisesRegex(ValueError,'reuse'):solve_camera_candidates(m,sphere_trajectory_view(m,arrays),contract=contract,
            resolution=m['render']['resolution'],inclined=False,audit_fn=None,framing_points=points)

    @unittest.skipUnless(importlib.util.find_spec('pybullet'),'requires project PyBullet')
    def test_cylinder_empty_aabb_corner_does_not_occlude(self):
        def build(pb):
            shape=pb.createCollisionShape(pb.GEOM_CYLINDER,radius=1.,height=2.)
            return [pb.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape)]
        with static_fixture_rays(build) as (blocked,corners):
            self.assertEqual(corners.shape,(8,3))
            self.assertFalse(blocked(np.array([.9,.9,3.]),np.array([[.9,.9,-3.]]))[0])
            self.assertTrue(blocked(np.array([0.,0.,3.]),np.array([[0.,0.,-3.]]))[0])
