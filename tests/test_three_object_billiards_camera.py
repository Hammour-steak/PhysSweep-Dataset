import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from tests import test_three_object_billiards_metadata as fixtures
from tools.rendering.three_object_billiards_camera import audit_billiards_camera, trajectory_view, static_support_rays
from tools.rendering.three_object_camera import solve_camera_candidates


class BilliardsCameraTests(unittest.TestCase):
    def inputs(self):
        m=fixtures.ThreeObjectBilliardsMetadataTests().candidate();objects=m['simulation']['objects']
        arrays={'object_ids':np.asarray([o['object_id'] for o in objects]),
            'position_m':np.repeat([[o['initial_state']['position_m'] for o in objects]],97,axis=0),
            'quaternion_wxyz':np.tile([1.,0.,0.,0.],(97,3,1)),
            'linear_velocity_m_s':np.zeros((97,3,3)),'angular_velocity_rad_s':np.zeros((97,3,3))}
        return m,arrays

    def test_all_objects_and_static_ray_observation_are_required(self):
        m,arrays=self.inputs();clear=lambda camera,points:np.zeros(len(points),dtype=bool)
        camera=solve_camera_candidates(m,trajectory_view(m,arrays),contract=m['three_object'],resolution=m['render']['resolution'],inclined=False,
            audit_fn=lambda m,_,c:audit_billiards_camera(m,arrays,c,static_occlusion=clear))
        self.assertTrue(camera['diagnostics']['admission']['passed'])
        self.assertFalse(audit_billiards_camera(m,arrays,camera,static_occlusion=lambda c,p:np.ones(len(p),dtype=bool))['passed'])
        with self.assertRaises(ValueError):audit_billiards_camera(m,arrays,camera,static_occlusion=None)
        damaged=copy.deepcopy(arrays);damaged['position_m'][-1,2,0]+=20
        audit=audit_billiards_camera(m,damaged,camera,static_occlusion=clear)
        self.assertFalse(audit['objects']['object_c']['passed']);self.assertTrue(audit['objects']['object_a']['passed'])

    def test_sweeps_cannot_solve_a_new_camera(self):
        m,arrays=self.inputs();m['sweep']={'kind':'sweep'}
        with self.assertRaisesRegex(ValueError,'reuse'):
            solve_camera_candidates(m,trajectory_view(m,arrays),contract=m['three_object'],resolution=m['render']['resolution'],inclined=False,audit_fn=None)

    @unittest.skipUnless(importlib.util.find_spec('pybullet'),'requires project PyBullet runtime')
    def test_static_triangle_gap_is_not_treated_as_solid_aabb(self):
        def fixture(pb,root,binding):
            vertices=[[-2,-1,0],[-1,-1,0],[-1,1,0],[-2,1,0],[1,-1,0],[2,-1,0],[2,1,0],[1,1,0]]
            shape=pb.createCollisionShape(pb.GEOM_MESH,vertices=vertices,indices=[0,1,2,0,2,3,4,5,6,4,6,7],flags=pb.GEOM_FORCE_CONCAVE_TRIMESH)
            return pb.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape)
        with patch('tools.rendering.three_object_billiards_camera.create_pybullet_static_support',side_effect=fixture):
            with static_support_rays(Path('.'),{}) as blocked:
                # Both segments cross the broad AABB; only one crosses a triangle.
                self.assertFalse(blocked(np.array([0.,0.,1.]),np.array([[0.,0.,-1.]]))[0])
                self.assertTrue(blocked(np.array([1.5,0.,1.]),np.array([[1.5,0.,-1.]]))[0])
