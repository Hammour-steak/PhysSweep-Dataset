import copy,unittest
import numpy as np
from tests import test_three_object_marble_metadata as fixtures
from tools.core.hashing import sha256_json
from tools.rendering.three_object_marble_camera import fixture_framing_points,audit_marble_camera
from tools.rendering.three_object_camera import solve_camera_candidates,sphere_trajectory_view

class MarbleCameraTests(unittest.TestCase):
    def inputs(self):
        instance=fixtures.ThreeObjectMarbleMetadataTests();self.addCleanup(instance.doCleanups);m,_=instance.candidate()
        # The unit source deliberately contains small test meshes, not the real asset.
        m['three_object']['camera_rules']['mandatory_fixture_components']=[c['id'] for c in m['physics']['fixture']['mesh_components']]+[c['id'] for c in m['physics']['fixture']['analytic_colliders'] if c['id']!='safety_floor']
        c=m['three_object'];c['contract_sha256']=sha256_json({k:v for k,v in c.items() if k!='contract_sha256'})
        objects=m['simulation']['objects']
        arrays={'object_ids':np.array([o['object_id'] for o in objects]),'position_m':np.repeat([[o['initial_state']['position_m'] for o in objects]],97,axis=0),
                'quaternion_wxyz':np.tile([1.,0.,0.,0.],(97,3,1)),'linear_velocity_m_s':np.zeros((97,3,3)),'angular_velocity_rad_s':np.zeros((97,3,3))}
        return m,arrays

    def test_mesh_binding_required_components_and_visited_floor(self):
        m,arrays=self.inputs();points=fixture_framing_points(fixtures.ROOT,m,arrays)
        self.assertEqual(points.shape,(9+4*8,3))
        floor=next(c for c in m['physics']['fixture']['analytic_colliders'] if c['id']=='safety_floor')
        changed=copy.deepcopy(arrays);changed['position_m'][-1,0]=[.2,.1,floor['position_m'][2]+floor['half_extents_m'][2]+.0225]
        visited=fixture_framing_points(fixtures.ROOT,m,changed)
        self.assertEqual(len(visited),len(points)+4)
        np.testing.assert_allclose(visited[-4:,2],.2)
        broken=copy.deepcopy(m);broken['three_object']['camera_rules']['mandatory_fixture_components'].append('missing')
        with self.assertRaisesRegex(ValueError,'missing mandatory'):fixture_framing_points(fixtures.ROOT,broken,arrays)
        binding=m['physics']['fixture']['mesh_components'][0]['collision'];(fixtures.ROOT/binding['path']).write_text('changed')
        with self.assertRaisesRegex(ValueError,'hash changed'):fixture_framing_points(fixtures.ROOT,m,arrays)

    def test_fixture_projection_occlusion_and_sweep_reuse_are_required(self):
        m,arrays=self.inputs();points=np.asarray([[-.4,0.,1.5],[.1,0.,1.7]])
        clear=lambda c,p:np.zeros(len(p),dtype=bool)
        camera=solve_camera_candidates(m,sphere_trajectory_view(m,arrays),contract=m['three_object'],resolution=m['render']['resolution'],inclined=False,framing_points=points,
            audit_fn=lambda m,_,c:audit_marble_camera(m,arrays,c,static_occlusion=clear,fixture_points=points))
        self.assertFalse(audit_marble_camera(m,arrays,camera,static_occlusion=clear,fixture_points=np.concatenate([points,[[20.,0.,1.]]]))['passed'])
        self.assertFalse(audit_marble_camera(m,arrays,camera,static_occlusion=lambda c,p:np.ones(len(p),dtype=bool),fixture_points=points)['passed'])
        with self.assertRaises(ValueError):audit_marble_camera(m,arrays,camera,static_occlusion=None,fixture_points=points)
        m['sweep']={'kind':'sweep'}
        with self.assertRaisesRegex(ValueError,'reuse'):solve_camera_candidates(m,sphere_trajectory_view(m,arrays),contract=m['three_object'],resolution=m['render']['resolution'],inclined=False,framing_points=points,audit_fn=None)
