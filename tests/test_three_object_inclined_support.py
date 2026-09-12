import copy
import math
from pathlib import Path
import unittest
import numpy as np
from tests.three_object_fixtures import source
from tools.sampling.three_object_sampling_request import load_pilot_rules
from tools.sampling.sample_three_object_base import build_three_object_scene
from tools.scene_rules.three_object import validate_host,validate_initial_layout
from tools.motion_rules.three_object.motion import validate_motion_contract
from tools.rendering.three_object_camera import solve_three_object_camera


def ramp_scene(roles=None,template='pair_control',matrix='three_object_d5b_sampling_matrix.json'):
    rules=load_pilot_rules(matrix_path=Path(matrix))
    host=source();s=host['metadata']['simulation']['support'];theta=math.radians(10)
    top=.8;thickness=.1;rise=8*math.tan(theta)
    s.update(support_shape='inclined_ramp',scene_class='ground_feature',surface_center_z_m=top,
      visual_geometry={'primitive':'solid_wedge','size_xy_m':[8,8],'base_z_m':top-rise/2,'high_top_z_m':top+rise/2,'slope_axis':'y'})
    s['surface_frame']={'normal':[0,-math.sin(theta),math.cos(theta)],'tangent_uphill':[0,math.cos(theta),math.sin(theta)],'tangent_cross':[1,0,0],'slope_angle_degrees':10}
    s['colliders'][0].update(position_m=[0,0,top-thickness/(2*math.cos(theta))],rotation_euler_degrees=[10,0,0])
    objects=[source(i) for i in range(3)]
    for i,row in enumerate(objects):
        obj=row['metadata']['simulation']['objects'][0];diameter=.2+.03*i
        obj['geometry']['size_m']=[diameter]*3;obj['collision_profile']['dimensions_m']=[diameter]*3
    scene=build_three_object_scene(host_source=host,object_sources=objects,rules=rules,scene_id='inclined_test',role_order=roles,template_id=template)
    return scene,rules


class InclinedSupportTest(unittest.TestCase):
    def test_independent_lane_lead_is_frozen_and_validated(self):
        for roles in (['P','Q','R'],['Q','R','P'],['R','P','Q']):
            scene,_=ramp_scene(roles,matrix='three_object_d5br2_sampling_matrix.json')
            q=scene['simulation']['objects'][roles.index('Q')];r=scene['simulation']['objects'][roles.index('R')]
            self.assertGreater(r['initial_state']['position_m'][0]-q['initial_state']['position_m'][0],sum(o['geometry']['size_m'][0]/2 for o in (q,r)))
            self.assertEqual(r['initial_state']['linear_velocity_m_s'],scene['simulation']['objects'][roles.index('P')]['initial_state']['linear_velocity_m_s'])
            bad=copy.deepcopy(scene);bad['three_object']['initial_parameters'].pop('R_lead_in_Q_R_radii')
            from tools.core.hashing import sha256_json
            bad['three_object']['contract_sha256']=sha256_json({k:v for k,v in bad['three_object'].items() if k!='contract_sha256'})
            with self.assertRaisesRegex(ValueError,'lead policy'):validate_motion_contract(bad)
            r['initial_state']['position_m'][0]+=.01
            with self.assertRaisesRegex(ValueError,'lead contradicts'):validate_motion_contract(scene)

    def test_camera_bounded_material_policy_is_explicit_and_validated(self):
        scene,_=ramp_scene(matrix='three_object_d5br2_sampling_matrix.json')
        policy=scene['three_object']['initial_parameters']['inclined_material_policy']
        self.assertEqual(policy['contact_friction_minimum'],.32)
        self.assertEqual(policy['rolling_friction'],.018)
        for role,object_id in scene['three_object']['roles'].items():
            obj=next(row for row in scene['simulation']['objects'] if row['object_id']==object_id)
            source=policy['source_values_by_role'][role]
            self.assertEqual(obj['material']['contact_friction'],max(.32,source['contact_friction']))
            self.assertEqual(obj['material']['rolling_friction'],.018)
        bad=copy.deepcopy(scene)
        bad['simulation']['objects'][0]['material']['rolling_friction']=.002
        from tools.core.hashing import sha256_json
        bad['three_object']['contract_sha256']=sha256_json({k:v for k,v in bad['three_object'].items() if k!='contract_sha256'})
        with self.assertRaisesRegex(ValueError,'material values contradict'):
            validate_motion_contract(bad)

    def test_contact_friction_sweep_is_the_only_material_policy_exception(self):
        scene,_=ramp_scene(matrix='three_object_d5br2_sampling_matrix.json')
        target=scene['simulation']['objects'][0]
        swept=copy.deepcopy(scene)
        swept['sweep']={'kind':'sweep','axis':'contact_friction','target_object_id':target['object_id']}
        swept['simulation']['objects'][0]['material']['contact_friction']=.21
        validate_motion_contract(swept)
        swept['simulation']['objects'][1]['material']['contact_friction']=.22
        with self.assertRaisesRegex(ValueError,'material values contradict'):
            validate_motion_contract(swept)
        rolling=copy.deepcopy(scene)
        rolling['sweep']={'kind':'sweep','axis':'contact_friction','target_object_id':target['object_id']}
        rolling['simulation']['objects'][0]['material']['rolling_friction']=.002
        with self.assertRaisesRegex(ValueError,'material values contradict'):
            validate_motion_contract(rolling)
    def test_different_radii_rest_on_actual_plane_in_all_target_roles(self):
        for roles in (['P','Q','R'],['Q','R','P'],['R','P','Q']):
            scene,rules=ramp_scene(roles)
            for obj in scene['simulation']['objects']:
                x,y,z=obj['initial_state']['position_m'];radius=obj['geometry']['size_m'][0]/2
                self.assertAlmostEqual((z-.8)*math.cos(math.radians(10))-y*math.sin(math.radians(10)),radius,places=12)
                contact=obj['initial_state']['contact_point_m']
                self.assertAlmostEqual(contact[2],.8+contact[1]*math.tan(math.radians(10)),places=12)
                self.assertEqual(obj['material']['mass_kg'],.5)
            bad=copy.deepcopy(scene);bad['simulation']['objects'][0]['initial_state']['position_m'][2]=.8+bad['simulation']['objects'][0]['geometry']['size_m'][0]/2
            with self.assertRaisesRegex(ValueError,'rest on ramp'):validate_initial_layout(bad,rules['scene'],.001)

    def test_scope_rejects_unsupported_templates_and_inconsistent_host(self):
        with self.assertRaisesRegex(ValueError,'only implemented'):ramp_scene(template='chain_transfer')
        scene,rules=ramp_scene()
        for key in ('surface_frame','visual_geometry'):
            bad=copy.deepcopy(scene)
            if key=='surface_frame':bad['simulation']['support'][key]['normal']=[0,0,1]
            else:bad['simulation']['support'][key]['high_top_z_m']+=.01
            with self.assertRaises(ValueError):validate_host(bad,rules['scene'])
        bad=copy.deepcopy(scene);bad['simulation']['objects'][0]['initial_state']['linear_velocity_m_s'][1]=.1
        with self.assertRaisesRegex(ValueError,'initial motion'):validate_motion_contract(bad)

    def test_finite_primary_surface_checked_even_with_loose_declared_bounds(self):
        scene,rules=ramp_scene();s=scene['simulation']['support'];s['safe_surface_bounds']={'x':[-100,100],'y':[-100,100]}
        obj=scene['simulation']['objects'][0];obj['initial_state']['position_m'][0]=4.1
        with self.assertRaisesRegex(ValueError,'finite ramp'):validate_initial_layout(scene,rules['scene'],.001)

    def test_camera_angles_reference_surface_normal(self):
        scene,_=ramp_scene();trajectory={}
        for obj in scene['simulation']['objects']:
            oid=obj['object_id'];p=np.repeat([obj['initial_state']['position_m']],97,axis=0);r=obj['geometry']['size_m'][0]/2
            trajectory[oid+'__position_m']=p;trajectory[oid+'__aabb_min_m']=p-r;trajectory[oid+'__aabb_max_m']=p+r
        camera=solve_three_object_camera(scene,trajectory)
        direction=np.asarray(camera['position_m'])-camera['target_m'];direction/=np.linalg.norm(direction)
        normal=scene['simulation']['support']['surface_frame']['normal']
        self.assertAlmostEqual(float(direction@normal),math.sin(math.radians(camera['diagnostics']['selected_elevation_degrees'])),places=12)
        self.assertEqual(camera['structure_context'],'inclined_surface')


if __name__=='__main__':unittest.main()
