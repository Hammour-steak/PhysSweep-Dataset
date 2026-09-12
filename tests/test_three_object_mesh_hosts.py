import copy
import unittest
from pathlib import Path

import numpy as np

from tests.three_object_fixtures import source
from tools.motion_rules.three_object.motion import center_layout_envelope,validate_motion_contract
from tools.sampling.sample_three_object_base import build_three_object_scene
from tools.sampling.three_object_coverage import _mesh_timing_parameters
from tools.sampling.three_object_fingerprints import physics_fingerprint
from tools.sampling.three_object_sampling_request import load_pilot_rules
from tools.scene_rules.three_object import validate_host


class ThreeObjectMeshHostFoundationTests(unittest.TestCase):
    def test_exact_mesh_matrix_is_explicit_and_bounded(self) -> None:
        rules=load_pilot_rules(matrix_path=Path('three_object_d5j_sampling_matrix.json'))
        matrix=rules['matrix']
        assignments=matrix['mesh_host_assignments']
        self.assertEqual(matrix['coverage_mode'],'exact_mesh_host_v1')
        self.assertEqual(matrix['candidate_budget'],19)
        self.assertEqual(matrix['maximum_source_attempts_per_candidate'],1)
        self.assertEqual(len(assignments),len(matrix['required_support_asset_ids']))
        self.assertEqual({row['asset_id'] for row in assignments},set(matrix['required_support_asset_ids']))
        self.assertEqual(sum(row['template']=='chain_transfer' for row in assignments),10)
        self.assertEqual(sum(row['template']=='pair_control' for row in assignments),9)
        self.assertEqual(sum(row['scale']=='small' for row in assignments),5)

    def test_d5k_matrix_explicitly_excludes_failed_rattan_host(self) -> None:
        rules=load_pilot_rules(matrix_path=Path('three_object_d5k_sampling_matrix.json'))
        matrix=rules['matrix'];assignments=matrix['mesh_host_assignments']
        self.assertEqual(matrix['candidate_budget'],18)
        self.assertEqual(matrix['excluded_support_asset_ids'],['support_rattan_coffee_c434f22e'])
        self.assertEqual(len(assignments),len(matrix['required_support_asset_ids']))
        self.assertEqual({row['asset_id'] for row in assignments},set(matrix['required_support_asset_ids']))
        self.assertEqual(sum(row['template']=='chain_transfer' for row in assignments),9)
        self.assertEqual(sum(row['template']=='pair_control' for row in assignments),9)
        self.assertEqual(sum(row['motion_axis']=='world_x' for row in assignments),9)
        self.assertEqual(sum(row['motion_axis']=='world_y' for row in assignments),9)
        self.assertEqual(sum(row['scale']=='small' for row in assignments),5)
        self.assertEqual(sum(row['scale']=='medium' for row in assignments),13)
        self.assertEqual(matrix['exact_mesh_timing_policy']['nominal_first_contact_time_s'],.21)
        self.assertEqual(matrix['exact_mesh_timing_policy']['chain_transfer_minimum_driver_to_receiver_mass_ratio'],.8)
        by_asset={row['asset_id']:row for row in assignments}
        self.assertEqual(by_asset['sketchfab_bg_a776babee9144e71ba90b473a4a5cfaa']['template'],'pair_control')
        self.assertEqual(by_asset['sketchfab_bg_a776babee9144e71ba90b473a4a5cfaa']['scale'],'small')
        self.assertEqual(by_asset['sketchfab_bg_bb73599044cf49e186b75842d63a280e']['template'],'chain_transfer')

    def test_d5k_timing_uses_horizontal_sphere_contact_and_capacity(self) -> None:
        rules=load_pilot_rules(matrix_path=Path('three_object_d5k_sampling_matrix.json'))
        trio=[source(index) for index in range(3)]
        for template,maximum_speed in (('chain_transfer',1.1),('pair_control',.4)):
            timing=_mesh_timing_parameters(trio,template,'world_x',(2.,2.),rules)
            self.assertIsNotNone(timing)
            radii=[row['metadata']['simulation']['objects'][0]['geometry']['size_m'][0]/2 for row in trio]
            p,q=radii[:2];pair_radius=p+q
            contact_x=np.sqrt(pair_radius**2-(p-q)**2)
            observed=(timing['spacing_ratio']*pair_radius-contact_x)/timing['initial_speed_m_s']
            self.assertAlmostEqual(observed,timing['nominal_first_contact_time_s'],places=12)
            self.assertGreaterEqual(observed,.21)
            self.assertLessEqual(timing['initial_speed_m_s'],maximum_speed)
            self.assertLessEqual(timing['spacing_ratio'],timing['usable_spacing_ratio'])

    def test_exact_mesh_host_identity_must_match_collision_and_visual(self) -> None:
        host=source()['metadata']
        support=host['simulation']['support']
        support['colliders'][0]['collision_enabled']=False
        support['asset_id']='table_asset'
        support['collision_authority']='exact_static_proxy'
        support['exact_static_binding']={'asset_id':'table_asset'}
        host['appearance']['support_visual']={'visual_type':'mesh_support','asset_id':'table_asset'}
        validate_host(host,load_pilot_rules()['scene'],allow_exact_mesh=True)
        host['appearance']['support_visual']['asset_id']='other_asset'
        with self.assertRaisesRegex(ValueError,'inconsistent'):
            validate_host(host,load_pilot_rules()['scene'],allow_exact_mesh=True)

    def test_exact_mesh_content_and_transform_belong_to_physics_fingerprint(self) -> None:
        base=scene_for_fingerprint=build_three_object_scene(
            host_source=source(),
            object_sources=[source(index) for index in range(3)],
            rules=load_pilot_rules(),
            scene_id='mesh_fingerprint',
        )
        support=scene_for_fingerprint['simulation']['support']
        support['colliders'][0]['collision_enabled']=False
        support['exact_static_binding']={
            'representation':'static_concave_mesh',
            'mesh':{
                'sha256':'a'*64,
                'scale':[1.0,2.0,3.0],
                'base_position_m':[0.0,0.0,0.0],
                'base_orientation_quaternion_xyzw':[0.0,0.0,0.0,1.0],
            },
        }
        before=physics_fingerprint(base)
        changed=copy.deepcopy(base)
        changed['simulation']['support']['exact_static_binding']['mesh']['base_position_m'][0]=0.1
        self.assertNotEqual(physics_fingerprint(changed),before)
        changed=copy.deepcopy(base)
        changed['simulation']['support']['exact_static_binding']['mesh']['sha256']='b'*64
        self.assertNotEqual(physics_fingerprint(changed),before)
        changed=copy.deepcopy(base)
        changed['simulation']['support']['colliders'][0]['size_m'][0]=999.0
        self.assertEqual(physics_fingerprint(changed),before)

    def test_asymmetric_pair_clearance_envelope_is_rigidly_centered(self) -> None:
        xy={
            'P':np.asarray([-0.3,0.0]),
            'Q':np.asarray([0.0,0.0]),
            'R':np.asarray([-0.3,0.4]),
        }
        extents={
            'P':np.asarray([0.08,0.08,0.08]),
            'Q':np.asarray([0.10,0.10,0.10]),
            'R':np.asarray([0.06,0.06,0.06]),
        }
        centered,offset=center_layout_envelope(xy,extents,1.5)
        before={role:xy[role]-xy['Q'] for role in xy}
        after={role:centered[role]-centered['Q'] for role in centered}
        for role in before:
            np.testing.assert_allclose(after[role],before[role],rtol=0,atol=1e-12)
        low=np.min(np.asarray([centered[r]-1.5*extents[r][:2] for r in centered]),axis=0)
        high=np.max(np.asarray([centered[r]+1.5*extents[r][:2] for r in centered]),axis=0)
        np.testing.assert_allclose(low,-high,rtol=0,atol=1e-12)
        self.assertFalse(np.allclose(offset,0.0,rtol=0,atol=1e-12))

    def test_world_y_chain_is_a_rigid_rotation_of_world_x(self) -> None:
        rules = load_pilot_rules()
        kwargs = dict(
            host_source=source(),
            object_sources=[source(index) for index in range(3)],
            rules=rules,
            scene_id='axis_test',
            spacing_ratio=1.5,
            template_id='chain_transfer',
        )
        world_x = build_three_object_scene(**kwargs)
        world_y = build_three_object_scene(**kwargs, motion_axis='world_y')
        validate_motion_contract(world_y)
        x_positions = np.asarray([
            row['initial_state']['position_m'] for row in world_x['simulation']['objects']
        ])
        y_positions = np.asarray([
            row['initial_state']['position_m'] for row in world_y['simulation']['objects']
        ])
        x_velocities = np.asarray([
            row['initial_state']['linear_velocity_m_s'] for row in world_x['simulation']['objects']
        ])
        y_velocities = np.asarray([
            row['initial_state']['linear_velocity_m_s'] for row in world_y['simulation']['objects']
        ])
        expected_positions = copy.deepcopy(x_positions)
        expected_positions[:, 0] = -x_positions[:, 1]
        expected_positions[:, 1] = x_positions[:, 0]
        expected_velocities = copy.deepcopy(x_velocities)
        expected_velocities[:, 0] = -x_velocities[:, 1]
        expected_velocities[:, 1] = x_velocities[:, 0]
        np.testing.assert_allclose(y_positions, expected_positions, rtol=0, atol=1e-12)
        np.testing.assert_allclose(y_velocities, expected_velocities, rtol=0, atol=1e-12)
        self.assertEqual(world_y['three_object']['approach_direction_world'], [0.0, 1.0, 0.0])
        self.assertEqual(world_y['three_object']['layout_axis'], 'world_y')
        self.assertNotEqual(physics_fingerprint(world_x), physics_fingerprint(world_y))

    def test_unknown_motion_axis_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'motion axis'):
            build_three_object_scene(
                host_source=source(),
                object_sources=[source(index) for index in range(3)],
                rules=load_pilot_rules(),
                scene_id='bad_axis',
                motion_axis='diagonal',
            )


if __name__ == '__main__':
    unittest.main()
