import unittest
from pathlib import Path

import numpy as np

from tests.three_object_fixtures import source
from tools.core.hashing import sha256_json
from tools.motion_rules.three_object.motion import validate_motion_contract
from tools.sampling.sample_three_object_base import build_three_object_scene
from tools.sampling.three_object_sampling_request import load_pilot_rules


MATRIX=Path('three_object_d6a_motion_matrix.json')


def primitive_source(index,shape):
    row=source(index);obj=row['metadata']['simulation']['objects'][0]
    size=[.18,.24,.30] if shape=='cuboid' else [.18,.18,.30]
    obj['geometry']={'type':shape,'size_m':size}
    obj['collision_profile']={'type':shape,'dimensions_m':size}
    obj['semantic_type']=shape
    return row


def candidate(template,variant,axis='world_x',objects=None):
    return build_three_object_scene(host_source=source(),object_sources=objects or [source(i) for i in range(3)],
        rules=load_pilot_rules(matrix_path=MATRIX),scene_id='motion_diversity_test',template_id=template,
        layout_variant=variant,motion_axis=axis,speed_m_s=1.15,spacing_ratio=1.75,offset_ratio=.52)


class ThreeObjectMotionDiversityTest(unittest.TestCase):
    def local_components(self,scene):
        contract=scene['three_object'];by_id={o['object_id']:o for o in scene['simulation']['objects']}
        direction=np.asarray(contract['approach_direction_world']);lateral=np.array([-direction[1],direction[0],0.])
        positions={r:np.asarray(by_id[oid]['initial_state']['position_m']) for r,oid in contract['roles'].items()}
        velocities={r:np.asarray(by_id[oid]['initial_state']['linear_velocity_m_s']) for r,oid in contract['roles'].items()}
        return direction,lateral,positions,velocities

    def test_legacy_default_stays_axis_aligned(self):
        legacy=build_three_object_scene(host_source=source(),object_sources=[source(i) for i in range(3)],
            rules=load_pilot_rules(),scene_id='legacy_motion_test')
        self.assertNotIn('layout_variant',legacy['three_object'])
        self.assertNotIn('layout_variant',legacy['three_object']['initial_parameters'])
        validate_motion_contract(legacy)

    def test_angled_chain_turns_both_ways_on_both_axes(self):
        for axis in ('world_x','world_y'):
            for side,sign in (('left',1),('right',-1)):
                scene=candidate('chain_transfer',f'angled_chain_{side}',axis)
                direction,lateral,positions,_=self.local_components(scene)
                self.assertGreater(sign*float((positions['Q']-positions['P'])@lateral),0)
                self.assertGreater(sign*float((positions['R']-positions['Q'])@lateral),0)
                self.assertGreater(float((positions['R']-positions['Q'])@direction),0)
                validate_motion_contract(scene)

    def test_successive_hit_offset_is_mirrored(self):
        left=candidate('successive_hits','offset_successive_left')
        right=candidate('successive_hits','offset_successive_right')
        for scene,sign in ((left,1),(right,-1)):
            _,lateral,positions,_=self.local_components(scene)
            self.assertGreater(sign*float((positions['Q']-positions['P'])@lateral),0)
            self.assertGreater(sign*float((positions['P']-positions['R'])@lateral),0)
            validate_motion_contract(scene)

    def test_crossing_control_moves_perpendicular_toward_pair_lane(self):
        for axis in ('world_x','world_y'):
            for side,sign in (('left',1),('right',-1)):
                scene=candidate('pair_control',f'crossing_control_{side}',axis)
                direction,lateral,positions,velocities=self.local_components(scene)
                self.assertLess(float((positions['R']-positions['P'])@direction),0.)
                self.assertGreater(sign*float((positions['R']-positions['P'])@lateral),0)
                self.assertAlmostEqual(float(velocities['R']@direction),0.)
                self.assertLess(sign*float(velocities['R']@lateral),0)
                validate_motion_contract(scene)

    def test_pair_control_explicitly_admits_a_moving_primitive_R(self):
        rules=load_pilot_rules(matrix_path=MATRIX)
        for shape in ('cuboid','cylinder'):
            objects=[source(0),source(1),primitive_source(2,shape)]
            scene=build_three_object_scene(host_source=source(),object_sources=objects,rules=rules,
                scene_id='moving_primitive_test',template_id='pair_control',layout_variant='crossing_control_left')
            by_id={o['object_id']:o for o in scene['simulation']['objects']}
            self.assertEqual(by_id[scene['three_object']['roles']['R']]['geometry']['type'],shape)
            self.assertAlmostEqual(np.linalg.norm(by_id[scene['three_object']['roles']['R']]['initial_state']['linear_velocity_m_s']),1.1)
            validate_motion_contract(scene)

    def test_crossing_sphere_uses_visible_transverse_speed(self):
        scene=candidate('pair_control','crossing_control_left')
        by_id={o['object_id']:o for o in scene['simulation']['objects']}
        role_r=scene['three_object']['roles']['R']
        self.assertAlmostEqual(
            np.linalg.norm(by_id[role_r]['initial_state']['linear_velocity_m_s']),
            .75*1.15,
        )
        validate_motion_contract(scene)

    def test_tampered_variant_geometry_is_rejected_after_rehash(self):
        scene=candidate('pair_control','crossing_control_left')
        role_r=scene['three_object']['roles']['R'];obj=next(o for o in scene['simulation']['objects'] if o['object_id']==role_r)
        obj['initial_state']['linear_velocity_m_s']=[1.15,0.,0.]
        scene['three_object']['contract_sha256']=sha256_json({k:v for k,v in scene['three_object'].items() if k!='contract_sha256'})
        with self.assertRaisesRegex(ValueError,'initial motion'):
            validate_motion_contract(scene)


if __name__=='__main__':unittest.main()
