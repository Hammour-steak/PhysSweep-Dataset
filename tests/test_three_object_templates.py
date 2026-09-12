import copy
import itertools
import importlib.util
import unittest
from tests.three_object_fixtures import source
from tools.sampling.sample_three_object_base import build_three_object_scene
from tools.sampling.three_object_sampling_request import load_pilot_rules
from tools.motion_rules.three_object.motion import validate_motion_contract
from tools.release.base_release_schema import _clean_text
from tools.dataset_contract.object_identity_contract import attach_object_identity


def candidate(template,roles=None):
    return build_three_object_scene(host_source=source(),object_sources=[source(i) for i in range(3)],
        rules=load_pilot_rules(),scene_id='three_object_template_test',template_id=template,role_order=roles)


class ThreeObjectTemplatesTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('pybullet'),'requires physics runtime')
    def test_four_templates_have_real_positive_and_missing_contact_examples(self):
        from tools.physics.simulate_pybullet_rigid import simulate
        for template in ('chain_transfer','successive_hits','converging_hits','pair_control'):
            for positive in (True,False):
                with self.subTest(template=template,positive=positive):
                    objects=[source(i) for i in range(3)]
                    for row in objects:
                        obj=row['metadata']['simulation']['objects'][0]
                        obj['geometry']['size_m']=[.2 if positive else .4]*3
                        obj['collision_profile']['dimensions_m']=obj['geometry']['size_m'][:]
                        obj['material'].update(contact_friction=.01,rolling_friction=0.,spinning_friction=0.)
                    scene=build_three_object_scene(host_source=source(),object_sources=objects,rules=load_pilot_rules(),
                        scene_id='template_physics_example',template_id=template,
                        speed_m_s=.9 if positive else .3,spacing_ratio=2. if positive else 4.)
                    trajectory,audit=simulate(scene)
                    integrity=[c for c in audit['checks'] if c['category']=='integrity']
                    self.assertTrue(all(c['passed'] for c in integrity),integrity)
                    self.assertEqual(audit['passed'],positive,[(c['id'],c['value']) for c in audit['checks'] if not c['passed']])
                    if not positive:
                        self.assertTrue(any(not c['passed'] for c in audit['checks'] if c['category']=='base_semantics'))

    def test_all_templates_and_role_permutations_bind_initial_text(self):
        for template in ('chain_transfer','successive_hits','converging_hits','pair_control'):
            for roles in itertools.permutations('PQR'):
                with self.subTest(template=template,roles=roles):
                    scene=candidate(template,list(roles));validate_motion_contract(scene)
                    text=_clean_text('generic',scene,{f'object_{i}':'ball' for i in 'abc'})
                    self.assertEqual(len(text['object_mentions']),3)
                    for mention in text['object_mentions']:
                        a,b=mention['char_span'];self.assertEqual(text['caption'][a:b],'the ball')
                    self.assertNotIn('will ',text['caption']);self.assertNotIn('collid',text['caption'])
                    before=copy.deepcopy(scene['object_identity']['text'])
                    scene['simulation']['objects'][0]['material']['mass_kg']*=4
                    attach_object_identity(scene)
                    self.assertEqual(scene['object_identity']['text'],before)

    def test_direction_rest_and_layout_contradictions_are_rejected(self):
        for template in ('converging_hits','pair_control'):
            scene=candidate(template)
            scene['simulation']['objects'][2]['initial_state']['linear_velocity_m_s']=[0,0,0]
            with self.assertRaises(ValueError):validate_motion_contract(scene)
            with self.assertRaises(ValueError):attach_object_identity(scene)
        scene=candidate('successive_hits')
        scene['simulation']['objects'][2]['initial_state']['position_m'][1]=.4
        with self.assertRaises(ValueError):validate_motion_contract(scene)

    def test_successive_constructor_has_offset_and_no_midrun_controls(self):
        scene=candidate('successive_hits');objects=scene['simulation']['objects']
        self.assertGreater(objects[1]['initial_state']['position_m'][1],0)
        self.assertLess(objects[2]['initial_state']['position_m'][1],0)
        self.assertFalse(any(k in scene['simulation'] for k in ('forces','impulses','controls','callbacks')))
        with self.assertRaises(ValueError):
            build_three_object_scene(host_source=source(),object_sources=[source(i) for i in range(3)],
                rules=load_pilot_rules(),scene_id='invalid',template_id='successive_hits',offset_ratio=0.)


if __name__=='__main__':unittest.main()
