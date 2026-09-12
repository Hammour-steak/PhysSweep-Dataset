import copy
import itertools
import unittest
from tests.three_object_fixtures import scene,source
from tools.dataset_contract.object_identity_contract import attach_object_identity
from tools.release.base_release_schema import _clean_text
from tools.sampling.sample_three_object_base import build_three_object_scene
from tools.sampling.three_object_sampling_request import load_pilot_rules
from tools.motion_rules.three_object.motion import validate_motion_contract


class ThreeObjectCaptionTests(unittest.TestCase):
    def test_all_six_role_permutations_bind_repeated_labels_in_identity_order(self):
        for order in itertools.permutations('PQR'):
            with self.subTest(order=order):
                m=scene(list(order));validate_motion_contract(m)
                text=_clean_text('generic',m,{o:'ball' for o in ('object_a','object_b','object_c')})
                self.assertEqual([r['object_id'] for r in text['object_mentions']],['object_a','object_b','object_c'])
                for r in text['object_mentions']:
                    a,b=r['char_span'];self.assertEqual(text['caption'][a:b],'the ball')
                self.assertNotIn('collid',text['caption']);self.assertNotIn('will ',text['caption'])

    def test_unknown_template_or_false_rest_description_rejected(self):
        m=scene();m['three_object']['template']['id']='unknown'
        with self.assertRaises(ValueError):attach_object_identity(m)
        m=scene();m['simulation']['objects'][1]['initial_state']['angular_velocity_rad_s']=[1,0,0]
        with self.assertRaises(ValueError):attach_object_identity(m)

    def test_material_intervention_does_not_change_caption(self):
        m=scene(['R','P','Q']);before=copy.deepcopy(m['object_identity']['text'])
        m['simulation']['objects'][0]['material']['mass_kg']*=4
        attach_object_identity(m)
        self.assertEqual(before,m['object_identity']['text'])

    def test_compiler_preserves_appearance_and_drops_old_admission(self):
        host=source();objects=[source(i) for i in range(3)];before=copy.deepcopy([host,objects])
        m=build_three_object_scene(host_source=host,object_sources=objects,rules=load_pilot_rules(),scene_id='test')
        self.assertEqual(before,[host,objects])
        self.assertEqual(m['qa']['status'],'sampled_pending_simulation')
        self.assertNotIn('sweep',m)
        self.assertEqual(m['render_request']['samples'],load_pilot_rules()['media']['samples'])
        for i,obj in enumerate(m['simulation']['objects']):
            self.assertEqual(obj['visual_profile'],objects[i]['metadata']['simulation']['objects'][0]['visual_profile'])

    def test_too_small_or_sloped_host_rejected(self):
        for change in ('small','slope'):
            host=source()
            if change=='small':host['metadata']['simulation']['support']['safe_surface_bounds']['x']=[-0.1,0.1]
            else:host['metadata']['simulation']['support']['surface_frame']['normal']=[0,0.1,0.995]
            with self.assertRaises(ValueError):build_three_object_scene(host_source=host,object_sources=[source(i) for i in range(3)],rules=load_pilot_rules(),scene_id='test')
