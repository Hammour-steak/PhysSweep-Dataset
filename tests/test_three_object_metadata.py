from __future__ import annotations
import copy
import unittest

from tools.sampling.three_object_sampling_request import load_pilot_rules, validate_rules, candidate_seed


class ThreeObjectRuleTests(unittest.TestCase):
    def setUp(self):
        self.rules=load_pilot_rules()

    def check(self, rules):
        validate_rules(**{k:rules[k] for k in ('matrix','motion','scene','sweep','dataset')})

    def test_unknown_target_and_output_rejected(self):
        for key,field,value in [('matrix','sweep_target_indices',[0,1,2]),('dataset','release_root','outputs/two_object'),('sweep','required_dynamic_objects',2)]:
            r=copy.deepcopy(self.rules); r[key][field]=value
            with self.assertRaises(ValueError): self.check(r)

    def test_conflicting_or_unclassified_contacts_rejected(self):
        for field,value in [('forbidden_pairs',[]),('required_pairs',[['P','Q'],['P','R']]),('first_contact_order',[['P','R']])]:
            r=copy.deepcopy(self.rules); r['motion']['templates'][0][field]=value
            with self.assertRaises(ValueError): self.check(r)

    def test_per_sweep_camera_and_invalid_threshold_rejected(self):
        r=copy.deepcopy(self.rules);r['scene']['camera']['policy']='per_sweep'
        with self.assertRaises(ValueError):self.check(r)
        r=copy.deepcopy(self.rules);r['motion']['thresholds']['minimum_event_gap_s']=float('nan')
        with self.assertRaises(ValueError):self.check(r)

    def test_seed_independent_of_evaluation_order(self):
        forward={i:candidate_seed(5,'cell',i,'physics') for i in range(10)}
        reverse={i:candidate_seed(5,'cell',i,'physics') for i in reversed(range(10))}
        self.assertEqual(forward,reverse)
        self.assertNotEqual(forward[0],candidate_seed(5,'cell',0,'camera'))
        self.assertEqual(len(set(forward.values())),10)

    def test_plan_resource_hashes_and_template_roles_are_explicit(self):
        self.assertEqual(self.rules['rules_sha256'],load_pilot_rules()['rules_sha256'])
        self.assertEqual(self.rules['matrix']['object_ids'],['object_a','object_b','object_c'])
        self.assertEqual(len(self.rules['motion']['templates']),4)
        self.assertFalse(self.rules['scene']['masks'])
