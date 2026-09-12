import copy
import importlib.util
import json
import unittest
import numpy as np

from tools.physics.contact_events import ContactEventCollector
from tools.motion_rules.three_object.interaction import audit_hard_results,expected_check_ids,validate_event_evidence,audit_three_object_motion
from tools.physics.simulate_pybullet_rigid import simulate,contact_interval_record
from tests.three_object_fixtures import scene


class ThreeObjectInteractionTests(unittest.TestCase):
    def test_contact_dedup_intervals_and_subframe_order(self):
        c=ContactEventCollector(['object_a','object_b','object_c'],480,0.0001)
        velocity={o:[0.,0.,0.] for o in c.ids}
        c.observe(0,{},velocity)
        c.observe(1,{('object_a','object_b'):-0.001,('object_b','object_a'):-0.001},velocity)
        c.observe(2,{('object_a','object_b'):-0.0005},velocity)
        c.observe(3,{('object_b','object_c'):-0.0002},velocity)
        events=c.evidence()['events']
        self.assertEqual(len(events),2)
        self.assertEqual((events[0]['start_substep'],events[0]['end_substep']),(1,2))
        self.assertEqual(events[1]['start_substep'],3)
        with self.assertRaises(ValueError):c.observe(5,{},velocity)

    def test_event_collection_does_not_remove_either_object_contact_points(self):
        class Physics:
            def getContactPoints(self,bodyA):
                bodyB=2 if bodyA==1 else 1
                return [(0,bodyA,bodyB,-1,-1,(0,0,0),(0,0,0),(1,0,0),-0.0001,0,0,(0,0,0),0,(0,0,0))]
        sink={}
        for body,other in [(1,2),(2,1)]:
            record=contact_interval_record(Physics(),body,{'support':0},0.2,{0:1.0},{str(other):other},{1:0.2,2:0.2},sink)
            self.assertEqual(record['all_contact_count'],1)
            self.assertEqual(record['object_contact_counts'][str(other)],1)
        self.assertEqual(len(sink),1)

    def test_empty_but_complete_events_differ_from_missing_observations(self):
        m=scene();c=ContactEventCollector(['object_a','object_b','object_c'],480,0.0001)
        velocity={o:[0.,0.,0.] for o in c.ids}
        for step in range(1921):c.observe(step,{},velocity)
        evidence=c.evidence();self.assertEqual(validate_event_evidence(evidence,m),[])
        evidence['observed_substeps']-=1
        with self.assertRaises(ValueError):validate_event_evidence(evidence,m)

    def test_sweep_semantics_are_diagnostic_but_unknown_missing_checks_fail(self):
        objects=scene()['simulation']['objects'];classes=expected_check_ids(objects)
        audit={'checks':[{'id':k,'category':v,'passed':v=='integrity'} for k,v in classes.items()]}
        self.assertTrue(all(audit_hard_results(objects,audit,True)))
        self.assertFalse(all(audit_hard_results(objects,audit,False)))
        for mutation in ('missing','unknown','category'):
            bad=copy.deepcopy(audit)
            if mutation=='missing':bad['checks'].pop()
            elif mutation=='unknown':bad['checks'].append({'id':'unknown','category':'base_semantics','passed':False})
            else:bad['checks'][0]['category']='base_semantics'
            with self.assertRaises(ValueError):audit_hard_results(objects,bad,True)

    @unittest.skipUnless(importlib.util.find_spec('pybullet'),'requires PyBullet')
    def test_real_three_body_transfer_and_missing_event_evidence(self):
        m=scene();trajectory,audit=simulate(m)
        self.assertTrue(audit['passed'],[(r['id'],r['value'],r['expected']) for r in audit['checks'] if not r['passed']])
        self.assertEqual(trajectory['object_c__position_m'].shape,(97,3))
        self.assertGreater(np.linalg.norm(trajectory['object_c__position_m'][-1]-trajectory['object_c__position_m'][0]),0.01)
        evidence=json.loads(trajectory['three_object_event_evidence_json'].item())
        self.assertEqual(len({tuple(e['object_ids']) for e in evidence['events']}),2)
        broken=copy.deepcopy(trajectory);del broken['three_object_event_evidence_json']
        with self.assertRaises(KeyError):audit_three_object_motion(m,broken)
