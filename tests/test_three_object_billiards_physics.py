import copy
import unittest
import numpy as np
from tests import test_three_object_billiards_metadata as fixtures
from tools.physics.resolved_simulation_scene import compile_resolved_scene
from tools.physics.three_object_specialized_simulation import audit_three_object_billiards
from tools.motion_rules.three_object.interaction import audit_hard_results
from tools.physics.contact_events import ContactEventCollector


class ThreeObjectBilliardsPhysicsTests(unittest.TestCase):
    def evidence(self):
        metadata = fixtures.ThreeObjectBilliardsMetadataTests().candidate()
        scene = compile_resolved_scene(metadata, fixtures.ROOT)
        objects = scene['objects']; count = scene['time']['frame_count']; hz = scene['time']['simulation_hz']
        initial = [o['initial_state'] for o in objects]; materials = [o['material'] for o in objects]
        arrays = {'time_s': np.arange(count)/scene['time']['output_fps']}
        for key in ('position_m', 'linear_velocity_m_s', 'angular_velocity_rad_s'):
            arrays[key] = np.repeat(np.asarray([[o[key] for o in initial]]), count, axis=0)
        arrays['quaternion_wxyz'] = np.tile([1., 0., 0., 0.], (count, 3, 1))
        arrays['runtime_material'] = np.asarray([[m[k] for k in ('mass_kg', 'contact_friction', 'contact_restitution')] for m in materials])
        arrays['runtime_material_extras'] = np.asarray([[m[k] for k in ('rolling_friction', 'spinning_friction', 'linear_damping', 'angular_damping')] for m in materials])
        arrays['inertia_diagonal_kg_m2'] = np.asarray([[.4*o['material']['mass_kg']*o['collision_proxy']['radius_m']**2]*3 for o in objects])
        arrays['adapter__minimum_contact_distance_m'] = np.zeros((count, 3))
        ids = [o['object_id'] for o in objects]; roles = metadata['three_object']['roles']
        collector = ContactEventCollector(ids, hz, metadata['three_object']['thresholds']['contact_distance_tolerance_m'])
        pq = tuple(sorted([roles['P'], roles['Q']])); qr = tuple(sorted([roles['Q'], roles['R']]))
        for step in range((count-1)*(hz//scene['time']['output_fps'])+1):
            v = {oid: [0., 0., 0.] for oid in ids}
            if step >= 1518: v[roles['Q']] = [.2, 0., 0.]
            collector.observe(step, {pq: 0.} if step == 1518 else ({qr: 0.} if step == 3643 else {}), v)
        support = scene['adapter_payload']['backend']['billiards_rules']['support_dynamics']
        execution = {'interaction_events': collector.evidence(), 'solver_execution': {}, 'contact_processing_execution': [],
            'runtime_support': [{'fixture_id': 'pool_table', 'mass_kg': 0., **support}],
            'runtime_proxies': [{'shape_type': 2, 'dimensions_m': [o['collision_proxy']['radius_m']]*3,
                'local_position_m': [0.]*3, 'local_quaternion_xyzw': [0., 0., 0., 1.]} for o in objects]}
        return scene, arrays, execution

    def test_complete_zero_contact_is_advisory_only_for_sweep(self):
        scene, arrays, execution = self.evidence()
        self.assertTrue(audit_three_object_billiards(scene, arrays, execution)['passed'])
        execution['interaction_events']['events'] = []
        self.assertFalse(audit_three_object_billiards(scene, arrays, execution)['passed'])
        scene['variant']['kind'] = 'sweep'
        audit = audit_three_object_billiards(scene, arrays, execution)
        self.assertTrue(audit['passed']); self.assertTrue(all(audit_hard_results(scene['objects'], audit, True)))
        self.assertEqual(len(audit['advisories']), 5)

    def test_incomplete_observation_and_runtime_evidence_never_pass_sweep(self):
        scene, arrays, execution = self.evidence(); scene['variant']['kind'] = 'sweep'
        for key in ('runtime_support', 'runtime_proxies'):
            damaged = copy.deepcopy(execution); damaged[key] = []
            with self.subTest(key=key), self.assertRaises(ValueError): audit_three_object_billiards(scene, arrays, damaged)
        damaged = copy.deepcopy(execution); damaged['interaction_events']['observed_substeps'] -= 1
        with self.assertRaises(ValueError): audit_three_object_billiards(scene, arrays, damaged)
        damaged_arrays = copy.deepcopy(arrays); damaged_arrays['adapter__minimum_contact_distance_m'] = np.zeros((1, 3))
        with self.assertRaises(ValueError): audit_three_object_billiards(scene, damaged_arrays, execution)

    def test_corrupt_numeric_evidence_stays_hard_and_check_list_is_exact(self):
        scene, arrays, execution = self.evidence(); scene['variant']['kind'] = 'sweep'
        for key, index, value, expected in [
            ('adapter__minimum_contact_distance_m', (1, 2), -.01, 'object_c__bounded_penetration'),
            ('inertia_diagonal_kg_m2', (1, 0), .5, 'object_b__runtime_inertia'),
            ('linear_velocity_m_s', (2, 0, 0), 100., 'bounded_group_energy'),
            ('runtime_material_extras', (2, 2), .8, 'object_c__runtime_material')]:
            damaged = copy.deepcopy(arrays); damaged[key][index] = value
            audit = audit_three_object_billiards(scene, damaged, execution)
            self.assertFalse(audit['passed']); self.assertFalse(next(c['passed'] for c in audit['checks'] if c['id'] == expected))
        audit = audit_three_object_billiards(scene, arrays, execution); audit['checks'].pop()
        with self.assertRaises(ValueError): audit_hard_results(scene['objects'], audit, True)

    def test_runtime_proxy_and_support_are_measured(self):
        scene, arrays, execution = self.evidence()
        execution['runtime_proxies'][2]['shape_type'] = 3
        execution['runtime_support'][0]['restitution'] = 0.
        audit = audit_three_object_billiards(scene, arrays, execution)
        failed = [c['id'] for c in audit['checks'] if not c['passed']]
        self.assertIn('object_c__runtime_proxy', failed)
        self.assertTrue(all(oid+'__runtime_support' in failed for oid in ('object_a', 'object_b', 'object_c')))
