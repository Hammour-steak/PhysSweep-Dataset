import copy
import unittest
import numpy as np
from tests import test_three_object_pinball_metadata as fixtures
from tools.physics.resolved_simulation_scene import compile_resolved_scene
from tools.physics.three_object_specialized_simulation import audit_three_object_pinball
from tools.physics.contact_events import ContactEventCollector
from tools.motion_rules.three_object.interaction import audit_hard_results


class ThreeObjectPinballPhysicsTests(unittest.TestCase):
    def evidence(self):
        scene = compile_resolved_scene(fixtures.ThreeObjectPinballMetadataTests().candidate(), fixtures.ROOT)
        objects = scene['objects']; count = scene['time']['frame_count']; hz = scene['time']['simulation_hz']
        arrays = {'time_s': np.arange(count) / scene['time']['output_fps']}
        for key in ('position_m', 'linear_velocity_m_s', 'angular_velocity_rad_s'):
            arrays[key] = np.repeat(np.asarray([[o['initial_state'][key] for o in objects]]), count, axis=0)
        arrays['quaternion_wxyz'] = np.tile([1., 0., 0., 0.], (count, 3, 1))
        for key, fields in [('runtime_material', ('mass_kg', 'contact_friction', 'contact_restitution')),
                            ('runtime_material_extras', ('rolling_friction', 'spinning_friction', 'linear_damping', 'angular_damping'))]:
            arrays[key] = np.asarray([[o['material'][k] for k in fields] for o in objects])
        arrays['inertia_diagonal_kg_m2'] = np.asarray([[.4*o['material']['mass_kg']*o['collision_proxy']['radius_m']**2]*3 for o in objects])
        arrays['adapter__minimum_contact_distance_m'] = np.zeros((count, 3))
        ids = [o['object_id'] for o in objects]
        collector = ContactEventCollector(ids, hz, scene['source_metadata']['three_object']['thresholds']['contact_distance_tolerance_m'])
        for step in range((count-1)*(hz//scene['time']['output_fps'])+1):
            collector.observe(step, {}, {oid: [0., 0., 0.] for oid in ids})
        fixture = scene['adapter_payload']['fixture']; material = fixture['material']
        execution = {'interaction_events': collector.evidence(), 'path_lengths': np.zeros(3), 'solver_execution': {}, 'contact_processing_execution': [],
            'runtime_support': [{'fixture_id': c['id'], 'mass_kg': 0., 'lateral_friction': material['contact_friction'],
                                 'restitution': material['contact_restitution']} for c in fixture['colliders']],
            'runtime_proxies': [{'shape_type': 2, 'dimensions_m': [o['collision_proxy']['radius_m']]*3,
                                'local_position_m': [0.]*3, 'local_quaternion_xyzw': [0., 0., 0., 1.]} for o in objects]}
        return scene, arrays, execution

    def test_zero_contact_and_short_path_are_only_sweep_advisories(self):
        scene, arrays, execution = self.evidence()
        self.assertFalse(audit_three_object_pinball(scene, arrays, execution)['passed'])
        scene['variant']['kind'] = 'sweep'
        audit = audit_three_object_pinball(scene, arrays, execution)
        self.assertTrue(audit['passed'])
        response = next(c for c in audit['checks'] if c['id'] == 'template_motion_response')
        self.assertFalse(response['passed']); self.assertEqual(response['severity'], 'advisory')
        self.assertEqual(response['path_evidence']['minimum_object_path_m'], .8)
        audit['checks'].pop()
        with self.assertRaises(ValueError): audit_hard_results(scene['objects'], audit, True)

    def test_each_static_body_and_numeric_integrity_remain_hard(self):
        scene, arrays, execution = self.evidence(); scene['variant']['kind'] = 'sweep'
        for field, index, value in [('adapter__minimum_contact_distance_m', (2, 2), -.002),
                                     ('runtime_material_extras', (2, 0), .9),
                                     ('inertia_diagonal_kg_m2', (1, 0), .9),
                                     ('linear_velocity_m_s', (2, 0, 0), 100.)]:
            damaged = copy.deepcopy(arrays); damaged[field][index] = value
            with self.subTest(field=field): self.assertFalse(audit_three_object_pinball(scene, damaged, execution)['passed'])
        for change in ('missing_peg', 'material', 'proxy'):
            damaged = copy.deepcopy(execution)
            if change == 'missing_peg': damaged['runtime_support'].pop()
            elif change == 'material': damaged['runtime_support'][-1]['restitution'] = 0.
            else: damaged['runtime_proxies'][2]['dimensions_m'][0] *= 2
            with self.subTest(change=change): self.assertFalse(audit_three_object_pinball(scene, arrays, damaged)['passed'])
        execution['interaction_events']['observed_substeps'] -= 1
        with self.assertRaises(ValueError): audit_three_object_pinball(scene, arrays, execution)

    def test_missing_or_inconsistent_path_evidence_stays_hard_for_sweeps(self):
        scene, arrays, execution = self.evidence(); scene['variant']['kind'] = 'sweep'
        for value in ([0., 0.], [0., float('nan'), 0.], [0., -1., 0.]):
            execution['path_lengths'] = value
            with self.assertRaises(ValueError): audit_three_object_pinball(scene, arrays, execution)
        execution['path_lengths'] = np.zeros(3); arrays['position_m'][3, 2, 0] += .01
        with self.assertRaises(ValueError): audit_three_object_pinball(scene, arrays, execution)
