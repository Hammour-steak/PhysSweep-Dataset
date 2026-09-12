"""Exercise material compilation, actual PyBullet calls and contact timing."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_resolved_simulation_scene as resolved_tests
from tests.test_two_object_sweep_admission import fixture
from tools.physics.resolved_simulation_scene import compile_resolved_scene, TWO_OBJECT_MATERIAL_EXTRAS
from tools.physics.pybullet_backend_dispatcher import _common_audit
from tools.physics import two_object_specialized_simulation as simulator

ROOT = resolved_tests.ROOT

try:
    import pybullet as pb
except ImportError:
    pb = None


def source_pair():
    source = resolved_tests.ResolvedSimulationSceneTests().marble_run_metadata()
    a = source['simulation']['objects'][0]
    a['object_id'] = 'object_a'
    b = copy.deepcopy(a); b['object_id'] = 'object_b'; b['initial_state']['position_m'][0] = .08
    source['simulation']['objects'] = [a, b]
    source['object_identity']['objects'] = [{'object_id': name, 'role': 'dynamic'} for name in ('object_a','object_b')]
    source['physics']['two_object_quality'] = {'maximum_first_pair_contact_time_s': 1., 'maximum_penetration_m': .01}
    return source


class TwoObjectDynamicsContractTests(unittest.TestCase):
    def test_compiler_preserves_extras_for_base_and_each_intervention_axis(self):
        source = source_pair()
        for axis in (None, 'mass_kg', 'contact_friction', 'contact_restitution'):
            with self.subTest(axis=axis):
                value = copy.deepcopy(source)
                if axis:
                    rows = []
                    for i, obj in enumerate(value['simulation']['objects']):
                        primary = {k: obj['material'][k] for k in ('mass_kg','contact_friction','contact_restitution')}
                        if i == 0: primary[axis] *= .5
                        rows.append({'object_id':obj['object_id'],'object_index':i,'material':primary})
                    value['sweep'] = {'kind':'sweep','target_object_id':'object_a','target_object_index':0,'parameter':axis,'value':rows[0]['material'][axis],'resolved_object_physics':rows}
                resolved = compile_resolved_scene(value, ROOT)
                for original, actual in zip(source['simulation']['objects'], resolved['objects']):
                    self.assertEqual({k:actual['material'][k] for k in TWO_OBJECT_MATERIAL_EXTRAS}, {k:original['material'][k] for k in TWO_OBJECT_MATERIAL_EXTRAS})

    def test_missing_or_invalid_extra_cannot_silently_become_zero(self):
        for bad in (None, -1., float('nan')):
            source = source_pair()
            if bad is None: del source['simulation']['objects'][0]['material']['linear_damping']
            else: source['simulation']['objects'][0]['material']['linear_damping'] = bad
            with self.assertRaises((KeyError, ValueError)):
                compile_resolved_scene(source, ROOT)

    def test_sweep_extra_mismatch_fails_integrity_without_outcome_constraints(self):
        for missing in (False, True):
            scene, trajectory, raw = fixture('marble_run_two_object_v1')
            if missing: trajectory.pop('runtime_material_extras')
            else: trajectory['runtime_material_extras'][0,2] = 0.
            audit = _common_audit(scene, trajectory, raw)
            self.assertFalse(audit['passed'])
            self.assertFalse(next(c for c in audit['checks'] if c['id']=='runtime_material_extras_exact')['passed'])

    @unittest.skipIf(pb is None, 'PyBullet unavailable')
    def test_actual_dynamics_call_and_readback_preserve_all_extras(self):
        scene = compile_resolved_scene(source_pair(), ROOT)
        calls = []
        class Recorder:
            def __getattr__(self, name): return getattr(pb, name)
            def changeDynamics(self, *args, **kwargs):
                calls.append(kwargs.copy()); return pb.changeDynamics(*args, **kwargs)
        client = pb.connect(pb.DIRECT)
        try:
            bodies, _, _, extras = simulator._create_spheres(Recorder(), scene)
            for i, body in enumerate(bodies):
                self.assertAlmostEqual(pb.getDynamicsInfo(body,-1)[6], .0025)
                self.assertAlmostEqual(pb.getDynamicsInfo(body,-1)[7], .0006)
                self.assertEqual([calls[i][k] for k in ('rollingFriction','spinningFriction','linearDamping','angularDamping')], [.0025,.0006,.025,.01])
            np.testing.assert_allclose(extras, np.tile([.0025,.0006,.025,.01],(2,1)))
        finally: pb.disconnect(client)

    @unittest.skipIf(pb is None, 'PyBullet unavailable')
    def test_contact_between_output_frames_is_retained_only_in_its_interval(self):
        objects = []
        for name, x, speed in [('object_a',-.03,1.),('object_b',.03,-1.)]:
            objects.append({'object_id':name,'collision_proxy':{'type':'sphere','radius_m':.01},'material':{'mass_kg':1.,'contact_friction':0.,'contact_restitution':1.,**{k:0. for k in TWO_OBJECT_MATERIAL_EXTRAS}},'initial_state':{'position_m':[x,0.,0.],'orientation_quaternion_xyzw':[0.,0.,0.,1.],'linear_velocity_m_s':[speed,0.,0.],'angular_velocity_rad_s':[0.,0.,0.]}})
        scene={'backend_binding':{'adapter_id':'passive_pinball_two_object_v1'},'objects':objects,'variant':{'kind':'base'},'world':{'gravity_m_s2':[0.,0.,0.]},'time':{'simulation_hz':2400,'output_fps':24,'frame_count':3},'adapter_payload':{'backend':{'physics':{'solver_iterations':100,'restitution_velocity_threshold_m_s':0.,'enable_cone_friction':True,'use_split_impulse':True}},'quality':{'maximum_first_pair_contact_time_s':1.,'maximum_penetration_m':.01,'minimum_path_length_per_object_m':0.,'minimum_distinct_fixture_contacts_per_object':0}}}
        scene['source_metadata'] = {'physics': {'engine': copy.deepcopy(scene['adapter_payload']['backend']['physics'])}}
        with patch.object(simulator,'_pinball_fixture',return_value=({},lambda *_:False)):
            arrays, audit = simulator.simulate_two_object_specialized(scene, ROOT)
        self.assertGreater(audit['metrics']['first_pair_contact_time_s'],0.)
        self.assertLess(audit['metrics']['first_pair_contact_time_s'],1/24)
        self.assertTrue(np.all(arrays['contact_count'][1] > 0))
        self.assertTrue(np.all(arrays['contact_count'][2] == 0))


if __name__ == '__main__': unittest.main()
