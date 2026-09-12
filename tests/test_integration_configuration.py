"""Mesh-impact substeps must be visible in resolution and runtime evidence."""
import copy
import unittest

from tests import test_resolved_simulation_scene as resolved_tests
from tests.test_two_object_sweep_admission import fixture
from tools.core.integration_configuration import generic_integration_configuration
from tools.physics.resolved_simulation_scene import compile_resolved_scene
from tools.physics.pybullet_backend_dispatcher import _common_audit


class IntegrationConfigurationTests(unittest.TestCase):
    def test_mesh_ballistic_trigger_changes_resolved_rate_once(self):
        metadata = resolved_tests.ResolvedSimulationSceneTests().generic_metadata()
        simulation = metadata['simulation']
        simulation['support'] = {'exact_static_binding': {'fixture': 'test'}}
        simulation['objects'][0]['expected_motion'] = {'contact_mode': 'ballistic_then_contact'}
        before = copy.deepcopy(metadata)
        from pathlib import Path
        resolved = compile_resolved_scene(metadata, Path(__file__).resolve().parents[1])
        nominal = simulation['time']['simulation_hz']
        self.assertEqual(resolved['time']['simulation_hz'], nominal * 2)
        self.assertEqual(generic_integration_configuration(resolved['source_metadata'])['simulation_hz'], nominal * 2)
        self.assertEqual(metadata, before)
        simulation['support'].pop('exact_static_binding')
        self.assertEqual(generic_integration_configuration(metadata)['simulation_hz'], nominal)

    def test_integration_mismatch_is_rejected_even_for_sweeps(self):
        for mutation in ('missing', 'readback', 'rate', 'internal_substeps'):
            scene, trajectory, audit = fixture()
            if mutation == 'missing':
                audit.pop('integration_execution')
            elif mutation == 'readback':
                audit['solver_execution']['reported_parameters']['fixedTimeStep'] = 1/480
            elif mutation == 'internal_substeps':
                audit['solver_execution']['reported_parameters']['numSubSteps'] = 2
            else:
                scene['time']['simulation_hz'] = 480
            result = _common_audit(scene, trajectory, audit)
            self.assertFalse(result['passed'], mutation)
            self.assertFalse(next(c for c in result['checks'] if c['id'] == 'runtime_integration_configuration_exact')['passed'])
