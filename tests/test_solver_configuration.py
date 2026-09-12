"""Non-default solver values must reach PyBullet and agree with exported values."""
import copy
import unittest
from unittest.mock import patch

from tests.test_two_object_sweep_admission import fixture
from tools.physics.pybullet_backend_dispatcher import _common_audit
from tools.physics.solver_configuration import (
    apply_solver_configuration, normalize_solver, resolved_solver_configuration,
)
from tools.physics.two_object_specialized_simulation import _configure_world
from tools.release.base_release_schema import _solver_contract

try:
    import pybullet as pb
except ImportError:
    pb = None


class SolverConfigurationTests(unittest.TestCase):
    def test_residual_threshold_is_exported_and_audited(self):
        for mutation in ('missing', 'wrong'):
            scene, trajectory, raw = fixture()
            exported = _solver_contract('generic_rigid_v1', scene['source_metadata'], scene)
            self.assertEqual(exported['solver_residual_threshold'], 0.0)
            if mutation == 'missing':
                raw['solver_execution']['pybullet_arguments'].pop('solverResidualThreshold')
            else:
                raw['solver_execution']['pybullet_arguments']['solverResidualThreshold'] = 1e-7
            audit = _common_audit(scene, trajectory, raw)
            self.assertFalse(next(c for c in audit['checks'] if c['id'] == 'runtime_solver_configuration_exact')['passed'])

    def test_explicit_residual_threshold_is_preserved_and_validated(self):
        scene, _, _ = fixture()
        original = scene['source_metadata']['simulation']['solver']
        self.assertEqual(normalize_solver({**original, 'solver_residual_threshold': 1e-8}, generic=True)['solver_residual_threshold'], 1e-8)
        for value in (-1, True, '0', float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_solver({**original, 'solver_residual_threshold': value}, generic=True)

    def test_specialized_solver_does_not_gain_a_generic_default(self):
        for adapter in ('billiards_two_object_v1', 'passive_pinball_two_object_v1', 'marble_run_two_object_v1'):
            scene, _, _ = fixture(adapter)
            self.assertNotIn('solver_residual_threshold', resolved_solver_configuration(adapter, scene['source_metadata'], scene))

    def test_source_backend_conflict_fails_simulation_and_export(self):
        scene, _, _ = fixture('passive_pinball_two_object_v1')
        source = scene['source_metadata']
        source['physics']['engine']['solver_iterations'] = 7
        for operation in (
            lambda: resolved_solver_configuration('passive_pinball_two_object_v1', source, scene),
            lambda: _solver_contract('passive_pinball_two_object_v1', source, scene),
            lambda: _configure_world(None, {**scene, 'time': {'simulation_hz': 240}}),
        ):
            with self.assertRaisesRegex(ValueError, 'conflicting source/backend'):
                operation()

    def test_generic_defaults_remain_true_but_explicit_false_is_preserved(self):
        scene, _, _ = fixture()
        source = scene['source_metadata']
        solver = source['simulation']['solver']
        for key in ('enable_cone_friction', 'use_split_impulse'):
            solver.pop(key)
        self.assertTrue(_solver_contract('generic_rigid_v1', source, scene)['enable_cone_friction'])
        solver.update(enable_cone_friction=False, use_split_impulse=False)
        exported = _solver_contract('generic_rigid_v1', source, scene)
        self.assertIs(exported['enable_cone_friction'], False)
        self.assertIs(exported['use_split_impulse'], False)

    def test_runtime_solver_mismatch_or_missing_evidence_fails_even_for_sweeps(self):
        for adapter in ('generic_rigid_v1', 'billiards_two_object_v1',
                        'passive_pinball_two_object_v1', 'marble_run_two_object_v1'):
            for mutation in ('missing', 'iterations', 'flags', 'readback'):
                with self.subTest(adapter=adapter, mutation=mutation):
                    scene, trajectory, raw = fixture(adapter)
                    if mutation == 'missing':
                        raw.pop('solver_execution')
                    elif mutation == 'readback':
                        raw['solver_execution']['reported_parameters']['numSolverIterations'] = 7
                    else:
                        key = 'numSolverIterations' if mutation == 'iterations' else 'enableConeFriction'
                        raw['solver_execution']['pybullet_arguments'][key] = 0
                    audit = _common_audit(scene, trajectory, raw)
                    self.assertFalse(audit['passed'])
                    self.assertFalse(next(c for c in audit['checks'] if c['id'] == 'runtime_solver_configuration_exact')['passed'])

    def test_invalid_options_do_not_silently_coerce(self):
        scene, _, _ = fixture()
        original = scene['source_metadata']['simulation']['solver']
        for key, invalid in [('solver_iterations', True), ('solver_iterations', 1.5),
                             ('enable_cone_friction', 'false'),
                             ('restitution_velocity_threshold_m_s', float('nan'))]:
            solver = {**original, key: invalid}
            with self.subTest(key=key, invalid=invalid), self.assertRaises(ValueError):
                normalize_solver(solver, generic=True)
        self.assertEqual(original, scene['source_metadata']['simulation']['solver'])

    @unittest.skipIf(pb is None, 'PyBullet unavailable')
    def test_real_pybullet_receives_nondefault_solver_configuration(self):
        scene, _, _ = fixture('passive_pinball_two_object_v1')
        scene['world'] = {'gravity_m_s2': [0, 0, 0]}
        scene['time']['simulation_hz'] = 240
        engine = scene['source_metadata']['physics']['engine']
        engine.update(solver_iterations=7, deterministic_overlapping_pairs=False,
                      enable_cone_friction=False, use_split_impulse=False)
        scene['adapter_payload']['backend']['physics'] = copy.deepcopy(engine)
        setter = pb.setPhysicsEngineParameter
        client = pb.connect(pb.DIRECT)
        try:
            with patch.object(pb, 'setPhysicsEngineParameter', wraps=setter) as call:
                execution = _configure_world(pb, scene)
            self.assertEqual(call.call_args.kwargs['numSolverIterations'], 7)
            self.assertEqual(call.call_args.kwargs['enableConeFriction'], 0)
            self.assertEqual(call.call_args.kwargs['useSplitImpulse'], 0)
            self.assertEqual(call.call_args.kwargs['deterministicOverlappingPairs'], 0)
            self.assertEqual(execution['reported_parameters']['numSolverIterations'], 7)
            self.assertEqual(_solver_contract('passive_pinball_two_object_v1', scene['source_metadata'], scene), engine)
            generic = normalize_solver(engine, generic=True)
            with patch.object(pb, 'setPhysicsEngineParameter', wraps=setter) as call:
                apply_solver_configuration(pb, generic)
            self.assertEqual(call.call_args.kwargs["solverResidualThreshold"], 0.0)
            self.assertEqual(call.call_args.kwargs['enableConeFriction'], 0)
            self.assertEqual(call.call_args.kwargs['useSplitImpulse'], 0)
        finally:
            pb.disconnect(client)


if __name__ == '__main__':
    unittest.main()
