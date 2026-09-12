"""Keep the simulated compound, resolved proxy and published proxy identical."""
import copy
from pathlib import Path
import unittest

import numpy as np

from tests import test_resolved_simulation_scene as resolved_tests
from tools.core.rigid_geometry import generic_collision_proxy, declared_collision_descriptors
from tools.physics.resolved_simulation_scene import compile_resolved_scene
from tools.physics.simulate_pybullet_rigid import create_dynamic_body
from tools.physics.physics_invariants import runtime_collision_descriptors
from tools.release.base_release_schema import _compact_objects

try:
    import pybullet as pb
except ImportError:
    pb = None


def fixture():
    source = resolved_tests.ResolvedSimulationSceneTests().generic_metadata()
    obj = source['simulation']['objects'][0]
    obj['geometry'] = {'type': 'cylinder', 'size_m': [.1, .1, .3]}
    obj['collision_profile'] = {'type': 'compound', 'colliders': [
        {'shape': 'cylinder', 'size_m': [.1, .1, .2], 'position_m': [0, 0, -.05], 'rotation_euler_degrees': [0, 0, 0]},
        {'shape': 'cylinder', 'size_m': [.04, .04, .1], 'position_m': [0, 0, .1], 'rotation_euler_degrees': [0, 0, 20]},
    ]}
    obj['material'].update(rolling_friction=0., spinning_friction=0., linear_damping=0., angular_damping=0.)
    other = copy.deepcopy(obj); other['object_id'] = 'object_b'; other['initial_state']['position_m'][0] = 1.
    source['simulation']['objects'].append(other)
    source['object_identity']['objects'] = [{'object_id': name, 'role': 'dynamic', 'asset_id': 'test_asset'} for name in ('object_a', 'object_b')]
    scene = compile_resolved_scene(source, Path('.'))
    trajectory = {'object_ids': ['object_a', 'object_b'], 'runtime_material': [[1., .4, .2]] * 2, 'inertia_diagonal_kg_m2': [[.001] * 3] * 2}
    return source, scene, trajectory


class CompoundCollisionContractTests(unittest.TestCase):
    def test_compiler_and_export_preserve_child_shapes_and_local_transforms(self):
        source, scene, trajectory = fixture()
        original = copy.deepcopy(source)
        objects, _ = _compact_objects('generic', source, scene, trajectory, '', None)
        for source_obj, resolved, exported in zip(source['simulation']['objects'], scene['objects'], objects):
            expected = source_obj['collision_profile']
            self.assertEqual(resolved['collision_proxy'], expected)
            self.assertEqual(exported['collision_proxy'], expected)
        self.assertEqual(source, original)

    def test_frozen_envelope_is_recovered_without_mutating_resolved_input(self):
        source, scene, trajectory = fixture()
        for obj in scene['objects']:
            obj['collision_proxy'] = copy.deepcopy(source['simulation']['objects'][0]['geometry'])
        original = copy.deepcopy(scene)
        objects, _ = _compact_objects('generic', source, scene, trajectory, '', None)
        self.assertEqual(objects[0]['collision_proxy'], source['simulation']['objects'][0]['collision_profile'])
        self.assertEqual(scene, original)
        scene['objects'][0]['collision_proxy']['size_m'][2] += .1
        with self.assertRaisesRegex(ValueError, 'collision proxy differs'):
            _compact_objects('generic', source, scene, trajectory, '', None)

    def test_invalid_or_missing_source_profile_is_rejected(self):
        source, _, _ = fixture()
        obj = source['simulation']['objects'][0]
        obj['collision_profile']['colliders'][0]['size_m'][0] = -1.
        with self.assertRaises(ValueError):
            generic_collision_proxy(obj)
        del obj['collision_profile']
        with self.assertRaises(ValueError):
            generic_collision_proxy(obj)

    @unittest.skipIf(pb is None, 'PyBullet unavailable')
    def test_export_recreates_actual_pybullet_compound_and_inertia(self):
        source, scene, trajectory = fixture()
        objects, _ = _compact_objects('generic', source, scene, trajectory, '', None)
        original = source['simulation']['objects'][0]
        rebuilt = copy.deepcopy(original); rebuilt['collision_profile'] = objects[0]['collision_proxy']
        client = pb.connect(pb.DIRECT)
        try:
            first = create_dynamic_body(pb, original); second = create_dynamic_body(pb, rebuilt)
            expected = declared_collision_descriptors(original)
            actual = runtime_collision_descriptors(pb, second)
            for key in expected:
                np.testing.assert_allclose(actual[key], expected[key], rtol=0., atol=1.e-12)
            self.assertEqual(pb.getDynamicsInfo(first, -1)[2], pb.getDynamicsInfo(second, -1)[2])
        finally:
            pb.disconnect(client)

    @unittest.skipIf(pb is None, 'PyBullet unavailable')
    def test_dispatch_refuses_a_resolved_envelope_instead_of_executing_other_geometry(self):
        from tools.physics.pybullet_backend_dispatcher import _generic
        source, scene, _ = fixture()
        scene['objects'][0]['collision_proxy'] = copy.deepcopy(source['simulation']['objects'][0]['geometry'])
        with self.assertRaisesRegex(ValueError, 'collision proxy differs'):
            _generic(scene)
