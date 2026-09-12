"""Behavior at the shared execution boundary, including subframe third-body events."""
import copy
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
from tools.physics.specialized_sphere_simulation import simulate_specialized_spheres


class SpecializedSphereExecutionTests(unittest.TestCase):
    def scene(self):
        return {
            'backend_binding': {'adapter_id': 'billiards_three_object_v1'},
            'time': {'frame_count': 3, 'output_fps': 24, 'simulation_hz': 240},
            'objects': [{'object_id': key, 'initial_state': {'position_m': [i, 0., 1.]}}
                        for i, key in enumerate(('P', 'Q', 'R'))],
        }

    def test_all_three_pairs_subframe_order_recontacts_and_endpoint_counts(self):
        scene = self.scene(); pb = MagicMock(); pb.connect.return_value = 0
        pb.getCollisionShapeData.return_value = [(0, -1, 2, (.1,.1,.1), b'', (0.,0.,0.), (0.,0.,0.,1.))]
        tick = [0]
        pb.stepSimulation.side_effect = lambda: tick.__setitem__(0, tick[0] + 1)
        pb.getBasePositionAndOrientation.side_effect = lambda body: ([body, tick[0] / 100., 1.], [0., 0., 0., 1.])
        pb.getBaseVelocity.return_value = ([0., .1, 0.], [0., 0., 0.])
        schedule = {2: [(0, 1), (0, 1)], 3: [(1, 2)], 7: [(0, 2)], 8: [(0, 1)], 9: [(0, 1)]}

        def contacts(bodyA, bodyB=None):
            result = []
            for a, b in schedule.get(tick[0], []):
                if bodyA == b: a, b = b, a
                if a == bodyA and (bodyB is None or b == bodyB):
                    result.append((0, a, b, 0, 0, [0., 0., 0.], [0., 0., 0.], 0, -1.e-5))
            return result

        pb.getContactPoints.side_effect = contacts
        arrays, evidence = simulate_specialized_spheres(
            scene, Path('.'), fixture_builder=lambda *_: ({}, lambda *_: False),
            pybullet_factory=lambda: pb, configure_world=lambda *_: {},
            create_spheres=lambda *_, **__: (list(range(3)), np.ones((3, 3)), np.ones((3, 3)), np.zeros((3, 4))),
            event_distance_tolerance_m=1.e-4,
        )
        events = evidence['interaction_events']; self.assertEqual(events['observed_substeps'], 21)
        self.assertEqual([(e['object_ids'], e['start_substep'], e['end_substep']) for e in events['events']],
                         [(['P', 'Q'], 2, 2), (['Q', 'R'], 3, 3), (['P', 'R'], 7, 7), (['P', 'Q'], 8, 9)])
        np.testing.assert_array_equal(arrays['position_m'][0], [[0., 0., 1.], [1., 0., 1.], [2., 0., 1.]])
        np.testing.assert_array_equal(arrays['contact_count'], [[0, 0, 0], [2, 2, 1], [0, 0, 0]])
        self.assertEqual(arrays['quaternion_wxyz'].shape, (3, 3, 4))
        pb.disconnect.assert_called_once_with(0)

    def test_unbound_adapter_count_and_missing_event_policy_rejected_before_connect(self):
        factory = MagicMock(); original = self.scene()
        for case in ('old_adapter', 'four_objects', 'missing_events'):
            scene = copy.deepcopy(original)
            if case == 'old_adapter': scene['backend_binding']['adapter_id'] = 'billiards_v4'
            if case == 'four_objects': scene['objects'].append(copy.deepcopy(scene['objects'][0]))
            with self.subTest(case=case), self.assertRaises(ValueError):
                simulate_specialized_spheres(scene, Path('.'), fixture_builder=MagicMock(), pybullet_factory=factory)
        factory.assert_not_called()
