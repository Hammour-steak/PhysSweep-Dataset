"""Explicit marble execution has three bodies and source-bound solver settings."""
import copy
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from tools.physics.solver_configuration import resolved_solver_configuration
from tools.physics.specialized_sphere_simulation import simulate_specialized_spheres

class ThreeObjectMarbleSolverTests(unittest.TestCase):
    def test_source_engine_required_and_conflict_rejected(self):
        values={'solver_iterations':180,'deterministic_overlapping_pairs':True,
            'restitution_velocity_threshold_m_s':.02,'enable_cone_friction':True,'use_split_impulse':True}
        source={'physics':{'engine':values}};scene={'adapter_payload':{'backend':{}}}
        self.assertEqual(resolved_solver_configuration('marble_run_three_object_v1',source,scene),
                         resolved_solver_configuration('marble_run_two_object_v1',source,scene))
        changed=copy.deepcopy(scene);changed['adapter_payload']['backend']['physics']={'solver_iterations':40}
        with self.assertRaisesRegex(ValueError,'conflicting'):
            resolved_solver_configuration('marble_run_three_object_v1',source,changed)
        with self.assertRaises(KeyError):resolved_solver_configuration('marble_run_three_object_v1',{},scene)

    def test_no_single_or_two_body_fallback_and_no_missing_event_policy(self):
        factory=MagicMock()
        for count,tolerance in ((1,.0001),(2,.0001),(3,None),(4,.0001)):
            with self.subTest(count=count),self.assertRaises(ValueError):
                simulate_specialized_spheres({'objects':[{}]*count,'backend_binding':{'adapter_id':'marble_run_three_object_v1'}},
                    Path('.'),fixture_builder=MagicMock(),pybullet_factory=factory,event_distance_tolerance_m=tolerance)
        factory.assert_not_called()
