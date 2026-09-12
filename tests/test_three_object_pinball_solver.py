"""New execution primitive uses declared pinball engine; no single-ball fallback."""
import copy
import unittest
from tools.physics.solver_configuration import resolved_solver_configuration


class ThreeObjectPinballSolverTests(unittest.TestCase):
    def test_actual_declared_engine_required_and_conflicts_rejected(self):
        values={'solver_iterations':180,'deterministic_overlapping_pairs':True,
            'restitution_velocity_threshold_m_s':.02,'enable_cone_friction':True,'use_split_impulse':True}
        source={'physics':{'engine':values}};scene={'adapter_payload':{'backend':{'physics':values}}}
        old=resolved_solver_configuration('passive_pinball_two_object_v1',source,scene)
        self.assertEqual(resolved_solver_configuration('passive_pinball_three_object_v1',source,scene),old)
        changed=copy.deepcopy(scene);changed['adapter_payload']['backend']['physics']['solver_iterations']=40
        with self.assertRaisesRegex(ValueError,'conflicting'):
            resolved_solver_configuration('passive_pinball_three_object_v1',source,changed)
        with self.assertRaises(KeyError):
            resolved_solver_configuration('passive_pinball_three_object_v1',{},scene)
