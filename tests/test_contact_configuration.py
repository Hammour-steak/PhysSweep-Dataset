"""Keep configurable contact thresholds consistent with actual execution."""
import copy
import unittest

from tests.test_two_object_sweep_admission import fixture
from tools.physics.contact_configuration import contact_processing_threshold
from tools.physics.pybullet_backend_dispatcher import _common_audit


class ContactConfigurationTests(unittest.TestCase):
    def test_nonzero_threshold_requires_matching_execution_evidence(self):
        scene, trajectory, raw = fixture('billiards_two_object_v1')
        scene['adapter_payload']['backend']['billiards_rules']['ball_dynamics']['contact_processing_threshold_m'] = .02
        self.assertEqual(contact_processing_threshold(scene), .02)
        self.assertFalse(_common_audit(scene, trajectory, raw)['passed'])
        for row in raw['contact_processing_execution']:
            row['contactProcessingThreshold'] = .02
        self.assertTrue(_common_audit(scene, trajectory, raw)['passed'])
        for altered in (None, raw['contact_processing_execution'][:1],
                        list(reversed(raw['contact_processing_execution']))):
            bad = copy.deepcopy(raw); bad['contact_processing_execution'] = altered
            self.assertFalse(_common_audit(scene, trajectory, bad)['passed'])

    def test_invalid_threshold_never_reaches_simulation(self):
        for value in (None, True, -.01, float('nan'), float('inf'), '.02'):
            scene, _, _ = fixture('billiards_two_object_v1')
            scene['adapter_payload']['backend']['billiards_rules']['ball_dynamics']['contact_processing_threshold_m'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                contact_processing_threshold(scene)
