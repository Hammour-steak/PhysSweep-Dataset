import copy
import tempfile
import unittest
from pathlib import Path

from tests.test_two_object_sweep_admission import fixture
from tools.core.hashing import sha256_file
from tools.core.json_io import write_json_atomic
from tools.release.admission_policy import admission_decision_matches, POLICY


class ReleaseAdmissionPolicyTests(unittest.TestCase):
    def evidence(self, root):
        scene, _, adapter = fixture()
        audit = {'checks':[{'id':'finite_state','passed':True},{'id':'adapter_hard_invariants','passed':False}], 'adapter_audit':adapter}
        write_json_atomic(root/'resolved.json',scene);write_json_atomic(root/'audit.json',audit)
        raw={'scene_id':'pair','audit_passed':False,'failed_checks':['adapter_hard_invariants'],'adapter_audit_passed':False,'resolved_scene_path':str(root/'resolved.json'),'resolved_scene_sha256':sha256_file(root/'resolved.json'),'audit_path':str(root/'audit.json'),'audit_sha256':sha256_file(root/'audit.json')}
        rows=[{'scene_id':'pair','kind':'sweep','passed':True,'previously_passed':False,'source_audit_sha256':raw['audit_sha256'],'adapter_integrity_passed':True,'failed_integrity_checks':[]}]
        write_json_atomic(root/'rows.json',rows)
        proof={'policy':POLICY,'status':'passed','sample_count':1,'records_path':str(root/'rows.json'),'records_sha256':sha256_file(root/'rows.json')}
        write_json_atomic(root/'proof.json',proof)
        effective={**raw,'audit_passed':True,'failed_checks':[],'original_audit_passed':False,'original_failed_checks':raw['failed_checks'].copy(),'admission_policy':POLICY,'admission_reclassification':{'path':str(root/'proof.json'),'sha256':sha256_file(root/'proof.json')}}
        return raw,effective

    def test_verified_effective_decision_preserves_raw_failed_audit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw,effective=self.evidence(root);before=copy.deepcopy(raw)
            self.assertTrue(admission_decision_matches(root,raw,effective));self.assertEqual(raw,before)

    def test_changed_proof_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw,effective=self.evidence(root)
            (root/'rows.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                admission_decision_matches(root,raw,effective)

    def test_unproven_or_cross_scene_decision_is_rejected(self):
        for changed in ({'admission_policy':'unknown'},{'scene_id':'another'},{'original_failed_checks':[]},{'adapter_audit_passed':True}):
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as d:
                root=Path(d);raw,effective=self.evidence(root);effective.update(changed)
                self.assertFalse(admission_decision_matches(root,raw,effective))

    def test_exact_legacy_decision_needs_no_reclassification(self):
        raw={'audit_passed':True,'adapter_audit_passed':True,'failed_checks':[]}
        self.assertTrue(admission_decision_matches(Path('.'),raw,raw.copy()))


if __name__ == '__main__': unittest.main()
