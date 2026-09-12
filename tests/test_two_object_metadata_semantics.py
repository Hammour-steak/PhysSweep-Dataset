import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.dataset_contract.two_object_metadata import validate_two_object_candidate_metadata
from tools.release.base_release_schema import _compact_semantics
from tools.sampling import sample_two_object_specialized as specialized
from tools.assets import billiards_template


class TwoObjectMetadataSemanticsTests(unittest.TestCase):
    def test_camera_guard_rejects_inherited_one_object_intent(self):
        scene = {
            "camera_request": {"profile": "front_left_oblique", "observation": {
                "intent": "joint_full_motion_envelope", "structure_context": "horizontal"}},
            "semantic_sampling": {"five_dimensions": {"camera_observation": {
                "camera_profile": "front_left_oblique",
                "observation_intent": "joint_full_motion_envelope",
                "structure_context": "horizontal"}}},
        }
        validate_two_object_candidate_metadata(Path("."), scene)
        scene["semantic_sampling"]["five_dimensions"]["camera_observation"]["observation_intent"] = "single_object"
        with self.assertRaisesRegex(ValueError, "camera semantics"):
            validate_two_object_candidate_metadata(Path("."), scene)

    def test_two_object_export_uses_motion_but_keeps_fixture_source_profile(self):
        scene = {"semantics": {"profile": "early_release_chain",
                 "motion_profile": "two_ball_catch_up", "dynamic_object_count": 2}}
        self.assertEqual(_compact_semantics(scene), {"profile": "two_ball_catch_up"})
        self.assertEqual(scene["semantics"]["profile"], "early_release_chain")
        scene["semantics"]["dynamic_object_count"] = 1
        self.assertEqual(_compact_semantics(scene), {"profile": "early_release_chain"})

    def test_candidate_rejects_admission_and_conflicting_motion(self):
        scene = {"semantics": {"scene_family": "marble_run", "dynamic_object_count": 2,
                 "motion_profile": "two_ball_catch_up"},
                 "simulation": {"interaction": {"motion_pattern": "two_ball_catch_up"}}}
        validate_two_object_candidate_metadata(Path("."), scene)
        scene["admission"] = {"dynamic_object_count": 1}
        with self.assertRaisesRegex(ValueError, "admission"):
            validate_two_object_candidate_metadata(Path("."), scene)
        del scene["admission"]
        scene["simulation"]["interaction"]["motion_pattern"] = "single_ball"
        with self.assertRaisesRegex(ValueError, "motion semantics"):
            validate_two_object_candidate_metadata(Path("."), scene)

    def test_catalog_refresh_requires_unchanged_fixture_and_runtime_validation(self):
        old = {
            "asset_id": "table", "usage_id": "playing_bed",
            "target_support_frame": {"size_xy_m": [2.2, 1.14], "center_xy_m": [0, 0], "plane_z_m": .78},
            "usage_contract": {"maximum_axis_scale_ratio": 1.1},
            "mesh": {"path": "mesh.obj", "sha256": "mesh"},
            "binding_sha256": "old", "catalog_record_sha256": "old-record",
        }
        templates = {"billiards": {"physical_proxy_catalog": {"path": "catalog.json"},
                     "physics": {"static_support_binding": old}, "semantic_rules": {"path": "old.json"}}}
        frozen = copy.deepcopy(templates)
        current = dict(old, binding_sha256="new", catalog_record_sha256="new-record")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "catalog.json").write_text("{}")
            (root / "rules.json").write_text("{}")
            with patch.object(billiards_template, "load_catalog", return_value=(
                    {"records_sha256": "records"}, [{"asset_id": "table"}])) as catalog, \
                 patch.object(billiards_template, "compile_static_support_binding", return_value=current), \
                 patch.object(billiards_template, "validate_static_support_binding_files") as files:
                prepared = specialized.prepare_specialized_templates(root, templates, Path("rules.json"))
                catalog.assert_called_once_with(root, root / "catalog.json", require_runtime_validation=True)
                files.assert_called_once_with(root, current)
                self.assertEqual(prepared["billiards"]["semantic_rules"]["path"], "rules.json")
                self.assertEqual(templates, frozen)
                current["mesh"] = {"path": "changed.obj", "sha256": "changed"}
                with self.assertRaisesRegex(ValueError, "changes the frozen fixture"):
                    specialized.prepare_specialized_templates(root, templates, Path("rules.json"))
