from __future__ import annotations

import unittest
import tempfile
from collections import Counter
from pathlib import Path

from tools.release.publish_specialized_release_extension import specialized_renderer_binding
from tools.release.specialized_release_extension import (
    index_replacements,
    load_extension_spec,
    project_root_reference,
    select_replacement_slots,
    stable_seed,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs/marble_run_v5_release_extension.json"


class SpecializedReleaseExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_extension_spec(ROOT, SPEC_PATH)

    def source_records(self) -> list[dict[str, object]]:
        return [
            {
                "index": index,
                "scene_id": f"source_{index:03d}",
                "metadata_path": f"source/{index:03d}/metadata.json",
                "metadata_sha256": f"{index:064x}",
                "pipeline": "generic_pybullet",
                "motion_intent": "drop_fall_1obj",
            }
            for index in range(1, 65)
        ]

    def test_project_root_references_are_relative_only_inside_the_project(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            nested = root / "datasets/snapshot"
            external = root.parent / "external-snapshot"
            self.assertEqual(project_root_reference(root, root), ".")
            self.assertEqual(
                project_root_reference(root, nested), "datasets/snapshot"
            )
            self.assertEqual(
                project_root_reference(root, external), str(external.resolve())
            )

    def test_spec_preserves_exact_release_size_and_changes_one_category(self) -> None:
        source = Counter(self.spec["source_release"]["pipeline_group_counts"])
        target = Counter(self.spec["target_release"]["pipeline_group_counts"])
        self.assertEqual(sum(source.values()), 3200)
        self.assertEqual(sum(target.values()), 3200)
        self.assertEqual(source["generic_pybullet"] - target["generic_pybullet"], 32)
        self.assertEqual(target["marble_run"] - source["marble_run"], 32)
        self.assertEqual(
            self.spec["group_contract"]["sample_count"],
            3200 * 13,
        )

    def test_renderer_binding_comes_from_the_specialized_registry(self) -> None:
        replacement = self.spec["replacement"]
        binding = specialized_renderer_binding(ROOT, replacement)
        self.assertEqual(binding["path"], "tools/rendering/render_marble_run_scene.py")

        changed = {
            **replacement,
            "scene_schema_version": "unexpected_schema",
        }
        with self.assertRaisesRegex(ValueError, "renderer schema mismatch"):
            specialized_renderer_binding(ROOT, changed)

    def test_slot_selection_is_order_independent_and_eligible(self) -> None:
        records = self.source_records()
        replacement = self.spec["replacement"]
        arguments = {
            "namespace": replacement["selection_namespace"],
            "source_pipeline": replacement["source_pipeline"],
            "motion_intent": replacement["source_motion_intent"],
        }
        first = select_replacement_slots(records, 32, 20260826, **arguments)
        second = select_replacement_slots(
            list(reversed(records)), 32, 20260826, **arguments
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        self.assertTrue(
            all(record["pipeline"] == "generic_pybullet" for record in first)
        )

    def test_candidate_seed_is_stable_positive_and_namespaced(self) -> None:
        self.assertEqual(stable_seed("candidate"), stable_seed("candidate"))
        self.assertNotEqual(stable_seed("candidate"), stable_seed("candidate-2"))
        self.assertGreater(stable_seed("candidate"), 0)

    def test_replacement_index_rejects_changed_slot_provenance(self) -> None:
        source = self.source_records()[:32]
        records = [
            {
                "index": original["index"],
                "scene_id": f"marble_{original['index']}",
                "metadata_path": f"marble/{original['index']}/metadata.json",
                "metadata_sha256": "a" * 64,
                "pipeline": "marble_run",
                "motion_intent": "drop_fall_1obj",
                "replaces_scene_id": original["scene_id"],
                "replaces_metadata_path": original["metadata_path"],
                "replaces_metadata_sha256": original["metadata_sha256"],
                "replaces_pipeline": original["pipeline"],
            }
            for original in source
        ]
        indexed, old_paths, new_paths = index_replacements(
            {"records": source},
            {"sample_count": 32, "records": records},
            self.spec,
        )
        self.assertEqual(len(indexed), 32)
        self.assertEqual(len(old_paths), 32)
        self.assertEqual(len(new_paths), 32)
        records[0]["replaces_metadata_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "preserved slot"):
            index_replacements(
                {"records": source},
                {"sample_count": 32, "records": records},
                self.spec,
            )


if __name__ == "__main__":
    unittest.main()
