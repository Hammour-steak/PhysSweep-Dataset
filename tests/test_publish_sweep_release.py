from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from publish_sweep_release import (  # noqa: E402
    merge_base_records,
    merge_generic_samples,
    select_source_records,
    validate_groups,
)


class SweepReleaseTests(unittest.TestCase):
    def test_validate_complete_group(self) -> None:
        records = [{"parent": "base.json", "kind": "base", "axis": None}]
        for axis in ("mass_kg", "contact_friction", "contact_restitution"):
            records.extend(
                {"parent": "base.json", "kind": "sweep", "axis": axis}
                for _ in range(4)
            )
        self.assertEqual(validate_groups(records), 1)

    def test_reject_incomplete_group(self) -> None:
        with self.assertRaisesRegex(ValueError, "13 records"):
            validate_groups([{"parent": "base.json", "kind": "base"}])

    def test_merge_base_records_replaces_exact_index(self) -> None:
        base = [
            {"index": 1, "metadata_path": "old-1.json"},
            {"index": 2, "metadata_path": "old-2.json"},
        ]
        replacement = {
            "index": 2,
            "metadata_path": "new-2.json",
            "replaces_metadata_path": "old-2.json",
        }
        self.assertEqual(
            merge_base_records(base, [replacement]),
            [base[0], replacement],
        )

    def test_select_source_records_excludes_replaced_parent(self) -> None:
        metadata = {
            "records": [
                {"scene_id": "old", "parent": "old-base.json"},
                {"scene_id": "keep", "parent": "keep-base.json"},
            ]
        }
        physics = {
            "records": [
                {"scene_id": "old", "ok": True, "audit_passed": True},
                {"scene_id": "keep", "ok": True, "audit_passed": True},
            ]
        }
        selected_metadata, selected_physics = select_source_records(
            metadata, physics, {"old-base.json"}
        )
        self.assertEqual([record["scene_id"] for record in selected_metadata], ["keep"])
        self.assertEqual([record["scene_id"] for record in selected_physics], ["keep"])

    def test_merge_generic_samples_preserves_slot_count(self) -> None:
        source = [
            {"scene_id": "old", "metadata_path": "old.json"},
            {"scene_id": "keep", "metadata_path": "keep.json"},
        ]
        replacement = {
            "replaces_metadata_path": "old.json",
            "candidate_scene_id": "new",
            "metadata_path": "new.json",
            "metadata_sha256": "metadata-hash",
            "simulation_record_path": "simulation.json",
            "trajectory_path": "trajectory.npz",
        }
        merged = merge_generic_samples(source, [replacement])
        self.assertEqual(len(merged), 2)
        self.assertEqual({sample["scene_id"] for sample in merged}, {"keep", "new"})


if __name__ == "__main__":
    unittest.main()
