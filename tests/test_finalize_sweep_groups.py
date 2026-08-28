from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_sweep_groups", ROOT / "tools/release/finalize_sweep_groups.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FinalizeSweepGroupsTest(unittest.TestCase):
    def source_records(self) -> list[dict]:
        records = [
            {
                "scene_id": "scene__base",
                "parent": "base/scene.json",
                "kind": "base",
                "axis": None,
                "level_index": None,
            }
        ]
        for axis in MODULE.AXES:
            for level in MODULE.DERIVED_LEVELS:
                records.append(
                    {
                        "scene_id": f"scene__{axis}_{level}",
                        "parent": "base/scene.json",
                        "kind": "sweep",
                        "axis": axis,
                        "level_index": level,
                    }
                )
        return records

    def test_complete_group_is_published(self) -> None:
        records = self.source_records()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            physics = root / "physics.json"
            MODULE.write_json(
                source,
                {
                    "schema_version": "physweep_physics_sweep_manifest_v2",
                    "sample_count": 13,
                    "records": records,
                },
            )
            MODULE.write_json(
                physics,
                {
                    "schema_version": "physweep_pybullet_batch_record_v1",
                    "dataset_id": "test",
                    "sample_count": 13,
                    "records": [
                        {
                            "scene_id": record["scene_id"],
                            "ok": True,
                            "audit_passed": True,
                        }
                        for record in records
                    ],
                },
            )
            accepted, report = MODULE.finalize(root, source, physics)
            self.assertEqual(accepted["accepted_group_count"], 1)
            self.assertEqual(accepted["sample_count"], 13)
            self.assertEqual(report["rejected_group_count"], 0)

    def test_one_failed_sample_rejects_the_whole_group(self) -> None:
        records = self.source_records()
        physics_records = [
            {
                "scene_id": record["scene_id"],
                "ok": True,
                "audit_passed": record["scene_id"] != "scene__mass_kg_0",
                "failed_checks": (
                    ["adapter_hard_invariants"]
                    if record["scene_id"] == "scene__mass_kg_0"
                    else []
                ),
            }
            for record in records
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            physics = root / "physics.json"
            MODULE.write_json(
                source,
                {
                    "schema_version": "physweep_physics_sweep_manifest_v2",
                    "sample_count": 13,
                    "records": records,
                },
            )
            MODULE.write_json(
                physics,
                {
                    "schema_version": "physweep_pybullet_batch_record_v1",
                    "dataset_id": "test",
                    "sample_count": 13,
                    "records": physics_records,
                },
            )
            accepted, report = MODULE.finalize(root, source, physics)
            self.assertEqual(accepted["accepted_group_count"], 0)
            self.assertEqual(accepted["sample_count"], 0)
            self.assertEqual(report["rejected_group_count"], 1)
            self.assertEqual(report["failed_check_counts"], {"adapter_hard_invariants": 1})

    def test_incomplete_axis_levels_are_rejected(self) -> None:
        records = self.source_records()[:-1]
        with self.assertRaisesRegex(ValueError, "cardinality"):
            MODULE.validate_source_groups(records)


if __name__ == "__main__":
    unittest.main()
