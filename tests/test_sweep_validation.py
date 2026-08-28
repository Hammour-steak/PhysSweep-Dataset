from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.core.sweep_values import (
    SWEEP_AXES,
    SWEEP_DERIVED_LEVELS,
    SWEEP_VARIANTS_PER_TARGET,
    sweep_group_size,
)
from tools.release.sweep_validation import validate_groups
from tools.release.source_release import (
    _validate_sweep_record_bindings,
    publish_source_release,
)


def sweep_records(target_count: int) -> list[dict]:
    records = [
        {
            "parent": "base/metadata.json",
            "scene_id": "canonical_base",
            "kind": "base",
        }
    ]
    for target_index in range(target_count):
        for axis in SWEEP_AXES:
            for level_index in SWEEP_DERIVED_LEVELS:
                records.append(
                    {
                        "parent": "base/metadata.json",
                        "scene_id": (
                            f"target_{target_index}_{axis}_{level_index}"
                        ),
                        "kind": "sweep",
                        "axis": axis,
                        "level_index": level_index,
                        "target_object_id": f"object_{target_index}",
                        "target_object_index": target_index,
                    }
                )
    return records


class SweepValidationTests(unittest.TestCase):
    def test_group_size_scales_by_target_object(self) -> None:
        self.assertEqual(SWEEP_VARIANTS_PER_TARGET, 12)
        self.assertEqual(sweep_group_size(1), 13)
        self.assertEqual(sweep_group_size(2), 25)
        self.assertEqual(sweep_group_size(3), 37)

    def test_two_object_group_requires_a_complete_grid_per_target(self) -> None:
        records = sweep_records(2)
        summary = validate_groups(records, expected_target_indices=(0, 1))
        self.assertEqual(summary.base_count, 1)
        self.assertEqual(summary.target_groups_per_base, 2)
        self.assertEqual(summary.derived_count, 24)
        with self.assertRaisesRegex(ValueError, "base plus complete target sweeps"):
            validate_groups(records[:-1], expected_target_indices=(0, 1))

    def test_target_identity_cannot_change_inside_a_group(self) -> None:
        records = sweep_records(2)
        records[-1]["target_object_id"] = "different_object"
        with self.assertRaisesRegex(ValueError, "identity differs"):
            validate_groups(records, expected_target_indices=(0, 1))

    def test_target_indices_and_ids_are_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "indices"):
            validate_groups(sweep_records(1), expected_target_indices=(False,))
        records = sweep_records(1)
        records[-1]["target_object_id"] = ""
        with self.assertRaisesRegex(ValueError, "id is invalid"):
            validate_groups(records, expected_target_indices=(0,))

    def test_manifest_coordinates_bind_to_metadata_and_object_order(self) -> None:
        record = {
            "scene_id": "scene",
            "parent": "base/metadata.json",
            "kind": "sweep",
            "axis": "mass_kg",
            "level_index": 0,
            "value": 0.5,
            "target_object_id": "object_b",
            "target_object_index": 1,
            "source_schema_version": "test_schema",
        }
        validated = [
            (
                Path("metadata.json"),
                {
                    "scene_id": "scene",
                    "sweep": {
                        "parent_metadata_path": "base/metadata.json",
                        "kind": "sweep",
                        "axis": "mass_kg",
                        "level_index": 0,
                        "value": 0.5,
                        "target_object_id": "object_b",
                        "target_object_index": 1,
                        "source_schema_version": "test_schema",
                    },
                },
                ("object_a", "object_b"),
            )
        ]
        _validate_sweep_record_bindings([record], validated)
        record["target_object_index"] = 0
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            _validate_sweep_record_bindings([record], validated)

    def test_two_object_source_release_publishes_twenty_four_variants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_records = [
                {
                    "scene_id": "base",
                    "metadata_path": "datasets/run/base/shared.json",
                    "metadata_sha256": "a" * 64,
                }
            ]
            metadata_records = sweep_records(2)
            for record in metadata_records:
                record.update(
                    {
                        "parent": "datasets/run/base/shared.json",
                        "path": "datasets/run/sweep/shared.json",
                        "metadata_sha256": "b" * 64,
                        "source_schema_version": "test_schema",
                    }
                )
            physics_records = [
                {
                    "scene_id": record["scene_id"],
                    "metadata_path": "datasets/run/sweep/shared.json",
                    "metadata_sha256": "b" * 64,
                    "source_schema_version": "test_schema",
                    "ok": True,
                    "audit_passed": True,
                    "failed_checks": [],
                }
                for record in metadata_records
            ]

            def write(path: Path, value: object) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value), encoding="utf-8")

            base_manifest = root / "datasets/run/base/manifest.json"
            metadata_manifest = root / "datasets/run/sweep/manifest.json"
            physics_manifest = root / "datasets/run/physics/manifest.json"
            write(
                base_manifest,
                {"sample_count": len(base_records), "records": base_records},
            )
            write(
                metadata_manifest,
                {
                    "sample_count": len(metadata_records),
                    "records": metadata_records,
                },
            )
            write(
                physics_manifest,
                {
                    "sample_count": len(physics_records),
                    "records": physics_records,
                },
            )
            shared = root / "datasets/run/sweep/shared.json"
            write(shared, {})
            output = root / "datasets/run/release"
            with patch(
                "tools.release.source_release._validate_object_count"
            ), patch(
                "tools.release.source_release._validate_sweep_record_bindings"
            ), patch(
                "tools.release.source_release.validate_source_artifacts"
            ), patch(
                "tools.release.source_release._verified_metadata",
                return_value=shared,
            ):
                release = publish_source_release(
                    root=root,
                    base_manifest_path=base_manifest,
                    sweep_metadata_manifest_path=metadata_manifest,
                    sweep_physics_manifest_path=physics_manifest,
                    output=output,
                    object_count=2,
                    dataset_id="physweep_two_object",
                    release_schema="physweep_two_object_source_release_v1",
                )
            self.assertEqual(release["base_count"], 1)
            self.assertEqual(release["derived_count"], 24)
            published = json.loads(
                (output / "metadata_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(published["group_size"], 25)


if __name__ == "__main__":
    unittest.main()
