from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from publish_sweep_release import merge_base_records, validate_groups  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
