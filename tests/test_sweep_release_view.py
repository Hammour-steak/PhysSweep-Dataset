from __future__ import annotations

import unittest

from tools.build_sweep_release_view import (
    DERIVED_LEVELS,
    SWEEP_AXES,
    sweep_descriptor,
    validate_groups,
)


class SweepReleaseViewTests(unittest.TestCase):
    def records(self) -> list[dict]:
        return [
            {
                "scene_id": f"scene_{axis}_{level}",
                "parent": "source/base.json",
                "target_object_id": "object_a",
                "axis": axis,
                "level_index": level,
                "value": float(level + 1),
            }
            for axis in SWEEP_AXES
            for level in DERIVED_LEVELS
        ]

    def test_group_index_requires_exact_one_factor_grid(self) -> None:
        base_by_source = {"source/base.json": {"scene_id": "group_a"}}
        base_groups = {"group_a": {"family": "generic"}}
        records = self.records()
        mapping = validate_groups(records, base_by_source, base_groups)
        self.assertEqual(set(mapping.values()), {"group_a"})
        with self.assertRaisesRegex(ValueError, "one-factor group"):
            validate_groups(records[:-1], base_by_source, base_groups)

    def test_sweep_descriptor_rejects_base_level(self) -> None:
        record = self.records()[0]
        self.assertEqual(
            set(sweep_descriptor(record)),
            {"target_object_id", "parameter", "level_index", "value"},
        )
        record["level_index"] = 2
        with self.assertRaisesRegex(ValueError, "invalid sweep descriptor"):
            sweep_descriptor(record)


if __name__ == "__main__":
    unittest.main()
