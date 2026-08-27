from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.build_sweep_release_view import (
    DERIVED_LEVELS,
    SWEEP_AXES,
    SWEEP_INDEX_FIELDS,
    group_manifest,
    sibling_release_roots,
    sweep_descriptor,
    sweep_sort_key,
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
        records[1]["scene_id"] = records[0]["scene_id"]
        with self.assertRaisesRegex(ValueError, "duplicate sweep scene id"):
            validate_groups(records, base_by_source, base_groups)

    def test_sweep_descriptor_rejects_base_level(self) -> None:
        record = self.records()[0]
        self.assertEqual(
            set(sweep_descriptor(record)),
            {"target_object_id", "parameter", "level_index", "value"},
        )
        record["level_index"] = 2
        with self.assertRaisesRegex(ValueError, "invalid sweep descriptor"):
            sweep_descriptor(record)
        self.assertNotIn("value", SWEEP_INDEX_FIELDS)
        self.assertNotIn("target_object_id", SWEEP_INDEX_FIELDS)

    def test_release_roots_are_siblings_and_axis_order_is_canonical(self) -> None:
        base, sweep = sibling_release_roots(Path("release/base"), Path("release/sweep"))
        self.assertEqual(base.parent, sweep.parent)
        with self.assertRaisesRegex(ValueError, "distinct siblings"):
            sibling_release_roots(Path("release/base"), Path("other/sweep"))
        indexed = [
            {"parameter": record["axis"], "level_index": record["level_index"]}
            for record in reversed(self.records())
        ]
        ordered = sorted(indexed, key=sweep_sort_key)
        self.assertEqual(
            [(record["parameter"], record["level_index"]) for record in ordered],
            [(axis, level) for axis in SWEEP_AXES for level in DERIVED_LEVELS],
        )

    def test_group_index_does_not_duplicate_physical_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            results = []
            for record in self.records():
                scene_id = record["scene_id"]
                sample = work / "generic" / scene_id
                sample.mkdir(parents=True)
                (sample / "metadata.json").write_text(
                    json.dumps(
                        {
                            "group_id": "group_a",
                            "sweep": {
                                "target_object_id": "object_a",
                                "parameter": record["axis"],
                                "level_index": record["level_index"],
                                "value": record["value"],
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                results.append(
                    {
                        "family": "generic",
                        "record": {
                            "scene_id": scene_id,
                            "metadata_sha256": "a" * 64,
                        },
                    }
                )
            manifest = group_manifest(
                output_name="sweep",
                results=list(reversed(results)),
                base_groups={
                    "group_a": {
                        "family": "generic",
                        "scene_id": "group_a__base",
                        "path": "base/generic/group_a__base",
                        "metadata_sha256": "b" * 64,
                    }
                },
                work=work,
            )
            group = manifest["records"][0]
            self.assertEqual(group["target_object_id"], "object_a")
            self.assertTrue(all(set(record) == SWEEP_INDEX_FIELDS for record in group["sweeps"]))


if __name__ == "__main__":
    unittest.main()
