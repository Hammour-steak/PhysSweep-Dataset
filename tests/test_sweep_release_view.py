from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.core.sweep_values import SWEEP_AXES, SWEEP_DERIVED_LEVELS
from tools.release.base_release_schema import SWEEP_SAMPLE_SCHEMA, sha256
from tools.release.base_release_view import (
    PipelineSpec,
    release_contract_fields,
    write_pipeline_manifests,
)
from tools.release.layout import release_roots
from tools.release.sweep_release_view import (
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
                "target_object_index": 0,
                "axis": axis,
                "level_index": level,
                "value": float(level + 1),
            }
            for axis in SWEEP_AXES
            for level in SWEEP_DERIVED_LEVELS
        ]

    def test_group_index_requires_exact_one_factor_grid(self) -> None:
        base_by_source = {"source/base.json": {"scene_id": "group_a"}}
        base_groups = {"group_a": {"family": "generic"}}
        records = self.records()
        mapping = validate_groups(
            records, base_by_source, base_groups, object_count=1
        )
        self.assertEqual(set(mapping.values()), {"group_a"})
        with self.assertRaisesRegex(ValueError, "one-factor group"):
            validate_groups(
                records[:-1], base_by_source, base_groups, object_count=1
            )
        records[1]["scene_id"] = records[0]["scene_id"]
        with self.assertRaisesRegex(ValueError, "duplicate sweep scene id"):
            validate_groups(records, base_by_source, base_groups, object_count=1)

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

    def test_sweep_uses_shared_release_manifest_writers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "generic").mkdir()
            spec = PipelineSpec("generic", "source_schema", work, work)
            bindings = write_pipeline_manifests(
                work=work,
                specs={spec.source_schema_version: spec},
                grouped={
                    "generic": [
                        {"scene_id": "scene_b", "metadata_sha256": "b" * 64},
                        {"scene_id": "scene_a", "metadata_sha256": "a" * 64},
                    ]
                },
                pipeline_schema="sweep_pipeline_schema",
                sample_schema=SWEEP_SAMPLE_SCHEMA,
            )
            path = work / "generic/manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                [record["scene_id"] for record in manifest["records"]],
                ["scene_a", "scene_b"],
            )
            self.assertEqual(manifest["sample_schema_version"], SWEEP_SAMPLE_SCHEMA)
            self.assertEqual(bindings["generic"]["manifest_sha256"], sha256(path))
        contracts = release_contract_fields(
            render_contract={"engine": "test"},
            sample_schema=SWEEP_SAMPLE_SCHEMA,
        )
        self.assertEqual(contracts["render_contract"], {"engine": "test"})
        self.assertEqual(contracts["sample_schema_version"], SWEEP_SAMPLE_SCHEMA)

    def test_release_roots_are_siblings_and_axis_order_is_canonical(self) -> None:
        base, sweep = release_roots(Path("outputs/one_object"), object_count=1)
        base, sweep = sibling_release_roots(base, sweep, object_count=1)
        self.assertEqual(base.parent, sweep.parent)
        _, staging = sibling_release_roots(
            base,
            sweep.with_name(".sweep.building"),
            object_count=1,
            allow_staging_markers=True,
        )
        self.assertEqual(staging.name, ".sweep.building")
        with self.assertRaisesRegex(ValueError, "one_object/sweep"):
            sibling_release_roots(
                base, sweep.with_name(".sweep.building"), object_count=1
            )
        with self.assertRaisesRegex(ValueError, "one_object/base"):
            sibling_release_roots(
                Path("release/base"), Path("other/sweep"), object_count=1
            )
        indexed = [
            {"parameter": record["axis"], "level_index": record["level_index"]}
            for record in reversed(self.records())
        ]
        ordered = sorted(indexed, key=sweep_sort_key)
        self.assertEqual(
            [(record["parameter"], record["level_index"]) for record in ordered],
            [
                (axis, level)
                for axis in SWEEP_AXES
                for level in SWEEP_DERIVED_LEVELS
            ],
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
                            "physics": {"objects": [{"object_id": "object_a"}]},
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
                object_count=1,
            )
            group = manifest["records"][0]
            self.assertEqual(group["target_object_id"], "object_a")
            self.assertTrue(all(set(record) == SWEEP_INDEX_FIELDS for record in group["sweeps"]))

    def test_multi_object_group_index_nests_complete_target_grids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            results = []
            for target_index, target_id in enumerate(("object_a", "object_b")):
                for axis in SWEEP_AXES:
                    for level in SWEEP_DERIVED_LEVELS:
                        scene_id = f"scene_{target_index}_{axis}_{level}"
                        sample = work / "generic" / scene_id
                        sample.mkdir(parents=True)
                        (sample / "metadata.json").write_text(
                            json.dumps(
                                {
                                    "group_id": "group_a",
                                    "physics": {
                                        "objects": [
                                            {"object_id": "object_a"},
                                            {"object_id": "object_b"},
                                        ]
                                    },
                                    "sweep": {
                                        "target_object_id": target_id,
                                        "parameter": axis,
                                        "level_index": level,
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
                results=results,
                base_groups={
                    "group_a": {
                        "family": "generic",
                        "scene_id": "group_a__base",
                        "path": "base/generic/group_a__base",
                        "metadata_sha256": "b" * 64,
                    }
                },
                work=work,
                object_count=2,
            )
            group = manifest["records"][0]
            self.assertEqual(manifest["schema_version"], "physweep_sweep_group_manifest_v2")
            self.assertEqual(manifest["sweep_count"], 24)
            self.assertEqual(
                [target["target_object_index"] for target in group["targets"]],
                [0, 1],
            )
            self.assertTrue(
                all(len(target["sweeps"]) == 12 for target in group["targets"])
            )


if __name__ == "__main__":
    unittest.main()
