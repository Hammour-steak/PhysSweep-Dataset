from __future__ import annotations

import unittest

from tools.prepare_passive_pinball_v4_replacements import (
    select_replacement_slots,
    stable_seed,
)
from tools.publish_passive_pinball_v4_release import (
    EXPECTED_SOURCE_PIPELINES,
    EXPECTED_V4_PIPELINES,
    replacement_index,
)


class PassivePinballV4ReleaseTests(unittest.TestCase):
    def source_records(self) -> list[dict[str, object]]:
        records = []
        for index in range(1, 101):
            records.append(
                {
                    "index": index,
                    "scene_id": f"source_{index:03d}",
                    "metadata_path": f"source/{index:03d}/metadata.json",
                    "metadata_sha256": f"{index:064x}",
                    "pipeline": "generic_pybullet",
                    "motion_intent": "drop_fall_1obj",
                }
            )
        records.append(
            {
                "index": 101,
                "scene_id": "not_eligible",
                "metadata_path": "source/not_eligible/metadata.json",
                "metadata_sha256": "f" * 64,
                "pipeline": "asset_proxy",
                "motion_intent": "drop_fall_1obj",
            }
        )
        return records

    def test_slot_selection_is_deterministic_and_eligible(self) -> None:
        records = self.source_records()
        first = select_replacement_slots(records, 32, 20260826)
        second = select_replacement_slots(list(reversed(records)), 32, 20260826)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)
        self.assertTrue(
            all(
                record["pipeline"] == "generic_pybullet"
                and record["motion_intent"] == "drop_fall_1obj"
                for record in first
            )
        )
        self.assertEqual(
            [record["index"] for record in first],
            sorted(record["index"] for record in first),
        )

    def test_candidate_seed_is_stable_and_positive(self) -> None:
        self.assertEqual(stable_seed("candidate"), stable_seed("candidate"))
        self.assertNotEqual(stable_seed("candidate"), stable_seed("candidate-2"))
        self.assertGreater(stable_seed("candidate"), 0)

    def test_replacement_index_requires_whole_slot_provenance(self) -> None:
        source_records = self.source_records()[:32]
        replacements = []
        for source in source_records:
            replacements.append(
                {
                    "index": source["index"],
                    "scene_id": f"replacement_{source['index']}",
                    "metadata_path": f"replacement/{source['index']}/metadata.json",
                    "metadata_sha256": "a" * 64,
                    "pipeline": "passive_pinball",
                    "motion_intent": "drop_fall_1obj",
                    "replaces_scene_id": source["scene_id"],
                    "replaces_metadata_path": source["metadata_path"],
                    "replaces_metadata_sha256": source["metadata_sha256"],
                    "replaces_pipeline": "generic_pybullet",
                }
            )
        indexed, old_paths, new_paths = replacement_index(
            {"records": source_records},
            {"sample_count": 32, "records": replacements},
        )
        self.assertEqual(len(indexed), 32)
        self.assertEqual(len(old_paths), 32)
        self.assertEqual(len(new_paths), 32)
        replacements[0]["motion_intent"] = "wall_impact_1obj"
        with self.assertRaisesRegex(ValueError, "preserved slot"):
            replacement_index(
                {"records": source_records},
                {"sample_count": 32, "records": replacements},
            )

    def test_v4_distribution_replaces_exactly_32_generic_groups(self) -> None:
        self.assertEqual(sum(EXPECTED_SOURCE_PIPELINES.values()), 3200)
        self.assertEqual(sum(EXPECTED_V4_PIPELINES.values()), 3200)
        self.assertEqual(
            EXPECTED_SOURCE_PIPELINES["generic_pybullet"]
            - EXPECTED_V4_PIPELINES["generic_pybullet"],
            32,
        )
        self.assertEqual(EXPECTED_V4_PIPELINES["passive_pinball"], 32)


if __name__ == "__main__":
    unittest.main()
