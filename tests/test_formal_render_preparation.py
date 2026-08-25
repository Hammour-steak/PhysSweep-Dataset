from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from prepare_formal_render_manifests import (  # noqa: E402
    review100_selection,
    select_records,
    sha256,
    stage_render_record,
    stress60_selection,
    validated_source_records,
)


class FormalRenderPreparationTests(unittest.TestCase):
    def test_all_selection_can_be_restricted_to_one_pipeline(self) -> None:
        manifest = {
            "records": [
                {"scene_id": "generic", "pipeline": "generic_pybullet"},
                {"scene_id": "pinball_a", "pipeline": "passive_pinball"},
                {"scene_id": "pinball_b", "pipeline": "passive_pinball"},
            ]
        }
        selected = select_records(manifest, "all", "passive_pinball")
        self.assertEqual(
            [record["scene_id"] for record in selected],
            ["pinball_a", "pinball_b"],
        )
        with self.assertRaisesRegex(ValueError, "requires --selection all"):
            select_records(manifest, "pilot20", "passive_pinball")
        with self.assertRaisesRegex(ValueError, "contains no billiards"):
            select_records(manifest, "all", "billiards")

    def test_source_manifest_requires_consistent_unique_records(self) -> None:
        manifest = {
            "schema_version": "physweep_one_object_decoupled_manifest_v3",
            "sample_count": 1,
            "records": [
                {
                    "index": 0,
                    "scene_id": "scene",
                    "pipeline": "generic_pybullet",
                }
            ],
        }
        self.assertEqual(validated_source_records(manifest), manifest["records"])
        manifest["records"].append(dict(manifest["records"][0]))
        manifest["sample_count"] = 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validated_source_records(manifest)

    def test_review100_selection_covers_formal_strata(self) -> None:
        records = []
        index = 0
        motions = (
            "slide_push_1obj",
            "roll_or_slide_1obj",
            "wall_impact_1obj",
            "edge_fall_1obj",
            "drop_fall_1obj",
            "projectile_1obj",
            "arc_projectile_1obj",
            "slope_slide_down_1obj",
            "slope_slide_up_1obj",
            "ramp_to_flat_1obj",
            "bounce_1obj",
        )
        for motion in motions:
            for _ in range(8):
                index += 1
                records.append(
                    {
                        "index": index,
                        "scene_id": f"scene_{index:03d}",
                        "pipeline": "generic_pybullet",
                        "motion_intent": motion,
                        "profile": "five_dimensional_matrix",
                        "environment_id": "generic_matrix",
                    }
                )
        for environment in (
            "curated_support_asset",
            "curated_support_with_prop",
            "workbench_single_object",
        ):
            for profile in (
                "resting_push",
                "edge_exit",
                "diagonal_push",
                "vertical_drop",
            ):
                for _ in range(3):
                    index += 1
                    records.append(
                        {
                            "index": index,
                            "scene_id": f"scene_{index:03d}",
                            "pipeline": "asset_proxy",
                            "motion_intent": "slide_push_1obj",
                            "profile": profile,
                            "environment_id": environment,
                        }
                    )
        for profile in ("single_ball_free_roll", "single_ball_rail_rebound"):
            index += 1
            records.append(
                {
                    "index": index,
                    "scene_id": f"scene_{index:03d}",
                    "pipeline": "billiards",
                    "motion_intent": "wall_impact_1obj",
                    "profile": profile,
                    "environment_id": "billiards_single_ball",
                }
            )

        selected = review100_selection(
            {"dataset_id": "test", "records": records}
        )

        self.assertEqual(len(selected), 100)
        self.assertEqual(
            sum(record["pipeline"] == "generic_pybullet" for record in selected),
            70,
        )
        self.assertEqual(
            sum(record["pipeline"] == "asset_proxy" for record in selected),
            28,
        )
        self.assertEqual(
            sum(record["pipeline"] == "billiards" for record in selected),
            2,
        )
        self.assertEqual(
            {
                record["motion_intent"]
                for record in selected
                if record["pipeline"] == "generic_pybullet"
            },
            set(motions),
        )

        for profile in ("dense_pinfield_descent", "offset_pinfield_descent"):
            index += 1
            records.append(
                {
                    "index": index,
                    "scene_id": f"scene_{index:03d}",
                    "pipeline": "passive_pinball",
                    "motion_intent": "drop_fall_1obj",
                    "profile": profile,
                    "environment_id": "passive_pinball_board",
                }
            )
        selected = review100_selection(
            {"dataset_id": "test-v4", "records": records}
        )
        self.assertEqual(len(selected), 100)
        self.assertEqual(
            sum(record["pipeline"] == "generic_pybullet" for record in selected),
            68,
        )
        self.assertEqual(
            sum(record["pipeline"] == "passive_pinball" for record in selected),
            2,
        )
        self.assertEqual(
            {
                record["profile"]
                for record in selected
                if record["pipeline"] == "passive_pinball"
            },
            {"dense_pinfield_descent", "offset_pinfield_descent"},
        )

    def test_stress60_selection_has_declared_strata(self) -> None:
        records = []
        index = 0
        for motion in (
            "wall_impact_1obj",
            "roll_or_slide_1obj",
            "projectile_1obj",
            "slope_slide_down_1obj",
            "slope_slide_up_1obj",
            "slide_push_1obj",
            "arc_projectile_1obj",
            "bounce_1obj",
            "drop_fall_1obj",
        ):
            for _ in range(4):
                index += 1
                records.append(
                    {
                        "index": index,
                        "scene_id": f"scene_{index:03d}",
                        "pipeline": "generic_pybullet",
                        "motion_intent": motion,
                        "profile": "five_dimensional_matrix",
                    }
                )
        for motion in ("edge_fall_1obj", "ramp_to_flat_1obj"):
            for _ in range(10):
                index += 1
                records.append(
                    {
                        "index": index,
                        "scene_id": f"scene_{index:03d}",
                        "pipeline": "generic_pybullet",
                        "motion_intent": motion,
                        "profile": "five_dimensional_matrix",
                    }
                )
        for profile in ("resting_push", "edge_exit", "diagonal", "vertical_drop"):
            for _ in range(2):
                index += 1
                records.append(
                    {
                        "index": index,
                        "scene_id": f"scene_{index:03d}",
                        "pipeline": "asset_proxy",
                        "motion_intent": "slide_push_1obj",
                        "profile": profile,
                    }
                )
        for profile in ("single_ball_free_roll", "single_ball_rail_rebound"):
            index += 1
            records.append(
                {
                    "index": index,
                    "scene_id": f"scene_{index:03d}",
                    "pipeline": "billiards",
                    "motion_intent": "wall_impact_1obj",
                    "profile": profile,
                }
            )

        selected = stress60_selection({"dataset_id": "test", "records": records})

        self.assertEqual(len(selected), 60)
        self.assertEqual(
            sum(record["pipeline"] == "generic_pybullet" for record in selected),
            50,
        )
        self.assertEqual(
            sum(record["pipeline"] == "asset_proxy" for record in selected), 8
        )
        self.assertEqual(
            sum(record["pipeline"] == "billiards" for record in selected), 2
        )
        self.assertEqual(
            sum(
                record["motion_intent"] in {"edge_fall_1obj", "ramp_to_flat_1obj"}
                for record in selected
            ),
            20,
        )

        for profile in ("dense_pinfield_descent", "offset_pinfield_descent"):
            index += 1
            records.append(
                {
                    "index": index,
                    "scene_id": f"scene_{index:03d}",
                    "pipeline": "passive_pinball",
                    "motion_intent": "drop_fall_1obj",
                    "profile": profile,
                }
            )
        selected = stress60_selection(
            {"dataset_id": "test-v4", "records": records}
        )
        self.assertEqual(len(selected), 60)
        self.assertEqual(
            sum(record["pipeline"] == "generic_pybullet" for record in selected),
            48,
        )
        self.assertEqual(
            sum(record["pipeline"] == "passive_pinball" for record in selected),
            2,
        )
        self.assertEqual(
            {
                record["profile"]
                for record in selected
                if record["pipeline"] == "passive_pinball"
            },
            {"dense_pinfield_descent", "offset_pinfield_descent"},
        )

    def test_render_override_preserves_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_path = root / "datasets/source/scene/metadata.json"
            metadata_path.parent.mkdir(parents=True)
            original = {"scene_id": "scene_001", "render": {"samples": 32}}
            metadata_path.write_text(json.dumps(original), encoding="utf-8")
            record = {
                "scene_id": "scene_001",
                "metadata_path": "datasets/source/scene/metadata.json",
                "metadata_sha256": sha256(metadata_path),
            }

            staged = stage_render_record(root, record, root / "outputs/staging/asset")

            self.assertEqual(staged["metadata_path"], record["metadata_path"])
            self.assertEqual(staged["metadata_sha256"], record["metadata_sha256"])
            self.assertEqual(
                staged["render_output"]["video_path"],
                "outputs/staging/asset/videos/scene_001.mp4",
            )
            self.assertEqual(json.loads(metadata_path.read_text()), original)
            self.assertFalse((root / "outputs/staging/asset/metadata").exists())

    def test_declared_source_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_path = root / "datasets/source/metadata.json"
            metadata_path.parent.mkdir(parents=True)
            metadata_path.write_text("{}", encoding="utf-8")
            record = {
                "scene_id": "scene_001",
                "metadata_path": "datasets/source/metadata.json",
                "metadata_sha256": "0" * 64,
            }
            with self.assertRaises(ValueError):
                stage_render_record(root, record, root / "outputs/staging")


if __name__ == "__main__":
    unittest.main()
