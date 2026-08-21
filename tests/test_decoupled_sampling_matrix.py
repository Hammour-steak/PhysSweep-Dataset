from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from sample_one_object_scene_matrix import (  # noqa: E402
    MATRIX_PATH,
    allocate_axis_counts,
    assign_environment_ids,
    build_schedule,
    generic_retry_seed,
    validate_matrix,
)


class DecoupledSamplingMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.matrix = json.loads(
            MATRIX_PATH.read_text(encoding="utf-8")
        )
        cls.production_spec = json.loads(
            (ROOT / "configs/production_video.json").read_text(
                encoding="utf-8"
            )
        )

    def test_matrix_only_references_reviewed_assets(self) -> None:
        validate_matrix(ROOT, self.matrix)

    def test_generic_retry_seeds_are_deterministic_and_slot_specific(self) -> None:
        seed = generic_retry_seed(20260804, 17, 2)
        self.assertEqual(seed, generic_retry_seed(20260804, 17, 2))
        self.assertNotEqual(seed, generic_retry_seed(20260804, 18, 2))
        self.assertNotEqual(seed, generic_retry_seed(20260804, 17, 3))
        with self.assertRaises(ValueError):
            generic_retry_seed(20260804, 17, 1)

    def test_dependency_and_implementation_sets_are_exact(self) -> None:
        self.assertEqual(
            set(self.matrix["dependencies"]),
            {
                "generic_sampling_bundle",
                "asset_proxy_registry",
                "physical_proxy_catalog",
                "asset_scene_composition",
                "asset_semantic_scene_rules",
                "visual_sampling",
                "physics_backend",
                "backend_capabilities",
                "production_video",
                "environment_collision_proxies",
                "environment_composition",
            },
        )
        self.assertEqual(
            set(self.matrix["implementation"]),
            {
                "matrix_sampler",
                "proxy_catalog",
                "asset_proxy_sampler",
                "billiards_generator",
                "billiards_renderer",
                "motion_rule_package",
                "motion_rule_contracts",
                "motion_rule_common",
                "motion_rule_registry",
                "motion_rule_planar",
                "motion_rule_ballistic",
                "motion_rule_incline",
                "motion_rule_transition",
            },
        )
        invalid = copy.deepcopy(self.matrix)
        invalid["dependencies"]["obsolete_rule"] = "configs/compatibility.json"
        with self.assertRaises(ValueError):
            validate_matrix(ROOT, invalid)

    def test_formal_motion_distribution_is_explicit(self) -> None:
        self.assertEqual(
            allocate_axis_counts(self.matrix["motion_intents"], 40, "motion"),
            {
                "slide_push_1obj": 5,
                "roll_or_slide_1obj": 4,
                "wall_impact_1obj": 4,
                "edge_fall_1obj": 3,
                "drop_fall_1obj": 4,
                "projectile_1obj": 4,
                "arc_projectile_1obj": 4,
                "slope_slide_down_1obj": 3,
                "slope_slide_up_1obj": 3,
                "ramp_to_flat_1obj": 3,
                "bounce_1obj": 3,
            },
        )

    def test_formal_environment_distribution_is_independent(self) -> None:
        self.assertEqual(
            allocate_axis_counts(self.matrix["environments"], 40, "environment"),
            {
                "generic_matrix": 29,
                "curated_support_asset": 4,
                "curated_support_with_prop": 4,
                "billiards_single_ball": 1,
                "workbench_single_object": 2,
            },
        )

    def test_motion_distribution_does_not_change_with_environment_weights(self) -> None:
        changed = copy.deepcopy(self.matrix)
        weights = {
            "generic_matrix": 0.80,
            "curated_support_asset": 0.05,
            "curated_support_with_prop": 0.05,
            "billiards_single_ball": 0.05,
            "workbench_single_object": 0.05,
        }
        for environment in changed["environments"]:
            environment["weight"] = weights[environment["id"]]
        baseline = build_schedule(self.matrix, 40, 20260723)
        alternate = build_schedule(changed, 40, 20260723)
        self.assertEqual(
            [record["motion_intent"] for record in baseline],
            [record["motion_intent"] for record in alternate],
        )
        self.assertNotEqual(
            Counter(record["environment_id"] for record in baseline),
            Counter(record["environment_id"] for record in alternate),
        )

    def test_every_binding_obeys_compatibility_table(self) -> None:
        schedule = build_schedule(self.matrix, 400, 20260723)
        environments = {
            environment["id"]: environment
            for environment in self.matrix["environments"]
        }
        for record in schedule:
            bindings = environments[record["environment_id"]]["motion_bindings"]
            self.assertIn(record["motion_intent"], bindings)
            self.assertIn(record["profile"], bindings[record["motion_intent"]])
            self.assertNotIn("scene_family", record)

    def test_formal_batch_reserves_every_motion_in_generic_matrix(self) -> None:
        schedule = build_schedule(self.matrix, 500, 20260729)
        generic = Counter(
            record["motion_intent"]
            for record in schedule
            if record["environment_id"] == "generic_matrix"
        )
        self.assertEqual(
            set(generic),
            {str(record["id"]) for record in self.matrix["motion_intents"]},
        )
        self.assertGreaterEqual(min(generic.values()), 16)

    def test_feasible_assignment_is_not_rejected_by_greedy_order(self) -> None:
        environments = [
            {
                "id": "catch_all",
                "motion_bindings": {"a": ["all"], "b": ["all"], "c": ["all"]},
            },
            {
                "id": "flexible",
                "motion_bindings": {"a": ["flex"], "b": ["flex"]},
            },
            {
                "id": "needs_a_and_c",
                "motion_bindings": {"a": ["narrow"], "c": ["narrow"]},
            },
        ]
        allocations = {"catch_all": 0, "flexible": 1, "needs_a_and_c": 2}
        for seed in range(30):
            assigned = assign_environment_ids(
                ["a", "b", "c"],
                environments,
                allocations,
                {"a", "b", "c"},
                random.Random(seed),
            )
            self.assertEqual(Counter(assigned), Counter(allocations))
            for motion, environment_id in zip(["a", "b", "c"], assigned):
                environment = next(
                    item for item in environments if item["id"] == environment_id
                )
                self.assertIn(motion, environment["motion_bindings"])

    def test_billiards_is_strongly_constrained(self) -> None:
        schedule = build_schedule(self.matrix, 400, 20260723)
        billiards = [
            record
            for record in schedule
            if record["environment_id"] == "billiards_single_ball"
        ]
        self.assertEqual(
            {
                (record["motion_intent"], record["profile"])
                for record in billiards
            },
            {
                ("roll_or_slide_1obj", "single_ball_free_roll"),
                ("wall_impact_1obj", "single_ball_rail_rebound"),
            },
        )
        self.assertEqual(
            {record["support_asset_id"] for record in billiards},
            {"sketchfab_bg_0f2ae181a2dd4b00a6ec25073692037f"},
        )
        self.assertEqual(
            {record["dynamic_asset_id"] for record in billiards},
            {None},
        )

    def test_common_motions_reach_generic_and_specialized_environments(self) -> None:
        schedule = build_schedule(self.matrix, 400, 20260723)
        environments_by_motion: dict[str, set[str]] = {}
        for record in schedule:
            environments_by_motion.setdefault(
                record["motion_intent"], set()
            ).add(record["environment_id"])
        self.assertIn("generic_matrix", environments_by_motion["slide_push_1obj"])
        self.assertIn(
            "curated_support_asset",
            environments_by_motion["slide_push_1obj"],
        )
        self.assertIn("generic_matrix", environments_by_motion["drop_fall_1obj"])
        self.assertIn(
            "workbench_single_object",
            environments_by_motion["drop_fall_1obj"],
        )
        self.assertIn(
            "billiards_single_ball",
            environments_by_motion["roll_or_slide_1obj"],
        )

    def test_specialized_schedule_obeys_dynamic_pools(self) -> None:
        schedule = build_schedule(self.matrix, 400, 20260723)
        environments = {
            environment["id"]: environment
            for environment in self.matrix["environments"]
        }
        for record in schedule:
            environment = environments[record["environment_id"]]
            entries = {
                entry["support_asset_id"]: entry["dynamic_pool_id"]
                for entry in environment.get("support_dynamic_entries", [])
            }
            pairs = {
                (
                    pair["support_asset_id"],
                    pair["static_prop_asset_id"],
                ): pair["dynamic_pool_id"]
                for pair in environment.get("support_prop_pairs", [])
            }
            if entries:
                pool_id = entries[record["support_asset_id"]]
                self.assertIn(
                    record["dynamic_asset_id"],
                    environment["dynamic_pools"][pool_id],
                )
                entry = next(
                    item
                    for item in environment["support_dynamic_entries"]
                    if item["support_asset_id"] == record["support_asset_id"]
                )
                self.assertIn(record["profile"], entry["profiles"])
            if pairs:
                pool_id = pairs[
                    (
                        record["support_asset_id"],
                        record["static_prop_asset_id"],
                    )
                ]
                self.assertIn(
                    record["dynamic_asset_id"],
                    environment["dynamic_pools"][pool_id],
                )

    def test_edge_exit_only_uses_reviewed_exit_supports(self) -> None:
        schedule = build_schedule(self.matrix, 500, 20260729)
        edge_records = [
            record
            for record in schedule
            if record["environment_id"] == "curated_support_asset"
            and record["profile"] == "edge_exit"
        ]
        self.assertTrue(edge_records)
        self.assertEqual(
            {record["support_asset_id"] for record in edge_records},
            {"sketchfab_bg_bb73599044cf49e186b75842d63a280e"},
        )

    def test_edge_exit_respects_dynamic_eligibility(self) -> None:
        eligible = {"sketchfab_4ae035ea89ea40bbaa82403b9c36afab"}
        schedule = build_schedule(
            self.matrix,
            500,
            20260802,
            profile_dynamic_eligibility={"edge_exit": eligible},
        )
        edge_records = [
            record
            for record in schedule
            if record["profile"] == "edge_exit"
        ]
        self.assertTrue(edge_records)
        self.assertEqual(
            {record["dynamic_asset_id"] for record in edge_records}, eligible
        )

    def test_schedule_is_seeded_and_reproducible(self) -> None:
        first = build_schedule(self.matrix, 40, 12345)
        second = build_schedule(self.matrix, 40, 12345)
        third = build_schedule(self.matrix, 40, 12346)
        self.assertEqual(first, second)
        self.assertNotEqual(first, third)

    def test_schedule_is_independent_of_python_hash_seed(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import json; "
                "from sample_one_object_scene_matrix import MATRIX_PATH, build_schedule; "
                "matrix=json.loads(MATRIX_PATH.read_text(encoding='utf-8')); "
                "print(json.dumps(build_schedule(matrix, 40, 12345), sort_keys=True))"
            ),
        ]
        outputs = []
        for hash_seed in ("1", "987654"):
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = hash_seed
            environment["PYTHONPATH"] = str(ROOT / "tools")
            outputs.append(
                subprocess.check_output(
                    command,
                    cwd=ROOT,
                    env=environment,
                    text=True,
                )
            )
        self.assertEqual(outputs[0], outputs[1])

    def test_formal_video_spec_is_consistent(self) -> None:
        spec = self.production_spec
        self.assertEqual(spec["duration_s"], 4.0)
        self.assertEqual(spec["output_fps"], 24)
        self.assertEqual(spec["frame_count"], 97)
        self.assertEqual(spec["resolution"], [1280, 720])
        self.assertEqual(spec["samples"], 32)
        self.assertEqual(
            spec["frame_count"],
            int(round(spec["duration_s"] * spec["output_fps"])) + 1,
        )


if __name__ == "__main__":
    unittest.main()
