from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from generate_billiards_scene import billiards_camera, simulate  # noqa: E402
from physical_proxy_catalog import load_catalog, records_by_id  # noqa: E402
from sample_one_object_scene_matrix import MATRIX_PATH, matrix_dependency_paths  # noqa: E402
from static_support_proxy import compile_static_support_binding  # noqa: E402


class BilliardsSceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
        cls.dependencies = matrix_dependency_paths(ROOT, matrix)
        registry = json.loads(
            cls.dependencies["asset_proxy_registry"].read_text(encoding="utf-8")
        )
        candidates = [
            record
            for record in registry["records"]
            if record["semantic_category"] == "support_game_table"
            and record["admission"].get("sampling_enabled", False)
        ]
        if len(candidates) != 1:
            raise AssertionError(f"expected one admitted game table, got {len(candidates)}")
        cls.support = candidates[0]
        _, proxy_records = load_catalog(
            ROOT, cls.dependencies["physical_proxy_catalog"]
        )
        cls.support_binding = compile_static_support_binding(
            records_by_id(proxy_records)[cls.support["asset_id"]],
            usage_id="curated_support",
        )
        cls.backend = json.loads(
            cls.dependencies["physics_backend"].read_text(encoding="utf-8")
        )
        cls.camera_views = json.loads(
            cls.dependencies["visual_sampling"].read_text(encoding="utf-8")
        )["specialized_camera_views"]

    @staticmethod
    def trajectory_digest(arrays: dict[str, np.ndarray]) -> str:
        digest = hashlib.sha256()
        for key in sorted(arrays):
            values = np.asarray(arrays[key])
            if np.issubdtype(values.dtype, np.floating):
                values = np.round(values.astype("<f8"), 8)
            else:
                values = values.astype(
                    values.dtype.newbyteorder("<"), copy=False
                )
            digest.update(key.encode("utf-8"))
            digest.update(str(values.shape).encode("ascii"))
            digest.update(str(values.dtype).encode("ascii"))
            digest.update(values.tobytes(order="C"))
        return digest.hexdigest()

    @classmethod
    def dependency_arguments(cls) -> list[str]:
        return [
            "--registry",
            str(cls.dependencies["asset_proxy_registry"]),
            "--catalog",
            str(cls.dependencies["physical_proxy_catalog"]),
            "--semantic-rules",
            str(cls.dependencies["asset_semantic_scene_rules"]),
            "--composition-rules",
            str(cls.dependencies["asset_scene_composition"]),
            "--backend",
            str(cls.dependencies["physics_backend"]),
            "--visual-rules",
            str(cls.dependencies["visual_sampling"]),
        ]

    def test_camera_is_seeded_bounded_and_reproducible(self) -> None:
        first = billiards_camera(123, "single_ball_free_roll")
        same = billiards_camera(123, "single_ball_free_roll")
        other = billiards_camera(124, "single_ball_free_roll")
        self.assertEqual(first, same)
        self.assertNotEqual(first, other)
        self.assertEqual(first["sensor_width_mm"], 36.0)
        rule = self.camera_views["single_ball_free_roll"]
        self.assertIn(
            first["mode"],
            {
                f"bounded_orbit_{int(value):+d}deg"
                for value in rule["yaw_offset_degrees"]
            },
        )
        self.assertIn(
            first["elevation_degrees"],
            {float(value) for value in rule["elevation_degrees"]},
        )
        self.assertGreaterEqual(first["position_m"][2], 2.0)
        self.assertLessEqual(first["position_m"][2], 3.2)
        self.assertLessEqual(abs(first["target_m"][0]), 0.08)
        self.assertLessEqual(abs(first["target_m"][1]), 0.06)

        self.assertEqual(
            first,
            {
                "seed": 123,
                "mode": "bounded_orbit_-15deg",
                "position_m": [1.840359, -3.851145, 2.580362],
                "target_m": [-0.030902, 0.01877, 0.68],
                "focal_length_mm": 50.0,
                "sensor_width_mm": 36.0,
                "elevation_degrees": 24.0,
            },
        )

    def test_single_ball_profiles_pass_physics_audit(self) -> None:
        for profile in (
            "single_ball_free_roll",
            "single_ball_rail_rebound",
        ):
            with self.subTest(profile=profile):
                arrays, audit, initial = simulate(
                    ROOT,
                    self.support_binding,
                    3.0,
                    24,
                    profile,
                    self.backend,
                )
                self.assertEqual(len(initial), 1)
                self.assertEqual(arrays["position_m"].shape, (73, 1, 3))
                self.assertTrue(audit["passed"], audit)

    def test_three_ball_profile_remains_supported(self) -> None:
        arrays, audit, initial = simulate(
            ROOT,
            self.support_binding,
            3.0,
            24,
            "three_ball_collision",
            self.backend,
        )
        self.assertEqual(len(initial), 3)
        self.assertEqual(arrays["position_m"].shape, (73, 3, 3))
        self.assertTrue(audit["passed"], audit)

    def test_published_profiles_keep_their_numeric_trajectory_contract(self) -> None:
        expected = {
            "single_ball_free_roll": (
                "3f97e5fea4b2798e24f4ab0b7d1e016a"
                "7c9e5632faaf354c5b1d88b42bcd9fca"
            ),
            "single_ball_rail_rebound": (
                "4bfae18ca5b357132e1e4085c5d005e7"
                "9254c7b2c7bbec4c68ab13c5bf175b8c"
            ),
            "three_ball_collision": (
                "f68d2f3a26cd4401635aada54b319e50"
                "2981e4e728b396b80366c31aeea07f7e"
            ),
        }
        for profile, expected_digest in expected.items():
            with self.subTest(profile=profile):
                arrays, audit, _ = simulate(
                    ROOT,
                    self.support_binding,
                    3.0,
                    24,
                    profile,
                    self.backend,
                )
                self.assertTrue(audit["passed"], audit)
                self.assertEqual(
                    self.trajectory_digest(arrays), expected_digest
                )

    def test_generator_writes_complete_visual_contract(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "datasets") as temporary:
            output = Path(temporary) / "billiards"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "generate_billiards_scene.py"),
                    "--root",
                    str(ROOT),
                    "--output",
                    str(output),
                    "--profile",
                    "single_ball_free_roll",
                    "--support-id",
                    self.support["asset_id"],
                    *self.dependency_arguments(),
                    "--duration",
                    "3.0",
                    "--fps",
                    "24",
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            metadata = json.loads(
                (output / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                metadata["schema_version"], "physweep_billiards_scene_v4"
            )
            self.assertIn("visual_rules", metadata)
            self.assertIn("lighting", metadata["render"]["environment"])
            self.assertEqual(
                metadata["assets"]["support_asset_id"],
                self.support["asset_id"],
            )
            self.assertEqual(
                metadata["physics"]["static_support_binding"]["asset_id"],
                self.support["asset_id"],
            )
            self.assertTrue((output / "simulation_record.json").is_file())

    def test_generator_rejects_an_unscheduled_support(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "datasets") as temporary:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "generate_billiards_scene.py"),
                    "--root",
                    str(ROOT),
                    "--output",
                    str(Path(temporary) / "billiards"),
                    "--profile",
                    "single_ball_free_roll",
                    "--support-id",
                    "not_an_admitted_support",
                    *self.dependency_arguments(),
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("requested='not_an_admitted_support'", completed.stdout)


if __name__ == "__main__":
    unittest.main()
