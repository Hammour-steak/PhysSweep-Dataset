#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from generate_marble_run_candidate import (  # noqa: E402
    build_metadata,
    derive_sweep_metadata,
    load_json,
    materialize_collision_meshes,
    project_path,
    simulate,
    validate_config,
    validate_one_factor_variants,
    write_json,
)


class MarbleRunCandidateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = ROOT / "configs/candidates/marble_run_v1.json"
        cls.config = load_json(cls.config_path)
        (ROOT / "outputs").mkdir(exist_ok=True)

    def test_candidate_is_not_release_enabled(self) -> None:
        admission = self.config["admission"]
        self.assertEqual(admission["status"], "candidate_only")
        self.assertIs(admission["release_enabled"], False)
        self.assertEqual(self.config["semantics"]["dynamic_object_count"], 1)
        self.assertEqual(self.config["semantics"]["dynamic_semantics"], ["marble"])

    def test_fixture_is_passive_and_explicit(self) -> None:
        fixture = self.config["fixture"]
        self.assertEqual(fixture["classification"], "static")
        self.assertEqual(
            fixture["representation"],
            "compound_exact_static_triangle_mesh_and_analytic_boxes",
        )
        self.assertEqual(len(fixture["mesh_components"]), 4)
        self.assertTrue(
            all(item["shape"] == "box" for item in fixture["analytic_colliders"])
        )
        ids = [
            item["id"]
            for item in fixture["mesh_components"] + fixture["analytic_colliders"]
        ]
        self.assertEqual(len(ids), len(set(ids)))
        backboard = self.config["render"]["context"]["backboard"]
        self.assertEqual(backboard["physics_role"], "render_only_context")
        self.assertEqual(
            self.config["render"]["fps"], self.config["physics"]["output_fps"]
        )

    def test_source_and_transform_contract(self) -> None:
        fixture = self.config["fixture"]
        matrix = np.asarray(fixture["source_to_world_rotation_matrix"], dtype=float)
        np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-9)
        self.assertAlmostEqual(float(np.linalg.det(matrix)), 1.0, places=9)
        x, y, z, w = fixture["source_to_world_orientation_quaternion_xyzw"]
        quaternion_matrix = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        np.testing.assert_allclose(matrix, quaternion_matrix, atol=1.0e-9)
        files = {item["path"]: item for item in self.config["source"]["files"]}
        topology = files["stl/straight-110.stl"]["topology"]
        self.assertEqual(topology["face_count"], 1880)
        self.assertTrue(topology["watertight_after_vertex_merge"])
        self.assertEqual(self.config["source"]["license"]["stl_objects"], "CC BY 4.0")

    def test_one_factor_contract_has_exactly_thirteen_records(self) -> None:
        source_root = project_path(ROOT, self.config["source"]["local_root"])
        if not source_root.is_dir():
            self.skipTest("candidate source checkout is intentionally not stored in Git")
        try:
            import trimesh  # noqa: F401
        except ImportError:
            self.skipTest("candidate mesh dependency is unavailable")
        source_paths = validate_config(ROOT, self.config)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temporary:
            output = Path(temporary)
            collision = materialize_collision_meshes(
                ROOT, output, self.config, source_paths
            )
            base = build_metadata(ROOT, self.config_path, self.config, collision)
            renderer = base["render"]["implementation"]
            self.assertEqual(renderer["path"], "tools/render_marble_run_candidate.py")
            self.assertEqual(len(renderer["sha256"]), 64)
            base_path = output / "metadata.json"
            write_json(base_path, base)
            variants = derive_sweep_metadata(
                base,
                ROOT,
                ROOT / "configs/physics_sweep.json",
                base_path,
            )
            validate_one_factor_variants(base, variants)
            self.assertEqual(len(variants), 13)
            self.assertEqual(
                [item["sweep"]["kind"] for item in variants].count("base"), 1
            )
            axes = [item["sweep"].get("axis") for item in variants[1:]]
            self.assertEqual(axes.count("mass_kg"), 4)
            self.assertEqual(axes.count("contact_friction"), 4)
            self.assertEqual(axes.count("contact_restitution"), 4)

    def test_runtime_is_deterministic_when_candidate_source_is_present(self) -> None:
        source_root = project_path(ROOT, self.config["source"]["local_root"])
        if not source_root.is_dir():
            self.skipTest("candidate source checkout is intentionally not stored in Git")
        try:
            import pybullet  # noqa: F401
            import trimesh  # noqa: F401
        except ImportError:
            self.skipTest("candidate runtime dependencies are unavailable")
        source_paths = validate_config(ROOT, self.config)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temporary:
            output = Path(temporary)
            collision = materialize_collision_meshes(
                ROOT, output, self.config, source_paths
            )
            metadata = build_metadata(
                ROOT, self.config_path, self.config, collision
            )
            first_arrays, first_audit = simulate(ROOT, metadata)
            second_arrays, second_audit = simulate(ROOT, metadata)
            self.assertTrue(first_audit["passed"], first_audit)
            self.assertEqual(first_audit, second_audit)
            self.assertEqual(first_arrays.keys(), second_arrays.keys())
            for key in first_arrays:
                np.testing.assert_array_equal(first_arrays[key], second_arrays[key])


if __name__ == "__main__":
    unittest.main()
