from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.generate_marble_run_candidate import materialize_collision_meshes
from tools.generate_marble_run_scene import (
    build_metadata,
    load_backend,
    marble_run_camera,
    simulate,
)
from tools.resolved_simulation_scene import compile_resolved_scene
from tools.specialized_backend_registry import specialized_by_pipeline


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/marble_run_backend.json"


class MarbleRunBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_promotion_binds_the_candidate_and_stays_single_object(self) -> None:
        admission = self.backend["admission"]
        self.assertEqual(admission["status"], "approved_specialized")
        self.assertIs(admission["release_enabled"], True)
        self.assertEqual(admission["dynamic_object_count"], 1)
        self.assertIs(admission["active_mechanisms_supported"], False)
        binding = self.backend["candidate_config"]
        path = ROOT / binding["path"]
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(), binding["sha256"]
        )

    def test_profiles_use_disjoint_audited_release_offsets(self) -> None:
        profile_offsets = {
            profile: [float(value) for value in rules["initial_track_offsets_m"]]
            for profile, rules in self.backend["profiles"].items()
        }
        self.assertEqual(
            set(profile_offsets), {"early_release_chain", "late_release_chain"}
        )
        flattened = [value for values in profile_offsets.values() for value in values]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertTrue(all(0.003 <= abs(value) <= 0.015 for value in flattened))

    def test_registry_declares_the_formal_renderer(self) -> None:
        record = specialized_by_pipeline(ROOT)["marble_run"]
        self.assertEqual(record["source_schema_version"], "physweep_marble_run_scene_v1")
        self.assertEqual(record["renderer_id"], "marble_run")
        self.assertEqual(record["sweep_branch"], "marble_run")

    def test_camera_is_seed_deterministic(self) -> None:
        candidate_path = ROOT / self.backend["candidate_config"]["path"]
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        arguments = (701, "early_release_chain", self.backend, candidate)
        self.assertEqual(marble_run_camera(*arguments), marble_run_camera(*arguments))

    def test_formal_metadata_compiles_and_simulates_deterministically(self) -> None:
        source_root = ROOT / "assets/library/github/jhpieper_marble_run"
        if not source_root.is_dir():
            self.skipTest("candidate source checkout is intentionally not stored in Git")
        try:
            import pybullet  # noqa: F401
            import trimesh  # noqa: F401
        except ImportError:
            self.skipTest("candidate runtime dependencies are unavailable")
        backend, candidate_path, candidate, source_paths = load_backend(
            ROOT, CONFIG_PATH
        )
        (ROOT / "outputs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temporary:
            output = Path(temporary)
            collision = materialize_collision_meshes(
                ROOT, output, candidate, source_paths
            )
            metadata = build_metadata(
                ROOT,
                output,
                CONFIG_PATH,
                backend,
                candidate_path,
                candidate,
                collision,
                701,
                "early_release_chain",
                "marble_run_contract_test",
            )
            self.assertEqual(metadata["simulation"]["time"]["frame_count"], 97)
            self.assertEqual(metadata["object_identity"]["object_order"], ["marble"])
            scene = compile_resolved_scene(metadata, ROOT)
            self.assertEqual(scene["backend_binding"]["adapter_id"], "marble_run_v1")
            first, first_audit = simulate(ROOT, metadata)
            second, second_audit = simulate(ROOT, metadata)
            self.assertTrue(first_audit["passed"], first_audit)
            self.assertEqual(first_audit, second_audit)
            self.assertEqual(first.keys(), second.keys())
            for field in first:
                np.testing.assert_array_equal(first[field], second[field])


if __name__ == "__main__":
    unittest.main()
