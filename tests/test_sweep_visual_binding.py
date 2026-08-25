from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from bind_physics_sweep_visuals import validated_sweep_samples


class SweepVisualBindingTest(unittest.TestCase):
    def manifest(self) -> dict:
        return {
            "schema_version": "physweep_pybullet_batch_record_v1",
            "sample_count": 1,
            "passed_count": 1,
            "rejected_count": 0,
            "error_count": 0,
            "records": [
                {
                    "scene_id": "sample_001",
                    "metadata_path": "/project/sample_001/metadata.json",
                    "trajectory_path": "/project/physics/sample_001/trajectory.npz",
                    "audit_path": "/project/physics/sample_001/trajectory_audit.json",
                    "ok": True,
                    "audit_passed": True,
                }
            ],
        }

    def test_accepts_fully_audited_batch(self) -> None:
        self.assertEqual(
            validated_sweep_samples(self.manifest()),
            [
                {
                    "scene_id": "sample_001",
                    "metadata_path": "/project/sample_001/metadata.json",
                    "trajectory_path": "/project/physics/sample_001/trajectory.npz",
                    "audit_path": "/project/physics/sample_001/trajectory_audit.json",
                    "simulation_record_path": "/project/physics/sample_001/simulation_record.json",
                }
            ],
        )

    def test_rejects_failed_batch(self) -> None:
        manifest = self.manifest()
        manifest["records"][0]["audit_passed"] = False
        manifest["passed_count"] = 0
        manifest["rejected_count"] = 1
        with self.assertRaises(ValueError):
            validated_sweep_samples(manifest)

    def test_rejects_derived_manifest_before_simulation(self) -> None:
        manifest = self.manifest()
        manifest["schema_version"] = "physweep_physics_sweep_manifest_v1"
        with self.assertRaises(ValueError):
            validated_sweep_samples(manifest)


if __name__ == "__main__":
    unittest.main()
