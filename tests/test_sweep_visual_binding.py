from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from bind_physics_sweep_visuals import bind_one, sha256, validated_sweep_samples


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


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

    def test_rejects_duplicate_scene_ids(self) -> None:
        manifest = self.manifest()
        manifest["records"].append(dict(manifest["records"][0]))
        manifest["sample_count"] = 2
        manifest["passed_count"] = 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validated_sweep_samples(manifest)

    def test_staged_binding_publishes_only_final_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sweep_path = root / "datasets/sweep.json"
            trajectory_path = root / "datasets/trajectory.npz"
            audit_path = root / "datasets/audit.json"
            simulation_path = root / "datasets/simulation_record.json"
            write_json(
                sweep_path,
                {
                    "scene_id": "derived",
                    "sweep": {"kind": "sweep", "parent_scene_id": "parent"},
                },
            )
            trajectory_path.write_bytes(b"trajectory")
            write_json(audit_path, {"passed": True})
            write_json(
                simulation_path,
                {
                    "scene_id": "derived",
                    "metadata_path": str(sweep_path),
                    "metadata_sha256": sha256(sweep_path),
                    "trajectory_sha256": sha256(trajectory_path),
                    "audit_sha256": sha256(audit_path),
                },
            )
            parent_path = root / "outputs/base/metadata/parent.json"
            write_json(
                parent_path,
                {
                    "schema_version": "physweep_pybullet_rigid_bound_metadata_v1",
                    "visualization": {"render": {}},
                },
            )
            staging = root / "outputs/.bound.tmp"
            published = root / "outputs/final"
            record = bind_one(
                root,
                {
                    "scene_id": "derived",
                    "metadata_path": str(sweep_path),
                    "trajectory_path": str(trajectory_path),
                    "audit_path": str(audit_path),
                    "simulation_record_path": str(simulation_path),
                },
                {"parent": {"metadata_path": str(parent_path)}},
                staging,
                published,
            )
            bound = json.loads(
                (staging / "metadata/derived.json").read_text(encoding="utf-8")
            )
            self.assertEqual(record["metadata_path"], "outputs/final/metadata/derived.json")
            self.assertEqual(
                bound["visualization"]["render"]["video_path"],
                "outputs/final/videos/derived.mp4",
            )
            self.assertNotIn(".bound.tmp", json.dumps(bound))


if __name__ == "__main__":
    unittest.main()
