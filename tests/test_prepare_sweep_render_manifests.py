import json
import tempfile
import unittest
from pathlib import Path

from tools.prepare_sweep_render_manifests import (
    dispatched_paths,
    select_complete_groups,
    sha256,
)


class PrepareSweepRenderManifestTests(unittest.TestCase):
    def test_selects_only_complete_groups(self) -> None:
        records = [
            {"parent": "a", "scene_id": f"a_{index}"} for index in range(13)
        ] + [{"parent": "b", "scene_id": f"b_{index}"} for index in range(13)]
        selected = select_complete_groups(records, {"b"})
        self.assertEqual(len(selected), 13)
        self.assertEqual({record["parent"] for record in selected}, {"b"})

    def test_rejects_incomplete_groups(self) -> None:
        with self.assertRaises(ValueError):
            select_complete_groups([{"parent": "a", "scene_id": "a_0"}], {"a"})

    def test_dispatched_paths_verify_simulation_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            metadata = root / "datasets/metadata.json"
            physics_root = root / "datasets/physics/scene"
            trajectory = physics_root / "trajectory.npz"
            audit = physics_root / "trajectory_audit.json"
            simulation = physics_root / "simulation_record.json"
            metadata.parent.mkdir(parents=True)
            physics_root.mkdir(parents=True)
            metadata.write_text("metadata", encoding="utf-8")
            trajectory.write_bytes(b"trajectory")
            audit.write_text("audit", encoding="utf-8")
            record = {
                "metadata_path": str(metadata),
                "metadata_sha256": sha256(metadata),
                "trajectory_path": str(trajectory),
                "trajectory_sha256": sha256(trajectory),
                "audit_path": str(audit),
                "audit_sha256": sha256(audit),
            }
            simulation.write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(
                dispatched_paths(root, record)["trajectory_path"],
                str(trajectory.relative_to(root)),
            )
            trajectory.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "trajectory hash"):
                dispatched_paths(root, record)


if __name__ == "__main__":
    unittest.main()
