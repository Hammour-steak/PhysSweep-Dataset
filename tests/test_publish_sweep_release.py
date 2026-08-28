from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]

from tools.release.publish_sweep_release import (  # noqa: E402
    main,
    merge_base_records,
    merge_generic_samples,
    select_source_records,
    sha256,
    validate_groups,
    validate_source_artifacts,
)


class SweepReleaseTests(unittest.TestCase):
    def test_main_publishes_complete_release_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            records = [{
                "parent": "base.json",
                "kind": "base",
                "axis": None,
                "scene_id": "base",
                "source_schema_version": "schema",
            }]
            for axis in ("mass_kg", "contact_friction", "contact_restitution"):
                records.extend(
                    {
                        "parent": "base.json",
                        "kind": "sweep",
                        "axis": axis,
                        "level_index": level,
                        "target_object_id": "object_a",
                        "target_object_index": 0,
                        "scene_id": f"{axis}_{level}",
                        "source_schema_version": "schema",
                    }
                    for level in (0, 1, 3, 4)
                )
            physics_records = []
            for record in records:
                scene_id = record["scene_id"]
                scene_root = root / "datasets/source" / scene_id
                metadata = scene_root / "metadata.json"
                resolved = scene_root / "resolved.json"
                trajectory = scene_root / "trajectory.npz"
                audit = scene_root / "audit.json"
                scene_root.mkdir(parents=True)
                for path, value in (
                    (metadata, "metadata"),
                    (resolved, "resolved"),
                    (trajectory, "trajectory"),
                    (audit, "audit"),
                ):
                    path.write_text(value, encoding="utf-8")
                record["path"] = str(metadata.relative_to(root))
                record["metadata_sha256"] = sha256(metadata)
                physics_records.append({
                    "scene_id": scene_id,
                    "source_schema_version": "schema",
                    "ok": True,
                    "audit_passed": True,
                    "failed_checks": [],
                    "metadata_path": str(metadata),
                    "metadata_sha256": sha256(metadata),
                    "resolved_scene_path": str(resolved),
                    "resolved_scene_sha256": sha256(resolved),
                    "trajectory_path": str(trajectory),
                    "trajectory_sha256": sha256(trajectory),
                    "audit_path": str(audit),
                    "audit_sha256": sha256(audit),
                })
            metadata_manifest = root / "datasets/source/metadata_manifest.json"
            physics_manifest = root / "datasets/source/physics_manifest.json"
            metadata_manifest.write_text(
                json.dumps({"sample_count": 13, "records": records}), encoding="utf-8"
            )
            physics_manifest.write_text(
                json.dumps({"sample_count": 13, "records": physics_records}),
                encoding="utf-8",
            )
            output = root / "datasets/release"
            argv = [
                "publish_sweep_release.py",
                "--root", str(root),
                "--metadata-manifest", str(metadata_manifest),
                "--physics-manifest", str(physics_manifest),
                "--output-dir", str(output),
            ]
            with patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
                main()
            release = json.loads((output / "manifest.json").read_text())
            self.assertEqual(release["sample_count"], 13)
            self.assertEqual(
                release["metadata_manifest"], "datasets/release/metadata_manifest.json"
            )
            self.assertFalse(any(output.parent.glob(".release.*")))

    def test_validate_complete_group(self) -> None:
        records = [{"parent": "base.json", "kind": "base", "axis": None}]
        for axis in ("mass_kg", "contact_friction", "contact_restitution"):
            records.extend(
                {
                    "parent": "base.json",
                    "kind": "sweep",
                    "axis": axis,
                    "level_index": level,
                    "target_object_id": "object_a",
                    "target_object_index": 0,
                }
                for level in (0, 1, 3, 4)
            )
        self.assertEqual(validate_groups(records), 1)

    def test_reject_duplicate_sweep_level(self) -> None:
        records = [{"parent": "base.json", "kind": "base", "axis": None}]
        for axis in ("mass_kg", "contact_friction", "contact_restitution"):
            levels = (0, 1, 3, 3) if axis == "mass_kg" else (0, 1, 3, 4)
            records.extend(
                {
                    "parent": "base.json",
                    "kind": "sweep",
                    "axis": axis,
                    "level_index": level,
                    "target_object_id": "object_a",
                    "target_object_index": 0,
                }
                for level in levels
            )
        with self.assertRaisesRegex(ValueError, "invalid sweep levels"):
            validate_groups(records)

    def test_reject_incomplete_group(self) -> None:
        with self.assertRaisesRegex(ValueError, "13 records"):
            validate_groups([{"parent": "base.json", "kind": "base"}])

    def test_merge_base_records_replaces_exact_index(self) -> None:
        base = [
            {"index": 1, "metadata_path": "old-1.json", "pipeline": "generic"},
            {"index": 2, "metadata_path": "old-2.json", "pipeline": "generic"},
        ]
        replacement = {
            "index": 2,
            "metadata_path": "new-2.json",
            "replaces_metadata_path": "old-2.json",
            "pipeline": "generic",
        }
        self.assertEqual(
            merge_base_records(base, [replacement]),
            [base[0], replacement],
        )

    def test_select_source_records_excludes_replaced_parent(self) -> None:
        metadata = {
            "records": [
                {"scene_id": "old", "parent": "old-base.json"},
                {"scene_id": "keep", "parent": "keep-base.json"},
            ]
        }
        physics = {
            "records": [
                {"scene_id": "old", "ok": True, "audit_passed": True},
                {"scene_id": "keep", "ok": True, "audit_passed": True},
            ]
        }
        selected_metadata, selected_physics = select_source_records(
            metadata, physics, {"old-base.json"}
        )
        self.assertEqual([record["scene_id"] for record in selected_metadata], ["keep"])
        self.assertEqual([record["scene_id"] for record in selected_physics], ["keep"])

    def test_merge_generic_samples_preserves_slot_count(self) -> None:
        source = [
            {"scene_id": "old", "metadata_path": "old.json"},
            {"scene_id": "keep", "metadata_path": "keep.json"},
        ]
        replacement = {
            "replaces_metadata_path": "old.json",
            "candidate_scene_id": "new",
            "metadata_path": "new.json",
            "metadata_sha256": "metadata-hash",
            "simulation_record_path": "simulation.json",
            "trajectory_path": "trajectory.npz",
        }
        merged = merge_generic_samples(source, [replacement])
        self.assertEqual(len(merged), 2)
        self.assertEqual({sample["scene_id"] for sample in merged}, {"keep", "new"})

    def test_validate_source_artifacts_checks_declared_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {
                name: root / f"datasets/{name}.bin"
                for name in ("metadata", "resolved", "trajectory", "audit")
            }
            for name, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode("ascii"))
            metadata = [{
                "scene_id": "scene",
                "path": str(paths["metadata"].relative_to(root)),
                "metadata_sha256": sha256(paths["metadata"]),
                "source_schema_version": "schema",
            }]
            physics = [{
                "scene_id": "scene",
                "metadata_path": str(paths["metadata"]),
                "metadata_sha256": sha256(paths["metadata"]),
                "source_schema_version": "schema",
                "resolved_scene_path": str(paths["resolved"]),
                "resolved_scene_sha256": sha256(paths["resolved"]),
                "trajectory_path": str(paths["trajectory"]),
                "trajectory_sha256": sha256(paths["trajectory"]),
                "audit_path": str(paths["audit"]),
                "audit_sha256": sha256(paths["audit"]),
            }]
            validate_source_artifacts(root, metadata, physics)
            paths["trajectory"].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
                validate_source_artifacts(root, metadata, physics)


if __name__ == "__main__":
    unittest.main()
