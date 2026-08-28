from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.core.hashing import sha256_file as sha256
from tools.release.audit_release_provenance import audit_release
from tools.release.one_object_source_release import publish_source_release


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class OneObjectSourceReleaseTests(unittest.TestCase):
    def test_fresh_release_binds_base_metadata_and_physics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_metadata = root / "datasets/run/base/metadata.json"
            write_json(base_metadata, {})
            base_manifest = root / "datasets/run/base/manifest.json"
            write_json(
                base_manifest,
                {
                    "dataset_id": "run",
                    "sample_count": 1,
                    "records": [
                        {
                            "scene_id": "base_scene",
                            "metadata_path": "datasets/run/base/metadata.json",
                            "metadata_sha256": sha256(base_metadata),
                        }
                    ],
                },
            )
            metadata_records = []
            physics_records = []
            descriptors = [(None, None, "base")]
            descriptors.extend(
                (axis, level, "sweep")
                for axis in (
                    "mass_kg",
                    "contact_friction",
                    "contact_restitution",
                )
                for level in (0, 1, 3, 4)
            )
            for index, (axis, level, kind) in enumerate(descriptors):
                scene_id = f"scene_{index}"
                metadata_path = root / f"datasets/run/sweep/{scene_id}.json"
                write_json(metadata_path, {})
                record = {
                    "scene_id": scene_id,
                    "path": str(metadata_path.relative_to(root)).replace("\\", "/"),
                    "metadata_sha256": sha256(metadata_path),
                    "parent": "datasets/run/base/metadata.json",
                    "kind": kind,
                    "axis": axis,
                    "level_index": level,
                    "target_object_id": "object_0" if kind == "sweep" else None,
                    "target_object_index": 0 if kind == "sweep" else None,
                    "source_schema_version": "test_schema",
                }
                metadata_records.append(record)
                physics_records.append(
                    {
                        "scene_id": scene_id,
                        "metadata_path": str(metadata_path.relative_to(root)).replace(
                            "\\", "/"
                        ),
                        "metadata_sha256": sha256(metadata_path),
                        "ok": True,
                        "audit_passed": True,
                        "failed_checks": [],
                        "source_schema_version": "test_schema",
                    }
                )
            metadata_manifest = root / "datasets/run/sweep/manifest.json"
            physics_manifest = root / "datasets/run/physics/manifest.json"
            write_json(
                metadata_manifest,
                {"sample_count": 13, "records": metadata_records},
            )
            write_json(
                physics_manifest,
                {"sample_count": 13, "records": physics_records},
            )
            output = root / "datasets/run/release"
            with patch(
                "tools.release.source_release.validate_object_identity",
                return_value={
                    "dynamic_object_count": 1,
                    "dynamic_object_ids": ["object_0"],
                },
            ), patch(
                "tools.release.source_release._validate_sweep_record_bindings"
            ), patch(
                "tools.release.source_release.validate_source_artifacts"
            ):
                release = publish_source_release(
                    root=root,
                    base_manifest_path=base_manifest,
                    sweep_metadata_manifest_path=metadata_manifest,
                    sweep_physics_manifest_path=physics_manifest,
                    output=output,
                )
            self.assertEqual(release["base_count"], 1)
            self.assertEqual(release["derived_count"], 12)
            self.assertTrue(audit_release(output / "manifest.json", root)["passed"])
            with self.assertRaises(FileExistsError):
                publish_source_release(
                    root=root,
                    base_manifest_path=base_manifest,
                    sweep_metadata_manifest_path=metadata_manifest,
                    sweep_physics_manifest_path=physics_manifest,
                    output=output,
                )


if __name__ == "__main__":
    unittest.main()
