from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.core.hashing import sha256_file
from tools.dataset_contract.object_identity_contract import attach_object_identity
from tools.sampling.assemble_two_object_base import assemble_base_manifest


def metadata(scene_id: str, schema: str) -> dict:
    motion_family = (
        "surface_dual_independent_2obj"
        if schema == "physweep_pybullet_rigid_metadata_v1"
        else "two_ball_direct_collision"
    )
    document = {
        "scene_id": scene_id,
        "schema_version": schema,
        "semantics": {"motion": {"family": motion_family}},
        "simulation": {
            "objects": [
                {
                    "object_id": object_id,
                    "body_model": "rigid_body",
                    "semantic_type": "ball",
                }
                for object_id in ("object_a", "object_b")
            ]
        },
    }
    attach_object_identity(document)
    return document


class TwoObjectDatasetPipelineTests(unittest.TestCase):
    def test_assembler_merges_hash_bound_generic_and_specialized_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            generic_path = root / "generic" / "metadata.json"
            specialized_path = root / "billiards" / "metadata.json"
            generic_path.parent.mkdir(parents=True)
            specialized_path.parent.mkdir(parents=True)
            generic_path.write_text(
                json.dumps(
                    metadata("generic_scene", "physweep_pybullet_rigid_metadata_v1")
                ),
                encoding="utf-8",
            )
            specialized_path.write_text(
                json.dumps(
                    metadata("billiards_scene", "physweep_billiards_scene_v4")
                ),
                encoding="utf-8",
            )
            generic_manifest = root / "generic_manifest.json"
            specialized_manifest = root / "specialized_manifest.json"
            generic_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "physweep_pybullet_base_manifest_v1",
                        "dataset_id": "physweep_two_object",
                        "sample_count": 1,
                        "samples": [
                            {
                                "scene_id": "generic_scene",
                                "metadata_path": generic_path.relative_to(root).as_posix(),
                                "metadata_sha256": sha256_file(generic_path),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            specialized_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "physweep_two_object_specialized_base_manifest_v1"
                        ),
                        "dataset_id": "physweep_two_object",
                        "sample_count": 1,
                        "samples": [
                            {
                                "scene_id": "billiards_scene",
                                "family": "billiards",
                                "metadata_path": (
                                    specialized_path.relative_to(root).as_posix()
                                ),
                                "metadata_sha256": sha256_file(specialized_path),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = assemble_base_manifest(
                root, generic_manifest, specialized_manifest
            )

        self.assertEqual(result["object_count"], 2)
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["family_counts"]["generic"], 1)
        self.assertEqual(result["family_counts"]["billiards"], 1)
        self.assertEqual(
            [record["scene_id"] for record in result["records"]],
            ["billiards_scene", "generic_scene"],
        )

    def test_assembler_rejects_a_changed_source_metadata_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            generic_path = root / "generic.json"
            specialized_path = root / "specialized.json"
            generic_path.write_text(
                json.dumps(
                    metadata("generic_scene", "physweep_pybullet_rigid_metadata_v1")
                ),
                encoding="utf-8",
            )
            specialized_path.write_text(
                json.dumps(
                    metadata("billiards_scene", "physweep_billiards_scene_v4")
                ),
                encoding="utf-8",
            )
            generic_manifest = root / "generic_manifest.json"
            specialized_manifest = root / "specialized_manifest.json"
            generic_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "physweep_pybullet_base_manifest_v1",
                        "dataset_id": "physweep_two_object",
                        "sample_count": 1,
                        "samples": [
                            {
                                "scene_id": "generic_scene",
                                "metadata_path": "generic.json",
                                "metadata_sha256": "0" * 64,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            specialized_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "physweep_two_object_specialized_base_manifest_v1"
                        ),
                        "dataset_id": "physweep_two_object",
                        "sample_count": 1,
                        "samples": [
                            {
                                "scene_id": "billiards_scene",
                                "family": "billiards",
                                "metadata_path": "specialized.json",
                                "metadata_sha256": sha256_file(specialized_path),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "metadata hash differs"):
                assemble_base_manifest(
                    root, generic_manifest, specialized_manifest
                )


if __name__ == "__main__":
    unittest.main()
