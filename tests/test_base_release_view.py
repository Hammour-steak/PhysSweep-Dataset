from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.audit_release_provenance import sha256
from tools.build_base_release_view import (
    PipelineSpec,
    build_view,
    verify_view,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class BaseReleaseViewTests(unittest.TestCase):
    def test_base_view_is_classified_hash_checked_and_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_dir = root / "datasets/one_object_v5/release"
            project_a = root / "source-a"
            project_b = root / "source-b"
            specs = []
            metadata_records = []
            physics_records = []
            base_records = []
            videos = []
            for index, (name, schema, project, has_masks) in enumerate(
                (
                    ("generic", "schema_generic", project_a, True),
                    ("asset", "schema_asset", project_b, False),
                )
            ):
                scene_id = f"scene_{index}__base"
                logical_base_id = f"logical_scene_{index}"
                source_metadata_path = f"source/base_{index}/metadata.json"
                metadata = project / "metadata" / f"{scene_id}.json"
                trajectory = project / "physics" / scene_id / "trajectory.npz"
                audit = project / "physics" / scene_id / "trajectory_audit.json"
                resolved = project / "physics" / scene_id / "resolved_scene.json"
                metadata.parent.mkdir(parents=True, exist_ok=True)
                metadata.write_text(scene_id, encoding="utf-8")
                trajectory.parent.mkdir(parents=True, exist_ok=True)
                trajectory.write_bytes(scene_id.encode("utf-8"))
                write_json(audit, {"passed": True})
                write_json(resolved, {"scene_id": scene_id})

                render_root = project / "render"
                frame_root = render_root / "frames" / scene_id
                frame_root.mkdir(parents=True)
                inspection = []
                for frame in (1, 2, 3):
                    path = frame_root / f"frame_{frame:04d}.png"
                    path.write_bytes(f"{scene_id}:{frame}".encode("utf-8"))
                    inspection.append(str(path))
                video = render_root / "videos" / f"{scene_id}.mp4"
                video.parent.mkdir(parents=True)
                video.write_bytes(f"video:{scene_id}".encode("utf-8"))
                videos.append(video)
                if has_masks:
                    mask = render_root / "masks" / scene_id / "object_a"
                    mask.mkdir(parents=True)
                    (mask / "frame_0001.png").write_bytes(b"mask")
                render_metadata = metadata
                if index == 0:
                    render_metadata = render_root / "metadata" / f"{scene_id}.json"
                    write_json(
                        render_metadata,
                        {
                            "source_metadata": {
                                "path": str(metadata),
                                "sha256": sha256(metadata),
                            },
                            "trajectory": {
                                "path": str(trajectory),
                                "sha256": sha256(trajectory),
                            },
                        },
                    )
                render_record = {
                    "scene_id": scene_id,
                    "metadata_path": str(render_metadata),
                    "metadata_sha256": sha256(render_metadata),
                    "video_path": str(video),
                    "video_sha256": sha256(video),
                    "inspection_frames": inspection,
                }
                write_json(frame_root / "render_record.json", render_record)

                metadata_records.append(
                    {
                        "scene_id": scene_id,
                        "kind": "base",
                        "parent": source_metadata_path,
                        "source_schema_version": schema,
                        "metadata_sha256": sha256(metadata),
                    }
                )
                physics_records.append(
                    {
                        "scene_id": scene_id,
                        "ok": True,
                        "audit_passed": True,
                        "failed_checks": [],
                        "metadata_path": str(metadata),
                        "metadata_sha256": sha256(metadata),
                        "trajectory_path": str(trajectory),
                        "trajectory_sha256": sha256(trajectory),
                        "audit_path": str(audit),
                        "audit_sha256": sha256(audit),
                        "resolved_scene_path": str(resolved),
                        "resolved_scene_sha256": sha256(resolved),
                    }
                )
                base_records.append(
                    {
                        "scene_id": logical_base_id,
                        "metadata_path": source_metadata_path,
                    }
                )
                specs.append(PipelineSpec(name, schema, project, render_root))

            metadata_records.append(
                {
                    "scene_id": "scene_0__sweep_object_a_mass_kg_00",
                    "kind": "sweep",
                    "parent": "generated/base_0/metadata.json",
                    "source_schema_version": "schema_generic",
                    "metadata_sha256": "0" * 64,
                }
            )
            base_path = release_dir / "base_manifest.json"
            metadata_path = release_dir / "metadata_manifest.json"
            physics_path = release_dir / "physics_manifest.json"
            write_json(base_path, {"sample_count": 2, "records": base_records})
            write_json(
                metadata_path,
                {"sample_count": 3, "records": metadata_records, "sources": []},
            )
            write_json(
                physics_path,
                {"sample_count": 2, "records": physics_records, "sources": []},
            )
            release_path = release_dir / "manifest.json"
            write_json(
                release_path,
                {
                    "dataset_id": "test_release",
                    "base_count": 2,
                    "base_manifest": (
                        "datasets/one_object_v5/release/base_manifest.json"
                    ),
                    "base_manifest_sha256": sha256(base_path),
                    "metadata_manifest": (
                        "datasets/one_object_v5/release/metadata_manifest.json"
                    ),
                    "metadata_manifest_sha256": sha256(metadata_path),
                    "physics_manifest": (
                        "datasets/one_object_v5/release/physics_manifest.json"
                    ),
                    "physics_manifest_sha256": sha256(physics_path),
                },
            )

            output = root / "outputs/one_object_release_v5/base"
            result = build_view(
                release_project_root=root,
                release_manifest=release_path,
                output=output,
                pipeline_specs=specs,
            )
            self.assertEqual(result["sample_count"], 2)
            self.assertEqual(result["pipeline_count"], 2)
            self.assertEqual(result["mask_count"], 1)
            self.assertTrue(
                (output / "generic/scene_0__base/video.mp4").is_symlink()
            )
            self.assertTrue((output / "generic/scene_0__base/masks").is_symlink())
            self.assertTrue(
                (output / "generic/scene_0__base/render_metadata.json").is_symlink()
            )
            self.assertFalse((output / "asset/scene_1__base/masks").exists())
            self.assertFalse(
                (output / "asset/scene_1__base/render_metadata.json").exists()
            )
            generic_manifest = json.loads(
                (output / "generic/manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                generic_manifest["records"][0]["logical_base_id"],
                "logical_scene_0",
            )
            self.assertEqual(
                generic_manifest["records"][0]["source_metadata_path"],
                "source/base_0/metadata.json",
            )
            self.assertEqual(verify_view(output), result)
            with self.assertRaises(FileExistsError):
                build_view(
                    release_project_root=root,
                    release_manifest=release_path,
                    output=output,
                    pipeline_specs=specs,
                )
            videos[0].write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "video.mp4 hash mismatch"):
                verify_view(output)


if __name__ == "__main__":
    unittest.main()
