from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.audit_release_provenance import sha256
from tools.base_release_schema import BASE_SAMPLE_SCHEMA, TRAJECTORY_FIELDS
from tools.build_base_release_view import PipelineSpec, build_view, verify_view


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class BaseReleaseViewTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "release view requires symlink support")
    def test_base_release_is_canonical_hash_checked_and_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_dir = root / "datasets/one_object_v5/release"
            specs = []
            metadata_records = []
            physics_records = []
            base_records = []
            videos = []
            audits = []
            for index, (name, schema, has_masks) in enumerate(
                (
                    ("generic", "schema_generic", True),
                    ("asset", "schema_asset", False),
                )
            ):
                project = root / f"source-{name}"
                scene_id = f"scene_{index}__base"
                group_id = f"logical_scene_{index}"
                source_metadata_path = f"source/base_{index}/metadata.json"
                metadata = project / "metadata" / f"{scene_id}.json"
                trajectory = project / "physics" / scene_id / "trajectory.npz"
                audit = project / "physics" / scene_id / "trajectory_audit.json"
                resolved = project / "physics" / scene_id / "resolved_scene.json"
                write_json(
                    metadata,
                    {
                        "schema_version": schema,
                        "scene_id": scene_id,
                        "seed": index + 1,
                        "semantics": {
                            "scene_family": name,
                            "profile": "drop",
                            "description": "One ball drops.",
                            "dynamic_object_count": 1,
                        },
                        "simulation": {
                            "objects": [
                                {
                                    "object_id": "ball",
                                    "visual": {
                                        "shape": "sphere",
                                        "radius_m": 0.1,
                                    },
                                }
                            ]
                        },
                        "render": {
                            "engine": "BLENDER_EEVEE",
                            "resolution": [1280, 720],
                            "samples": 16,
                        },
                        "object_identity": {
                            "objects": [
                                {
                                    "object_id": "ball",
                                    "role": "dynamic",
                                    "semantic_label": "ball",
                                    "asset_id": None,
                                    "mask_instance_id": 1,
                                }
                            ],
                            "text": {"caption": "The ball drops."},
                        },
                    },
                )
                trajectory.parent.mkdir(parents=True, exist_ok=True)
                time_s = np.asarray([0.0, 1.0], dtype=np.float64)
                position = np.zeros((2, 1, 3), dtype=np.float64)
                quaternion = np.zeros((2, 1, 4), dtype=np.float64)
                quaternion[:, :, 0] = 1.0
                np.savez_compressed(
                    trajectory,
                    schema_version=np.asarray("physweep_object_trajectory_v2"),
                    object_ids=np.asarray(["ball"]),
                    time_s=time_s,
                    position_m=position,
                    quaternion_wxyz=quaternion,
                    linear_velocity_m_s=np.zeros_like(position),
                    angular_velocity_rad_s=np.zeros_like(position),
                    contact_count=np.zeros((2, 1), dtype=np.int64),
                    runtime_material=np.asarray([[0.2, 0.3, 0.4]]),
                    inertia_diagonal_kg_m2=np.asarray([[0.001, 0.001, 0.001]]),
                    adapter__position_m=position,
                )
                write_json(audit, {"passed": True})
                audits.append(audit)
                write_json(
                    resolved,
                    {
                        "scene_id": scene_id,
                        "backend_binding": {
                            "backend_id": "pybullet_rigid",
                            "adapter_id": f"{name}_v1",
                        },
                        "time": {
                            "duration_s": 1.0,
                            "output_fps": 1,
                            "simulation_hz": 10,
                            "frame_count": 2,
                        },
                        "world": {"gravity_m_s2": [0.0, 0.0, -9.81]},
                        "objects": [
                            {
                                "object_id": "ball",
                                "object_index": 0,
                                "collision_proxy": {
                                    "type": "sphere",
                                    "radius_m": 0.1,
                                },
                                "material": {
                                    "mass_kg": 0.2,
                                    "contact_friction": 0.3,
                                    "contact_restitution": 0.4,
                                },
                                "initial_state": {
                                    "position_m": [0.0, 0.0, 1.0],
                                    "orientation_quaternion_xyzw": [
                                        0.0,
                                        0.0,
                                        0.0,
                                        1.0,
                                    ],
                                    "linear_velocity_m_s": [0.0, 0.0, 0.0],
                                    "angular_velocity_rad_s": [0.0, 0.0, 0.0],
                                },
                            }
                        ],
                    },
                )

                render_root = project / "render"
                frame_root = render_root / "frames" / scene_id
                frame_root.mkdir(parents=True)
                video = render_root / "videos" / f"{scene_id}.mp4"
                video.parent.mkdir(parents=True)
                video.write_bytes(f"video:{scene_id}".encode("utf-8"))
                videos.append(video)
                if has_masks:
                    mask = render_root / "masks" / scene_id / "ball"
                    mask.mkdir(parents=True)
                    (mask / "frame_0001.png").write_bytes(b"mask-1")
                    (mask / "frame_0002.png").write_bytes(b"mask-2")
                camera = {
                    "mode": "front",
                    "position_m": [2.0, -2.0, 1.5],
                    "target_m": [0.0, 0.0, 0.5],
                    "focal_length_mm": 50.0,
                    "sensor_width_mm": 36.0,
                }
                render_metadata = metadata
                render_record_camera = camera
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
                            "visualization": {"camera": camera},
                        },
                    )
                    render_record_camera = None
                render_record = {
                    "scene_id": scene_id,
                    "metadata_path": str(render_metadata),
                    "metadata_sha256": sha256(render_metadata),
                    "video_path": str(video),
                    "video_sha256": sha256(video),
                    "video_encoding": {
                        "codec": "H264",
                        "container": "MPEG4",
                        "fps": 1,
                    },
                }
                if render_record_camera is not None:
                    render_record["camera"] = render_record_camera
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
                    {"scene_id": group_id, "metadata_path": source_metadata_path}
                )
                specs.append(PipelineSpec(name, schema, project, render_root))

            metadata_records.append(
                {
                    "scene_id": "scene_0__sweep_ball_mass_kg_00",
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
                    "base_manifest": "datasets/one_object_v5/release/base_manifest.json",
                    "base_manifest_sha256": sha256(base_path),
                    "metadata_manifest": "datasets/one_object_v5/release/metadata_manifest.json",
                    "metadata_manifest_sha256": sha256(metadata_path),
                    "physics_manifest": "datasets/one_object_v5/release/physics_manifest.json",
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
            sample = output / "generic/scene_0__base"
            self.assertFalse((sample / "metadata.json").is_symlink())
            self.assertFalse((sample / "trajectory.npz").is_symlink())
            self.assertTrue((sample / "video.mp4").is_symlink())
            self.assertTrue((sample / "masks").is_symlink())
            self.assertEqual(
                {path.name for path in sample.iterdir()},
                {
                    "metadata.json",
                    "trajectory.npz",
                    "video.mp4",
                    "masks",
                    "mask_manifest.json",
                },
            )
            metadata = json.loads(
                (sample / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["schema_version"], BASE_SAMPLE_SCHEMA)
            self.assertEqual(metadata["group_id"], "logical_scene_0")
            with np.load(sample / "trajectory.npz", allow_pickle=False) as archive:
                self.assertEqual(tuple(archive.files), TRAJECTORY_FIELDS)
                self.assertNotIn("adapter__position_m", archive.files)
            root_manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("kind", root_manifest)
            for key in ("base_manifest", "metadata_manifest", "physics_manifest"):
                self.assertNotIn(f"{key}_sha256", root_manifest)
            self.assertEqual(
                set(root_manifest["pipelines"]["generic"]),
                {"manifest", "manifest_sha256"},
            )
            self.assertEqual(root_manifest["render_contract"]["resolution"], [1280, 720])
            generic_manifest = json.loads(
                (output / "generic/manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                set(generic_manifest["records"][0]),
                {"scene_id", "group_id", "metadata_sha256"},
            )
            self.assertEqual(verify_view(output), result)
            root_manifest["storage_mode"] = "unexpected"
            write_json(output / "manifest.json", root_manifest)
            with self.assertRaisesRegex(ValueError, "canonical PhysSweep"):
                verify_view(output)
            root_manifest["storage_mode"] = (
                "compact_metadata_with_absolute_artifact_symlinks"
            )
            write_json(output / "manifest.json", root_manifest)
            with self.assertRaises(FileExistsError):
                build_view(
                    release_project_root=root,
                    release_manifest=release_path,
                    output=output,
                    pipeline_specs=specs,
                )
            original_audit = audits[0].read_bytes()
            audits[0].write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "trajectory audit hash mismatch"):
                build_view(
                    release_project_root=root,
                    release_manifest=release_path,
                    output=output.parent / "tampered-audit-base",
                    pipeline_specs=specs,
                )
            audits[0].write_bytes(original_audit)
            videos[0].write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "video hash mismatch"):
                verify_view(output)


if __name__ == "__main__":
    unittest.main()
