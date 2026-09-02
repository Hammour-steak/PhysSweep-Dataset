from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from tools.rendering.render_asset_proxy_manifest import (  # noqa: E402
    MASK_REQUIRED_SCHEMAS,
    expected_instance_mask_directory,
    implementation_is_reusable,
    instance_masks_are_reusable,
    mask_record_is_reusable,
    output_path,
    render_samples_are_reusable,
    result_manifest_path,
    render_source_records,
    reusable_render_record as reusable_asset_record,
    sha256,
)
from tools.rendering.render_pybullet_manifest import (  # noqa: E402
    reusable_render_record as reusable_generic_record,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class RenderManifestResumeTests(unittest.TestCase):
    def test_two_object_specialized_renderer_accepts_all_three_fixture_schemas(self) -> None:
        renderers = {
            "two_object_specialized": (
                "tools/rendering/render_two_object_specialized_scene.py",
                "physweep_two_object_specialized_render_manifest_v1",
                "render_manifest.json",
                "two_object_specialized",
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = []
            for index, schema in enumerate(
                (
                    "physweep_billiards_scene_v4",
                    "physweep_passive_pinball_scene_v1",
                    "physweep_marble_run_scene_v1",
                )
            ):
                scene_id = f"scene_{index}"
                path = root / f"{scene_id}.json"
                write_json(
                    path,
                    {
                        "schema_version": schema,
                        "scene_id": scene_id,
                        "semantics": {"dynamic_object_count": 2},
                    },
                )
                samples.append(
                    {"scene_id": scene_id, "metadata_path": path.name}
                )
            records = render_source_records(
                root,
                {"samples": samples},
                "two_object_specialized",
                renderers,
            )
        self.assertEqual([record["scene_id"] for record in records], [
            "scene_0",
            "scene_1",
            "scene_2",
        ])

    def test_every_specialized_renderer_requires_materialized_masks(self) -> None:
        self.assertEqual(
            MASK_REQUIRED_SCHEMAS,
            {
                "physweep_asset_proxy_scene_v3",
                "physweep_billiards_scene_v4",
                "physweep_passive_pinball_scene_v1",
                "physweep_marble_run_scene_v1",
            },
        )

    def test_render_override_moves_asset_masks_with_video_and_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs/run/asset"
            source_record = {
                "scene_id": "scene",
                "render_output": {
                    "video_path": "outputs/run/asset/videos/scene.mp4",
                    "inspection_frame_dir": "outputs/run/asset/frames/scene",
                },
            }
            mask_contract = {
                "path": "datasets/source/masks/scene",
            }
            self.assertEqual(
                expected_instance_mask_directory(
                    root,
                    output,
                    source_record,
                    mask_contract,
                ),
                (output / "masks/scene").resolve(),
            )

    def test_only_sample_bound_specialized_schemas_require_render_samples(self) -> None:
        record: dict[str, object] = {}
        billiards = {
            "schema_version": "physweep_billiards_scene_v4",
            "render": {"samples": 16},
        }
        self.assertTrue(render_samples_are_reusable(billiards, record))
        for schema in (
            "physweep_passive_pinball_scene_v1",
            "physweep_marble_run_scene_v1",
        ):
            metadata = {"schema_version": schema, "render": {"samples": 16}}
            self.assertFalse(render_samples_are_reusable(metadata, record))
            record["render_samples"] = 16
            self.assertTrue(render_samples_are_reusable(metadata, record))
            record["render_samples"] = 8
            self.assertFalse(render_samples_are_reusable(metadata, record))
            record.clear()

        strict = {
            "schema_version": "physweep_billiards_scene_v4",
            "render": {
                "samples": 16,
                "evidence_contract": "physweep_specialized_render_evidence_v2",
            },
        }
        self.assertFalse(render_samples_are_reusable(strict, record))
        record["render_samples"] = 16
        self.assertTrue(render_samples_are_reusable(strict, record))

    def test_specialized_mask_reuse_requires_matching_scene_and_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask_root = root / "outputs/run/masks/scene"
            mask_dir = mask_root / "object"
            masks = [mask_dir / f"frame_{index:04d}.png" for index in (1, 2)]
            for mask in masks:
                mask.parent.mkdir(parents=True, exist_ok=True)
                mask.write_bytes(b"mask")
            manifest = {
                "schema_version": "physweep_instance_mask_manifest_v1",
                "scene_id": "scene",
                "object_id": "object",
                "frame_count": 2,
                "records": [
                    {"filename": mask.name, "sha256": sha256(mask)}
                    for mask in masks
                ],
            }
            manifest_path = mask_root / "mask_manifest.json"
            write_json(manifest_path, manifest)
            render_record = {
                "scene_id": "scene",
                "instance_mask_output": {
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": sha256(manifest_path),
                    "objects": {"object": {"directory": str(mask_dir)}},
                },
            }
            self.assertTrue(
                instance_masks_are_reusable(root, render_record, 2)
            )

            manifest["scene_id"] = "different_scene"
            write_json(manifest_path, manifest)
            render_record["instance_mask_output"]["manifest_sha256"] = sha256(
                manifest_path
            )
            self.assertFalse(
                instance_masks_are_reusable(root, render_record, 2)
            )

    def test_v2_mask_reuse_covers_every_identity_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask_root = root / "outputs/run/masks/scene"
            objects = {}
            manifest_objects = {}
            for instance_id, object_id in enumerate(("cue_ball", "object_ball"), 1):
                mask_dir = mask_root / object_id
                paths = [mask_dir / f"frame_{index:04d}.png" for index in (1, 2)]
                for path in paths:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(object_id.encode("utf-8"))
                objects[object_id] = {
                    "instance_id": instance_id,
                }
                manifest_objects[object_id] = {
                    **objects[object_id],
                    "validation": {
                        "initial_occupancy_fraction": 0.01,
                        "initial_soft_edge_fraction": 0.001,
                    },
                    "records": [
                        {"filename": path.name, "sha256": sha256(path)}
                        for path in paths
                    ],
                }
            manifest = {
                "schema_version": "physweep_instance_mask_manifest_v2",
                "scene_id": "scene",
                "frame_count": 2,
                "objects": manifest_objects,
            }
            manifest_path = mask_root / "mask_manifest.json"
            write_json(manifest_path, manifest)
            record = {
                "scene_id": "scene",
                "instance_mask_output": {
                    "render_samples": 8,
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": sha256(manifest_path),
                },
            }
            self.assertTrue(
                instance_masks_are_reusable(
                    root,
                    record,
                    2,
                    required=True,
                    expected_objects={
                        object_id: {"instance_id": value["instance_id"]}
                        for object_id, value in objects.items()
                    },
                    expected_directory=mask_root,
                    expected_render_samples=8,
                )
            )
            del manifest["objects"]["object_ball"]
            write_json(manifest_path, manifest)
            record["instance_mask_output"]["manifest_sha256"] = sha256(
                manifest_path
            )
            self.assertFalse(
                instance_masks_are_reusable(
                    root,
                    record,
                    2,
                    required=True,
                    expected_objects={
                        "cue_ball": {"instance_id": 1},
                        "object_ball": {"instance_id": 2},
                    },
                    expected_directory=mask_root,
                    expected_render_samples=8,
                )
            )
            self.assertFalse(instance_masks_are_reusable(root, {}, 2, required=True))

    def test_v2_evidence_binds_the_exact_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "tools/renderer.py"
            evidence = root / "tools/rendering/specialized_render_evidence.py"
            evidence.parent.mkdir(parents=True)
            script.write_text("renderer", encoding="utf-8")
            evidence.write_text("evidence", encoding="utf-8")
            metadata = {
                "render": {
                    "evidence_contract": "physweep_specialized_render_evidence_v2"
                },
                "implementation": {
                    "renderer": {
                        "path": str(script),
                        "sha256": sha256(script),
                    },
                    "render_evidence": {
                        "path": str(evidence),
                        "sha256": sha256(evidence),
                    },
                },
            }
            self.assertTrue(implementation_is_reusable(root, metadata, script))
            metadata["implementation"]["render_evidence"]["sha256"] = "0" * 64
            self.assertFalse(implementation_is_reusable(root, metadata, script))

    def test_mask_only_resume_binds_metadata_objects_code_and_egl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / "tools/rendering/render_asset_proxy_scene.py"
            evidence = root / "tools/rendering/specialized_render_evidence.py"
            script.parent.mkdir(parents=True)
            script.write_text("renderer", encoding="utf-8")
            evidence.write_text("evidence", encoding="utf-8")
            metadata_path = root / "datasets/scene.json"
            metadata = {
                "scene_id": "scene",
                "physics": {"frame_count": 2},
                "render": {"resolution": [320, 180], "samples": 16},
                "object_identity": {
                    "objects": [{"object_id": "object", "role": "dynamic"}],
                    "instance_masks": {
                        "objects": {"object": {"instance_id": 1}}
                    },
                },
            }
            write_json(metadata_path, metadata)
            mask_root = root / "outputs/run/masks/scene"
            object_root = mask_root / "object"
            paths = [object_root / f"frame_{frame:04d}.png" for frame in (1, 2)]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(path.name.encode("utf-8"))
            mask_manifest = {
                "schema_version": "physweep_instance_mask_manifest_v2",
                "scene_id": "scene",
                "frame_count": 2,
                "objects": {
                    "object": {
                        "instance_id": 1,
                        "validation": {
                            "initial_occupancy_fraction": 0.01,
                            "initial_soft_edge_fraction": 0.001,
                        },
                        "records": [
                            {"filename": path.name, "sha256": sha256(path)}
                            for path in paths
                        ],
                    }
                },
            }
            mask_manifest_path = mask_root / "mask_manifest.json"
            write_json(mask_manifest_path, mask_manifest)
            render_record = {
                "schema_version": "physweep_specialized_mask_render_record_v1",
                "scene_id": "scene",
                "metadata_path": str(metadata_path),
                "metadata_sha256": sha256(metadata_path),
                "render_scope": "instance_masks_only",
                "mask_resolution": [320, 180],
                "egl_device_verified": True,
                "instance_mask_output": {
                    "render_samples": 16,
                    "manifest_path": str(mask_manifest_path),
                    "manifest_sha256": sha256(mask_manifest_path),
                },
                "implementation": {
                    "renderer": {
                        "path": str(script),
                        "sha256": sha256(script),
                    },
                    "render_evidence": {
                        "path": str(evidence),
                        "sha256": sha256(evidence),
                    },
                },
            }
            self.assertTrue(
                mask_record_is_reusable(
                    root,
                    metadata_path,
                    metadata,
                    render_record,
                    mask_root,
                    script,
                )
            )
            render_record["egl_device_verified"] = False
            self.assertFalse(
                mask_record_is_reusable(
                    root,
                    metadata_path,
                    metadata,
                    render_record,
                    mask_root,
                    script,
                )
            )
            render_record["egl_device_verified"] = True
            render_record["implementation"]["renderer"]["sha256"] = "0" * 64
            self.assertFalse(
                mask_record_is_reusable(
                    root,
                    metadata_path,
                    metadata,
                    render_record,
                    mask_root,
                    script,
                )
            )

    def test_renderer_rejects_the_wrong_scene_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "datasets/scene.json"
            write_json(
                metadata,
                {"schema_version": "wrong", "scene_id": "scene"},
            )
            with self.assertRaisesRegex(ValueError, "wrong scene schema"):
                render_source_records(
                    root,
                    {"records": [{"metadata_path": str(metadata)}]},
                    "asset",
                    {"asset": ("", "", "", "physweep_asset_proxy_scene_v3")},
                )

    def test_renderer_rejects_a_scene_id_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "datasets/scene.json"
            write_json(
                metadata,
                {"schema_version": "schema", "scene_id": "../scene"},
            )
            with self.assertRaisesRegex(ValueError, "invalid scene id"):
                render_source_records(
                    root,
                    {"records": [{"metadata_path": str(metadata)}]},
                    "asset",
                    {"asset": ("", "", "", "schema")},
                )

    def test_output_path_rejects_non_output_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "root/outputs"):
                output_path(root, "datasets/formal/video.mp4")

    def test_result_manifest_remains_below_its_render_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs/run/branch"
            self.assertEqual(
                result_manifest_path(root, output, None, "render_manifest.json"),
                output / "render_manifest.json",
            )
            with self.assertRaisesRegex(ValueError, "below its output root"):
                result_manifest_path(
                    root,
                    output,
                    root / "outputs/run/other/render_manifest.json",
                    "render_manifest.json",
                )

    @patch(
        "tools.rendering.render_asset_proxy_manifest.video_has_expected_frame_count",
        return_value=True,
    )
    def test_asset_record_requires_verified_egl_provenance(
        self,
        _video_frame_count: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs/run/asset"
            frame_dir = output / "frames/scene"
            video = output / "videos/scene.mp4"
            source = root / "datasets/source.json"
            metadata_path = output / "metadata/scene.json"
            source.parent.mkdir(parents=True)
            source.write_text("source", encoding="utf-8")
            metadata = {
                "scene_id": "scene",
                "physics": {"frame_count": 3},
                "source_metadata": {
                    "path": str(source.relative_to(root)),
                    "sha256": sha256(source),
                },
            }
            write_json(metadata_path, metadata)
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            frames = [frame_dir / f"frame_{index:04d}.png" for index in range(1, 4)]
            for frame in frames:
                frame.parent.mkdir(parents=True, exist_ok=True)
                frame.write_bytes(b"frame")
            record = {
                "scene_id": "scene",
                "metadata_path": str(metadata_path),
                "metadata_sha256": sha256(source),
                "video_path": str(video),
                "video_sha256": sha256(video),
                "inspection_frames": [str(frame) for frame in frames],
            }
            source_record = {"scene_id": "scene"}
            self.assertFalse(
                reusable_asset_record(
                    root,
                    output,
                    source_record,
                    metadata_path,
                    metadata,
                    frame_dir,
                    video,
                    record,
                    3,
                )
            )
            log = output / "logs/scene.log"
            log.parent.mkdir(parents=True)
            log.write_text(
                "PhysSweep EGL selector: CUDA device 3 matched EGL index 0",
                encoding="utf-8",
            )
            self.assertFalse(
                reusable_asset_record(
                    root,
                    output,
                    source_record,
                    metadata_path,
                    metadata,
                    frame_dir,
                    video,
                    record,
                    3,
                )
            )
            record["metadata_sha256"] = sha256(metadata_path)
            self.assertTrue(
                reusable_asset_record(
                    root,
                    output,
                    source_record,
                    metadata_path,
                    metadata,
                    frame_dir,
                    video,
                    record,
                    3,
                )
            )

    @patch(
        "tools.rendering.render_pybullet_manifest.video_has_expected_frame_count",
        return_value=True,
    )
    def test_generic_record_requires_complete_masks_and_hashes(
        self,
        _video_frame_count: object,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs/run/generic"
            metadata_path = output / "metadata/scene.json"
            trajectory = output / "physics/scene.npz"
            video = output / "videos/scene.mp4"
            frame_dir = output / "frames/scene"
            mask_dir = output / "masks/scene/object"
            script = root / "tools/rendering/render_pybullet_rigid.py"
            script.parent.mkdir(parents=True)
            script.write_text("renderer", encoding="utf-8")
            trajectory.parent.mkdir(parents=True)
            trajectory.write_bytes(b"trajectory")
            video.parent.mkdir(parents=True)
            video.write_bytes(b"video")
            frames = [frame_dir / f"frame_{index:04d}.png" for index in (1, 2, 3)]
            for frame in frames:
                frame.parent.mkdir(parents=True, exist_ok=True)
                frame.write_bytes(b"frame")
            masks = [mask_dir / f"frame_{index:04d}.png" for index in (1, 2)]
            for mask in masks:
                mask.parent.mkdir(parents=True, exist_ok=True)
                mask.write_bytes(b"mask")
            metadata = {
                "trajectory": {
                    "path": str(trajectory.relative_to(root)),
                    "sha256": sha256(trajectory),
                },
                "visualization": {
                    "render": {
                        "frame_start": 1,
                        "frame_end": 2,
                        "video_path": str(video.relative_to(root)),
                        "inspection_frame_dir": str(frame_dir.relative_to(root)),
                        "inspection_frames": [1, 2, 3],
                    }
                },
            }
            write_json(metadata_path, metadata)
            sample = {
                "scene_id": "scene",
                "metadata_sha256": sha256(metadata_path),
            }
            record = {
                "implementation": {
                    "path": str(script),
                    "sha256": sha256(script),
                },
                "scene_id": "scene",
                "render_scope": "full_animation",
                "metadata_path": str(metadata_path),
                "metadata_sha256": sha256(metadata_path),
                "trajectory_path": str(trajectory),
                "trajectory_sha256": sha256(trajectory),
                "video_path": str(video),
                "video_sha256": sha256(video),
                "inspection_frames": [str(frame) for frame in frames],
                "egl_device_verified": True,
                "instance_mask_output": {
                    "objects": {"object": {"directory": str(mask_dir)}},
                    "validation": {"objects": {"object": {"frame_count": 2}}},
                },
            }
            self.assertTrue(
                reusable_generic_record(
                    root,
                    output,
                    sample,
                    metadata_path,
                    metadata,
                    record,
                    False,
                    3,
                    script,
                )
            )
            script.write_text("changed renderer", encoding="utf-8")
            self.assertFalse(
                reusable_generic_record(
                    root,
                    output,
                    sample,
                    metadata_path,
                    metadata,
                    record,
                    False,
                    3,
                    script,
                )
            )
            script.write_text("renderer", encoding="utf-8")
            masks[-1].unlink()
            self.assertFalse(
                reusable_generic_record(
                    root,
                    output,
                    sample,
                    metadata_path,
                    metadata,
                    record,
                    False,
                    3,
                    script,
                )
            )


if __name__ == "__main__":
    unittest.main()
