from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from render_asset_proxy_manifest import (  # noqa: E402
    output_path,
    render_source_records,
    reusable_render_record as reusable_asset_record,
    sha256,
)
from render_pybullet_manifest import (  # noqa: E402
    reusable_render_record as reusable_generic_record,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class RenderManifestResumeTests(unittest.TestCase):
    def test_billiards_legacy_paths_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = root / "datasets/scene.json"
            write_json(
                metadata,
                {
                    "schema_version": "physweep_billiards_scene_v4",
                    "scene_id": "scene",
                },
            )
            records = render_source_records(
                root,
                {"billiards_metadata_paths": [str(metadata.relative_to(root))]},
                "billiards",
            )
            self.assertEqual(records[0]["scene_id"], "scene")

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
                )

    def test_output_path_rejects_non_output_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "root/outputs"):
                output_path(root, "datasets/formal/video.mp4")

    def test_asset_record_requires_verified_egl_provenance(self) -> None:
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

    def test_generic_record_requires_complete_masks_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs/run/generic"
            metadata_path = output / "metadata/scene.json"
            trajectory = output / "physics/scene.npz"
            video = output / "videos/scene.mp4"
            frame_dir = output / "frames/scene"
            mask_dir = output / "masks/scene/object"
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
                )
            )
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
                )
            )


if __name__ == "__main__":
    unittest.main()
