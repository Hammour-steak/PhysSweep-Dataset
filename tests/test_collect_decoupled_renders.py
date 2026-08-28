from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from tools.rendering.collect_decoupled_renders import collect, sha256  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class CollectedRenderTests(unittest.TestCase):
    def fixture(self, root: Path, *, video_hash: str | None = None) -> dict[str, Path]:
        source_manifest = root / "datasets/formal/manifest.json"
        metadata = root / "datasets/formal/scene/metadata.json"
        trajectory = root / "datasets/formal/scene/trajectory.npz"
        bound_metadata = root / "outputs/.staging_current_v4/generic/metadata/scene.json"
        video = root / "outputs/.staging_current_v4/generic/videos/scene.mp4"
        write_json(source_manifest, {"dataset_id": "formal"})
        write_json(metadata, {"scene_id": "scene"})
        trajectory.parent.mkdir(parents=True, exist_ok=True)
        trajectory.write_bytes(b"trajectory")
        write_json(
            bound_metadata,
            {
                "scene_id": "scene",
                "visualization": {
                    "render": {
                        "video_path": "outputs/.staging_current_v4/generic/videos/scene.mp4",
                        "inspection_frame_dir": "outputs/.staging_current_v4/generic/frames/scene",
                    }
                },
            },
        )
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"video")
        declared_video_hash = video_hash or sha256(video)
        staged = root / "outputs/.staging_current_v4/staged_manifest.json"
        write_json(
            staged,
            {
                "schema_version": "physweep_one_object_decoupled_manifest_v3",
                "dataset_id": "formal__all",
                "sample_count": 1,
                "source_manifest": "datasets/formal/manifest.json",
                "records": [
                    {
                        "index": 1,
                        "scene_id": "outer",
                        "motion_intent": "slide_push_1obj",
                        "environment_id": "generic_matrix",
                        "profile": "five_dimensional_matrix",
                        "pipeline": "generic_pybullet",
                        "metadata_path": "datasets/formal/scene/metadata.json",
                        "metadata_sha256": sha256(metadata),
                    }
                ],
            },
        )
        render_record = {
            "scene_id": "scene",
            "ok": True,
            "gpu": 0,
            "egl_device_verified": True,
            "render_record": {
                "schema_version": "physweep_pybullet_render_record_v1",
                "scene_id": "scene",
                "metadata_path": str(bound_metadata),
                "metadata_sha256": sha256(bound_metadata),
                "trajectory_path": str(trajectory),
                "trajectory_sha256": sha256(trajectory),
                "video_path": str(video),
                "video_sha256": declared_video_hash,
                "inspection_frames": [],
                "render_engine": "BLENDER_EEVEE",
                "video_encoding": {"fps": 24},
            },
        }
        inspection_frame = (
            root / "outputs/.staging_current_v4/generic/frames/scene/frame_0001.png"
        )
        inspection_frame.parent.mkdir(parents=True)
        inspection_frame.write_bytes(b"frame")
        render_record["render_record"]["inspection_frames"] = [str(inspection_frame)]
        branch_paths = {}
        for name, records in {
            "generic": [render_record],
            "asset": [],
            "billiards": [],
        }.items():
            branch = root / f"outputs/.staging_current_v4/{name}/render_manifest.json"
            write_json(
                branch,
                {
                    "sample_count": len(records),
                    "records": records,
                    "egl_device_selector": {"binary_sha256": "selector"},
                },
            )
            branch_paths[name] = branch
        return {
            "manifest": staged,
            "generic": branch_paths["generic"],
            "asset": branch_paths["asset"],
            "billiards": branch_paths["billiards"],
        }

    def run_collect(self, root: Path, paths: dict[str, Path]) -> dict[str, object]:
        return collect(
            root=root,
            manifest_path=paths["manifest"],
            generic_render_manifest_path=paths["generic"],
            asset_render_manifest_path=paths["asset"],
            billiards_render_manifest_path=paths["billiards"],
            output=root / "outputs/current",
            overwrite=False,
        )

    def test_collection_is_self_contained_after_staging_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_collect(root, self.fixture(root))
            record = result["records"][0]
            self.assertEqual(
                result["schema_version"],
                "physweep_decoupled_collected_renders_v4",
            )
            self.assertNotIn(".staging_current_v4", json.dumps(result))
            self.assertEqual(result["source_manifest"], "datasets/formal/manifest.json")
            retained = root / record["effective_render_metadata_path"]
            self.assertTrue(retained.is_file())
            self.assertEqual(sha256(retained), record["effective_render_metadata_sha256"])
            retained_value = json.loads(retained.read_text(encoding="utf-8"))
            self.assertNotIn(".staging_current_v4", json.dumps(retained_value))
            retained_frames = root / retained_value["visualization"]["render"][
                "inspection_frame_dir"
            ]
            self.assertEqual(
                [path.name for path in retained_frames.iterdir()], ["frame_0001.png"]
            )
            shutil_target = root / record["video_path"]
            self.assertTrue(shutil_target.is_file())
            self.assertEqual(sha256(shutil_target), record["sha256"])

    def test_declared_render_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.fixture(root, video_hash="0" * 64)
            with self.assertRaises(ValueError):
                self.run_collect(root, paths)

    def test_failed_overwrite_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.fixture(root, video_hash="0" * 64)
            output = root / "outputs/current"
            sentinel = output / "sentinel"
            sentinel.parent.mkdir(parents=True)
            sentinel.write_text("keep", encoding="utf-8")
            with self.assertRaises(ValueError):
                collect(
                    root=root,
                    manifest_path=paths["manifest"],
                    generic_render_manifest_path=paths["generic"],
                    asset_render_manifest_path=paths["asset"],
                    billiards_render_manifest_path=paths["billiards"],
                    output=output,
                    overwrite=True,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_raw_source_manifest_is_rejected_before_output_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.fixture(root)
            staged = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            staged.pop("source_manifest")
            write_json(paths["manifest"], staged)
            output = root / "outputs/current"
            with self.assertRaisesRegex(ValueError, "staged_manifest.json"):
                self.run_collect(root, paths)
            self.assertFalse(output.exists())

    def test_only_required_specialized_branch_is_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self.fixture(root)
            staged = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            staged["schema_version"] = "physweep_one_object_decoupled_manifest_v4"
            staged["records"][0]["pipeline"] = "passive_pinball"
            write_json(paths["manifest"], staged)

            generic = json.loads(paths["generic"].read_text(encoding="utf-8"))
            envelope = generic["records"][0]
            envelope["scene_id"] = "outer"
            envelope["render_record"]["scene_id"] = "outer"
            specialized = root / "outputs/.staging_current_v4/pinball/render_manifest.json"
            write_json(specialized, generic)

            result = collect(
                root=root,
                manifest_path=paths["manifest"],
                generic_render_manifest_path=None,
                asset_render_manifest_path=None,
                billiards_render_manifest_path=None,
                output=root / "outputs/current",
                overwrite=False,
                specialized_render_manifest_paths={"passive_pinball": specialized},
            )

            self.assertEqual(result["records"][0]["pipeline"], "passive_pinball")
            self.assertEqual(
                set(result["render_runtime"]["branch_manifest_sha256"]),
                {"passive_pinball"},
            )


if __name__ == "__main__":
    unittest.main()
