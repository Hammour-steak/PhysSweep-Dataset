from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.rendering.freeze_asset_sweep_cameras import freeze_cameras, json_bytes, sha256_bytes


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(value))


class FreezeAssetSweepCameraTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[Path, Path]:
        branch = root / "outputs/run/sweep/asset"
        parent = "parent"
        records = []
        source_path = root / "datasets/source.json"
        write_json(source_path, {"scene_id": parent})
        source_hash = sha256_bytes(source_path.read_bytes())
        for index in range(13):
            kind = "base" if index == 0 else "sweep"
            scene_id = f"scene_{index:02d}"
            metadata_path = branch / "metadata" / f"{scene_id}.json"
            metadata = {
                "schema_version": "physweep_asset_proxy_scene_v3",
                "scene_id": scene_id,
                "sweep": {"kind": kind, "parent_scene_id": parent},
                "source_metadata": {
                    "path": str(source_path.relative_to(root)),
                    "sha256": source_hash,
                },
            }
            write_json(metadata_path, metadata)
            frame_dir = branch / "frames" / scene_id
            record = {
                "scene_id": scene_id,
                "metadata_path": str(metadata_path.relative_to(root)),
                "metadata_sha256": sha256_bytes(json_bytes(metadata)),
                "render_output": {
                    "inspection_frame_dir": str(frame_dir.relative_to(root)),
                    "video_path": str((branch / "videos" / f"{scene_id}.mp4").relative_to(root)),
                },
            }
            records.append(record)
            if kind == "base":
                write_json(
                    frame_dir / "render_record.json",
                    {
                        "schema_version": "physweep_asset_proxy_render_record_v1",
                        "scene_id": scene_id,
                        "metadata_sha256": source_hash,
                        "camera": {
                            "solver_version": "asset_motion_structure_camera_v2",
                            "position_m": [2.0, -1.0, 2.0],
                            "target_m": [0.0, 0.0, 0.5],
                            "focal_length_mm": 48.0,
                            "focus_span_m": 1.2,
                        },
                    },
                )
        manifest = {"sample_count": 13, "records": records}
        base = {"sample_count": 1, "records": records[:1]}
        derived = {"sample_count": 12, "records": records[1:]}
        manifest_path = branch / "render_input_manifest.json"
        base_path = branch / "base_render_input_manifest.json"
        write_json(manifest_path, manifest)
        write_json(base_path, base)
        write_json(branch / "derived_render_input_manifest.json", derived)
        return manifest_path, base_path

    def test_only_derived_metadata_receives_frozen_camera_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, base_path = self.fixture(root)
            for path in (
                manifest_path,
                base_path,
                manifest_path.with_name("derived_render_input_manifest.json"),
            ):
                legacy = json.loads(path.read_text(encoding="utf-8"))
                for record in legacy["records"]:
                    record.pop("metadata_sha256")
                write_json(path, legacy)
            freeze_cameras(root, manifest_path, base_path)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            derived = json.loads(
                manifest_path.with_name("derived_render_input_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            derived_hashes = {
                record["scene_id"]: record["metadata_sha256"]
                for record in derived["records"]
            }
            for index, record in enumerate(manifest["records"]):
                metadata_path = root / record["metadata_path"]
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    record["metadata_sha256"], sha256_bytes(metadata_path.read_bytes())
                )
                if index == 0:
                    self.assertNotIn("camera_binding", metadata)
                else:
                    self.assertEqual(
                        metadata["camera_binding"]["source_base_scene_id"], "scene_00"
                    )
                    self.assertEqual(
                        derived_hashes[record["scene_id"]], record["metadata_sha256"]
                    )

            freeze_cameras(root, manifest_path, base_path)

    def test_incomplete_group_is_rejected_before_metadata_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, base_path = self.fixture(root)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            missing = manifest["records"].pop()
            manifest["sample_count"] -= 1
            write_json(manifest_path, manifest)
            before = (root / missing["metadata_path"]).read_bytes()

            with self.assertRaisesRegex(ValueError, "partition"):
                freeze_cameras(root, manifest_path, base_path)
            self.assertEqual((root / missing["metadata_path"]).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
