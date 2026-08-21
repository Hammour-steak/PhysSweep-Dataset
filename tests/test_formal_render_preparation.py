from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from prepare_formal_render_manifests import (  # noqa: E402
    sha256,
    stage_render_record,
)


class FormalRenderPreparationTests(unittest.TestCase):
    def test_render_override_preserves_source_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_path = root / "datasets/source/scene/metadata.json"
            metadata_path.parent.mkdir(parents=True)
            original = {"scene_id": "scene_001", "render": {"samples": 32}}
            metadata_path.write_text(json.dumps(original), encoding="utf-8")
            record = {
                "scene_id": "scene_001",
                "metadata_path": "datasets/source/scene/metadata.json",
                "metadata_sha256": sha256(metadata_path),
            }

            staged = stage_render_record(root, record, root / "outputs/staging/asset")

            self.assertEqual(staged["metadata_path"], record["metadata_path"])
            self.assertEqual(staged["metadata_sha256"], record["metadata_sha256"])
            self.assertEqual(
                staged["render_output"]["video_path"],
                "outputs/staging/asset/videos/scene_001.mp4",
            )
            self.assertEqual(json.loads(metadata_path.read_text()), original)
            self.assertFalse((root / "outputs/staging/asset/metadata").exists())

    def test_declared_source_hash_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata_path = root / "datasets/source/metadata.json"
            metadata_path.parent.mkdir(parents=True)
            metadata_path.write_text("{}", encoding="utf-8")
            record = {
                "scene_id": "scene_001",
                "metadata_path": "datasets/source/metadata.json",
                "metadata_sha256": "0" * 64,
            }
            with self.assertRaises(ValueError):
                stage_render_record(root, record, root / "outputs/staging")


if __name__ == "__main__":
    unittest.main()
