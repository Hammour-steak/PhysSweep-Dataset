from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.render_sweep_release import load_release, sha256


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class RenderSweepReleaseTests(unittest.TestCase):
    def fixture(self, root: Path) -> Path:
        base_path = root / "datasets/release/base.json"
        write_json(
            base_path,
            {"sample_count": 1, "records": [{"scene_id": "scene"}]},
        )
        release_path = root / "datasets/release/manifest.json"
        write_json(
            release_path,
            {
                "schema_version": "physweep_one_object_sweep_release_v1",
                "base_group_count": 1,
                "sample_count": 13,
                "base_manifest": str(base_path.relative_to(root)),
                "base_manifest_sha256": sha256(base_path),
            },
        )
        return release_path

    def test_release_and_base_provenance_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_path = self.fixture(root)
            _, release, base_path = load_release(root, release_path.relative_to(root))
            self.assertEqual(release["sample_count"], 13)
            self.assertEqual(base_path.name, "base.json")

            base_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_release(root, release_path)


if __name__ == "__main__":
    unittest.main()
