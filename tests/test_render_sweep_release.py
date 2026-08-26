from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.render_sweep_release import (
    generic_render_command,
    load_release,
    release_metadata_selection,
    sha256,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class RenderSweepReleaseTests(unittest.TestCase):
    def fixture(
        self,
        root: Path,
        schema: str = "physweep_one_object_sweep_release_v1",
    ) -> Path:
        base_path = root / "datasets/release/base.json"
        write_json(
            base_path,
            {
                "sample_count": 1,
                "records": [
                    {"scene_id": "scene", "pipeline": "generic_pybullet"}
                ],
            },
        )
        release_path = root / "datasets/release/manifest.json"
        metadata_path = root / "datasets/release/metadata.json"
        write_json(
            metadata_path,
            {
                "sample_count": 13,
                "records": [
                    {
                        "scene_id": f"scene_{index:02d}",
                        "parent": "parent",
                        "source_schema_version": "physweep_pybullet_rigid_metadata_v1",
                        "kind": "base" if index == 0 else "sweep",
                    }
                    for index in range(13)
                ],
            },
        )
        write_json(
            release_path,
            {
                "schema_version": schema,
                "base_group_count": 1,
                "sample_count": 13,
                "pipeline_group_counts": {"generic_pybullet": 1},
                "base_manifest": str(base_path.relative_to(root)),
                "base_manifest_sha256": sha256(base_path),
                "metadata_manifest": str(metadata_path.relative_to(root)),
                "metadata_manifest_sha256": sha256(metadata_path),
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

    def test_release_rejects_wrong_base_pipeline_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_path = self.fixture(root)
            release = json.loads(release_path.read_text(encoding="utf-8"))
            base_path = root / str(release["base_manifest"])
            base = json.loads(base_path.read_text(encoding="utf-8"))
            base["records"][0]["pipeline"] = "asset_proxy"
            write_json(base_path, base)
            release["base_manifest_sha256"] = sha256(base_path)
            write_json(release_path, release)
            with self.assertRaisesRegex(ValueError, "base manifest contract"):
                load_release(root, release_path)

    def test_v1_release_derives_legacy_pipeline_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_path = self.fixture(root)
            release = json.loads(release_path.read_text(encoding="utf-8"))
            release.pop("pipeline_group_counts")
            write_json(release_path, release)
            _, release, _ = load_release(root, release_path)
            metadata_path, digest = release_metadata_selection(root, release)
            self.assertEqual(digest, sha256(metadata_path))

    def test_v4_release_schema_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_path = self.fixture(
                root, "physweep_one_object_sweep_release_v2"
            )
            _, release, _ = load_release(root, release_path)
            self.assertEqual(
                release["schema_version"],
                "physweep_one_object_sweep_release_v2",
            )

    def test_v5_release_schema_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_path = self.fixture(
                root, "physweep_one_object_sweep_release_v3"
            )
            _, release, _ = load_release(root, release_path)
            self.assertEqual(
                release["schema_version"],
                "physweep_one_object_sweep_release_v3",
            )

    def test_render_selection_is_hash_bound_to_complete_release_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            release_path = self.fixture(root)
            _, release, _ = load_release(root, release_path)
            metadata_path, metadata_sha256 = release_metadata_selection(
                root, release
            )
            self.assertEqual(metadata_sha256, sha256(metadata_path))

            metadata_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "metadata manifest hash mismatch"):
                release_metadata_selection(root, release)

    def test_render_selection_requires_complete_generic_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            release_path = self.fixture(root)
            _, release, _ = load_release(root, release_path)
            metadata_path = root / str(release["metadata_manifest"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["records"][0]["source_schema_version"] = "other"
            write_json(metadata_path, metadata)
            release["metadata_manifest_sha256"] = sha256(metadata_path)
            with self.assertRaisesRegex(ValueError, "generic metadata groups"):
                release_metadata_selection(root, release)
            metadata["records"][0]["source_schema_version"] = (
                "physweep_pybullet_rigid_metadata_v1"
            )
            metadata["records"][0]["kind"] = "sweep"
            write_json(metadata_path, metadata)
            release["metadata_manifest_sha256"] = sha256(metadata_path)
            with self.assertRaisesRegex(ValueError, "generic metadata groups"):
                release_metadata_selection(root, release)

    def test_generic_render_command_cannot_drop_release_selection(self) -> None:
        root = Path("/project")
        output = root / "outputs/release"
        selection = root / "datasets/release/metadata.json"
        command = generic_render_command(
            "python",
            root,
            output,
            4,
            "0,1,2,3",
            "base",
            selection,
            "a" * 64,
        )
        self.assertEqual(command.count("--selection-manifest"), 1)
        self.assertEqual(
            command[command.index("--selection-manifest") + 1], str(selection)
        )
        self.assertEqual(
            command[command.index("--selection-manifest-sha256") + 1], "a" * 64
        )
        self.assertTrue(command[-1].endswith("/base_render_manifest.json"))
        with self.assertRaisesRegex(ValueError, "unsupported generic render kind"):
            generic_render_command(
                "python", root, output, 1, "0", "all", selection, "a" * 64
            )

if __name__ == "__main__":
    unittest.main()
