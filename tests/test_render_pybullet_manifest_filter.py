from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.render_pybullet_manifest import (
    result_manifest_name,
    select_release_records,
    select_sweep_kind,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RenderPybulletSweepKindTests(unittest.TestCase):
    def test_partitioned_runs_keep_separate_result_manifests(self) -> None:
        self.assertEqual(result_manifest_name(None), "render_manifest.json")
        self.assertEqual(result_manifest_name("base"), "base_render_manifest.json")
        self.assertEqual(
            result_manifest_name("sweep"), "derived_render_manifest.json"
        )

    def sample(self, root: Path, scene_id: str, kind: str) -> dict[str, str]:
        metadata_path = root / "metadata" / f"{scene_id}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps({"scene_id": scene_id, "sweep": {"kind": kind}}),
            encoding="utf-8",
        )
        return {
            "scene_id": scene_id,
            "kind": kind,
            "metadata_path": metadata_path.relative_to(root).as_posix(),
            "metadata_sha256": sha256(metadata_path),
        }

    def test_partitions_from_hashed_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            base = self.sample(root, "base", "base")
            derived = self.sample(root, "derived", "sweep")
            self.assertEqual(select_sweep_kind(root, [base, derived], "base"), [base])
            self.assertEqual(
                select_sweep_kind(root, [base, derived], "sweep"), [derived]
            )

    def test_rejects_manifest_metadata_kind_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sample = self.sample(root, "base", "base")
            sample["kind"] = "sweep"
            with self.assertRaisesRegex(ValueError, "sweep kinds differ"):
                select_sweep_kind(root, [sample], "base")

    def test_rejects_metadata_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            sample = self.sample(root, "base", "base")
            sample["metadata_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "metadata hash mismatch"):
                select_sweep_kind(root, [sample], "base")

    def test_release_manifest_selects_exact_source_schema_scene_ids(self) -> None:
        samples = [
            {"scene_id": "generic_base", "kind": "base"},
            {"scene_id": "generic_sweep", "kind": "sweep"},
            {"scene_id": "retired", "kind": "base"},
        ]
        selection = {
            "sample_count": 3,
            "records": [
                {
                    "scene_id": "generic_base",
                    "kind": "base",
                    "source_schema_version": "generic",
                },
                {
                    "scene_id": "generic_sweep",
                    "kind": "sweep",
                    "source_schema_version": "generic",
                },
                {
                    "scene_id": "specialized",
                    "kind": "base",
                    "source_schema_version": "specialized",
                },
            ],
        }
        self.assertEqual(
            select_release_records(samples, selection, "generic"), samples[:2]
        )

    def test_release_manifest_requires_every_selected_scene(self) -> None:
        selection = {
            "sample_count": 1,
            "records": [{"scene_id": "missing", "kind": "base"}],
        }
        with self.assertRaisesRegex(ValueError, "lacks 1 selected"):
            select_release_records([], selection, None)

    def test_release_manifest_rejects_kind_mismatch(self) -> None:
        selection = {
            "sample_count": 1,
            "records": [{"scene_id": "scene", "kind": "sweep"}],
        }
        with self.assertRaisesRegex(ValueError, "sweep kinds differ"):
            select_release_records(
                [{"scene_id": "scene", "kind": "base"}], selection, None
            )


if __name__ == "__main__":
    unittest.main()
