from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from tools.assets.select_environment_review_samples import select_review_samples  # noqa: E402


def write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EnvironmentReviewSelectionTests(unittest.TestCase):
    def test_selects_one_sample_per_approved_profile_in_review_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            samples = []
            for index, profile_id in enumerate(("env_b", "procedural", "env_a")):
                metadata_path = root / "scenes" / str(index) / "metadata.json"
                metadata = {
                    "appearance": {
                        "scene_visual": {
                            "id": profile_id,
                            "composition": (
                                {"review_status": "approved"}
                                if profile_id != "procedural"
                                else None
                            ),
                        }
                    }
                }
                digest = write_json(metadata_path, metadata)
                samples.append(
                    {
                        "scene_id": str(index),
                        "metadata_path": str(metadata_path.relative_to(root)),
                        "metadata_sha256": digest,
                    }
                )
            manifest = {"samples": samples}
            composition = {
                "policy": {"admitted_review_status": "approved"},
                "records": [
                    {"profile_id": "env_a", "review_status": "approved"},
                    {"profile_id": "env_paused", "review_status": "paused"},
                    {"profile_id": "env_b", "review_status": "approved"},
                ],
            }
            selected, counts = select_review_samples(
                root, manifest, composition, samples_per_profile=1
            )
            self.assertEqual([record["scene_id"] for record in selected], ["2", "0"])
            self.assertEqual(counts, {"env_a": 1, "env_b": 1})

    def test_rejects_incomplete_approved_profile_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            manifest = {"samples": []}
            composition = {
                "policy": {"admitted_review_status": "approved"},
                "records": [
                    {"profile_id": "env_a", "review_status": "approved"}
                ],
            }
            with self.assertRaisesRegex(ValueError, "lacks approved profiles"):
                select_review_samples(
                    root, manifest, composition, samples_per_profile=1
                )


if __name__ == "__main__":
    unittest.main()
