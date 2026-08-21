import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_pybullet_batch import manifest_samples  # noqa: E402


class PyBulletBatchRunnerTests(unittest.TestCase):
    def test_base_manifest_is_normalized(self) -> None:
        sample = {"scene_id": "base", "metadata_path": "base/metadata.json"}
        dataset_id, samples = manifest_samples(
            {
                "schema_version": "physweep_pybullet_base_manifest_v1",
                "dataset_id": "base_set",
                "sample_count": 1,
                "samples": [sample],
            }
        )
        self.assertEqual(dataset_id, "base_set")
        self.assertEqual(samples, [sample])

    def test_sweep_manifest_is_normalized(self) -> None:
        dataset_id, samples = manifest_samples(
            {
                "schema_version": "physweep_physics_sweep_manifest_v1",
                "dataset_id": "sweep_set",
                "sample_count": 1,
                "error_count": 0,
                "records": [
                    {
                        "scene_id": "sweep",
                        "path": "sweep/metadata.json",
                    }
                ],
            }
        )
        self.assertEqual(dataset_id, "sweep_set")
        self.assertEqual(
            samples,
            [{"scene_id": "sweep", "metadata_path": "sweep/metadata.json"}],
        )

    def test_derivation_errors_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "derivation errors"):
            manifest_samples(
                {
                    "schema_version": "physweep_physics_sweep_manifest_v1",
                    "dataset_id": "bad",
                    "error_count": 1,
                    "records": [],
                }
            )


if __name__ == "__main__":
    unittest.main()
