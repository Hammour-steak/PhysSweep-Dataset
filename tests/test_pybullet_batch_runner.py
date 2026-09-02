import json
import unittest
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from tools.physics.run_pybullet_batch import (  # noqa: E402
    batch_failed,
    default_output_root,
    group_samples_by_schema,
    manifest_samples,
    prepare_output_root,
    worker_context,
)


class PyBulletBatchRunnerTests(unittest.TestCase):
    def test_default_output_stays_inside_the_source_dataset(self) -> None:
        manifest = Path("datasets/batch/manifest.json")
        self.assertEqual(
            default_output_root(manifest), Path("datasets/batch/physics")
        )

    def test_audit_rejections_can_be_returned_to_a_resampling_caller(self) -> None:
        self.assertFalse(
            batch_failed(
                rejected_count=1,
                error_count=0,
                allow_audit_rejections=True,
            )
        )
        self.assertTrue(
            batch_failed(
                rejected_count=1,
                error_count=0,
                allow_audit_rejections=False,
            )
        )
        self.assertTrue(
            batch_failed(
                rejected_count=0,
                error_count=1,
                allow_audit_rejections=True,
            )
        )

    def test_workers_use_fresh_spawned_processes(self) -> None:
        self.assertEqual(worker_context().get_start_method(), "spawn")

    def test_samples_are_isolated_by_source_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            records = {
                "generic.json": ("generic", "physweep_pybullet_rigid_metadata_v1"),
                "billiards.json": ("billiards", "physweep_billiards_scene_v4"),
            }
            for name, (scene_id, schema) in records.items():
                (root / name).write_text(
                    json.dumps({"scene_id": scene_id, "schema_version": schema})
                )
            samples = [
                {
                    "scene_id": scene_id,
                    "metadata_path": name,
                    "source_schema_version": schema,
                }
                for name, (scene_id, schema) in records.items()
            ]
            groups = group_samples_by_schema(root, samples)
        self.assertEqual(set(groups), {
            "physweep_pybullet_rigid_metadata_v1",
            "physweep_billiards_scene_v4",
        })
        self.assertEqual(groups["physweep_billiards_scene_v4"][0]["scene_id"], "billiards")

    def test_declared_schema_must_match_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "scene.json").write_text(
                json.dumps({"scene_id": "scene", "schema_version": "actual"})
            )
            with self.assertRaisesRegex(ValueError, "schema differs"):
                group_samples_by_schema(
                    root,
                    [{
                        "scene_id": "scene",
                        "metadata_path": "scene.json",
                        "source_schema_version": "declared",
                    }],
                )

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

    def test_specialized_two_object_manifest_is_normalized(self) -> None:
        sample = {
            "scene_id": "specialized",
            "metadata_path": "scenes/specialized/metadata.json",
        }
        dataset_id, samples = manifest_samples(
            {
                "schema_version": "physweep_two_object_specialized_base_manifest_v1",
                "dataset_id": "physweep_two_object",
                "sample_count": 1,
                "samples": [sample],
            }
        )
        self.assertEqual(dataset_id, "physweep_two_object")
        self.assertEqual(samples, [sample])

    def test_sweep_manifest_is_normalized(self) -> None:
        dataset_id, samples = manifest_samples(
            {
                "schema_version": "physweep_physics_sweep_manifest_v2",
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
            [
                {
                    "scene_id": "sweep",
                    "metadata_path": "sweep/metadata.json",
                    "source_schema_version": None,
                }
            ],
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

    def test_duplicate_scene_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate scene ids"):
            manifest_samples(
                {
                    "schema_version": "physweep_pybullet_base_manifest_v1",
                    "dataset_id": "duplicates",
                    "sample_count": 2,
                    "samples": [
                        {"scene_id": "same", "metadata_path": "a.json"},
                        {"scene_id": "same", "metadata_path": "b.json"},
                    ],
                }
            )

    def test_unsafe_scene_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsafe scene ids"):
            manifest_samples(
                {
                    "schema_version": "physweep_pybullet_base_manifest_v1",
                    "dataset_id": "unsafe",
                    "sample_count": 1,
                    "samples": [
                        {"scene_id": "../escape", "metadata_path": "scene.json"}
                    ],
                }
            )

    def test_nonempty_output_requires_explicit_permission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "physics"
            output.mkdir()
            (output / "stale.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "not empty"):
                prepare_output_root(output, allow_existing=False)
            prepare_output_root(output, allow_existing=True)

    def test_mixed_schema_sweep_is_forwarded_to_dispatcher(self) -> None:
        _, samples = manifest_samples(
            {
                "schema_version": "physweep_physics_sweep_manifest_v1",
                "dataset_id": "mixed",
                "sample_count": 1,
                "error_count": 0,
                "records": [
                    {
                        "scene_id": "asset_sweep",
                        "path": "sweep/metadata.json",
                        "source_schema_version": "physweep_asset_proxy_scene_v3",
                    }
                ],
            }
        )
        self.assertEqual(
            samples[0]["source_schema_version"], "physweep_asset_proxy_scene_v3"
        )


if __name__ == "__main__":
    unittest.main()
