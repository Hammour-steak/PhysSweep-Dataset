from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from tools.core.hashing import sha256_file
from tools.dataset_contract.object_identity_contract import attach_object_identity
from tools.sampling.assemble_two_object_base import assemble_base_manifest


def metadata(scene_id: str, schema: str) -> dict:
    motion_family = (
        "surface_dual_independent_2obj"
        if schema == "physweep_pybullet_rigid_metadata_v1"
        else "two_ball_direct_collision"
    )
    document = {
        "scene_id": scene_id,
        "schema_version": schema,
        "semantics": {"motion": {"family": motion_family}},
        "simulation": {
            "interaction": {"motion_pattern": motion_family},
            "objects": [
                {
                    "object_id": object_id,
                    "body_model": "rigid_body",
                    "semantic_type": "ball",
                }
                for object_id in ("object_a", "object_b")
            ]
        },
    }
    attach_object_identity(document)
    return document


class TwoObjectDatasetPipelineTests(unittest.TestCase):
    def test_admission_rejects_old_sweep_configuration_even_with_same_group_size(self):
        from tools.cli.two_object_admission import _validate_sweep_metadata, PHYSICS_SWEEP_CONFIG
        from tools.sampling.derive_physics_sweep import DEFAULT_CONFIG
        from tests.test_sweep_validation import sweep_records
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "configs").mkdir()
            (root / "configs/physics_sweep.json").write_bytes(DEFAULT_CONFIG.read_bytes())
            config = root / PHYSICS_SWEEP_CONFIG
            config.write_bytes(DEFAULT_CONFIG.with_name("two_object_physics_sweep.json").read_bytes())
            path = root / "manifest.json"
            manifest = {"config": {"path": str(PHYSICS_SWEEP_CONFIG), "sha256": sha256_file(config)},
                        "records": sweep_records(1)}
            path.write_text(json.dumps(manifest), encoding="utf-8")
            _validate_sweep_metadata(root, path)
            manifest["config"] = {"path": "configs/physics_sweep.json", "sha256": sha256_file(DEFAULT_CONFIG)}
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                _validate_sweep_metadata(root, path)

    def test_render_plan_counts_one_target_for_generic_and_specialized_families(self):
        from tools.cli.generate_two_object_dataset import generation_layout, render_sweep
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            layout = generation_layout(root, "single_target_test", root / "outputs/two_object")
            layout.sweep_render.mkdir(parents=True)
            plan_path = layout.sweep_render / "manifest.json"
            plan = {
                "object_count": 2, "sweep_target_object_indices": [0],
                "branch_counts": {family: 13 for family in (
                    "generic", "billiards", "passive_pinball", "marble_run")},
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with patch("tools.cli.generate_two_object_dataset.run") as run, patch(
                "tools.cli.generate_two_object_dataset.run_once"
            ), patch("tools.cli.generate_two_object_dataset.verify_render_manifest") as verify:
                render_sweep(root=root, layout=layout, workers=1, gpus="0", resume=False)
            self.assertEqual([call.args[1] for call in verify.call_args_list], [1] * 4 + [12] * 4)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands), 8)
            for index, command in enumerate(commands):
                result = Path(command[command.index('--result-manifest') + 1])
                self.assertEqual(result.name, 'base_render_manifest.json' if index < 4 else 'derived_render_manifest.json')
                self.assertNotIn('tools.rendering.bind_pybullet_visuals', command)
                self.assertEqual(command[command.index('--blender') + 1], str(Path('runtime/blender-3.4.0-linux-x64/blender')))
            plan["sweep_target_object_indices"] = [0, 1]
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "object_a-only"):
                render_sweep(root=root, layout=layout, workers=1, gpus="0", resume=True)

    def test_assembler_merges_hash_bound_generic_and_specialized_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            generic_path = root / "generic" / "metadata.json"
            specialized_path = root / "billiards" / "metadata.json"
            generic_path.parent.mkdir(parents=True)
            specialized_path.parent.mkdir(parents=True)
            generic_path.write_text(
                json.dumps(
                    metadata("generic_scene", "physweep_pybullet_rigid_metadata_v1")
                ),
                encoding="utf-8",
            )
            specialized_path.write_text(
                json.dumps(
                    metadata("billiards_scene", "physweep_billiards_scene_v4")
                ),
                encoding="utf-8",
            )
            generic_manifest = root / "generic_manifest.json"
            specialized_manifest = root / "specialized_manifest.json"
            generic_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "physweep_pybullet_base_manifest_v1",
                        "dataset_id": "physweep_two_object",
                        "sample_count": 1,
                        "samples": [
                            {
                                "scene_id": "generic_scene",
                                "metadata_path": generic_path.relative_to(root).as_posix(),
                                "metadata_sha256": sha256_file(generic_path),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            specialized_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "physweep_two_object_specialized_base_manifest_v1"
                        ),
                        "dataset_id": "physweep_two_object",
                        "sample_count": 1,
                        "samples": [
                            {
                                "scene_id": "billiards_scene",
                                "family": "billiards",
                                "metadata_path": (
                                    specialized_path.relative_to(root).as_posix()
                                ),
                                "metadata_sha256": sha256_file(specialized_path),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = assemble_base_manifest(
                root, generic_manifest, specialized_manifest
            )

        self.assertEqual(result["object_count"], 2)
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["family_counts"]["generic"], 1)
        self.assertEqual(result["family_counts"]["billiards"], 1)
        self.assertEqual(
            [record["scene_id"] for record in result["records"]],
            ["billiards_scene", "generic_scene"],
        )

    def test_assembler_rejects_a_changed_source_metadata_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            generic_path = root / "generic.json"
            specialized_path = root / "specialized.json"
            generic_path.write_text(
                json.dumps(
                    metadata("generic_scene", "physweep_pybullet_rigid_metadata_v1")
                ),
                encoding="utf-8",
            )
            specialized_path.write_text(
                json.dumps(
                    metadata("billiards_scene", "physweep_billiards_scene_v4")
                ),
                encoding="utf-8",
            )
            generic_manifest = root / "generic_manifest.json"
            specialized_manifest = root / "specialized_manifest.json"
            generic_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": "physweep_pybullet_base_manifest_v1",
                        "dataset_id": "physweep_two_object",
                        "sample_count": 1,
                        "samples": [
                            {
                                "scene_id": "generic_scene",
                                "metadata_path": "generic.json",
                                "metadata_sha256": "0" * 64,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            specialized_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "physweep_two_object_specialized_base_manifest_v1"
                        ),
                        "dataset_id": "physweep_two_object",
                        "sample_count": 1,
                        "samples": [
                            {
                                "scene_id": "billiards_scene",
                                "family": "billiards",
                                "metadata_path": "specialized.json",
                                "metadata_sha256": sha256_file(specialized_path),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "metadata hash differs"):
                assemble_base_manifest(
                    root, generic_manifest, specialized_manifest
                )


if __name__ == "__main__":
    unittest.main()
