import importlib.util
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.dataset_contract.gt_scene_input import interaction_collider_ids


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PipelineBoundaryTest(unittest.TestCase):
    def test_gt_interaction_requires_an_explicit_multi_object_target(self):
        metadata = {
            "simulation": {
                "support": {
                    "colliders": [{"id": "floor", "role": "primary_support"}]
                },
                "objects": [
                    {"object_id": "a", "expected_motion": {}},
                    {
                        "object_id": "b",
                        "expected_motion": {
                            "required_collider_contact_id": "floor"
                        },
                    },
                ],
            }
        }
        with self.assertRaisesRegex(ValueError, "controlled_object_id is required"):
            interaction_collider_ids(metadata)
        self.assertEqual(interaction_collider_ids(metadata, "b"), ("floor",))

    def test_model_source_is_not_part_of_dataset_repository(self):
        self.assertFalse((ROOT / "tools/model_training").exists())
        self.assertFalse((ROOT / "tools/training_data").exists())
        self.assertFalse((ROOT / "configs/training").exists())

    def test_dataset_config_has_no_model_settings(self):
        module = load_module(
            "dataset_build_entry",
            ROOT / "tools/cli/build_one_object_dataset.py",
        )
        config = module.load_config(ROOT / "configs/datasets/one_object.json")
        self.assertEqual(
            set(config), {"schema_version", "release_root", "object_count"}
        )
        self.assertEqual(config["release_root"], "outputs/one_object")
        self.assertEqual(config["object_count"], 1)

    def test_dataset_entry_requires_explicit_published_sources(self):
        module = load_module(
            "dataset_publish_entry",
            ROOT / "tools/cli/build_one_object_dataset.py",
        )
        specs = module.pipeline_specs(
            [("generic", "schema", "project", "render", "masks")]
        )
        self.assertEqual(specs[0].name, "generic")
        self.assertEqual(specs[0].source_schema_version, "schema")

    def test_generation_plan_is_complete_and_registry_driven(self):
        module = load_module(
            "dataset_generation_entry",
            ROOT / "tools/cli/generate_one_object_dataset.py",
        )
        plan = module.generation_plan(
            ROOT, "smoke", Path("outputs/smoke/one_object")
        )
        self.assertEqual(
            plan["stages"],
            [
                "sample_and_audit_base",
                "stage_and_render_base",
                "derive_and_audit_sweep",
                "publish_source_release",
                "stage_and_render_derived_sweep",
                "materialize_canonical_base_and_sweep",
                "verify_canonical_release",
            ],
        )
        pipelines = {record["pipeline"] for record in plan["pipelines"]}
        self.assertEqual(
            pipelines,
            {
                "generic_pybullet",
                "asset_proxy",
                "billiards",
                "passive_pinball",
                "marble_run",
            },
        )
        self.assertTrue(plan["layout"]["canonical_release"].endswith("one_object"))
        source = (ROOT / "tools/cli/generate_one_object_dataset.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("tools.rendering.freeze_asset_sweep_cameras", source)
        for renderer in (
            "render_passive_pinball_scene.py",
            "render_marble_run_scene.py",
        ):
            renderer_source = (ROOT / "tools/rendering" / renderer).read_text(
                encoding="utf-8"
            )
            self.assertIn("include_mask_output=True", renderer_source)
            self.assertIn("args.instance_mask_dir", renderer_source)

    def test_matrix_sampler_defaults_to_the_formal_release_size(self):
        tree = ast.parse(
            (ROOT / "tools/sampling/sample_one_object_scene_matrix.py").read_text(
                encoding="utf-8"
            )
        )
        defaults = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "DEFAULT_MATRIX_COUNT"
        }
        self.assertEqual(defaults, {"DEFAULT_MATRIX_COUNT": 3200})

    def test_dataset_entry_propagates_the_one_object_boundary(self):
        module = load_module(
            "dataset_boundary_entry",
            ROOT / "tools/cli/build_one_object_dataset.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            release_root = Path(directory) / "one_object"
            with (
                patch.object(module, "verify_base_view", return_value={"passed": True}) as base,
                patch.object(module, "verify_sweep_view", return_value={"passed": True}) as sweep,
            ):
                module.verify_dataset(release_root)
        base.assert_called_once_with(
            release_root / "base", expected_object_count=1
        )
        sweep.assert_called_once_with(
            release_root / "sweep",
            base_root=release_root / "base",
            expected_object_count=1,
        )

    def test_base_and_sweep_render_sources_are_distinct(self):
        module = load_module(
            "dataset_distinct_render_entry",
            ROOT / "tools/cli/build_one_object_dataset.py",
        )
        base_spec = module.pipeline_specs(
            [("generic", "schema", "project", "base_render", "base_masks")]
        )[0]
        sweep_spec = module.pipeline_specs(
            [("generic", "schema", "project", "sweep_render", "sweep_masks")]
        )[0]
        with tempfile.TemporaryDirectory() as directory:
            release_root = Path(directory) / "one_object"
            with (
                patch.object(module, "build_base_view", return_value={"passed": True}) as base,
                patch.object(module, "build_sweep_view", return_value={"passed": True}) as sweep,
            ):
                module.publish_dataset(
                    release_project_root=Path(directory),
                    release_manifest=Path("release.json"),
                    release_root=release_root,
                    base_specs=[base_spec],
                    sweep_specs=[sweep_spec],
                    workers=2,
                    resume=False,
                )
        self.assertEqual(base.call_args.kwargs["pipeline_specs"], [base_spec])
        self.assertEqual(sweep.call_args.kwargs["pipeline_specs"], [sweep_spec])

    def test_generator_publishes_sweep_baselines_as_canonical_bases(self):
        module = load_module(
            "dataset_sweep_baseline_entry",
            ROOT / "tools/cli/generate_one_object_dataset.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "datasets/work/release"
            metadata = release / "metadata.json"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "source_schema_version": "physweep_pybullet_rigid_metadata_v1"
                            },
                            {"source_schema_version": "asset_schema"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (release / "manifest.json").write_text(
                json.dumps({"metadata_manifest": str(metadata.relative_to(root))}),
                encoding="utf-8",
            )
            layout = module.generation_layout(
                root, "work", Path("outputs/work/one_object")
            )
            with patch.object(
                module,
                "load_specialized_backends",
                return_value=[
                    {
                        "source_schema_version": "asset_schema",
                        "sweep_branch": "asset",
                    }
                ],
            ):
                base_specs, sweep_specs = module.release_specs(root, layout)
        base_roots = {spec.name: spec.render_root for spec in base_specs}
        sweep_roots = {spec.name: spec.render_root for spec in sweep_specs}
        self.assertEqual(base_roots, sweep_roots)
        self.assertEqual(
            base_roots["generic"], layout.sweep_render / "generic" / "bound"
        )
        self.assertEqual(base_roots["asset"], layout.sweep_render / "asset")

    def test_bound_manifest_bootstraps_scene_export_without_published_dataset(self):
        module = load_module(
            "scene_export_entry",
            ROOT / "tools/training_export/build_gt_training_scenes.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "bound.json"
            manifest.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "scene_id": "scene_a",
                                "metadata_path": "outputs/release/generic/metadata/scene_a.json",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            records = module._base_records_from_bound_manifest(manifest)
        self.assertEqual(records[0]["base_scene_id"], "scene_a")
        self.assertEqual(
            records[0]["conditioning"]["first_frame"],
            "outputs/release/generic/frames/scene_a/frame_0001.png",
        )

    def test_published_paths_resolve_only_from_project_root(self):
        module = load_module(
            "point_trajectory_export_entry",
            ROOT / "tools/training_export/export_point_trajectories.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "physweep"
            root.mkdir()
            source = root / "outputs" / "trajectory.npz"
            source.parent.mkdir()
            source.write_bytes(b"trajectory")
            self.assertEqual(
                module.project_path(root, "outputs/trajectory.npz"), source.resolve()
            )
            with self.assertRaisesRegex(ValueError, "project-relative"):
                module.project_path(root, str(source.resolve()))
            with self.assertRaisesRegex(ValueError, "project-relative"):
                module.project_path(root, "../trajectory.npz")

    def test_release_audit_requires_an_explicit_path_base(self):
        module = load_module(
            "training_dataset_audit_entry",
            ROOT / "tools/training_export/audit_training_dataset.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "datasets" / "physweep_training"
            release.mkdir(parents=True)
            (release / "summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "project-root-relative"):
                module.audit(release, root, forbid_approximations=True)

    def test_source_ownership_is_one_way(self):
        dataset_source = (ROOT / "tools/cli/build_one_object_dataset.py").read_text()
        dataset_imports = {
            node.names[0].name.split(".")[0]
            for node in ast.walk(ast.parse(dataset_source))
            if isinstance(node, ast.Import)
        }
        for forbidden in ("torch", "wan_training", "train_wan", "cache_wan"):
            self.assertNotIn(forbidden, dataset_imports)

    def test_specialized_render_records_hash_the_bound_metadata(self):
        for name in (
            "render_asset_proxy_scene.py",
            "render_billiards_scene.py",
            "render_passive_pinball_scene.py",
            "render_marble_run_scene.py",
        ):
            source = (ROOT / "tools/rendering" / name).read_text(encoding="utf-8")
            self.assertIn(
                '"metadata_sha256": sha256(metadata_path)',
                source,
            )
            self.assertNotIn(
                '"metadata_sha256": simulation_record["metadata"]["sha256"]',
                source,
            )

    def test_method_specific_trajectory_rasterization_is_absent(self):
        source = (
            ROOT / "tools/dataset_contract/point_trajectory.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("rasterize_projected_tracks", source)
        self.assertNotIn("cover_center_crop_coordinates", source)
        self.assertNotIn("depth_normalization", source)

    def test_obsolete_scene_condition_schema_is_absent(self):
        self.assertFalse((ROOT / "tools/dataset_contract/scene_condition.py").exists())
        for path in (ROOT / "tools/dataset_contract").glob("*.py"):
            self.assertNotIn(
                "physweep.scene_condition.v1",
                path.read_text(encoding="utf-8"),
            )

    def test_method_brand_and_model_artifacts_are_absent(self):
        forbidden = ("PhyContext", "PhysBind", "phycontext.", "physbind")
        tracked_files = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        for relative in tracked_files:
            path = ROOT / relative
            if path == Path(__file__).resolve() or not path.is_file():
                continue
            if path.suffix not in {".py", ".md", ".json", ".yml", ".yaml", ".sh"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                self.assertNotIn(token, text, f"method token {token!r} in {path}")


if __name__ == "__main__":
    unittest.main()
