import importlib.util
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class PipelineBoundaryTest(unittest.TestCase):
    def test_model_source_is_not_part_of_dataset_repository(self):
        self.assertFalse((ROOT / "tools/model_training").exists())
        self.assertFalse((ROOT / "tools/training_data").exists())
        self.assertFalse((ROOT / "configs/training").exists())

    def test_dataset_config_has_no_model_settings(self):
        module = load_module(
            "dataset_build_entry",
            ROOT / "tools/dataset_generation/build_one_object_dataset.py",
        )
        config = module.load_config(ROOT / "configs/datasets/one_object.json")
        self.assertTrue(config["release_root"].startswith("datasets/"))
        commands = [token for stage in config["stages"] for token in stage["command"]]
        self.assertFalse(any("tools/training_data/" in token for token in commands))
        self.assertFalse(any("tools/model_training/" in token for token in commands))

    def test_bound_manifest_bootstraps_scene_export_without_published_dataset(self):
        module = load_module(
            "scene_export_entry",
            ROOT / "tools/dataset_generation/build_gt_training_scenes.py",
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
            ROOT / "tools/dataset_generation/export_point_trajectories.py",
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
            ROOT / "tools/dataset_generation/audit_training_dataset.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "datasets" / "physweep_training"
            release.mkdir(parents=True)
            (release / "summary.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "project-root-relative"):
                module.audit(release, root, forbid_approximations=True)

    def test_source_ownership_is_one_way(self):
        dataset_source = (ROOT / "tools/dataset_generation/build_one_object_dataset.py").read_text()
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
            source = (ROOT / "tools" / name).read_text(encoding="utf-8")
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
        for path in ROOT.rglob("*"):
            if path == Path(__file__).resolve():
                continue
            if not path.is_file() or any(
                part in {".git", "__pycache__"} for part in path.parts
            ):
                continue
            if path.suffix not in {".py", ".md", ".json", ".yml", ".yaml", ".sh"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in forbidden:
                self.assertNotIn(token, text, f"method token {token!r} in {path}")


if __name__ == "__main__":
    unittest.main()
