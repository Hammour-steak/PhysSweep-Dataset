from __future__ import annotations

import os
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.cli import dataset_generation as generation
from tools.cli.two_object_admission import _camera_failure_manifest, _validate_base_binding
from tools.core.hashing import sha256_file


class DatasetGenerationTests(unittest.TestCase):
    def test_child_imports_the_running_code_not_the_data_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tools').mkdir()
            (root / 'tools/__init__.py').write_text('raise RuntimeError("stale code")')
            previous = Path.cwd()
            try:
                os.chdir(root)
                result = generation.run([
                    sys.executable, '-c',
                    'from pathlib import Path; import tools; '
                    f'assert str(Path(tools.__file__).resolve().parent) == {str(generation.CODE_ROOT / "tools")!r}',
                ])
            finally:
                os.chdir(previous)
            self.assertEqual(result.returncode, 0)

    def test_resume_binds_executable_code_but_ignores_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'tools').mkdir()
            source = root / 'tools/example.py'
            source.write_text('VALUE = 1\n')
            with patch.object(generation, 'CODE_ROOT', root):
                first = generation.generation_code_sha256()
                plan = root / 'generation_plan.json'
                generation.bind_generation_plan(plan, {'generator_code_sha256': first}, False)
                (root / 'outputs').mkdir()
                (root / 'outputs/result.txt').write_text('finished')
                self.assertEqual(generation.generation_code_sha256(), first)
                generation.bind_generation_plan(plan, {'generator_code_sha256': first}, True)
                source.write_text('VALUE = 2\n')
                changed = generation.generation_code_sha256()
                self.assertNotEqual(first, changed)
                with self.assertRaisesRegex(ValueError, 'frozen generation plan'):
                    generation.bind_generation_plan(plan, {'generator_code_sha256': changed}, True)

    def test_run_once_requires_a_completion_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / 'manifest.json'
            with patch.object(generation, 'run') as run:
                with self.assertRaisesRegex(RuntimeError, 'completion artifact'):
                    generation.run_once(['command'], completion=result, resume=False)
                result.write_text('{}')
                run.reset_mock()
                generation.run_once(['command'], completion=result, resume=True)
                run.assert_not_called()
                with self.assertRaises(FileExistsError):
                    generation.run_once(['command'], completion=result, resume=False)

    def test_camera_success_without_manifest_is_not_a_zero_exit_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generic = root / 'generic.json'
            physics = root / 'physics.json'
            generic.write_text(json.dumps({'samples': [{'scene_id': 'base'}]}))
            physics.write_text(json.dumps({'records': [{'scene_id': 'base'}]}))
            with patch('tools.cli.two_object_admission.run', return_value=subprocess.CompletedProcess([], 0)):
                with self.assertRaisesRegex(RuntimeError, 'neither a bound nor a failure manifest'):
                    _camera_failure_manifest(root=root, render_root=root / 'render',
                        generic_manifest=generic, base_physics=physics, workers=1)

    def test_two_object_seed_is_explicitly_specialized(self):
        from tools.cli.generate_two_object_dataset import parse_args
        argv = ['generate', '--work-id', 'test', '--released-base-manifest', 'released.json',
                '--source-root', '.', '--source-manifest', 'source.json',
                '--billiards-template', 'b.json', '--passive-pinball-template', 'p.json',
                '--marble-run-template', 'm.json', '--specialized-seed', '123']
        with patch('sys.argv', argv):
            args = parse_args()
        self.assertEqual(args.specialized_seed, 123)
        self.assertFalse(hasattr(args, 'seed'))



    def test_camera_reuse_checks_scene_set_and_source_hashes_without_resolving(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "source.json"
            source.write_text('{"scene_id":"base"}')
            trajectory = root / "trajectory.npz"
            trajectory.write_bytes(b"unchanged trajectory")
            metadata_path = root / "bound.json"
            metadata = {
                "schema_version": "physweep_pybullet_rigid_bound_metadata_v1",
                "scene_id": "base",
                "source_metadata": {"path": source.name, "sha256": sha256_file(source)},
                "trajectory": {"path": trajectory.name, "sha256": sha256_file(trajectory)},
                "visualization": {"camera": {"position_m": [1, 2, 3]}},
            }
            metadata_path.write_text(json.dumps(metadata))
            manifest_path = root / "bound_manifest.json"
            rules_path = root / "camera_rules.json"
            rules_path.write_text('{}')
            manifest = {
                "schema_version": "physweep_pybullet_bound_manifest_v2",
                "camera_rules": {"path": rules_path.name, "sha256": sha256_file(rules_path)},
                "sample_count": 1,
                "samples": [{"scene_id": "base", "metadata_path": metadata_path.name,
                             "metadata_sha256": sha256_file(metadata_path)}],
            }
            manifest_path.write_text(json.dumps(manifest))
            physics_path = root / "physics.json"
            physics_path.write_text(json.dumps({"records": [{
                "scene_id": "base", "metadata_path": str(source), "metadata_sha256": sha256_file(source),
                "trajectory_path": str(trajectory), "trajectory_sha256": sha256_file(trajectory),
            }]}))
            with patch("tools.rendering.camera_solver.solve_camera", side_effect=AssertionError("camera must be reused")):
                _validate_base_binding(root, manifest_path, physics_path, {"base"})
            with self.assertRaisesRegex(ValueError, "different scenes"):
                _validate_base_binding(root, manifest_path, physics_path, {"other"})
            for path in (source, trajectory):
                original = path.read_bytes()
                path.write_bytes(b"changed")
                with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, "binding differs"):
                    _validate_base_binding(root, manifest_path, physics_path, {"base"})
                path.write_bytes(original)
            rules_path.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "base camera rules changed"):
                _validate_base_binding(root, manifest_path, physics_path, {"base"})
            rules_path.write_text('{}')
            metadata_path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "base camera metadata changed"):
                _validate_base_binding(root, manifest_path, physics_path, {"base"})

    def test_camera_stage_filters_specialized_physics_and_reuses_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            generic = root / "generic.json"
            physics = root / "physics.json"
            generic.write_text(json.dumps({"samples": [{"scene_id": "base"}]}))
            physics.write_text(json.dumps({
                "schema_version": "physweep_pybullet_batch_record_v1",
                "sample_count": 2, "passed_count": 2, "rejected_count": 0, "error_count": 0,
                "records": [{"scene_id": "base", "ok": True, "audit_passed": True},
                            {"scene_id": "billiards", "ok": True, "audit_passed": True}],
            }))
            render = root / "outputs/admission"
            def complete(command, **kwargs):
                selection = json.loads(Path(command[command.index("--manifest") + 1]).read_text())
                self.assertEqual([r["scene_id"] for r in selection["records"]], ["base"])
                self.assertEqual(selection["sample_count"], 1)
                self.assertEqual(selection["passed_count"], 1)
                self.assertEqual(selection["source_manifest"], str(generic))
                (render / "generic/bound_manifest.json").write_text("{}")
                return subprocess.CompletedProcess(command, 0)
            with patch("tools.cli.two_object_admission.run", side_effect=complete) as run:
                for _ in range(2):
                    self.assertIsNone(_camera_failure_manifest(
                        root=root, render_root=render, generic_manifest=generic,
                        base_physics=physics, workers=1))
                run.assert_called_once()


    def test_render_completion_checks_records_not_just_summary_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "render.json"
            manifest = {"sample_count": 2, "success_count": 2, "failure_count": 0}
            for records in (None, [], [{"scene_id": "a", "ok": True}],
                            [{"scene_id": "a", "ok": True}, {"scene_id": "b", "ok": False}],
                            [{"scene_id": "a", "ok": True}, {"scene_id": "a", "ok": True}],
                            [{"scene_id": "a", "ok": True}, {"ok": True}]):
                path.write_text(json.dumps({**manifest, "records": records}))
                with self.subTest(records=records), self.assertRaises(ValueError):
                    generation.verify_render_manifest(path, 2)
            path.write_text(json.dumps({**manifest, "records": [
                {"scene_id": "a", "ok": True}, {"scene_id": "b", "ok": True}]}))
            generation.verify_render_manifest(path, 2)


    def test_admission_resume_requires_and_validates_the_frozen_camera_binding(self):
        from tools.cli import two_object_admission as admission
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            layout = generation.generation_layout(root, "resume_test", root / "outputs/two_object", object_count=2)
            files = [
                layout.base_manifest, layout.base_dataset / "physics/manifest.json",
                layout.sweep_metadata / "manifest.json", layout.sweep_physics / "manifest.json",
                layout.base_render / "generic/bound_manifest.json",
            ]
            for path in files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}")
            kwargs = dict(root=root, layout=layout, initial_generic_manifest=root / "generic.json",
                          specialized_manifest=root / "specialized.json", physics_workers=1,
                          camera_workers=1, max_attempts=1, resume=True)
            with patch.object(admission, "_validate_sweep_metadata"), patch.object(admission, "_generic_scene_ids", return_value={"base"}), patch.object(admission, "_physics_rejections", return_value=[]), patch.object(admission, "_validate_base_binding") as validate, patch.object(admission, "run_once", side_effect=RuntimeError("admission must run")) as stage:
                result = admission.admit_two_object_groups(**kwargs)
                self.assertEqual(result.base_bound, files[-1])
                validate.assert_called_once_with(root, files[-1], files[1], {"base"})
                stage.assert_not_called()
                files[-1].unlink()
                with self.assertRaisesRegex(RuntimeError, "admission must run"):
                    admission.admit_two_object_groups(**kwargs)

if __name__ == '__main__':
    unittest.main()
