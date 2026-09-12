"""Exercise the real CLI stage transitions with expensive work isolated."""
import argparse
import copy
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from tools.cli import generate_two_object_dataset as generator
from tools.cli.two_object_admission import AdmittedManifests
from tools.core.hashing import sha256_file
from tools.core.json_io import write_json_atomic_sorted as write


class RenderStageTests(unittest.TestCase):
    def test_metadata_admission_camera_render_resume_keeps_sampling_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory).resolve()
            args = argparse.Namespace(root=root, work_id='staged', release_root=Path('outputs/two_object'),
                released_base_manifest=root/'released.json', source_root=root, source_manifest=Path('source.json'),
                billiards_template=root/'b.json', passive_pinball_template=root/'p.json', marble_run_template=root/'m.json',
                generic_limit=1, specialized_seed=7, physics_workers=1, render_workers=1, gpus='0',
                max_admission_attempts=8, resume=False, plan_only=False, metadata_only=True,
                admission_only=False, sampling_config=None, accepted_base_cameras=None)
            layout = generator.generation_layout(root, args.work_id, args.release_root)
            counts = {'generic': 1, 'billiards': 3, 'passive_pinball': 3, 'marble_run': 3}
            sampled = {'object_count': 2, 'family_counts': counts, 'sample_count': 10,
                       'records': [], 'status': 'sampled_pending_simulation'}
            stack.enter_context(patch.object(generator, 'parse_args', return_value=args))
            plan = {'request': {'seed': 7}, 'code': 'fixed'}
            stack.enter_context(patch.object(generator, 'generation_plan', side_effect=lambda *a, **kw: copy.deepcopy(plan)))
            stack.enter_context(patch.object(generator, 'sampling_settings', return_value=(None, {}, counts, 1)))
            stack.enter_context(patch.object(generator, 'assemble_base_manifest', return_value=sampled))
            def run_stage(command, *, completion, resume):
                if completion.exists():
                    self.assertTrue(resume)
                    return
                if command[2] == 'tools.rendering.prepare_sweep_render_manifests':
                    binding = generator.bind_render_stage(root, layout, plan_path, args.accepted_base_cameras, resume=True)
                    write(completion, {'source_release': (layout.source_release/'manifest.json').relative_to(root).as_posix(),
                        'source_release_sha256': sha256_file(layout.source_release/'manifest.json'),
                        'source_staged_base_manifest': layout.base_manifest.relative_to(root).as_posix(),
                        'source_staged_base_manifest_sha256': sha256_file(layout.base_manifest),
                        'accepted_base_cameras': binding})
                else:
                    write(completion, {})
            stack.enter_context(patch.object(generator, 'run_once', side_effect=run_stage))
            admitted = AdmittedManifests(layout.base_manifest, layout.base_dataset/'physics/manifest.json',
                layout.sweep_metadata/'manifest.json', layout.sweep_physics/'manifest.json',
                layout.base_render/'generic/bound_manifest.json')
            def admit(**kwargs):
                for path in (admitted.base, admitted.base_physics, admitted.sweep_metadata, admitted.sweep_physics, admitted.base_bound):
                    if not path.exists():
                        write(path, sampled if path == admitted.base else {})
                return admitted
            admission = stack.enter_context(patch.object(generator, 'admit_two_object_groups', side_effect=admit))
            stack.enter_context(patch.object(generator, 'publish_source_release',
                side_effect=lambda **kw: write(layout.source_release/'manifest.json', {'status': 'published'})))
            render = stack.enter_context(patch.object(generator, 'render_sweep'))
            publish = stack.enter_context(patch.object(generator, 'publish_dataset', return_value={'status': 'complete'}))
            stack.enter_context(patch.object(generator, 'release_specs', return_value=[]))

            generator.main()
            plan_path = root/'outputs/staged/generation_plan.json'
            checkpoint = layout.base_dataset/'sampled_manifest.json'
            before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (plan_path, checkpoint)}
            admission.assert_not_called(); render.assert_not_called()
            args.resume = True; args.metadata_only = False; args.admission_only = True
            generator.main()
            self.assertTrue(admitted.base_physics.is_file())
            self.assertFalse((layout.source_release/'manifest.json').exists())
            render.assert_not_called(); publish.assert_not_called()
            args.admission_only = False
            args.accepted_base_cameras = root/'accepted.json'
            write(args.accepted_base_cameras, {'camera': 'reviewed after admission'})
            generator.main()
            render.assert_called_once(); publish.assert_called_once()
            generator.main()
            self.assertEqual(render.call_count, 2)
            for path, expected in before.items():
                self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), expected)
            frozen_render = plan_path.with_name('render_execution_plan.json')
            self.assertEqual(json.loads(frozen_render.read_text())['accepted_base_cameras']['sha256'], sha256_file(args.accepted_base_cameras))

            write(args.accepted_base_cameras, {'camera': 'changed after render preparation'})
            with self.assertRaisesRegex(ValueError, 'prepared render stage'):
                generator.main()
            self.assertEqual(render.call_count, 2)
            plan['request']['seed'] = 8
            with self.assertRaisesRegex(ValueError, 'frozen generation plan'):
                generator.main()

    def test_unprepared_inputs_can_be_corrected_without_rebinding_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            layout = generator.generation_layout(root, 'staged', Path('outputs/two_object'))
            plan = root/'outputs/staged/generation_plan.json'
            write(plan, {'seed': 7})
            write(layout.base_manifest, {})
            release = layout.source_release/'manifest.json'
            write(release, {})
            before = plan.read_bytes()
            camera = root/'camera.json'
            write(camera, {'status': 'pending'})
            binding = generator.bind_render_stage(root, layout, plan, camera, resume=True)
            frozen = plan.with_name('render_execution_plan.json')
            self.assertFalse(frozen.exists())
            legacy = {'schema_version': 'physweep_two_object_render_execution_plan_v1',
                      'accepted_base_cameras': binding}
            for key, path in [('generation_plan', plan), ('source_release', release),
                              ('admitted_base', layout.base_manifest)]:
                legacy[key] = {'path': path.relative_to(root).as_posix(), 'sha256': sha256_file(path)}
            write(frozen, legacy)
            write(camera, {'status': 'accepted'})
            corrected = generator.bind_render_stage(root, layout, plan, camera, resume=True)
            self.assertNotEqual(binding, corrected)
            write(layout.sweep_render/'manifest.json', {
                'source_release': legacy['source_release']['path'],
                'source_release_sha256': legacy['source_release']['sha256'],
                'source_staged_base_manifest': legacy['admitted_base']['path'],
                'source_staged_base_manifest_sha256': legacy['admitted_base']['sha256'],
                'accepted_base_cameras': corrected})
            generator.bind_render_stage(root, layout, plan, camera, resume=True)
            self.assertEqual(json.loads(frozen.read_text())['accepted_base_cameras'], corrected)
            self.assertEqual(plan.read_bytes(), before)

    def test_unbound_render_checkpoint_cannot_silently_gain_a_camera(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            layout = generator.generation_layout(root, 'staged', Path('outputs/two_object'))
            plan_path = root/'outputs/staged/generation_plan.json'
            write(plan_path, {'seed': 7})
            write(layout.base_manifest, {'base': 'frozen'})
            release = layout.source_release/'manifest.json'
            write(release, {'release': 'frozen'})
            write(layout.sweep_render/'manifest.json', {
                'source_release': release.relative_to(root).as_posix(), 'source_release_sha256': sha256_file(release),
                'source_staged_base_manifest': layout.base_manifest.relative_to(root).as_posix(),
                'source_staged_base_manifest_sha256': sha256_file(layout.base_manifest)})
            camera = root/'accepted.json'; write(camera, {'camera': 'new'})
            with self.assertRaisesRegex(ValueError, 'prepared render stage'):
                generator.bind_render_stage(root, layout, plan_path, camera, resume=True)
            self.assertFalse(plan_path.with_name('render_execution_plan.json').exists())
            prepared = layout.sweep_render/'manifest.json'
            previous = json.loads(prepared.read_text())
            previous['accepted_base_cameras'] = {'path': './accepted.json', 'sha256': sha256_file(camera)}
            write(prepared, previous)
            binding = generator.bind_render_stage(root, layout, plan_path, camera, resume=True)
            self.assertEqual(binding, {'path': 'accepted.json', 'sha256': sha256_file(camera)})


if __name__ == '__main__':
    unittest.main()
