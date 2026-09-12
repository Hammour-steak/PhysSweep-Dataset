"""Code provenance supports a distinct source tree while data stays contained."""
import tempfile
import unittest
from pathlib import Path

from tools.core.hashing import implementation_file_binding, relative_file_binding
from tools.rendering.render_asset_proxy_manifest import implementation_is_reusable, EVIDENCE_CONTRACT


class ImplementationSourceRootTests(unittest.TestCase):
    def test_external_code_is_bound_and_validated_against_the_actual_renderer(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)/"data"
            data.mkdir()
            code = Path(directory)/"code/tools/rendering"
            code.mkdir(parents=True)
            renderer = code/"render_two_object_specialized_scene.py"
            helper = code/"specialized_render_evidence.py"
            renderer.write_text("renderer")
            helper.write_text("evidence")
            binding = implementation_file_binding(data, renderer)
            self.assertTrue(Path(binding['path']).is_absolute())
            with self.assertRaises(ValueError):
                relative_file_binding(data, renderer)
            metadata = {'render': {'evidence_contract': EVIDENCE_CONTRACT}, 'implementation': {
                'renderer': binding, 'render_evidence': implementation_file_binding(data, helper)}}
            self.assertTrue(implementation_is_reusable(data, metadata, renderer))
            old = data/'tools/rendering/render_two_object_specialized_scene.py'
            old.parent.mkdir(parents=True)
            old.write_text("renderer")
            metadata['implementation']['renderer'] = implementation_file_binding(data, old)
            self.assertFalse(implementation_is_reusable(data, metadata, renderer))
            metadata['implementation']['renderer'] = binding
            helper.write_text("changed")
            self.assertFalse(implementation_is_reusable(data, metadata, renderer))

    def test_specialized_worker_forwards_data_root_to_blender(self):
        from contextlib import contextmanager
        from unittest.mock import patch
        from tools.rendering import render_asset_proxy_manifest as driver
        from tools.core.json_io import write_json_atomic as write
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"data"
            output = root/"outputs/smoke"
            metadata_path = output/"metadata/a.json"
            paths = {'inspection_frame_dir': str(output/"frames/a"),
                     'video_path': str(output/"videos/a.mp4")}
            write(metadata_path, {'render': paths})
            script = Path(directory)/"code/tools/rendering/render_two_object_specialized_scene.py"
            record = {'scene_id': 'a', 'metadata_path': str(metadata_path)}
            @contextmanager
            def environment(*args):
                yield {}, 'selector'
            with patch.object(driver, 'isolated_blender_environment', environment), patch.object(driver.subprocess, 'run', side_effect=RuntimeError('captured')) as invoke:
                with self.assertRaisesRegex(RuntimeError, 'captured'):
                    driver.worker(root, root/'blender', script, record, output, 0, root/'selector', False)
            argv = invoke.call_args.args[0]
            self.assertEqual(argv[argv.index('--python')+1], str(script))
            self.assertEqual(argv[argv.index('--root')+1], str(root))
