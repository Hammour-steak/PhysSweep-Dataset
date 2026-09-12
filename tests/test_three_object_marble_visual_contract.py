import unittest
import tempfile,json
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
from tests import test_three_object_marble_metadata as fixtures
from tools.physics.resolved_simulation_scene import compile_resolved_scene
from tools.physics.specialized_backend_registry import load_specialized_backends
from tools.release.base_release_schema import _compact_objects,build_fixture_payload,_solver_contract,_compact_semantics


class MarbleVisualContractTests(unittest.TestCase):
    def test_render_worker_passes_data_root_without_mask_arguments(self):
        from tools.rendering.render_asset_proxy_manifest import worker
        @contextmanager
        def environment(*a):yield {},'marker'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);output=root/'outputs';output.mkdir()
            path=output/'metadata.json';path.write_text(json.dumps({'render':{'inspection_frame_dir':'outputs/frames/scene','video_path':'outputs/videos/scene.mp4'}}))
            script=root/'render_three_object_marble_scene.py'
            with patch('tools.rendering.render_asset_proxy_manifest.isolated_blender_environment',environment),patch('tools.rendering.render_asset_proxy_manifest.subprocess.run',side_effect=RuntimeError('captured command')) as run:
                with self.assertRaisesRegex(RuntimeError,'captured command'):
                    worker(root,root/'blender',script,{'scene_id':'scene','metadata_path':str(path)},output,0,root/'selector',False)
            command=run.call_args.args[0]
            self.assertEqual(command[command.index('--root')+1],str(root));self.assertNotIn('--instance-mask-dir',command)

    def test_common_export_keeps_three_materials_fixture_and_motion(self):
        for target in ('P','Q','R'):
            instance=fixtures.ThreeObjectMarbleMetadataTests();self.addCleanup(instance.doCleanups)
            m,_=instance.candidate(target);scene=compile_resolved_scene(m,fixtures.ROOT)
            info={'object_ids':[o['object_id'] for o in scene['objects']],
                'runtime_material':[[o['material'][k] for k in ('mass_kg','contact_friction','contact_restitution')] for o in scene['objects']],
                'inertia_diagonal_kg_m2':[[.001]*3]*3}
            objects,_=_compact_objects('marble_run',m,scene,info,'track',{})
            self.assertEqual(len(objects),3)
            self.assertEqual(len({tuple(o['visual']['base_color_srgb_rgba']) for o in objects}),3)
            for actual,source in zip(objects,m['simulation']['objects']):
                self.assertEqual(actual['object_id'],source['object_id'])
                self.assertEqual(actual['material']['rolling_friction'],source['material']['rolling_friction'])
                self.assertEqual(actual['material']['linear_damping'],source['material']['linear_damping'])
            fixture=build_fixture_payload('marble_run_three_object_v1',m,scene)
            self.assertEqual(len(fixture['physical']['fixture']['mesh_components']),len(m['physics']['fixture']['mesh_components']))
            self.assertEqual(len(fixture['physical']['fixture']['analytic_colliders']),5)
            self.assertEqual(_solver_contract('marble_run_three_object_v1',m,scene)['solver_iterations'],180)
            self.assertEqual(_compact_semantics(m)['profile'],'ordered_contacts')
        old={'schema_version':'physweep_marble_run_scene_v1','semantics':{'dynamic_object_count':1,'profile':'early_release_chain','motion_profile':'unrelated'}}
        self.assertEqual(_compact_semantics(old)['profile'],'early_release_chain')

    def test_marble_three_registry_is_explicit_and_excluded_from_older_counts(self):
        for count in (1,2):
            self.assertNotIn('marble_run_three_object',{r['pipeline'] for r in load_specialized_backends(fixtures.ROOT,object_count=count)})
        records=load_specialized_backends(fixtures.ROOT,object_count=3)
        r=next(r for r in records if r['pipeline']=='marble_run_three_object')
        self.assertEqual(r['dynamic_object_counts'],[3]);self.assertEqual(r['source_schema_version'],'physweep_marble_run_three_object_scene_v1')
