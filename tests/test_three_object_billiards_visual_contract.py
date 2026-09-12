import ast
import itertools
import tempfile
import unittest
from pathlib import Path
from tests import test_three_object_billiards_metadata as fixtures
from tools.physics.resolved_simulation_scene import compile_resolved_scene
from tools.release.base_release_schema import _compact_objects,build_fixture_payload,_solver_contract,_compact_semantics
from tools.rendering.render_asset_proxy_manifest import implementation_is_reusable,render_record_implementation_is_reusable
from tools.core.hashing import implementation_file_binding


class BilliardsVisualContractTests(unittest.TestCase):
    def test_public_motion_profile_preserves_the_frozen_three_object_type(self):
        m=fixtures.ThreeObjectBilliardsMetadataTests().candidate()
        self.assertEqual(_compact_semantics(m)['profile'],'chain_transfer')
        legacy={'schema_version':'physweep_billiards_scene_v4','semantics':{'dynamic_object_count':3,'profile':'legacy_three_ball','motion_profile':'unrelated'}}
        self.assertEqual(_compact_semantics(legacy)['profile'],'legacy_three_ball')
    def test_actual_slot_order_and_exported_object_appearance_agree(self):
        tree=ast.parse((fixtures.ROOT/'tools/rendering/specialized_sphere_rendering.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_fixture')
        palette={'cue_ball':'white','object_ball_1':'yellow','object_ball_2':'red'}
        namespace={'Any':object,'PROJECT_ROOT':fixtures.ROOT,'sha256':lambda p:'hash',
            'read_json':lambda p:{'records':[{'asset_id':'table','component_policy':{}}]},
            'add_support':lambda *a,**k:[],'hidden_ball_materials':lambda *a:palette}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'fixture','exec'),namespace)
        for slots in itertools.permutations(palette):
            m=fixtures.ThreeObjectBilliardsMetadataTests().candidate()
            for obj,slot in zip(m['simulation']['objects'],slots):obj['visual_profile']['material_slot']=slot
            scene=compile_resolved_scene(m,fixtures.ROOT)
            m['composition_rules']={'path':'composition.json','sha256':'hash'};m['assets']={'support_asset_id':'table'}
            _,materials=namespace['_fixture'](m,'billiards')
            info={'object_ids':[o['object_id'] for o in scene['objects']],
                'runtime_material':[[o['material'][k] for k in ('mass_kg','contact_friction','contact_restitution')] for o in scene['objects']],
                'inertia_diagonal_kg_m2':[[.001]*3]*3}
            objects,_=_compact_objects('billiards',m,scene,info,'table',{'table':palette})
            self.assertEqual(materials,[o['visual']['material_template']['source_object_name'] for o in objects])
            self.assertEqual(materials,[palette[s] for s in slots])
            self.assertEqual(build_fixture_payload('billiards_three_object_v1',m,scene)['physical']['support_dynamics'],
                             scene['adapter_payload']['backend']['billiards_rules']['support_dynamics'])
            self.assertEqual(_solver_contract('billiards_three_object_v1',m,scene)['solver_iterations'],180)

    def test_shared_render_core_is_bound_and_reuse_detects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);script=root/'render_three_object_billiards_scene.py';helper=root/'specialized_render_evidence.py';core=root/'specialized_sphere_rendering.py'
            for p in (script,helper,core):p.write_text('original')
            implementation={k:implementation_file_binding(root,p) for k,p in [('renderer',script),('render_evidence',helper),('sphere_render_core',core)]}
            metadata={'render':{'evidence_contract':'physweep_specialized_render_evidence_v2'},'implementation':implementation}
            self.assertTrue(implementation_is_reusable(root,metadata,script))
            self.assertTrue(render_record_implementation_is_reusable(root,{'implementation':implementation},script))
            core.write_text('changed')
            self.assertFalse(implementation_is_reusable(root,metadata,script))
            self.assertFalse(render_record_implementation_is_reusable(root,{'implementation':implementation},script))
