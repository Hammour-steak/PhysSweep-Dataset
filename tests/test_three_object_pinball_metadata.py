import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools.core.hashing import sha256_json
from tools.physics.generate_passive_pinball_scene import build_metadata
from tools.physics.resolved_simulation_scene import compile_resolved_scene
from tools.sampling.sample_three_object_pinball import load_pinball_rules,build_three_object_pinball_scene
from tools.sampling.derive_physics_sweep import load_sweep_config
from tools.sampling.three_object_sweeps import derive_group
from tools.sampling.three_object_fingerprints import physics_fingerprint
from tools.dataset_contract.three_object_group import validate_group_inputs
from tools.motion_rules.three_object.pinball import validate_pinball_contract
ROOT=Path(__file__).resolve().parents[1]


class ThreeObjectPinballMetadataTests(unittest.TestCase):
    def candidate(self,target='P',source_kind='base'):
        path=ROOT/'configs/passive_pinball_backend.json';config=json.loads(path.read_text())
        m=build_metadata(ROOT,ROOT/'outputs/unit_pinball',path,config,1,'dense_pinfield_descent','unit_source')
        m['admission']={'passed':True};m['sweep']={'kind':source_kind}
        sources=[{'source':{'scene_id':f'source_{i}'},'metadata':copy.deepcopy(m)} for i in range(3)]
        rules=load_pinball_rules(ROOT,ROOT);roles=[target]+[r for r in ('P','Q','R') if r!=target]
        with patch('tools.sampling.sample_three_object_pinball.choose_specialized_environment',return_value={'room':{'half_extent_m':4.2},'role':'gritty_low_priority'}):
            return build_three_object_pinball_scene(data_root=ROOT,host_source=sources[0],object_sources=sources,rules=rules,scene_id='pinball_'+target,
                role_order=roles,palette_order=['blue','green','amber'],parameters=rules['config']['initial_limits']['parameter_cases'][0],
                seeds={'physics':1,'appearance':2,'camera':3},requested_view='pinfield_front',background_profile='lab_storage')

    def test_all_target_roles_keep_source_materials_caption_and_thirteen_members(self):
        config_path=ROOT/'configs/three_object_pinball_physics_sweep.json';config=load_sweep_config(config_path)
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            for role in ('P','Q','R'):
                m=self.candidate(role);p=Path(temporary)/(m['scene_id']+'.json');p.write_text(json.dumps(m))
                self.assertNotIn('admission',m);self.assertNotIn('quality',m['physics']);self.assertNotIn('camera',m)
                scene=compile_resolved_scene(m,ROOT);self.assertEqual(scene['backend_binding']['adapter_id'],'passive_pinball_three_object_v1')
                self.assertTrue(all(len(o['material'])==7 for o in scene['objects']))
                self.assertEqual(scene['objects'][0]['material']['rolling_friction'],.0025)
                text=m['object_identity']['text'];self.assertNotIn('priority',text['caption']);self.assertNotIn('parallel',text['caption'])
                self.assertIn('starts moving down the board',text['caption']);self.assertEqual(text['template_version'],'physweep_object_caption_v9')
                self.assertEqual(m['three_object']['roles'][role],'object_a')
                members=derive_group(m,p,ROOT,config,config_path,{},{});self.assertEqual(len(members),13)
                damaged=copy.deepcopy(members);damaged[0]['simulation']['time']['simulation_hz']*=2
                with self.assertRaises(ValueError):validate_group_inputs(m,damaged,[0])

    def test_fixture_and_initial_state_changes_fail_and_rendering_does_not_add_physics(self):
        m=self.candidate();bad=copy.deepcopy(m);bad['physics']['fixture']['material']['contact_friction']=.7
        with self.assertRaisesRegex(ValueError,'fixture changed'):validate_pinball_contract(bad)
        bad=copy.deepcopy(m);bad['simulation']['objects'][1]['initial_state']['linear_velocity_m_s'][0]=.1
        c=bad['three_object'];c['contract_sha256']=sha256_json({k:v for k,v in c.items() if k!='contract_sha256'})
        with self.assertRaisesRegex(ValueError,'initial state'):validate_pinball_contract(bad)
        changed=copy.deepcopy(m);changed['render']['samples']*=2;changed['physics']['engine']['solver_iterations']*=2
        self.assertEqual(physics_fingerprint(m,root=ROOT),physics_fingerprint(changed,root=ROOT))
        changed['simulation']['objects'][0]['material']['mass_kg']*=2
        self.assertNotEqual(physics_fingerprint(m,root=ROOT),physics_fingerprint(changed,root=ROOT))

    def test_sweep_variant_cannot_be_a_new_base_source(self):
        with self.assertRaisesRegex(ValueError,'base generation sources'):
            self.candidate(source_kind='sweep')
