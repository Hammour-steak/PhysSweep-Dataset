import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from tools.core.hashing import sha256_file,sha256_json
from tools.physics.resolved_simulation_scene import compile_resolved_scene
from tools.sampling.sample_three_object_marble import load_marble_rules,build_three_object_marble_scene
from tools.sampling.released_object_sources import localize_marble_source_rows
from tools.sampling.derive_physics_sweep import load_sweep_config
from tools.sampling.three_object_sweeps import derive_group
from tools.sampling.three_object_fingerprints import physics_fingerprint
from tools.dataset_contract.three_object_group import validate_group_inputs
from tools.motion_rules.three_object.marble import validate_marble_contract
ROOT=Path(__file__).resolve().parents[1]

class ThreeObjectMarbleMetadataTests(unittest.TestCase):
    def candidate(self,target='P',source_kind='base'):
        temp=tempfile.TemporaryDirectory(dir=ROOT);self.addCleanup(temp.cleanup);directory=Path(temp.name)
        mesh=directory/'source.obj';mesh.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n');digest=sha256_file(mesh)
        backend=ROOT/'configs/marble_run_backend.json';candidate=json.loads((ROOT/'configs/candidates/marble_run_v1.json').read_text())
        dynamic=candidate['dynamic_object'];fixture=candidate['fixture'];radius=dynamic['radius_m']
        m={'schema_version':'physweep_marble_run_scene_v1','scene_id':'unit_source','sweep':{'kind':source_kind},'admission':{'passed':True},
           'semantics':{'dynamic_object_count':1,'scene_family':'marble_run'},'physics':{'profile':'early_release_chain',
           'backend_config':{'path':str(backend),'sha256':sha256_file(backend)},'engine':{k:candidate['physics'][k] for k in ('solver_iterations','deterministic_overlapping_pairs','restitution_velocity_threshold_m_s','enable_cone_friction','use_split_impulse')},
           'fixture':{'mesh_components':[{'id':'track_1','collision':{'path':str(mesh),'sha256':digest},'source_path':str(mesh),'source_sha256':digest,'mesh_scale':[.003]*3,'base_position_m':[0.,0.,1.55],'base_orientation_quaternion_xyzw':fixture['source_to_world_orientation_quaternion_xyzw']}],
                      'analytic_colliders':copy.deepcopy(fixture['analytic_colliders']),'mesh_material':fixture['material'],'analytic_material':fixture['catch_material']},'quality':{'old_passed':True}},
           'simulation':{'time':{'duration_s':4.,'output_fps':24,'simulation_hz':3840,'frame_count':97},'world':{'gravity_m_s2':[0.,0.,-9.81]},
              'objects':[{'object_id':'marble','body_model':'rigid_body','is_dynamic':True,'semantic_type':'marble','collision_proxy':{'type':'sphere','radius_m':radius},'material':dynamic['material'],
                          'initial_state':dynamic['initial_state'],'visual':{'shape':'sphere','radius_m':radius,'color_rgba':dynamic['color_rgba']}}]}}
        original_mesh=m['physics']['fixture']['mesh_components'][0]
        m['physics']['fixture']['mesh_components']=[{**copy.deepcopy(original_mesh),'id':part['id'],'base_position_m':copy.deepcopy(part['base_position_m'])} for part in fixture['mesh_components']]
        sources=[{'source':{'scene_id':f'source_{i}'},'metadata':copy.deepcopy(m)} for i in range(3)]
        sources=localize_marble_source_rows(sources,ROOT,ROOT,directory/'localized');rules=load_marble_rules(ROOT,ROOT)
        roles=[target]+[r for r in ('P','Q','R') if r!=target]
        with patch('tools.sampling.sample_three_object_marble.choose_specialized_environment',return_value={'room':{'half_extent_m':4.2},'role':'gritty_low_priority'}):
            result=build_three_object_marble_scene(data_root=ROOT,host_source=sources[0],object_sources=sources,rules=rules,scene_id='marble_'+target,
                role_order=roles,palette_order=['blue','green','amber'],parameters=rules['config']['initial_limits']['parameter_cases'][0],
                seeds={'physics':1,'appearance':2,'camera':3},requested_view='track_front',background_profile='lab_storage')
        return result,directory

    def test_all_roles_and_thirteen_members_preserve_source_materials(self):
        config_path=ROOT/'configs/three_object_marble_physics_sweep.json';config=load_sweep_config(config_path)
        for role in ('P','Q','R'):
            m,directory=self.candidate(role);p=directory/'metadata.json';p.write_text(json.dumps(m))
            self.assertNotIn('admission',m);self.assertNotIn('quality',m['physics']);self.assertNotIn('camera',m)
            scene=compile_resolved_scene(m,ROOT);self.assertEqual(scene['backend_binding']['adapter_id'],'marble_run_three_object_v1')
            self.assertEqual(scene['time']['simulation_hz'],7680);self.assertTrue(all(len(o['material'])==7 for o in scene['objects']))
            text=m['object_identity']['text'];self.assertEqual(text['template_version'],'physweep_object_caption_v10')
            for word in ('priority','collide','transfer','rolling','catch tray'):self.assertNotIn(word,text['caption'])
            self.assertEqual(m['three_object']['roles'][role],'object_a');self.assertEqual(len(text['object_mentions']),3)
            members=derive_group(m,p,ROOT,config,config_path,{},{});self.assertEqual(len(members),13)
            bad=copy.deepcopy(members);bad[0]['simulation']['time']['simulation_hz']*=2
            with self.assertRaises(ValueError):validate_group_inputs(m,bad,[0])

    def test_fixture_initial_state_and_schema_cannot_be_silently_changed(self):
        m,_=self.candidate();bad=copy.deepcopy(m);bad['physics']['fixture']['mesh_material']['contact_friction']=.8
        with self.assertRaisesRegex(ValueError,'fixture'):validate_marble_contract(bad)
        bad=copy.deepcopy(m);bad['simulation']['objects'][1]['initial_state']['linear_velocity_m_s'][0]=.1
        with self.assertRaisesRegex(ValueError,'initial state'):validate_marble_contract(bad)
        bad=copy.deepcopy(m);bad['three_object']['template']['id']='chain_transfer';c=bad['three_object'];c['contract_sha256']=sha256_json({k:v for k,v in c.items() if k!='contract_sha256'})
        with self.assertRaisesRegex(ValueError,'ordered-contact'):validate_marble_contract(bad)
        with self.assertRaisesRegex(ValueError,'base generation sources'):self.candidate(source_kind='sweep')

    def test_content_signature_ignores_copied_paths_labels_and_execution(self):
        m,_=self.candidate('P');changed,_=self.candidate('Q')
        self.assertEqual(physics_fingerprint(m,root=ROOT),physics_fingerprint(changed,root=ROOT))
        changed['simulation']['time']['simulation_hz']*=2;changed['physics']['engine']['solver_iterations']*=2;changed['render']['samples']*=2
        self.assertEqual(physics_fingerprint(m,root=ROOT),physics_fingerprint(changed,root=ROOT))
        changed['simulation']['objects'][0]['material']['mass_kg']*=2
        self.assertNotEqual(physics_fingerprint(m,root=ROOT),physics_fingerprint(changed,root=ROOT))

    def test_localized_geometry_hash_is_checked_on_compile(self):
        m,_=self.candidate();p=ROOT/m['physics']['fixture']['mesh_components'][0]['collision']['path'];p.write_text('changed')
        with self.assertRaisesRegex(ValueError,'collision mesh hash'):compile_resolved_scene(m,ROOT)
