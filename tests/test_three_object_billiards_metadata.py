import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from tools.core.hashing import sha256_file, sha256_json
from tools.motion_rules.three_object.billiards import validate_billiards_contract
from tools.sampling.sample_three_object_billiards import load_billiards_rules, build_three_object_billiards_scene
from tools.sampling.derive_physics_sweep import load_sweep_config
from tools.sampling.three_object_sweeps import derive_group
from tools.sampling.three_object_fingerprints import physics_fingerprint
from tools.physics.resolved_simulation_scene import compile_resolved_scene
from tools.dataset_contract.three_object_group import validate_group_inputs
from tools.dataset_contract.object_identity_contract import attach_object_identity

ROOT = Path(__file__).resolve().parents[1]


class ThreeObjectBilliardsMetadataTests(unittest.TestCase):
    def test_hdri_priority_is_not_a_room_description(self):
        scene=self.candidate()
        environment=scene['render']['environment'];environment.update(role='gritty_low_priority',tier='rare',room={'half_extent_m':3.4})
        attach_object_identity(scene)
        text=scene['object_identity']['text']
        self.assertTrue(text['caption'].startswith('In an indoor room, '))
        self.assertNotIn('priority',text['caption']);self.assertEqual(text['template_version'],'physweep_object_caption_v8')
        del environment['room'];attach_object_identity(scene)
        self.assertTrue(scene['object_identity']['text']['caption'].startswith('the billiard ball'))

    def candidate(self, target='P'):
        backend=ROOT/'configs/pybullet_backend.json';config=json.loads(backend.read_text());dynamics=config['billiards_rules']['ball_dynamics']
        fixture={'asset_id':'table','mesh':{'sha256':'a'*64,'scale':[1.,1.,1.],'base_position_m':[0.,0.,0.],
            'base_orientation_quaternion_xyzw':[0.,0.,0.,1.]},
            'target_support_frame':{'safe_surface':{'center_xy_m':[0.,0.],'size_xy_m':[2.04,.98],'z_m':.78}},
            'visual':{'path':'table.glb','sha256':'b'*64}}
        metadata={'schema_version':'physweep_billiards_scene_v4','semantics':{'dynamic_object_count':1},
            'physics':{'backend_config':{'path':'configs/pybullet_backend.json','sha256':sha256_file(backend)},
                'ball_radius_m':.028575,'ball_mass_kg':.17,'static_support_binding':fixture,'simulation_hz':6072,
                'runtime_material':{'mass_kg':.17,'contact_friction':dynamics['lateral_friction'],'contact_restitution':dynamics['restitution']}},
            'render':{'environment':{'name':'bound_room','strength':.3}},'sweep':{'kind':'base'},'admission':{'passed':True}}
        sources=[{'source':{'scene_id':str(i)},'metadata':copy.deepcopy(metadata)} for i in range(3)]
        order=[target]+[r for r in ('P','Q','R') if r!=target]
        with patch('tools.sampling.sample_three_object_billiards.refresh_billiards_fixture',side_effect=lambda _,m:copy.deepcopy(m)):
            return build_three_object_billiards_scene(data_root=ROOT,host_source=sources[0],object_sources=sources,
                rules=load_billiards_rules(ROOT),scene_id='billiards_three_test_'+target,role_order=order,
                material_slots=['object_ball_2','cue_ball','object_ball_1'],
                parameters={'speed_m_s':.5,'PQ_surface_gap_m':.12,'QR_surface_gap_m':.10,'lane_y_m':.02,'heading':1},
                seeds={'physics':1,'appearance':2,'camera':3},requested_view='side_oblique')

    def test_all_target_roles_resolve_explicitly_and_derive_exact_groups(self):
        config_path=ROOT/'configs/three_object_billiards_physics_sweep.json';config=load_sweep_config(config_path)
        for role in ('P','Q','R'):
            with self.subTest(role=role),tempfile.TemporaryDirectory(dir=ROOT) as directory:
                scene=self.candidate(role);path=Path(directory)/'candidate.json';path.write_text(json.dumps(scene))
                self.assertNotIn('admission',scene);self.assertNotIn('initial_states',scene['physics'])
                resolved=compile_resolved_scene(scene,ROOT)
                self.assertEqual(resolved['backend_binding']['adapter_id'],'billiards_three_object_v1')
                self.assertEqual(len(resolved['objects']),3)
                self.assertTrue(all(len(obj['material'])==7 for obj in resolved['objects']))
                members=derive_group(scene,path,ROOT,config,config_path,{}, {})
                self.assertEqual(len(members),13)
                self.assertEqual(scene['three_object']['roles'][role],'object_a')
                text=scene['object_identity']['text']['caption']
                self.assertIn('starts',text);self.assertIn('billiards table',text);self.assertNotIn('collide with',text)
                damaged=copy.deepcopy(members);damaged[0]['render']['environment']['strength']=.9
                with self.assertRaises(ValueError):validate_group_inputs(scene,damaged,[0])

    def test_rehashed_state_changes_fail_and_solver_changes_do_not_increase_diversity(self):
        scene=self.candidate();modified=copy.deepcopy(scene)
        modified['simulation']['objects'][1]['initial_state']['linear_velocity_m_s']=[.2,0.,0.]
        c=modified['three_object'];c['contract_sha256']=sha256_json({k:v for k,v in c.items() if k!='contract_sha256'})
        with self.assertRaisesRegex(ValueError,'initial state'):validate_billiards_contract(modified)
        other=copy.deepcopy(scene);other['physics']['simulation_hz']*=2
        self.assertEqual(physics_fingerprint(scene,root=ROOT),physics_fingerprint(other,root=ROOT))
        other['simulation']['objects'][0]['material']['mass_kg']*=2
        self.assertNotEqual(physics_fingerprint(scene,root=ROOT),physics_fingerprint(other,root=ROOT))
