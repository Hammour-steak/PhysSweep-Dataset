import copy
import tempfile
import unittest
from pathlib import Path
from tools.core.json_io import write_json
from tools.sampling.derive_physics_sweep import load_sweep_config,derive_one,normalize_canonical_base
from tools.dataset_contract.three_object_group import validate_group_inputs
from tests.three_object_fixtures import scene

ROOT=Path(__file__).resolve().parents[1]


def group(parent,base_path,root):
    config_path=ROOT/'configs/three_object_physics_sweep.json'
    config=load_sweep_config(config_path)
    # Pin the same portable configuration inside this test's isolated root.
    local=root/'config.json';write_json(local,config)
    members=[]
    for axis in config['axes']:
        for level in range(5):
            if level==2 and axis!='mass_kg':continue
            m=derive_one(parent,base_path,root,config,local,axis,level,{}, {},target_object_index=0)
            members.append(normalize_canonical_base(m) if level==2 else m)
    return members


class ThreeObjectSweepTests(unittest.TestCase):
    def test_base_at_sweep_domain_endpoint_is_rejected_without_clamping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=scene();parent['simulation']['objects'][0]['material']['contact_restitution']=0.8
            p=root/'parent.json';write_json(p,parent)
            with self.assertRaises(ValueError):group(parent,p,root)

    def test_exact_13_preserve_nonintervened_inputs_for_all_target_roles(self):
        for order in (['P','Q','R'],['Q','R','P'],['R','P','Q']):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);parent=scene(order);p=root/'parent.json';write_json(p,parent)
                members=group(parent,p,root)
                self.assertTrue(validate_group_inputs(parent,members,[0])['passed'])

    def test_solver_camera_and_non_target_mutations_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=scene();p=root/'parent.json';write_json(p,parent);members=group(parent,p,root)
            for mutation in ('solver','camera','body'):
                wrong=copy.deepcopy(members)
                if mutation=='solver':wrong[0]['simulation']['solver']['iterations']+=1
                elif mutation=='camera':wrong[0]['camera_request']['requested_view_family']='side_oblique'
                else:wrong[0]['simulation']['objects'][2]['material']['mass_kg']+=1
                with self.assertRaises(ValueError):validate_group_inputs(parent,wrong,[0])
            with self.assertRaises(ValueError):validate_group_inputs(parent,members,[0,1,2])
            with self.assertRaises(ValueError):validate_group_inputs(parent,members[:-1],[0])
