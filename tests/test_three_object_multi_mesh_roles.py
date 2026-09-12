import unittest
from collections import Counter
from pathlib import Path

from tests.three_object_fixtures import source
from tools.motion_rules.three_object.motion import validate_motion_contract
from tools.sampling.sample_three_object_base import build_three_object_scene
from tools.sampling.three_object_coverage import build_plan,_multi_mesh_initial_parameters
from tools.sampling.three_object_sampling_request import load_pilot_rules


PAIR_MATRIX=Path('three_object_d6e_multi_mesh_pair_sampling_matrix.json')
ROLES=('P','Q','R')
ASSET_GEOMETRY={
    'sketchfab_046b94a7775e496aac26ad8c699496e6':('cylinder','small'),
    'sketchfab_1551616939cc4da7a9d731dfddd4090f':('cylinder','small'),
    'sketchfab_4ae035ea89ea40bbaa82403b9c36afab':('cuboid','large'),
    'sketchfab_640b0c8287274629a7f4ff3ce74a5999':('cylinder','small'),
    'sketchfab_76fa80950b1f48a8ad6a8441fe443241':('cuboid','large'),
    'sketchfab_b0902ff83cf946c984f9f7e0197ddb7f':('cuboid','medium'),
    'sketchfab_beb69bc08a8f487ab8c5207fb155cbf2':('cylinder','large'),
    'sketchfab_dae37cc4869a4155b6999efe26df710c':('cuboid','large'),
    'sketchfab_ebb7c4102dc94ef7ba14d9c5df43c448':('cylinder','large'),
    'sketchfab_f4d11e48cc52479c802ec2cf0d629f68':('cuboid','small'),
}


def primitive_row(index,shape,visual,scale='medium',family='generic'):
    row=source(index);obj=row['metadata']['simulation']['objects'][0]
    size=[.18,.24,.20] if shape=='cuboid' else ([.18,.18,.24] if shape=='cylinder' else [.18,.18,.18])
    obj['geometry']={'type':shape,'size_m':size};obj['collision_profile']={'type':shape,'dimensions_m':size}
    obj['semantic_type']=shape;obj['visual_profile']['id']=visual
    row['metadata']['semantic_sampling']['five_dimensions']['foreground_object']['scale_bin']=scale
    row['source']['source_family']=family
    return row


def pool_fixture(rules):
    generic=rules['matrix']['required_generic_visual_asset_ids'];objects=[]
    for index,visual in enumerate(generic):
        shape='sphere' if index<22 else ('cuboid' if index<57 else 'cylinder')
        objects.append(primitive_row(index,shape,visual))
    for offset,(visual,(shape,scale)) in enumerate(ASSET_GEOMETRY.items(),start=len(objects)):
        objects.append(primitive_row(offset,shape,visual,scale,'asset'))
    hosts=[]
    for host_index,host in enumerate(('ground_flat','raised_flat')):
        for category_index,category in enumerate(('home_office','lab_studio')):
            row=source(1000+10*host_index+category_index)
            row['metadata']['simulation']['support']['scene_class']=host
            row['metadata']['appearance']['scene_visual']['environment_category']=category
            hosts.append(row)
    return {'objects':objects,'hosts':hosts,'summary':{'fixture':True}}


class ThreeObjectMultiMeshRolesTest(unittest.TestCase):
    def test_pair_only_repair_keeps_full_visual_coverage(self):
        rules=load_pilot_rules(matrix_path=PAIR_MATRIX);plan=build_plan(pool_fixture(rules),rules)
        self.assertEqual(Counter(cell['template'] for cell in plan['cells']),{'pair_control':42})
        self.assertEqual(Counter(cell['motion_axis'] for cell in plan['cells']),{'world_x':21,'world_y':21})
        self.assertEqual(Counter(cell['target_role'] for cell in plan['cells']),{'P':14,'Q':14,'R':14})
        visuals={value for cell in plan['cells'] for role,value in cell['required_visual_asset_by_role'].items()
            if cell['required_source_family_by_role'][role]=='generic'}
        self.assertEqual(visuals,set(rules['matrix']['required_generic_visual_asset_ids']))

    def test_pair_only_nonball_driver_gap_uses_released_friction(self):
        rules=load_pilot_rules(matrix_path=PAIR_MATRIX)
        objects={role:primitive_row(index,shape,f'visual_{index}')['metadata']['simulation']['objects'][0]
            for index,(role,shape) in enumerate(zip(ROLES,('cuboid','cylinder','cuboid')))}
        objects['P']['material']['contact_friction']=.6
        construction=build_plan(pool_fixture(rules),rules)['construction']
        speed,spacing,_,time_s=_multi_mesh_initial_parameters(objects,{'template':'pair_control','motion_axis':'world_x'},rules,construction)
        self.assertAlmostEqual(time_s,.22);self.assertGreater(speed,.6*9.81*.22)
        self.assertGreaterEqual(spacing,1.5);self.assertLessEqual(spacing,4.)

    def test_axis_aligned_pair_admits_nonball_P_Q_and_R(self):
        rules=load_pilot_rules(matrix_path=PAIR_MATRIX)
        objects=[primitive_row(index,shape,f'visual_{index}') for index,shape in enumerate(('cuboid','cylinder','cuboid'))]
        scene=build_three_object_scene(host_source=source(99),object_sources=objects,rules=rules,
            scene_id='multi_mesh_pair_control',template_id='pair_control',speed_m_s=.8,spacing_ratio=2.)
        self.assertEqual([obj['geometry']['type'] for obj in scene['simulation']['objects']],['cuboid','cylinder','cuboid'])
        validate_motion_contract(scene)

    def test_sphere_contact_estimators_remain_sphere_only(self):
        rules=load_pilot_rules(matrix_path=PAIR_MATRIX)
        objects=[primitive_row(index,shape,f'visual_{index}') for index,shape in enumerate(('cuboid','sphere','sphere'))]
        for template in ('successive_hits','converging_hits'):
            with self.assertRaisesRegex(ValueError,'shape contradicts'):
                build_three_object_scene(host_source=source(99),object_sources=objects,rules=rules,
                    scene_id='rejected_'+template,template_id=template)


if __name__=='__main__':unittest.main()
