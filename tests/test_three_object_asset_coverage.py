import copy
import unittest
from collections import Counter
from pathlib import Path
from tests.three_object_fixtures import source
from tools.sampling.three_object_coverage import build_plan,bound_object_pool
from tools.sampling.three_object_sampling_request import load_pilot_rules,validate_rules


def pool_fixture():
    objects=[]
    for shape in ('sphere','cuboid','cylinder'):
        for index in range(3):
            row=source(len(objects));obj=row['metadata']['simulation']['objects'][0]
            obj['geometry']['type']=shape;obj['visual_profile']['id']=shape+'_'+str(index)
            row['metadata']['semantic_sampling']['five_dimensions']['foreground_object']['scale_bin']='medium';objects.append(row)
    hosts=[]
    for host in ('ground_flat','raised_flat'):
        for category in ('home_office','lab_studio'):
            row=source();row['metadata']['simulation']['support']['scene_class']=host
            row['metadata']['appearance']['scene_visual']['environment_category']=category;hosts.append(row)
    return {'objects':objects,'hosts':hosts,'summary':{}}


class AssetCoverageTests(unittest.TestCase):
    def test_every_asset_has_both_prespecified_templates_and_stable_cells(self):
        pool=pool_fixture();rules=load_pilot_rules(matrix_path=Path('three_object_d5f_sampling_matrix.json'));rules['matrix']['candidate_budget']=18
        plan=build_plan(pool,rules);self.assertEqual(plan['candidate_count'],18)
        targets=Counter((c['required_visual_asset_by_role']['R'],c['template']) for c in plan['cells'])
        self.assertEqual(len(targets),18);self.assertEqual(set(targets.values()),{1})
        self.assertEqual(set(Counter(c['target_role'] for c in plan['cells']).values()),{6})
        changed=copy.deepcopy(pool);changed['objects'].reverse();changed['hosts'].reverse()
        self.assertEqual(plan,build_plan(changed,rules))
        rules['matrix']['candidate_budget']=20
        with self.assertRaisesRegex(ValueError,'frozen candidate budget'):build_plan(pool,rules)

    def test_absent_target_never_falls_back_to_another_asset(self):
        rows=pool_fixture()['objects'];pools={('sphere','medium'):rows[:3]}
        cell={'shape_by_role':{'P':'sphere'},'scale_by_role':{'P':'medium'}}
        self.assertIs(bound_object_pool(pools,cell,'P'),pools[('sphere','medium')])
        cell['required_visual_asset_by_role']={'P':'missing'};self.assertEqual(bound_object_pool(pools,cell,'P'),[])
        cell['required_visual_asset_by_role']={'P':'sphere_1'};self.assertEqual(bound_object_pool(pools,cell,'P'),[rows[1]])
        cell['required_visual_asset_by_role']={'unknown':'sphere_1'}
        with self.assertRaises(ValueError):bound_object_pool(pools,cell,'P')

    def test_missing_scale_is_reported_without_substitution(self):
        pool=pool_fixture();pool['objects'][-1]['metadata']['semantic_sampling']['five_dimensions']['foreground_object']['scale_bin']='large'
        rules=load_pilot_rules(matrix_path=Path('three_object_d5f_sampling_matrix.json'));rules['matrix']['candidate_budget']=16
        plan=build_plan(pool,rules)
        self.assertEqual(plan['unavailable_medium_visual_asset_ids'],['cylinder_2'])
        self.assertNotIn('cylinder_2',plan['required_visual_asset_ids'])

    def test_unknown_mode_is_rejected(self):
        rules=load_pilot_rules();rules['matrix']['coverage_mode']='typo'
        with self.assertRaisesRegex(ValueError,'coverage mode'):validate_rules(**{k:rules[k] for k in ('matrix','motion','scene','sweep','dataset')})


if __name__=='__main__':unittest.main()
