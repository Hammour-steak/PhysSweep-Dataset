import copy
import unittest
from collections import Counter
from pathlib import Path
from tests.three_object_fixtures import scene,source
from tools.sampling.three_object_fingerprints import physics_fingerprint
from tools.sampling.three_object_coverage import build_plan
from tools.sampling.three_object_sampling_request import load_pilot_rules


class ThreeObjectCoverageTests(unittest.TestCase):
    def test_execution_changes_do_not_create_unique_physical_scenes(self):
        base=scene();changed=copy.deepcopy(base)
        changed['simulation']['solver']['iterations']+=30
        changed['simulation']['time'].update(simulation_hz=960,output_fps=48,frame_count=193)
        self.assertEqual(physics_fingerprint(changed),physics_fingerprint(base))
        changed['simulation']['world']['gravity_m_s2'][2]=-8.0
        self.assertNotEqual(physics_fingerprint(changed),physics_fingerprint(base))

    def test_physical_uniqueness_ignores_identity_and_appearance_but_not_material(self):
        base=scene();before=physics_fingerprint(base)
        changed=copy.deepcopy(base)
        changed['simulation']['objects'].reverse()
        for i,obj in enumerate(changed['simulation']['objects']):
            obj['object_id']='renamed_'+str(i)
            obj['visual_profile']['id']='different_texture'
        self.assertEqual(physics_fingerprint(changed),before)
        changed['simulation']['objects'][0]['material']['mass_kg']*=2
        self.assertNotEqual(physics_fingerprint(changed),before)

    def test_pilot_allocates_120_explicit_cells_across_templates_hosts_and_targets(self):
        objects=[]
        for scale in ('small','medium','large'):
            for index in range(3):
                row=source(index);row['metadata']['semantic_sampling']['five_dimensions']['foreground_object']['scale_bin']=scale
                objects.append(row)
        hosts=[]
        for host_class in ('ground_flat','raised_flat'):
            for category in ('simple','workshop','outdoor'):
                row=source();row['metadata']['simulation']['support']['scene_class']=host_class
                row['metadata']['appearance']['scene_visual']['environment_category']=category;hosts.append(row)
        plan=build_plan({'objects':objects,'hosts':hosts,'summary':{}},load_pilot_rules())
        self.assertEqual(plan['candidate_count'],120)
        self.assertEqual(len({c['cell_id'] for c in plan['cells']}),120)
        self.assertEqual(set(Counter((c['template'],c['host_class'],c['target_role']) for c in plan['cells']).values()),{5})
        self.assertEqual(set(Counter(c['requested_camera_family'] for c in plan['cells']).values()),{40})
        self.assertEqual(len(plan['unallocated_scale_patterns']),22)

    def test_inclined_plan_freezes_camera_bounded_initial_domain(self):
        objects=[]
        for scale in ('small','medium'):
            for index in range(3):
                row=source(index)
                row['metadata']['semantic_sampling']['five_dimensions']['foreground_object']['scale_bin']=scale
                objects.append(row)
        hosts=[]
        for host_class in ('ground_feature','raised_feature'):
            row=source()
            row['metadata']['simulation']['support']['scene_class']=host_class
            row['metadata']['appearance']['scene_visual']['environment_category']='simple'
            hosts.append(row)
        rules=load_pilot_rules(matrix_path=Path('three_object_d5br2_sampling_matrix.json'))
        plan=build_plan({'objects':objects,'hosts':hosts,'summary':{}},rules)
        self.assertEqual(plan['candidate_count'],24)
        self.assertEqual(set(Counter(cell['requested_camera_family'] for cell in plan['cells']).values()),{8})
        for cell in plan['cells']:
            self.assertEqual(cell['initial_speed_m_s'],.8)
            expected=1.7 if cell['scale_by_role']['P']=='small' else 1.5
            self.assertEqual(cell['spacing_ratio'],expected)
        self.assertEqual(rules['motion']['initial_state']['inclined_material_policy']['rolling_friction'],.018)


if __name__=='__main__':unittest.main()
