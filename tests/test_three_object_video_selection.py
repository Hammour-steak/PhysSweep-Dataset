import copy
import unittest
from tests.three_object_fixtures import scene
from tools.sampling.three_object_video_selection import build_video_plan,TEMPLATES


def candidates():
    values=[]
    for template in TEMPLATES:
        for index,role in enumerate(('P','Q','R')):
            m=scene();m['scene_id']=f'{template}_{role}'
            m['coverage']={'cell':{'template':template,'target_role':role,'host_class':('ground_flat','raised_flat')[index%2],
                'requested_camera_family':('front_oblique','side_oblique','rear_oblique')[index],'scale_by_role':dict.fromkeys(('P','Q','R'),'medium')}}
            m['appearance']['scene_visual']['environment_category']=('minimal','lab_studio','home_office')[index]
            m['appearance']['materials']['support_surface']={'record':{'asset_id':f'support_{index}'}}
            values.append(m)
    return values


class ThreeObjectVideoSelectionTests(unittest.TestCase):
    def test_order_independent_base_selection_with_explicit_same_template_reserves(self):
        base=candidates();plan=build_video_plan(base)
        self.assertEqual(plan,build_video_plan(list(reversed(base))))
        primary=[slot['candidate_order'][0] for slot in plan['slots']]
        self.assertEqual(len(primary),8);self.assertEqual(len(set(primary)),8)
        for slot in plan['slots']:
            self.assertLessEqual(len(slot['candidate_order']),4)
            self.assertTrue(all(scene_id.startswith(slot['template']+'_') for scene_id in slot['candidate_order']))
        bad=copy.deepcopy(base);bad[0]['sweep']={'kind':'sweep'}
        with self.assertRaisesRegex(ValueError,'base inputs only'):build_video_plan(bad)

    def test_missing_template_capacity_is_not_reallocated(self):
        base=[m for m in candidates() if not m['scene_id'].startswith('chain_transfer_') or m['scene_id'].endswith('_P')]
        plan=build_video_plan(base)
        empty=[slot for slot in plan['slots'] if not slot['candidate_order']]
        self.assertEqual([s['slot_id'] for s in empty],['chain_transfer_1'])
        self.assertEqual(len(plan['slots']),8)


if __name__=='__main__':unittest.main()
