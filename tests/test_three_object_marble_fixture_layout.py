import copy,unittest
from tests import test_three_object_marble_metadata as fixtures
from tools.core.hashing import sha256_json
from tools.core.marble_fixture_layout import construct_marble_fixture,validate_marble_fixture_construction
from tools.motion_rules.three_object.marble import validate_marble_contract

class MarbleFixtureLayoutTests(unittest.TestCase):
    def candidate(self):
        instance=fixtures.ThreeObjectMarbleMetadataTests();self.addCleanup(instance.doCleanups);return instance.candidate()[0]

    def test_only_terminal_track_and_catch_placement_change(self):
        m=self.candidate();c=m['three_object'];self.assertEqual(c['schema_version'],'physweep_three_object_marble_motion_contract_v2')
        binding=c['fixture_construction'];original=binding['source_fixture'];fixture=m['physics']['fixture']
        self.assertEqual(fixture['mesh_components'],original['mesh_components'][:3]);self.assertEqual(len(original['mesh_components']),4)
        for before,after in zip(original['analytic_colliders'],fixture['analytic_colliders']):
            expected=copy.deepcopy(before)
            if before['id']!='safety_floor':expected['position_m']=[a+b for a,b in zip(before['position_m'],[-.36,0.,.09])]
            self.assertEqual(after,expected)
        self.assertEqual(m['source_binding']['fixture_construction'],binding)
        validate_marble_fixture_construction(fixture,binding)

    def test_source_transform_cannot_be_changed_by_only_rehashing_output(self):
        m=self.candidate();c=m['three_object'];bad=copy.deepcopy(m)
        bad['physics']['fixture']['analytic_colliders'][0]['half_extents_m'][0]*=2
        b=bad['three_object'];b['fixture_sha256']=sha256_json(bad['physics']['fixture']);b['contract_sha256']=sha256_json({k:v for k,v in b.items() if k!='contract_sha256'})
        with self.assertRaisesRegex(ValueError,'source transformation'):validate_marble_contract(bad)
        binding=copy.deepcopy(c['fixture_construction']);binding['source_fixture']['mesh_material']['contact_friction']+=.01
        with self.assertRaisesRegex(ValueError,'original.*hash'):validate_marble_fixture_construction(m['physics']['fixture'],binding)
        layout=copy.deepcopy(c['fixture_construction']['layout']);layout['catch_translation_m'][0]=-.2
        with self.assertRaisesRegex(ValueError,'translation'):construct_marble_fixture(c['fixture_construction']['source_fixture'],layout)

    def test_v1_full_source_fixture_still_valid_but_cannot_hide_transform(self):
        m=self.candidate();c=m['three_object'];binding=c['fixture_construction'];m['physics']['fixture']=copy.deepcopy(binding['source_fixture'])
        c['schema_version']='physweep_three_object_marble_motion_contract_v1';c.pop('fixture_construction');m['source_binding'].pop('fixture_construction')
        c['fixture_sha256']=sha256_json(m['physics']['fixture']);c['contract_sha256']=sha256_json({k:v for k,v in c.items() if k!='contract_sha256'})
        validate_marble_contract(m)
        c['fixture_construction']=binding;c['contract_sha256']=sha256_json({k:v for k,v in c.items() if k!='contract_sha256'})
        with self.assertRaisesRegex(ValueError,'v1 cannot'):validate_marble_contract(m)
