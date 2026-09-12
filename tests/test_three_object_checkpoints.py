import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.core.hashing import sha256_file


@unittest.skipUnless(importlib.util.find_spec('pybullet'),'requires physics runtime')
class ThreeObjectCheckpointTests(unittest.TestCase):
    def test_resume_rejects_same_content_at_a_different_source_path(self):
        from tools.cli.generate_three_object_dataset import simulation
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);output=root/'physics';output.mkdir()
            source=root/'metadata.json';source.write_text(json.dumps({'scene_id':'candidate'}))
            record={'scene_id':'candidate','audit_passed':True,'failed_checks':[]}
            for key in ('metadata','resolved_scene','trajectory','audit'):
                path=source if key=='metadata' else root/key
                if key!='metadata':path.write_bytes(b'unchanged artifact')
                record[key+'_path']=str(path);record[key+'_sha256']=sha256_file(path)
            (output/'simulation_record.json').write_text(json.dumps(record))
            with patch('tools.cli.generate_three_object_dataset.dispatch_simulation',side_effect=AssertionError('must reuse')):
                self.assertEqual(simulation(source,output,root,True),record)
                other=root/'another_metadata.json';other.write_bytes(source.read_bytes())
                with self.assertRaisesRegex(ValueError,'path changed'):simulation(other,output,root,True)
                (root/'audit').write_bytes(b'corrupted audit')
                with self.assertRaisesRegex(ValueError,'audit changed'):simulation(source,output,root,True)




if __name__=='__main__':unittest.main()
