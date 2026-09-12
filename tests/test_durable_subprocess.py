"""Kill the coordinator while its child runs, then recover the same job."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from tools.core.durable_subprocess import process_identity

ROOT = Path(__file__).resolve().parents[1]
COORDINATOR = """from pathlib import Path
import sys
from tools.core.durable_subprocess import run_stage
p=Path(sys.argv[1])
run_stage(p, lambda: p.with_suffix('.done').exists(), interval=.02)
"""


@unittest.skipUnless(sys.platform == 'linux', 'Linux process/lock recovery contract')
class DurableSubprocessTests(unittest.TestCase):
    def launch(self, path):
        return subprocess.Popen([sys.executable, '-B', '-c', COORDINATOR, str(path)],
                                cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def wait_for(self, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.02)
        self.fail('timed out waiting for job state')

    def test_live_child_and_completed_restart_do_not_duplicate_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); spec = root / 'publish.json'
            code = """from pathlib import Path
import time
p=Path('publish')
with p.with_suffix('.count').open('a') as f: f.write('run\\n')
while not p.with_suffix('.release').exists(): time.sleep(.02)
p.with_suffix('.done').touch()
"""
            spec.write_text(json.dumps({'command': [sys.executable, '-B', '-c', code], 'cwd': str(root)}))
            first = self.launch(spec)
            second = None
            try:
                self.wait_for(lambda: (root / 'publish.count').exists())
                identity = json.loads(spec.with_suffix('.state.json').read_text())['identity']
                first.kill(); first.wait(timeout=5)
                self.assertEqual(process_identity(identity['pid']), identity)
                second = self.launch(spec)
                time.sleep(.15)
                self.assertIsNone(second.poll())
                self.assertEqual((root / 'publish.count').read_text().splitlines(), ['run'])
                (root / 'publish.release').touch()
                self.assertEqual(second.wait(timeout=10), 0)
                third = self.launch(spec)
                self.assertEqual(third.wait(timeout=10), 0)
                self.assertEqual((root / 'publish.count').read_text().splitlines(), ['run'])
            finally:
                (root / 'publish.release').touch()
                for process in (first, second):
                    if process is not None and process.poll() is None:
                        process.kill(); process.wait(timeout=5)

    def test_failed_partial_stage_retries_on_next_invocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); spec = root / 'publish.json'
            code = """from pathlib import Path
import sys
p=Path('publish.partial')
if not p.exists(): p.write_text('preserve me'); sys.exit(3)
assert p.read_text()=='preserve me'
Path('publish.done').touch()
"""
            spec.write_text(json.dumps({'command': [sys.executable, '-B', '-c', code], 'cwd': str(root)}))
            self.assertNotEqual(self.launch(spec).wait(timeout=10), 0)
            self.assertEqual(self.launch(spec).wait(timeout=10), 0)
            self.assertEqual((root / 'publish.partial').read_text(), 'preserve me')
