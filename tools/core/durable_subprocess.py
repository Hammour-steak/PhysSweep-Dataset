"""Resume a Linux job across coordinator restarts without duplicating its child.

The child inherits an already locked file descriptor, records its identity, and
executes the command in place. Thus even the spawn/receipt window is protected.
The caller supplies artifact verification; a process exit alone is not success.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable

from tools.core.json_io import write_json_atomic


def process_identity(pid: int) -> dict | None:
    root = Path('/proc') / str(pid)
    try:
        fields = (root / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': fields[19],
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    except FileNotFoundError:
        return None


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execute(spec_path: Path, lock_fd: int, expected_digest: str) -> None:
    if _digest(spec_path) != expected_digest:
        raise ValueError('stage specification changed before execution')
    spec = json.loads(spec_path.read_text())
    os.fstat(lock_fd)
    os.set_inheritable(lock_fd, True)
    write_json_atomic(spec_path.with_suffix('.state.json'), {
        'spec_sha256': expected_digest, 'identity': process_identity(os.getpid()),
        'started_at': time.time(), 'command': spec['command'],
    })
    os.chdir(spec['cwd'])
    os.execvpe(spec['command'][0], spec['command'], dict(os.environ, **spec.get('env', {})))


def run_stage(spec_path: Path, is_complete: Callable[[], bool], *,
              interval: float = 1.0, on_wait: Callable[[], None] | None = None) -> None:
    """Attach to a live stage, or resume incomplete artifacts once per invocation.

    Specifications must be immutable. A failed child is reported to the caller;
    a subsequent coordinator invocation may retry using the command's resume mode.
    """
    import fcntl

    spec_path = spec_path.resolve()
    digest = _digest(spec_path)
    observed_job = False
    child = None
    while True:
        if _digest(spec_path) != digest:
            raise ValueError('stage specification changed')
        with spec_path.with_suffix('.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                observed_job = True
                state_path = spec_path.with_suffix('.state.json')
                if state_path.exists():
                    state = json.loads(state_path.read_text())
                    if state['spec_sha256'] != digest:
                        raise ValueError('running stage belongs to another specification')
            else:
                if child is not None:
                    child.wait()
                if is_complete():
                    return
                if observed_job:
                    raise RuntimeError(f'stage exited with incomplete or invalid artifacts: {spec_path}')
                # The child owns the same locked open-file description from the
                # instant it is spawned, even if this coordinator then disappears.
                child = subprocess.Popen(
                    [sys.executable, '-B', '-m', __name__, '--run', str(spec_path),
                     str(lock.fileno()), digest],
                    pass_fds=(lock.fileno(),), start_new_session=True,
                )
                observed_job = True
        if on_wait is not None:
            on_wait()
        time.sleep(interval)


if __name__ == '__main__':
    if len(sys.argv) != 5 or sys.argv[1] != '--run':
        raise SystemExit('internal usage: --run SPEC LOCK_FD SHA256')
    _execute(Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
