"""Small checked-process runner for deterministic orchestration tools."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_checked(command: list[str], root: Path) -> None:
    """Run a command and include a bounded output tail in failures."""

    completed = subprocess.run(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise RuntimeError(f"command failed: {' '.join(command)}\n{tail}")
