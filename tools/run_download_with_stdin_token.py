#!/usr/bin/env python3
"""Run the visual-environment downloader with a token read from stdin."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(project_root / "tools/download_sketchfab_visual_environments.py"),
    ]
    command.extend(sys.argv[1:])
    environment = dict(os.environ)
    if not any(argument in {"-h", "--help"} for argument in sys.argv[1:]):
        token = sys.stdin.readline().strip()
        if not token:
            raise SystemExit("missing token on stdin")
        environment["SKETCHFAB_API_TOKEN"] = token
    result = subprocess.run(command, env=environment, stdin=subprocess.DEVNULL, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
