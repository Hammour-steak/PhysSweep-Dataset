#!/usr/bin/env python3
"""Publish a fresh audited one-object source release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.release.one_object_source_release import publish_source_release


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--sweep-metadata-manifest", type=Path, required=True)
    parser.add_argument("--sweep-physics-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    release = publish_source_release(
        root=args.root,
        base_manifest_path=args.base_manifest,
        sweep_metadata_manifest_path=args.sweep_metadata_manifest,
        sweep_physics_manifest_path=args.sweep_physics_manifest,
        output=args.output,
    )
    print(json.dumps(release, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
