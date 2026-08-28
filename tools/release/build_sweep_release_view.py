#!/usr/bin/env python3
"""Build or verify the canonical derived-only PhysSweep sweep release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.release.base_release_view import (
    DEFAULT_RELEASE_ROOT,
    PipelineSpec,
    one_object_release_roots,
)
from tools.release.sweep_release_view import build_view, verify_view


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-project-root", type=Path)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=DEFAULT_RELEASE_ROOT,
        help="Canonical output root; must be named one_object.",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--pipeline",
        nargs=5,
        action="append",
        metavar=("NAME", "SOURCE_SCHEMA", "PROJECT_ROOT", "RENDER_ROOT", "MASK_ROOT"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_root, output = one_object_release_roots(args.release_root)
    if args.verify_only:
        result = verify_view(output, base_root=base_root)
    else:
        if args.release_project_root is None or args.release_manifest is None:
            raise SystemExit(
                "--release-project-root and --release-manifest are required when building"
            )
        specs = [
            PipelineSpec(name, schema, Path(root), Path(render), Path(masks))
            for name, schema, root, render, masks in (args.pipeline or [])
        ]
        result = build_view(
            release_project_root=args.release_project_root,
            release_manifest=args.release_manifest,
            base_root=base_root,
            output=output,
            pipeline_specs=specs,
            workers=args.workers,
            resume=args.resume,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
