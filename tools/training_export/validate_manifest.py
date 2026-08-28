#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from tools.dataset_contract.schema import validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a PhysSweep training manifest")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = args.project_root / manifest
    result = validate_manifest(
        manifest.resolve(),
        args.project_root.resolve(),
        check_files=args.check_files,
    )
    print(json.dumps({"passed": True, **result}, indent=2))


if __name__ == "__main__":
    main()
