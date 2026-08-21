#!/usr/bin/env python3
"""Validate and atomically publish the active physical proxy catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from physical_proxy_catalog import (
    CATALOG_VERSION,
    load_catalog,
    read_jsonl,
    sha256,
    summarize_records,
    validate_catalog,
    validate_runtime_catalog,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = Path("configs/physical_proxy_pipeline.json")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def project_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"release path is outside the project: {path}") from exc


def build_manifest(root: Path, policy_path: Path) -> dict[str, Any]:
    policy = load_json(policy_path)
    catalog = policy["catalog"]
    records_path = project_path(root, catalog["records"])
    validation_path = project_path(root, catalog["validation"])
    records = read_jsonl(records_path)
    validation = load_json(validation_path)

    if validation.get("catalog_records_sha256") != sha256(records_path):
        raise ValueError("proxy validation was not produced for the current records")
    counts = validation.get("counts", {})
    if int(counts.get("failed", -1)) != 0:
        raise ValueError("proxy validation contains failures")
    if int(counts.get("tested", -1)) != int(counts.get("passed", -2)):
        raise ValueError("proxy validation is incomplete")

    manifest = {
        "version": CATALOG_VERSION,
        "policy_path": project_relative(root, policy_path),
        "policy_sha256": sha256(policy_path),
        "generator_path": project_relative(root, Path(__file__)),
        "generator_sha256": sha256(Path(__file__)),
        "records_path": project_relative(root, records_path),
        "records_sha256": sha256(records_path),
        "counts": summarize_records(records),
        "scope": policy["scope"],
        "validation": {
            "path": project_relative(root, validation_path),
            "sha256": sha256(validation_path),
            "version": validation["version"],
            "scope": validation["scope"],
            "counts": counts,
        },
    }
    validate_catalog(manifest, records, root)
    validate_runtime_catalog(manifest, records, root)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--policy", type=Path, default=POLICY_PATH)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="atomically replace the active manifest after all checks pass",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    policy_path = project_path(root, args.policy)
    policy = load_json(policy_path)
    manifest = build_manifest(root, policy_path)
    if args.promote:
        manifest_path = project_path(root, policy["catalog"]["manifest"])
        write_json(manifest_path, manifest)
        load_catalog(root, manifest_path, require_runtime_validation=True)
    print(json.dumps({"promoted": bool(args.promote), **manifest["counts"]}, indent=2))


if __name__ == "__main__":
    main()
