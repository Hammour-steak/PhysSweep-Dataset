#!/usr/bin/env python3
"""Audit canonical PhysSweep metadata object identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.dataset_contract.object_identity_contract import (
    OBJECT_IDENTITY_SCHEMA_VERSION,
    validate_object_identity,
)

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metadata_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("metadata.json") if path.is_file())


def audit(path: Path) -> dict[str, Any]:
    metadata = load_json(path)
    had_contract = isinstance(metadata.get("object_identity"), dict)
    if not had_contract:
        raise ValueError("metadata has no object_identity contract")
    result = validate_object_identity(metadata)
    return {
        "path": str(path),
        "had_contract": had_contract,
        **result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    paths = metadata_paths(args.input.resolve())
    if not paths:
        raise SystemExit(f"no metadata.json found under {args.input}")
    records = []
    errors = []
    for path in paths:
        try:
            records.append(audit(path))
        except Exception as exc:  # noqa: BLE001 - audit must report every file
            errors.append({"path": str(path), "error": str(exc)})
    report = {
        "schema_version": "physweep_object_identity_audit_v1",
        "contract_schema_version": OBJECT_IDENTITY_SCHEMA_VERSION,
        "input": str(args.input.resolve()),
        "metadata_count": len(paths),
        "valid_count": len(records),
        "error_count": len(errors),
        "records": records,
        "errors": errors,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
