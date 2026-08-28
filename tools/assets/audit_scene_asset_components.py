#!/usr/bin/env python3
"""Run normalized Blender component inspection for every non-dynamic asset."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--registry",
        type=Path,
        default=PROJECT_ROOT / "configs/asset_proxy_registry.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--blender",
        type=Path,
        default=PROJECT_ROOT / "runtime/blender-3.4.0-linux-x64/blender",
    )
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def inspect_one(
    root: Path,
    registry_path: Path,
    output: Path,
    blender: Path,
    script: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    inventory_path = output / "inventories" / f"{record['asset_id']}.json"
    log_path = output / "logs" / f"{record['asset_id']}.log"
    command = [
        str(blender),
        "--background",
        "--python",
        str(script),
        "--",
        "--root",
        str(root),
        "--registry",
        str(registry_path),
        "--asset-id",
        str(record["asset_id"]),
        "--output",
        str(inventory_path),
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    output_exists = inventory_path.exists()
    inventory = load_json(inventory_path) if output_exists else None
    ok = completed.returncode == 0 and inventory is not None
    return {
        "asset_id": record["asset_id"],
        "name": record["name"],
        "asset_role": record["asset_role"],
        "ok": ok,
        "returncode": completed.returncode,
        "wall_time_s": round(time.perf_counter() - started, 6),
        "inventory_path": str(inventory_path),
        "log_path": str(log_path),
        "mesh_component_count": inventory["mesh_component_count"] if inventory else None,
        "total_triangle_count": inventory["total_triangle_count"] if inventory else None,
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    registry_path = args.registry.resolve()
    output = args.output.resolve()
    registry = load_json(registry_path)
    records = [
        record for record in registry["records"] if record["asset_role"] != "dynamic_object"
    ]
    script = root / "tools/assets/inspect_scene_asset_components.py"
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        results = list(
            executor.map(
                lambda record: inspect_one(
                    root,
                    registry_path,
                    output,
                    args.blender.resolve(),
                    script,
                    record,
                ),
                records,
            )
        )
    summary = {
        "schema_version": "physweep_scene_asset_component_audit_v1",
        "asset_count": len(results),
        "success_count": sum(result["ok"] for result in results),
        "failure_count": sum(not result["ok"] for result in results),
        "wall_time_s": round(time.perf_counter() - started, 6),
        "records": results,
    }
    write_json(output / "manifest.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    failures = [result for result in results if not result["ok"]]
    if failures:
        print(json.dumps(failures, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
