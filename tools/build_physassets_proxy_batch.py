#!/usr/bin/env python3
"""Resumable batch driver for PhysAssets primitive proxy generation."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_one(row: dict, root: Path, output: Path, worker: Path, timeout: int, force: bool) -> dict:
    asset_dir = output / str(row["sample_id"])
    result_path = asset_dir / "proxy.json"
    if result_path.exists() and not force:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        return {"sample_id": str(row["sample_id"]), "status": result["admission"], "cached": True, "timing_seconds": result.get("timing_seconds")}
    command = [
        sys.executable, str(worker), "--source", str(root / row["mesh_path"]),
        "--output", str(asset_dir), "--sample-id", str(row["sample_id"]),
        "--uid", row["objaverse_uid"], "--object-name", row.get("object_name", ""),
        "--material", row.get("material", ""),
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if completed.returncode != 0:
            return {"sample_id": str(row["sample_id"]), "status": "generation_error", "error": completed.stderr[-1200:], "timing_seconds": time.monotonic() - started}
        result = json.loads(result_path.read_text(encoding="utf-8"))
        return {"sample_id": str(row["sample_id"]), "status": result["admission"], "cached": False, "timing_seconds": time.monotonic() - started}
    except subprocess.TimeoutExpired:
        return {"sample_id": str(row["sample_id"]), "status": "timeout", "timing_seconds": time.monotonic() - started}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = [r for r in read_rows(args.index) if r.get("proxy_status") == "proxy_candidate"]
    rows.sort(key=lambda r: int(r["sample_id"]))
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)
    rows = rows[args.offset:]
    if args.limit is not None:
        rows = rows[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "batch_manifest.jsonl"
    worker = Path(__file__).with_name("generate_physassets_primitive_proxy.py")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, row, args.project_root, args.output, worker, args.timeout, args.force): row for row in rows}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            with manifest.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=True) + "\n")
            print(json.dumps(result, ensure_ascii=True), flush=True)
    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    summary = {"requested": len(rows), "counts": counts, "workers": args.workers, "timeout_seconds": args.timeout}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
