#!/usr/bin/env python3
"""Download Sketchfab visual environments with immutable provenance."""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json
from tools.assets import sketchfab_download


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def request_json(url: str, token: str, attempts: int = 5) -> dict[str, Any]:
    return sketchfab_download.request_json(url, token, attempts=attempts)


def download_file(url: str, output: Path) -> None:
    sketchfab_download.download_file(
        url, output, user_agent="physweep-visual-environment-curation/1.0", timeout=900,
    )


def download_candidate(
    candidate: dict[str, Any],
    *,
    token: str,
    policy: dict[str, Any],
    output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    return sketchfab_download.download_candidate(
        candidate, token=token, policy=policy, output_root=output_root,
        overwrite=overwrite, category_field="environment_category", role_field="intended_role",
        request_json_fn=request_json, download_file_fn=download_file,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "assets/library/sketchfab/visual_environments_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    token = os.environ.get("SKETCHFAB_API_TOKEN")
    if not token:
        raise SystemExit("SKETCHFAB_API_TOKEN is required")

    manifest = load_json(args.manifest)
    policy = manifest["policy"]

    def worker(candidate: dict[str, Any]) -> dict[str, Any]:
        return download_candidate(
            candidate,
            token=token,
            policy=policy,
            output_root=args.output_root,
            overwrite=bool(args.overwrite),
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(worker, manifest["candidates"]))
    for record in records:
        print(record["status"], record["candidate_id"], record["size_bytes"])

    output = {
        "version": str(
            manifest.get(
                "download_manifest_version",
                "physweep_sketchfab_visual_environment_downloads_v1",
            )
        ),
        "source_manifest": str(args.manifest),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
        "summary": {"requested": len(records), "available": len(records)},
    }
    write_json(args.output_root / "download_manifest.json", output)
    print("manifest", args.output_root / "download_manifest.json")


if __name__ == "__main__":
    main()
