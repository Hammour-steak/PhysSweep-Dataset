#!/usr/bin/env python3
"""Download reviewed Sketchfab support assets with license provenance."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def request_json(url: str, token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Token {token}", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "physweep-support-asset-curation/1.0"}
    )
    with tempfile.NamedTemporaryFile(delete=False, dir=output.parent) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                shutil.copyfileobj(response, temporary)
            temporary_path.replace(output)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise


def noai_declared(model: dict[str, Any]) -> bool:
    searchable = [str(model.get("name", "")), str(model.get("description", ""))]
    searchable.extend(str(tag.get("name", "")) for tag in model.get("tags", []))
    return "noai" in " ".join(searchable).lower().replace("-", "")


def choose_glb(downloads: dict[str, Any]) -> dict[str, Any]:
    glb = downloads.get("glb")
    if not isinstance(glb, dict) or not glb.get("url"):
        raise ValueError("candidate has no downloadable GLB archive")
    return glb


def download_candidate(
    candidate: dict[str, Any],
    *,
    token: str,
    policy: dict[str, Any],
    output_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    allowed = {str(value) for value in policy["allowed_license_slugs"]}
    uid = str(candidate["source_uid"])
    model = request_json(f"https://api.sketchfab.com/v3/models/{uid}", token)
    license_record = model.get("license") or {}
    license_slug = str(license_record.get("slug", ""))
    if license_slug not in allowed:
        raise ValueError(f"disallowed license for {uid}: {license_slug}")
    if bool(policy.get("reject_noai", True)) and noai_declared(model):
        raise ValueError(f"NoAI candidate rejected: {uid}")
    if not bool(model.get("isDownloadable", False)):
        raise ValueError(f"candidate is not downloadable: {uid}")

    asset_dir = output_root / str(candidate["candidate_id"])
    archive = asset_dir / "model.glb"
    status = "exists"
    if overwrite or not archive.exists():
        download = choose_glb(
            request_json(f"https://api.sketchfab.com/v3/models/{uid}/download", token)
        )
        download_file(str(download["url"]), archive)
        status = "downloaded"
    author = model.get("user") or {}
    attribution = {
        **candidate,
        "source_name": model.get("name"),
        "author": {
            "uid": author.get("uid"),
            "username": author.get("username"),
            "display_name": author.get("displayName"),
            "profile_url": author.get("profileUrl"),
        },
        "license": {
            "slug": license_slug,
            "label": license_record.get("label"),
            "url": license_record.get("url"),
        },
        "is_downloadable": bool(model.get("isDownloadable")),
        "noai_detected": noai_declared(model),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "archive_path": str(archive),
        "archive_sha256": sha256(archive),
        "archive_size_bytes": archive.stat().st_size,
    }
    write_json(asset_dir / "attribution.json", attribution)
    return {
        "candidate_id": candidate["candidate_id"],
        "source_uid": uid,
        "name": model.get("name"),
        "semantic_category": candidate["semantic_category"],
        "status": status,
        "archive_kind": "glb",
        "archive_path": str(archive),
        "size_bytes": archive.stat().st_size,
        "sha256": attribution["archive_sha256"],
        "license": attribution["license"],
        "author": attribution["author"],
        "viewer_url": candidate["viewer_url"],
        "intended_proxy": candidate["intended_proxy"],
    }


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
        default=PROJECT_ROOT / "assets/library/sketchfab/support_ramps",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
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
        "version": "physweep_sketchfab_support_downloads_v1",
        "source_manifest": str(args.manifest),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records": records,
        "summary": {"requested": len(records), "available": len(records)},
    }
    write_json(args.output_root / "download_manifest.json", output)
    print("manifest", args.output_root / "download_manifest.json")


if __name__ == "__main__":
    main()
