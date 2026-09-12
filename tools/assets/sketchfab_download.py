"""Shared transport and attribution for reviewed Sketchfab assets."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.assets.sketchfab_policy import noai_declared, require_glb_download
from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import write_json


def request_json(url: str, token: str, attempts: int = 5) -> dict[str, Any]:
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            headers={"Authorization": f"Token {token}", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")



def download_file(url: str, output: Path, *, user_agent: str, timeout: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=output.parent) as temporary:
            temporary_path = Path(temporary.name)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                shutil.copyfileobj(response, temporary)
        # Close and flush before publishing, including on Windows.
        temporary_path.replace(output)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def download_candidate(
    candidate: dict[str, Any],
    *,
    token: str,
    policy: dict[str, Any],
    output_root: Path,
    overwrite: bool,
    category_field: str,
    role_field: str,
    request_json_fn: Callable[[str, str], dict[str, Any]],
    download_file_fn: Callable[[str, Path], None],
) -> dict[str, Any]:
    allowed = {str(value) for value in policy["allowed_license_slugs"]}
    uid = str(candidate["source_uid"])
    model = request_json_fn(f"https://api.sketchfab.com/v3/models/{uid}", token)
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
        download = require_glb_download(
            request_json_fn(f"https://api.sketchfab.com/v3/models/{uid}/download", token)
        )
        download_file_fn(str(download["url"]), archive)
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
        category_field: candidate[category_field],
        "status": status,
        "archive_kind": "glb",
        "archive_path": str(archive),
        "size_bytes": archive.stat().st_size,
        "sha256": attribution["archive_sha256"],
        "license": attribution["license"],
        "author": attribution["author"],
        "viewer_url": candidate["viewer_url"],
        role_field: candidate[role_field],
    }
