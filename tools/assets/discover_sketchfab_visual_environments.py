#!/usr/bin/env python3
"""Discover reviewable Sketchfab visual-environment candidates."""

from __future__ import annotations

import argparse
import io
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


API_ROOT = "https://api.sketchfab.com/v3"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def request_bytes(url: str, *, attempts: int = 5) -> bytes:
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json,image/*",
                    "User-Agent": "physweep-visual-environment-curation/1.0",
                },
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
        except (TimeoutError, urllib.error.URLError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def request_json(url: str) -> dict[str, Any]:
    return json.loads(request_bytes(url).decode("utf-8"))


def noai_declared(model: dict[str, Any]) -> bool:
    fields = [model.get("name"), model.get("description")]
    fields.extend(tag.get("name") for tag in model.get("tags", []) if isinstance(tag, dict))
    return "noai" in " ".join(str(value or "") for value in fields).lower().replace("-", "")


def largest_thumbnail(model: dict[str, Any]) -> str | None:
    images = (model.get("thumbnails") or {}).get("images") or []
    if not images:
        return None
    return str(max(images, key=lambda item: int(item.get("width", 0))).get("url") or "") or None


def license_slug(license_record: dict[str, Any]) -> str:
    slug = str(license_record.get("slug") or "")
    if slug:
        return slug
    return {
        "CC Attribution": "by",
        "CC0 Public Domain": "cc0",
    }.get(str(license_record.get("label") or ""), "unknown")


def candidate_from_model(
    model: dict[str, Any],
    *,
    environment_category: str,
    query: str,
) -> dict[str, Any]:
    license_record = model.get("license") or {}
    user = model.get("user") or {}
    archives = model.get("archives") or {}
    return {
        "environment_category": environment_category,
        "query": query,
        "source_uid": model.get("uid"),
        "name": model.get("name"),
        "viewer_url": model.get("viewerUrl"),
        "description": str(model.get("description") or "")[:1200],
        "face_count": model.get("faceCount"),
        "vertex_count": model.get("vertexCount"),
        "like_count": model.get("likeCount"),
        "view_count": model.get("viewCount"),
        "animation_count": model.get("animationCount"),
        "is_downloadable": bool(model.get("isDownloadable")),
        "license": {
            "slug": license_slug(license_record),
            "label": license_record.get("label"),
            "url": license_record.get("url"),
        },
        "author": {
            "uid": user.get("uid"),
            "username": user.get("username"),
            "display_name": user.get("displayName"),
        },
        "archives": archives,
        "tags": [tag.get("name") for tag in model.get("tags", []) if isinstance(tag, dict)],
        "thumbnail_url": largest_thumbnail(model),
        "noai_detected": noai_declared(model),
    }


def search_query(query: str, count: int) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {
            "type": "models",
            "q": query,
            "downloadable": "true",
            "sort_by": "-likeCount",
            "count": count,
        }
    )
    payload = request_json(f"{API_ROOT}/search?{params}")
    return list(payload.get("results", []))


def admitted(candidate: dict[str, Any], policy: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not candidate["is_downloadable"]:
        reasons.append("not_downloadable")
    if candidate["license"]["slug"] not in set(policy["allowed_license_slugs"]):
        reasons.append("license_not_allowed")
    if policy.get("reject_noai", True) and candidate["noai_detected"]:
        reasons.append("noai")
    faces = int(candidate.get("face_count") or 0)
    if faces < int(policy.get("min_face_count", 0)):
        reasons.append("too_few_faces")
    if faces > int(policy.get("max_face_count", 10**12)):
        reasons.append("too_many_faces")
    if not candidate.get("thumbnail_url"):
        reasons.append("missing_thumbnail")
    return not reasons, reasons


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


def download_thumbnail(candidate: dict[str, Any], output: Path) -> Path:
    path = output / candidate["environment_category"] / f"{candidate['source_uid']}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        image = Image.open(io.BytesIO(request_bytes(candidate["thumbnail_url"]))).convert("RGB")
        image.thumbnail((960, 720), Image.Resampling.LANCZOS)
        image.save(path, quality=92)
    return path


def fit_text(draw: ImageDraw.ImageDraw, text: str, width: int) -> str:
    text = " ".join(text.split())
    while text and draw.textbbox((0, 0), text)[2] > width:
        text = text[:-1]
    return text.rstrip() + ("..." if text else "")


def make_sheet(candidates: list[dict[str, Any]], output: Path, columns: int = 4) -> None:
    if not candidates:
        return
    tile_w, tile_h, label_h = 360, 270, 64
    rows = math.ceil(len(candidates) / columns)
    canvas = Image.new("RGB", (columns * tile_w, rows * (tile_h + label_h)), (28, 29, 31))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, candidate in enumerate(candidates):
        x = (index % columns) * tile_w
        y = (index // columns) * (tile_h + label_h)
        source = Image.open(candidate["thumbnail_path"]).convert("RGB")
        source.thumbnail((tile_w, tile_h), Image.Resampling.LANCZOS)
        px = x + (tile_w - source.width) // 2
        py = y + (tile_h - source.height) // 2
        canvas.paste(source, (px, py))
        lines = [
            f"{index + 1:02d} {candidate['environment_category']} | {candidate['source_uid'][:8]}",
            fit_text(draw, str(candidate["name"]), tile_w - 12),
            f"faces={candidate['face_count']} likes={candidate['like_count']} lic={candidate['license']['slug']}",
        ]
        for line_index, line in enumerate(lines):
            draw.text((x + 6, y + tile_h + 5 + line_index * 17), line, font=font, fill=(235, 235, 235))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-query", type=int, default=12)
    parser.add_argument("--terms-per-category", type=int, default=1)
    parser.add_argument("--preview-per-category", type=int, default=5)
    args = parser.parse_args()

    query_manifest = read_json(args.queries)
    policy = query_manifest["policy"]
    search_rows: list[tuple[str, str, dict[str, Any]]] = []
    for entry in query_manifest["queries"]:
        for query in entry["search_terms"][: args.terms_per_category]:
            for model in search_query(query, args.per_query):
                search_rows.append((entry["environment_category"], query, model))
        print("searched", entry["environment_category"], flush=True)

    unique_uids = sorted({str(row[2].get("uid")) for row in search_rows})

    records: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for category, query, model in search_rows:
        uid = str(model.get("uid"))
        if not uid or (category, uid) in seen_pairs:
            continue
        seen_pairs.add((category, uid))
        candidate = candidate_from_model(model, environment_category=category, query=query)
        candidate["auto_admitted"], candidate["auto_rejection_reasons"] = admitted(candidate, policy)
        records.append(candidate)

    admitted_records = [record for record in records if record["auto_admitted"]]
    by_category: dict[str, list[dict[str, Any]]] = {}
    for record in admitted_records:
        by_category.setdefault(record["environment_category"], []).append(record)
    preview_records: list[dict[str, Any]] = []
    for category, values in by_category.items():
        values.sort(key=lambda value: (-int(value.get("like_count") or 0), int(value.get("face_count") or 0)))
        review_values = values[: args.preview_per_category]
        for candidate in review_values:
            try:
                candidate["thumbnail_path"] = str(download_thumbnail(candidate, args.output / "thumbnails"))
                preview_records.append(candidate)
            except Exception as exc:  # pylint: disable=broad-except
                candidate["auto_rejection_reasons"].append(f"thumbnail_download_failed:{exc}")
        make_sheet(
            [value for value in review_values if value.get("thumbnail_path")],
            args.output / "contact_sheets" / f"{safe_name(category)}.jpg",
        )
    make_sheet(preview_records, args.output / "contact_sheets" / "all_candidates.jpg")

    report = {
        "version": "physweep_visual_environment_discovery_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "query_manifest": str(args.queries),
        "policy": policy,
        "records": records,
        "summary": {
            "search_rows": len(search_rows),
            "unique_models": len(unique_uids),
            "auto_admitted": len(admitted_records),
            "auto_rejected": len(records) - len(admitted_records),
            "previewed": len(preview_records),
            "admitted_by_category": {key: len(value) for key, value in sorted(by_category.items())},
        },
    }
    write_json(args.output / "discovery_report.json", report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
