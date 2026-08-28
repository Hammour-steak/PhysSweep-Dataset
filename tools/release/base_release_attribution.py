#!/usr/bin/env python3
"""Asset identity, appearance-template, and attribution helpers for base releases."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from tools.release.audit_release_provenance import sha256

ATTRIBUTION_SCHEMA = "physweep_asset_attribution_manifest_v2"


def source_record_hash(value: Any) -> str:
    payload = (
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def collect_asset_ids(value: Any, destination: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "asset_id" and isinstance(item, str):
                destination.add(item)
            collect_asset_ids(item, destination)
    elif isinstance(value, list):
        for item in value:
            collect_asset_ids(item, destination)


def load_billiards_templates(project_root: Path) -> dict[str, dict[str, str]]:
    path = project_root / "configs" / "asset_scene_composition.json"
    catalog = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, str]] = {}
    for record in catalog["records"]:
        templates = record.get("component_policy", {}).get("material_template_objects")
        if isinstance(templates, dict):
            result[str(record["asset_id"])] = {
                str(object_id): str(object_name)
                for object_id, object_name in templates.items()
            }
    return result


def build_attribution(
    project_root: Path,
    used_asset_ids: set[str],
    used_hdri: dict[str, str],
) -> dict[str, Any]:
    if not used_asset_ids and not used_hdri:
        return {
            "schema_version": ATTRIBUTION_SCHEMA,
            "record_count": 0,
            "counts": {
                "physassets_objaverse": 0,
                "sketchfab": 0,
                "sketchfab_support_surface": 0,
                "polyhaven_material": 0,
                "polyhaven_hdri": 0,
                "redistribution_verification_required": 0,
            },
            "records": [],
        }
    third_party = json.loads(
        (project_root / "assets" / "THIRD_PARTY_ASSETS.json").read_text(encoding="utf-8")
    )
    sketchfab = {
        str(item["asset_id"]): item for item in third_party["sketchfab"]["records"]
    }
    profiles = json.loads(
        (
            project_root / "configs" / "physassets_core_object_profiles_source.json"
        ).read_text(encoding="utf-8")
    )
    physassets: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for profile in profiles["profiles"]:
        for variant in profile["visual_variants"]:
            physassets[str(variant["asset_id"])] = (profile, variant)
    material_catalog = json.loads(
        (project_root / "assets" / "manifests" / "polyhaven_render_library.json").read_text(
            encoding="utf-8"
        )
    )
    materials = {str(item["asset_id"]): item for item in material_catalog["assets"]}
    hdri_catalog = json.loads(
        (project_root / "assets" / "manifests" / "hdri_admission.json").read_text(
            encoding="utf-8"
        )
    )
    hdris = {str(item["name"]): item for item in hdri_catalog["records"]}

    physasset_records = []
    sketchfab_records = []
    support_surface_records = []
    material_records = []
    unknown = []
    for asset_id in sorted(used_asset_ids):
        if asset_id.startswith("physassets_"):
            if asset_id not in physassets:
                unknown.append(asset_id)
                continue
            profile, variant = physassets[asset_id]
            physasset_records.append(
                {
                    "asset_id": asset_id,
                    "kind": "dynamic_object",
                    "provider": "PhysAssets/Objaverse",
                    "semantic_profile_id": str(profile["id"]),
                    "content_sha256": str(variant["sha256"]),
                    "license": copy.deepcopy(variant["license"]),
                    "source_record_sha256": source_record_hash(variant),
                }
            )
        elif asset_id.startswith("sketchfab_"):
            if asset_id not in sketchfab:
                unknown.append(asset_id)
                continue
            item = sketchfab[asset_id]
            source_path = project_root / str(item["asset_path"])
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            sketchfab_records.append(
                {
                    "asset_id": asset_id,
                    "kind": str(item["family"]),
                    "provider": "Sketchfab",
                    "name": str(item["name"]),
                    "author": copy.deepcopy(item["author"]),
                    "license": copy.deepcopy(item["license"]),
                    "viewer_url": str(item["viewer_url"]),
                    "content_sha256": sha256(source_path),
                    "source_record_sha256": source_record_hash(item),
                }
            )
        elif asset_id.startswith("support_"):
            attribution_path = (
                project_root
                / "assets"
                / "library"
                / "sketchfab"
                / "support_surfaces_v2"
                / asset_id
                / "attribution.json"
            )
            if not attribution_path.is_file():
                unknown.append(asset_id)
                continue
            item = json.loads(attribution_path.read_text(encoding="utf-8"))
            if str(item["candidate_id"]) != asset_id:
                raise ValueError(f"support attribution identity differs: {asset_id}")
            source_path = project_root / str(item["archive_path"])
            if sha256(source_path) != str(item["archive_sha256"]):
                raise ValueError(f"support source hash differs: {asset_id}")
            support_surface_records.append(
                {
                    "asset_id": asset_id,
                    "kind": "support_surface",
                    "provider": "Sketchfab",
                    "name": str(item["name"]),
                    "author": copy.deepcopy(item["author"]),
                    "license": copy.deepcopy(item["license"]),
                    "viewer_url": str(item["viewer_url"]),
                    "content_sha256": str(item["archive_sha256"]),
                    "source_record_sha256": source_record_hash(item),
                }
            )
        elif asset_id in materials:
            item = materials[asset_id]
            source_name = re.sub(r"_\d+k$", "", asset_id)
            material_root = project_root / str(item["path"])
            content_files = []
            for content_path in sorted(path for path in material_root.rglob("*") if path.is_file()):
                if content_path.is_symlink():
                    raise ValueError(f"material content is a symlink: {content_path}")
                content_files.append(
                    {
                        "path": content_path.relative_to(material_root).as_posix(),
                        "sha256": sha256(content_path),
                    }
                )
            if not content_files:
                raise ValueError(f"material has no content files: {asset_id}")
            material_records.append(
                {
                    "asset_id": asset_id,
                    "kind": "material",
                    "provider": "Poly Haven",
                    "license": "CC0",
                    "source_url": f"https://polyhaven.com/a/{source_name}",
                    "content_files": content_files,
                    "source_record_sha256": source_record_hash(item),
                }
            )
        else:
            unknown.append(asset_id)
    if unknown:
        raise ValueError(f"unattributed asset ids: {unknown}")

    hdri_records = []
    for name, expected_hash in sorted(used_hdri.items()):
        if name not in hdris or str(hdris[name]["sha256"]) != expected_hash:
            raise ValueError(f"unattributed or hash-mismatched HDRI: {name}")
        item = hdris[name]
        hdri_records.append(
            {
                "asset_id": name,
                "kind": "hdri",
                "provider": "Poly Haven",
                "license": "CC0",
                "source_url": f"https://polyhaven.com/a/{name}",
                "content_sha256": expected_hash,
                "source_record_sha256": source_record_hash(item),
            }
        )

    records = [
        *physasset_records,
        *sketchfab_records,
        *support_surface_records,
        *material_records,
        *hdri_records,
    ]
    return {
        "schema_version": ATTRIBUTION_SCHEMA,
        "record_count": len(records),
        "counts": {
            "physassets_objaverse": len(physasset_records),
            "sketchfab": len(sketchfab_records),
            "sketchfab_support_surface": len(support_surface_records),
            "polyhaven_material": len(material_records),
            "polyhaven_hdri": len(hdri_records),
            "redistribution_verification_required": sum(
                bool(item["license"].get("verification_required_before_redistribution"))
                for item in physasset_records
            ),
        },
        "records": records,
    }
