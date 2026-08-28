#!/usr/bin/env python3
"""Audit the active asset-ingestion release and its sampling handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.assets.physical_proxy_catalog import load_catalog, records_by_id
from tools.assets.scene_kit_compiler import validate_registry_counts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = Path("configs/asset_ingestion_contract.json")
TEXT_SUFFIXES = {".json", ".md", ".py", ".sh"}
FORBIDDEN_ACTIVE_REFERENCE = "archive" + "/config_history"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_active_references(root: Path, excluded: set[Path] | None = None) -> None:
    excluded = {path.resolve() for path in (excluded or set())}
    scan_roots = [root / "configs", root / "docs", root / "tools", root / "tests"]
    files = [root / "README.md"]
    files.extend(
        path
        for scan_root in scan_roots
        for path in scan_root.rglob("*")
        if path.is_file() and path.suffix in TEXT_SUFFIXES and path.resolve() not in excluded
    )
    stale = [
        path.relative_to(root).as_posix()
        for path in files
        if FORBIDDEN_ACTIVE_REFERENCE in path.read_text(encoding="utf-8")
    ]
    if stale:
        raise ValueError(f"active files reference deleted config history: {stale}")


def audit_contract(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_json(contract_path)
    provenance = {
        key: {
            "path": project_path(root, value["path"]),
            "sha256": str(value["sha256"]),
        }
        for key, value in contract.get("immutable_provenance", {}).items()
    }
    invalid_provenance = [
        key
        for key, value in provenance.items()
        if not value["path"].is_file()
        or sha256(value["path"]) != value["sha256"]
    ]
    if invalid_provenance:
        raise ValueError(f"immutable asset provenance changed: {invalid_provenance}")
    audit_active_references(
        root, {value["path"] for value in provenance.values()}
    )
    release_paths = {
        key: project_path(root, value)
        for key, value in contract["active_release"].items()
    }
    implementation_paths = {
        key: project_path(root, value)
        for key, value in contract["implementations"].items()
    }
    missing = [
        f"{kind}:{key}:{path}"
        for kind, paths in (
            ("release", release_paths),
            ("implementation", implementation_paths),
        )
        for key, path in paths.items()
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"asset-ingestion contract has missing paths: {missing}")

    bundle = load_json(release_paths["sampling_bundle"])
    matrix = load_json(release_paths["sampling_matrix"])
    expected_bundle_paths = {
        "object_profiles": "object_profiles",
        "object_visual_preflight": "object_visual_preflight",
        "object_visual_preflight_report": "object_visual_preflight_report",
        "object_visual_curation": "object_visual_curation",
        "object_visual_repairs": "object_visual_repairs",
        "asset_proxy_registry": "asset_proxy_registry",
        "physical_proxy_catalog": "physical_proxy_catalog",
        "scene_mesh_profiles": "scene_mesh_profiles",
        "support_mesh_profiles": "support_mesh_profiles",
        "environment_collision_proxies": "environment_collision_proxies",
        "environment_composition": "environment_composition",
    }
    mismatches = [
        key
        for key, bundle_key in expected_bundle_paths.items()
        if project_path(root, bundle[bundle_key]) != release_paths[key]
    ]
    if project_path(
        root, matrix["dependencies"]["generic_sampling_bundle"]
    ) != release_paths["sampling_bundle"]:
        mismatches.append("sampling_matrix.generic_sampling_bundle")
    if mismatches:
        raise ValueError(
            f"asset-ingestion and sampling releases disagree: {mismatches}"
        )

    registry = load_json(release_paths["asset_proxy_registry"])
    validate_registry_counts(registry)
    catalog_manifest, catalog_records = load_catalog(
        root,
        release_paths["physical_proxy_catalog"],
        require_runtime_validation=True,
    )
    catalog = records_by_id(catalog_records)
    enabled_registry_ids = {
        str(record["asset_id"])
        for record in registry["records"]
        if record["admission"].get("sampling_enabled", False)
    }
    missing_catalog = sorted(enabled_registry_ids - set(catalog))
    unready_registry = sorted(
        asset_id
        for asset_id in enabled_registry_ids & set(catalog)
        if not catalog[asset_id]["admission"]["sampling_ready"]
    )
    if missing_catalog or unready_registry:
        raise ValueError(
            "enabled registry assets lack ready physical proxies: "
            f"missing={missing_catalog}, unready={unready_registry}"
        )

    profiles = load_json(release_paths["object_profiles"])["profiles"]
    foreground_ids = {
        str(visual["asset_id"])
        for profile in profiles
        for visual in profile["visual_variants"]
    }
    missing_foreground = sorted(foreground_ids - set(catalog))
    unready_foreground = sorted(
        asset_id
        for asset_id in foreground_ids & set(catalog)
        if not catalog[asset_id]["admission"]["sampling_ready"]
    )
    if missing_foreground or unready_foreground:
        raise ValueError(
            "foreground profiles lack ready physical proxies: "
            f"missing={missing_foreground}, unready={unready_foreground}"
        )

    local_sketchfab_ids = {
        path.parent.name
        for path in (root / "assets/library/sketchfab").glob("**/model.glb")
    }
    catalog_sketchfab_ids = {
        str(record["asset_id"])
        for record in catalog_records
        if record["source"]["collection"] == "sketchfab"
    }
    if local_sketchfab_ids != catalog_sketchfab_ids:
        raise ValueError(
            "local Sketchfab inventory and catalog dispositions differ: "
            f"missing={sorted(local_sketchfab_ids - catalog_sketchfab_ids)}, "
            f"extra={sorted(catalog_sketchfab_ids - local_sketchfab_ids)}"
        )

    return {
        "contract_version": contract["version"],
        "sampling_bundle_version": bundle["version"],
        "sampling_matrix_version": matrix["version"],
        "catalog_records": len(catalog_records),
        "local_sketchfab_assets": len(local_sketchfab_ids),
        "foreground_profiles": len(profiles),
        "enabled_registry_assets": len(enabled_registry_ids),
        "status": "clean",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    contract_path = project_path(root, args.contract)
    print(json.dumps(audit_contract(root, contract_path), indent=2))


if __name__ == "__main__":
    main()
