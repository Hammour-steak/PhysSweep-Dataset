#!/usr/bin/env python3
"""Bind repaired visuals and build the complete per-object curation ledger."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json
from tools.assets.physical_proxy_catalog import load_catalog, records_by_id
from tools.assets.static_support_proxy import record_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--source-profiles",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--preflight",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--repairs",
        type=Path,
        default=Path("configs/object_visual_repairs.json"),
    )
    parser.add_argument(
        "--output-profiles",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-curation",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--profiles-version",
        default="physweep_physassets_core_object_profiles_v4",
    )
    parser.add_argument(
        "--curation-version",
        default="physweep_object_visual_curation_v8",
    )
    return parser.parse_args()


def resolved(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_proxy_evidence_sources(
    root: Path,
    source: dict[str, Any],
    policy: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    evidence_policy = policy["proxy_evidence"]
    core_path = root / str(evidence_policy["core_index"])
    overlay_root = root / str(evidence_policy["overlay_root"])
    catalog_path = root / str(evidence_policy["physical_proxy_catalog"])
    if not core_path.is_file() or not overlay_root.is_dir():
        raise FileNotFoundError("candidate proxy evidence source is missing")
    core_records = read_jsonl(core_path)
    core_by_sample = {str(record["sample_id"]): record for record in core_records}
    if len(core_by_sample) != len(core_records):
        raise ValueError("candidate proxy core index has duplicate sample ids")
    expected_samples = {
        str(profile["source_review"]["sample_id"])
        for profile in source["profiles"]
    }
    if set(core_by_sample) != expected_samples:
        raise ValueError("candidate proxy core index differs from source profiles")
    catalog_manifest, catalog_records = load_catalog(
        root, catalog_path, require_runtime_validation=True
    )
    validation_descriptor = catalog_manifest.get("validation")
    if not isinstance(validation_descriptor, dict):
        raise ValueError("candidate proxy catalog has no runtime validation")
    validation_path = root / str(validation_descriptor["path"])
    validation = load_json(validation_path)
    validation_records = [
        record
        for record in validation.get("records", [])
        if record.get("probe") == "analytic_drop"
    ]
    validation_by_asset = {
        str(record["asset_id"]): record for record in validation_records
    }
    if len(validation_by_asset) != len(validation_records):
        raise ValueError(
            "candidate proxy analytic-drop validation has duplicate asset ids"
        )
    required_views = [str(value) for value in evidence_policy["required_overlay_views"]]
    expected_sources = {
        "core_index": {
            "path": core_path.relative_to(root).as_posix(),
            "sha256": sha256(core_path),
            "record_count": len(core_records),
        },
        "overlay_root": {
            "path": overlay_root.relative_to(root).as_posix(),
            "required_views": required_views,
        },
        "physical_proxy_catalog": {
            "path": catalog_path.relative_to(root).as_posix(),
            "sha256": sha256(catalog_path),
            "records_sha256": str(catalog_manifest["records_sha256"]),
        },
        "physical_proxy_validation": {
            "path": validation_path.relative_to(root).as_posix(),
            "sha256": sha256(validation_path),
            "version": str(validation["version"]),
            "catalog_records_sha256": str(validation["catalog_records_sha256"]),
            "record_count": len(validation.get("records", [])),
        },
    }
    if report.get("proxy_evidence_sources") != expected_sources:
        raise ValueError("candidate proxy evidence source binding is stale")
    return {
        "policy": evidence_policy,
        "core_by_sample": core_by_sample,
        "overlay_root": overlay_root,
        "required_views": required_views,
        "catalog_manifest": catalog_manifest,
        "catalog_by_asset": records_by_id(catalog_records),
        "validation_by_asset": validation_by_asset,
    }


def validate_proxy_evidence_record(
    root: Path,
    profile: dict[str, Any],
    record: dict[str, Any],
    context: dict[str, Any],
) -> None:
    asset_id = str(record["visual_asset_id"])
    evidence = record.get("proxy_evidence", {})
    sample_id = str(profile["source_review"]["sample_id"])
    core = context["core_by_sample"].get(sample_id)
    if core is None:
        raise ValueError(f"candidate proxy core record is missing: {asset_id}")
    if (
        evidence.get("status") != "verified"
        or str(evidence.get("sample_id")) != sample_id
        or evidence.get("core_record_sha256") != record_sha256(core)
        or evidence.get("proxy_method") != str(core["method"])
        or evidence.get("blender_alignment_fit")
        != core["blender_alignment_fit"]
        or not math.isclose(
            float(evidence.get("proxy_to_visual_hull_volume_ratio", math.inf)),
            float(core["proxy_to_visual_hull_volume_ratio"]),
            rel_tol=1.0e-8,
            abs_tol=1.0e-9,
        )
    ):
        raise ValueError(f"candidate proxy core evidence is stale: {asset_id}")
    if (
        float(evidence["proxy_to_visual_hull_volume_ratio"])
        > float(context["policy"]["maximum_proxy_to_visual_hull_volume_ratio"])
        or not bool(evidence["blender_alignment_fit"]["passed"])
        or float(evidence["blender_alignment_fit"]["maximum_relative_error"])
        > float(context["policy"]["maximum_alignment_relative_error"])
    ):
        raise ValueError(f"candidate proxy evidence exceeds policy: {asset_id}")
    source_proxy = evidence.get("source_proxy", {})
    proxy_path = root / str(source_proxy.get("path", ""))
    if (
        str(source_proxy.get("path"))
        != str(profile["source_review"]["proxy_json"])
        or not proxy_path.is_file()
        or sha256(proxy_path) != str(source_proxy.get("sha256"))
    ):
        raise ValueError(f"candidate source proxy evidence is stale: {asset_id}")
    overlay_views = evidence.get("overlay_views", [])
    if (
        len(overlay_views) != len(context["required_views"])
        or {str(view.get("view")) for view in overlay_views}
        != set(context["required_views"])
    ):
        raise ValueError(f"candidate proxy overlay evidence is incomplete: {asset_id}")
    for view in overlay_views:
        path = root / str(view["path"])
        if not path.is_file() or sha256(path) != str(view["sha256"]):
            raise ValueError(f"candidate proxy overlay evidence mismatch: {asset_id}")
    catalog_record = context["catalog_by_asset"].get(asset_id)
    if catalog_record is None:
        raise ValueError(f"candidate proxy catalog record is missing: {asset_id}")
    colliders = catalog_record["proxy"]["colliders"]
    expected_catalog = {
        "asset_id": asset_id,
        "catalog_record_sha256": record_sha256(catalog_record),
        "catalog_records_sha256": str(
            context["catalog_manifest"]["records_sha256"]
        ),
        "representation": str(catalog_record["proxy"]["representation"]),
        "collider_shape": str(colliders[0]["shape"]),
        "collider_size_m": [float(value) for value in colliders[0]["size_m"]],
        "qa_grade": str(catalog_record["qa"]["grade"]),
        "catalog_visual": {
            "path": str(catalog_record["source"]["visual_path"]),
            "sha256": str(catalog_record["source"]["sha256"]),
        },
    }
    if evidence.get("physical_proxy_catalog") != expected_catalog:
        raise ValueError(f"candidate physical proxy evidence is stale: {asset_id}")
    probe = context["validation_by_asset"].get(asset_id)
    if probe is None:
        raise ValueError(f"candidate runtime proxy probe is missing: {asset_id}")
    expected_probe = {
        "record_sha256": record_sha256(probe),
        "probe": str(probe["probe"]),
        "passed": bool(probe["passed"]),
    }
    if evidence.get("runtime_probe") != expected_probe or not expected_probe["passed"]:
        raise ValueError(f"candidate runtime proxy probe is stale: {asset_id}")


def validated_repair_records(
    root: Path, repairs: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for recipe in repairs["records"]:
        asset_id = str(recipe["visual_asset_id"])
        if asset_id in result:
            raise ValueError(f"duplicate visual repair recipe: {asset_id}")
        output = root / str(recipe["output_path"])
        report_path = output.parent / "repair.json"
        if not output.is_file() or not report_path.is_file():
            raise FileNotFoundError(f"repair output is incomplete: {asset_id}")
        report = load_json(report_path)
        if report["visual_asset_id"] != asset_id:
            raise ValueError(f"repair report asset mismatch: {asset_id}")
        if report["source"] != {
            "path": str(recipe["source_path"]),
            "sha256": str(recipe["source_sha256"]),
        }:
            raise ValueError(f"repair report source mismatch: {asset_id}")
        expected_admitted = {
            "path": str(recipe["output_path"]),
            "sha256": sha256(output),
        }
        if report["admitted_visual"] != expected_admitted:
            raise ValueError(f"repair report output mismatch: {asset_id}")
        if report["operations"] != recipe["operations"]:
            raise ValueError(f"repair operations mismatch: {asset_id}")
        for view in report["verification"]["review_views"]:
            path = root / str(view["path"])
            if not path.is_file() or sha256(path) != str(view["sha256"]):
                raise ValueError(f"repair review evidence mismatch: {asset_id}")
        result[asset_id] = {"recipe": recipe, "report": report}
    return result


def validated_preflight_records(
    root: Path,
    source_path: Path,
    source: dict[str, Any],
    policy_path: Path,
    policy: dict[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    expected_source_path = source_path.relative_to(root).as_posix()
    if str(policy["source_profiles"]) != expected_source_path:
        raise ValueError("candidate preflight targets a different source profile file")
    report_path = root / str(policy["output_report"])
    if not report_path.is_file():
        raise FileNotFoundError("candidate preflight report is missing")
    report = load_json(report_path)
    if report.get("version") != str(policy["report_version"]):
        raise ValueError("unsupported candidate preflight report")
    if not bool(report.get("complete_profile_set")):
        raise ValueError("candidate preflight report is not a complete profile audit")
    if report.get("policy") != {
        "path": policy_path.relative_to(root).as_posix(),
        "sha256": sha256(policy_path),
    }:
        raise ValueError("candidate preflight policy binding is stale")
    if report.get("source") != {
        "path": expected_source_path,
        "sha256": sha256(source_path),
        "version": str(source["version"]),
        "profile_count": len(source["profiles"]),
    }:
        raise ValueError("candidate preflight source binding is stale")
    proxy_evidence_context = validate_proxy_evidence_sources(
        root, source, policy, report
    )
    profiles_by_id = {
        str(profile["id"]): profile for profile in source["profiles"]
    }
    records = report["records"]
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        asset_id = str(record["visual_asset_id"])
        if asset_id in result:
            raise ValueError(f"duplicate candidate preflight record: {asset_id}")
        review_views = record.get("review_views", [])
        if len(review_views) != 4:
            raise ValueError(f"candidate preflight views are incomplete: {asset_id}")
        for view in review_views:
            path = root / str(view["path"])
            if not path.is_file() or sha256(path) != str(view["sha256"]):
                raise ValueError(f"candidate preflight evidence mismatch: {asset_id}")
        profile = profiles_by_id.get(str(record["profile_id"]))
        if profile is None:
            raise ValueError(f"candidate preflight profile is unknown: {asset_id}")
        validate_proxy_evidence_record(
            root, profile, record, proxy_evidence_context
        )
        result[asset_id] = record
    if report.get("counts") != {
        "profiles": len(source["profiles"]),
        "visuals": len(records),
        "source_verified": sum(
            record["disposition"] == "source_verified" for record in records
        ),
        "repair_required": sum(
            record["disposition"] == "repair_required" for record in records
        ),
        "manual_review_required": sum(
            record["disposition"] == "manual_review_required"
            for record in records
        ),
        "reviewed_transparency": sum(
            record.get("reviewed_transparency") is not None for record in records
        ),
        "proxy_evidence_verified": sum(
            record.get("proxy_evidence", {}).get("status") == "verified"
            for record in records
        ),
    }:
        raise ValueError("candidate preflight counts are stale")
    return report_path, report, result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_path = resolved(root, args.source_profiles)
    preflight_policy_path = resolved(root, args.preflight)
    repairs_path = resolved(root, args.repairs)
    output_profiles_path = resolved(root, args.output_profiles)
    output_curation_path = resolved(root, args.output_curation)
    source = load_json(source_path)
    preflight_policy = load_json(preflight_policy_path)
    repairs = load_json(repairs_path)
    preflight_report_path, _, preflight_by_asset = validated_preflight_records(
        root,
        source_path,
        source,
        preflight_policy_path,
        preflight_policy,
    )
    repaired_by_asset = validated_repair_records(root, repairs)

    profiles = copy.deepcopy(source)
    profiles["version"] = str(args.profiles_version)
    profiles.setdefault("policy", {})["visual_curation"] = {
        "source_assets_are_immutable": True,
        "candidate_preflight_is_mandatory": True,
        "only_admitted_visual_paths_may_enter_sampling": True,
    }
    curation_records = []
    seen_repaired_assets: set[str] = set()
    for profile in profiles["profiles"]:
        admitted_visuals = []
        for visual in profile["visual_variants"]:
            if visual["type"] != "mesh":
                continue
            asset_id = str(visual["asset_id"])
            source_visual = {
                "path": str(visual["path"]),
                "sha256": str(visual["sha256"]),
            }
            preflight = preflight_by_asset.get(asset_id)
            if preflight is None:
                raise ValueError(f"visual has no candidate preflight record: {asset_id}")
            expected_collision = {
                "type": str(profile["collision"]["type"]),
                "dimensions_m": list(profile["collision"]["dimensions_m"]),
                "proxy_json": str(profile["source_review"]["proxy_json"]),
            }
            if (
                str(preflight["profile_id"]) != str(profile["id"])
                or str(preflight["semantic_category"])
                != str(profile["semantic_category"])
                or preflight["source_visual"] != source_visual
                or preflight["collision_proxy"] != expected_collision
            ):
                raise ValueError(f"candidate preflight no longer matches profile: {asset_id}")
            source_preflight = {
                "disposition": str(preflight["disposition"]),
                "issues": list(preflight["issues"]),
                "visibility_issues": list(preflight["visibility_issues"]),
                "recommended_operations": list(
                    preflight["recommended_operations"]
                ),
                "reviewed_transparency": copy.deepcopy(
                    preflight.get("reviewed_transparency")
                ),
                "review_views": copy.deepcopy(preflight["review_views"]),
                "review_metrics": copy.deepcopy(preflight["review_metrics"]),
                "proxy_evidence": copy.deepcopy(preflight["proxy_evidence"]),
            }
            repair = repaired_by_asset.get(asset_id)
            if repair is None:
                if (
                    source_preflight["disposition"] != "source_verified"
                    or source_preflight["issues"]
                    or source_preflight["visibility_issues"]
                    or source_preflight["recommended_operations"]
                ):
                    raise ValueError(
                        f"candidate has an unresolved preflight finding: {asset_id}"
                    )
                status = "source_verified"
                admitted = copy.deepcopy(source_visual)
                verification = {
                    "method": "candidate_preflight_four_view_material_and_proxy_review",
                    "source_preflight": source_preflight,
                }
            else:
                recipe = repair["recipe"]
                report = repair["report"]
                if str(recipe["profile_id"]) != str(profile["id"]):
                    raise ValueError(f"repair profile mismatch: {asset_id}")
                if report["source"] != source_visual:
                    raise ValueError(f"repair source no longer matches profile: {asset_id}")
                if source_preflight["disposition"] != "repair_required":
                    raise ValueError(f"repair has no preflight finding: {asset_id}")
                if source_preflight["recommended_operations"] != recipe["operations"]:
                    raise ValueError(f"repair does not resolve preflight finding: {asset_id}")
                admitted = copy.deepcopy(report["admitted_visual"])
                visual["source_visual"] = source_visual
                visual["path"] = admitted["path"]
                visual["sha256"] = admitted["sha256"]
                visual["curation"] = {
                    "status": "repaired_verified",
                    "repair_report": str(Path(admitted["path"]).parent / "repair.json"),
                }
                status = "repaired_verified"
                verification = {
                    "method": "repair_reimport_geometry_and_four_view_review",
                    "repair_operations": list(report["operations"]),
                    "repair_reason": str(report["reason"]),
                    "repair_report": str(Path(admitted["path"]).parent / "repair.json"),
                    "review_views": copy.deepcopy(report["verification"]["review_views"]),
                    "source_preflight": source_preflight,
                }
                seen_repaired_assets.add(asset_id)
            if (
                source_preflight["proxy_evidence"]["physical_proxy_catalog"][
                    "catalog_visual"
                ]
                != admitted
            ):
                raise ValueError(
                    f"physical proxy catalog does not bind the admitted visual: {asset_id}"
                )
            admitted_visuals.append(
                {
                    "visual_asset_id": asset_id,
                    "status": status,
                    "source_visual": source_visual,
                    "admitted_visual": admitted,
                    "verification": verification,
                }
            )
        if not admitted_visuals:
            raise ValueError(f"profile has no mesh visual to curate: {profile['id']}")
        curation_records.append(
            {
                "profile_id": str(profile["id"]),
                "admission": "active",
                "visuals": admitted_visuals,
            }
        )

    missing_repairs = set(repaired_by_asset) - seen_repaired_assets
    if missing_repairs:
        raise ValueError(f"repair recipes do not bind to a profile: {sorted(missing_repairs)}")
    used_assets = {
        str(visual["visual_asset_id"])
        for record in curation_records
        for visual in record["visuals"]
    }
    unused_preflight = set(preflight_by_asset) - used_assets
    if unused_preflight:
        raise ValueError(
            f"candidate preflight has visuals outside the profiles: {sorted(unused_preflight)}"
        )
    write_json(output_profiles_path, profiles)

    statuses = Counter(
        visual["status"]
        for record in curation_records
        for visual in record["visuals"]
    )
    curation = {
        "version": str(args.curation_version),
        "source": {
            "object_profiles_version": profiles["version"],
            "object_profiles_sha256": sha256(output_profiles_path),
            "profile_count": len(profiles["profiles"]),
            "candidate_preflight_policy": {
                "path": preflight_policy_path.relative_to(root).as_posix(),
                "sha256": sha256(preflight_policy_path),
            },
            "candidate_preflight_report": {
                "path": preflight_report_path.relative_to(root).as_posix(),
                "sha256": sha256(preflight_report_path),
            },
        },
        "policy": {
            "every_profile_requires_an_explicit_curation_record": True,
            "raw_sources_are_immutable": True,
            "repairs_bind_to_new_hashed_assets": True,
            "unverified_or_missing_visuals_are_a_hard_error": True,
            "curation_happens_before_matrix_compilation": True,
            "implicit_source_approval_is_forbidden": True,
            "sampling_exclusion_is_not_a_repair_strategy": True,
            "proxy_fit_overlays_and_runtime_probe_are_mandatory": True,
        },
        "records": curation_records,
        "counts": {
            "profiles": len(curation_records),
            "active_profiles": len(curation_records),
            "visuals": sum(len(record["visuals"]) for record in curation_records),
            "source_verified": statuses["source_verified"],
            "repaired_verified": statuses["repaired_verified"],
        },
    }
    write_json(output_curation_path, curation)
    print(json.dumps(curation["counts"], indent=2))


if __name__ == "__main__":
    main()
