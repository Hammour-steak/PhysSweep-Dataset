#!/usr/bin/env python3
"""Compile data-driven PhysSweep profiles into the stable backend contract."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.assets.physical_proxy_catalog import (
    load_catalog,
    records_by_id,
    validate_curated_registry_bindings,
    validate_object_profile_bindings,
)
from tools.assets.static_support_proxy import record_sha256


TOPOLOGY_TO_SUPPORT_SHAPE = {
    "flat_surface": "rectangular_slab",
    "inclined_ramp": "inclined_ramp",
    "tray": "tray_surface",
    "pedestal": "pedestal_block",
    "pocketed_table": "pocketed_table",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _unique(records: list[dict[str, Any]], key: str, source: str) -> None:
    values = [str(record[key]) for record in records]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {key} in {source}")


def validate_registry_counts(registry: dict[str, Any]) -> None:
    records = registry["records"]
    expected = {
        "by_asset_role": dict(sorted(Counter(
            str(record["asset_role"]) for record in records
        ).items())),
        "by_proxy_kind": dict(sorted(Counter(
            str(record["proxy"]["kind"]) for record in records
        ).items())),
        "sampling_enabled": sum(
            bool(record["admission"].get("sampling_enabled", False))
            for record in records
        ),
        "total": len(records),
    }
    if registry["counts"] != expected:
        raise ValueError("asset proxy registry counts do not match records")


def validate_object_visual_curation(
    root: Path,
    object_profiles_path: Path,
    object_source: dict[str, Any],
    preflight_policy_path: Path,
    preflight_report_path: Path,
    preflight_report: dict[str, Any],
    curation: dict[str, Any],
) -> None:
    source = curation["source"]
    if str(source["object_profiles_version"]) != str(object_source["version"]):
        raise ValueError("object visual curation targets a different profile version")
    if str(source["object_profiles_sha256"]) != sha256(object_profiles_path):
        raise ValueError("object visual curation targets a different profile revision")

    profiles = object_source["profiles"]
    if int(source["profile_count"]) != len(profiles):
        raise ValueError("object visual curation profile count is stale")
    if source.get("candidate_preflight_policy") != {
        "path": preflight_policy_path.relative_to(root).as_posix(),
        "sha256": sha256(preflight_policy_path),
    }:
        raise ValueError("object visual curation preflight policy binding is stale")
    if source.get("candidate_preflight_report") != {
        "path": preflight_report_path.relative_to(root).as_posix(),
        "sha256": sha256(preflight_report_path),
    }:
        raise ValueError("object visual curation preflight report binding is stale")
    if not bool(preflight_report.get("complete_profile_set")):
        raise ValueError("object visual candidate preflight is incomplete")
    preflight_policy = load_json(preflight_policy_path)
    evidence_policy = preflight_policy["proxy_evidence"]
    core_path = root / str(evidence_policy["core_index"])
    overlay_root = root / str(evidence_policy["overlay_root"])
    catalog_path = root / str(evidence_policy["physical_proxy_catalog"])
    if not core_path.is_file() or not overlay_root.is_dir():
        raise FileNotFoundError("object proxy evidence source is missing")
    core_records = read_jsonl(core_path)
    core_by_sample = {str(record["sample_id"]): record for record in core_records}
    if len(core_by_sample) != len(core_records):
        raise ValueError("object proxy evidence has duplicate core sample ids")
    if set(core_by_sample) != {
        str(profile["source_review"]["sample_id"]) for profile in profiles
    }:
        raise ValueError("object proxy core index differs from active profiles")
    catalog_manifest, catalog_records = load_catalog(
        root, catalog_path, require_runtime_validation=True
    )
    catalog_by_asset = records_by_id(catalog_records)
    validation_descriptor = catalog_manifest.get("validation")
    if not isinstance(validation_descriptor, dict):
        raise ValueError("object proxy catalog has no runtime validation")
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
            "object proxy analytic-drop validation has duplicate asset ids"
        )
    required_overlay_views = [
        str(value) for value in evidence_policy["required_overlay_views"]
    ]
    expected_proxy_sources = {
        "core_index": {
            "path": core_path.relative_to(root).as_posix(),
            "sha256": sha256(core_path),
            "record_count": len(core_records),
        },
        "overlay_root": {
            "path": overlay_root.relative_to(root).as_posix(),
            "required_views": required_overlay_views,
        },
        "physical_proxy_catalog": {
            "path": catalog_path.relative_to(root).as_posix(),
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
    actual_proxy_sources = copy.deepcopy(
        preflight_report.get("proxy_evidence_sources", {})
    )
    actual_proxy_sources.get("physical_proxy_catalog", {}).pop("sha256", None)
    if actual_proxy_sources != expected_proxy_sources:
        raise ValueError("object proxy evidence source binding is stale")
    preflight_records = preflight_report["records"]
    preflight_by_asset = {
        str(record["visual_asset_id"]): record for record in preflight_records
    }
    if len(preflight_by_asset) != len(preflight_records):
        raise ValueError("object visual candidate preflight has duplicate assets")
    profiles_by_id = {str(profile["id"]): profile for profile in profiles}
    expected_visual_assets = {
        str(variant["asset_id"])
        for profile in profiles
        for variant in profile["visual_variants"]
        if variant["type"] == "mesh"
    }
    if set(preflight_by_asset) != expected_visual_assets:
        raise ValueError(
            "object visual candidate preflight does not cover active visuals exactly"
        )
    records = curation["records"]
    curated_ids = [str(record["profile_id"]) for record in records]
    if len(curated_ids) != len(set(curated_ids)):
        raise ValueError("object visual curation has duplicate profile records")
    if set(curated_ids) != set(profiles_by_id):
        raise ValueError("object visual curation does not cover every profile exactly once")
    if not bool(curation["policy"].get("sampling_exclusion_is_not_a_repair_strategy")):
        raise ValueError("object visual curation still permits exclusion as repair")
    if not bool(curation["policy"].get("implicit_source_approval_is_forbidden")):
        raise ValueError("object visual curation still permits implicit source approval")
    if not bool(
        curation["policy"].get(
            "proxy_fit_overlays_and_runtime_probe_are_mandatory"
        )
    ):
        raise ValueError("object visual curation does not require proxy evidence")

    verified_hashes: dict[str, str] = {}

    def validate_file(binding: dict[str, Any], context: str) -> None:
        if set(binding) != {"path", "sha256"}:
            raise ValueError(f"visual binding is incomplete: {context}")
        relative_path = str(binding["path"])
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"visual binding is missing: {context}: {relative_path}")
        if relative_path not in verified_hashes:
            verified_hashes[relative_path] = sha256(path)
        actual = verified_hashes[relative_path]
        if actual != str(binding["sha256"]):
            raise ValueError(f"visual binding hash mismatch: {context}")

    status_counts: Counter[str] = Counter()
    visual_count = 0
    for record in records:
        profile_id = str(record["profile_id"])
        if record.get("admission") != "active":
            raise ValueError(f"curated profile is not active: {profile_id}")
        profile = profiles_by_id[profile_id]
        variants = {
            str(variant["asset_id"]): variant
            for variant in profile["visual_variants"]
            if variant["type"] == "mesh"
        }
        visuals = record["visuals"]
        visual_ids = [str(visual["visual_asset_id"]) for visual in visuals]
        if len(visual_ids) != len(set(visual_ids)) or set(visual_ids) != set(variants):
            raise ValueError(f"curated visual set differs from profile: {profile_id}")
        for visual in visuals:
            asset_id = str(visual["visual_asset_id"])
            context = f"{profile_id}:{asset_id}"
            variant = variants[asset_id]
            source_visual = visual["source_visual"]
            admitted_visual = visual["admitted_visual"]
            validate_file(source_visual, f"{context}:source")
            validate_file(admitted_visual, f"{context}:admitted")
            preflight = preflight_by_asset.get(asset_id)
            if preflight is None:
                raise ValueError(f"visual has no candidate preflight evidence: {context}")
            expected_preflight = {
                "disposition": str(preflight["disposition"]),
                "issues": list(preflight["issues"]),
                "visibility_issues": list(preflight["visibility_issues"]),
                "recommended_operations": list(
                    preflight["recommended_operations"]
                ),
                "reviewed_transparency": preflight.get(
                    "reviewed_transparency"
                ),
                "review_views": preflight["review_views"],
                "review_metrics": preflight["review_metrics"],
                "proxy_evidence": preflight["proxy_evidence"],
            }
            verification = visual.get("verification", {})
            if verification.get("source_preflight") != expected_preflight:
                raise ValueError(f"candidate preflight evidence is stale: {context}")
            if preflight.get("source_visual") != source_visual:
                raise ValueError(f"candidate preflight source mismatch: {context}")
            review_views = preflight.get("review_views", [])
            if len(review_views) != 4:
                raise ValueError(f"candidate preflight views are incomplete: {context}")
            for view in review_views:
                validate_file(view, f"{context}:raw-review")
            proxy_evidence = preflight.get("proxy_evidence", {})
            sample_id = str(profile["source_review"]["sample_id"])
            core_record = core_by_sample.get(sample_id)
            if (
                core_record is None
                or proxy_evidence.get("status") != "verified"
                or str(proxy_evidence.get("sample_id")) != sample_id
                or proxy_evidence.get("core_record_sha256")
                != record_sha256(core_record)
                or proxy_evidence.get("proxy_method")
                != str(core_record["method"])
                or proxy_evidence.get("blender_alignment_fit")
                != core_record["blender_alignment_fit"]
                or float(
                    proxy_evidence.get(
                        "proxy_to_visual_hull_volume_ratio", float("inf")
                    )
                )
                != float(core_record["proxy_to_visual_hull_volume_ratio"])
            ):
                raise ValueError(f"object proxy core evidence is stale: {context}")
            if (
                float(proxy_evidence["proxy_to_visual_hull_volume_ratio"])
                > float(
                    evidence_policy[
                        "maximum_proxy_to_visual_hull_volume_ratio"
                    ]
                )
                or not bool(proxy_evidence["blender_alignment_fit"]["passed"])
                or float(
                    proxy_evidence["blender_alignment_fit"][
                        "maximum_relative_error"
                    ]
                )
                > float(evidence_policy["maximum_alignment_relative_error"])
            ):
                raise ValueError(f"object proxy evidence exceeds policy: {context}")
            source_proxy = proxy_evidence.get("source_proxy", {})
            if str(source_proxy.get("path")) != str(
                profile["source_review"]["proxy_json"]
            ):
                raise ValueError(f"object source proxy path is stale: {context}")
            validate_file(source_proxy, f"{context}:source-proxy")
            overlay_views = proxy_evidence.get("overlay_views", [])
            if (
                len(overlay_views) != len(required_overlay_views)
                or {str(item.get("view")) for item in overlay_views}
                != set(required_overlay_views)
            ):
                raise ValueError(f"object proxy overlays are incomplete: {context}")
            for view in overlay_views:
                binding = {"path": view["path"], "sha256": view["sha256"]}
                validate_file(binding, f"{context}:proxy-overlay:{view['view']}")
            catalog_record = catalog_by_asset.get(asset_id)
            if catalog_record is None:
                raise ValueError(f"object proxy catalog record is missing: {context}")
            colliders = catalog_record["proxy"]["colliders"]
            expected_catalog_evidence = {
                "asset_id": asset_id,
                "catalog_record_sha256": record_sha256(catalog_record),
                "catalog_records_sha256": str(
                    catalog_manifest["records_sha256"]
                ),
                "representation": str(
                    catalog_record["proxy"]["representation"]
                ),
                "collider_shape": str(colliders[0]["shape"]),
                "collider_size_m": [
                    float(value) for value in colliders[0]["size_m"]
                ],
                "qa_grade": str(catalog_record["qa"]["grade"]),
                "catalog_visual": {
                    "path": str(catalog_record["source"]["visual_path"]),
                    "sha256": str(catalog_record["source"]["sha256"]),
                },
            }
            if (
                proxy_evidence.get("physical_proxy_catalog")
                != expected_catalog_evidence
            ):
                raise ValueError(f"object proxy catalog evidence is stale: {context}")
            if expected_catalog_evidence["catalog_visual"] != admitted_visual:
                raise ValueError(
                    f"object proxy catalog does not bind admitted visual: {context}"
                )
            probe = validation_by_asset.get(asset_id)
            if probe is None:
                raise ValueError(f"object proxy runtime probe is missing: {context}")
            expected_probe = {
                "record_sha256": record_sha256(probe),
                "probe": str(probe["probe"]),
                "passed": bool(probe["passed"]),
            }
            if (
                proxy_evidence.get("runtime_probe") != expected_probe
                or not expected_probe["passed"]
            ):
                raise ValueError(f"object proxy runtime probe is stale: {context}")
            if {
                "path": str(variant["path"]),
                "sha256": str(variant["sha256"]),
            } != admitted_visual:
                raise ValueError(f"profile does not bind its admitted visual: {context}")
            status = str(visual["status"])
            status_counts[status] += 1
            visual_count += 1
            if status == "source_verified":
                if admitted_visual != source_visual:
                    raise ValueError(f"source-verified visual changed asset: {context}")
                if (
                    preflight["disposition"] != "source_verified"
                    or preflight["issues"]
                    or preflight["visibility_issues"]
                    or preflight["recommended_operations"]
                ):
                    raise ValueError(f"unrepaired finding entered sampling: {context}")
                continue
            if status != "repaired_verified":
                raise ValueError(f"unsupported visual curation status: {context}: {status}")
            if admitted_visual == source_visual:
                raise ValueError(f"repaired visual still binds the raw source: {context}")
            if variant.get("source_visual") != source_visual:
                raise ValueError(f"repaired profile lost raw source provenance: {context}")
            variant_curation = variant.get("curation", {})
            report_path = str(verification.get("repair_report", ""))
            if variant_curation != {
                "status": "repaired_verified",
                "repair_report": report_path,
            }:
                raise ValueError(f"repaired profile curation binding is stale: {context}")
            report_file = root / report_path
            if not report_file.is_file():
                raise FileNotFoundError(f"repair report is missing: {context}")
            report = load_json(report_file)
            if report.get("source") != source_visual:
                raise ValueError(f"repair report source mismatch: {context}")
            if report.get("admitted_visual") != admitted_visual:
                raise ValueError(f"repair report admitted visual mismatch: {context}")
            if report.get("operations") != verification.get("repair_operations"):
                raise ValueError(f"repair operation evidence mismatch: {context}")
            if preflight["disposition"] != "repair_required":
                raise ValueError(f"repair has no candidate preflight finding: {context}")
            if report.get("operations") != preflight["recommended_operations"]:
                raise ValueError(f"repair does not resolve candidate finding: {context}")
            if report.get("reason") != verification.get("repair_reason"):
                raise ValueError(f"repair reason evidence mismatch: {context}")
            review_views = report.get("verification", {}).get("review_views", [])
            if review_views != verification.get("review_views") or len(review_views) != 4:
                raise ValueError(f"repair review evidence is incomplete: {context}")
            for view in review_views:
                validate_file(view, f"{context}:review")

    counts = {
        "profiles": len(profiles),
        "active_profiles": len(profiles),
        "visuals": visual_count,
        "source_verified": status_counts["source_verified"],
        "repaired_verified": status_counts["repaired_verified"],
    }
    if curation["counts"] != counts:
        raise ValueError("object visual curation counts are stale")


def compile_object_profile(
    profile: dict[str, Any], mesh_alignment_coordinate_frame: str = "raw_gltf_z_up"
) -> dict[str, Any]:
    variants = copy.deepcopy(profile.get("visual_variants", [profile.get("visual")]))
    if not variants or any(variant is None for variant in variants):
        raise ValueError(f"object profile has no visual variants: {profile['id']}")
    if any("id" not in variant for variant in variants):
        raise ValueError(f"visual variant lacks id: {profile['id']}")
    collision = copy.deepcopy(profile["collision"])
    physics = copy.deepcopy(profile["physics"])
    collision_type = str(collision["type"])
    dimensions = [float(value) for value in collision["dimensions_m"]]
    if collision_type not in {"sphere", "cuboid", "cylinder"}:
        raise ValueError(f"unsupported collision proxy: {collision_type}")
    if len(dimensions) != 3 or min(dimensions) <= 0.0:
        raise ValueError(f"invalid collision dimensions: {profile['id']}")
    ids = [str(variant["id"]) for variant in variants]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate visual variant id: {profile['id']}")
    for visual in variants:
        if visual["type"] not in {"primitive", "mesh"}:
            raise ValueError(f"unsupported visual type: {visual['type']}")
        if visual["type"] == "mesh":
            visual.setdefault(
                "alignment_coordinate_frame", mesh_alignment_coordinate_frame
            )
            required = {
                "asset_id", "path", "sha256", "alignment_euler_degrees",
                "alignment_coordinate_frame", "canonical_extent_m", "license",
            }
            missing = required - set(visual)
            if missing:
                raise ValueError(f"mesh visual lacks {sorted(missing)}: {profile['id']}")
    primitive = next((variant for variant in variants if variant["type"] == "primitive"), None)
    fallback_visual = primitive or variants[0]
    if "material_hint" not in fallback_visual or "color" not in fallback_visual:
        raise ValueError(f"object profile visual lacks material defaults: {profile['id']}")
    return {
        "label": str(profile["id"]),
        "semantic_category": str(profile["semantic_category"]),
        "shape": collision_type,
        "pose_profile": str(profile.get("pose_profile", "support_normal")),
        "excluded_motion_families": [
            str(value) for value in profile.get("excluded_motion_families", [])
        ],
        "size": dimensions,
        "hint": str(fallback_visual["material_hint"]),
        "color": [float(value) for value in fallback_visual["color"]],
        "friction": float(physics["friction"]),
        "restitution": float(physics["restitution"]),
        "mass": [float(value) for value in physics["mass_range_kg"]],
        "visual_profile": fallback_visual,
        "visual_variants": variants,
        "collision_profile": collision,
    }


def compile_scene_kit(kit: dict[str, Any]) -> dict[str, Any]:
    topology = str(kit["topology"])
    if topology not in TOPOLOGY_TO_SUPPORT_SHAPE:
        raise ValueError(f"unsupported scene topology: {topology}")
    structure = copy.deepcopy(kit.get("structure", {}))
    style = str(structure.get("style", "none"))
    placement: dict[str, Any] = {
        "support_shape": TOPOLOGY_TO_SUPPORT_SHAPE[topology],
        "structure_style": style,
        "show_table_legs": style == "legs",
        "ground_surface": style in {"ground", "corridor"},
    }
    key_map = {
        "thickness_m": "thickness",
        "family": "structure_family",
        "rail_height_m": "rail_height_m",
        "rail_width_m": "rail_width_m",
        "side_rail_height_m": "side_rail_height_m",
        "side_rail_width_m": "side_rail_width_m",
        "landing_length_m": "landing_length_m",
        "anchor_low_edge_to_floor": "anchor_low_edge_to_floor",
        "base_platform_top_z_m": "base_platform_top_z_m",
        "base_platform_margin_m": "base_platform_margin_m",
        "base_platform_thickness_m": "base_platform_thickness_m",
        "pocket_radius_m": "pocket_radius_m",
        "slope_axis": "slope_axis",
        "slope_rise_m": "slope_rise_m",
        "motion_axis": "motion_axis",
        "maximum_planar_trajectory_distance_m": "maximum_planar_trajectory_distance_m",
        "corridor_wall_height_m": "corridor_wall_height_m",
        "corridor_wall_thickness_m": "corridor_wall_thickness_m",
        "camera_clearance_m": "camera_clearance_m",
    }
    for source, target in key_map.items():
        if source in structure:
            placement[target] = structure[source]
    if "maximum_planar_trajectory_distance_m" in placement and float(
        placement["maximum_planar_trajectory_distance_m"]
    ) <= 0.0:
        raise ValueError(f"scene kit has invalid trajectory distance: {kit['id']}")
    if "motion_axis" in placement and placement["motion_axis"] not in {"x", "y"}:
        raise ValueError(f"scene kit has invalid motion axis: {kit['id']}")
    corridor_fields = {
        "corridor_wall_height_m",
        "corridor_wall_thickness_m",
        "camera_clearance_m",
    }
    if style == "corridor":
        missing = {"motion_axis", *corridor_fields} - set(placement)
        if missing:
            raise ValueError(
                f"corridor scene kit lacks {sorted(missing)}: {kit['id']}"
            )
    elif corridor_fields & set(placement):
        raise ValueError(f"non-corridor scene kit declares walls: {kit['id']}")
    if "camera_clearance_m" in placement and float(
        placement["camera_clearance_m"]
    ) <= 0.0:
        raise ValueError(f"scene kit has invalid camera clearance: {kit['id']}")
    material_indices = kit.get("material_indices", {})
    record = {
        "label": str(kit["id"]),
        "scene_class": str(kit["scene_class"]),
        "topology": topology,
        "theme": str(kit["theme"]),
        "size": [float(value) for value in kit["dimensions_m"]],
        "top_z": float(kit["surface_z_m"]),
        "surface_index": int(material_indices.get("surface", 0)),
        "wall_index": int(material_indices.get("wall", 0)),
        "overrides": {"placement": placement},
    }
    if "visual" in kit:
        record["visual_profile"] = copy.deepcopy(kit["visual"])
    if "allowed_motions" in kit:
        record["allowed_motions"] = [str(value) for value in kit["allowed_motions"]]
        if not record["allowed_motions"] or len(record["allowed_motions"]) != len(
            set(record["allowed_motions"])
        ):
            raise ValueError(f"scene kit has invalid allowed motions: {kit['id']}")
    if "environment_categories" in kit:
        record["environment_categories"] = [
            str(value) for value in kit["environment_categories"]
        ]
        if not record["environment_categories"] or len(
            record["environment_categories"]
        ) != len(set(record["environment_categories"])):
            raise ValueError(
                f"scene kit has invalid environment categories: {kit['id']}"
            )
    variants = kit.get("geometry_variants", [])
    if variants:
        variant_ids = [str(variant["id"]) for variant in variants]
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError(f"scene kit has duplicate geometry variants: {kit['id']}")
        compiled_variants = []
        for variant in variants:
            unsupported = set(variant) - {
                "id",
                "dimensions_m",
                "surface_z_m",
                "structure",
            }
            if unsupported:
                raise ValueError(
                    f"scene kit geometry variant has unsupported fields: "
                    f"{kit['id']}/{variant['id']}: {sorted(unsupported)}"
                )
            resolved_kit = copy.deepcopy(kit)
            resolved_kit.pop("geometry_variants", None)
            for field in ("dimensions_m", "surface_z_m"):
                if field in variant:
                    resolved_kit[field] = copy.deepcopy(variant[field])
            resolved_structure = copy.deepcopy(resolved_kit.get("structure", {}))
            resolved_structure.update(copy.deepcopy(variant.get("structure", {})))
            resolved_kit["structure"] = resolved_structure
            compiled = compile_scene_kit(resolved_kit)
            compiled_variants.append(
                {
                    "id": str(variant["id"]),
                    "size": compiled["size"],
                    "top_z": compiled["top_z"],
                    "overrides": compiled["overrides"],
                }
            )
        record["geometry_variants"] = compiled_variants
    return record


def support_is_compatible(
    support: dict[str, Any], motion: str, compatibility: dict[str, Any]
) -> bool:
    try:
        rule = compatibility["motions"][motion]
    except KeyError as exc:
        raise ValueError(f"motion lacks compatibility rule: {motion}") from exc
    return (
        str(support["scene_class"]) in set(rule["scene_classes"])
        and str(support["topology"]) in set(rule["topologies"])
        and motion in set(support.get("allowed_motions", [motion]))
    )


def validate_generic_capabilities(
    compiled: dict[str, Any],
    backend: dict[str, Any],
    capabilities: dict[str, Any],
) -> None:
    scope = capabilities["generic_base_scope"]
    if capabilities["active_backend"] != backend["backend_id"]:
        raise ValueError("backend capability id does not match active backend")
    if capabilities["metadata_schema"] != "physweep_pybullet_rigid_metadata_v1":
        raise ValueError("backend capability metadata schema is unsupported")
    if int(scope["dynamic_body_count"]) != 1 or scope["body_model"] != "rigid_body":
        raise ValueError("generic capability scope must describe one rigid body")
    expected_sets = {
        "motions": {str(value) for value in compiled["axes"]["motion_axis"]},
        "shapes": {
            str(record["shape"]) for record in compiled["axes"]["object_axis"]
        },
        "scene_classes": {
            str(record["label"])
            for record in compiled["axes"]["scene_class_axis"]
        },
        "support_types": {
            str(record["label"]) for record in compiled["axes"]["support_axis"]
        },
        "support_shapes": {
            str(record["overrides"]["placement"]["support_shape"])
            for record in compiled["axes"]["support_axis"]
        },
    }
    for key, expected in expected_sets.items():
        declared = {str(value) for value in scope[key]}
        if declared != expected:
            raise ValueError(
                f"generic backend capability {key} does not match compiled rules"
            )


def bind_environment_compositions(
    composition_source: dict[str, Any],
    mesh_profiles: list[dict[str, Any]],
    scene_source: dict[str, Any],
    base: dict[str, Any],
) -> None:
    if composition_source.get("schema_version") != (
        "physweep_visual_environment_composition_v1"
    ):
        raise ValueError("unsupported visual environment composition schema")
    records = composition_source["records"]
    _unique(records, "profile_id", "visual environment compositions")
    by_profile = {str(record["profile_id"]): record for record in records}
    profile_ids = {str(profile["id"]) for profile in mesh_profiles}
    if set(by_profile) != profile_ids:
        raise ValueError("environment composition set does not match mesh profiles")

    support_by_id = {
        str(kit["id"]): kit
        for kit in scene_source["kits"]
        if bool(kit.get("sampling_enabled", True))
    }
    motion_ids = {str(value) for value in base["axes"]["motion_axis"]}
    extent_ids = {
        str(record["label"])
        for record in base["axes"]["trajectory_extent_axis"]
    }
    policy = composition_source["policy"]
    admitted_status = str(policy["admitted_review_status"])
    supported_modes = {str(value) for value in policy["supported_modes"]}
    admitted_scene_classes = {
        str(value) for value in policy["integrated_environment_scene_classes"]
    }
    minimum_margin = float(policy["minimum_clearance_safety_margin_m"])

    for profile in mesh_profiles:
        record = copy.deepcopy(by_profile[str(profile["id"])])
        if str(record["asset_id"]) != str(profile["asset"]["asset_id"]):
            raise ValueError(f"environment composition asset mismatch: {profile['id']}")
        status = str(record["review_status"])
        if status != admitted_status:
            if not str(record.get("reason", "")).strip():
                raise ValueError(
                    f"paused environment lacks a review reason: {profile['id']}"
                )
            profile["composition"] = record
            continue

        mode = str(record.get("composition_mode", ""))
        if mode not in supported_modes:
            raise ValueError(f"unsupported environment composition: {profile['id']}")
        surface = record["action_surface"]
        anchor = [float(value) for value in surface["anchor_local_m"]]
        if len(anchor) != 3:
            raise ValueError(f"invalid environment action anchor: {profile['id']}")
        audited_radius = float(surface["audited_clear_radius_m"])
        admitted_radius = float(surface["admitted_action_radius_m"])
        if admitted_radius <= 0.0 or audited_radius - admitted_radius < minimum_margin:
            raise ValueError(
                f"environment action clearance margin is too small: {profile['id']}"
            )
        contract = profile["asset"]["collision_proxy"]["transform_contract"]
        if str(contract["frame"]) != str(policy["action_surface_frame"]):
            raise ValueError(f"environment action frame mismatch: {profile['id']}")
        if abs(anchor[2] - float(contract["authoritative_floor_z_m"])) > 1.0e-5:
            raise ValueError(f"environment action floor mismatch: {profile['id']}")
        camera = record["camera"]
        required_camera_fields = {
            "preferred_local_azimuth_degrees",
            "maximum_local_azimuth_deviation_degrees",
            "preferred_elevation_degrees",
            "minimum_elevation_degrees",
            "maximum_elevation_degrees",
            "minimum_distance_m",
            "maximum_distance_m",
            "target_depth_offset_m",
            "target_lateral_offset_m",
            "target_z_offset_m",
            "focal_length_cap_mm",
            "reviewed_view",
        }
        if set(camera) != required_camera_fields:
            raise ValueError(
                f"environment camera contract is incomplete: {profile['id']}"
            )
        float(camera["preferred_local_azimuth_degrees"])
        maximum_azimuth_deviation = float(
            camera["maximum_local_azimuth_deviation_degrees"]
        )
        minimum_elevation = float(camera["minimum_elevation_degrees"])
        preferred_elevation = float(camera["preferred_elevation_degrees"])
        maximum_elevation = float(camera["maximum_elevation_degrees"])
        minimum_distance = float(camera["minimum_distance_m"])
        maximum_distance = float(camera["maximum_distance_m"])
        if not 0.0 <= maximum_azimuth_deviation <= 30.0:
            raise ValueError(
                f"invalid environment azimuth corridor: {profile['id']}"
            )
        if not (
            0.0 < minimum_elevation
            <= preferred_elevation
            <= maximum_elevation
            < 90.0
        ):
            raise ValueError(
                f"invalid environment elevation corridor: {profile['id']}"
            )
        if not 1.0 <= minimum_distance <= maximum_distance:
            raise ValueError(
                f"invalid environment camera distance corridor: {profile['id']}"
            )
        target_depth_offset = float(camera["target_depth_offset_m"])
        float(camera["target_lateral_offset_m"])
        float(camera["target_z_offset_m"])
        focal_length_cap = float(camera["focal_length_cap_mm"])
        if not 0.0 <= target_depth_offset <= 2.0:
            raise ValueError(
                f"invalid environment target depth offset: {profile['id']}"
            )
        if not 20.0 <= focal_length_cap <= 55.0:
            raise ValueError(
                f"invalid environment focal-length cap: {profile['id']}"
            )
        if not str(camera["reviewed_view"]).strip():
            raise ValueError(f"environment reviewed view is empty: {profile['id']}")
        bindings = record.get("bindings", [])
        if not bindings:
            raise ValueError(f"approved environment has no bindings: {profile['id']}")
        for binding in bindings:
            support_ids = {str(value) for value in binding["support_ids"]}
            motion_families = {
                str(value) for value in binding["motion_families"]
            }
            if not support_ids or not motion_families:
                raise ValueError(f"empty environment binding: {profile['id']}")
            if not support_ids <= set(support_by_id):
                raise ValueError(f"unknown environment support: {profile['id']}")
            if any(
                str(support_by_id[support_id]["scene_class"])
                not in admitted_scene_classes
                for support_id in support_ids
            ):
                raise ValueError(
                    f"environment binding uses an unreviewed scene class: {profile['id']}"
                )
            if not motion_families <= motion_ids:
                raise ValueError(f"unknown environment motion: {profile['id']}")
            trajectory_extents = {
                str(value) for value in binding.get("trajectory_extents", extent_ids)
            }
            if not trajectory_extents or not trajectory_extents <= extent_ids:
                raise ValueError(f"invalid environment trajectory extent: {profile['id']}")
        profile["composition"] = record


def load_sampling_bundle(root: Path, bundle_path: Path) -> dict[str, Any]:
    bundle = load_json(bundle_path)
    required_dependencies = {
        "base_rules",
        "object_profiles",
        "object_visual_preflight",
        "object_visual_preflight_report",
        "object_visual_curation",
        "object_visual_repairs",
        "scene_kits",
        "scene_visual_profiles",
        "scene_mesh_profiles",
        "environment_composition",
        "environment_collision_proxies",
        "support_mesh_profiles",
        "asset_proxy_registry",
        "physical_proxy_catalog",
        "compatibility",
        "visual_sampling",
        "backend",
        "backend_capabilities",
        "material_manifest",
        "hdri_manifest",
    }
    missing_dependencies = required_dependencies - set(bundle)
    if missing_dependencies:
        raise ValueError(
            f"sampling bundle lacks dependencies: {sorted(missing_dependencies)}"
        )
    for key in sorted(required_dependencies):
        if not (root / str(bundle[key])).is_file():
            raise FileNotFoundError(f"sampling bundle dependency is missing: {key}")
    required_implementation = {
        "sampler",
        "compiler",
        "geometry",
        "inclined_support",
        "support_structure",
        "time_step",
        "simulator",
        "trajectory_audit",
        "proxy_catalog",
        "static_support_binding",
        "camera_geometry",
        "environment_collision",
        "batch_runner",
        "visual_preflight",
        "visual_repair",
        "visual_curation",
    }
    if bundle.get("policy", {}).get(
        "motion_rules_are_grouped_and_registry_dispatched"
    ):
        required_implementation.update(
            {
                "motion_rule_package",
                "motion_rule_contracts",
                "motion_rule_common",
                "motion_rule_registry",
                "motion_rule_planar",
                "motion_rule_ballistic",
                "motion_rule_incline",
                "motion_rule_transition",
            }
        )
    declared_implementation = bundle.get("implementation", {})
    missing_implementation = required_implementation - set(declared_implementation)
    if missing_implementation:
        raise ValueError(
            f"sampling bundle lacks implementation files: {sorted(missing_implementation)}"
        )
    for key, relative_path in sorted(declared_implementation.items()):
        if not (root / str(relative_path)).is_file():
            raise FileNotFoundError(f"sampling implementation is missing: {key}")
    base = load_json(root / bundle["base_rules"])
    object_profiles_path = root / bundle["object_profiles"]
    object_source = load_json(object_profiles_path)
    preflight_policy_path = root / bundle["object_visual_preflight"]
    preflight_policy = load_json(preflight_policy_path)
    if str(
        preflight_policy["proxy_evidence"]["physical_proxy_catalog"]
    ) != str(bundle["physical_proxy_catalog"]):
        raise ValueError(
            "object proxy preflight and sampling bundle use different catalogs"
        )
    preflight_report_path = root / bundle["object_visual_preflight_report"]
    preflight_report = load_json(preflight_report_path)
    object_visual_curation = load_json(root / bundle["object_visual_curation"])
    scene_source = load_json(root / bundle["scene_kits"])
    compatibility = load_json(root / bundle["compatibility"])
    backend = load_json(root / bundle["backend"])
    backend_capabilities = load_json(root / bundle["backend_capabilities"])
    scene_visual_profiles = load_json(root / bundle["scene_visual_profiles"])
    scene_mesh_profiles = load_json(root / bundle["scene_mesh_profiles"])
    environment_composition = load_json(root / bundle["environment_composition"])
    environment_collision_proxies = load_json(
        root / bundle["environment_collision_proxies"]
    )
    visual_rules = load_json(root / bundle["visual_sampling"])
    support_mesh_profiles = load_json(root / bundle["support_mesh_profiles"])
    asset_proxy_registry = load_json(root / bundle["asset_proxy_registry"])
    physical_proxy_manifest, physical_proxy_records = load_catalog(
        root,
        root / bundle["physical_proxy_catalog"],
        require_runtime_validation=True,
    )
    validate_registry_counts(asset_proxy_registry)
    validate_object_visual_curation(
        root,
        object_profiles_path,
        object_source,
        preflight_policy_path,
        preflight_report_path,
        preflight_report,
        object_visual_curation,
    )
    validate_object_profile_bindings(
        physical_proxy_records, object_source["profiles"]
    )
    validate_curated_registry_bindings(
        physical_proxy_records, asset_proxy_registry
    )
    physical_proxy_by_id = records_by_id(physical_proxy_records)
    _unique(object_source["profiles"], "id", bundle["object_profiles"])
    _unique(scene_source["kits"], "id", bundle["scene_kits"])
    _unique(
        scene_visual_profiles["profiles"],
        "id",
        bundle["scene_visual_profiles"],
    )
    _unique(
        scene_mesh_profiles["profiles"],
        "id",
        bundle["scene_mesh_profiles"],
    )
    _unique(
        support_mesh_profiles["profiles"],
        "id",
        bundle["support_mesh_profiles"],
    )
    _unique(
        asset_proxy_registry["records"],
        "asset_id",
        bundle["asset_proxy_registry"],
    )
    procedural_profiles = copy.deepcopy(scene_visual_profiles["profiles"])
    for profile in procedural_profiles:
        profile.setdefault("visual_type", "procedural_room")
        clear_lane = profile.get("clear_lane_half_width_m")
        if clear_lane is not None:
            clear_lane = float(clear_lane)
            if clear_lane <= 0.0:
                raise ValueError(
                    f"scene visual clear lane must be positive: {profile['id']}"
                )
            for piece in profile.get("set_pieces", []):
                lateral = abs(float(piece["offset_lateral_outward_z"][0]))
                half_width = float(piece["size_m"][0]) / 2.0
                if lateral - half_width < clear_lane:
                    raise ValueError(
                        "scene visual set piece enters the clear motion lane: "
                        f"{profile['id']}:{piece['id']}"
                    )
    mesh_profiles = copy.deepcopy(scene_mesh_profiles["profiles"])
    bind_environment_compositions(
        environment_composition,
        mesh_profiles,
        scene_source,
        base,
    )
    environment_proxy_by_asset = {
        str(record["asset_id"]): record
        for record in environment_collision_proxies["records"]
    }
    if len(environment_proxy_by_asset) != len(
        environment_collision_proxies["records"]
    ):
        raise ValueError("duplicate environment collision proxy")
    all_scene_visual_profiles = procedural_profiles + mesh_profiles
    _unique(all_scene_visual_profiles, "id", "combined scene visual profiles")
    allowed_environment_categories = {
        str(value) for value in visual_rules["environment_categories"]
    }
    for profile in all_scene_visual_profiles:
        category = str(profile.get("environment_category", ""))
        if category not in allowed_environment_categories:
            raise ValueError(
                f"invalid environment category for {profile['id']}: {category}"
            )
    for profile in mesh_profiles:
        if profile.get("visual_type") != "mesh_backdrop":
            raise ValueError(f"invalid mesh scene visual type: {profile['id']}")
        asset = profile.get("asset", {})
        required = {
            "asset_id", "path", "sha256", "source_bbox_size",
            "normalization_axis", "target_extent_m", "front_view_yaw_degrees",
            "license", "collision_proxy",
        }
        missing = required - set(asset)
        if missing:
            raise ValueError(
                f"mesh scene visual lacks {sorted(missing)}: {profile['id']}"
            )
        if asset["normalization_axis"] not in {"x", "y", "z"}:
            raise ValueError(f"invalid normalization axis: {profile['id']}")
        if len(asset["source_bbox_size"]) != 3 or min(asset["source_bbox_size"]) <= 0:
            raise ValueError(f"invalid source bounds: {profile['id']}")
        if float(asset["target_extent_m"]) <= 0:
            raise ValueError(f"invalid target extent: {profile['id']}")
        proxy = asset["collision_proxy"]
        source_proxy = environment_proxy_by_asset.get(str(asset["asset_id"]))
        if source_proxy is None or str(source_proxy["profile_id"]) != str(
            profile["id"]
        ):
            raise ValueError(f"mesh environment proxy is missing: {profile['id']}")
        expected_proxy = source_proxy["proxy"]
        for key in ("representation", "method", "path", "sha256"):
            if str(proxy[key]) != str(expected_proxy[key]):
                raise ValueError(
                    f"mesh environment proxy field is stale: {profile['id']}:{key}"
                )
        if proxy["representation"] != "static_concave_mesh":
            raise ValueError(f"invalid environment proxy type: {profile['id']}")
        proxy_path = root / str(proxy["path"])
        if not proxy_path.is_file() or sha256(proxy_path) != str(proxy["sha256"]):
            raise ValueError(f"environment proxy hash mismatch: {profile['id']}")
    if set(environment_proxy_by_asset) != {
        str(profile["asset"]["asset_id"]) for profile in mesh_profiles
    }:
        raise ValueError("environment proxy set does not match mesh profiles")
    active_support_ids = {
        str(kit["id"])
        for kit in scene_source["kits"]
        if bool(kit.get("sampling_enabled", True))
    }
    support_mesh_required = {
        "asset_id", "path", "sha256", "source_bbox_size",
        "source_support_plane_z_from_bottom", "alignment_yaw_degrees",
        "material_policy", "requires_image_texture", "license",
    }
    support_policy = support_mesh_profiles["policy"]
    if support_policy.get("collision_authority") != (
        "exact_static_proxy_when_selected"
    ):
        raise ValueError("support mesh policy does not select exact collision")
    if not bool(support_policy.get("immutable_binding_required")):
        raise ValueError("support mesh policy does not require immutable binding")
    if support_policy.get("fallback_phase") != "before_metadata_freeze_only":
        raise ValueError("support mesh fallback is allowed after metadata freeze")
    for profile in support_mesh_profiles["profiles"]:
        if profile.get("visual_type") != "mesh_support":
            raise ValueError(f"invalid support mesh visual type: {profile['id']}")
        missing = support_mesh_required - set(profile)
        if missing:
            raise ValueError(
                f"support mesh profile lacks {sorted(missing)}: {profile['id']}"
            )
        if not set(profile["support_ids"]) <= active_support_ids:
            raise ValueError(f"support mesh references inactive support: {profile['id']}")
        if len(profile["source_bbox_size"]) != 3 or min(profile["source_bbox_size"]) <= 0:
            raise ValueError(f"invalid support mesh bounds: {profile['id']}")
        if not 0 < float(profile["source_support_plane_z_from_bottom"]) <= float(
            profile["source_bbox_size"][2]
        ):
            raise ValueError(f"invalid source support plane: {profile['id']}")
        if profile["material_policy"] not in {
            "embedded_pbr", "support_surface_pbr_override"
        }:
            raise ValueError(f"invalid support material policy: {profile['id']}")
        proxy_record = physical_proxy_by_id[str(profile["asset_id"])]
        if proxy_record["proxy"]["representation"] != "static_concave_mesh":
            raise ValueError(
                f"support mesh lacks a static catalog proxy: {profile['id']}"
            )
        if not proxy_record["admission"]["sampling_ready"]:
            raise ValueError(
                f"support mesh catalog proxy is not sampling-ready: {profile['id']}"
            )
        active_usage_ids = {
            str(usage["id"])
            for usage in proxy_record["proxy"]["usages"]
            if bool(usage.get("active"))
        }
        expected_usage_ids = {
            f"generic_support:{support_id}"
            for support_id in profile["support_ids"]
        }
        if not expected_usage_ids <= active_usage_ids:
            raise ValueError(
                f"support mesh lacks active generic usages: {profile['id']}"
            )

    registry_by_asset = {
        str(record["asset_id"]): record
        for record in asset_proxy_registry["records"]
    }
    for profile in mesh_profiles:
        asset_id = str(profile["asset"]["asset_id"])
        registry_record = registry_by_asset[asset_id]
        if registry_record["asset_role"] != "static_environment":
            raise ValueError(f"mesh backdrop lacks a static physical role: {profile['id']}")
        if registry_record["proxy"]["kind"] != "static_environment_mesh":
            raise ValueError(f"mesh backdrop lacks collision: {profile['id']}")
        if not registry_record["proxy"].get("colliders"):
            raise ValueError(f"mesh backdrop proxy is empty: {profile['id']}")

    proxy_kinds = {
        "none",
        "dynamic_rigid",
        "static_compound",
        "support_compound",
        "static_environment_mesh",
    }
    for record in asset_proxy_registry["records"]:
        kind = str(record["proxy"]["kind"])
        if kind not in proxy_kinds:
            raise ValueError(f"invalid unified asset proxy kind: {record['asset_id']}")
        if kind == "none" and record["proxy"].get("colliders"):
            raise ValueError(f"proxy none has colliders: {record['asset_id']}")
        if kind != "none" and not record["proxy"].get("colliders"):
            raise ValueError(f"physical asset proxy has no colliders: {record['asset_id']}")

    result = copy.deepcopy(base)
    mesh_alignment_coordinate_frame = str(
        object_source["policy"]["mesh_alignment_coordinate_frame"]
    )
    result["axes"]["object_axis"] = [
        compile_object_profile(profile, mesh_alignment_coordinate_frame)
        for profile in object_source["profiles"]
    ]
    collision_motion_exclusions = object_source["policy"].get(
        "collision_motion_exclusions", {}
    )
    unknown_collision_types = set(collision_motion_exclusions) - {
        "sphere", "cuboid", "cylinder"
    }
    if unknown_collision_types:
        raise ValueError(
            "object policy excludes motions for unknown collision types: "
            f"{sorted(unknown_collision_types)}"
        )
    for obj in result["axes"]["object_axis"]:
        obj["excluded_motion_families"] = sorted(
            set(obj["excluded_motion_families"])
            | {
                str(motion)
                for motion in collision_motion_exclusions.get(obj["shape"], [])
            }
        )
    result["axes"]["support_axis"] = [
        compile_scene_kit(kit)
        for kit in scene_source["kits"]
        if bool(kit.get("sampling_enabled", True))
    ]
    result["architecture"] = {
        "bundle_version": str(bundle["version"]),
        "object_profiles_version": str(object_source["version"]),
        "object_visual_curation_version": str(object_visual_curation["version"]),
        "object_visual_curation_counts": copy.deepcopy(
            object_visual_curation["counts"]
        ),
        "scene_kits_version": str(scene_source["version"]),
        "compatibility_version": str(compatibility["version"]),
        "backend_capabilities_version": str(backend_capabilities["version"]),
        "compatibility": compatibility,
        "visual_sampling": copy.deepcopy(
            object_source.get("visual_sampling", {"target_mesh_fraction": 0.0})
        ),
        "scene_visual_profiles": {
            "version": (
                f"{scene_visual_profiles['version']}+{scene_mesh_profiles['version']}"
            ),
            "policy": {
                **copy.deepcopy(scene_visual_profiles["policy"]),
                **copy.deepcopy(scene_mesh_profiles["policy"]),
            },
            "sampling": copy.deepcopy(scene_mesh_profiles["sampling"]),
            "profiles": all_scene_visual_profiles,
        },
        "environment_composition": {
            "version": str(environment_composition["schema_version"]),
            "policy": copy.deepcopy(environment_composition["policy"]),
        },
        "environment_collision_proxies": copy.deepcopy(
            environment_collision_proxies
        ),
        "support_mesh_profiles": copy.deepcopy(support_mesh_profiles),
        "asset_proxy_registry": copy.deepcopy(asset_proxy_registry),
        "physical_proxy_catalog": {
            "version": str(physical_proxy_manifest["version"]),
            "manifest_path": str(bundle["physical_proxy_catalog"]),
            "records_sha256": str(physical_proxy_manifest["records_sha256"]),
            "counts": copy.deepcopy(physical_proxy_manifest["counts"]),
            "active_records": [
                copy.deepcopy(record)
                for record in physical_proxy_records
                if bool(record["admission"]["active_matrix_selected"])
            ],
        },
    }
    validate_generic_capabilities(result, backend, backend_capabilities)
    for motion in result["axes"]["motion_axis"]:
        if motion not in compatibility["motions"]:
            raise ValueError(f"motion lacks compatibility rule: {motion}")
    active_motions = set(result["axes"]["motion_axis"])
    for obj in result["axes"]["object_axis"]:
        unknown = set(obj["excluded_motion_families"]) - active_motions
        if unknown:
            raise ValueError(
                f"object excludes unknown motion families: {obj['label']}: {sorted(unknown)}"
            )
    for support in result["axes"]["support_axis"]:
        unknown_motions = set(support.get("allowed_motions", [])) - active_motions
        if unknown_motions:
            raise ValueError(
                f"scene kit admits unknown motions: {support['label']}: "
                f"{sorted(unknown_motions)}"
            )
        unknown_categories = set(support.get("environment_categories", [])) - set(
            visual_rules["environment_categories"]
        )
        if unknown_categories:
            raise ValueError(
                f"scene kit admits unknown environment categories: "
                f"{support['label']}: {sorted(unknown_categories)}"
            )
        if not any(
            support_is_compatible(support, motion, compatibility)
            for motion in result["axes"]["motion_axis"]
        ):
            raise ValueError(f"scene kit is unreachable: {support['label']}")
    return result
