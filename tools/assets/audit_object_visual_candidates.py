#!/usr/bin/env python3
"""Preflight every raw PhysAssets visual before curation and sampling."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.core.json_io import read_json as load_json
from tools.core.json_io import read_jsonl
from tools.core.json_io import write_json
from tools.assets.repair_physassets_visuals import (
    clear_scene,
    imported_meshes,
    mesh_signature,
    render_review,
    sha256,
)
from tools.assets.physical_proxy_catalog import load_catalog, records_by_id
from tools.assets.static_support_proxy import record_sha256


def blender_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/object_visual_preflight.json"),
    )
    return parser.parse_args(values)


def project_relative_path(root: Path, value: str) -> str:
    normalized = str(value).replace("\\", "/")
    root_text = root.as_posix().rstrip("/")
    if normalized.startswith(f"{root_text}/"):
        return normalized[len(root_text) + 1 :]
    marker = "/physweep/"
    if marker in normalized:
        return normalized.rsplit(marker, 1)[1]
    path = Path(normalized)
    if not path.is_absolute():
        return path.as_posix()
    raise ValueError(f"evidence path is outside the project: {value}")


def load_proxy_evidence_context(
    root: Path, evidence_policy: dict[str, Any]
) -> dict[str, Any]:
    core_path = root / str(evidence_policy["core_index"])
    overlay_root = root / str(evidence_policy["overlay_root"])
    catalog_path = root / str(evidence_policy["physical_proxy_catalog"])
    if not core_path.is_file() or not overlay_root.is_dir():
        raise FileNotFoundError("proxy review evidence is incomplete")
    core_records = read_jsonl(core_path)
    core_by_sample = {str(record["sample_id"]): record for record in core_records}
    if len(core_by_sample) != len(core_records):
        raise ValueError("proxy review core index has duplicate sample ids")
    catalog_manifest, catalog_records = load_catalog(
        root, catalog_path, require_runtime_validation=True
    )
    validation_descriptor = catalog_manifest.get("validation")
    if not isinstance(validation_descriptor, dict):
        raise ValueError("physical proxy catalog has no runtime validation")
    validation_path = root / str(validation_descriptor["path"])
    validation_report = load_json(validation_path)
    if validation_report.get("catalog_records_sha256") != catalog_manifest[
        "records_sha256"
    ]:
        raise ValueError("physical proxy validation targets a stale catalog")
    validation_records = [
        record
        for record in validation_report.get("records", [])
        if record.get("probe") == "analytic_drop"
    ]
    validation_by_asset = {
        str(record["asset_id"]): record for record in validation_records
    }
    if len(validation_by_asset) != len(validation_records):
        raise ValueError("physical proxy analytic-drop validation has duplicate asset ids")
    required_views = [str(value) for value in evidence_policy["required_overlay_views"]]
    if len(required_views) != len(set(required_views)) or not required_views:
        raise ValueError("proxy overlay view policy is invalid")
    return {
        "core_path": core_path,
        "core_records": core_records,
        "core_by_sample": core_by_sample,
        "overlay_root": overlay_root,
        "required_views": required_views,
        "catalog_path": catalog_path,
        "catalog_manifest": catalog_manifest,
        "catalog_by_asset": records_by_id(catalog_records),
        "validation_path": validation_path,
        "validation_report": validation_report,
        "validation_by_asset": validation_by_asset,
    }


def proxy_evidence_for_visual(
    root: Path,
    profile: dict[str, Any],
    visual: dict[str, Any],
    evidence_policy: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    profile_id = str(profile["id"])
    asset_id = str(visual["asset_id"])
    sample_id = str(profile["source_review"]["sample_id"])
    core = context["core_by_sample"].get(sample_id)
    if core is None:
        raise ValueError(f"profile has no proxy review core record: {profile_id}")
    if core.get("final_status") != "core" or core.get("final_review_reasons"):
        raise ValueError(f"proxy review is not finally admitted: {profile_id}")
    alignment = core.get("blender_alignment_fit", {})
    alignment_error = float(alignment.get("maximum_relative_error", math.inf))
    maximum_alignment_error = float(
        evidence_policy["maximum_alignment_relative_error"]
    )
    if not bool(alignment.get("passed")) or alignment_error > maximum_alignment_error:
        raise ValueError(f"proxy alignment review failed: {profile_id}")
    profile_alignment_error = float(
        profile["source_review"]["alignment_relative_error"]
    )
    if not math.isclose(
        alignment_error, profile_alignment_error, rel_tol=1.0e-6, abs_tol=1.0e-9
    ):
        raise ValueError(f"proxy alignment evidence is stale: {profile_id}")
    fit_ratio = float(core["proxy_to_visual_hull_volume_ratio"])
    maximum_fit_ratio = float(
        evidence_policy["maximum_proxy_to_visual_hull_volume_ratio"]
    )
    if fit_ratio > maximum_fit_ratio:
        raise ValueError(f"proxy fit ratio exceeds policy: {profile_id}")
    source_path = project_relative_path(root, str(core["source_glb"]))
    proxy_path = project_relative_path(root, str(core["proxy_json"]))
    if source_path != str(visual["path"]):
        raise ValueError(f"proxy review visual path is stale: {profile_id}")
    if proxy_path != str(profile["source_review"]["proxy_json"]):
        raise ValueError(f"proxy review JSON path is stale: {profile_id}")
    if str(core["method"]) != str(profile["source_review"]["proxy_method"]):
        raise ValueError(f"proxy review method is stale: {profile_id}")
    proxy_file = root / proxy_path
    if not proxy_file.is_file():
        raise FileNotFoundError(f"source proxy JSON is missing: {profile_id}")

    catalog_record = context["catalog_by_asset"].get(asset_id)
    if catalog_record is None:
        raise ValueError(f"candidate has no physical proxy catalog record: {asset_id}")
    admission = catalog_record["admission"]
    if (
        catalog_record["classification"]["role"] != "dynamic_object"
        or not bool(admission["active_matrix_selected"])
        or not bool(admission["sampling_ready"])
        or catalog_record["proxy"]["representation"] != "analytic_compound"
    ):
        raise ValueError(f"candidate proxy is not active and sampleable: {asset_id}")
    colliders = catalog_record["proxy"]["colliders"]
    expected_shape = {
        "cuboid": "box",
        "sphere": "sphere",
        "cylinder": "cylinder",
    }[str(profile["collision"]["type"])]
    expected_size = [float(value) for value in profile["collision"]["dimensions_m"]]
    if len(colliders) != 1 or str(colliders[0]["shape"]) != expected_shape:
        raise ValueError(f"candidate catalog proxy shape differs: {asset_id}")
    collider_size = [float(value) for value in colliders[0]["size_m"]]
    if any(
        not math.isclose(left, right, rel_tol=1.0e-8, abs_tol=1.0e-9)
        for left, right in zip(collider_size, expected_size)
    ):
        raise ValueError(f"candidate catalog proxy dimensions differ: {asset_id}")
    catalog_source = catalog_record["source"]
    if (
        str(catalog_source.get("active_profile_id")) != profile_id
        or str(catalog_source.get("proxy_record_path")) != proxy_path
        or str(catalog_source.get("proxy_record_sha256")) != sha256(proxy_file)
    ):
        raise ValueError(f"candidate catalog source binding differs: {asset_id}")
    catalog_visual_path = root / str(catalog_source.get("visual_path", ""))
    if (
        not catalog_visual_path.is_file()
        or sha256(catalog_visual_path) != str(catalog_source.get("sha256"))
    ):
        raise ValueError(f"candidate catalog visual binding differs: {asset_id}")
    catalog_fit = float(
        catalog_record["qa"]["fit_quality"]["proxy_to_visual_hull_volume_ratio"]
    )
    if not math.isclose(catalog_fit, fit_ratio, rel_tol=1.0e-8, abs_tol=1.0e-9):
        raise ValueError(f"candidate catalog fit evidence differs: {asset_id}")

    probe = context["validation_by_asset"].get(asset_id)
    if (
        probe is None
        or not bool(probe.get("passed"))
        or probe.get("probe") != "analytic_drop"
        or probe.get("representation") != "analytic_compound"
    ):
        raise ValueError(f"candidate proxy lacks a passing runtime probe: {asset_id}")

    overlay_views = []
    for view in context["required_views"]:
        matches = sorted(context["overlay_root"].glob(f"{sample_id}_*_{view}.png"))
        if len(matches) != 1:
            raise ValueError(
                f"proxy overlay evidence is ambiguous or missing: {asset_id}:{view}"
            )
        overlay_views.append(
            {
                "view": view,
                "path": matches[0].relative_to(root).as_posix(),
                "sha256": sha256(matches[0]),
            }
        )
    return {
        "status": "verified",
        "sample_id": sample_id,
        "core_record_sha256": record_sha256(core),
        "proxy_method": str(core["method"]),
        "source_proxy": {"path": proxy_path, "sha256": sha256(proxy_file)},
        "proxy_to_visual_hull_volume_ratio": fit_ratio,
        "blender_alignment_fit": alignment,
        "overlay_views": overlay_views,
        "physical_proxy_catalog": {
            "asset_id": asset_id,
            "catalog_record_sha256": record_sha256(catalog_record),
            "catalog_records_sha256": str(
                context["catalog_manifest"]["records_sha256"]
            ),
            "representation": str(catalog_record["proxy"]["representation"]),
            "collider_shape": str(colliders[0]["shape"]),
            "collider_size_m": collider_size,
            "qa_grade": str(catalog_record["qa"]["grade"]),
            "catalog_visual": {
                "path": str(catalog_source["visual_path"]),
                "sha256": str(catalog_source["sha256"]),
            },
        },
        "runtime_probe": {
            "record_sha256": record_sha256(probe),
            "probe": str(probe["probe"]),
            "passed": bool(probe["passed"]),
        },
    }


def material_records(meshes) -> list[dict[str, Any]]:
    materials = {}
    for obj in meshes:
        for slot in obj.material_slots:
            material = slot.material
            if material is None or material.name in materials:
                continue
            alpha_default = None
            alpha_links = 0
            if material.use_nodes and material.node_tree:
                shader = material.node_tree.nodes.get("Principled BSDF")
                if shader is not None:
                    alpha_default = float(shader.inputs["Alpha"].default_value)
                    alpha_links = len(shader.inputs["Alpha"].links)
            materials[material.name] = {
                "name": material.name,
                "blend_method": str(material.blend_method),
                "diffuse_alpha": float(material.diffuse_color[3]),
                "principled_alpha_default": alpha_default,
                "principled_alpha_links": alpha_links,
                "use_backface_culling": bool(material.use_backface_culling),
            }
    return [materials[name] for name in sorted(materials)]


def opaque_issues(
    materials: list[dict[str, Any]], allowed_transparent: set[str]
) -> list[str]:
    issues = []
    for material in materials:
        name = str(material["name"])
        if name in allowed_transparent:
            continue
        if material["blend_method"] != "OPAQUE":
            issues.append(f"non_opaque_blend_method:{name}")
        if float(material["diffuse_alpha"]) < 0.999:
            issues.append(f"diffuse_alpha_below_one:{name}")
        alpha_default = material["principled_alpha_default"]
        if alpha_default is not None and float(alpha_default) < 0.999:
            issues.append(f"principled_alpha_below_one:{name}")
        if int(material["principled_alpha_links"]) > 0:
            issues.append(f"principled_alpha_linked:{name}")
    return issues


def nonopaque_material_names(materials: list[dict[str, Any]]) -> set[str]:
    names = set()
    for material in materials:
        alpha_default = material["principled_alpha_default"]
        if (
            material["blend_method"] != "OPAQUE"
            or float(material["diffuse_alpha"]) < 0.999
            or (alpha_default is not None and float(alpha_default) < 0.999)
            or int(material["principled_alpha_links"]) > 0
        ):
            names.add(str(material["name"]))
    return names


def hash_views(root: Path, paths: list[str]) -> list[dict[str, str]]:
    return [
        {
            "path": str(Path(path).relative_to(root)),
            "sha256": sha256(Path(path)),
        }
        for path in paths
    ]


def review_metrics(path: Path, background_threshold: float) -> dict[str, Any]:
    import bpy
    import numpy as np

    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = (int(value) for value in image.size)
        pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(
            height, width, 4
        )[:, :, :3]
        border = max(2, min(width, height) // 32)
        corners = np.concatenate(
            [
                pixels[:border, :border].reshape(-1, 3),
                pixels[:border, -border:].reshape(-1, 3),
                pixels[-border:, :border].reshape(-1, 3),
                pixels[-border:, -border:].reshape(-1, 3),
            ],
            axis=0,
        )
        background = np.median(corners, axis=0)
        distance = np.linalg.norm(pixels - background, axis=2)
        subject = distance >= background_threshold
        occupancy = float(np.mean(subject))
        if np.any(subject):
            rows, columns = np.nonzero(subject)
            bounds = {
                "x_min_fraction": float(columns.min() / width),
                "x_max_fraction": float((columns.max() + 1) / width),
                "y_min_fraction": float(rows.min() / height),
                "y_max_fraction": float((rows.max() + 1) / height),
            }
        else:
            bounds = None
        return {
            "view": path.stem,
            "subject_fraction": occupancy,
            "subject_bounds": bounds,
            "background_rgb": [float(value) for value in background],
        }
    finally:
        bpy.data.images.remove(image)


def process_visual(
    root: Path,
    evidence_root: Path,
    profile: dict[str, Any],
    visual: dict[str, Any],
    opaque_categories: set[str],
    transparency_exception: dict[str, Any] | None,
    review_policy: dict[str, Any],
    proxy_evidence_policy: dict[str, Any],
    proxy_evidence_context: dict[str, Any],
) -> dict[str, Any]:
    import bpy

    profile_id = str(profile["id"])
    asset_id = str(visual["asset_id"])
    source = root / str(visual["path"])
    if not source.is_file():
        raise FileNotFoundError(source)
    actual_sha = sha256(source)
    if actual_sha != str(visual["sha256"]):
        raise ValueError(f"source hash mismatch: {profile_id}:{asset_id}")

    clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(source))
    meshes = imported_meshes()
    if not meshes:
        raise ValueError(f"candidate has no mesh: {profile_id}:{asset_id}")
    signature = mesh_signature(meshes)
    if min(float(value) for value in signature["extent"]) <= 0.0:
        raise ValueError(f"candidate has a degenerate extent: {profile_id}:{asset_id}")
    materials = material_records(meshes)
    if not materials:
        raise ValueError(f"candidate has no material: {profile_id}:{asset_id}")

    category = str(profile["semantic_category"])
    allowed_transparent = set(
        transparency_exception["material_names"] if transparency_exception else []
    )
    material_names = {str(material["name"]) for material in materials}
    if not allowed_transparent <= material_names:
        raise ValueError(f"reviewed transparency material is stale: {asset_id}")
    if not allowed_transparent <= nonopaque_material_names(materials):
        raise ValueError(f"reviewed transparency exception is no longer needed: {asset_id}")
    issues = (
        opaque_issues(materials, allowed_transparent)
        if category in opaque_categories
        else []
    )
    review_dir = evidence_root / asset_id / "raw_review"
    review_paths = render_review(meshes, review_dir)
    review_views = hash_views(root, review_paths)
    metrics = [
        review_metrics(
            Path(path),
            float(review_policy["review_background_distance_threshold"]),
        )
        for path in review_paths
    ]
    minimum_subject = float(review_policy["minimum_review_subject_fraction"])
    maximum_subject = float(review_policy["maximum_review_subject_fraction"])
    visibility_issues = []
    for metric in metrics:
        fraction = float(metric["subject_fraction"])
        if fraction < minimum_subject:
            visibility_issues.append(f"subject_not_visible:{metric['view']}")
        elif fraction > maximum_subject:
            visibility_issues.append(f"subject_overfills_frame:{metric['view']}")
    if visibility_issues:
        disposition = "manual_review_required"
    elif issues:
        disposition = "repair_required"
    else:
        disposition = "source_verified"
    operations = ["force_opaque_materials"] if issues else []
    return {
        "profile_id": profile_id,
        "visual_asset_id": asset_id,
        "semantic_category": category,
        "source_visual": {"path": str(visual["path"]), "sha256": actual_sha},
        "collision_proxy": {
            "type": str(profile["collision"]["type"]),
            "dimensions_m": list(profile["collision"]["dimensions_m"]),
            "proxy_json": str(profile["source_review"]["proxy_json"]),
        },
        "proxy_evidence": proxy_evidence_for_visual(
            root,
            profile,
            visual,
            proxy_evidence_policy,
            proxy_evidence_context,
        ),
        "mesh_signature": signature,
        "materials": materials,
        "issues": issues,
        "visibility_issues": visibility_issues,
        "disposition": disposition,
        "recommended_operations": operations,
        "reviewed_transparency": transparency_exception,
        "review_views": review_views,
        "review_metrics": metrics,
    }


def main() -> None:
    import numpy as np

    if "bool" not in np.__dict__:
        np.bool = np.bool_
    args = blender_args()
    root = args.root.resolve()
    policy_path = args.policy if args.policy.is_absolute() else root / args.policy
    policy = load_json(policy_path)
    profiles_path = root / str(policy["source_profiles"])
    profiles = load_json(profiles_path)
    report_path = root / str(policy["output_report"])
    evidence_root = root / str(policy["review_root"])
    opaque_categories = set(policy["policy"]["opaque_semantic_categories"])
    transparency_exceptions = {
        str(record["visual_asset_id"]): record
        for record in policy.get("reviewed_transparency_exceptions", [])
    }
    if len(transparency_exceptions) != len(
        policy.get("reviewed_transparency_exceptions", [])
    ):
        raise ValueError("duplicate reviewed transparency exception")
    proxy_evidence_policy = policy["proxy_evidence"]
    proxy_evidence_context = load_proxy_evidence_context(
        root, proxy_evidence_policy
    )
    records = []
    for profile in profiles["profiles"]:
        mesh_visuals = [
            visual
            for visual in profile["visual_variants"]
            if visual["type"] == "mesh"
        ]
        if not mesh_visuals:
            raise ValueError(f"profile has no mesh visual: {profile['id']}")
        for visual in mesh_visuals:
            records.append(
                process_visual(
                    root,
                    evidence_root,
                    profile,
                    visual,
                    opaque_categories,
                    transparency_exceptions.get(str(visual["asset_id"])),
                    policy["policy"],
                    proxy_evidence_policy,
                    proxy_evidence_context,
                )
            )

    expected_ids = {
        str(profile["id"])
        for profile in profiles["profiles"]
    }
    actual_ids = {str(record["profile_id"]) for record in records}
    if actual_ids != expected_ids:
        raise ValueError("candidate preflight did not cover every selected profile")
    expected_samples = {
        str(profile["source_review"]["sample_id"])
        for profile in profiles["profiles"]
    }
    if set(proxy_evidence_context["core_by_sample"]) != expected_samples:
        raise ValueError("proxy review core index differs from the selected profiles")
    actual_assets = {str(record["visual_asset_id"]) for record in records}
    if not set(transparency_exceptions) <= actual_assets:
        raise ValueError("reviewed transparency exception does not bind to a visual")
    report = {
        "version": str(policy["report_version"]),
        "policy": {"path": str(policy_path.relative_to(root)), "sha256": sha256(policy_path)},
        "source": {
            "path": str(profiles_path.relative_to(root)),
            "sha256": sha256(profiles_path),
            "version": str(profiles["version"]),
            "profile_count": len(expected_ids),
        },
        "proxy_evidence_sources": {
            "core_index": {
                "path": proxy_evidence_context["core_path"].relative_to(root).as_posix(),
                "sha256": sha256(proxy_evidence_context["core_path"]),
                "record_count": len(proxy_evidence_context["core_records"]),
            },
            "overlay_root": {
                "path": proxy_evidence_context["overlay_root"].relative_to(root).as_posix(),
                "required_views": proxy_evidence_context["required_views"],
            },
            "physical_proxy_catalog": {
                "path": proxy_evidence_context["catalog_path"].relative_to(root).as_posix(),
                "sha256": sha256(proxy_evidence_context["catalog_path"]),
                "records_sha256": str(
                    proxy_evidence_context["catalog_manifest"]["records_sha256"]
                ),
            },
            "physical_proxy_validation": {
                "path": proxy_evidence_context["validation_path"].relative_to(root).as_posix(),
                "sha256": sha256(proxy_evidence_context["validation_path"]),
                "version": str(
                    proxy_evidence_context["validation_report"]["version"]
                ),
                "catalog_records_sha256": str(
                    proxy_evidence_context["validation_report"][
                        "catalog_records_sha256"
                    ]
                ),
                "record_count": len(
                    proxy_evidence_context["validation_report"]["records"]
                ),
            },
        },
        "complete_profile_set": True,
        "records": records,
        "counts": {
            "profiles": len(expected_ids),
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
                record["reviewed_transparency"] is not None for record in records
            ),
            "proxy_evidence_verified": sum(
                record["proxy_evidence"]["status"] == "verified"
                for record in records
            ),
        },
    }
    write_json(report_path, report)
    print(json.dumps({"report": str(report_path), **report["counts"]}, indent=2))


if __name__ == "__main__":
    main()
