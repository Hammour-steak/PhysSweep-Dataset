#!/usr/bin/env python3
"""Compile reviewed visual environments into immutable scene profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def indexed(records: list[dict[str, Any]], key: str, label: str) -> dict[str, dict[str, Any]]:
    result = {str(record[key]): record for record in records}
    if len(result) != len(records):
        raise ValueError(f"duplicate {key} in {label}")
    return result


def merge_indexed_documents(
    paths: list[Path], collection_key: str, record_key: str, label: str
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    merged: dict[str, dict[str, Any]] = {}
    sources: list[str] = []
    for path in paths:
        document = load_json(path)
        records = indexed(document[collection_key], record_key, str(path))
        overlap = set(merged) & set(records)
        if overlap:
            raise ValueError(f"duplicate {record_key} across {label}: {sorted(overlap)}")
        merged.update(records)
        sources.append(str(path))
    return merged, sources


def project_relative(path: str) -> str:
    source = Path(path)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    source = source.resolve()
    try:
        return str(source.relative_to(PROJECT_ROOT.resolve()))
    except ValueError as exc:
        raise ValueError(f"asset is outside project root: {source}") from exc


def normalization_axis(review_yaw: int) -> tuple[str, str]:
    yaw = review_yaw % 360
    if yaw in {0, 180}:
        return "x", "y"
    if yaw in {90, 270}:
        return "y", "x"
    raise ValueError(f"review yaw must be a right angle: {review_yaw}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, action="append", required=True)
    parser.add_argument("--downloads", type=Path, action="append", required=True)
    parser.add_argument("--inspection", type=Path, action="append", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--scene-kits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--version", default="physweep_scene_mesh_profiles_v4"
    )
    args = parser.parse_args()

    annotations_doc = load_json(args.annotations)
    scene_kits_doc = load_json(args.scene_kits)
    candidates, candidate_sources = merge_indexed_documents(
        args.candidates, "candidates", "candidate_id", "candidates"
    )
    downloads, download_sources = merge_indexed_documents(
        args.downloads, "records", "candidate_id", "downloads"
    )
    inspections, inspection_sources = merge_indexed_documents(
        args.inspection, "records", "candidate_id", "inspection"
    )
    reviews = indexed(annotations_doc["reviews"], "candidate_id", "annotations")
    scene_kits = indexed(scene_kits_doc["kits"], "id", "scene kits")
    sampling_support_ids = {
        support_id
        for support_id, support in scene_kits.items()
        if bool(support.get("sampling_enabled", False))
    }
    expected_ids = set(candidates)
    for label, values in (("downloads", downloads), ("inspection", inspections), ("annotations", reviews)):
        if set(values) != expected_ids:
            raise ValueError(f"{label} candidate set does not match source manifest")

    accepted_reviews = [record for record in annotations_doc["reviews"] if record["decision"] == "accepted"]
    target = int(annotations_doc["policy"]["target_accepted_count"])
    render_category_by_semantic_group = {
        str(key): str(value)
        for key, value in annotations_doc["policy"][
            "render_category_by_semantic_group"
        ].items()
    }
    if len(accepted_reviews) != target:
        raise ValueError(f"expected {target} accepted environments, got {len(accepted_reviews)}")
    profiles = []
    covered_support_ids: set[str] = set()
    for review in accepted_reviews:
        candidate_id = str(review["candidate_id"])
        candidate = candidates[candidate_id]
        download = downloads[candidate_id]
        inspection = inspections[candidate_id]
        if inspection.get("status") != "ok":
            raise ValueError(f"failed Blender import: {candidate_id}")
        if int(inspection.get("texture_image_count", 0)) < 1:
            raise ValueError(f"accepted environment has no image texture: {candidate_id}")
        camera_context = review.get("camera_context")
        if not isinstance(camera_context, dict):
            raise ValueError(f"accepted environment lacks camera_context: {candidate_id}")
        required_context_fields = {
            "depth_offset_m",
            "lateral_offset_m",
            "target_z_offset_m",
            "focal_length_cap_mm",
        }
        optional_context_fields = {
            "minimum_elevation_degrees",
            "maximum_elevation_degrees",
        }
        if not required_context_fields.issubset(camera_context) or (
            set(camera_context) - required_context_fields - optional_context_fields
        ):
            raise ValueError(f"invalid camera_context fields: {candidate_id}")
        depth_offset = float(camera_context["depth_offset_m"])
        lateral_offset = float(camera_context["lateral_offset_m"])
        target_z_offset = float(camera_context["target_z_offset_m"])
        focal_cap = float(camera_context["focal_length_cap_mm"])
        if not 0.0 <= depth_offset <= 2.5:
            raise ValueError(f"camera depth offset is out of range: {candidate_id}")
        if not -1.0 <= lateral_offset <= 1.0:
            raise ValueError(f"camera lateral offset is out of range: {candidate_id}")
        if not 0.0 <= target_z_offset <= 1.0:
            raise ValueError(f"camera z offset is out of range: {candidate_id}")
        if not 28.0 <= focal_cap <= 44.0:
            raise ValueError(f"camera focal cap is out of range: {candidate_id}")
        minimum_elevation = camera_context.get("minimum_elevation_degrees")
        maximum_elevation = camera_context.get("maximum_elevation_degrees")
        if minimum_elevation is not None:
            minimum_elevation = float(minimum_elevation)
            if not 10.0 <= minimum_elevation <= 50.0:
                raise ValueError(f"camera minimum elevation is out of range: {candidate_id}")
        if maximum_elevation is not None:
            maximum_elevation = float(maximum_elevation)
            if not 10.0 <= maximum_elevation <= 50.0:
                raise ValueError(f"camera maximum elevation is out of range: {candidate_id}")
        if (
            minimum_elevation is not None
            and maximum_elevation is not None
            and minimum_elevation > maximum_elevation
        ):
            raise ValueError(f"camera elevation limits are reversed: {candidate_id}")
        excluded_names = [
            str(value) for value in review.get("exclude_object_names", [])
        ]
        excluded_prefixes = [
            str(value) for value in review.get("exclude_object_name_prefixes", [])
        ]
        if len(excluded_names) != len(set(excluded_names)):
            raise ValueError(f"duplicate excluded object name: {candidate_id}")
        if len(excluded_prefixes) != len(set(excluded_prefixes)):
            raise ValueError(f"duplicate excluded object prefix: {candidate_id}")
        if any(not value for value in [*excluded_names, *excluded_prefixes]):
            raise ValueError(f"empty excluded object selector: {candidate_id}")
        if excluded_names or excluded_prefixes:
            inspected_names = {
                str(value) for value in inspection.get("mesh_object_names", [])
            }
            if not inspected_names:
                raise ValueError(
                    f"shell editing requires inspected mesh names: {candidate_id}"
                )
            missing_names = set(excluded_names) - inspected_names
            missing_prefixes = [
                prefix
                for prefix in excluded_prefixes
                if not any(name.startswith(prefix) for name in inspected_names)
            ]
            if missing_names or missing_prefixes:
                raise ValueError(
                    f"reviewed shell selectors do not match import: {candidate_id} "
                    f"names={sorted(missing_names)} prefixes={missing_prefixes}"
                )
            remaining_names = {
                name
                for name in inspected_names
                if name not in excluded_names
                and not any(name.startswith(prefix) for prefix in excluded_prefixes)
            }
            if not remaining_names:
                raise ValueError(f"shell editing removes every mesh: {candidate_id}")
        raw_face_exclusions = review.get("source_space_face_exclusions", [])
        if not isinstance(raw_face_exclusions, list):
            raise ValueError(f"source-space face exclusions must be a list: {candidate_id}")
        source_space_face_exclusions = list(raw_face_exclusions)
        source_bbox_min = [float(value) for value in inspection["bbox_min"]]
        source_bbox_max = [float(value) for value in inspection["bbox_max"]]
        source_axis_index = {"x": 0, "y": 1, "z": 2}
        normalized_face_exclusions = []
        exclusion_signatures = set()
        for selector in source_space_face_exclusions:
            if not isinstance(selector, dict) or set(selector) != {
                "axis",
                "comparison",
                "value",
            }:
                raise ValueError(f"invalid source-space face selector: {candidate_id}")
            axis = str(selector["axis"])
            comparison = str(selector["comparison"])
            if axis not in source_axis_index or comparison not in {
                "at_or_above",
                "at_or_below",
            }:
                raise ValueError(f"unsupported source-space face selector: {candidate_id}")
            value = float(selector["value"])
            index = source_axis_index[axis]
            if not source_bbox_min[index] < value < source_bbox_max[index]:
                raise ValueError(f"shell face threshold is outside source bounds: {candidate_id}")
            signature = (axis, comparison, value)
            if signature in exclusion_signatures:
                raise ValueError(f"duplicate source-space face selector: {candidate_id}")
            exclusion_signatures.add(signature)
            normalized_face_exclusions.append(
                {"axis": axis, "comparison": comparison, "value": value}
            )
        source_size = [float(value) for value in inspection["bbox_size"]]
        width_axis, depth_axis = normalization_axis(int(review["review_yaw_degrees"]))
        axis_index = {"x": 0, "y": 1, "z": 2}
        target_extent = float(review["target_extent_m"])
        scale = target_extent / source_size[axis_index[width_axis]]
        floor_alignment = review.get("floor_alignment")
        if not isinstance(floor_alignment, dict):
            raise ValueError(f"accepted environment lacks floor_alignment: {candidate_id}")
        required_floor_fields = {
            "method",
            "source_floor_z",
            "surface_overlap_m",
            "evidence",
        }
        if set(floor_alignment) != required_floor_fields:
            raise ValueError(f"invalid floor_alignment fields: {candidate_id}")
        source_bbox_min_z = float(inspection["bbox_min"][2])
        source_bbox_max_z = float(inspection["bbox_max"][2])
        floor_method = str(floor_alignment["method"])
        if floor_method == "source_bbox_min":
            if floor_alignment["source_floor_z"] is not None:
                raise ValueError(f"bbox-min floor must omit source z: {candidate_id}")
            source_floor_z = source_bbox_min_z
        elif floor_method == "reviewed_horizontal_surface":
            source_floor_z = float(floor_alignment["source_floor_z"])
            if not source_bbox_min_z <= source_floor_z <= source_bbox_max_z:
                raise ValueError(f"reviewed floor is outside source bounds: {candidate_id}")
        else:
            raise ValueError(f"unsupported floor alignment method: {candidate_id}")
        surface_overlap = float(floor_alignment["surface_overlap_m"])
        if not 0.0 <= surface_overlap <= 0.10:
            raise ValueError(f"surface overlap is out of range: {candidate_id}")
        anchor_z = -(
            (source_floor_z - source_bbox_min_z) * scale
        ) - surface_overlap
        scaled_depth = source_size[axis_index[depth_axis]] * scale
        wall_distance = 0.5 * scaled_depth + float(review["front_clearance_m"])
        support_ids = [str(value) for value in review["support_ids"]]
        if not support_ids or len(support_ids) != len(set(support_ids)):
            raise ValueError(f"invalid support_ids: {candidate_id}")
        unknown_support_ids = set(support_ids) - sampling_support_ids
        if unknown_support_ids:
            raise ValueError(
                f"environment references unavailable supports: {candidate_id} "
                f"{sorted(unknown_support_ids)}"
            )
        covered_support_ids.update(support_ids)
        themes = sorted({str(scene_kits[value]["theme"]) for value in support_ids})
        scene_classes = sorted(
            {str(scene_kits[value]["scene_class"]) for value in support_ids}
        )
        semantic_environment_group = str(review["environment_category"])
        if semantic_environment_group not in render_category_by_semantic_group:
            raise ValueError(
                f"semantic environment group lacks render category: "
                f"{semantic_environment_group}"
            )
        profiles.append(
            {
                "id": str(review["profile_id"]),
                "visual_type": "mesh_backdrop",
                "environment_category": render_category_by_semantic_group[
                    semantic_environment_group
                ],
                "semantic_environment_group": semantic_environment_group,
                "semantic_environment": str(review["semantic_environment"]),
                "quality_tier": str(review["quality_tier"]),
                "themes": themes,
                "scene_classes": scene_classes,
                "support_ids": support_ids,
                "back_wall_distance_m": round(wall_distance, 6),
                "wall_enabled": False,
                "side_wall": None,
                "decor": [],
                "camera_context": {
                    "depth_offset_m": depth_offset,
                    "lateral_offset_m": lateral_offset,
                    "target_z_offset_m": target_z_offset,
                    "focal_length_cap_mm": focal_cap,
                    **(
                        {"minimum_elevation_degrees": minimum_elevation}
                        if minimum_elevation is not None
                        else {}
                    ),
                    **(
                        {"maximum_elevation_degrees": maximum_elevation}
                        if maximum_elevation is not None
                        else {}
                    ),
                },
                "asset": {
                    "asset_id": candidate_id,
                    "path": project_relative(str(download["archive_path"])),
                    "sha256": str(download["sha256"]),
                    "source_bbox_size": source_size,
                    "normalization_axis": width_axis,
                    "target_extent_m": target_extent,
                    "front_view_yaw_degrees": float((-int(review["review_yaw_degrees"])) % 360),
                    "lateral_offset_m": float(review.get("lateral_offset_m", 0.0)),
                    "outward_offset_m": float(review.get("outward_offset_m", 0.0)),
                    "anchor_z_m": round(anchor_z, 6),
                    "floor_alignment": {
                        "method": floor_method,
                        "source_bbox_min_z": source_bbox_min_z,
                        "source_floor_z": source_floor_z,
                        "scale": round(scale, 9),
                        "surface_overlap_m": surface_overlap,
                        "computed_anchor_z_m": round(anchor_z, 6),
                        "evidence": str(floor_alignment["evidence"]),
                    },
                    "exclude_object_names": excluded_names,
                    "exclude_object_name_prefixes": excluded_prefixes,
                    "source_space_face_exclusions": normalized_face_exclusions,
                    "license": str(download["license"]["label"]),
                    "source_uid": str(candidate["source_uid"]),
                    "review_yaw_degrees": int(review["review_yaw_degrees"]),
                    "front_clearance_m": float(review["front_clearance_m"])
                }
            }
        )

    missing_support_ids = sampling_support_ids - covered_support_ids
    if missing_support_ids:
        raise ValueError(
            f"sampling supports lack a mesh environment: {sorted(missing_support_ids)}"
        )

    output = {
        "version": str(args.version),
        "sampling": {
            "target_mesh_fraction": 0.4,
            "exact_batch_ratio": True
        },
        "policy": {
            "visual_only": True,
            "requires_image_texture": True,
            "requires_hash_match": True,
            "never_changes_collision_or_trajectory": True,
            "placement_anchor": "camera_relative_behind_motion",
            "accepted_profile_count": target,
            "per_asset_web_review_required": True,
            "per_asset_multiview_review_required": True,
            "closed_shells_without_reviewed_opening_forbidden": True,
            "reviewed_shell_opening_selectors_are_import_validated": True,
            "panorama_spheres_forbidden": True,
            "distance_rule": "half_scaled_depth_plus_reviewed_front_clearance",
            "compatibility_rule": "exact_reviewed_support_ids",
            "floor_alignment_rule": (
                "anchor=-(reviewed_floor_z-source_bbox_min_z)*scale-surface_overlap"
            ),
        },
        "sources": {
            "candidates": candidate_sources,
            "downloads": download_sources,
            "inspection": inspection_sources,
            "annotations": str(args.annotations),
            "scene_kits": str(args.scene_kits)
        },
        "profiles": profiles
    }
    write_json(args.output, output)
    print("profiles", len(profiles))
    for profile in profiles:
        print(
            profile["id"],
            profile["asset"]["normalization_axis"],
            profile["asset"]["target_extent_m"],
            profile["back_wall_distance_m"]
        )


if __name__ == "__main__":
    main()
