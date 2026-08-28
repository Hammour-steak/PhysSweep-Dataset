#!/usr/bin/env python3
"""Compile reviewed support downloads into versioned registry and composition files."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json_sorted as write_json
from tools.assets.build_asset_proxy_registry import validate_record


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolved(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def build_support_record(
    download: dict[str, Any],
    measurement: dict[str, Any],
    annotation: dict[str, Any],
) -> dict[str, Any]:
    plane_rank = int(annotation["plane_rank"])
    planes = measurement["dominant_upward_planes"]
    if not 0 <= plane_rank < len(planes):
        raise ValueError(f"invalid plane rank for {annotation['asset_id']}")
    plane = planes[plane_rank]
    source_bounds = [
        float(value)
        for value in annotation.get("source_support_bounds_xy", plane["xy_bounds"])
    ]
    target_size = [float(value) for value in annotation["target_support_size_xy_m"]]
    target_z = float(annotation["target_support_plane_z_m"])
    thickness = float(annotation["collider_thickness_m"])
    collider_fraction = [
        float(value)
        for value in annotation.get("collider_size_fraction_xy", [1.0, 1.0])
    ]
    safe_fraction = [float(value) for value in annotation["safe_surface_fraction_xy"]]
    exit_directions = [
        [float(value) for value in direction]
        for direction in annotation.get("clear_exit_directions_xy", [])
    ]
    if (
        len(source_bounds) != 4
        or source_bounds[1] <= source_bounds[0]
        or source_bounds[3] <= source_bounds[2]
        or len(target_size) != 2
        or min(target_size) <= 0.0
        or len(collider_fraction) != 2
        or min(collider_fraction) <= 0.0
        or max(collider_fraction) > 1.0
        or len(safe_fraction) != 2
        or min(safe_fraction) <= 0.0
        or max(safe_fraction) >= 1.0
        or not 0.0 < thickness < target_z
        or any(
            len(direction) != 2
            or not math.isclose(
                math.hypot(*direction), 1.0, rel_tol=1.0e-6, abs_tol=1.0e-8
            )
            for direction in exit_directions
        )
        or ("edge_exit" in annotation.get("profiles", []) and not exit_directions)
    ):
        raise ValueError(f"invalid support annotation: {annotation['asset_id']}")
    collider_size = [
        target_size[axis] * collider_fraction[axis] for axis in range(2)
    ]
    shape = str(annotation["collider_shape"])
    if shape == "cylinder":
        if abs(collider_size[0] - collider_size[1]) > 1.0e-8:
            raise ValueError(f"cylinder support must be circular: {annotation['asset_id']}")
    elif shape != "box":
        raise ValueError(f"unsupported support collider: {shape}")
    record = {
        "asset_id": str(annotation["asset_id"]),
        "name": str(download["name"]),
        "semantic_category": str(download["semantic_category"]),
        "asset_role": "interactive_support",
        "visual": {
            "path": str(download["archive_path"]),
            "sha256": str(download["sha256"]),
            "source_bbox_size": [float(value) for value in measurement["bbox_size"]],
            "alignment_euler_degrees": [0.0, 0.0, 0.0],
            "source_support_plane_z_from_bottom": float(plane["z_from_bottom"]),
            "source_support_bounds_xy": source_bounds,
            "target_support_size_xy_m": target_size,
            "scale_policy": "support_surface_frame_nonuniform_v1",
            "license": str(download["license"]["label"]),
        },
        "proxy": {
            "kind": "support_compound",
            "body_type": "static",
            "colliders": [
                {
                    "id": "tabletop",
                    "shape": shape,
                    "size_m": [*collider_size, thickness],
                    "position_m": [0.0, 0.0, target_z - thickness * 0.5],
                    "rotation_euler_degrees": [0.0, 0.0, 0.0],
                }
            ],
            "usable_surfaces": [
                {
                    "id": "tabletop",
                    "size_xy_m": [
                        target_size[axis] * safe_fraction[axis] for axis in range(2)
                    ],
                    "center_xy_m": [0.0, 0.0],
                    "z_m": target_z,
                }
            ],
        },
        "admission": {
            "sampling_enabled": True,
            "status": "approved",
            "reason": str(annotation["review_reason"]),
        },
        "review": {
            "geometry_gate": "measured_dominant_support_plane_v2",
            "visual_status": "turntable_reviewed_2026_08_04",
            "physics_status": "exact_static_mesh_probe_required_v2",
            "review_reason": str(annotation["review_reason"]),
        },
    }
    if exit_directions:
        record["proxy"]["interaction_policy"] = {
            "clear_exit_directions_xy": exit_directions,
            "edge_exit_lateral_center_fraction": 0.0,
        }
    return record


def composition_record(
    record: dict[str, Any], annotation: dict[str, Any]
) -> dict[str, Any]:
    profiles = set(str(value) for value in annotation.get("profiles", []))
    allowed = []
    if profiles & {"resting_push", "diagonal_push"}:
        allowed.extend(["generic_tabletop_motion", "slide_push"])
    if "vertical_drop" in profiles:
        allowed.append("drop_to_surface")
    if "edge_exit" in profiles:
        allowed.append("edge_exit")
    if not allowed:
        raise ValueError(f"support has no scene capability: {record['asset_id']}")
    return {
        "asset_id": str(record["asset_id"]),
        "name": str(record["name"]),
        "sampling_status": "ready_generic",
        "handling": "keep_whole_support",
        "component_policy": {
            "mode": "keep_whole",
            "interaction_surface": "registered_continuous_tabletop",
        },
        "scene_fit": {
            "allowed": allowed,
            "forbidden": [],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--base-registry", type=Path, required=True)
    parser.add_argument("--base-composition", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--registry-output", type=Path, required=True)
    parser.add_argument("--composition-output", type=Path, required=True)
    parser.add_argument("--registry-version", required=True)
    parser.add_argument("--composition-version", required=True)
    parser.add_argument("--base-matrix", type=Path)
    parser.add_argument("--matrix-output", type=Path)
    parser.add_argument("--matrix-version")
    parser.add_argument("--sampling-bundle", type=Path)
    parser.add_argument("--physical-proxy-catalog", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    base_registry = load_json(resolved(root, args.base_registry))
    base_composition = load_json(resolved(root, args.base_composition))
    downloads = load_json(resolved(root, args.download_manifest))["records"]
    measurements = load_json(resolved(root, args.measurements))["records"]
    annotations_document = load_json(resolved(root, args.annotations))
    by_download = {str(item["candidate_id"]): item for item in downloads}
    by_measurement = {str(item["asset_id"]): item for item in measurements}
    annotations = {
        str(item["asset_id"]): item for item in annotations_document["records"]
    }
    if set(by_download) != set(by_measurement) or set(by_download) != set(annotations):
        raise ValueError("download, measurement, and annotation coverage differs")
    admitted = [
        build_support_record(by_download[asset_id], by_measurement[asset_id], annotation)
        for asset_id, annotation in annotations.items()
        if annotation["decision"] == "admit"
    ]
    rejected = {
        asset_id: str(annotation["reason"])
        for asset_id, annotation in annotations.items()
        if annotation["decision"] == "reject"
    }
    unknown = sorted(
        asset_id
        for asset_id, annotation in annotations.items()
        if annotation["decision"] not in {"admit", "reject"}
    )
    if unknown:
        raise ValueError(f"unknown support decisions: {unknown}")

    registry_records = copy.deepcopy(base_registry["records"]) + admitted
    registry_records.sort(key=lambda item: str(item["asset_id"]))
    ids = [str(item["asset_id"]) for item in registry_records]
    if len(ids) != len(set(ids)):
        raise ValueError("expanded registry contains duplicate asset ids")
    for record in registry_records:
        validate_record(record)
    registry = {
        "version": str(args.registry_version),
        "policy": {
            **copy.deepcopy(base_registry["policy"]),
            "support_expansion": {
                "annotations": str(args.annotations),
                "admitted": [str(item["asset_id"]) for item in admitted],
                "rejected": rejected,
            },
        },
        "counts": {
            "total": len(registry_records),
            "by_asset_role": dict(
                sorted(Counter(item["asset_role"] for item in registry_records).items())
            ),
            "by_proxy_kind": dict(
                sorted(Counter(item["proxy"]["kind"] for item in registry_records).items())
            ),
            "sampling_enabled": sum(
                bool(item["admission"]["sampling_enabled"]) for item in registry_records
            ),
        },
        "records": registry_records,
    }

    composition_records = copy.deepcopy(base_composition["records"])
    composition_records.extend(
        composition_record(record, annotations[str(record["asset_id"])])
        for record in admitted
    )
    composition_records.sort(key=lambda item: str(item["asset_id"]))
    composition = {
        "version": str(args.composition_version),
        "policy": {
            **copy.deepcopy(base_composition["policy"]),
            "review_date": "2026-08-04",
        },
        "records": composition_records,
    }
    write_json(resolved(root, args.registry_output), registry)
    write_json(resolved(root, args.composition_output), composition)
    matrix_args = [
        args.base_matrix,
        args.matrix_output,
        args.matrix_version,
        args.sampling_bundle,
        args.physical_proxy_catalog,
    ]
    if any(matrix_args) and not all(matrix_args):
        raise ValueError("all matrix expansion arguments must be provided together")
    matrix_summary = None
    if args.base_matrix is not None:
        matrix = copy.deepcopy(load_json(resolved(root, args.base_matrix)))
        matrix["version"] = str(args.matrix_version)
        matrix["dependencies"]["generic_sampling_bundle"] = (
            args.sampling_bundle.as_posix()
        )
        matrix["dependencies"]["asset_proxy_registry"] = (
            args.registry_output.as_posix()
        )
        matrix["dependencies"]["physical_proxy_catalog"] = (
            args.physical_proxy_catalog.as_posix()
        )
        matrix["dependencies"]["asset_scene_composition"] = (
            args.composition_output.as_posix()
        )
        environments = {
            str(item["id"]): item for item in matrix["environments"]
        }
        curated = environments["curated_support_asset"]
        existing_supports = {
            str(item["support_asset_id"])
            for item in curated["support_dynamic_entries"]
        }
        additions = []
        valid_pools = set(curated["dynamic_pools"])
        valid_profiles = {
            profile
            for values in curated["motion_bindings"].values()
            for profile in values
        }
        for record in admitted:
            annotation = annotations[str(record["asset_id"])]
            pool_id = str(annotation["dynamic_pool_id"])
            profiles = [str(value) for value in annotation["profiles"]]
            if pool_id not in valid_pools or not set(profiles) <= valid_profiles:
                raise ValueError(
                    f"unsupported matrix binding for {record['asset_id']}"
                )
            if str(record["asset_id"]) in existing_supports:
                raise ValueError(
                    f"support already exists in base matrix: {record['asset_id']}"
                )
            additions.append(
                {
                    "support_asset_id": str(record["asset_id"]),
                    "dynamic_pool_id": pool_id,
                    "profiles": profiles,
                }
            )
        curated["support_dynamic_entries"].extend(additions)
        scene_support_ids = {
            str(item["support_asset_id"])
            for item in curated["support_dynamic_entries"]
        }
        for environment_id in ("billiards_single_ball", "workbench_single_object"):
            scene_support_ids.update(
                str(value)
                for value in environments[environment_id]["support_asset_ids"]
            )
        if len(scene_support_ids) != 20:
            raise ValueError(
                f"expanded matrix has {len(scene_support_ids)} support scenes, expected 20"
            )
        matrix["policy"]["declared_interactive_support_scene_count"] = 20
        matrix["policy"]["support_scene_expansion_source"] = (
            args.annotations.as_posix()
        )
        write_json(resolved(root, args.matrix_output), matrix)
        matrix_summary = {
            "new_support_entries": len(additions),
            "interactive_support_scenes": len(scene_support_ids),
        }
    print(
        json.dumps(
            {
                "registry_counts": registry["counts"],
                "new_supports": len(admitted),
                "rejected_downloads": len(rejected),
                "matrix": matrix_summary,
            },
            indent=2,
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
