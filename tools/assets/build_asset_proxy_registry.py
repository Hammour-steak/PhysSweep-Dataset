#!/usr/bin/env python3
"""Build and validate the explicit Sketchfab asset proxy registry."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json_sorted as write_json
from tools.core.rigid_geometry import positive_vector as _positive_vector


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def validate_collider(collider: dict[str, Any], asset_id: str) -> None:
    shape = str(collider["shape"])
    if shape not in {"box", "sphere", "cylinder"}:
        raise ValueError(f"unsupported collider shape for {asset_id}: {shape}")
    _positive_vector(collider["size_m"], 3, f"collider size for {asset_id}")
    if len(collider.get("position_m", [])) != 3:
        raise ValueError(f"invalid collider position for {asset_id}")
    if len(collider.get("rotation_euler_degrees", [])) != 3:
        raise ValueError(f"invalid collider rotation for {asset_id}")


def validate_record(record: dict[str, Any]) -> None:
    required = {
        "asset_id", "name", "semantic_category", "asset_role", "visual",
        "proxy", "admission", "review",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"asset record lacks {sorted(missing)}: {record.get('asset_id')}")
    visual = record["visual"]
    if len(str(visual["sha256"])) != 64:
        raise ValueError(f"invalid asset hash: {record['asset_id']}")
    include_names = [str(value) for value in visual.get("include_object_names", [])]
    if len(include_names) != len(set(include_names)) or any(not value for value in include_names):
        raise ValueError(f"invalid visual component selection: {record['asset_id']}")
    variant_names = [str(value) for value in visual.get("variant_object_names", [])]
    if len(variant_names) != len(set(variant_names)) or any(not value for value in variant_names):
        raise ValueError(f"invalid visual component variants: {record['asset_id']}")
    if include_names and variant_names:
        raise ValueError(f"fixed and variant visual selection conflict: {record['asset_id']}")
    proxy = record["proxy"]
    kind = str(proxy["kind"])
    if kind not in {"none", "dynamic_rigid", "static_compound", "support_compound"}:
        raise ValueError(f"invalid proxy kind for {record['asset_id']}: {kind}")
    colliders = proxy.get("colliders", [])
    if kind == "none" and colliders:
        raise ValueError(f"proxy none has colliders: {record['asset_id']}")
    if kind != "none" and not colliders:
        raise ValueError(f"physical proxy has no colliders: {record['asset_id']}")
    collider_ids = [str(item["id"]) for item in colliders]
    if len(collider_ids) != len(set(collider_ids)):
        raise ValueError(f"duplicate collider id: {record['asset_id']}")
    for collider in colliders:
        validate_collider(collider, str(record["asset_id"]))
    if kind == "support_compound":
        raw_bounds = visual.get("source_support_bounds_xy")
        aligned_bounds = visual.get("source_support_bounds_xy_aligned_relative")
        if (raw_bounds is None) == (aligned_bounds is None):
            raise ValueError(
                f"support requires exactly one source bounds representation: {record['asset_id']}"
            )
        source_bounds = [float(value) for value in (raw_bounds or aligned_bounds)]
        if (
            len(source_bounds) != 4
            or source_bounds[1] <= source_bounds[0]
            or source_bounds[3] <= source_bounds[2]
        ):
            raise ValueError(f"invalid source support bounds: {record['asset_id']}")
        if float(visual.get("source_support_plane_z_from_bottom", 0.0)) <= 0.0:
            raise ValueError(f"invalid source support plane: {record['asset_id']}")
        _positive_vector(
            visual.get("target_support_size_xy_m", []),
            2,
            f"target support size for {record['asset_id']}",
        )
        surfaces = proxy.get("usable_surfaces", [])
        if not surfaces:
            raise ValueError(f"support proxy lacks usable surface: {record['asset_id']}")
        for surface in surfaces:
            _positive_vector(surface["size_xy_m"], 2, f"surface size for {record['asset_id']}")
            if len(surface["center_xy_m"]) != 2 or float(surface["z_m"]) <= 0.0:
                raise ValueError(f"invalid support surface for {record['asset_id']}")


def deep_update(target: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def foreground_record(
    source: dict[str, Any], annotation: dict[str, Any] | None = None
) -> dict[str, Any]:
    preprocessing = source["preprocessing"]
    simulation = source["simulation_profile"]
    proxy = simulation["collision_proxy"]
    shape = {"box": "box", "sphere": "sphere", "cylinder": "cylinder"}[
        str(proxy["proxy_type"])
    ]
    status = str(source["admission_status"])
    approved = status == "physweep_v0_approved"
    record = {
        "asset_id": str(source["candidate_id"]),
        "name": str(source["name"]),
        "semantic_category": str(source["semantic_category"]),
        "asset_role": "dynamic_object",
        "visual": {
            "path": str(source["archive_path"]),
            "sha256": str(source["sha256"]),
            "source_bbox_size": [float(v) for v in preprocessing["source_bbox"]],
            "alignment_euler_degrees": [
                float(v) for v in preprocessing["visual_alignment_degrees"]
            ],
            "canonical_extent_m": [
                float(v) for v in preprocessing["normalized_visual_extent_m"]
            ],
            "scale_policy": "uniform_preserve_aspect",
            "license": str(source["license"]["label"]),
        },
        "proxy": {
            "kind": "dynamic_rigid",
            "body_type": "dynamic",
            "colliders": [
                {
                    "id": "body",
                    "shape": shape,
                    "size_m": [float(v) for v in proxy["extent_m"]],
                    "position_m": [0.0, 0.0, 0.0],
                    "rotation_euler_degrees": [0.0, 0.0, 0.0],
                }
            ],
            "mass_range_kg": [float(v) for v in simulation["canonical_scale"]["mass_range_kg"]],
            "material": copy.deepcopy(simulation["default_material"]),
        },
        "admission": {
            "sampling_enabled": approved,
            "status": "approved" if approved else "review",
            "reason": source.get("notes", ""),
        },
        "review": {
            "geometry_gate": str(preprocessing["geometry_gate"]),
            "visual_status": str(preprocessing["visual_review_status"]),
            "physics_status": "probe_required_v1",
            "review_reason": preprocessing.get("review_reason"),
        },
    }
    if annotation:
        deep_update(record, {key: value for key, value in annotation.items() if key != "asset_id"})
    return record


def background_record(source: dict[str, Any], annotation: dict[str, Any]) -> dict[str, Any]:
    record = copy.deepcopy(annotation)
    record.update(
        {
            "asset_id": str(source["candidate_id"]),
            "name": str(source["name"]),
            "semantic_category": str(source["semantic_category"]),
            "visual": {
                "path": str(source["archive_path"]),
                "sha256": str(source["sha256"]),
                "license": str(source["license"]["label"]),
                **copy.deepcopy(annotation.get("visual", {})),
            },
        }
    )
    return record


def build_registry(
    background_manifest: dict[str, Any],
    foreground_manifest: dict[str, Any],
    annotations: dict[str, Any],
    foreground_annotations: dict[str, Any],
) -> dict[str, Any]:
    background = {str(item["candidate_id"]): item for item in background_manifest["records"]}
    annotated = {str(item["asset_id"]): item for item in annotations["records"]}
    if set(background) != set(annotated):
        missing = sorted(set(background) - set(annotated))
        extra = sorted(set(annotated) - set(background))
        raise ValueError(f"background annotation coverage mismatch missing={missing} extra={extra}")
    records = [background_record(background[key], annotated[key]) for key in sorted(background)]
    foreground_overrides = {
        str(item["asset_id"]): item for item in foreground_annotations["records"]
    }
    foreground_ids = {str(item["candidate_id"]) for item in foreground_manifest["records"]}
    if not set(foreground_overrides) <= foreground_ids:
        raise ValueError("foreground annotation references an unknown asset")
    records.extend(
        foreground_record(item, foreground_overrides.get(str(item["candidate_id"])))
        for item in foreground_manifest["records"]
    )
    records.sort(key=lambda item: str(item["asset_id"]))
    ids = [str(item["asset_id"]) for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate asset id in unified registry")
    for record in records:
        validate_record(record)
    return {
        "version": "physweep_asset_proxy_registry_v1",
        "policy": copy.deepcopy(annotations["policy"]),
        "counts": {
            "total": len(records),
            "by_asset_role": dict(sorted(Counter(item["asset_role"] for item in records).items())),
            "by_proxy_kind": dict(sorted(Counter(item["proxy"]["kind"] for item in records).items())),
            "sampling_enabled": sum(bool(item["admission"]["sampling_enabled"]) for item in records),
        },
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--background-manifest",
        type=Path,
        default=PROJECT_ROOT / "assets/manifests/sketchfab_background_admission_v1.json",
    )
    parser.add_argument(
        "--foreground-manifest",
        type=Path,
        default=PROJECT_ROOT / "assets/manifests/sketchfab_foreground_source_v1.json",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=PROJECT_ROOT
        / "configs/source_annotations/background_asset_proxy_annotations_v1.json",
    )
    parser.add_argument(
        "--foreground-annotations",
        type=Path,
        default=PROJECT_ROOT
        / "configs/source_annotations/foreground_asset_proxy_annotations_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = build_registry(
        load_json(args.background_manifest),
        load_json(args.foreground_manifest),
        load_json(args.annotations),
        load_json(args.foreground_annotations),
    )
    write_json(args.output, registry)
    print(json.dumps(registry["counts"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
