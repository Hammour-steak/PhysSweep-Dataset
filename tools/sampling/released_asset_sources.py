"""Normalize released 1obj asset-proxy objects for object-count consumers.

This module verifies and converts one released dynamic asset only. Scene-specific
shape, motion, host and camera admission remain the caller's responsibility.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from tools.core.hashing import sha256_file as sha256, sha256_json
from tools.core.json_io import read_json
from tools.sampling.released_object_sources import declared_within


def asset_scale_bin(size: np.ndarray, eligibility: dict[str, Any]) -> str:
    extent = float(np.max(size))
    for scale_bin, maximum in eligibility[
        "asset_scale_bin_maximum_extent_m"
    ].items():
        if maximum is None or extent <= float(maximum) + 1.0e-12:
            return str(scale_bin)
    raise ValueError("asset scale-bin contract has no open final interval")


def asset_proxy_definition(
    colliders: Any, eligibility: dict[str, Any], asset_id: str
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    """Return the layout envelope and exact admitted collision declaration."""
    if not isinstance(colliders, list) or not colliders:
        raise ValueError(f"asset source has no collision proxy: {asset_id}")
    canonical: list[dict[str, Any]] = []
    for collider in colliders:
        if not isinstance(collider, dict):
            raise ValueError(f"asset source has an invalid proxy: {asset_id}")
        position = np.asarray(collider.get("position_m"), dtype=np.float64)
        rotation = np.asarray(
            collider.get("rotation_euler_degrees"), dtype=np.float64
        )
        size = np.asarray(collider.get("size_m"), dtype=np.float64)
        shape = str(collider.get("shape", ""))
        if (
            position.shape != (3,)
            or rotation.shape != (3,)
            or size.shape != (3,)
            or not np.isfinite(position).all()
            or not np.isfinite(rotation).all()
            or not np.isfinite(size).all()
            or bool(np.any(size <= 0.0))
            or shape not in {"box", "sphere", "cylinder"}
            or (
                shape == "sphere"
                and not np.allclose(size, size[0], atol=1.0e-8, rtol=0.0)
            )
            or (
                shape == "cylinder"
                and not np.isclose(size[0], size[1], atol=1.0e-8, rtol=0.0)
            )
        ):
            raise ValueError(f"asset source has an invalid proxy: {asset_id}")
        canonical.append(
            {
                "shape": shape,
                "size_m": size.tolist(),
                "position_m": position.tolist(),
                "rotation_euler_degrees": rotation.tolist(),
            }
        )

    if len(canonical) == 1:
        collider = canonical[0]
        if not np.allclose(collider["position_m"], 0.0, atol=1.0e-8, rtol=0.0):
            return None, None, "offset_primitive_proxy"
        if not np.allclose(
            collider["rotation_euler_degrees"], 0.0, atol=1.0e-8, rtol=0.0
        ):
            return None, None, "rotated_primitive_proxy"
        shape = {"box": "cuboid"}.get(collider["shape"], collider["shape"])
        geometry = {"type": shape, "size_m": collider["size_m"]}
        return geometry, copy.deepcopy(geometry), "eligible"

    if any(collider["shape"] != "cylinder" for collider in canonical):
        return None, None, "non_axisymmetric_compound_proxy"
    if any(
        not np.allclose(
            collider["rotation_euler_degrees"], 0.0, atol=1.0e-8, rtol=0.0
        )
        for collider in canonical
    ):
        return None, None, "rotated_compound_proxy"
    if any(
        not np.allclose(collider["position_m"][:2], 0.0, atol=1.0e-8, rtol=0.0)
        for collider in canonical
    ):
        return None, None, "off_axis_compound_proxy"
    z_intervals = sorted(
        (
            collider["position_m"][2] - collider["size_m"][2] * 0.5,
            collider["position_m"][2] + collider["size_m"][2] * 0.5,
        )
        for collider in canonical
    )
    connected_upper = z_intervals[0][1]
    for lower, upper in z_intervals[1:]:
        if lower > connected_upper + 1.0e-8:
            return None, None, "disconnected_compound_proxy"
        connected_upper = max(connected_upper, upper)
    lower_z = z_intervals[0][0]
    upper_z = connected_upper
    center_offset = abs((lower_z + upper_z) * 0.5)
    if center_offset > float(
        eligibility["asset_proxy_maximum_aabb_center_offset_m"]
    ) + 1.0e-12:
        return None, None, "off_center_compound_proxy"
    diameter = max(float(collider["size_m"][0]) for collider in canonical)
    geometry = {
        "type": "cylinder",
        "size_m": [diameter, diameter, upper_z - lower_z],
    }
    return geometry, {"type": "compound", "colliders": canonical}, "eligible"


def asset_object_template(
    *,
    source_root: Path,
    runtime_root: Path,
    generation_metadata: dict[str, Any],
    release_metadata: dict[str, Any],
    eligibility: dict[str, Any],
    registry_cache: dict[tuple[str, str], dict[str, dict[str, Any]]],
    visual_hash_cache: dict[str, str],
    include_compound_inertia_reference: bool = False,
) -> tuple[dict[str, Any] | None, str]:
    registry_binding = generation_metadata.get("registry")
    if not isinstance(registry_binding, dict) or set(registry_binding) != {
        "path",
        "sha256",
    }:
        raise ValueError("asset source lacks a pinned registry")
    registry_path = declared_within(
        source_root, Path(str(registry_binding["path"]))
    )
    registry_hash = str(registry_binding["sha256"])
    if sha256(registry_path) != registry_hash:
        raise ValueError("asset source registry hash mismatch")
    cache_key = (str(registry_path), registry_hash)
    if cache_key not in registry_cache:
        registry = read_json(registry_path)
        records = registry.get("records")
        if not isinstance(records, list):
            raise ValueError("asset source registry has no records")
        by_id = {str(record["asset_id"]): record for record in records}
        if len(by_id) != len(records):
            raise ValueError("asset source registry has duplicate ids")
        registry_cache[cache_key] = by_id
    asset_id = str(generation_metadata["assets"]["dynamic_asset_id"])
    dynamic = registry_cache[cache_key].get(asset_id)
    if dynamic is None:
        raise ValueError(f"asset source is absent from its registry: {asset_id}")
    geometry, collision, reason = asset_proxy_definition(
        dynamic.get("proxy", {}).get("colliders"), eligibility, asset_id
    )
    if geometry is None or collision is None:
        return None, reason
    size = np.asarray(geometry["size_m"], dtype=np.float64)
    visual = dynamic.get("visual", {})
    canonical_extent = np.asarray(visual.get("canonical_extent_m"), dtype=np.float64)
    alignment = np.asarray(
        visual.get("alignment_euler_degrees"), dtype=np.float64
    )
    if (
        canonical_extent.shape != (3,)
        or not np.isfinite(canonical_extent).all()
        or bool(np.any(canonical_extent <= 0.0))
        or alignment.shape != (3,)
        or not np.isfinite(alignment).all()
        or not str(visual.get("path", ""))
        or not str(visual.get("sha256", ""))
    ):
        raise ValueError(f"asset source lacks a canonical visual extent: {asset_id}")
    visual_path = declared_within(runtime_root, Path(str(visual["path"])))
    visual_path_key = str(visual_path)
    actual_visual_hash = visual_hash_cache.get(visual_path_key)
    if actual_visual_hash is None:
        actual_visual_hash = sha256(visual_path)
        visual_hash_cache[visual_path_key] = actual_visual_hash
    if actual_visual_hash != str(visual["sha256"]):
        raise ValueError(f"asset source visual hash mismatch: {asset_id}")
    ratios = size / canonical_extent
    uniform_ratio = float(np.median(ratios))
    relative_error = np.abs(canonical_extent * uniform_ratio - size) / size
    if float(np.max(relative_error)) > 0.06:
        return None, "visual_proxy_extent_mismatch"
    release_objects = release_metadata.get("physics", {}).get("objects")
    semantic_objects = release_metadata.get("semantics", {}).get("objects")
    if (
        not isinstance(release_objects, list)
        or len(release_objects) != 1
        or not isinstance(semantic_objects, list)
        or len(semantic_objects) != 1
        or str(release_objects[0].get("asset_id", "")) != asset_id
    ):
        raise ValueError(f"released asset source identity is inconsistent: {asset_id}")
    material = copy.deepcopy(release_objects[0]["material"])
    if collision["type"] == "compound" and include_compound_inertia_reference:
        reference_mass = float(material["mass_kg"])
        released_mass = float(release_objects[0]["material"]["mass_kg"])
        inertia = np.asarray(
            release_objects[0].get("inertia_diagonal_kg_m2"), dtype=np.float64
        )
        if (
            not np.isfinite(reference_mass)
            or reference_mass <= 0.0
            or not np.isclose(reference_mass, released_mass, atol=1.0e-12, rtol=0.0)
            or inertia.shape != (3,)
            or not np.isfinite(inertia).all()
            or bool(np.any(inertia <= 0.0))
        ):
            raise ValueError(
                f"released compound asset lacks a valid inertia reference: {asset_id}"
            )
        collision["inertia_reference"] = {
            "schema_version": "physweep_compound_inertia_reference_v1",
            "method": "released_base_runtime_v1",
            "reference_mass_kg": reference_mass,
            "diagonal_kg_m2": inertia.tolist(),
            "collision_profile_sha256": sha256_json(
                {"type": "compound", "colliders": collision["colliders"]}
            ),
        }
    semantic_label = str(semantic_objects[0].get("semantic_label", "")).strip()
    if not semantic_label:
        raise ValueError(f"released asset source has no semantic label: {asset_id}")
    template = {
        "schema_version": "physweep_pybullet_rigid_metadata_v1",
        "scene_id": str(generation_metadata["scene_id"]),
        "sweep": {"kind": "base"},
        "simulation": {
            "objects": [
                {
                    "object_id": "object_a",
                    "body_model": eligibility["body_model"],
                    "semantic_type": semantic_label,
                    "geometry": geometry,
                    "collision_profile": collision,
                    "material": material,
                    "initial_state": {
                        "pose_profile": eligibility["required_pose_profile"],
                        "position_m": [0.0, 0.0, 0.0],
                        "orientation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "linear_velocity_m_s": [0.0, 0.0, 0.0],
                        "angular_velocity_rad_s": [0.0, 0.0, 0.0],
                    },
                    "visual_profile": {
                        "id": asset_id,
                        "type": "mesh",
                        "asset_id": asset_id,
                        "path": str(visual["path"]),
                        "sha256": str(visual["sha256"]),
                        "alignment_coordinate_frame": "blender_imported_z_up",
                        "alignment_euler_degrees": alignment.tolist(),
                        "material_policy": "source_or_bound_fallback",
                    },
                }
            ]
        },
        "semantic_sampling": {
            "five_dimensions": {
                "foreground_object": {
                    "semantic_category": semantic_label,
                    "scale_bin": asset_scale_bin(size, eligibility),
                    "uniform_scale": 1.0,
                }
            }
        },
        "appearance": {"materials": {}},
    }
    return template, "eligible"
