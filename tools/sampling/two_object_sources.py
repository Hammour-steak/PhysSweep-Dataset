"""Resolve released 1obj metadata into strict 2obj object and host sources."""

from __future__ import annotations

import copy
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json


def declared_within(root: Path, value: Path) -> Path:
    candidate = value if value.is_absolute() else root / value
    absolute = Path(os.path.abspath(candidate))
    absolute.relative_to(root)
    return absolute


def _release_metadata_records(
    *,
    root: Path,
    released_path: Path,
    released: dict[str, Any],
    families: tuple[str, ...],
) -> dict[str, tuple[str, Path, dict[str, Any], str]]:
    base_root = released_path.parent
    result: dict[str, tuple[str, Path, dict[str, Any], str]] = {}
    for family in families:
        binding = released.get("pipelines", {}).get(family)
        if not isinstance(binding, dict) or set(binding) != {
            "manifest",
            "manifest_sha256",
        }:
            raise ValueError(f"released 1obj base lacks the {family} pipeline")
        manifest_path = declared_within(base_root, Path(str(binding["manifest"])))
        if sha256(manifest_path) != str(binding["manifest_sha256"]):
            raise ValueError(f"released {family} pipeline manifest hash mismatch")
        manifest = read_json(manifest_path)
        records = manifest.get("records")
        if (
            manifest.get("schema_version") != "physweep_base_pipeline_view_v12"
            or manifest.get("pipeline") != family
            or not isinstance(records, list)
            or int(manifest.get("sample_count", -1)) != len(records)
        ):
            raise ValueError(f"released {family} pipeline manifest is invalid")
        for record in records:
            scene_id = str(record.get("scene_id", ""))
            metadata_path = manifest_path.parent / scene_id / "metadata.json"
            expected_hash = str(record.get("metadata_sha256", ""))
            if (
                not scene_id
                or scene_id in result
                or not expected_hash
                or sha256(metadata_path) != expected_hash
            ):
                raise ValueError(f"released {family} sample is invalid: {scene_id}")
            metadata = read_json(metadata_path)
            if (
                metadata.get("schema_version") != "physweep_base_sample_v11"
                or str(metadata.get("scene_id", "")) != scene_id
            ):
                raise ValueError(f"released {family} metadata identity is invalid")
            result[scene_id] = (family, metadata_path, metadata, expected_hash)
    return result


def _source_reference(
    *,
    root: Path,
    source_root: Path,
    generation_record: dict[str, Any],
    generation_metadata: dict[str, Any],
    release_record: tuple[str, Path, dict[str, Any], str],
) -> dict[str, Any]:
    family, release_path, _, release_hash = release_record
    generation_path = declared_within(
        source_root, Path(str(generation_record["path"]))
    )
    return {
        "scene_id": str(generation_metadata["scene_id"]),
        "source_family": family,
        "release_metadata": {
            "path": release_path.relative_to(root).as_posix(),
            "sha256": release_hash,
        },
        "generation_metadata": {
            "path": generation_path.relative_to(source_root).as_posix(),
            "sha256": str(generation_record["metadata_sha256"]),
        },
    }


def _asset_scale_bin(size: np.ndarray, eligibility: dict[str, Any]) -> str:
    extent = float(np.max(size))
    for scale_bin, maximum in eligibility[
        "asset_scale_bin_maximum_extent_m"
    ].items():
        if maximum is None or extent <= float(maximum) + 1.0e-12:
            return str(scale_bin)
    raise ValueError("asset scale-bin contract has no open final interval")


def _asset_object_template(
    *,
    source_root: Path,
    runtime_root: Path,
    generation_metadata: dict[str, Any],
    release_metadata: dict[str, Any],
    eligibility: dict[str, Any],
    registry_cache: dict[tuple[str, str], dict[str, dict[str, Any]]],
    visual_hash_cache: dict[str, str],
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
    colliders = dynamic.get("proxy", {}).get("colliders")
    if not isinstance(colliders, list) or len(colliders) != 1:
        return None, "compound_proxy"
    collider = colliders[0]
    position = np.asarray(collider.get("position_m"), dtype=np.float64)
    rotation = np.asarray(
        collider.get("rotation_euler_degrees"), dtype=np.float64
    )
    if (
        position.shape != (3,)
        or rotation.shape != (3,)
        or not np.isfinite(position).all()
        or not np.isfinite(rotation).all()
    ):
        raise ValueError(f"asset source has an invalid proxy transform: {asset_id}")
    if not np.allclose(position, 0.0, atol=1.0e-8, rtol=0.0):
        return None, "offset_primitive_proxy"
    if not np.allclose(rotation, 0.0, atol=1.0e-8, rtol=0.0):
        return None, "rotated_primitive_proxy"
    collider_shape = str(collider.get("shape", ""))
    shape = {"box": "cuboid"}.get(collider_shape, collider_shape)
    if shape not in {"sphere", "cuboid", "cylinder"}:
        return None, "unsupported_primitive_proxy"
    size = np.asarray(collider.get("size_m"), dtype=np.float64)
    if (
        size.shape != (3,)
        or not np.isfinite(size).all()
        or bool(np.any(size <= 0.0))
        or (
            shape == "sphere"
            and not np.allclose(size, size[0], atol=1.0e-8, rtol=0.0)
        )
        or (
            shape == "cylinder"
            and not np.isclose(size[0], size[1], atol=1.0e-8, rtol=0.0)
        )
    ):
        raise ValueError(f"asset source has an invalid primitive proxy: {asset_id}")
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
    semantic_label = str(semantic_objects[0].get("semantic_label", "")).strip()
    if not semantic_label:
        raise ValueError(f"released asset source has no semantic label: {asset_id}")
    geometry = {"type": shape, "size_m": size.tolist()}
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
                    "collision_profile": copy.deepcopy(geometry),
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
                    "scale_bin": _asset_scale_bin(size, eligibility),
                    "uniform_scale": 1.0,
                }
            }
        },
        "appearance": {"materials": {}},
    }
    return template, "eligible"


def released_source_pool(
    *,
    root: Path,
    released_base_manifest_path: Path,
    source_root: Path,
    source_manifest_path: Path,
    matrix: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load generic hosts and exact generic/asset objects from the 1obj release."""

    source_contract = matrix["candidate_pool"]["source_release"]
    eligibility = matrix["candidate_pool"]["object_eligibility"]
    host_eligibility = matrix["candidate_pool"]["host_eligibility"]
    released_path = released_base_manifest_path.resolve()
    released_path.relative_to(root)
    generation_path = declared_within(source_root, source_manifest_path)
    released = read_json(released_path)
    generation = read_json(generation_path)
    if released.get("schema_version") != source_contract[
        "released_base_manifest_schema_version"
    ]:
        raise ValueError("released 1obj base manifest has the wrong schema")
    provenance = released.get("provenance", {}).get(
        "source_generation_release_metadata"
    )
    if not isinstance(provenance, dict) or (
        provenance.get("schema_version")
        != source_contract["generation_manifest_schema_version"]
        or provenance.get("manifest_sha256") != sha256(generation_path)
    ):
        raise ValueError("released 1obj base does not name this generation manifest")
    records = generation.get("records")
    if (
        generation.get("schema_version")
        != source_contract["generation_manifest_schema_version"]
        or generation.get("dataset_id") != released.get("dataset_id")
        or not isinstance(records, list)
        or int(generation.get("sample_count", -1)) != len(records)
        or int(generation.get("group_count", -1))
        != int(released.get("sample_count", -2))
    ):
        raise ValueError("generation manifest contradicts the released 1obj base")
    base_records = [
        record
        for record in records
        if str(record.get("kind")) == source_contract["sample_kind"]
    ]
    base_ids = [str(record.get("scene_id", "")) for record in base_records]
    if (
        len(base_records) != int(generation["group_count"])
        or any(not value for value in base_ids)
        or len(base_ids) != len(set(base_ids))
    ):
        raise ValueError("generation manifest has invalid canonical base records")

    family_schemas = {
        str(record["generation_metadata_schema_version"]): str(record["id"])
        for record in matrix["candidate_pool"]["object_source_families"]
    }
    released_records = _release_metadata_records(
        root=root,
        released_path=released_path,
        released=released,
        families=tuple(family_schemas.values()),
    )
    generation_family_ids = {
        str(record["scene_id"])
        for record in base_records
        if str(record.get("source_schema_version")) in family_schemas
    }
    if generation_family_ids != set(released_records):
        raise ValueError(
            "released generic/asset samples differ from generation metadata"
        )

    objects: list[dict[str, Any]] = []
    hosts: list[dict[str, Any]] = []
    asset_rejections: Counter[str] = Counter()
    registry_cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    visual_hash_cache: dict[str, str] = {}
    rejected_host_role_count = 0
    rejected_bounded_camera_host_count = 0
    rejected_visual_type_host_count = 0
    allowed_scenes = set(matrix["scene_compatibility"]["allowed_scene_classes"])
    allowed_visual_types = set(host_eligibility["allowed_visual_types"])
    allowed_scale_bins = set(eligibility["scale_bins"])
    geometry_types = {
        str(record["geometry_type"])
        for record in matrix["shape_families"]["families"]
    }

    for record in base_records:
        schema = str(record.get("source_schema_version", ""))
        source_family = family_schemas.get(schema)
        if source_family is None:
            continue
        scene_id = str(record["scene_id"])
        release_record = released_records[scene_id]
        if release_record[0] != source_family:
            raise ValueError(f"source family changed after release: {scene_id}")
        metadata_path = declared_within(source_root, Path(str(record["path"])))
        if sha256(metadata_path) != str(record["metadata_sha256"]):
            raise ValueError(f"source metadata changed after release: {scene_id}")
        metadata = read_json(metadata_path)
        release_metadata = release_record[2]
        if (
            metadata.get("schema_version") != schema
            or str(metadata.get("scene_id", "")) != scene_id
            or release_metadata.get("lineage", {}).get(
                "source_generation_metadata_sha256"
            )
            != str(record["metadata_sha256"])
        ):
            raise ValueError(f"source metadata identity is invalid: {scene_id}")
        source = _source_reference(
            root=root,
            source_root=source_root,
            generation_record=record,
            generation_metadata=metadata,
            release_record=release_record,
        )

        if source_family == "generic":
            simulation_objects = metadata.get("simulation", {}).get("objects")
            if not isinstance(simulation_objects, list) or len(simulation_objects) != 1:
                raise ValueError("eligible generic metadata must contain one object")
            obj = simulation_objects[0]
            support = metadata["simulation"]["support"]
            scene_class = str(support.get("scene_class", ""))
            if scene_class in allowed_scenes:
                roles = {
                    str(collider.get("role", ""))
                    for collider in support.get("colliders", [])
                    if isinstance(collider, dict)
                }
                required_roles = set(host_eligibility["required_collider_roles"])
                allowed_roles = set(host_eligibility["allowed_collider_roles"])
                if support.get("support_shape") == host_eligibility["support_shape"]:
                    motion_neutral = required_roles.issubset(roles) and roles.issubset(
                        allowed_roles
                    )
                    camera_unbounded = support.get("camera_envelope") is None
                    if motion_neutral and camera_unbounded:
                        scene_visual = metadata["appearance"]["scene_visual"]
                        visual_id = str(scene_visual.get("id", ""))
                        visual_type = str(scene_visual.get("visual_type", ""))
                        if not visual_id or not visual_type:
                            raise ValueError(
                                "eligible two-object host lacks a visual profile"
                            )
                        if visual_type not in allowed_visual_types:
                            rejected_visual_type_host_count += 1
                        else:
                            hosts.append(
                                {
                                    "metadata": metadata,
                                    "source": source,
                                    "scene_class": scene_class,
                                    "visual_profile_id": visual_id,
                                    "visual_type": visual_type,
                                }
                            )
                    elif not motion_neutral:
                        rejected_host_role_count += 1
                    else:
                        rejected_bounded_camera_host_count += 1
            template = metadata
        else:
            template, reason = _asset_object_template(
                source_root=source_root,
                runtime_root=root,
                generation_metadata=metadata,
                release_metadata=release_metadata,
                eligibility=eligibility,
                registry_cache=registry_cache,
                visual_hash_cache=visual_hash_cache,
            )
            if template is None:
                asset_rejections[reason] += 1
                continue
            obj = template["simulation"]["objects"][0]

        geometry = obj.get("geometry", {})
        shape = str(geometry.get("type", ""))
        collision = obj.get("collision_profile", {})
        pose_profile = str(obj.get("initial_state", {}).get("pose_profile", ""))
        if (
            obj.get("body_model") != eligibility["body_model"]
            or shape not in geometry_types
            or pose_profile != eligibility["required_pose_profile"]
        ):
            continue
        size = np.asarray(geometry.get("size_m"), dtype=np.float64)
        if (
            size.shape != (3,)
            or not np.isfinite(size).all()
            or bool(np.any(size <= 0.0))
            or (
                shape == "sphere"
                and not np.allclose(size, size[0], atol=1.0e-8, rtol=0.0)
            )
            or (
                shape == "cylinder"
                and not np.isclose(size[0], size[1], atol=1.0e-8, rtol=0.0)
            )
            or collision.get("type") != shape
        ):
            raise ValueError("eligible object lacks a matching primitive proxy")
        foreground = template["semantic_sampling"]["five_dimensions"][
            "foreground_object"
        ]
        scale_bin = str(foreground.get("scale_bin", ""))
        visual_profile_id = str(obj.get("visual_profile", {}).get("id", ""))
        if scale_bin not in allowed_scale_bins or not visual_profile_id:
            raise ValueError("eligible object lacks scale or a visual profile")
        objects.append(
            {
                "metadata": template,
                "source": source,
                "source_family": source_family,
                "shape_family_id": shape,
                "scale_bin": scale_bin,
                "visual_profile_id": visual_profile_id,
            }
        )

    if not objects or not hosts:
        raise ValueError("released 1obj metadata yields no eligible 2obj sources")
    audit = {
        "released_base_count": len(base_records),
        "released_object_family_counts": dict(
            sorted(Counter(value[0] for value in released_records.values()).items())
        ),
        "eligible_object_count": len(objects),
        "eligible_object_source_family_counts": dict(
            sorted(Counter(record["source_family"] for record in objects).items())
        ),
        "eligible_object_source_family_shape_scale_counts": dict(
            sorted(
                Counter(
                    ":".join(
                        [
                            str(record["source_family"]),
                            str(record["shape_family_id"]),
                            str(record["scale_bin"]),
                        ]
                    )
                    for record in objects
                ).items()
            )
        ),
        "eligible_asset_visual_count": len(
            {
                record["visual_profile_id"]
                for record in objects
                if record["source_family"] == "asset"
            }
        ),
        "asset_rejection_counts": dict(sorted(asset_rejections.items())),
        "eligible_host_count": len(hosts),
        "rejected_motion_specific_host_count": rejected_host_role_count,
        "rejected_bounded_camera_host_count": rejected_bounded_camera_host_count,
        "rejected_visual_type_host_count": rejected_visual_type_host_count,
        "object_scale_bin_counts": dict(
            sorted(Counter(record["scale_bin"] for record in objects).items())
        ),
        "object_shape_counts": dict(
            sorted(Counter(record["shape_family_id"] for record in objects).items())
        ),
        "object_visual_profile_count": len(
            {record["visual_profile_id"] for record in objects}
        ),
        "host_scene_class_counts": dict(
            sorted(Counter(record["scene_class"] for record in hosts).items())
        ),
        "host_visual_profile_count": len(
            {record["visual_profile_id"] for record in hosts}
        ),
    }
    return objects, hosts, audit
