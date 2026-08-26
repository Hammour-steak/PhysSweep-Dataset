#!/usr/bin/env python3
"""Canonical consumer-facing schema for compact PhysSweep base samples."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


BASE_SAMPLE_SCHEMA = "physweep_base_sample_v1"
TRAJECTORY_SCHEMA = "physweep_object_trajectory_v3"
MASK_MANIFEST_SCHEMA = "physweep_instance_mask_manifest_v3"

TRAJECTORY_FIELDS = (
    "schema_version",
    "object_ids",
    "time_s",
    "position_m",
    "quaternion_wxyz",
    "linear_velocity_m_s",
    "angular_velocity_rad_s",
    "contact_count",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_file(path: Path, expected_hash: str, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    actual = sha256(path)
    if actual != expected_hash:
        raise ValueError(
            f"{label} hash mismatch: expected={expected_hash} actual={actual}"
        )
    return path


def linked_file(path: Path, source: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def write_json(path: Path, value: Any) -> None:
    """Write deterministic JSON atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        temporary.write_text(
            json.dumps(
                value,
                indent=2,
                ensure_ascii=True,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write an NPZ whose bytes do not depend on wall-clock ZIP timestamps."""
    if tuple(arrays) != TRAJECTORY_FIELDS:
        raise ValueError("canonical trajectory fields are incomplete or out of order")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp-{os.getpid()}-{time.time_ns()}.npz")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for key, raw in arrays.items():
                array = np.asarray(raw)
                payload = io.BytesIO()
                np.lib.format.write_array(payload, array, allow_pickle=False)
                info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _without_none(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): copy.deepcopy(item) for key, item in value.items() if item is not None}


def _string_array(value: np.ndarray) -> list[str]:
    return [str(item) for item in np.asarray(value).reshape(-1).tolist()]


def canonical_trajectory(source: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Strip adapter/render channels while returning invariant object properties."""
    with np.load(source, allow_pickle=False) as archive:
        missing = sorted(set(TRAJECTORY_FIELDS[1:]) - set(archive.files))
        for required in ("runtime_material", "inertia_diagonal_kg_m2"):
            if required not in archive.files:
                missing.append(required)
        if missing:
            raise ValueError(f"trajectory lacks canonical fields: {', '.join(missing)}")
        arrays = {
            "schema_version": np.asarray(TRAJECTORY_SCHEMA),
            **{
                key: np.array(archive[key], copy=True)
                for key in TRAJECTORY_FIELDS
                if key != "schema_version"
            },
        }
        runtime_material = np.asarray(archive["runtime_material"], dtype=np.float64)
        inertia = np.asarray(archive["inertia_diagonal_kg_m2"], dtype=np.float64)

    object_ids = _string_array(arrays["object_ids"])
    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    position = np.asarray(arrays["position_m"], dtype=np.float64)
    quaternion = np.asarray(arrays["quaternion_wxyz"], dtype=np.float64)
    linear = np.asarray(arrays["linear_velocity_m_s"], dtype=np.float64)
    angular = np.asarray(arrays["angular_velocity_rad_s"], dtype=np.float64)
    contact = np.asarray(arrays["contact_count"])
    frame_count = int(time_s.shape[0])
    object_count = len(object_ids)
    expected = {
        "position_m": (frame_count, object_count, 3),
        "quaternion_wxyz": (frame_count, object_count, 4),
        "linear_velocity_m_s": (frame_count, object_count, 3),
        "angular_velocity_rad_s": (frame_count, object_count, 3),
        "contact_count": (frame_count, object_count),
    }
    actual = {
        "position_m": position.shape,
        "quaternion_wxyz": quaternion.shape,
        "linear_velocity_m_s": linear.shape,
        "angular_velocity_rad_s": angular.shape,
        "contact_count": contact.shape,
    }
    if actual != expected:
        raise ValueError(f"canonical trajectory shape mismatch: {actual}")
    if len(object_ids) != len(set(object_ids)) or any(not value for value in object_ids):
        raise ValueError("trajectory object_ids must be nonempty and unique")
    if frame_count < 2 or not np.isfinite(time_s).all() or not np.all(np.diff(time_s) > 0.0):
        raise ValueError("trajectory time axis is invalid")
    if not all(np.isfinite(value).all() for value in (position, quaternion, linear, angular)):
        raise ValueError("trajectory contains non-finite kinematics")
    norm_error = float(np.max(np.abs(np.linalg.norm(quaternion, axis=2) - 1.0)))
    if norm_error > 1.0e-6:
        raise ValueError(f"trajectory quaternion norm error: {norm_error}")
    if not np.issubdtype(contact.dtype, np.integer) or np.any(contact < 0):
        raise ValueError("trajectory contact_count must be non-negative integers")
    if runtime_material.shape != (object_count, 3) or not np.isfinite(runtime_material).all():
        raise ValueError("trajectory runtime_material is invalid")
    if inertia.shape != (object_count, 3) or not np.isfinite(inertia).all() or np.any(inertia <= 0.0):
        raise ValueError("trajectory inertia is invalid")
    return arrays, {
        "object_ids": object_ids,
        "frame_count": frame_count,
        "runtime_material": runtime_material,
        "inertia_diagonal_kg_m2": inertia,
    }


def _orientation_wxyz(initial: Mapping[str, Any]) -> list[float]:
    if "orientation_quaternion_wxyz" in initial:
        value = [float(item) for item in initial["orientation_quaternion_wxyz"]]
    elif "orientation_quaternion_xyzw" in initial:
        x, y, z, w = [float(item) for item in initial["orientation_quaternion_xyzw"]]
        value = [w, x, y, z]
    else:
        raise ValueError("initial state has no orientation quaternion")
    norm = math.sqrt(sum(item * item for item in value))
    if len(value) != 4 or abs(norm - 1.0) > 1.0e-6:
        raise ValueError("initial orientation is not a unit quaternion")
    return value


def _compact_camera(
    source: Mapping[str, Any],
    render_record: Mapping[str, Any],
    render_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    candidate = _mapping(render_record.get("camera"))
    if not candidate and render_metadata:
        candidate = _mapping(_mapping(render_metadata.get("visualization")).get("camera"))
    if not candidate:
        candidate = _mapping(source.get("camera"))
    required = ("position_m", "target_m", "focal_length_mm")
    if any(key not in candidate for key in required):
        raise ValueError("final camera is absent from rendered base artifacts")
    sensor_width = candidate.get("sensor_width_mm")
    if sensor_width is None:
        sensor_width = _mapping(source.get("camera")).get("sensor_width_mm")
    if sensor_width is None:
        if source.get("schema_version") != "physweep_asset_proxy_scene_v3":
            raise ValueError("final camera has no sensor_width_mm")
        # The reviewed asset-proxy v3 renderer fixed Blender's sensor to 36 mm
        # but its legacy render record omitted that constant.
        sensor_width = 36.0
    result = {
        "position_m": [float(item) for item in candidate["position_m"]],
        "target_m": [float(item) for item in candidate["target_m"]],
        "focal_length_mm": float(candidate["focal_length_mm"]),
        "sensor_width_mm": float(sensor_width),
    }
    for key in (
        "mode",
        "profile",
        "observation_intent",
        "structure_context",
        "clip_start_m",
        "clip_end_m",
    ):
        if candidate.get(key) is not None:
            result[key] = copy.deepcopy(candidate[key])
    if len(result["position_m"]) != 3 or len(result["target_m"]) != 3:
        raise ValueError("final camera vectors must be three-dimensional")
    if not all(math.isfinite(float(item)) for item in (*result["position_m"], *result["target_m"])):
        raise ValueError("final camera contains non-finite coordinates")
    return result


def _compact_semantics(source: Mapping[str, Any], family: str) -> dict[str, Any]:
    semantic_sampling = _mapping(source.get("semantic_sampling"))
    dimensions = _mapping(semantic_sampling.get("five_dimensions"))
    if dimensions:
        motion = _mapping(dimensions.get("motion"))
        foreground = _mapping(dimensions.get("foreground_object"))
        support = _mapping(dimensions.get("support_interaction"))
        observation = _mapping(dimensions.get("camera_observation"))
        appearance = _mapping(dimensions.get("appearance_lighting"))
        return {
            "motion": _without_none(
                {key: motion.get(key) for key in ("family", "subtype", "direction", "trajectory_extent")}
            ),
            "object": _without_none(
                {
                    key: foreground.get(key)
                    for key in (
                        "object_type",
                        "semantic_category",
                        "shape",
                        "visual_asset_id",
                        "scale_bin",
                        "uniform_scale",
                    )
                }
            ),
            "support": _without_none(
                {
                    key: support.get(key)
                    for key in (
                        "scene_class",
                        "support_type",
                        "support_layout",
                        "scene_theme",
                    )
                }
            ),
            "observation": _without_none(
                {
                    key: observation.get(key)
                    for key in ("camera_profile", "observation_intent", "structure_context")
                }
            ),
            "appearance": _without_none(
                {
                    key: appearance.get(key)
                    for key in ("surface_family", "environment_category", "contrast_policy", "hdri_role")
                }
            ),
        }
    semantics = _mapping(source.get("semantics"))
    if semantics:
        return _without_none(
            {
                "scene_family": semantics.get("scene_family", family),
                "profile": semantics.get("profile"),
                "description": semantics.get("description"),
            }
        )
    physics = _mapping(source.get("physics"))
    return _without_none(
        {
            "scene_family": family,
            "profile": physics.get("motion_profile") or physics.get("profile"),
            "description": source.get("dynamic_asset_name"),
        }
    )


def _compact_environment(source: Mapping[str, Any]) -> dict[str, Any] | None:
    environment = _mapping(_mapping(source.get("appearance")).get("hdri"))
    if not environment:
        environment = _mapping(_mapping(source.get("render")).get("environment"))
    if not environment:
        return None
    return _without_none(
        {
            key: environment.get(key)
            for key in ("name", "sha256", "role", "strength", "rotation_degrees")
        }
    )


def _compact_material_bindings(source: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    materials = _mapping(_mapping(source.get("appearance")).get("materials"))
    for role, raw in materials.items():
        binding = _mapping(raw)
        record = _mapping(binding.get("record"))
        compact = _without_none(
            {
                "asset_id": record.get("asset_id"),
                "texture_scale": binding.get("texture_scale"),
                "semantic_color_srgb": binding.get("semantic_color_srgb"),
                "semantic_color_mix": binding.get("semantic_color_mix"),
            }
        )
        if compact:
            result[str(role)] = compact
    return result


def _compact_fixture(
    source: Mapping[str, Any], render_record: Mapping[str, Any]
) -> dict[str, Any] | None:
    physics = _mapping(source.get("physics"))
    support_binding = _mapping(physics.get("static_support_binding"))
    environment_binding = _mapping(source.get("environment_binding"))
    simulation_support = _mapping(_mapping(source.get("simulation")).get("support"))
    semantics = _mapping(source.get("semantics"))
    assets = _mapping(source.get("assets"))
    binding_hash = (
        render_record.get("fixture_sha256")
        or render_record.get("support_binding_sha256")
        or support_binding.get("binding_sha256")
        or environment_binding.get("binding_sha256")
    )
    fixture = _without_none(
        {
            "id": (
                assets.get("support_asset_id")
                or semantics.get("profile")
                or simulation_support.get("semantic_type")
            ),
            "representation": (
                _mapping(physics.get("fixture")).get("representation")
                or support_binding.get("representation")
                or simulation_support.get("collision_authority")
            ),
            "binding_sha256": binding_hash,
        }
    )
    return fixture or None


def _compact_lighting(render_record: Mapping[str, Any]) -> dict[str, Any] | None:
    adaptation = _mapping(render_record.get("lighting_adaptation"))
    if not adaptation:
        return None
    return _without_none(
        {
            "policy_version": adaptation.get("policy_version"),
            "decision": adaptation.get("decision"),
            "exposure_ev": adaptation.get("result_exposure_ev"),
            "world_strength_scale": adaptation.get("world_strength_scale"),
            "fill_light_scale": adaptation.get("fill_light_scale"),
        }
    )


def _source_visual_by_id(source: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    simulation = _mapping(source.get("simulation"))
    for raw in simulation.get("objects", []):
        record = _mapping(raw)
        if record.get("object_id") and record.get("visual"):
            result[str(record["object_id"])] = copy.deepcopy(record["visual"])
    return result


def _compact_objects(
    source: Mapping[str, Any],
    resolved_scene: Mapping[str, Any],
    trajectory_info: Mapping[str, Any],
) -> list[dict[str, Any]]:
    identity = _mapping(source.get("object_identity"))
    identities = {
        str(record["object_id"]): _mapping(record)
        for record in identity.get("objects", [])
    }
    resolved_objects = [_mapping(value) for value in resolved_scene.get("objects", [])]
    object_ids = list(trajectory_info["object_ids"])
    if [str(value.get("object_id")) for value in resolved_objects] != object_ids:
        raise ValueError("resolved scene object order differs from trajectory")
    if set(identities) != set(object_ids):
        raise ValueError("object identity differs from canonical trajectory")
    visual_by_id = _source_visual_by_id(source)
    runtime_material = np.asarray(trajectory_info["runtime_material"], dtype=np.float64)
    inertia = np.asarray(trajectory_info["inertia_diagonal_kg_m2"], dtype=np.float64)
    result = []
    for array_index, (object_id, raw) in enumerate(zip(object_ids, resolved_objects)):
        record = identities[object_id]
        material = _mapping(raw.get("material"))
        expected_material = np.asarray(
            [
                float(material["mass_kg"]),
                float(material["contact_friction"]),
                float(material["contact_restitution"]),
            ]
        )
        if not np.allclose(runtime_material[array_index], expected_material, rtol=0.0, atol=1.0e-7):
            raise ValueError(f"runtime material differs for {object_id}")
        initial = _mapping(raw.get("initial_state"))
        compact = {
            "object_id": object_id,
            "array_index": array_index,
            "object_valid": True,
            "role": str(record.get("role", "dynamic")),
            "semantic_label": str(record.get("semantic_label", object_id)),
            "mask_instance_id": int(record.get("mask_instance_id", array_index + 1)),
            "collision_proxy": copy.deepcopy(raw["collision_proxy"]),
            "material": {
                "mass_kg": float(expected_material[0]),
                "contact_friction": float(expected_material[1]),
                "contact_restitution": float(expected_material[2]),
            },
            "inertia_diagonal_kg_m2": [float(item) for item in inertia[array_index]],
            "initial_state": {
                "position_m": [float(item) for item in initial["position_m"]],
                "quaternion_wxyz": _orientation_wxyz(initial),
                "linear_velocity_m_s": [float(item) for item in initial["linear_velocity_m_s"]],
                "angular_velocity_rad_s": [float(item) for item in initial["angular_velocity_rad_s"]],
            },
        }
        if record.get("asset_id") is not None:
            compact["asset_id"] = str(record["asset_id"])
        if object_id in visual_by_id:
            compact["visual"] = visual_by_id[object_id]
        result.append(compact)
    return result


def build_mask_manifest(
    *, scene_id: str, mask_root: Path, objects: list[dict[str, Any]]
) -> dict[str, Any]:
    expected_ids = [str(record["object_id"]) for record in objects]
    actual_ids = sorted(path.name for path in mask_root.iterdir() if path.is_dir())
    if sorted(expected_ids) != actual_ids:
        raise ValueError(f"mask object ids differ for {scene_id}")
    records = []
    frame_count: int | None = None
    for record in objects:
        object_id = str(record["object_id"])
        paths = sorted((mask_root / object_id).glob("frame_*.png"))
        if not paths:
            raise ValueError(f"mask frames are missing for {scene_id}/{object_id}")
        expected_names = [f"frame_{index:04d}.png" for index in range(1, len(paths) + 1)]
        if [path.name for path in paths] != expected_names:
            raise ValueError(f"mask frame sequence is not contiguous for {scene_id}/{object_id}")
        if frame_count is None:
            frame_count = len(paths)
        elif frame_count != len(paths):
            raise ValueError(f"mask object frame counts differ for {scene_id}")
        records.append(
            {
                "object_id": object_id,
                "instance_id": int(record["mask_instance_id"]),
                "frame_sha256": [sha256(path) for path in paths],
            }
        )
    return {
        "schema_version": MASK_MANIFEST_SCHEMA,
        "scene_id": scene_id,
        "frame_count": int(frame_count or 0),
        "objects": records,
    }


def build_base_metadata(
    *,
    family: str,
    group_id: str,
    source: Mapping[str, Any],
    source_metadata_sha256: str,
    resolved_scene: Mapping[str, Any],
    render_record: Mapping[str, Any],
    render_metadata: Mapping[str, Any] | None,
    trajectory_info: Mapping[str, Any],
    trajectory_sha256: str,
    video_sha256: str,
    mask_manifest_sha256: str | None,
) -> dict[str, Any]:
    scene_id = str(source["scene_id"])
    if scene_id != str(resolved_scene.get("scene_id")) or scene_id != str(render_record.get("scene_id")):
        raise ValueError("source, physics, and render scene ids differ")
    objects = _compact_objects(source, resolved_scene, trajectory_info)
    camera = _compact_camera(source, render_record, render_metadata)
    render = _mapping(source.get("render_request")) or _mapping(source.get("render"))
    resolution = render.get("resolution")
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError("render resolution is absent from source metadata")
    visual: dict[str, Any] = {
        "camera": camera,
        "resolution": [int(item) for item in resolution],
    }
    environment = _compact_environment(source)
    if environment:
        visual["environment"] = environment
    materials = _compact_material_bindings(source)
    if materials:
        visual["materials"] = materials
    assets = _without_none(_mapping(source.get("assets")))
    if assets:
        visual["assets"] = assets
    lighting = _compact_lighting(render_record)
    if lighting:
        visual["lighting"] = lighting

    backend = _mapping(resolved_scene.get("backend_binding"))
    physics: dict[str, Any] = {
        "backend": _without_none(
            {key: backend.get(key) for key in ("backend_id", "adapter_id")}
        ),
        "time": copy.deepcopy(resolved_scene["time"]),
        "world": copy.deepcopy(resolved_scene["world"]),
        "objects": objects,
    }
    fixture = _compact_fixture(source, render_record)
    if fixture:
        physics["fixture"] = fixture

    artifacts: dict[str, Any] = {
        "trajectory": {
            "path": "trajectory.npz",
            "sha256": trajectory_sha256,
            "schema_version": TRAJECTORY_SCHEMA,
        },
        "video": {"path": "video.mp4", "sha256": video_sha256},
    }
    if mask_manifest_sha256 is not None:
        artifacts["masks"] = {
            "path": "masks",
            "manifest_path": "mask_manifest.json",
            "manifest_sha256": mask_manifest_sha256,
            "schema_version": MASK_MANIFEST_SCHEMA,
        }

    caption = _mapping(_mapping(source.get("object_identity")).get("text")).get("caption")
    metadata = {
        "schema_version": BASE_SAMPLE_SCHEMA,
        "scene_id": scene_id,
        "group_id": group_id,
        "kind": "base",
        "family": family,
        "seed": int(source["seed"]),
        "semantics": _compact_semantics(source, family),
        "physics": physics,
        "visual": visual,
        "artifacts": artifacts,
        "lineage": {
            "source_schema_version": str(source["schema_version"]),
            "source_metadata_sha256": source_metadata_sha256,
        },
    }
    if caption:
        metadata["caption"] = str(caption)
    return metadata


def materialize_base_sample(
    *,
    target: Path,
    family: str,
    group_id: str,
    source_metadata_path: Path,
    source_metadata_sha256: str,
    resolved_scene_path: Path,
    resolved_scene_sha256: str,
    render_record_path: Path,
    render_record_sha256: str,
    trajectory_source_path: Path,
    trajectory_source_sha256: str,
    video_source_path: Path,
    video_sha256: str,
    masks_source_path: Path | None,
    render_metadata_path: Path | None = None,
    render_metadata_sha256: str | None = None,
) -> dict[str, Any]:
    """Materialize one compact sample from verified generation artifacts."""
    source_metadata_path = verified_file(
        source_metadata_path,
        source_metadata_sha256,
        f"{group_id} source metadata",
    )
    resolved_scene_path = verified_file(
        resolved_scene_path,
        resolved_scene_sha256,
        f"{group_id} resolved scene",
    )
    render_record_path = verified_file(
        render_record_path,
        render_record_sha256,
        f"{group_id} render record",
    )
    trajectory_source_path = verified_file(
        trajectory_source_path,
        trajectory_source_sha256,
        f"{group_id} source trajectory",
    )
    video_source_path = verified_file(
        video_source_path,
        video_sha256,
        f"{group_id} video",
    )
    render_metadata = None
    if render_metadata_path is not None:
        if render_metadata_sha256 is None:
            raise ValueError("render metadata path has no hash")
        render_metadata = json.loads(
            verified_file(
                render_metadata_path,
                render_metadata_sha256,
                f"{group_id} render metadata",
            ).read_text(encoding="utf-8")
        )

    source = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    resolved_scene = json.loads(resolved_scene_path.read_text(encoding="utf-8"))
    render_record = json.loads(render_record_path.read_text(encoding="utf-8"))
    scene_id = str(source.get("scene_id", ""))
    if not scene_id:
        raise ValueError("source metadata has no scene_id")
    arrays, trajectory_info = canonical_trajectory(trajectory_source_path)
    target.mkdir(parents=True)
    trajectory_path = target / "trajectory.npz"
    write_deterministic_npz(trajectory_path, arrays)
    trajectory_hash = sha256(trajectory_path)
    linked_file(target / "video.mp4", video_source_path)

    provisional = build_base_metadata(
        family=family,
        group_id=group_id,
        source=source,
        source_metadata_sha256=source_metadata_sha256,
        resolved_scene=resolved_scene,
        render_record=render_record,
        render_metadata=render_metadata,
        trajectory_info=trajectory_info,
        trajectory_sha256=trajectory_hash,
        video_sha256=video_sha256,
        mask_manifest_sha256=None,
    )
    mask_hash = None
    if masks_source_path is not None:
        if not masks_source_path.is_dir():
            raise FileNotFoundError(f"{scene_id} masks: {masks_source_path}")
        linked_file(target / "masks", masks_source_path)
        mask_manifest = build_mask_manifest(
            scene_id=scene_id,
            mask_root=masks_source_path,
            objects=provisional["physics"]["objects"],
        )
        mask_manifest_path = target / "mask_manifest.json"
        write_json(mask_manifest_path, mask_manifest)
        mask_hash = sha256(mask_manifest_path)

    metadata = build_base_metadata(
        family=family,
        group_id=group_id,
        source=source,
        source_metadata_sha256=source_metadata_sha256,
        resolved_scene=resolved_scene,
        render_record=render_record,
        render_metadata=render_metadata,
        trajectory_info=trajectory_info,
        trajectory_sha256=trajectory_hash,
        video_sha256=video_sha256,
        mask_manifest_sha256=mask_hash,
    )
    validate_base_metadata(metadata)
    metadata_path = target / "metadata.json"
    write_json(metadata_path, metadata)
    return {
        "scene_id": scene_id,
        "group_id": group_id,
        "sample_path": scene_id,
        "metadata_sha256": sha256(metadata_path),
        "source_metadata_sha256": source_metadata_sha256,
        "has_masks": mask_hash is not None,
    }


def validate_base_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if metadata.get("schema_version") != BASE_SAMPLE_SCHEMA or metadata.get("kind") != "base":
        raise ValueError("not canonical PhysSweep base metadata")
    scene_id = str(metadata.get("scene_id", ""))
    group_id = str(metadata.get("group_id", ""))
    family = str(metadata.get("family", ""))
    if not scene_id or not group_id or not family:
        raise ValueError("base identity is incomplete")
    physics = _mapping(metadata.get("physics"))
    objects = physics.get("objects", [])
    ids = [str(record.get("object_id", "")) for record in objects]
    indices = [record.get("array_index") for record in objects]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("canonical objects have invalid object ids")
    if indices != list(range(len(objects))) or not all(record.get("object_valid") is True for record in objects):
        raise ValueError("canonical object axis is invalid")
    for record in objects:
        inertia = np.asarray(record.get("inertia_diagonal_kg_m2"), dtype=np.float64)
        if inertia.shape != (3,) or not np.isfinite(inertia).all() or np.any(inertia <= 0.0):
            raise ValueError("canonical object inertia is invalid")
        if set(record.get("initial_state", {})) != {
            "position_m",
            "quaternion_wxyz",
            "linear_velocity_m_s",
            "angular_velocity_rad_s",
        }:
            raise ValueError("canonical initial state is incomplete")
    camera = _mapping(_mapping(metadata.get("visual")).get("camera"))
    if any(key not in camera for key in ("position_m", "target_m", "focal_length_mm", "sensor_width_mm")):
        raise ValueError("canonical final camera is incomplete")
    artifacts = _mapping(metadata.get("artifacts"))
    for key in ("trajectory", "video"):
        binding = _mapping(artifacts.get(key))
        if not binding.get("path") or len(str(binding.get("sha256", ""))) != 64:
            raise ValueError(f"canonical artifact binding is invalid: {key}")
    return {
        "schema_version": BASE_SAMPLE_SCHEMA,
        "scene_id": scene_id,
        "group_id": group_id,
        "family": family,
        "object_ids": ids,
        "mask_bound": "masks" in artifacts,
    }
