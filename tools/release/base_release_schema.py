#!/usr/bin/env python3
"""Canonical consumer-facing schema for compact PhysSweep base samples."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from tools.core.json_io import write_json_atomic_sorted as write_json
from tools.core.sweep_values import SWEEP_AXES, SWEEP_DERIVED_LEVELS
from tools.release.audit_release_provenance import sha256

BASE_SAMPLE_SCHEMA = "physweep_base_sample_v11"
SWEEP_SAMPLE_SCHEMA = "physweep_sweep_sample_v1"
TRAJECTORY_SCHEMA = "physweep_object_trajectory_v4"
MASK_MANIFEST_SCHEMA = "physweep_instance_mask_manifest_v4"
FIXTURE_SCHEMA = "physweep_static_fixture_v1"

SAMPLE_ENTRIES = frozenset(
    {
        "metadata.json",
        "trajectory.npz",
        "video.mp4",
        "mask_manifest.json",
        "masks",
    }
)
SAMPLE_LAYOUT_CONTRACT = {
    "sample_directory": "{family}/{scene_id}",
    "metadata": "metadata.json",
    "trajectory": "trajectory.npz",
    "video": "video.mp4",
    "mask_manifest": "mask_manifest.json",
    "masks": "masks/{object_id}/frame_{one_based_frame:04d}.png",
    "fixture": "fixtures/{metadata.physics.fixture.sha256}.json",
}
COMMON_SAMPLE_METADATA_FIELDS = frozenset(
    {
        "schema_version",
        "scene_id",
        "group_id",
        "family",
        "sample_kind",
        "seed",
        "semantics",
        "physics",
        "visual",
        "text",
        "artifacts",
        "lineage",
    }
)
SAMPLE_METADATA_FIELDS = {
    "base": COMMON_SAMPLE_METADATA_FIELDS,
    "sweep": COMMON_SAMPLE_METADATA_FIELDS | {"sweep"},
}

DYNAMIC_MATERIAL_FIELDS = (
    *SWEEP_AXES,
    "rolling_friction",
    "spinning_friction",
    "linear_damping",
    "angular_damping",
    "contact_processing_threshold_m",
)
ASSET_LABELS = {
    "decorated dinner plate": "decorated dinner plate",
    "plastic cup": "plastic cup",
    "toy car": "toy car",
    "glass bottle": "glass bottle",
    "a wooden crate": "wooden crate",
    "rubber band ball": "rubber band ball",
    "gas cylinder": "gas cylinder",
    "remote control": "remote control",
    "ceramic bowl": "ceramic bowl",
    "modern generic smartphone": "smartphone",
    "tin can": "tin can",
    "baluster vase, from a five piece garniture": "baluster vase",
    "ceramic mug": "ceramic mug",
    "closed magazine": "closed magazine",
    "tools / 4 way lug wrench": "four-way lug wrench",
    "cardboard box": "cardboard box",
}

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

def verified_file(path: Path, expected_hash: str, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    actual = sha256(path)
    if actual != expected_hash:
        raise ValueError(
            f"{label} hash mismatch: expected={expected_hash} actual={actual}"
        )
    return path


def materialized_file(path: Path, source: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source.resolve(), path)
    except OSError:
        if path.exists():
            raise
        shutil.copy2(source, path)


def safe_path_component(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"invalid {label}: {value!r}")
    component = value
    if (
        not component
        or Path(component).name != component
        or component in {".", ".."}
    ):
        raise ValueError(f"invalid {label}: {component!r}")
    return component


def write_content_addressed_json(root: Path, directory: str, value: Any) -> tuple[str, str]:
    payload = (
        json.dumps(value, indent=2, ensure_ascii=True, sort_keys=True) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    relative = Path(directory) / f"{digest}.json"
    target = root / relative
    if target.exists():
        if sha256(target) != digest:
            raise ValueError(f"content-address collision: {target}")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{time.time_ns()}")
        try:
            temporary.write_bytes(payload)
            try:
                os.link(temporary, target)
            except FileExistsError:
                pass
        finally:
            temporary.unlink(missing_ok=True)
        if sha256(target) != digest:
            raise ValueError(f"content-addressed write failed: {target}")
    return digest, relative.as_posix()


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


def canonical_quaternion_sign(value: Any) -> list[float]:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("invalid initial quaternion")
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > 1.0e-6:
        raise ValueError("initial quaternion is not normalized")
    for component in quaternion:
        if abs(float(component)) > 1.0e-12:
            if component < 0.0:
                quaternion = -quaternion
            break
    return [float(item) for item in quaternion]


def canonical_trajectory(
    source: Path,
    initial_quaternions: list[list[float]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Strip adapter channels and normalize the public trajectory contract."""
    with np.load(source, allow_pickle=False) as archive:
        missing = sorted(set(TRAJECTORY_FIELDS[1:]) - set(archive.files))
        for required in ("runtime_material", "inertia_diagonal_kg_m2"):
            if required not in archive.files:
                missing.append(required)
        if missing:
            raise ValueError(f"trajectory lacks canonical fields: {', '.join(missing)}")
        arrays = {"schema_version": np.asarray(TRAJECTORY_SCHEMA)}
        arrays.update(
            {
                key: np.array(archive[key], copy=True)
                for key in TRAJECTORY_FIELDS
                if key != "schema_version"
            }
        )
        runtime_material = np.asarray(archive["runtime_material"], dtype=np.float64)
        inertia = np.asarray(archive["inertia_diagonal_kg_m2"], dtype=np.float64)

    object_ids = [
        str(item)
        for item in np.asarray(arrays["object_ids"]).reshape(-1).tolist()
    ]
    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    position = np.asarray(arrays["position_m"], dtype=np.float64)
    quaternion = np.asarray(arrays["quaternion_wxyz"], dtype=np.float64)
    linear = np.asarray(arrays["linear_velocity_m_s"], dtype=np.float64)
    angular = np.asarray(arrays["angular_velocity_rad_s"], dtype=np.float64)
    contact = np.asarray(arrays["contact_count"], dtype=np.int32)
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
    if initial_quaternions is not None:
        initial = np.asarray(initial_quaternions, dtype=np.float64)
        if initial.shape != (object_count, 4):
            raise ValueError("initial quaternion/object shape differs")
        for object_index in range(object_count):
            if float(np.dot(quaternion[0, object_index], initial[object_index])) < 0.0:
                quaternion[0, object_index] *= -1.0
            for frame in range(1, frame_count):
                if float(
                    np.dot(
                        quaternion[frame - 1, object_index],
                        quaternion[frame, object_index],
                    )
                ) < 0.0:
                    quaternion[frame, object_index] *= -1.0
    if np.any(np.sum(quaternion[1:] * quaternion[:-1], axis=2) < -1.0e-12):
        raise ValueError("trajectory quaternion sign continuity failed")
    arrays.update(
        {
            "time_s": time_s,
            "position_m": position,
            "quaternion_wxyz": quaternion,
            "linear_velocity_m_s": linear,
            "angular_velocity_rad_s": angular,
            "contact_count": contact,
        }
    )
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
    return canonical_quaternion_sign(value)


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
    source_camera = _mapping(source.get("camera"))
    clip_start = candidate.get("clip_start_m", source_camera.get("clip_start_m"))
    clip_end = candidate.get("clip_end_m", source_camera.get("clip_end_m"))
    if clip_start is None or clip_end is None:
        # Asset and specialized renderers use this explicit fixed camera range.
        # Generic renders carry their per-sample values in the camera binding.
        if str(source.get("schema_version")) == "physweep_pybullet_rigid_metadata_v1":
            raise ValueError("final generic camera has no clipping range")
        clip_start = 0.03
        clip_end = 100.0
    result["clip_start_m"] = float(clip_start)
    result["clip_end_m"] = float(clip_end)
    if len(result["position_m"]) != 3 or len(result["target_m"]) != 3:
        raise ValueError("final camera vectors must be three-dimensional")
    if not all(math.isfinite(float(item)) for item in (*result["position_m"], *result["target_m"])):
        raise ValueError("final camera contains non-finite coordinates")
    return result


def _dynamic_material_extras(
    source: Mapping[str, Any],
    resolved_scene: Mapping[str, Any],
    object_id: str,
) -> dict[str, float]:
    """Recover non-swept dynamics that were applied by the simulator."""
    candidates: list[Mapping[str, Any]] = []
    simulation = _mapping(source.get("simulation"))
    for raw in simulation.get("objects", []):
        record = _mapping(raw)
        if str(record.get("object_id")) == object_id:
            candidates.append(_mapping(record.get("material")))

    adapter_payload = _mapping(resolved_scene.get("adapter_payload"))
    backend = _mapping(adapter_payload.get("backend"))
    adapter_id = str(_mapping(resolved_scene.get("backend_binding")).get("adapter_id"))
    if adapter_id == "asset_proxy_v3":
        candidates.append(
            _mapping(
                _mapping(
                    _mapping(backend.get("asset_proxy_rules")).get("contact")
                ).get("dynamic_defaults")
            )
        )
    elif adapter_id == "billiards_v4":
        candidates.append(
            _mapping(
                _mapping(backend.get("billiards_rules")).get("ball_dynamics")
            )
        )

    result: dict[str, float] = {}
    for key in (
        "rolling_friction",
        "spinning_friction",
        "linear_damping",
        "angular_damping",
    ):
        for candidate in candidates:
            if candidate.get(key) is not None:
                result[key] = float(candidate[key])
                break
        if key not in result:
            raise ValueError(f"resolved dynamic material lacks {key} for {object_id}")
        if not math.isfinite(result[key]) or result[key] < 0.0:
            raise ValueError(f"invalid {key} for {object_id}")
    return result


def _solver_contract(
    adapter_id: str,
    source: Mapping[str, Any],
    resolved_scene: Mapping[str, Any],
) -> dict[str, Any]:
    if adapter_id == "generic_rigid_v1":
        result = copy.deepcopy(_mapping(source.get("simulation")).get("solver"))
        result.setdefault("enable_cone_friction", True)
        result.setdefault("use_split_impulse", True)
    else:
        payload = _mapping(resolved_scene.get("adapter_payload"))
        backend = _mapping(payload.get("backend"))
        if adapter_id == "asset_proxy_v3":
            engine = _mapping(_mapping(backend.get("asset_proxy_rules")).get("engine"))
            result = {
                "solver_iterations": int(_mapping(backend.get("engine"))["solver_iterations"]),
                "deterministic_overlapping_pairs": True,
                "restitution_velocity_threshold_m_s": float(
                    engine["restitution_velocity_threshold_m_s"]
                ),
                "enable_cone_friction": bool(engine["enable_cone_friction"]),
                "use_split_impulse": bool(engine["use_split_impulse"]),
            }
        elif adapter_id == "billiards_v4":
            engine = _mapping(_mapping(backend.get("billiards_rules")).get("engine"))
            result = {
                "solver_iterations": int(engine["solver_iterations"]),
                "deterministic_overlapping_pairs": True,
                "restitution_velocity_threshold_m_s": float(
                    engine["restitution_velocity_threshold_m_s"]
                ),
                "enable_cone_friction": bool(engine["enable_cone_friction"]),
                "use_split_impulse": bool(engine["use_split_impulse"]),
            }
        else:
            result = copy.deepcopy(
                _mapping(_mapping(source.get("physics")).get("engine"))
            )
    if "iterations" in result:
        if "solver_iterations" in result:
            raise ValueError("both solver iteration field names are present")
        result["solver_iterations"] = result.pop("iterations")
    required = {
        "solver_iterations",
        "deterministic_overlapping_pairs",
        "restitution_velocity_threshold_m_s",
        "enable_cone_friction",
        "use_split_impulse",
    }
    if not required.issubset(result):
        raise ValueError(f"resolved solver contract is incomplete for {adapter_id}")
    allowed = required | {"contact_breaking_threshold_m"}
    return {
        key: copy.deepcopy(value)
        for key, value in result.items()
        if key in allowed
    }


def _normalized_label(family: str, value: str) -> str:
    label = value.strip().lower()
    if family == "generic":
        match = re.fullmatch(r"physassets\s+\d+\s+(.+)", label)
        if match is not None:
            return match.group(1)
    elif family == "asset":
        if label in ASSET_LABELS:
            return ASSET_LABELS[label]
    elif family == "billiards" and label == "object 1":
        return "cue ball"
    if not label or re.search(r"physassets\s+\d+|object\s+\d+", label):
        raise ValueError(f"unreviewed semantic label: {value!r}")
    return label


def _normalized_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _clean_text(
    family: str,
    source: Mapping[str, Any],
    labels: Mapping[str, str],
) -> dict[str, Any]:
    identity_text = _mapping(_mapping(source.get("object_identity")).get("text"))
    caption = str(identity_text.get("caption", ""))
    raw_mentions = identity_text.get("object_mentions", [])
    if not caption or not isinstance(raw_mentions, list) or not raw_mentions:
        raise ValueError("source object text is incomplete")
    search_start = 0
    normalized_surfaces = []
    for raw in raw_mentions:
        mention = _mapping(raw)
        object_id = str(mention.get("object_id", ""))
        old_surface = str(mention.get("text", ""))
        if object_id not in labels or not old_surface:
            raise ValueError("source caption mention is incomplete")
        start = caption.find(old_surface, search_start)
        if start < 0:
            raise ValueError("source caption mentions do not follow object order")
        surface = f"the {labels[object_id]}"
        caption = caption[:start] + surface + caption[start + len(old_surface) :]
        search_start = start + len(surface)
        normalized_surfaces.append((object_id, surface))
    if family == "generic":
        caption = caption.replace(" 1obj scenario", " scenario")
    if family in {"asset", "billiards"}:
        caption = re.sub(r" on the sketchfab bg [0-9a-f]{32}(?=\.)", "", caption)
        caption = re.sub(
            r" on the support [a-z0-9 ]+ [0-9a-f]{8}(?=\.)",
            " on the support",
            caption,
        )
    caption = re.sub(r"\s+", " ", caption).strip()
    if not caption.endswith(".") or re.search(
        r"physassets\s+\d+|sketchfab\s+bg\s+[0-9a-f]{16,}|object\s+\d+|\b1obj\b",
        caption,
    ):
        raise ValueError(f"internal identifier remains in caption: {caption!r}")
    mentions = []
    search_start = 0
    for object_id, surface in normalized_surfaces:
        start = caption.find(surface, search_start)
        if start < 0:
            raise ValueError("normalized caption mention is missing")
        mentions.append(
            {"object_id": object_id, "char_span": [start, start + len(surface)]}
        )
        search_start = start + len(surface)
    return {"caption": caption, "object_mentions": mentions}


def _compact_semantics(source: Mapping[str, Any]) -> dict[str, Any]:
    semantic_sampling = _mapping(source.get("semantic_sampling"))
    dimensions = _mapping(semantic_sampling.get("five_dimensions"))
    if dimensions:
        motion = _mapping(dimensions.get("motion"))
        support = _mapping(dimensions.get("support_interaction"))
        observation = _mapping(dimensions.get("camera_observation"))
        appearance = _mapping(dimensions.get("appearance_lighting"))
        return {
            "motion": _without_none(
                {key: motion.get(key) for key in ("family", "subtype", "direction", "trajectory_extent")}
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
                "profile": semantics.get("profile"),
                "description": semantics.get("description"),
            }
        )
    physics = _mapping(source.get("physics"))
    return _without_none(
        {
            "profile": physics.get("motion_profile") or physics.get("profile"),
            "description": source.get("dynamic_asset_name"),
        }
    )


def _compact_generic_object_annotations(
    source: Mapping[str, Any], object_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """Normalize legacy 1obj or explicit multi-object sampling semantics."""

    dimensions = _mapping(
        _mapping(source.get("semantic_sampling")).get("five_dimensions")
    )
    singular = dimensions.get("foreground_object")
    plural = dimensions.get("foreground_objects")
    if singular is not None and plural is not None:
        raise ValueError("generic source declares singular and plural object semantics")
    if plural is None:
        if len(object_ids) != 1 or not isinstance(singular, Mapping):
            raise ValueError(
                "generic multi-object source requires foreground_objects semantics"
            )
        records = [{"object_id": object_ids[0], **dict(singular)}]
    else:
        if not isinstance(plural, list) or not plural:
            raise ValueError("foreground_objects semantics must be a non-empty list")
        records = [_mapping(value) for value in plural]

    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        object_id = str(record.get("object_id", ""))
        if not object_id or object_id in by_id:
            raise ValueError("generic object semantics contain missing or duplicate ids")
        compact = _without_none(
            {
                key: record.get(key)
                for key in ("semantic_category", "scale_bin", "uniform_scale")
            }
        )
        if set(compact) != {"semantic_category", "scale_bin", "uniform_scale"}:
            raise ValueError(f"generic semantic annotations differ for {object_id}")
        by_id[object_id] = compact
    if list(by_id) != object_ids:
        raise ValueError("generic object semantic order differs from the object axis")
    return by_id


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


def _fixture_descriptor(
    source: Mapping[str, Any], render_record: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
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
        }
    )
    if set(fixture) != {"id", "representation"}:
        raise ValueError("fixture identity/representation is incomplete")
    if not isinstance(binding_hash, str) or len(binding_hash) != 64:
        raise ValueError("fixture source binding hash is incomplete")
    return fixture, binding_hash


def _strip_visual_fields(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_visual_fields(item) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    result = {}
    for key, item in value.items():
        if key in {
            "visual",
            "color_rgba",
            "source_path",
            "source_sha256",
            "review",
            "admission",
            "diagnostics",
        }:
            continue
        result[key] = _strip_visual_fields(item)
    return result


def build_fixture_payload(
    adapter_id: str,
    source: Mapping[str, Any],
    resolved_scene: Mapping[str, Any],
) -> dict[str, Any]:
    payload = _mapping(resolved_scene.get("adapter_payload"))
    if adapter_id == "generic_rigid_v1":
        physical = {
            "support": copy.deepcopy(_mapping(source.get("simulation"))["support"])
        }
    elif adapter_id == "asset_proxy_v3":
        backend = _mapping(payload.get("backend"))
        contact = _mapping(_mapping(backend.get("asset_proxy_rules")).get("contact"))
        physical = {
            "static_support_binding": copy.deepcopy(payload["static_support_binding"]),
            "static_prop_record": copy.deepcopy(payload.get("static_prop_record")),
            "static_prop_binding": copy.deepcopy(payload.get("static_prop_binding")),
            "static_dynamics": {
                "ground": copy.deepcopy(contact["ground"]),
                "support": copy.deepcopy(contact["support"]),
                "static_prop": copy.deepcopy(contact["static_prop"]),
            },
        }
    elif adapter_id == "billiards_v4":
        rules = _mapping(_mapping(payload.get("backend")).get("billiards_rules"))
        physical = {
            "static_support_binding": copy.deepcopy(payload["static_support_binding"]),
            "support_dynamics": copy.deepcopy(rules["support_dynamics"]),
        }
    else:
        physical = {"fixture": copy.deepcopy(payload["fixture"])}
    return {
        "schema_version": FIXTURE_SCHEMA,
        "adapter_id": adapter_id,
        "physical": _strip_visual_fields(physical),
    }


def localize_fixture_assets(
    value: Any,
    *,
    project_root: Path,
    release_root: Path,
    context: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, list):
        return [
            localize_fixture_assets(
                item,
                project_root=project_root,
                release_root=release_root,
                context=context + (str(index),),
            )
            for index, item in enumerate(value)
        ]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    result = {
        str(key): localize_fixture_assets(
            item,
            project_root=project_root,
            release_root=release_root,
            context=context + (str(key),),
        )
        for key, item in value.items()
    }
    is_collision_asset = (
        isinstance(result.get("path"), str)
        and isinstance(result.get("sha256"), str)
        and any(part in {"mesh", "collision"} for part in context)
        and "visual" not in context
    )
    if is_collision_asset:
        source = Path(result["path"])
        if not source.is_absolute():
            source = project_root / source
        expected = str(result["sha256"])
        verified_file(source, expected, "fixture collision asset")
        suffix = source.suffix.lower() or ".bin"
        relative = Path("fixture_assets") / f"{expected}{suffix}"
        target = release_root / relative
        if not target.exists():
            try:
                materialized_file(target, source)
            except FileExistsError:
                pass
        verified_file(target, expected, "localized fixture collision asset")
        result["path"] = relative.as_posix()
    return result


def _compact_lighting(render_record: Mapping[str, Any]) -> dict[str, Any] | None:
    adaptation = _mapping(render_record.get("lighting_adaptation"))
    if not adaptation:
        return None
    return _without_none(
        {
            "exposure_ev": adaptation.get("result_exposure_ev"),
            "world_strength_scale": adaptation.get("world_strength_scale"),
            "fill_light_scale": adaptation.get("fill_light_scale"),
        }
    )


def _source_base_color_by_id(source: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    simulation = _mapping(source.get("simulation"))
    for raw in simulation.get("objects", []):
        record = _mapping(raw)
        visual = _mapping(record.get("visual"))
        if record.get("object_id") and visual.get("color_rgba") is not None:
            result[str(record["object_id"])] = copy.deepcopy(visual["color_rgba"])
    return result


def _compact_collision_proxy(raw: Mapping[str, Any]) -> dict[str, Any]:
    proxy = copy.deepcopy(raw)
    if "colliders" in proxy:
        return {
            "type": "compound",
            "colliders": [
                {
                    key: copy.deepcopy(collider[key])
                    for key in ("shape", "size_m", "position_m", "rotation_euler_degrees")
                }
                for collider in proxy["colliders"]
            ],
        }
    if proxy.get("type") == "sphere" and "size_m" in proxy:
        size = [float(value) for value in proxy["size_m"]]
        if len(size) != 3 or max(size) - min(size) > 1.0e-9:
            raise ValueError("sphere collision proxy is not isotropic")
        return {"type": "sphere", "radius_m": size[0] / 2.0}
    return proxy


def _compact_objects(
    family: str,
    source: Mapping[str, Any],
    resolved_scene: Mapping[str, Any],
    trajectory_info: Mapping[str, Any],
    fixture_id: str,
    billiards_templates: Mapping[str, Mapping[str, str]] | None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
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
    base_color_by_id = _source_base_color_by_id(source)
    runtime_material = np.asarray(trajectory_info["runtime_material"], dtype=np.float64)
    inertia = np.asarray(trajectory_info["inertia_diagonal_kg_m2"], dtype=np.float64)
    source_objects = {
        str(_mapping(value).get("object_id")): _mapping(value)
        for value in _mapping(source.get("simulation")).get("objects", [])
    }
    result = []
    labels: dict[str, str] = {}
    for array_index, (object_id, raw) in enumerate(zip(object_ids, resolved_objects)):
        record = identities[object_id]
        labels[object_id] = _normalized_label(
            family, str(record.get("semantic_label", object_id))
        )
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
        extra_material = _dynamic_material_extras(source, resolved_scene, object_id)
        initial = _mapping(raw.get("initial_state"))
        collision_proxy = _compact_collision_proxy(_mapping(raw["collision_proxy"]))
        compact = {
            "object_id": object_id,
            "object_valid": True,
            "collision_proxy": collision_proxy,
            "material": {
                "mass_kg": float(expected_material[0]),
                "contact_friction": float(expected_material[1]),
                "contact_restitution": float(expected_material[2]),
                **extra_material,
                "contact_processing_threshold_m": 0.0,
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
        adapter_id = str(_mapping(resolved_scene.get("backend_binding")).get("adapter_id"))
        if adapter_id == "generic_rigid_v1":
            geometry = _mapping(source_objects.get(object_id, {}).get("geometry"))
            size_m = [float(value) for value in geometry.get("size_m", [])]
            if len(size_m) != 3 or min(size_m) <= 0.0:
                raise ValueError(f"invalid generic source geometry for {object_id}")
            compact["ccd_swept_sphere_radius_m"] = 0.22 * min(size_m)
        if object_id in base_color_by_id:
            compact["visual"] = {
                "base_color_srgb_rgba": base_color_by_id[object_id]
            }
        if family == "billiards":
            templates = _mapping((billiards_templates or {}).get(fixture_id))
            if object_id not in templates or "asset_id" in compact or "visual" in compact:
                raise ValueError(f"billiards appearance template is incomplete: {object_id}")
            compact["visual"] = {
                "material_template": {
                    "source_fixture_asset_id": fixture_id,
                    "source_object_name": str(templates[object_id]),
                }
            }
        appearance_sources = int("asset_id" in compact)
        visual = _mapping(compact.get("visual"))
        appearance_sources += int("base_color_srgb_rgba" in visual)
        appearance_sources += int("material_template" in visual)
        if appearance_sources != 1:
            raise ValueError(f"object has no unique appearance source: {object_id}")
        result.append(compact)
    return result, labels


def build_mask_manifest(
    *, scene_id: str, mask_root: Path, objects: list[dict[str, Any]]
) -> dict[str, Any]:
    expected_ids = [
        safe_path_component(record["object_id"], "mask object id")
        for record in objects
    ]
    actual_ids = sorted(path.name for path in mask_root.iterdir() if path.is_dir())
    if sorted(expected_ids) != actual_ids:
        raise ValueError(f"mask object ids differ for {scene_id}")
    records = []
    frame_count: int | None = None
    for record in objects:
        object_id = safe_path_component(record["object_id"], "mask object id")
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
                "frame_sha256": [sha256(path) for path in paths],
            }
        )
    return {
        "schema_version": MASK_MANIFEST_SCHEMA,
        "scene_id": scene_id,
        "frame_count": int(frame_count or 0),
        "objects": records,
    }


def _build_sample_metadata(
    *,
    sample_kind: str,
    sweep: Mapping[str, Any] | None,
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
    fixture_sha256: str,
    billiards_templates: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    scene_id = str(source["scene_id"])
    if scene_id != str(resolved_scene.get("scene_id")) or scene_id != str(render_record.get("scene_id")):
        raise ValueError("source, physics, and render scene ids differ")
    fixture_identity, source_fixture_binding_sha256 = _fixture_descriptor(
        source, render_record
    )
    objects, labels = _compact_objects(
        family,
        source,
        resolved_scene,
        trajectory_info,
        str(fixture_identity["id"]),
        billiards_templates,
    )
    camera = _compact_camera(source, render_record, render_metadata)
    render_config = source.get("render_request") or source.get("render")
    if not isinstance(render_config, Mapping):
        raise ValueError("source render configuration is absent")
    render_samples = render_record.get("render_samples")
    if render_samples is None:
        render_samples = render_config.get("samples")
    if render_samples is None or int(render_samples) <= 0:
        raise ValueError("render sample count is invalid")
    visual: dict[str, Any] = {
        "camera": camera,
        "render_samples": int(render_samples),
    }
    environment = _compact_environment(source)
    if environment:
        visual["environment"] = environment
    materials = _compact_material_bindings(source)
    materials.pop("dynamic_object", None)
    if materials:
        visual["materials"] = materials
    static_prop = _mapping(source.get("assets")).get("static_prop_asset_id")
    if static_prop is not None:
        visual["assets"] = {"static_prop_asset_id": str(static_prop)}
    lighting = _compact_lighting(render_record)
    if lighting:
        visual["lighting"] = lighting

    semantics = _compact_semantics(source)
    annotations_by_id: dict[str, dict[str, Any]] = {}
    if family == "generic":
        annotations_by_id = _compact_generic_object_annotations(
            source, [str(record["object_id"]) for record in objects]
        )
    else:
        dimensions = _mapping(
            _mapping(source.get("semantic_sampling")).get("five_dimensions")
        )
        if (
            "foreground_object" in dimensions
            or "foreground_objects" in dimensions
        ):
            raise ValueError(
                "non-generic sample declares generic object annotations"
            )
    semantic_objects = []
    for physics_object in objects:
        object_id = str(physics_object["object_id"])
        semantic_object = {
            "object_id": object_id,
            "semantic_label": labels[object_id],
        }
        if object_id in annotations_by_id:
            semantic_object.update(copy.deepcopy(annotations_by_id[object_id]))
        semantic_objects.append(semantic_object)
    semantics["objects"] = semantic_objects
    appearance = semantics.get("appearance")
    if not isinstance(appearance, dict):
        appearance = {}
    environment = _mapping(visual.get("environment"))
    if appearance.get("hdri_role") == environment.get("role"):
        appearance.pop("hdri_role", None)
    support = semantics.get("support")
    if not isinstance(support, dict):
        support = {}
    if support.get("support_type") == fixture_identity.get("id"):
        support.pop("support_type", None)
    if len(semantic_objects) == 1 and "description" in semantics:
        identity_objects = _mapping(source.get("object_identity")).get("objects", [])
        redundant_labels = {_normalized_text(semantic_objects[0]["semantic_label"])}
        redundant_labels.update(
            _normalized_text(_mapping(value).get("semantic_label", ""))
            for value in identity_objects
        )
        if _normalized_text(semantics["description"]) in redundant_labels:
            semantics.pop("description")
    text = _clean_text(family, source, labels)

    backend = _mapping(resolved_scene.get("backend_binding"))
    physics: dict[str, Any] = {
        "backend": _without_none(
            {key: backend.get(key) for key in ("backend_id", "adapter_id")}
        ),
        "time": {
            key: copy.deepcopy(resolved_scene["time"][key])
            for key in ("duration_s", "output_fps", "simulation_hz")
        },
        "world": {
            "gravity_m_s2": [
                float(item) for item in resolved_scene["world"]["gravity_m_s2"]
            ]
        },
        "objects": objects,
        "solver": _solver_contract(str(backend["adapter_id"]), source, resolved_scene),
        "fixture": {
            **fixture_identity,
            "sha256": fixture_sha256,
        },
    }

    artifacts: dict[str, Any] = {
        "trajectory": {"sha256": trajectory_sha256},
        "video": {"sha256": video_sha256},
    }
    if sample_kind not in {"base", "sweep"} or (sample_kind == "base") != (sweep is None):
        raise ValueError("sample kind and sweep descriptor disagree")
    metadata = {
        "schema_version": (
            BASE_SAMPLE_SCHEMA if sample_kind == "base" else SWEEP_SAMPLE_SCHEMA
        ),
        "scene_id": scene_id,
        "group_id": group_id,
        "family": family,
        "sample_kind": sample_kind,
        "sweep": copy.deepcopy(dict(sweep)) if sweep is not None else None,
        "seed": int(source["seed"]),
        "semantics": semantics,
        "physics": physics,
        "visual": visual,
        "text": text,
        "artifacts": artifacts,
        "lineage": {
            "source_generation_metadata_sha256": source_metadata_sha256,
            "source_fixture_binding_sha256": source_fixture_binding_sha256,
        },
    }
    if sweep is None:
        metadata.pop("sweep")
    return metadata


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
    fixture_sha256: str,
    billiards_templates: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    return _build_sample_metadata(
        sample_kind="base",
        sweep=None,
        family=family,
        group_id=group_id,
        source=source,
        source_metadata_sha256=source_metadata_sha256,
        resolved_scene=resolved_scene,
        render_record=render_record,
        render_metadata=render_metadata,
        trajectory_info=trajectory_info,
        trajectory_sha256=trajectory_sha256,
        video_sha256=video_sha256,
        fixture_sha256=fixture_sha256,
        billiards_templates=billiards_templates,
    )


def build_sweep_metadata(
    *,
    sweep: Mapping[str, Any],
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
    fixture_sha256: str,
    billiards_templates: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    return _build_sample_metadata(
        sample_kind="sweep",
        sweep=sweep,
        family=family,
        group_id=group_id,
        source=source,
        source_metadata_sha256=source_metadata_sha256,
        resolved_scene=resolved_scene,
        render_record=render_record,
        render_metadata=render_metadata,
        trajectory_info=trajectory_info,
        trajectory_sha256=trajectory_sha256,
        video_sha256=video_sha256,
        fixture_sha256=fixture_sha256,
        billiards_templates=billiards_templates,
    )


def _materialize_sample(
    *,
    sample_kind: str,
    sweep: Mapping[str, Any] | None,
    target: Path,
    family: str,
    group_id: str,
    source: Mapping[str, Any],
    source_metadata_sha256: str,
    resolved_scene: Mapping[str, Any],
    render_record: Mapping[str, Any],
    trajectory_source_path: Path,
    video_source_path: Path,
    video_sha256: str,
    masks_source_path: Path,
    release_root: Path,
    source_project_root: Path,
    billiards_templates: Mapping[str, Mapping[str, str]] | None = None,
    render_metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], str]:
    """Materialize one compact sample from verified generation artifacts."""
    source = dict(source)
    resolved_scene = dict(resolved_scene)
    render_record = dict(render_record)
    render_metadata = dict(render_metadata) if render_metadata is not None else None
    scene_id = str(source.get("scene_id", ""))
    if not scene_id:
        raise ValueError("source metadata has no scene_id")
    resolved_objects = [_mapping(value) for value in resolved_scene.get("objects", [])]
    initial_quaternions = [
        _orientation_wxyz(_mapping(value.get("initial_state")))
        for value in resolved_objects
    ]
    arrays, trajectory_info = canonical_trajectory(
        trajectory_source_path, initial_quaternions
    )
    adapter_id = str(_mapping(resolved_scene.get("backend_binding")).get("adapter_id"))
    fixture_payload = localize_fixture_assets(
        build_fixture_payload(adapter_id, source, resolved_scene),
        project_root=source_project_root.resolve(),
        release_root=release_root.resolve(),
    )
    fixture_sha256, fixture_path = write_content_addressed_json(
        release_root.resolve(), "fixtures", fixture_payload
    )
    if fixture_path != f"fixtures/{fixture_sha256}.json":
        raise ValueError("fixture path is not derived from its hash")
    target.mkdir(parents=True)
    trajectory_path = target / "trajectory.npz"
    write_deterministic_npz(trajectory_path, arrays)
    trajectory_hash = sha256(trajectory_path)
    materialized_file(target / "video.mp4", video_source_path)

    metadata = _build_sample_metadata(
        sample_kind=sample_kind,
        sweep=sweep,
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
        fixture_sha256=fixture_sha256,
        billiards_templates=billiards_templates,
    )
    if not masks_source_path.is_dir():
        raise FileNotFoundError(f"{scene_id} masks: {masks_source_path}")
    masks_target = target / "masks"
    masks_target.mkdir()
    for physics_object in metadata["physics"]["objects"]:
        object_id = safe_path_component(physics_object["object_id"], "mask object id")
        source_directory = masks_source_path / object_id
        source_frames = sorted(source_directory.glob("frame_*.png"))
        if not source_frames:
            raise ValueError(f"mask frames are missing for {scene_id}/{object_id}")
        target_directory = masks_target / object_id
        target_directory.mkdir()
        for index, source_frame in enumerate(source_frames, start=1):
            expected_name = f"frame_{index:04d}.png"
            if source_frame.name != expected_name:
                raise ValueError(f"mask frame sequence differs for {scene_id}/{object_id}")
            target_frame = target_directory / expected_name
            with Image.open(source_frame) as image:
                if image.mode == "RGBA":
                    alpha = image.getchannel("A")
                elif image.mode == "L":
                    alpha = image.copy()
                else:
                    raise ValueError(f"unsupported source mask mode: {source_frame}")
                alpha.save(
                    target_frame,
                    format="PNG",
                    compress_level=9,
                    optimize=False,
                )
    mask_manifest = build_mask_manifest(
        scene_id=scene_id,
        mask_root=masks_target,
        objects=metadata["physics"]["objects"],
    )
    if int(mask_manifest["frame_count"]) != int(trajectory_info["frame_count"]):
        raise ValueError(f"mask and trajectory frame counts differ for {scene_id}")
    mask_manifest_path = target / "mask_manifest.json"
    write_json(mask_manifest_path, mask_manifest)
    metadata["artifacts"]["masks"] = {
        "manifest_sha256": sha256(mask_manifest_path),
    }
    if sample_kind == "base":
        validate_base_metadata(metadata)
    else:
        validate_sweep_metadata(metadata)
    metadata_path = target / "metadata.json"
    write_json(metadata_path, metadata)
    return (
        {
            "scene_id": scene_id,
            "metadata_sha256": sha256(metadata_path),
        },
        fixture_sha256,
    )


def materialize_base_sample(
    *,
    target: Path,
    family: str,
    group_id: str,
    source: Mapping[str, Any],
    source_metadata_sha256: str,
    resolved_scene: Mapping[str, Any],
    render_record: Mapping[str, Any],
    trajectory_source_path: Path,
    video_source_path: Path,
    video_sha256: str,
    masks_source_path: Path,
    release_root: Path,
    source_project_root: Path,
    billiards_templates: Mapping[str, Mapping[str, str]] | None = None,
    render_metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], str]:
    return _materialize_sample(
        sample_kind="base",
        sweep=None,
        target=target,
        family=family,
        group_id=group_id,
        source=source,
        source_metadata_sha256=source_metadata_sha256,
        resolved_scene=resolved_scene,
        render_record=render_record,
        trajectory_source_path=trajectory_source_path,
        video_source_path=video_source_path,
        video_sha256=video_sha256,
        masks_source_path=masks_source_path,
        release_root=release_root,
        source_project_root=source_project_root,
        billiards_templates=billiards_templates,
        render_metadata=render_metadata,
    )


def materialize_sweep_sample(
    *,
    sweep: Mapping[str, Any],
    target: Path,
    family: str,
    group_id: str,
    source: Mapping[str, Any],
    source_metadata_sha256: str,
    resolved_scene: Mapping[str, Any],
    render_record: Mapping[str, Any],
    trajectory_source_path: Path,
    video_source_path: Path,
    video_sha256: str,
    masks_source_path: Path,
    release_root: Path,
    source_project_root: Path,
    billiards_templates: Mapping[str, Mapping[str, str]] | None = None,
    render_metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, str], str]:
    return _materialize_sample(
        sample_kind="sweep",
        sweep=sweep,
        target=target,
        family=family,
        group_id=group_id,
        source=source,
        source_metadata_sha256=source_metadata_sha256,
        resolved_scene=resolved_scene,
        render_record=render_record,
        trajectory_source_path=trajectory_source_path,
        video_source_path=video_source_path,
        video_sha256=video_sha256,
        masks_source_path=masks_source_path,
        release_root=release_root,
        source_project_root=source_project_root,
        billiards_templates=billiards_templates,
        render_metadata=render_metadata,
    )


def _validate_sample_metadata(
    metadata: Mapping[str, Any],
    *,
    sample_kind: str,
    schema_version: str,
) -> dict[str, Any]:
    if metadata.get("schema_version") != schema_version:
        raise ValueError(f"not canonical PhysSweep {sample_kind} metadata")
    required = SAMPLE_METADATA_FIELDS.get(sample_kind)
    if required is None:
        raise ValueError(f"invalid sample kind: {sample_kind}")
    if set(metadata) != required or metadata.get("sample_kind") != sample_kind:
        raise ValueError(f"canonical {sample_kind} fields are invalid")
    scene_id = str(metadata.get("scene_id", ""))
    group_id = str(metadata.get("group_id", ""))
    family = str(metadata.get("family", ""))
    if not scene_id or not group_id or not family:
        raise ValueError(f"{sample_kind} identity is incomplete")
    physics = _mapping(metadata.get("physics"))
    if set(physics) != {"backend", "time", "world", "objects", "solver", "fixture"}:
        raise ValueError("canonical physics fields are invalid")
    if set(_mapping(physics.get("backend"))) != {"backend_id", "adapter_id"}:
        raise ValueError("canonical backend binding is invalid")
    world = _mapping(physics.get("world"))
    gravity = np.asarray(world.get("gravity_m_s2"), dtype=np.float64)
    if (
        set(world) != {"gravity_m_s2"}
        or gravity.shape != (3,)
        or not np.isfinite(gravity).all()
    ):
        raise ValueError("canonical world fields are invalid")
    solver = _mapping(physics.get("solver"))
    solver_required = {
        "solver_iterations",
        "deterministic_overlapping_pairs",
        "restitution_velocity_threshold_m_s",
        "enable_cone_friction",
        "use_split_impulse",
    }
    if set(solver) not in (
        solver_required,
        solver_required | {"contact_breaking_threshold_m"},
    ):
        raise ValueError("canonical solver fields are invalid")
    fixture = _mapping(physics.get("fixture"))
    if set(fixture) != {"id", "representation", "sha256"}:
        raise ValueError("canonical fixture binding is invalid")
    objects = physics.get("objects", [])
    ids = [
        safe_path_component(record.get("object_id", ""), "object id")
        for record in objects
    ]
    if not 1 <= len(ids) <= 3 or len(ids) != len(set(ids)):
        raise ValueError("canonical objects have invalid object ids")
    if not all(
        record.get("object_valid") is True
        and not {"array_index", "role", "mask_instance_id", "semantic_label"}.intersection(record)
        for record in objects
    ):
        raise ValueError("canonical object axis is invalid")
    for record in objects:
        common_object_fields = {
            "object_id",
            "object_valid",
            "collision_proxy",
            "material",
            "inertia_diagonal_kg_m2",
            "initial_state",
        }
        if set(record) - common_object_fields - {
            "asset_id",
            "visual",
            "ccd_swept_sphere_radius_m",
        }:
            raise ValueError("canonical object fields are invalid")
        if "ccd_swept_sphere_radius_m" in record and (
            not math.isfinite(float(record["ccd_swept_sphere_radius_m"]))
            or float(record["ccd_swept_sphere_radius_m"]) <= 0.0
        ):
            raise ValueError("canonical object CCD radius is invalid")
        material = _mapping(record.get("material"))
        if set(material) != set(DYNAMIC_MATERIAL_FIELDS):
            raise ValueError("canonical dynamic material is incomplete")
        values = {key: float(material[key]) for key in DYNAMIC_MATERIAL_FIELDS}
        if (
            not all(math.isfinite(value) for value in values.values())
            or values["mass_kg"] <= 0.0
            or values["contact_friction"] < 0.0
            or not 0.0 <= values["contact_restitution"] <= 1.0
            or any(values[key] < 0.0 for key in DYNAMIC_MATERIAL_FIELDS[3:])
            or values["contact_processing_threshold_m"] != 0.0
        ):
            raise ValueError("canonical dynamic material is invalid")
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
        proxy = _mapping(record.get("collision_proxy"))
        proxy_fields = {
            "sphere": {"type", "radius_m"},
            "compound": {"type", "colliders"},
        }.get(str(proxy.get("type")), {"type", "size_m"})
        if set(proxy) != proxy_fields:
            raise ValueError("canonical collision proxy is invalid")
        if proxy.get("type") == "compound" and any(
            set(collider)
            != {"shape", "size_m", "position_m", "rotation_euler_degrees"}
            for collider in proxy["colliders"]
        ):
            raise ValueError("canonical compound collider is invalid")
        visual_object = _mapping(record.get("visual"))
        appearance_sources = int("asset_id" in record)
        appearance_sources += int("base_color_srgb_rgba" in visual_object)
        appearance_sources += int("material_template" in visual_object)
        if (
            appearance_sources != 1
            or "color_rgba" in visual_object
            or ("asset_id" in record and "visual" in record)
            or (
                "base_color_srgb_rgba" in visual_object
                and set(visual_object) != {"base_color_srgb_rgba"}
            )
            or (
                "material_template" in visual_object
                and (
                    set(visual_object) != {"material_template"}
                    or set(_mapping(visual_object["material_template"]))
                    != {"source_fixture_asset_id", "source_object_name"}
                )
            )
        ):
            raise ValueError("canonical object appearance is invalid")
    if set(physics.get("time", {})) != {"duration_s", "output_fps", "simulation_hz"}:
        raise ValueError("canonical time contract is invalid")
    semantics = _mapping(metadata.get("semantics"))
    if set(semantics) - {
        "objects",
        "profile",
        "description",
        "motion",
        "observation",
        "support",
        "appearance",
    }:
        raise ValueError("canonical semantic fields are invalid")
    semantic_objects = semantics.get("objects", [])
    semantic_ids = [
        safe_path_component(record.get("object_id", ""), "semantic object id")
        for record in semantic_objects
    ]
    if (
        "scene_family" in semantics
        or "object" in semantics
        or semantic_ids != ids
        or any(
            not str(record.get("semantic_label", "")).strip()
            for record in semantic_objects
        )
    ):
        raise ValueError("canonical semantics duplicate top-level family")
    labels = {
        str(record["object_id"]): str(record["semantic_label"])
        for record in semantic_objects
    }
    text = _mapping(metadata.get("text"))
    if set(text) != {"caption", "object_mentions"}:
        raise ValueError("canonical text fields are invalid")
    caption = str(text["caption"])
    mentioned = set()
    for mention in text["object_mentions"]:
        object_id = str(mention.get("object_id", ""))
        span = mention.get("char_span")
        if (
            set(mention) != {"object_id", "char_span"}
            or object_id not in labels
            or not isinstance(span, list)
            or len(span) != 2
            or not all(isinstance(value, int) for value in span)
            or not 0 <= span[0] < span[1] <= len(caption)
            or caption[span[0] : span[1]] != f"the {labels[object_id]}"
        ):
            raise ValueError("canonical text mention is invalid")
        mentioned.add(object_id)
    if not set(ids).issubset(mentioned):
        raise ValueError("canonical text does not mention every object")
    visual = _mapping(metadata.get("visual"))
    camera = _mapping(visual.get("camera"))
    required_camera = {
        "position_m",
        "target_m",
        "focal_length_mm",
        "sensor_width_mm",
        "clip_start_m",
        "clip_end_m",
    }
    if set(camera) != required_camera:
        raise ValueError("canonical final camera is incomplete")
    if (
        float(camera["clip_start_m"]) <= 0.0
        or float(camera["clip_end_m"]) <= float(camera["clip_start_m"])
    ):
        raise ValueError("canonical final camera clipping range is invalid")
    lighting = _mapping(visual.get("lighting"))
    if (
        set(visual) - {"camera", "render_samples", "environment", "materials", "assets", "lighting"}
        or isinstance(visual.get("render_samples"), bool)
        or int(visual.get("render_samples", 0)) <= 0
        or "resolution" in visual
        or (
            lighting
            and set(lighting)
            != {"exposure_ev", "world_strength_scale", "fill_light_scale"}
        )
    ):
        raise ValueError("canonical visual contract is invalid")
    artifacts = _mapping(metadata.get("artifacts"))
    trajectory = _mapping(artifacts.get("trajectory"))
    if (
        set(trajectory) != {"sha256"}
        or len(str(trajectory.get("sha256", ""))) != 64
    ):
        raise ValueError("canonical trajectory binding is invalid")
    video = _mapping(artifacts.get("video"))
    if (
        set(video) != {"sha256"}
        or len(str(video.get("sha256", ""))) != 64
    ):
        raise ValueError("canonical video binding is invalid")
    masks = _mapping(artifacts.get("masks"))
    if (
        set(masks) != {"manifest_sha256"}
        or len(str(masks.get("manifest_sha256", ""))) != 64
    ):
        raise ValueError("canonical mask binding is invalid")
    if set(artifacts) != {"trajectory", "video", "masks"}:
        raise ValueError("canonical artifact set is invalid")
    lineage = _mapping(metadata.get("lineage"))
    if (
        set(lineage)
        != {"source_generation_metadata_sha256", "source_fixture_binding_sha256"}
        or any(len(str(value)) != 64 for value in lineage.values())
    ):
        raise ValueError("canonical lineage is invalid")
    return {
        "schema_version": schema_version,
        "scene_id": scene_id,
        "group_id": group_id,
        "family": family,
        "object_ids": ids,
    }


def validate_base_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_sample_metadata(
        metadata,
        sample_kind="base",
        schema_version=BASE_SAMPLE_SCHEMA,
    )


def validate_sweep_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    summary = _validate_sample_metadata(
        metadata,
        sample_kind="sweep",
        schema_version=SWEEP_SAMPLE_SCHEMA,
    )
    sweep = _mapping(metadata.get("sweep"))
    if set(sweep) != {
        "target_object_id",
        "parameter",
        "level_index",
        "value",
    }:
        raise ValueError("canonical sweep descriptor fields are invalid")
    target_object_id = safe_path_component(
        sweep.get("target_object_id", ""), "sweep target object id"
    )
    parameter = str(sweep.get("parameter", ""))
    level_index = sweep.get("level_index")
    value = sweep.get("value")
    if (
        target_object_id not in summary["object_ids"]
        or parameter not in SWEEP_AXES
        or isinstance(level_index, bool)
        or not isinstance(level_index, int)
        or level_index not in SWEEP_DERIVED_LEVELS
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError("canonical sweep descriptor is invalid")
    target = next(
        record
        for record in metadata["physics"]["objects"]
        if record["object_id"] == target_object_id
    )
    if not math.isclose(
        float(target["material"][parameter]),
        float(value),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("sweep descriptor value differs from object material")
    return summary
