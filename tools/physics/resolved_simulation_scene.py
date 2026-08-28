#!/usr/bin/env python3
"""Compile heterogeneous PhysSweep metadata into one immutable physics scene."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.rigid_geometry import finite_vector as _finite_vector


RESOLVED_SCENE_VERSION = "physweep_resolved_simulation_scene_v1"
GENERIC_SCHEMA = "physweep_pybullet_rigid_metadata_v1"
ASSET_SCHEMA = "physweep_asset_proxy_scene_v3"
BILLIARDS_SCHEMA = "physweep_billiards_scene_v4"
PASSIVE_PINBALL_SCHEMA = "physweep_passive_pinball_scene_v1"
MARBLE_RUN_SCHEMA = "physweep_marble_run_scene_v1"
SUPPORTED_SCHEMAS = {
    GENERIC_SCHEMA,
    ASSET_SCHEMA,
    BILLIARDS_SCHEMA,
    PASSIVE_PINBALL_SCHEMA,
    MARBLE_RUN_SCHEMA,
}


def _project_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_pinned_json(root: Path, binding: dict[str, Any], label: str) -> dict[str, Any]:
    path = _project_path(root, str(binding["path"]))
    if not path.exists():
        raise FileNotFoundError(f"missing {label}: {path}")
    expected = binding.get("sha256")
    if expected is not None and sha256(path) != str(expected):
        raise ValueError(f"{label} hash mismatch: {path}")
    return load_json(path)


def _finite_quaternion(value: Any, label: str) -> list[float]:
    result = _finite_vector(value, 4, label)
    norm = math.sqrt(sum(item * item for item in result))
    if abs(norm - 1.0) > 1.0e-6:
        raise ValueError(f"{label} is not a unit quaternion: {value}")
    return result


def _identity_objects(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    identity = metadata.get("object_identity", {})
    records = identity.get("objects", [])
    dynamic = [record for record in records if record.get("role") == "dynamic"]
    order = [str(record.get("object_id", "")) for record in dynamic]
    if not order or any(not object_id for object_id in order):
        raise ValueError("metadata has no valid dynamic object identity")
    if len(order) != len(set(order)):
        raise ValueError("dynamic object ids are not unique")
    return dynamic


def _resolved_materials(
    metadata: dict[str, Any], object_ids: list[str]
) -> dict[str, dict[str, float]]:
    sweep = metadata.get("sweep", {})
    records = sweep.get("resolved_object_physics")
    if records is None:
        return {}
    if not records:
        raise ValueError("resolved object physics must not be empty")
    record_ids = [str(record.get("object_id", "")) for record in records]
    record_indices = [record.get("object_index") for record in records]
    if record_ids != object_ids or record_indices != list(range(len(object_ids))):
        raise ValueError(
            "resolved object physics order and indices must match dynamic objects"
        )
    result: dict[str, dict[str, float]] = {}
    for record in records:
        object_id = str(record["object_id"])
        if object_id in result:
            raise ValueError(f"duplicate resolved material for {object_id}")
        material = record["material"]
        result[object_id] = {
            "mass_kg": float(material["mass_kg"]),
            "contact_friction": float(material["contact_friction"]),
            "contact_restitution": float(material["contact_restitution"]),
        }
    return result


def _material(value: dict[str, Any]) -> dict[str, float]:
    result = {
        "mass_kg": float(value["mass_kg"]),
        "contact_friction": float(value["contact_friction"]),
        "contact_restitution": float(value["contact_restitution"]),
    }
    if not all(math.isfinite(item) for item in result.values()):
        raise ValueError("dynamic object material values must be finite")
    if result["mass_kg"] <= 0.0:
        raise ValueError("dynamic object mass must be positive")
    if result["contact_friction"] < 0.0:
        raise ValueError("contact friction must be non-negative")
    if not 0.0 <= result["contact_restitution"] <= 1.0:
        raise ValueError("contact restitution must be in [0, 1]")
    return result


def _variant(metadata: dict[str, Any]) -> dict[str, Any]:
    sweep = metadata.get("sweep")
    if not sweep:
        return {
            "kind": "base",
            "target_object_id": None,
            "target_object_index": None,
            "parameter": None,
            "value": None,
        }
    kind = str(sweep.get("kind", "sweep"))
    target = sweep.get("target_object_id")
    target_index = sweep.get("target_object_index")
    parameter = sweep.get("parameter")
    value = sweep.get("value")
    if kind == "base":
        null_fields = (
            "target_object_id",
            "target_object_index",
            "parameter",
            "axis",
            "value",
            "level_index",
        )
        if any(sweep.get(field) is not None for field in null_fields):
            raise ValueError("canonical base must not bind a target or sweep value")
    elif kind == "sweep":
        if target is None or target_index is None or parameter is None or value is None:
            raise ValueError("sweep variant lacks target, index, parameter, or value")
        if parameter not in {"mass_kg", "contact_friction", "contact_restitution"}:
            raise ValueError(f"unsupported sweep parameter: {parameter}")
        if sweep.get("axis", parameter) != parameter:
            raise ValueError("sweep axis and parameter differ")
        if not math.isfinite(float(value)):
            raise ValueError("sweep value must be finite")
    else:
        raise ValueError(f"unsupported variant kind: {kind}")
    return {
        "kind": kind,
        "target_object_id": target,
        "target_object_index": target_index,
        "parameter": parameter,
        "value": value,
    }


def _validate_time_and_world(scene: dict[str, Any]) -> None:
    time = scene["time"]
    duration_s = float(time["duration_s"])
    output_fps = int(time["output_fps"])
    simulation_hz = int(time["simulation_hz"])
    frame_count = int(time["frame_count"])
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration must be finite and positive")
    if output_fps <= 0 or simulation_hz <= 0 or simulation_hz % output_fps:
        raise ValueError("simulation rate must be a positive multiple of output fps")
    if frame_count != int(round(duration_s * output_fps)) + 1:
        raise ValueError("frame count differs from duration and output fps")
    _finite_vector(scene["world"]["gravity_m_s2"], 3, "gravity")


def _generic_scene(metadata: dict[str, Any], root: Path) -> dict[str, Any]:
    simulation = metadata["simulation"]
    identities = _identity_objects(metadata)
    source_objects = simulation["objects"]
    if len(source_objects) != len(identities):
        raise ValueError("generic simulation and identity object counts differ")
    resolved = _resolved_materials(
        metadata, [str(source["object_id"]) for source in source_objects]
    )
    objects = []
    for index, source in enumerate(source_objects):
        object_id = str(source["object_id"])
        if object_id != str(identities[index]["object_id"]):
            raise ValueError("generic object order differs from object identity")
        material = _material(resolved.get(object_id, source["material"]))
        initial = source["initial_state"]
        objects.append(
            {
                "object_id": object_id,
                "object_index": index,
                "collision_proxy": copy.deepcopy(source["geometry"]),
                "initial_state": {
                    "position_m": _finite_vector(initial["position_m"], 3, "position"),
                    "orientation_quaternion_wxyz": _finite_quaternion(
                        initial["orientation_quaternion_wxyz"], "orientation"
                    ),
                    "linear_velocity_m_s": _finite_vector(
                        initial["linear_velocity_m_s"], 3, "linear velocity"
                    ),
                    "angular_velocity_rad_s": _finite_vector(
                        initial["angular_velocity_rad_s"], 3, "angular velocity"
                    ),
                },
                "material": material,
                "inertia_policy": "pybullet_from_collision_proxy_and_mass",
            }
        )
    return {
        "backend_binding": {
            "backend_id": "pybullet_rigid",
            "adapter_id": "generic_rigid_v1",
            "capability": "rigid_objects_with_analytic_or_exact_static_support",
            "supported_dynamic_object_counts": [1],
        },
        "time": copy.deepcopy(simulation["time"]),
        "world": copy.deepcopy(simulation["world"]),
        "objects": objects,
        "adapter_payload": {},
    }


def _registry_index(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = document.get("records", [])
    result = {str(record["asset_id"]): record for record in records}
    if len(result) != len(records):
        raise ValueError("asset registry contains duplicate asset ids")
    return result


def _asset_scene(metadata: dict[str, Any], root: Path) -> dict[str, Any]:
    identities = _identity_objects(metadata)
    if len(identities) != 1:
        raise ValueError("asset proxy v3 currently carries exactly one dynamic object")
    registry = _load_pinned_json(root, metadata["registry"], "asset registry")
    by_id = _registry_index(registry)
    asset_id = str(metadata["assets"]["dynamic_asset_id"])
    if asset_id not in by_id:
        raise ValueError(f"dynamic asset is missing from registry: {asset_id}")
    dynamic = by_id[asset_id]
    object_id = str(identities[0]["object_id"])
    resolved = _resolved_materials(metadata, [object_id])
    fallback = metadata["physics"].get("runtime_material")
    if fallback is None:
        proxy_material = dynamic["proxy"]["material"]
        fallback = {
            "mass_kg": metadata["physics"]["mass_kg"],
            "contact_friction": proxy_material["friction"],
            "contact_restitution": proxy_material["restitution"],
        }
    material = _material(resolved.get(object_id, fallback))
    initial = metadata["physics"]["initial_state"]
    static_prop_id = metadata["assets"].get("static_prop_asset_id")
    static_prop_record = None
    if static_prop_id is not None:
        static_prop_record = by_id.get(str(static_prop_id))
        if static_prop_record is None:
            raise ValueError(f"static prop is missing from registry: {static_prop_id}")
    backend = _load_pinned_json(
        root, metadata["physics"]["backend_config"], "PyBullet backend"
    )
    return {
        "backend_binding": {
            "backend_id": "pybullet_rigid",
            "adapter_id": "asset_proxy_v3",
            "capability": "reviewed_compound_dynamic_and_exact_static_support",
            "supported_dynamic_object_counts": [1],
        },
        "time": {
            "duration_s": float(metadata["physics"]["duration_s"]),
            "output_fps": int(metadata["physics"]["output_fps"]),
            "simulation_hz": int(metadata["physics"]["simulation_hz"]),
            "frame_count": int(metadata["physics"]["frame_count"]),
        },
        "world": {"gravity_m_s2": [0.0, 0.0, -9.81]},
        "objects": [
            {
                "object_id": object_id,
                "object_index": 0,
                "collision_proxy": copy.deepcopy(dynamic["proxy"]),
                "initial_state": {
                    "position_m": _finite_vector(initial["position_m"], 3, "position"),
                    "orientation_quaternion_xyzw": _finite_quaternion(
                        initial["orientation_quaternion_xyzw"], "orientation"
                    ),
                    "linear_velocity_m_s": _finite_vector(
                        initial["linear_velocity_m_s"], 3, "linear velocity"
                    ),
                    "angular_velocity_rad_s": _finite_vector(
                        initial["angular_velocity_rad_s"], 3, "angular velocity"
                    ),
                },
                "material": material,
                "inertia_policy": "pybullet_from_collision_proxy_and_mass",
            }
        ],
        "adapter_payload": {
            "dynamic_record": copy.deepcopy(dynamic),
            "static_support_binding": copy.deepcopy(
                metadata["physics"]["static_support_binding"]
            ),
            "static_prop_record": copy.deepcopy(static_prop_record),
            "static_prop_binding": copy.deepcopy(metadata["physics"].get("static_prop")),
            "motion_profile": str(metadata["physics"]["motion_profile"]),
            "expected_motion": copy.deepcopy(metadata["physics"]["expected_motion"]),
            "backend": backend,
        },
    }


def _billiards_scene(metadata: dict[str, Any], root: Path) -> dict[str, Any]:
    identities = _identity_objects(metadata)
    physics = metadata["physics"]
    initial_by_id = {
        str(record["object_id"]): record for record in physics["initial_states"]
    }
    identity_ids = [str(identity["object_id"]) for identity in identities]
    if len(initial_by_id) != len(physics["initial_states"]):
        raise ValueError("billiards initial states contain duplicate object ids")
    if set(initial_by_id) != set(identity_ids):
        raise ValueError("billiards initial states differ from dynamic object identity")
    resolved = _resolved_materials(metadata, identity_ids)
    backend = _load_pinned_json(root, physics["backend_config"], "PyBullet backend")
    dynamics = backend["billiards_rules"]["ball_dynamics"]
    objects = []
    for index, identity in enumerate(identities):
        object_id = str(identity["object_id"])
        if object_id not in initial_by_id:
            raise ValueError(f"billiards initial state is missing: {object_id}")
        initial = initial_by_id[object_id]
        fallback = physics.get("runtime_material") or {
            "mass_kg": physics["ball_mass_kg"],
            "contact_friction": dynamics["lateral_friction"],
            "contact_restitution": dynamics["restitution"],
        }
        material = _material(resolved.get(object_id, fallback))
        objects.append(
            {
                "object_id": object_id,
                "object_index": index,
                "collision_proxy": {
                    "type": "sphere",
                    "size_m": [2.0 * float(physics["ball_radius_m"])] * 3,
                },
                "initial_state": {
                    "position_m": _finite_vector(initial["position_m"], 3, "position"),
                    "orientation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                    "linear_velocity_m_s": _finite_vector(
                        initial["velocity_m_s"], 3, "linear velocity"
                    ),
                    "angular_velocity_rad_s": [0.0, 0.0, 0.0],
                },
                "material": material,
                "inertia_policy": "pybullet_from_collision_proxy_and_mass",
            }
        )
    return {
        "backend_binding": {
            "backend_id": "pybullet_rigid",
            "adapter_id": "billiards_v4",
            "capability": "one_or_more_spheres_with_exact_table_support",
            "supported_dynamic_object_counts": [1, 3],
        },
        "time": {
            "duration_s": float(physics["duration_s"]),
            "output_fps": int(physics["output_fps"]),
            "simulation_hz": int(physics["simulation_hz"]),
            "frame_count": int(physics["frame_count"]),
        },
        "world": {"gravity_m_s2": [0.0, 0.0, -9.81]},
        "objects": objects,
        "adapter_payload": {
            "static_support_binding": copy.deepcopy(physics["static_support_binding"]),
            "profile": str(physics["profile"]),
            "backend": backend,
        },
    }


def _single_sphere_fixture_scene(
    metadata: dict[str, Any],
    root: Path,
    *,
    label: str,
    backend_schema: str,
    adapter_id: str,
    capability: str,
) -> dict[str, Any]:
    identities = _identity_objects(metadata)
    simulation = metadata["simulation"]
    source_objects = simulation["objects"]
    if len(source_objects) != 1 or len(identities) != 1:
        raise ValueError(f"{label} requires exactly one dynamic object")
    source = source_objects[0]
    object_id = str(source["object_id"])
    if object_id != str(identities[0]["object_id"]):
        raise ValueError(f"{label} object identity differs from simulation")
    proxy = source["collision_proxy"]
    if proxy.get("type") != "sphere" or float(proxy.get("radius_m", 0.0)) <= 0.0:
        raise ValueError(f"{label} dynamic collision proxy must be a sphere")
    physics = metadata["physics"]
    backend = _load_pinned_json(root, physics["backend_config"], f"{label} backend")
    if backend.get("schema_version") != backend_schema:
        raise ValueError(f"unsupported {label} backend config")
    profile = str(physics["profile"])
    if profile != str(metadata["semantics"]["profile"]):
        raise ValueError(f"{label} physics and semantic profiles differ")
    if profile not in backend["profiles"]:
        raise ValueError(f"undeclared {label} profile: {profile}")
    resolved = _resolved_materials(metadata, [object_id])
    material = _material(resolved.get(object_id, source["material"]))
    initial = source["initial_state"]
    return {
        "backend_binding": {
            "backend_id": "pybullet_rigid",
            "adapter_id": adapter_id,
            "capability": capability,
            "supported_dynamic_object_counts": [1],
        },
        "time": copy.deepcopy(simulation["time"]),
        "world": copy.deepcopy(simulation["world"]),
        "objects": [
            {
                "object_id": object_id,
                "object_index": 0,
                "collision_proxy": copy.deepcopy(proxy),
                "initial_state": {
                    "position_m": _finite_vector(
                        initial["position_m"], 3, "position"
                    ),
                    "orientation_quaternion_xyzw": _finite_quaternion(
                        initial["orientation_quaternion_xyzw"], "orientation"
                    ),
                    "linear_velocity_m_s": _finite_vector(
                        initial["linear_velocity_m_s"], 3, "linear velocity"
                    ),
                    "angular_velocity_rad_s": _finite_vector(
                        initial["angular_velocity_rad_s"], 3, "angular velocity"
                    ),
                },
                "material": material,
                "inertia_policy": "pybullet_from_collision_proxy_and_mass",
            }
        ],
        "adapter_payload": {
            "profile": profile,
            "fixture": copy.deepcopy(physics["fixture"]),
            "quality": copy.deepcopy(physics["quality"]),
        },
    }


def _passive_pinball_scene(metadata: dict[str, Any], root: Path) -> dict[str, Any]:
    return _single_sphere_fixture_scene(
        metadata,
        root,
        label="passive-pinball",
        backend_schema="physweep_passive_pinball_backend_v1",
        adapter_id="passive_pinball_v1",
        capability="one_sphere_with_exact_passive_pinfield_fixture",
    )


def _marble_run_scene(metadata: dict[str, Any], root: Path) -> dict[str, Any]:
    return _single_sphere_fixture_scene(
        metadata,
        root,
        label="marble-run",
        backend_schema="physweep_marble_run_backend_v1",
        adapter_id="marble_run_v1",
        capability="one_sphere_with_exact_passive_track_fixture",
    )


def compile_resolved_scene(
    metadata: dict[str, Any], root: Path, metadata_path: Path | None = None
) -> dict[str, Any]:
    """Resolve one source schema without resampling or changing physical values."""
    schema = str(metadata.get("schema_version", ""))
    if schema not in SUPPORTED_SCHEMAS:
        raise ValueError(f"unsupported source metadata schema: {schema!r}")
    compiler = {
        GENERIC_SCHEMA: _generic_scene,
        ASSET_SCHEMA: _asset_scene,
        BILLIARDS_SCHEMA: _billiards_scene,
        PASSIVE_PINBALL_SCHEMA: _passive_pinball_scene,
        MARBLE_RUN_SCHEMA: _marble_run_scene,
    }[schema]
    compiled = compiler(metadata, root)
    scene = {
        "schema_version": RESOLVED_SCENE_VERSION,
        "scene_id": str(metadata["scene_id"]),
        "source_schema_version": schema,
        "source_metadata_path": str(metadata_path) if metadata_path else None,
        "source_metadata_sha256": sha256(metadata_path) if metadata_path else None,
        "variant": _variant(metadata),
        **compiled,
        "source_metadata": copy.deepcopy(metadata),
    }
    object_ids = [record["object_id"] for record in scene["objects"]]
    supported_counts = scene["backend_binding"]["supported_dynamic_object_counts"]
    if len(object_ids) not in supported_counts:
        raise ValueError(
            f"adapter does not support {len(object_ids)} dynamic objects"
        )
    _validate_time_and_world(scene)
    target = scene["variant"]["target_object_id"]
    if target is not None and target not in object_ids:
        raise ValueError(f"variant target is not a dynamic object: {target}")
    if target is not None:
        target_index = int(scene["variant"]["target_object_index"])
        if target_index < 0 or target_index >= len(object_ids):
            raise ValueError("variant target object index is out of range")
        if object_ids[target_index] != target:
            raise ValueError("variant target object id and index differ")
        target_object = scene["objects"][target_index]
        parameter = str(scene["variant"]["parameter"])
        if target_object["material"][parameter] != float(scene["variant"]["value"]):
            raise ValueError("sweep value differs from resolved target material")
    return scene
