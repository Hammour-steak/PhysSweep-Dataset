#!/usr/bin/env python3
"""Validate and resolve two-object rules for frozen specialized fixtures."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RULES = Path("configs/two_object_specialized_scene_rules.json")
SCHEMA_VERSION = "physweep_two_object_specialized_scene_rules_v1"
SCENE_FAMILIES = ("billiards", "passive_pinball", "marble_run")
OBJECT_IDS = ("object_a", "object_b")


def _project_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _finite_vector(value: Any, size: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain {size} values")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must be finite")
    return result


def load_two_object_specialized_rules(
    root: Path = PROJECT_ROOT,
    path: str | Path = DEFAULT_RULES,
) -> dict[str, Any]:
    source = _project_path(root.resolve(), path)
    document = json.loads(source.read_text(encoding="utf-8"))
    validate_two_object_specialized_rules(document, root.resolve())
    return document


def family_index(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(record["id"]): record for record in contract["scene_families"]}


def profile_index(
    contract: dict[str, Any],
) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    result: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for family in contract["scene_families"]:
        for profile in family["profiles"]:
            profile_id = str(profile["id"])
            if profile_id in result:
                raise ValueError(f"duplicate specialized profile: {profile_id}")
            result[profile_id] = (family, profile)
    return result


def _validate_source(root: Path, family: dict[str, Any]) -> None:
    binding = family.get("source_config")
    if not isinstance(binding, dict) or set(binding) != {"path", "schema_version"}:
        raise ValueError(f"{family.get('id')} source config is incomplete")
    path = _project_path(root, str(binding["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    source = json.loads(path.read_text(encoding="utf-8"))
    source_version = source.get("schema_version", source.get("version"))
    if source_version != binding["schema_version"]:
        raise ValueError(f"{family['id']} source config schema changed")


def _validate_profile(family_id: str, profile: dict[str, Any]) -> None:
    if profile.get("interaction_class") != "interacting":
        raise ValueError(f"{profile.get('id')} must be an interacting profile")
    if profile.get("contact_requirement") != "must_contact":
        raise ValueError(f"{profile.get('id')} must require pair contact")
    if not isinstance(profile.get("description"), str) or not profile["description"].strip():
        raise ValueError(f"{profile.get('id')} lacks a description")
    objects = profile.get("objects")
    if not isinstance(objects, list) or [
        obj.get("object_id") for obj in objects
    ] != list(OBJECT_IDS):
        raise ValueError(f"{profile.get('id')} object order changed")
    if any(not str(obj.get("semantic_label", "")).strip() for obj in objects):
        raise ValueError(f"{profile.get('id')} has an empty object label")
    if family_id == "billiards":
        for obj in objects:
            _finite_vector(obj.get("position_xy_m"), 2, "billiards position")
            _finite_vector(obj.get("linear_velocity_xy_m_s"), 2, "billiards velocity")
    elif family_id == "passive_pinball":
        for obj in objects:
            _finite_vector(obj.get("local_position_m"), 2, "pinball local position")
            _finite_vector(obj.get("local_velocity_m_s"), 2, "pinball local velocity")
    elif family_id == "marble_run":
        for obj in objects:
            offset = float(obj.get("initial_track_offset_m"))
            velocity = float(obj.get("track_velocity_m_s"))
            if not math.isfinite(offset) or not math.isfinite(velocity):
                raise ValueError("marble-run initial state must be finite")
    else:
        raise ValueError(f"unsupported specialized family: {family_id}")
    quality = profile.get("quality")
    if not isinstance(quality, dict):
        raise ValueError(f"{profile.get('id')} lacks quality rules")
    gap = float(quality.get("minimum_initial_surface_gap_m", -1.0))
    first_contact = float(quality.get("maximum_first_pair_contact_time_s", 0.0))
    penetration = float(quality.get("maximum_penetration_m", 0.0))
    if (
        gap < 0.0
        or not math.isfinite(gap)
        or first_contact <= 0.0
        or not 0.0 < penetration <= 0.001
    ):
        raise ValueError(f"{profile.get('id')} has invalid pair-contact bounds")


def _camera_view_index(camera: dict[str, Any]) -> dict[str, dict[str, Any]]:
    views = camera.get("view_families")
    if not isinstance(views, list) or len(views) < 3:
        raise ValueError("specialized camera contract requires at least three views")
    result: dict[str, dict[str, Any]] = {}
    for view in views:
        view_id = str(view.get("id", ""))
        if not view_id or view_id in result:
            raise ValueError("specialized camera view ids must be unique and non-empty")
        azimuth = float(view.get("azimuth_offset_degrees"))
        elevation = float(view.get("elevation_offset_degrees"))
        distance_scale = float(view.get("distance_scale"))
        if (
            not all(
                math.isfinite(value)
                for value in (azimuth, elevation, distance_scale)
            )
            or abs(azimuth) > 30.0
            or abs(elevation) > 15.0
            or not 0.85 <= distance_scale <= 1.15
        ):
            raise ValueError(f"invalid specialized camera view: {view_id}")
        result[view_id] = view
    return result


def validate_two_object_specialized_rules(
    contract: dict[str, Any], root: Path = PROJECT_ROOT
) -> None:
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported two-object specialized-scene rules")
    object_contract = contract.get("object_contract")
    if not isinstance(object_contract, dict):
        raise ValueError("specialized object contract is missing")
    if (
        int(object_contract.get("object_count", -1)) != 2
        or tuple(object_contract.get("object_ids", ())) != OBJECT_IDS
        or object_contract.get("shape") != "sphere"
        or object_contract.get("body_model") != "rigid_body"
        or object_contract.get("initial_contact_is_forbidden") is not True
        or object_contract.get("active_mechanisms_supported") is not False
    ):
        raise ValueError("specialized two-object identity contract changed")
    families = contract.get("scene_families")
    if not isinstance(families, list) or [
        record.get("id") for record in families
    ] != list(SCENE_FAMILIES):
        raise ValueError("specialized scene-family coverage is incomplete")
    profile_ids: set[str] = set()
    for family in families:
        family_id = str(family["id"])
        _validate_source(root.resolve(), family)
        fixture = family.get("fixture_contract")
        camera = family.get("camera_contract")
        if (
            not isinstance(fixture, dict)
            or fixture.get("source_family") != family_id
            or fixture.get("dynamic_shape") != "sphere"
            or not isinstance(camera, dict)
            or camera.get("full_fixture_visible") is not True
            or camera.get("full_pair_motion_envelope_visible") is not True
        ):
            raise ValueError(f"{family_id} fixture or camera contract is incomplete")
        camera_views = _camera_view_index(camera)
        profiles = family.get("profiles")
        if not isinstance(profiles, list) or len(profiles) < 3:
            raise ValueError(f"{family_id} requires at least three profiles")
        assigned_views: set[str] = set()
        for profile in profiles:
            profile_id = str(profile.get("id", ""))
            if not profile_id or profile_id in profile_ids:
                raise ValueError(f"duplicate or empty specialized profile: {profile_id}")
            profile_ids.add(profile_id)
            _validate_profile(family_id, profile)
            view_id = str(profile.get("camera_view_family_id", ""))
            if view_id not in camera_views:
                raise ValueError(f"{profile_id} references an unknown camera view")
            assigned_views.add(view_id)
        if assigned_views != set(camera_views):
            raise ValueError(f"{family_id} does not exercise every camera view")
    expected_policy = {
        "fixture_is_static": True,
        "pair_contact_is_deferred": True,
        "post_contact_outcome_is_not_preclassified": True,
        "one_factor_sweep_keeps_fixture_and_both_initial_states_fixed": True,
        "object_identity_order_is_fixed": True,
    }
    if contract.get("policy") != expected_policy:
        raise ValueError("specialized two-object policy changed")


def resolve_billiards_initial_states(
    profile: dict[str, Any], *, bed_z_m: float, ball_radius_m: float
) -> list[dict[str, Any]]:
    center_z = float(bed_z_m) + float(ball_radius_m) + 0.001
    result = []
    for source in profile["objects"]:
        x, y = _finite_vector(source["position_xy_m"], 2, "billiards position")
        vx, vy = _finite_vector(source["linear_velocity_xy_m_s"], 2, "billiards velocity")
        result.append(
            {
                "object_id": str(source["object_id"]),
                "semantic_label": str(source["semantic_label"]),
                "position_m": [x, y, center_z],
                "orientation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "linear_velocity_m_s": [vx, vy, 0.0],
                "angular_velocity_rad_s": [0.0, 0.0, 0.0],
            }
        )
    validate_resolved_initial_clearance(result, 2.0 * float(ball_radius_m), profile)
    return result


def resolve_pinball_initial_states(
    profile: dict[str, Any],
    *,
    fixture_source: dict[str, Any],
    fixture_frame: dict[str, Any],
    ball_radius_m: float,
) -> list[dict[str, Any]]:
    top = np.asarray(fixture_source["top_center_m"], dtype=np.float64)
    right = np.asarray(fixture_frame["right"], dtype=np.float64)
    down = np.asarray(fixture_frame["down"], dtype=np.float64)
    normal = np.asarray(fixture_frame["normal"], dtype=np.float64)
    normal_offset = (
        float(fixture_source["board_thickness_m"]) / 2.0
        + float(ball_radius_m)
        + 0.0005
    )
    result = []
    for source in profile["objects"]:
        x, local_down = _finite_vector(source["local_position_m"], 2, "pinball local position")
        vx, vdown = _finite_vector(source["local_velocity_m_s"], 2, "pinball local velocity")
        position = top + x * right + local_down * down + normal_offset * normal
        velocity = vx * right + vdown * down
        result.append(
            {
                "object_id": str(source["object_id"]),
                "semantic_label": str(source["semantic_label"]),
                "position_m": position.tolist(),
                "orientation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                "linear_velocity_m_s": velocity.tolist(),
                "angular_velocity_rad_s": [0.0, 0.0, 0.0],
            }
        )
    validate_resolved_initial_clearance(result, 2.0 * float(ball_radius_m), profile)
    return result


def resolve_marble_run_initial_states(
    profile: dict[str, Any], *, base_initial_state: dict[str, Any], ball_radius_m: float
) -> list[dict[str, Any]]:
    base_position = np.asarray(base_initial_state["position_m"], dtype=np.float64)
    result = []
    for source in profile["objects"]:
        position = base_position.copy()
        position[0] += float(source["initial_track_offset_m"])
        result.append(
            {
                "object_id": str(source["object_id"]),
                "semantic_label": str(source["semantic_label"]),
                "position_m": position.tolist(),
                "orientation_quaternion_xyzw": copy.deepcopy(
                    base_initial_state["orientation_quaternion_xyzw"]
                ),
                "linear_velocity_m_s": [float(source["track_velocity_m_s"]), 0.0, 0.0],
                "angular_velocity_rad_s": [0.0, 0.0, 0.0],
            }
        )
    validate_resolved_initial_clearance(result, 2.0 * float(ball_radius_m), profile)
    return result


def resolve_specialized_camera_binding(
    family: dict[str, Any],
    profile: dict[str, Any],
    base_binding: dict[str, Any],
) -> dict[str, Any]:
    """Apply a small reviewed orbit offset without weakening fixture coverage."""

    views = _camera_view_index(family["camera_contract"])
    view_id = str(profile["camera_view_family_id"])
    if view_id not in views:
        raise ValueError(f"unknown specialized camera view: {view_id}")
    view = views[view_id]
    position = np.asarray(
        _finite_vector(base_binding.get("position_m"), 3, "camera position"),
        dtype=np.float64,
    )
    target = np.asarray(
        _finite_vector(base_binding.get("target_m"), 3, "camera target"),
        dtype=np.float64,
    )
    offset = position - target
    distance = float(np.linalg.norm(offset))
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("specialized base camera has zero distance")
    azimuth = math.atan2(float(offset[1]), float(offset[0])) + math.radians(
        float(view["azimuth_offset_degrees"])
    )
    base_elevation = math.asin(float(offset[2]) / distance)
    elevation = base_elevation + math.radians(float(view["elevation_offset_degrees"]))
    if not math.radians(-80.0) < elevation < math.radians(80.0):
        raise ValueError("specialized camera elevation is outside the safe orbit")
    distance *= float(view["distance_scale"])
    horizontal = distance * math.cos(elevation)
    resolved = copy.deepcopy(base_binding)
    resolved["position_m"] = [
        round(float(target[0]) + horizontal * math.cos(azimuth), 9),
        round(float(target[1]) + horizontal * math.sin(azimuth), 9),
        round(float(target[2]) + distance * math.sin(elevation), 9),
    ]
    resolved["view_family_id"] = view_id
    resolved["specialized_orbit_offset"] = {
        "azimuth_degrees": float(view["azimuth_offset_degrees"]),
        "elevation_degrees": float(view["elevation_offset_degrees"]),
        "distance_scale": float(view["distance_scale"]),
    }
    return resolved


def validate_resolved_initial_clearance(
    states: list[dict[str, Any]], diameter_m: float, profile: dict[str, Any]
) -> float:
    if [record.get("object_id") for record in states] != list(OBJECT_IDS):
        raise ValueError("resolved specialized object order changed")
    positions = [np.asarray(record["position_m"], dtype=np.float64) for record in states]
    center_distance = float(np.linalg.norm(positions[0] - positions[1]))
    surface_gap = center_distance - float(diameter_m)
    minimum = float(profile["quality"]["minimum_initial_surface_gap_m"])
    if surface_gap + 1.0e-9 < minimum:
        raise ValueError(
            f"{profile['id']} initial surface gap {surface_gap:.6f} is below {minimum:.6f}"
        )
    return surface_gap
