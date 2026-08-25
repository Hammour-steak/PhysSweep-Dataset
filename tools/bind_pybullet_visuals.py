#!/usr/bin/env python3
"""Bind deterministic camera, lighting, and render paths to rigid trajectories."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from object_identity_contract import attach_object_identity
except ModuleNotFoundError:  # import when the script is loaded as tools.* in tests
    from tools.object_identity_contract import attach_object_identity

try:
    from camera_geometry import (
        camera_azimuth_offsets,
        inclined_surface_side_readability,
    )
except ModuleNotFoundError as exc:
    if exc.name != "camera_geometry":
        raise
    from tools.camera_geometry import (
        camera_azimuth_offsets,
        inclined_surface_side_readability,
    )

try:
    from environment_collision import validate_environment_binding
except ModuleNotFoundError as exc:
    if exc.name != "environment_collision":
        raise
    from tools.environment_collision import validate_environment_binding

try:
    from trajectory_contract import object_trajectory_view
except ModuleNotFoundError as exc:
    if exc.name != "trajectory_contract":
        raise
    from tools.trajectory_contract import object_trajectory_view


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_RULES_PATH = PROJECT_ROOT / "configs/one_object_sampling_rules.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_points(
    points: np.ndarray,
    position: np.ndarray,
    target: np.ndarray,
    lens_mm: float,
    sensor_width_mm: float,
    aspect: float,
) -> np.ndarray:
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    if np.linalg.norm(right) < 1.0e-8:
        right = np.asarray([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = points - position
    depth = relative @ forward
    horizontal = relative @ right
    vertical = relative @ up
    half_width = np.maximum(depth * sensor_width_mm / (2.0 * lens_mm), 1.0e-8)
    half_height = half_width / aspect
    return np.column_stack(
        [
            0.5 + horizontal / (2.0 * half_width),
            0.5 + vertical / (2.0 * half_height),
            depth,
        ]
    )


def image_center_visibility_mask(projected: np.ndarray) -> np.ndarray:
    """Return whether projected centers lie inside the actual image."""

    projected = np.asarray(projected, dtype=np.float64)
    if projected.ndim != 2 or projected.shape[1] != 3:
        raise ValueError("projected centers must have shape (N, 3)")
    return np.logical_and.reduce(
        [
            projected[:, 0] >= 0.0,
            projected[:, 0] <= 1.0,
            projected[:, 1] >= 0.0,
            projected[:, 1] <= 1.0,
            projected[:, 2] > 0.1,
        ]
    )


def camera_inside_structural_envelope(
    metadata: dict[str, Any],
    position: np.ndarray,
) -> bool:
    """Keep cameras inside structures whose walls define the usable space."""

    support = metadata["simulation"]["support"]
    envelope = support.get("camera_envelope")
    if envelope is None:
        return True
    if str(envelope.get("type")) != "paired_parallel_walls":
        raise ValueError("unsupported structural camera envelope")
    motion_axis = str(envelope["motion_axis"])
    if motion_axis not in {"x", "y"}:
        raise ValueError("structural camera-envelope axis must be x or y")
    cross_axis = 1 if motion_axis == "x" else 0
    wall_ids = {str(value) for value in envelope["collider_ids"]}
    walls = [
        collider
        for collider in support.get("colliders", [])
        if str(collider.get("id")) in wall_ids
    ]
    if len(wall_ids) != 2 or len(walls) != 2:
        raise ValueError("paired-wall camera envelope requires two colliders")
    clearance_m = float(envelope["clearance_m"])
    if clearance_m <= 0.0:
        raise ValueError("structural camera-envelope clearance must be positive")
    lower_inner = -math.inf
    upper_inner = math.inf
    ceiling = math.inf
    for wall in walls:
        center = np.asarray(wall["position_m"], dtype=np.float64)
        size = np.asarray(wall["size_m"], dtype=np.float64)
        if center[cross_axis] < 0.0:
            lower_inner = max(
                lower_inner, float(center[cross_axis] + 0.5 * size[cross_axis])
            )
        else:
            upper_inner = min(
                upper_inner, float(center[cross_axis] - 0.5 * size[cross_axis])
            )
        ceiling = min(ceiling, float(center[2] + 0.5 * size[2]))
    position = np.asarray(position, dtype=np.float64)
    return bool(
        lower_inner + clearance_m <= position[cross_axis] <= upper_inner - clearance_m
        and position[2] <= ceiling - clearance_m
    )


def camera_target_centers(
    focus_points: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return primary-observation and complete-trajectory target centers."""

    focus_points = np.asarray(focus_points, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if focus_points.ndim != 2 or focus_points.shape[1] != 3:
        raise ValueError("focus points must have shape (N, 3)")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("trajectory positions must have shape (N, 3)")
    primary_target = (focus_points.min(axis=0) + focus_points.max(axis=0)) / 2.0
    full_target = (positions.min(axis=0) + positions.max(axis=0)) / 2.0
    return primary_target, full_target


def camera_azimuth_degrees(rules: dict[str, Any], profile: str) -> float:
    for record in rules["axes"]["camera_axis"]:
        if str(record["label"]) == profile:
            return float(record["overrides"]["view_rule"]["azimuth_degrees"])
    raise ValueError(f"unknown camera profile: {profile}")


def camera_elevation_degrees(rules: dict[str, Any], profile: str) -> float:
    for record in rules["axes"]["camera_axis"]:
        if str(record["label"]) == profile:
            return float(record["overrides"]["view_rule"]["elevation_degrees"])
    raise ValueError(f"unknown camera profile: {profile}")


def sampled_motion_points(
    trajectory: dict[str, np.ndarray],
    object_id: str,
    observation: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(trajectory[f"{object_id}__position_m"], dtype=np.float64)
    lower = np.asarray(trajectory[f"{object_id}__aabb_min_m"], dtype=np.float64)
    upper = np.asarray(trajectory[f"{object_id}__aabb_max_m"], dtype=np.float64)
    event = observation["focus_event"]
    event_type = str(event["type"])
    post_event_frames = max(2, int(event.get("post_event_frames", 2)))

    def first_contact_index(key: str) -> int:
        if key not in trajectory:
            raise ValueError(f"camera observation has no trajectory channel: {key}")
        contacts = np.flatnonzero(np.asarray(trajectory[key], dtype=np.int64) > 0)
        if not contacts.size:
            raise ValueError(f"camera observation event did not occur: {key}")
        return int(contacts[0])

    key_indices = [0]
    if event_type == "fraction":
        fraction = float(event["fraction"])
        if not 0.0 < fraction <= 1.0:
            raise ValueError("camera observation fraction must be in (0, 1]")
        primary_end = max(2, min(len(positions), int(math.ceil(len(positions) * fraction))))
    elif event_type == "primary_support_contact":
        event_index = first_contact_index(
            f"{object_id}__primary_support_contact_count"
        )
        primary_end = min(len(positions), event_index + post_event_frames)
        key_indices.append(event_index)
    elif event_type == "collider_contact":
        event_index = first_contact_index(
            f"{object_id}__collider_contact_count__{event['collider_id']}"
        )
        primary_end = min(len(positions), event_index + post_event_frames)
        key_indices.append(event_index)
    elif event_type == "first_contact":
        contact_keys = [f"{object_id}__primary_support_contact_count"]
        contact_keys.extend(
            key
            for key in trajectory
            if key.startswith(f"{object_id}__collider_contact_count__")
        )
        occurred = []
        for key in contact_keys:
            contacts = np.flatnonzero(np.asarray(trajectory[key], dtype=np.int64) > 0)
            if contacts.size:
                occurred.append(int(contacts[0]))
        if not occurred:
            raise ValueError("camera observation expected a contact event")
        event_index = min(occurred)
        primary_end = min(len(positions), event_index + post_event_frames)
        key_indices.append(event_index)
    else:
        raise ValueError(f"unknown camera focus event: {event_type}")

    key_indices.append(primary_end - 1)
    if str(observation["intent"]) == "ballistic_arc":
        key_indices.append(int(np.argmax(positions[:primary_end, 2])))
    sampled = np.linspace(0, primary_end - 1, min(18, primary_end), dtype=int)
    indices = np.unique(np.concatenate([sampled, np.asarray(key_indices, dtype=int)]))
    points = []
    for index in indices:
        points.append(positions[index])
        for x in (lower[index, 0], upper[index, 0]):
            for y in (lower[index, 1], upper[index, 1]):
                for z in (lower[index, 2], upper[index, 2]):
                    points.append([x, y, z])
    return np.asarray(points, dtype=np.float64), positions, indices


def support_context_points(
    metadata: dict[str, Any],
    azimuth_degrees: float,
    focus_xy: np.ndarray,
    observation: dict[str, Any],
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    support = metadata["simulation"]["support"]
    outward = np.asarray(
        [math.cos(math.radians(azimuth_degrees)), math.sin(math.radians(azimuth_degrees))],
        dtype=np.float64,
    )
    lateral = np.asarray([-outward[1], outward[0]], dtype=np.float64)
    bounds = support["safe_surface_bounds"]
    lower = np.asarray([float(bounds["x"][0]), float(bounds["y"][0])])
    upper = np.asarray([float(bounds["x"][1]), float(bounds["y"][1])])
    center = np.clip(np.asarray(focus_xy, dtype=np.float64), lower, upper)

    ray_limits = []
    for axis in range(2):
        if abs(outward[axis]) <= 1.0e-8:
            continue
        boundary = upper[axis] if outward[axis] > 0.0 else lower[axis]
        ray_limits.append((boundary - center[axis]) / outward[axis])
    positive_limits = [value for value in ray_limits if value >= 0.0]
    edge_distance = min(positive_limits) if positive_limits else 0.0
    edge = np.clip(center + outward * edge_distance * 0.88, lower, upper)
    trajectory_span = float(np.linalg.norm(np.ptp(positions[:, :2], axis=0)))
    local_span = min(0.60, max(0.18, trajectory_span * 0.25))
    side_a = np.clip(center + lateral * local_span, lower, upper)
    side_b = np.clip(center - lateral * local_span, lower, upper)
    inward = np.clip(center - outward * local_span, lower, upper)

    slope_angle = math.radians(float(support["surface_frame"]["slope_angle_degrees"]))
    support_top = float(support["surface_center_z_m"])

    def surface_point(xy: np.ndarray) -> list[float]:
        return [
            float(xy[0]),
            float(xy[1]),
            support_top + float(xy[1]) * math.tan(slope_angle) + 0.01,
        ]

    points = [surface_point(value) for value in (center, edge, side_a, side_b, inward)]
    # A horizontal support is identified by the local contact patch, not by a
    # distant boundary of an arbitrarily large floor plane.
    required_indices = [0, 2, 3]
    context = str(observation["structure_context"])

    if (
        context == "horizontal_surface"
        and str(support.get("scene_class")) == "raised_flat"
        and support.get("motion_axis") in {"x", "y"}
        and "maximum_planar_trajectory_distance_m" in support
    ):
        axis = 0 if support["motion_axis"] == "x" else 1
        cross_axis = 1 - axis
        support_center = np.clip(np.zeros(2, dtype=np.float64), lower, upper)
        axis_half_span = min(
            0.45 * float(upper[axis] - lower[axis]),
            0.5 * float(support["maximum_planar_trajectory_distance_m"]),
        )
        cross_half_span = 0.30 * float(upper[cross_axis] - lower[cross_axis])
        long_axis_anchors = []
        for sign in (-1.0, 1.0):
            value = support_center.copy()
            value[axis] += sign * axis_half_span
            long_axis_anchors.append(np.clip(value, lower, upper))
        for sign in (-1.0, 1.0):
            value = support_center.copy()
            value[cross_axis] += sign * cross_half_span
            long_axis_anchors.append(np.clip(value, lower, upper))
        required_indices = list(
            range(len(points), len(points) + 1 + len(long_axis_anchors))
        )
        points.extend(
            surface_point(value)
            for value in (support_center, *long_axis_anchors)
        )

    if context in {"inclined_surface", "ramp_and_landing"}:
        visual_geometry = support.get("visual_geometry")
        if (
            isinstance(visual_geometry, dict)
            and str(visual_geometry.get("primitive")) == "solid_wedge"
        ):
            width, length = [
                float(value) for value in visual_geometry["size_xy_m"]
            ]
            low_z = float(visual_geometry["base_z_m"]) + 0.01
            high_z = float(visual_geometry["high_top_z_m"]) + 0.01
            wedge_corners = [
                [-width / 2.0, -length / 2.0, low_z],
                [width / 2.0, -length / 2.0, low_z],
                [-width / 2.0, length / 2.0, high_z],
                [width / 2.0, length / 2.0, high_z],
            ]
            required_indices = list(
                range(len(points), len(points) + len(wedge_corners))
            )
            points.extend(wedge_corners)
        else:
            low = np.asarray([center[0], lower[1]], dtype=np.float64)
            high = np.asarray([center[0], upper[1]], dtype=np.float64)
            required_indices = [len(points), len(points) + 1]
            points.extend([surface_point(low), surface_point(high)])

    collider_id = observation["focus_event"].get("collider_id")
    if context == "impact_boundary":
        collider = next(
            (
                value
                for value in support["colliders"]
                if str(value["id"]) == str(collider_id)
            ),
            None,
        )
        if collider is None:
            raise ValueError(f"camera context collider is missing: {collider_id}")
        size = np.asarray(collider["size_m"], dtype=np.float64)
        transform = rotation_matrix_xyz_degrees(collider["rotation_euler_degrees"])
        origin = np.asarray(collider["position_m"], dtype=np.float64)
        local = np.asarray(
            [
                [0.0, 0.0, size[2] / 2.0],
                [-size[0] / 2.0, 0.0, size[2] / 2.0],
                [size[0] / 2.0, 0.0, size[2] / 2.0],
            ],
            dtype=np.float64,
        )
        required_indices = list(range(len(points), len(points) + len(local)))
        points.extend((local @ transform.T + origin).tolist())

    if context == "ramp_and_landing":
        collider = next(
            (
                value
                for value in support["colliders"]
                if str(value["id"]) == str(collider_id)
            ),
            None,
        )
        if collider is None:
            raise ValueError(f"camera context collider is missing: {collider_id}")
        size = np.asarray(collider["size_m"], dtype=np.float64)
        transform = rotation_matrix_xyz_degrees(collider["rotation_euler_degrees"])
        origin = np.asarray(collider["position_m"], dtype=np.float64)
        local_landing = transform.T @ (positions[-1] - origin)
        local_landing[:2] = np.clip(
            local_landing[:2],
            -size[:2] / 2.0,
            size[:2] / 2.0,
        )
        local_landing[2] = size[2] / 2.0 + 0.01
        landing = local_landing @ transform.T + origin
        required_indices.append(len(points))
        points.append(landing.tolist())

    if context == "edge_and_landing":
        travel = positions[-1, :2] - positions[0, :2]
        if np.linalg.norm(travel) > 1.0e-8:
            travel /= np.linalg.norm(travel)
        else:
            travel = outward
        limits = []
        for axis in range(2):
            if abs(travel[axis]) <= 1.0e-8:
                continue
            boundary = upper[axis] if travel[axis] > 0.0 else lower[axis]
            limits.append((boundary - center[axis]) / travel[axis])
        positive = [value for value in limits if value >= 0.0]
        edge_xy = np.clip(
            center + travel * (min(positive) if positive else 0.0), lower, upper
        )
        landing = positions[-1].copy()
        landing[2] = min(float(landing[2]), support_top)
        required_indices = [len(points), len(points) + 1]
        points.extend([surface_point(edge_xy), landing.tolist()])

    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(required_indices, dtype=int),
    )


def rotation_matrix_xyz_degrees(euler_degrees: list[float]) -> np.ndarray:
    x, y, z = np.radians(np.asarray(euler_degrees, dtype=np.float64))
    rx = np.asarray([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.asarray([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.asarray([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def segment_intersects_box(
    start: np.ndarray, end: np.ndarray, collider: dict[str, Any]
) -> bool:
    """Return whether an oriented box blocks the open segment to a target point."""

    center = np.asarray(collider["position_m"], dtype=np.float64)
    half = np.asarray(collider["size_m"], dtype=np.float64) / 2.0
    rotation = rotation_matrix_xyz_degrees(collider["rotation_euler_degrees"])
    local_start = rotation.T @ (np.asarray(start, dtype=np.float64) - center)
    local_end = rotation.T @ (np.asarray(end, dtype=np.float64) - center)
    direction = local_end - local_start
    lower_t, upper_t = 0.0, 0.985
    for axis in range(3):
        if abs(direction[axis]) <= 1.0e-10:
            if local_start[axis] < -half[axis] or local_start[axis] > half[axis]:
                return False
            continue
        first = (-half[axis] - local_start[axis]) / direction[axis]
        second = (half[axis] - local_start[axis]) / direction[axis]
        near, far = sorted((float(first), float(second)))
        lower_t = max(lower_t, near)
        upper_t = min(upper_t, far)
        if lower_t > upper_t:
            return False
    return upper_t >= 0.0 and lower_t <= 0.985


def segments_intersect_box(
    start: np.ndarray, ends: np.ndarray, collider: dict[str, Any]
) -> np.ndarray:
    """Vectorized equivalent of segment_intersects_box for one camera pose."""

    points = np.asarray(ends, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("segment endpoints must have shape (N, 3)")
    center = np.asarray(collider["position_m"], dtype=np.float64)
    half = np.asarray(collider["size_m"], dtype=np.float64) / 2.0
    rotation = rotation_matrix_xyz_degrees(collider["rotation_euler_degrees"])
    local_start = rotation.T @ (np.asarray(start, dtype=np.float64) - center)
    local_ends = (points - center) @ rotation
    directions = local_ends - local_start
    lower_t = np.zeros(len(points), dtype=np.float64)
    upper_t = np.full(len(points), 0.985, dtype=np.float64)
    possible = np.ones(len(points), dtype=bool)
    for axis in range(3):
        direction = directions[:, axis]
        parallel = np.abs(direction) <= 1.0e-10
        possible &= ~(
            parallel
            & (
                (local_start[axis] < -half[axis])
                | (local_start[axis] > half[axis])
            )
        )
        nonparallel = ~parallel
        if np.any(nonparallel):
            first = (-half[axis] - local_start[axis]) / direction[nonparallel]
            second = (half[axis] - local_start[axis]) / direction[nonparallel]
            lower_t[nonparallel] = np.maximum(
                lower_t[nonparallel], np.minimum(first, second)
            )
            upper_t[nonparallel] = np.minimum(
                upper_t[nonparallel], np.maximum(first, second)
            )
        possible &= lower_t <= upper_t
    return possible & (upper_t >= 0.0) & (lower_t <= 0.985)


def unoccluded_fraction(
    camera_position: np.ndarray,
    points: np.ndarray,
    colliders: list[dict[str, Any]],
) -> float:
    blockers = [
        collider
        for collider in colliders
        if bool(collider.get("visible", True))
        and collider.get("primitive") == "box"
    ]
    points = np.asarray(points, dtype=np.float64)
    if not len(points):
        return 0.0
    blocked = np.zeros(len(points), dtype=bool)
    for collider in blockers:
        blocked |= segments_intersect_box(camera_position, points, collider)
        if np.all(blocked):
            break
    return float((~blocked).mean())


def camera_occlusion_colliders(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every frozen box that can visibly block the selected camera."""

    support_blockers = [
        collider
        for collider in metadata["simulation"]["support"]["colliders"]
        if bool(collider.get("visible", True))
        and bool(collider.get("occludes_camera", False))
        and collider.get("primitive") == "box"
    ]
    environment_blockers = [
        collider
        for collider in metadata.get("environment_binding", {}).get(
            "colliders", []
        )
        if bool(collider.get("visible", True))
        and bool(collider.get("collision_enabled", True))
        and collider.get("primitive") == "box"
    ]
    return [*support_blockers, *environment_blockers]


def solve_camera(
    metadata: dict[str, Any], trajectory: dict[str, np.ndarray], rules: dict[str, Any]
) -> dict[str, Any]:
    request = metadata["camera_request"]
    profile = str(request["profile"])
    observation = request["observation"]
    object_id = str(metadata["simulation"]["objects"][0]["object_id"])
    focus_points, positions, center_indices = sampled_motion_points(
        trajectory,
        object_id,
        observation,
    )
    azimuth_base = camera_azimuth_degrees(rules, profile)
    is_ramp = metadata["simulation"]["support"]["support_shape"] == "inclined_ramp"
    minimum_ramp_side_readability = float(
        rules["camera_observation"][
            "minimum_inclined_surface_side_readability"
        ]
    )
    object_size = np.asarray(
        metadata["simulation"]["objects"][0]["geometry"]["size_m"], dtype=np.float64
    )
    object_lower = np.asarray(
        trajectory[f"{object_id}__aabb_min_m"], dtype=np.float64
    )
    object_upper = np.asarray(
        trajectory[f"{object_id}__aabb_max_m"], dtype=np.float64
    )
    initial_corners = np.asarray(
        [
            [x, y, z]
            for x in (object_lower[0, 0], object_upper[0, 0])
            for y in (object_lower[0, 1], object_upper[0, 1])
            for z in (object_lower[0, 2], object_upper[0, 2])
        ],
        dtype=np.float64,
    )
    support_top = float(metadata["simulation"]["support"]["surface_center_z_m"])
    primary_target, full_target = camera_target_centers(focus_points, positions)
    structure_target = None
    if str(observation["structure_context"]) in {
        "ramp_and_landing",
        "edge_and_landing",
    }:
        context_points, required_anchor_indices = support_context_points(
            metadata,
            camera_azimuth_degrees(rules, profile),
            primary_target[:2],
            observation,
            positions[center_indices],
        )
        framing_points = np.vstack(
            [positions[center_indices], context_points[required_anchor_indices]]
        )
        structure_target = (
            framing_points.min(axis=0) + framing_points.max(axis=0)
        ) / 2.0
    requested_target_blend = float(request["full_trajectory_camera_target_blend"])
    requested_lens = min(44.0, float(request["focal_length_mm"]))
    scene_visual = metadata["appearance"]["scene_visual"]
    camera_context = scene_visual.get("camera_context", {})
    composition = scene_visual.get("composition")
    reviewed_camera = (
        composition.get("camera", {})
        if isinstance(composition, dict)
        and str(composition.get("review_status")) == "approved"
        else {}
    )
    maximum_local_azimuth_deviation = (
        float(reviewed_camera["maximum_local_azimuth_deviation_degrees"])
        if reviewed_camera
        else None
    )
    reviewed_preferred_elevation = (
        float(reviewed_camera["preferred_elevation_degrees"])
        if reviewed_camera
        else None
    )
    context_depth_offset = float(
        reviewed_camera["target_depth_offset_m"]
        if reviewed_camera
        else camera_context.get("depth_offset_m", 0.0)
    )
    context_lateral_offset = float(
        reviewed_camera["target_lateral_offset_m"]
        if reviewed_camera
        else camera_context.get("lateral_offset_m", 0.0)
    )
    context_target_z_offset = float(
        reviewed_camera["target_z_offset_m"]
        if reviewed_camera
        else camera_context.get("target_z_offset_m", 0.0)
    )
    context_focal_cap = (
        float(reviewed_camera["focal_length_cap_mm"])
        if reviewed_camera
        else camera_context.get("focal_length_cap_mm")
    )
    if context_focal_cap is not None:
        requested_lens = min(requested_lens, float(context_focal_cap))
    focal_length_candidates = [requested_lens]
    focal_length_candidates.extend(
        value for value in (40.0, 36.0, 32.0, 28.0) if value < requested_lens
    )
    sensor_width = 32.0
    aspect = 16.0 / 9.0
    planar_object_size = max(float(object_size[0]), float(object_size[1]))
    object_size_scale = min(1.60, max(0.80, math.sqrt(planar_object_size / 0.18)))
    preferred_span = float(observation["preferred_object_span_ndc"]) * object_size_scale
    minimum_median_object_span = float(
        observation["minimum_median_object_span_ndc"]
    )
    minimum_center_fraction = float(
        request.get("minimum_primary_trajectory_center_visible_fraction", 0.75)
    )
    minimum_full_center_fraction = float(
        request["minimum_full_trajectory_center_visible_fraction"]
    )
    if str(observation["intent"]) == "ballistic_arc":
        minimum_full_center_fraction = max(
            minimum_full_center_fraction,
            float(
                rules["camera_observation"][
                    "minimum_ballistic_full_trajectory_center_visible_fraction"
                ]
            ),
        )
    minimum_object_span = float(request["minimum_initial_object_span_ndc"])
    minimum_initial_object_visible_fraction = float(
        request.get("minimum_initial_object_visible_fraction", 0.75)
    )
    initial_object_center_margin = float(
        request.get("initial_object_center_margin_ndc", 0.05)
    )
    if not 0.0 <= initial_object_center_margin < 0.5:
        raise ValueError("initial object center margin must be in [0, 0.5)")
    soft_maximum_focus_span = float(request["soft_maximum_focus_span_ndc"])
    maximum_focus_span = float(request["maximum_focus_span_ndc"])
    focus_span_penalty_weight = float(request["focus_span_penalty_weight"])
    if not 0.0 < soft_maximum_focus_span < maximum_focus_span:
        raise ValueError("camera focus-span limits must be positive and ordered")
    maximum_camera_distance = float(request["maximum_camera_distance_m"])
    support_size = np.asarray(metadata["simulation"]["support"]["size_m"][:2], dtype=np.float64)
    support_diagonal = float(np.linalg.norm(support_size))
    minimum_camera_distance = max(
        float(request.get("minimum_camera_distance_floor_m", 1.45)),
        float(request.get("minimum_camera_distance_offset_m", 0.50))
        + support_diagonal
        * float(request.get("minimum_camera_distance_support_diagonal_scale", 0.36)),
    )
    if reviewed_camera:
        minimum_camera_distance = max(
            minimum_camera_distance,
            float(reviewed_camera["minimum_distance_m"]),
        )
    distance_allowance = float(
        request.get("maximum_camera_distance_above_minimum_m", 1.70)
    )
    if camera_context:
        distance_allowance += (
            1.50 * context_depth_offset + 0.50 * context_target_z_offset
        )
    maximum_camera_distance = min(
        maximum_camera_distance,
        minimum_camera_distance + distance_allowance,
    )
    if reviewed_camera:
        maximum_camera_distance = min(
            maximum_camera_distance,
            float(reviewed_camera["maximum_distance_m"]),
        )
    if maximum_camera_distance < minimum_camera_distance:
        raise ValueError(
            "reviewed environment camera corridor cannot frame support for "
            f"{metadata['scene_id']}: minimum={minimum_camera_distance:.6f} "
            f"maximum={maximum_camera_distance:.6f}"
        )
    preferred_camera_distance = minimum_camera_distance + float(
        request.get("preferred_camera_distance_offset_m", 0.25)
    )
    distance_penalty_weight = float(request.get("camera_distance_penalty_weight", 0.08))
    maximum_object_span = float(request.get("maximum_initial_object_span_ndc", 0.22))
    requested_minimum_elevation = request.get("minimum_camera_elevation_degrees")
    requested_maximum_elevation = request.get("maximum_camera_elevation_degrees")
    minimum_elevation_candidates = [
        float(value)
        for value in (
            requested_minimum_elevation,
            camera_context.get("minimum_elevation_degrees"),
            reviewed_camera.get("minimum_elevation_degrees")
            if reviewed_camera
            else None,
        )
        if value is not None
    ]
    maximum_elevation_candidates = [
        float(value)
        for value in (
            requested_maximum_elevation,
            camera_context.get("maximum_elevation_degrees"),
            reviewed_camera.get("maximum_elevation_degrees")
            if reviewed_camera
            else None,
        )
        if value is not None
    ]
    minimum_camera_elevation = (
        max(minimum_elevation_candidates) if minimum_elevation_candidates else None
    )
    maximum_camera_elevation = (
        min(maximum_elevation_candidates) if maximum_elevation_candidates else None
    )
    if (
        minimum_camera_elevation is not None
        and maximum_camera_elevation is not None
        and minimum_camera_elevation > maximum_camera_elevation
    ):
        raise ValueError(
            "camera elevation constraints do not overlap for "
            f"{metadata['scene_id']}: motion_min={requested_minimum_elevation}, "
            f"motion_max={requested_maximum_elevation}, "
            "environment_min="
            f"{camera_context.get('minimum_elevation_degrees')}, "
            "environment_max="
            f"{camera_context.get('maximum_elevation_degrees')}"
        )
    minimum_context_fraction = float(
        request.get("minimum_support_context_visible_fraction", 0.80)
    )
    minimum_anchor_fraction = float(
        observation["minimum_anchor_visible_fraction"]
    )
    minimum_anchor_unoccluded = float(
        observation["minimum_anchor_unoccluded_fraction"]
    )
    minimum_primary_unoccluded = float(
        request.get("minimum_primary_trajectory_unoccluded_fraction", 0.90)
    )
    minimum_full_unoccluded = float(
        request.get("minimum_full_trajectory_unoccluded_fraction", 0.70)
    )
    colliders = camera_occlusion_colliders(metadata)
    has_camera_blockers = bool(colliders)
    full_occlusion_indices = np.unique(
        np.linspace(0, len(positions) - 1, min(18, len(positions)), dtype=int)
    )
    def evaluate_target(
        motion_target: np.ndarray,
        target_blend: float,
        focal_length_mm: float,
        wide_azimuth: bool = False,
        absolute_elevations: tuple[float, ...] | None = None,
        target_mode: str = "motion",
    ) -> list[dict[str, Any]]:
        result = []
        azimuth_offsets = camera_azimuth_offsets(
            maximum_deviation_degrees=maximum_local_azimuth_deviation,
            wide=wide_azimuth,
            has_camera_blockers=has_camera_blockers,
        )
        for azimuth_offset in azimuth_offsets:
            azimuth = math.radians(azimuth_base + azimuth_offset)
            outward_direction = np.asarray(
                [math.cos(azimuth), math.sin(azimuth)], dtype=np.float64
            )
            lateral_direction = np.asarray(
                [-outward_direction[1], outward_direction[0]], dtype=np.float64
            )
            target = np.asarray(motion_target, dtype=np.float64).copy()
            target[:2] += (
                -outward_direction * context_depth_offset
                + lateral_direction * context_lateral_offset
            )
            target[2] += context_target_z_offset
            ramp_side_readability = inclined_surface_side_readability(
                azimuth_base + azimuth_offset
            )
            if (
                is_ramp
                and ramp_side_readability < minimum_ramp_side_readability
            ):
                continue
            context_points, required_anchor_indices = support_context_points(
                metadata,
                azimuth_base + azimuth_offset,
                motion_target[:2],
                observation,
                positions[center_indices],
            )
            base_elevation = (
                reviewed_preferred_elevation
                if reviewed_preferred_elevation is not None
                else camera_elevation_degrees(rules, profile)
            )
            if is_ramp:
                base_elevation = min(base_elevation, 42.0)
            context_elevations = (
                (base_elevation, base_elevation + 10.0, base_elevation - 8.0,
                 base_elevation + 18.0, base_elevation - 14.0)
                if camera_context
                else (base_elevation, base_elevation + 10.0,
                      base_elevation - 8.0, base_elevation + 18.0)
            )
            elevation_candidates = (
                tuple(dict.fromkeys(float(value) for value in absolute_elevations))
                if absolute_elevations is not None
                else context_elevations
            )
            for elevation_degrees in elevation_candidates:
                if (
                    minimum_camera_elevation is not None
                    and elevation_degrees < float(minimum_camera_elevation)
                ):
                    continue
                if (
                    maximum_camera_elevation is not None
                    and elevation_degrees > float(maximum_camera_elevation)
                ):
                    continue
                elevation_offset = elevation_degrees - base_elevation
                elevation = math.radians(elevation_degrees)
                distance_samples = max(
                    48,
                    int(round((maximum_camera_distance - minimum_camera_distance) / 0.05)) + 1,
                )
                for distance in np.linspace(
                    minimum_camera_distance, maximum_camera_distance, distance_samples
                ):
                    horizontal_distance = float(distance) * math.cos(elevation)
                    position = target + np.asarray(
                        [
                            horizontal_distance * math.cos(azimuth),
                            horizontal_distance * math.sin(azimuth),
                            float(distance) * math.sin(elevation),
                        ]
                    )
                    if not camera_inside_structural_envelope(metadata, position):
                        continue
                    projected_focus = project_points(
                        focus_points,
                        position,
                        target,
                        focal_length_mm,
                        sensor_width,
                        aspect,
                    )
                    projected_centers = project_points(
                        positions[center_indices],
                        position,
                        target,
                        focal_length_mm,
                        sensor_width,
                        aspect,
                    )
                    projected_full_centers = project_points(
                        positions,
                        position,
                        target,
                        focal_length_mm,
                        sensor_width,
                        aspect,
                    )
                    projected_context = project_points(
                        context_points,
                        position,
                        target,
                        focal_length_mm,
                        sensor_width,
                        aspect,
                    )
                    projected_initial_corners = project_points(
                        initial_corners,
                        position,
                        target,
                        focal_length_mm,
                        sensor_width,
                        aspect,
                    )
                    center_inside = image_center_visibility_mask(projected_centers)
                    center_fraction = float(center_inside.mean())
                    full_center_inside = image_center_visibility_mask(
                        projected_full_centers
                    )
                    full_center_fraction = float(full_center_inside.mean())
                    initial_inside = bool(
                        initial_object_center_margin
                        <= projected_centers[0, 0]
                        <= 1.0 - initial_object_center_margin
                        and initial_object_center_margin
                        <= projected_centers[0, 1]
                        <= 1.0 - initial_object_center_margin
                        and projected_centers[0, 2] > 0.1
                    )
                    initial_corner_inside = np.logical_and.reduce(
                        [
                            projected_initial_corners[:, 0] >= 0.0,
                            projected_initial_corners[:, 0] <= 1.0,
                            projected_initial_corners[:, 1] >= 0.0,
                            projected_initial_corners[:, 1] <= 1.0,
                            projected_initial_corners[:, 2] > 0.1,
                        ]
                    )
                    initial_object_visible_fraction = float(initial_corner_inside.mean())
                    initial_object_inside = (
                        initial_object_visible_fraction
                        >= minimum_initial_object_visible_fraction
                    )
                    context_inside = np.logical_and.reduce(
                        [
                            projected_context[:, 0] >= -0.12,
                            projected_context[:, 0] <= 1.12,
                            projected_context[:, 1] >= -0.12,
                            projected_context[:, 1] <= 1.12,
                            projected_context[:, 2] > 0.1,
                        ]
                    )
                    context_fraction = float(context_inside.mean())
                    anchor_fraction = float(
                        context_inside[required_anchor_indices].mean()
                    )
                    object_span = max(
                        float(np.ptp(projected_initial_corners[:, 0])),
                        float(np.ptp(projected_initial_corners[:, 1])),
                    )
                    primary_object_spans = []
                    for index in center_indices:
                        corners = np.asarray(
                            [
                                [x, y, z]
                                for x in (object_lower[index, 0], object_upper[index, 0])
                                for y in (object_lower[index, 1], object_upper[index, 1])
                                for z in (object_lower[index, 2], object_upper[index, 2])
                            ],
                            dtype=np.float64,
                        )
                        projected = project_points(
                            corners,
                            position,
                            target,
                            focal_length_mm,
                            sensor_width,
                            aspect,
                        )
                        primary_object_spans.append(
                            max(
                                float(np.ptp(projected[:, 0])),
                                float(np.ptp(projected[:, 1])),
                            )
                        )
                    median_object_span = float(np.median(primary_object_spans))
                    focus_span = max(
                        float(np.ptp(projected_focus[:, 0])),
                        float(np.ptp(projected_focus[:, 1])),
                    )
                    primary_unoccluded = unoccluded_fraction(
                        position, positions[center_indices], colliders
                    )
                    full_unoccluded = unoccluded_fraction(
                        position, positions[full_occlusion_indices], colliders
                    )
                    target_unoccluded = unoccluded_fraction(
                        position, target[None, :], colliders
                    )
                    required_anchors_unoccluded = unoccluded_fraction(
                        position,
                        context_points[required_anchor_indices],
                        colliders,
                    )
                    admissible = (
                        initial_inside
                        and initial_object_inside
                        and center_fraction >= minimum_center_fraction
                        and full_center_fraction >= minimum_full_center_fraction
                        and context_fraction >= minimum_context_fraction
                        and anchor_fraction >= minimum_anchor_fraction
                        and focus_span <= maximum_focus_span
                        and object_span >= minimum_object_span
                        and object_span <= maximum_object_span
                        and median_object_span >= minimum_median_object_span
                        and primary_unoccluded >= minimum_primary_unoccluded
                        and full_unoccluded >= minimum_full_unoccluded
                        and required_anchors_unoccluded
                        >= minimum_anchor_unoccluded
                    )
                    score = (
                        abs(median_object_span - preferred_span) * 5.0
                        + (1.0 - center_fraction) * 2.0
                        + (1.0 - full_center_fraction) * 0.65
                        + (1.0 - context_fraction) * 0.75
                        + (1.0 - anchor_fraction) * 1.25
                        + (1.0 - primary_unoccluded) * 3.0
                        + (1.0 - full_unoccluded) * 0.8
                        + (1.0 - required_anchors_unoccluded) * 1.25
                        + max(0.0, focus_span - soft_maximum_focus_span)
                        * focus_span_penalty_weight
                        + abs(float(distance) - preferred_camera_distance)
                        * distance_penalty_weight
                        + abs(azimuth_offset) * 0.001
                        + abs(elevation_offset) * 0.015
                        + (1.0 - ramp_side_readability) * (0.8 if is_ramp else 0.0)
                    )
                    result.append(
                        {
                            "admissible": admissible,
                            "score": score,
                            "position": position,
                            "target": target,
                            "motion_target": motion_target,
                            "target_blend": target_blend,
                            "target_mode": target_mode,
                            "distance": float(distance),
                            "focal_length_mm": float(focal_length_mm),
                            "azimuth_degrees": azimuth_base + azimuth_offset,
                            "elevation_degrees": math.degrees(elevation),
                            "center_fraction": center_fraction,
                            "full_center_fraction": full_center_fraction,
                            "context_fraction": context_fraction,
                            "anchor_fraction": anchor_fraction,
                            "object_span": object_span,
                            "median_object_span": median_object_span,
                            "initial_object_visible_fraction": initial_object_visible_fraction,
                            "focus_span": focus_span,
                            "primary_unoccluded_fraction": primary_unoccluded,
                            "full_unoccluded_fraction": full_unoccluded,
                            "target_unoccluded_fraction": target_unoccluded,
                            "required_structure_anchors_unoccluded_fraction": (
                                required_anchors_unoccluded
                            ),
                            "ramp_side_readability": ramp_side_readability,
                            "constraints": {
                                "initial_center_inside": initial_inside,
                                "initial_object_inside": initial_object_inside,
                                "primary_center_fraction": center_fraction >= minimum_center_fraction,
                                "full_center_fraction": full_center_fraction >= minimum_full_center_fraction,
                                "support_context_fraction": context_fraction >= minimum_context_fraction,
                                "required_structure_anchor_fraction": anchor_fraction >= minimum_anchor_fraction,
                                "focus_span": focus_span <= maximum_focus_span,
                                "object_span": object_span >= minimum_object_span,
                                "maximum_object_span": object_span <= maximum_object_span,
                                "median_object_span": median_object_span >= minimum_median_object_span,
                                "primary_unoccluded_fraction": primary_unoccluded >= minimum_primary_unoccluded,
                                "full_unoccluded_fraction": full_unoccluded >= minimum_full_unoccluded,
                                "required_structure_anchors_unoccluded_fraction": (
                                    required_anchors_unoccluded
                                    >= minimum_anchor_unoccluded
                                ),
                            },
                        }
                    )
        return result

    candidates: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []
    attempted_blends = []
    attempted_focal_lengths = []
    used_safety_elevation_fallback = False

    def search_focal_length(focal_length_mm: float) -> list[dict[str, Any]]:
        nonlocal used_safety_elevation_fallback
        for target_blend in dict.fromkeys(
            (requested_target_blend, requested_target_blend / 2.0, 0.0)
        ):
            attempted_blends.append(target_blend)
            motion_target = (
                (1.0 - target_blend) * primary_target
                + target_blend * full_target
            )
            motion_target[2] = max(
                max(0.18, support_top - 0.45),
                min(float(motion_target[2]), support_top + 0.78),
            )
            target_candidates = evaluate_target(
                motion_target,
                target_blend,
                focal_length_mm,
            )
            candidates.extend(target_candidates)
            accepted = [
                record for record in target_candidates if record["admissible"]
            ]
            if accepted:
                return accepted
        if structure_target is not None:
            target_candidates = evaluate_target(
                structure_target,
                requested_target_blend,
                focal_length_mm,
                wide_azimuth=True,
                target_mode="motion_and_required_structure",
            )
            candidates.extend(target_candidates)
            accepted = [
                record for record in target_candidates if record["admissible"]
            ]
            if accepted:
                return accepted
        for initial_bias in (0.20, 0.35, 0.50):
            attempted_blends.append(-initial_bias)
            motion_target = (
                (1.0 - initial_bias) * primary_target
                + initial_bias * positions[0]
            )
            motion_target[2] = max(
                max(0.18, support_top - 0.45),
                min(float(motion_target[2]), support_top + 0.78),
            )
            target_candidates = evaluate_target(
                motion_target,
                -initial_bias,
                focal_length_mm,
                wide_azimuth=initial_bias >= 0.35,
            )
            candidates.extend(target_candidates)
            accepted = [
                record for record in target_candidates if record["admissible"]
            ]
            if accepted:
                return accepted
        safety_elevations = (20.0, 24.0, 27.0, 32.0, 38.0, 42.0, 48.0)
        for target_blend in dict.fromkeys(
            (requested_target_blend, requested_target_blend / 2.0, 0.0)
        ):
            motion_target = (
                (1.0 - target_blend) * primary_target
                + target_blend * full_target
            )
            motion_target[2] = max(
                max(0.18, support_top - 0.45),
                min(float(motion_target[2]), support_top + 0.78),
            )
            target_candidates = evaluate_target(
                motion_target,
                target_blend,
                focal_length_mm,
                wide_azimuth=True,
                absolute_elevations=safety_elevations,
            )
            candidates.extend(target_candidates)
            accepted = [
                record for record in target_candidates if record["admissible"]
            ]
            if accepted:
                used_safety_elevation_fallback = True
                return accepted
        return []

    for focal_length_mm in focal_length_candidates:
        attempted_focal_lengths.append(focal_length_mm)
        admitted = search_focal_length(focal_length_mm)
        if admitted:
            break
    if not candidates:
        raise ValueError(
            f"camera solver evaluated no poses for {metadata['scene_id']}"
        )
    best = min(admitted or candidates, key=lambda record: record["score"])
    camera_variant_index = 0
    if admitted:
        # Geometry decides which poses are valid. The scene seed only chooses
        # among near-equivalent valid poses, preserving reproducibility while
        # avoiding a systematic collapse to the nominal camera angle.
        quality_limit = float(best["score"]) + 0.65
        quality_pool = [
            record for record in admitted if float(record["score"]) <= quality_limit
        ]
        scene_digest = hashlib.sha256(
            f"generic-camera:{metadata['scene_id']}".encode("utf-8")
        ).digest()
        azimuths = sorted(
            {round(float(record["azimuth_degrees"]), 6) for record in quality_pool}
        )
        elevations = sorted(
            {round(float(record["elevation_degrees"]), 6) for record in quality_pool}
        )
        preferred_azimuth = azimuths[int(scene_digest[0]) % len(azimuths)]
        preferred_elevation = elevations[int(scene_digest[1]) % len(elevations)]
        ranked_pool = sorted(
            quality_pool,
            key=lambda record: (
                abs(float(record["azimuth_degrees"]) - preferred_azimuth) / 24.0
                + abs(float(record["elevation_degrees"]) - preferred_elevation) / 8.0,
                float(record["score"]),
            ),
        )
        best = ranked_pool[0]
        camera_variant_index = quality_pool.index(best)
    if not best["admissible"]:
        largest_object = max(candidates, key=lambda record: record["object_span"])
        closest_to_object_threshold = min(
            candidates,
            key=lambda record: abs(record["object_span"] - minimum_object_span),
        )
        best_structure_target = min(
            (
                record
                for record in candidates
                if record["target_mode"] == "motion_and_required_structure"
            ),
            key=lambda record: record["score"],
            default=None,
        )
        best_anchor_coverage = min(
            candidates,
            key=lambda record: (-record["anchor_fraction"], record["score"]),
        )
        raise ValueError(
            f"camera solver found no admissible pose for {metadata['scene_id']}: "
            f"best={best}; closest_object_threshold={closest_to_object_threshold}; "
            f"largest_object={largest_object}; "
            f"best_structure_target={best_structure_target}; "
            f"best_anchor_coverage={best_anchor_coverage}"
        )
    return {
        "solver_version": "motion_structure_camera_v12",
        "profile": profile,
        "observation_intent": str(observation["intent"]),
        "structure_context": str(observation["structure_context"]),
        "position_m": [round(float(value), 6) for value in best["position"]],
        "target_m": [round(float(value), 6) for value in best["target"]],
        "focal_length_mm": round(float(best["focal_length_mm"]), 6),
        "sensor_width_mm": sensor_width,
        "clip_start_m": 0.05,
        "clip_end_m": 100.0,
        "diagnostics": {
            "deterministic_variant_index": camera_variant_index,
            "selected_azimuth_degrees": round(float(best["azimuth_degrees"]), 6),
            "selected_elevation_degrees": round(float(best["elevation_degrees"]), 6),
            "evaluated_candidates": len(candidates),
            "attempted_full_trajectory_target_blends": [
                round(float(value), 6) for value in attempted_blends
            ],
            "attempted_focal_lengths_mm": [
                round(float(value), 6) for value in attempted_focal_lengths
            ],
            "selected_focal_length_mm": round(
                float(best["focal_length_mm"]), 6
            ),
            "selected_full_trajectory_target_blend": round(
                float(best["target_blend"]), 6
            ),
            "selected_target_mode": str(best["target_mode"]),
            "motion_target_m": [
                round(float(value), 6) for value in best["motion_target"]
            ],
            "environment_camera_context": {
                "enabled": bool(camera_context),
                "depth_offset_m": round(context_depth_offset, 6),
                "lateral_offset_m": round(context_lateral_offset, 6),
                "target_z_offset_m": round(context_target_z_offset, 6),
                "focal_length_cap_mm": (
                    round(float(context_focal_cap), 6)
                    if context_focal_cap is not None
                    else None
                ),
                "minimum_elevation_degrees": (
                    round(float(minimum_camera_elevation), 6)
                    if minimum_camera_elevation is not None
                    else None
                ),
                "maximum_elevation_degrees": (
                    round(float(maximum_camera_elevation), 6)
                    if maximum_camera_elevation is not None
                    else None
                ),
                "reviewed_camera_corridor": (
                    {
                        "preferred_local_azimuth_degrees": round(
                            float(
                                reviewed_camera[
                                    "preferred_local_azimuth_degrees"
                                ]
                            ),
                            6,
                        ),
                        "maximum_local_azimuth_deviation_degrees": round(
                            float(maximum_local_azimuth_deviation), 6
                        ),
                        "preferred_elevation_degrees": round(
                            float(reviewed_preferred_elevation), 6
                        ),
                        "minimum_distance_m": round(
                            float(reviewed_camera["minimum_distance_m"]), 6
                        ),
                        "maximum_distance_m": round(
                            float(reviewed_camera["maximum_distance_m"]), 6
                        ),
                        "target_depth_offset_m": round(
                            float(reviewed_camera["target_depth_offset_m"]), 6
                        ),
                        "target_lateral_offset_m": round(
                            float(reviewed_camera["target_lateral_offset_m"]),
                            6,
                        ),
                        "target_z_offset_m": round(
                            float(reviewed_camera["target_z_offset_m"]), 6
                        ),
                        "focal_length_cap_mm": round(
                            float(reviewed_camera["focal_length_cap_mm"]), 6
                        ),
                    }
                    if reviewed_camera
                    else None
                ),
            },
            "used_safety_elevation_fallback": used_safety_elevation_fallback,
            "score": round(float(best["score"]), 6),
            "distance_m": round(float(best["distance"]), 6),
            "minimum_distance_m": round(float(minimum_camera_distance), 6),
            "preferred_distance_m": round(float(preferred_camera_distance), 6),
            "maximum_distance_m": round(float(maximum_camera_distance), 6),
            "azimuth_degrees": round(float(best["azimuth_degrees"]), 6),
            "elevation_degrees": round(float(best["elevation_degrees"]), 6),
            "ramp_side_readability": round(
                float(best["ramp_side_readability"]), 6
            ),
            "primary_center_visible_fraction": round(float(best["center_fraction"]), 6),
            "primary_center_evaluated_frame_count": int(len(center_indices)),
            "full_trajectory_center_visible_fraction": round(
                float(best["full_center_fraction"]), 6
            ),
            "full_trajectory_center_evaluated_frame_count": int(len(positions)),
            "center_visibility_bounds_ndc": [0.0, 1.0],
            "support_context_visible_fraction": round(float(best["context_fraction"]), 6),
            "required_structure_anchor_visible_fraction": round(
                float(best["anchor_fraction"]), 6
            ),
            "initial_object_span_ndc": round(float(best["object_span"]), 6),
            "initial_object_visible_fraction": round(
                float(best["initial_object_visible_fraction"]), 6
            ),
            "preferred_object_span_ndc": round(float(preferred_span), 6),
            "median_primary_object_span_ndc": round(
                float(best["median_object_span"]), 6
            ),
            "focus_span_ndc": round(float(best["focus_span"]), 6),
            "primary_trajectory_unoccluded_fraction": round(
                float(best["primary_unoccluded_fraction"]), 6
            ),
            "full_trajectory_unoccluded_fraction": round(
                float(best["full_unoccluded_fraction"]), 6
            ),
            "target_unoccluded_fraction": round(
                float(best["target_unoccluded_fraction"]), 6
            ),
            "required_structure_anchors_unoccluded_fraction": round(
                float(best["required_structure_anchors_unoccluded_fraction"]),
                6,
            ),
            "minimum_required_structure_anchor_unoccluded_fraction": round(
                minimum_anchor_unoccluded, 6
            ),
        },
    }


def shadow_readable_lighting(metadata: dict[str, Any]) -> dict[str, Any]:
    object_size = np.asarray(
        metadata["simulation"]["objects"][0]["geometry"]["size_m"],
        dtype=np.float64,
    )
    footprint_m = float(max(object_size[0], object_size[1]))
    thickness_m = float(min(object_size))
    key_size_m = float(np.clip(4.5 * footprint_m, 0.95, 1.60))
    key_energy_w = min(460.0, 160.0 * key_size_m**2)
    fill_energy_w = min(55.0, 0.18 * key_energy_w)
    contact_shadow_bias_m = float(np.clip(0.05 * thickness_m, 0.0005, 0.004))
    contact_shadow_distance_m = float(np.clip(2.0 * footprint_m, 0.20, 0.60))
    return {
        "rule_version": "object_scale_shadow_readability_v1",
        "object_footprint_m": round(footprint_m, 6),
        "object_thickness_m": round(thickness_m, 6),
        "key_size_m": round(key_size_m, 6),
        "key_energy_w": round(key_energy_w, 6),
        "key_energy_per_square_meter": 160.0,
        "fill_energy_w": round(fill_energy_w, 6),
        "contact_shadow_bias_m": round(contact_shadow_bias_m, 6),
        "contact_shadow_distance_m": round(contact_shadow_distance_m, 6),
        "contact_shadow_thickness": 0.05,
    }


def frozen_environment_binding(
    metadata: dict[str, Any], camera: dict[str, Any]
) -> dict[str, Any]:
    """Bind lighting around geometry whose visual/collision pose is already frozen."""

    binding = validate_environment_binding(metadata)
    scene_anchor = np.asarray(
        camera.get("diagnostics", {}).get("motion_target_m", camera["target_m"]),
        dtype=np.float64,
    )
    camera_position = np.asarray(camera["position_m"], dtype=np.float64)
    outward = camera_position[:2] - scene_anchor[:2]
    outward /= np.linalg.norm(outward)
    lateral = np.asarray([-outward[1], outward[0]])
    key_xy = scene_anchor[:2] + outward * 1.10 + lateral * 1.35
    fill_xy = scene_anchor[:2] + outward * 0.65 - lateral * 1.25
    lighting = shadow_readable_lighting(metadata)
    return {
        "profile_id": str(binding["profile_id"]),
        "environment_binding_sha256": str(binding["binding_sha256"]),
        "collision_authority": (
            "frozen_static_environment_proxy_plus_analytic_action_surface"
        ),
        "static_background_objects": copy.deepcopy(binding["visual_objects"]),
        "shadow_readability_rule": lighting,
        "key_light": {
            "type": "AREA",
            "position_m": [
                float(key_xy[0]),
                float(key_xy[1]),
                float(scene_anchor[2] + 2.5),
            ],
            "target_m": [float(value) for value in scene_anchor],
            "energy_w": lighting["key_energy_w"],
            "size_m": lighting["key_size_m"],
            "cast_shadow": True,
            "contact_shadow": True,
            "contact_shadow_bias_m": lighting["contact_shadow_bias_m"],
            "contact_shadow_distance_m": lighting["contact_shadow_distance_m"],
            "contact_shadow_thickness": lighting["contact_shadow_thickness"],
        },
        "fill_light": {
            "type": "AREA",
            "position_m": [
                float(fill_xy[0]),
                float(fill_xy[1]),
                float(scene_anchor[2] + 1.65),
            ],
            "target_m": [float(value) for value in scene_anchor],
            "energy_w": lighting["fill_energy_w"],
            "size_m": 3.0,
            "cast_shadow": False,
            "contact_shadow": False,
        },
    }


def resolve_render_request(
    metadata: dict[str, Any],
    resolution: tuple[int, int] | None,
    samples: int | None,
) -> tuple[tuple[int, int], int]:
    render_request = metadata["render_request"]
    resolved_resolution = (
        resolution
        if resolution is not None
        else tuple(int(value) for value in render_request["resolution"])
    )
    if len(resolved_resolution) != 2 or min(resolved_resolution) <= 0:
        raise ValueError("render resolution must contain two positive values")
    resolved_samples = int(
        samples if samples is not None else render_request["samples"]
    )
    if resolved_samples <= 0:
        raise ValueError("render samples must be positive")
    return (int(resolved_resolution[0]), int(resolved_resolution[1])), resolved_samples


def bind_scene(
    root: Path,
    metadata_path: Path,
    simulation_record_path: Path,
    trajectory_path: Path,
    output_root: Path,
    rules: dict[str, Any],
    resolution: tuple[int, int] | None,
    samples: int | None,
) -> dict[str, Any]:
    metadata = load_json(metadata_path)
    resolution, samples = resolve_render_request(metadata, resolution, samples)
    scene_id = str(metadata["scene_id"])
    simulation_record = load_json(simulation_record_path)
    if simulation_record.get("schema_version") not in {
        "physweep_pybullet_simulation_record_v1",
        "physweep_dispatched_simulation_record_v1",
    }:
        raise ValueError("unsupported PyBullet simulation record")
    record_scene = simulation_record.get("scene_id")
    expected_scene = scene_id
    if record_scene != expected_scene:
        raise ValueError("simulation record scene id mismatch")
    if sha256(metadata_path) != simulation_record.get("metadata_sha256"):
        raise ValueError("source metadata changed after simulation")
    if Path(simulation_record["metadata_path"]).resolve() != metadata_path.resolve():
        raise ValueError("simulation record metadata path mismatch")
    if Path(simulation_record["trajectory_path"]).resolve() != trajectory_path.resolve():
        raise ValueError("simulation record trajectory path mismatch")
    if sha256(trajectory_path) != simulation_record.get("trajectory_sha256"):
        raise ValueError("trajectory changed after simulation")
    audit_path = Path(simulation_record["audit_path"])
    if sha256(audit_path) != simulation_record.get("audit_sha256"):
        raise ValueError("trajectory audit changed after simulation")
    with np.load(trajectory_path) as source:
        trajectory = {key: source[key] for key in source.files}
    trajectory = object_trajectory_view(metadata, trajectory)
    camera = solve_camera(metadata, trajectory, rules)
    environment = frozen_environment_binding(metadata, camera)
    support_static_objects = copy.deepcopy(
        metadata["simulation"]["support"]["colliders"]
    )
    support_visual = metadata["appearance"].get(
        "support_visual",
        {
            "id": "procedural_support_proxy",
            "visual_type": "procedural_proxy",
        },
    )
    support_visual_binding = {
        "requested_profile": str(support_visual["id"]),
        "requested_type": str(support_visual["visual_type"]),
        "selected_type": "procedural_proxy",
        "fallback_reason": None,
        "selection_phase": "source_metadata_before_simulation",
    }
    support = metadata["simulation"]["support"]
    scene_composition = metadata["appearance"]["scene_visual"].get(
        "composition"
    )
    integrated_ground = (
        isinstance(scene_composition, dict)
        and str(scene_composition.get("review_status")) == "approved"
        and str(scene_composition.get("composition_mode")) == "integrated_ground"
    )
    if integrated_ground:
        scene_class = str(support["scene_class"])
        hidden_roles = (
            {"primary_support", "environment_floor"}
            if scene_class == "ground_flat"
            else {"environment_floor"}
        )
        for record in support_static_objects:
            if str(record["role"]) in hidden_roles:
                record["visible"] = False
        if str(support_visual["visual_type"]) == "mesh_support":
            raise ValueError(
                "integrated ground metadata cannot bind a second support mesh"
            )
    visual_geometry = support.get("visual_geometry")
    if (
        str(support_visual["visual_type"]) == "procedural_proxy"
        and isinstance(visual_geometry, dict)
        and str(visual_geometry.get("primitive")) == "solid_wedge"
    ):
        for record in support_static_objects:
            if bool(record.get("render_replaced_by_solid_wedge", False)):
                record["visible"] = False
        support_static_objects.append(
            {
                "id": "solid_ramp_wedge",
                "primitive": "solid_wedge",
                "role": "render_only_support",
                "material_role": "support_surface",
                "structure_material_role": "support_structure",
                "size_xy_m": [
                    float(value) for value in visual_geometry["size_xy_m"]
                ],
                "base_z_m": float(visual_geometry["base_z_m"]),
                "high_top_z_m": float(visual_geometry["high_top_z_m"]),
                "slope_axis": str(visual_geometry["slope_axis"]),
                "visible": True,
                "collision_enabled": False,
            }
        )
        support_visual_binding["selected_type"] = "procedural_solid_wedge"
    if str(support_visual["visual_type"]) == "mesh_support":
        if str(support["topology"]) != "flat_surface":
            raise ValueError(
                f"support mesh requires a flat surface: {support['semantic_type']}"
            )
        binding = support.get("exact_static_binding")
        if binding is None:
            raise ValueError(
                f"mesh support was not frozen before simulation: {expected_scene}"
            )
        if str(binding["asset_id"]) != str(support_visual["asset_id"]):
            raise ValueError(
                f"support visual and collision assets differ: {expected_scene}"
            )
        if str(support["collision_authority"]) != "exact_static_proxy":
            raise ValueError(
                f"mesh support lacks exact collision authority: {expected_scene}"
            )
        support_visual_binding["selected_type"] = "exact_static_proxy"
        support_visual_binding["binding_sha256"] = str(
            binding["binding_sha256"]
        )
        for record in support_static_objects:
            if str(record["role"]) in {
                "primary_support",
                "support_structure",
            }:
                record["visible"] = False
        support_static_objects.append(
            {
                "id": "support_visual_mesh",
                "primitive": "exact_support_visual",
                "role": "authoritative_support_visual",
                "material_role": "support_surface",
                "binding": copy.deepcopy(binding),
                "material_policy": str(support_visual["material_policy"]),
                "requires_image_texture": bool(
                    support_visual["requires_image_texture"]
                ),
                "license": str(support_visual["license"]),
                "visible": True,
                "collision_enabled": False,
                "occludes_camera": False,
            }
        )
    frame_count = int(metadata["simulation"]["time"]["frame_count"])
    inspection_frames = sorted({1, max(1, (frame_count + 1) // 2), frame_count})
    bound = copy.deepcopy(metadata)
    attach_object_identity(
        bound,
        trajectory_path=str(trajectory_path.relative_to(root)),
        mask_path=str((output_root / "masks" / scene_id).relative_to(root)),
    )
    bound["schema_version"] = "physweep_pybullet_rigid_bound_metadata_v1"
    bound["source_metadata"] = {
        "path": str(metadata_path.relative_to(root)),
        "sha256": sha256(metadata_path),
    }
    bound["trajectory"] = {
        "path": str(trajectory_path.relative_to(root)),
        "sha256": sha256(trajectory_path),
    }
    bound["simulation_record"] = {
        "path": str(simulation_record_path.relative_to(root)),
        "sha256": sha256(simulation_record_path),
    }
    bound["visualization"] = {
        "binding_version": "physweep_pybullet_visual_binding_v3",
        "support_visual_binding": support_visual_binding,
        "camera": camera,
        "materials": metadata["appearance"]["materials"],
        "hdri": metadata["appearance"]["hdri"],
        "static_objects": support_static_objects,
        "environment": environment,
        "render": {
            "engine": "BLENDER_EEVEE",
            "resolution_x": int(resolution[0]),
            "resolution_y": int(resolution[1]),
            "resolution_percentage": 100,
            "samples": int(samples),
            "fps": int(metadata["simulation"]["time"]["output_fps"]),
            "frame_start": 1,
            "frame_end": frame_count,
            "video_path": str((output_root / "videos" / f"{scene_id}.mp4").relative_to(root)),
            "inspection_frame_dir": str((output_root / "frames" / scene_id).relative_to(root)),
            "instance_mask_dir": str((output_root / "masks" / scene_id).relative_to(root)),
            "inspection_frames": inspection_frames,
            "color_management": {
                "view_transform": "Filmic",
                "look": "Medium High Contrast",
                "exposure": 0.0,
                "gamma": 1.0,
            },
        },
    }
    output_path = output_root / "metadata" / f"{scene_id}.json"
    write_json(output_path, bound)
    return {
        "scene_id": scene_id,
        "metadata_path": str(output_path.relative_to(root)),
        "metadata_sha256": sha256(output_path),
        "trajectory_path": str(trajectory_path.relative_to(root)),
        "camera_diagnostics": camera["diagnostics"],
    }


def parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must look like 640x360")
    width, height = [int(item) for item in parts]
    if min(width, height) <= 0:
        raise argparse.ArgumentTypeError("resolution must be positive")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        help="Explicit override; otherwise inherit render_request.resolution.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Explicit override; otherwise inherit render_request.samples.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def manifest_rules_path(root: Path, manifest: dict[str, Any]) -> Path:
    declared = manifest.get("rules_path")
    path = root / str(declared) if declared else ACTIVE_RULES_PATH
    path = path.resolve()
    expected = manifest.get("rules_sha256")
    if expected is not None and sha256(path) != str(expected):
        raise ValueError(f"rules hash mismatch: {path}")
    return path


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest = load_json(args.manifest.resolve())
    rules_path = manifest_rules_path(root, manifest)
    rules = load_json(rules_path)
    output_root = args.output_root.resolve()
    samples = list(manifest["samples"])
    if args.limit is not None:
        samples = samples[: args.limit]
    def sample_path(sample: dict[str, Any], key: str, fallback: Path) -> Path:
        value = sample.get(key)
        if value is None:
            return fallback
        path = Path(str(value))
        return path if path.is_absolute() else root / path

    jobs = [
        (
            root,
            (metadata_path := root / str(sample["metadata_path"])),
            sample_path(
                sample,
                "simulation_record_path",
                metadata_path.parent / "physics" / "simulation_record.json",
            ),
            sample_path(
                sample,
                "trajectory_path",
                metadata_path.parent / "physics" / "trajectory.npz",
            ),
            output_root,
            rules,
            args.resolution,
            args.samples,
        )
        for sample in samples
    ]
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.workers == 1:
        records = [bind_scene(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            records = list(executor.map(bind_scene, *zip(*jobs)))
    bound_manifest = {
        "schema_version": "physweep_pybullet_bound_manifest_v2",
        "source_manifest": str(args.manifest.resolve()),
        "output_root": str(output_root),
        "implementation": {
            "path": str(Path(__file__).resolve().relative_to(root)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "camera_rules": {
            "path": str(rules_path.relative_to(root)),
            "sha256": sha256(rules_path),
        },
        "sample_count": len(records),
        "samples": records,
    }
    write_json(output_root / "bound_manifest.json", bound_manifest)
    print(f"bound manifest: {output_root / 'bound_manifest.json'}")
    print(f"samples: {len(records)}")


if __name__ == "__main__":
    main()
