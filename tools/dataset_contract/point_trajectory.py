#!/usr/bin/env python3
"""Dense identity-preserving point trajectories for PhysSweep.

The simulation trajectory remains the source of truth for object poses.  This
module materializes fixed, identity-preserving surface points from those poses
and projects them into the solved camera.  The object axis is explicit so the
same file format works for one, two, and three dynamic objects.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


POINT_TRAJECTORY_SCHEMA = "physweep.point_trajectories.v1"
POINT_COUNT = 2048
MAX_OBJECTS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_quaternion_wxyz(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm must be positive")
    return quaternion / norm


def quaternion_wxyz_to_matrix(value: np.ndarray) -> np.ndarray:
    """Return a rotation matrix mapping body-frame column vectors to world."""
    w, x, y, z = _normalize_quaternion_wxyz(value)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if transform.shape != (4, 4):
        raise ValueError("transform must have shape [4, 4]")
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    return (homogeneous @ transform.T)[:, :3]


def camera_from_world_from_spec(camera: dict[str, Any]) -> np.ndarray:
    position = np.asarray(camera["position_m"], dtype=np.float64)
    target = np.asarray(camera["target_m"], dtype=np.float64)
    if position.shape != (3,) or target.shape != (3,):
        raise ValueError("camera position and target must have shape [3]")
    forward = target - position
    forward /= max(float(np.linalg.norm(forward)), 1.0e-12)
    world_up = np.asarray(camera.get("world_up", [0.0, 0.0, 1.0]), dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= max(float(np.linalg.norm(right)), 1.0e-12)
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1.0e-12)
    rotation = np.stack([right, up, forward], axis=0)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ position
    return transform


def camera_intrinsics_from_spec(
    camera: dict[str, Any], image_size_px: tuple[int, int]
) -> np.ndarray:
    width, height = [int(value) for value in image_size_px]
    focal = float(camera["focal_length_mm"])
    sensor_width = float(camera["sensor_width_mm"])
    focal_px = focal / sensor_width * width
    return np.asarray(
        [
            [focal_px, 0.0, width * 0.5],
            [0.0, focal_px, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def project_camera_points(
    points_camera_m: np.ndarray,
    intrinsics: np.ndarray,
    image_size_px: tuple[int, int],
    clip_start_m: float = 0.03,
    clip_end_m: float = 100.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points using PhysSweep's right/up/forward convention."""
    points = np.asarray(points_camera_m, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("camera points must have shape [T, N, 3]")
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must have shape [3, 3]")
    width, height = [int(value) for value in image_size_px]
    z = points[..., 2]
    safe_z = np.maximum(z, 1.0e-8)
    u = intrinsics[0, 0] * points[..., 0] / safe_z + intrinsics[0, 2]
    v = intrinsics[1, 2] - intrinsics[1, 1] * points[..., 1] / safe_z
    tracks = np.stack([u, v], axis=-1)
    valid = (
        np.isfinite(tracks).all(axis=-1)
        & np.isfinite(z)
        & (z > float(clip_start_m))
        & (z < float(clip_end_m))
        & (u >= 0.0)
        & (u < float(width))
        & (v >= 0.0)
        & (v < float(height))
    )
    return tracks.astype(np.float32), z.astype(np.float32), valid.astype(np.uint8)


def rigid_points_from_poses(
    initial_points_camera_m: np.ndarray,
    positions_world_m: np.ndarray,
    quaternions_wxyz: np.ndarray,
    camera_from_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Materialize fixed body points from a rigid PyBullet pose trajectory."""
    initial_points = np.asarray(initial_points_camera_m, dtype=np.float64)
    positions = np.asarray(positions_world_m, dtype=np.float64)
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64)
    camera_from_world = np.asarray(camera_from_world, dtype=np.float64)
    if initial_points.ndim != 2 or initial_points.shape[1] != 3:
        raise ValueError("initial points must have shape [N, 3]")
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [T, 3]")
    if quaternions.shape != (len(positions), 4):
        raise ValueError("quaternions must have shape [T, 4]")
    if camera_from_world.shape != (4, 4):
        raise ValueError("camera_from_world must have shape [4, 4]")
    world_from_camera = np.linalg.inv(camera_from_world)
    initial_points_world = transform_points(initial_points, world_from_camera)
    initial_rotation = quaternion_wxyz_to_matrix(quaternions[0])
    local_points = (initial_points_world - positions[0][None, :]) @ initial_rotation
    rotations = np.stack(
        [quaternion_wxyz_to_matrix(quaternion) for quaternion in quaternions], axis=0
    )
    points_world = np.einsum("nj,tij->tni", local_points, rotations)
    points_world += positions[:, None, :]
    points_camera = np.einsum(
        "tni,ji->tnj", points_world, camera_from_world[:3, :3]
    )
    points_camera += camera_from_world[:3, 3][None, None, :]
    initial_alignment_error = float(
        np.max(np.linalg.norm(points_camera[0] - initial_points, axis=-1))
    )
    if not np.isfinite(initial_alignment_error):
        raise ValueError("initial point alignment is non-finite")
    return points_world.astype(np.float32), points_camera.astype(np.float32), initial_alignment_error


def _scalar_string(value: np.ndarray | str) -> str:
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.ndim == 0:
        return str(array.item())
    if len(array) != 1:
        raise ValueError("expected a scalar string")
    return str(array[0])


def load_scene_object_points(path: Path) -> dict[str, np.ndarray]:
    """Load one or more fixed 2048-point object blocks from a scene condition."""
    with np.load(path, allow_pickle=False) as archive:
        if "object_xyz_camera_m" not in archive.files:
            raise ValueError(f"scene condition has no object points: {path}")
        points = np.asarray(archive["object_xyz_camera_m"], dtype=np.float32)
        metadata = (
            json.loads(_scalar_string(archive["metadata_json"]))
            if "metadata_json" in archive.files
            else {}
        )
        if points.ndim == 2:
            points = points[None, ...]
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("object_xyz_camera_m must have shape [O, N, 3]")
        ids = None
        for key in ("object_ids", "object_entity_ids", "controlled_object_ids"):
            if key in archive.files:
                ids = [_scalar_string(value) for value in np.asarray(archive[key]).reshape(-1)]
                break
        if ids is None:
            metadata_ids = metadata.get("object_ids") or metadata.get("object_entity_ids")
            if metadata_ids:
                ids = [str(value) for value in metadata_ids]
        if ids is None:
            ids = [str(metadata.get("controlled_object_id", "object_a"))]
        if len(ids) != len(points):
            raise ValueError(
                f"scene object id count does not match point blocks: {len(ids)} != {len(points)}"
            )
        if len(set(ids)) != len(ids):
            raise ValueError("scene object ids must be unique")
        for object_id, block in zip(ids, points):
            if block.shape[0] != POINT_COUNT:
                raise ValueError(
                    f"{object_id} has {block.shape[0]} points; expected {POINT_COUNT}"
                )
            if not np.isfinite(block).all():
                raise ValueError(f"{object_id} contains non-finite points")
        return {object_id: block for object_id, block in zip(ids, points)}


def load_scene_camera(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Load camera geometry from the exact GT surface referenced by a scene condition."""
    with np.load(path, allow_pickle=False) as archive:
        if {"camera_from_world", "camera_intrinsics", "image_size_px"}.issubset(
            archive.files
        ):
            return (
                np.asarray(archive["camera_from_world"], dtype=np.float64),
                np.asarray(archive["camera_intrinsics"], dtype=np.float64),
                tuple(int(value) for value in archive["image_size_px"]),
            )
        if "metadata_json" not in archive.files:
            raise ValueError(f"scene condition has no camera source: {path}")
        metadata = json.loads(_scalar_string(archive["metadata_json"]))
    source_surface = metadata.get("source_surface")
    if not source_surface:
        raise ValueError(f"scene condition does not identify a camera source: {path}")
    source_path = Path(str(source_surface))
    if not source_path.is_file():
        raise FileNotFoundError(f"camera source is missing: {source_surface}")
    with np.load(source_path, allow_pickle=False) as archive:
        return (
            np.asarray(archive["camera_from_world"], dtype=np.float64),
            np.asarray(archive["camera_intrinsics"], dtype=np.float64),
            tuple(int(value) for value in archive["image_size_px"]),
        )


def trajectory_object_ids(trajectory: dict[str, np.ndarray]) -> list[str]:
    suffix = "__position_m"
    ids = [key[: -len(suffix)] for key in trajectory if key.endswith(suffix)]
    if not ids:
        raise ValueError("trajectory contains no object position channels")
    return sorted(ids)


def build_point_trajectory(
    scene_path: Path,
    trajectory_path: Path,
    *,
    clip_start_m: float = 0.03,
    clip_end_m: float = 100.0,
) -> dict[str, np.ndarray]:
    """Build a multi-object identity-preserving point trajectory cache."""
    object_points = load_scene_object_points(scene_path)
    camera_from_world, intrinsics, image_size_px = load_scene_camera(scene_path)
    with np.load(trajectory_path, allow_pickle=False) as archive:
        trajectory = {key: archive[key] for key in archive.files}
    object_ids = trajectory_object_ids(trajectory)
    if set(object_ids) != set(object_points):
        raise ValueError(
            "scene and trajectory object ids differ: "
            f"scene={sorted(object_points)} trajectory={object_ids}"
        )
    time_s = np.asarray(trajectory["time_s"], dtype=np.float64)
    if time_s.ndim != 1 or len(time_s) < 2 or not np.isfinite(time_s).all():
        raise ValueError("trajectory time_s must be a finite [T] array")
    ordered_ids = [object_id for object_id in object_points if object_id in object_ids]
    world_blocks = []
    camera_blocks = []
    alignment_errors = []
    for object_id in ordered_ids:
        positions = np.asarray(trajectory[f"{object_id}__position_m"], dtype=np.float64)
        quaternions = np.asarray(
            trajectory[f"{object_id}__quaternion_wxyz"], dtype=np.float64
        )
        if len(positions) != len(time_s):
            raise ValueError(f"{object_id} trajectory length differs from time_s")
        world, camera, alignment_error = rigid_points_from_poses(
            object_points[object_id], positions, quaternions, camera_from_world
        )
        world_blocks.append(world)
        camera_blocks.append(camera)
        alignment_errors.append(alignment_error)
    points_world = np.stack(world_blocks, axis=1)
    points_camera = np.stack(camera_blocks, axis=1)
    tracks, depth, valid = project_camera_points(
        points_camera.reshape(len(time_s), -1, 3),
        intrinsics,
        image_size_px,
        clip_start_m=clip_start_m,
        clip_end_m=clip_end_m,
    )
    tracks = tracks.reshape(len(time_s), len(ordered_ids), POINT_COUNT, 2)
    depth = depth.reshape(len(time_s), len(ordered_ids), POINT_COUNT)
    valid = valid.reshape(len(time_s), len(ordered_ids), POINT_COUNT)
    payload = {
        "time_s": time_s.astype(np.float32),
        "object_ids": np.asarray(ordered_ids),
        "points_world_m": points_world.astype(np.float32),
        "points_camera_m": points_camera.astype(np.float32),
        "tracks_xy_px": tracks.astype(np.float32),
        "depth_m": depth.astype(np.float32),
        "valid": valid.astype(np.uint8),
        "initial_points_camera_m": points_camera[0].astype(np.float32),
        "camera_from_world": camera_from_world.astype(np.float32),
        "camera_intrinsics": intrinsics.astype(np.float32),
        "image_size_px": np.asarray(image_size_px, dtype=np.int32),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "schema": POINT_TRAJECTORY_SCHEMA,
                    "point_count": POINT_COUNT,
                    "object_count": len(ordered_ids),
                    "object_ids": ordered_ids,
                    "coordinate_frame_world": "pybullet_world_xyz",
                    "coordinate_frame_camera": "camera_right_up_forward",
                    "track_definition": "perspective_projection_of_fixed_material_points",
                    "visibility_definition": "in_frame_and_clip_validity;_not_a_z_buffer",
                    "initial_alignment_error_m": alignment_errors,
                    "clip_start_m": float(clip_start_m),
                    "clip_end_m": float(clip_end_m),
                },
                sort_keys=True,
            )
        ),
    }
    validate_point_trajectory(payload)
    return payload


def validate_point_trajectory(payload: dict[str, np.ndarray]) -> None:
    required = {
        "time_s",
        "object_ids",
        "points_world_m",
        "points_camera_m",
        "tracks_xy_px",
        "depth_m",
        "valid",
        "initial_points_camera_m",
        "camera_from_world",
        "camera_intrinsics",
        "image_size_px",
        "metadata_json",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"point trajectory is missing fields: {', '.join(missing)}")
    points_world = np.asarray(payload["points_world_m"])
    points_camera = np.asarray(payload["points_camera_m"])
    tracks = np.asarray(payload["tracks_xy_px"])
    depth = np.asarray(payload["depth_m"])
    valid = np.asarray(payload["valid"])
    if points_world.ndim != 4 or points_world.shape[-1] != 3:
        raise ValueError("points_world_m must have shape [T, O, 2048, 3]")
    if points_world.shape[2] != POINT_COUNT:
        raise ValueError("point trajectory must contain exactly 2048 points per object")
    if points_camera.shape != points_world.shape:
        raise ValueError("world and camera point trajectories must have equal shapes")
    expected_tracks = points_world.shape[:3] + (2,)
    if tracks.shape != expected_tracks:
        raise ValueError(f"tracks_xy_px must have shape {expected_tracks}")
    if depth.shape != points_world.shape[:3] or valid.shape != points_world.shape[:3]:
        raise ValueError("depth and valid arrays must match [T, O, 2048]")
    object_ids = np.asarray(payload["object_ids"]).reshape(-1)
    if len(object_ids) != points_world.shape[1] or len(set(map(str, object_ids))) != len(object_ids):
        raise ValueError("object_ids must be unique and match the object axis")
    if len(object_ids) > MAX_OBJECTS:
        raise ValueError(f"at most {MAX_OBJECTS} dynamic objects are supported")
    for name in ("points_world_m", "points_camera_m", "tracks_xy_px", "depth_m"):
        if not np.isfinite(payload[name]).all():
            raise ValueError(f"{name} contains non-finite values")
    if not np.isfinite(payload["time_s"]).all():
        raise ValueError("time_s contains non-finite values")


def save_point_trajectory(path: Path, payload: dict[str, np.ndarray]) -> None:
    validate_point_trajectory(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
