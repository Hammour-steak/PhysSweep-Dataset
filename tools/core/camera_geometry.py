"""Shared camera-angle geometry for sampling and visual binding."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from tools.core.rigid_geometry import (
    object_contact_offset_m,
    quaternion_matrix_wxyz,
    upright_pair_center_distance_m,
)


BASE_AZIMUTH_OFFSETS_DEGREES = (0.0, -12.0, 12.0, -24.0, 24.0)
WIDE_AZIMUTH_OFFSETS_DEGREES = (
    -48.0,
    48.0,
    -72.0,
    72.0,
    -96.0,
    96.0,
)
BLOCKER_AZIMUTH_OFFSETS_DEGREES = (
    -48.0,
    48.0,
    -72.0,
    72.0,
    -96.0,
    96.0,
    -120.0,
    120.0,
    -144.0,
    144.0,
    168.0,
)


def camera_azimuth_offsets(
    *,
    maximum_deviation_degrees: float | None = None,
    wide: bool = False,
    has_camera_blockers: bool = False,
) -> list[float]:
    offsets = list(BASE_AZIMUTH_OFFSETS_DEGREES)
    if wide:
        offsets.extend(WIDE_AZIMUTH_OFFSETS_DEGREES)
    if has_camera_blockers:
        offsets.extend(BLOCKER_AZIMUTH_OFFSETS_DEGREES)
    offsets = list(dict.fromkeys(offsets))
    if maximum_deviation_degrees is not None:
        maximum = float(maximum_deviation_degrees)
        offsets = [value for value in offsets if abs(value) <= maximum]
    return offsets


def inclined_surface_side_readability(azimuth_degrees: float) -> float:
    """Return side-view strength for ramps whose uphill tangent is local +Y."""

    return abs(math.cos(math.radians(float(azimuth_degrees))))


def pair_approach_axis_xy(
    approach_axis_xyz: Sequence[float],
) -> tuple[float, float]:
    """Project one finite pair-approach axis onto the horizontal plane."""

    if len(approach_axis_xyz) != 3:
        raise ValueError("pair approach axis must contain three components")
    x, y, z = [float(value) for value in approach_axis_xyz]
    norm = math.hypot(x, y)
    if (
        not all(math.isfinite(value) for value in (x, y, z))
        or norm <= 1.0e-8
    ):
        raise ValueError("pair approach axis needs a finite horizontal projection")
    return x / norm, y / norm


def deterministic_pair_side_azimuths(
    scene_id: str, approach_axis_xyz: Sequence[float]
) -> tuple[float, float]:
    """Return both side-on pair views in deterministic preference order."""

    x, y = pair_approach_axis_xy(approach_axis_xyz)
    approach_degrees = math.degrees(math.atan2(y, x))
    digest = hashlib.sha256(
        f"joint-camera-side:{scene_id}".encode("utf-8")
    ).digest()
    preferred_side = -1.0 if digest[0] % 2 else 1.0
    preferred = approach_degrees + preferred_side * 90.0
    return preferred, preferred + 180.0


def pair_view_azimuth_degrees(
    approach_axis_xyz: Sequence[float], relative_azimuth_degrees: float
) -> float:
    """Resolve a declared pair-relative view into one world-space azimuth."""

    x, y = pair_approach_axis_xy(approach_axis_xyz)
    relative = float(relative_azimuth_degrees)
    if not math.isfinite(relative) or not -180.0 <= relative <= 180.0:
        raise ValueError("pair-relative camera azimuth must lie in [-180, 180]")
    return math.degrees(math.atan2(y, x)) + relative


def pair_view_azimuth_candidates(
    approach_axis_xyz: Sequence[float],
    relative_azimuth_degrees: float,
    maximum_deviation_degrees: float,
) -> tuple[float, ...]:
    """Return deterministic candidates inside one pair-relative view family."""

    requested = pair_view_azimuth_degrees(
        approach_axis_xyz, relative_azimuth_degrees
    )
    relative = float(relative_azimuth_degrees)
    maximum = float(maximum_deviation_degrees)
    if not math.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("pair camera deviation must be finite and positive")
    side_target = 90.0 if relative >= 0.0 else -90.0
    toward_side = side_target - relative
    if abs(toward_side) <= 1.0e-8:
        return (requested,)
    serialization_margin = min(1.0e-3, 0.01 * maximum)
    extent = min(maximum - serialization_margin, abs(toward_side))
    direction = math.copysign(1.0, toward_side)
    offsets = (
        0.0,
        direction * extent,
        direction * 0.5 * extent,
        -direction * 0.5 * extent,
        -direction * extent,
    )
    return tuple(requested + value for value in dict.fromkeys(offsets))


def pair_view_elevation_candidates(
    minimum_degrees: float,
    preferred_degrees: float,
    maximum_degrees: float,
) -> tuple[float, ...]:
    """Return the preferred pair elevation and deterministic interior options."""

    minimum, preferred, maximum = map(
        float, (minimum_degrees, preferred_degrees, maximum_degrees)
    )
    if (
        not all(math.isfinite(value) for value in (minimum, preferred, maximum))
        or not minimum <= preferred <= maximum
        or minimum == maximum
    ):
        raise ValueError("pair camera elevation interval is invalid")
    candidates = (
        preferred,
        0.5 * (preferred + maximum),
        0.5 * (preferred + minimum),
        0.25 * preferred + 0.75 * maximum,
        0.25 * preferred + 0.75 * minimum,
    )
    return tuple(dict.fromkeys(candidates))


def _primitive_camera_model(
    obj: dict[str, Any],
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    geometry = obj["geometry"]
    shape = str(geometry["type"])
    size = np.asarray(geometry["size_m"], dtype=np.float64)
    rotation = np.asarray(
        quaternion_matrix_wxyz(
            obj["initial_state"]["orientation_quaternion_wxyz"]
        ),
        dtype=np.float64,
    )
    if shape == "sphere":
        half_extents = np.repeat(0.5 * size[0], 3)
    elif shape == "cuboid":
        half_extents = np.abs(rotation) @ (0.5 * size)
    elif shape == "cylinder":
        axis = rotation[:, 2]
        radial = 0.5 * size[0] * np.sqrt(
            np.maximum(0.0, 1.0 - axis**2)
        )
        half_extents = radial + 0.5 * size[2] * np.abs(axis)
    else:
        raise ValueError("camera eligibility received an unknown primitive shape")
    return shape, size, rotation, half_extents


def _primitive_radii_along_world_directions(
    model: tuple[str, np.ndarray, np.ndarray, np.ndarray],
    directions: np.ndarray,
) -> np.ndarray:
    shape, size, rotation, _ = model
    local = np.asarray(directions, dtype=np.float64) @ rotation
    if shape == "sphere":
        return 0.5 * float(size[0]) * np.linalg.norm(local, axis=1)
    if shape == "cuboid":
        return np.abs(local) @ (0.5 * size)
    return (
        0.5 * size[0] * np.linalg.norm(local[:, :2], axis=1)
        + 0.5 * size[2] * np.abs(local[:, 2])
    )


def pair_camera_geometry_eligible(
    scene: dict[str, Any], minimum_object_extent_m: float
) -> bool:
    """Check whether a supported pair can meet its camera geometry."""

    minimum_extent = float(minimum_object_extent_m)
    if not math.isfinite(minimum_extent) or minimum_extent < 0.0:
        raise ValueError("pair camera minimum object extent cannot be negative")
    objects = scene["simulation"]["objects"]
    interaction = scene["simulation"]["interaction"]
    support = scene["simulation"]["support"]
    if len(objects) != 2 or support["support_shape"] not in {
        "rectangular_slab",
        "inclined_ramp",
    }:
        raise ValueError("pair camera geometry requires two supported objects")

    approach = np.asarray(
        interaction["approach_axis_xyz"], dtype=np.float64
    )
    approach_norm = float(np.linalg.norm(approach))
    if (
        approach.shape != (3,)
        or not np.all(np.isfinite(approach))
        or approach_norm <= 1.0e-12
    ):
        raise ValueError("pair camera approach axis must be finite and nonzero")
    approach /= approach_norm
    frame = support["surface_frame"]
    if support["support_shape"] == "inclined_ramp":
        longitudinal = np.asarray(
            frame["tangent_uphill"], dtype=np.float64
        )
        lateral = -np.asarray(frame["tangent_cross"], dtype=np.float64)
    else:
        longitudinal = np.asarray(
            frame["tangent_cross"], dtype=np.float64
        )
        lateral = np.asarray(frame["tangent_uphill"], dtype=np.float64)
    normal = np.asarray(frame["normal"], dtype=np.float64)
    local_direction = [
        float(approach @ longitudinal),
        float(approach @ lateral),
    ]
    shapes = [str(obj["geometry"]["type"]) for obj in objects]
    sizes = [list(map(float, obj["geometry"]["size_m"])) for obj in objects]
    primitive_models = [_primitive_camera_model(obj) for obj in objects]
    planar_contact_distance = upright_pair_center_distance_m(
        shapes[0], sizes[0], shapes[1], sizes[1], local_direction
    )
    support_offsets = [
        object_contact_offset_m(shape, size)
        for shape, size in zip(shapes, sizes, strict=True)
    ]
    contact_delta = (
        approach * planar_contact_distance
        + normal * (support_offsets[1] - support_offsets[0])
    )
    approach_xy = np.asarray(
        pair_approach_axis_xy(interaction["approach_axis_xyz"]),
        dtype=np.float64,
    )
    minimum_pair_ratio = float(
        interaction[
            "minimum_pair_keyframe_projected_center_separation_to_radius_sum_ratio"
        ]
    )
    aspect = 16.0 / 9.0

    azimuths = pair_view_azimuth_candidates(
        interaction["approach_axis_xyz"],
        interaction["camera_relative_azimuth_degrees"],
        interaction["maximum_camera_view_azimuth_deviation_degrees"],
    )
    elevations = pair_view_elevation_candidates(
        interaction["minimum_camera_elevation_degrees"],
        interaction["preferred_camera_elevation_degrees"],
        interaction["maximum_camera_elevation_degrees"],
    )
    azimuth_grid = np.repeat(np.radians(azimuths), len(elevations))
    elevation_grid = np.tile(np.radians(elevations), len(azimuths))
    cos_elevation = np.cos(elevation_grid)
    outward = np.column_stack(
        [
            cos_elevation * np.cos(azimuth_grid),
            cos_elevation * np.sin(azimuth_grid),
            np.sin(elevation_grid),
        ]
    )
    forward = -outward
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right, axis=1)[:, None]
    up = np.cross(right, forward)

    eligible = np.ones(len(right), dtype=bool)
    for model in primitive_models:
        projected_extent = 2.0 * np.maximum(
            _primitive_radii_along_world_directions(model, right),
            _primitive_radii_along_world_directions(model, up),
        )
        if minimum_extent > 0.0:
            eligible &= projected_extent + 1.0e-12 >= minimum_extent

    projected_delta = np.column_stack(
        [right @ contact_delta, aspect * (up @ contact_delta)]
    )
    projected_separation = np.linalg.norm(projected_delta, axis=1)
    eligible &= projected_separation > 1.0e-12
    separation_direction = projected_delta / np.maximum(
        projected_separation[:, None], 1.0e-12
    )
    radius_sum = np.zeros(len(right), dtype=np.float64)
    for model, shape in zip(primitive_models, shapes, strict=True):
        if shape == "sphere":
            world_directions = (
                separation_direction[:, :1] * right
                + aspect * separation_direction[:, 1:] * up
            )
            radii = _primitive_radii_along_world_directions(
                model, world_directions
            )
        else:
            half_xy = model[3][:2]
            footprint_radius = float(np.abs(approach_xy) @ half_xy)
            footprint_vector = np.asarray(
                [
                    footprint_radius * approach_xy[0],
                    footprint_radius * approach_xy[1],
                    0.0,
                ]
            )
            projected_footprint = np.column_stack(
                [
                    right @ footprint_vector,
                    aspect * (up @ footprint_vector),
                ]
            )
            radii = np.abs(
                np.sum(projected_footprint * separation_direction, axis=1)
            )
        radius_sum += radii
    ratio = projected_separation / np.maximum(radius_sum, 1.0e-12)
    return bool(np.any(eligible & (ratio + 1.0e-12 >= minimum_pair_ratio)))


def camera_corridor_admits_inclined_surface(
    *,
    base_azimuth_degrees: float,
    maximum_deviation_degrees: float,
    minimum_side_readability: float,
) -> bool:
    return any(
        inclined_surface_side_readability(base_azimuth_degrees + offset)
        >= float(minimum_side_readability)
        for offset in camera_azimuth_offsets(
            maximum_deviation_degrees=maximum_deviation_degrees
        )
    )


def seeded_view_order(
    values: Sequence[float], digest: bytes, byte_index: int
) -> list[float]:
    """Rotate an allowed view pool reproducibly without changing membership."""

    ordered = [float(value) for value in values]
    if not ordered:
        raise ValueError("camera view pool must not be empty")
    start = int(digest[byte_index]) % len(ordered)
    return ordered[start:] + ordered[:start]


def blocker_safe_seeded_view_order(
    values: Sequence[float],
    digest: bytes,
    byte_index: int,
    blocker_offset_xy: Sequence[float],
) -> list[float]:
    """Prefer a blocker-safe half of the pool while preserving seeded diversity."""

    ordered = seeded_view_order(values, digest, byte_index)
    offset_x, offset_y = [float(value) for value in blocker_offset_xy]
    ranked = sorted(
        ordered,
        key=lambda degrees: (
            offset_x * math.cos(math.radians(degrees))
            + offset_y * math.sin(math.radians(degrees))
        ),
    )
    safe_count = max(2, (len(ranked) + 1) // 2)
    safe = set(ranked[:safe_count])
    return [value for value in ordered if value in safe] + [
        value for value in ordered if value not in safe
    ]
