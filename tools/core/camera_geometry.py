"""Shared camera-angle geometry for sampling and visual binding."""

from __future__ import annotations

import math
from collections.abc import Sequence


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
