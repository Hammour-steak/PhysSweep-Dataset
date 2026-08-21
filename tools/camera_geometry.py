"""Shared camera-angle geometry for sampling and visual binding."""

from __future__ import annotations

import math


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
