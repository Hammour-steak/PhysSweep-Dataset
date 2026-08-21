"""Pure calculations shared by grouped motion rules."""

from __future__ import annotations

import math

import numpy as np


def distance_lower_bound(
    threshold: float, absolute_tolerance: float, relative_tolerance: float
) -> float:
    tolerance = max(absolute_tolerance, abs(threshold) * relative_tolerance)
    return max(0.0, threshold - tolerance)


def distance_upper_bound(
    threshold: float, absolute_tolerance: float, relative_tolerance: float
) -> float:
    tolerance = max(absolute_tolerance, abs(threshold) * relative_tolerance)
    return threshold + tolerance


def precontact_lateral_drift(
    positions: np.ndarray, first_contact_index: int | None
) -> float:
    index = (
        len(positions) - 1
        if first_contact_index is None
        else max(0, first_contact_index - 1)
    )
    return float(np.linalg.norm(positions[index, :2] - positions[0, :2]))


def sampled_extremum_tolerance(
    gravity_m_s2: float,
    output_fps: float,
    interval_error_multiplier: float,
) -> float:
    if output_fps <= 0.0:
        raise ValueError("output fps must be positive")
    return interval_error_multiplier * abs(gravity_m_s2) / (
        8.0 * output_fps**2
    )


def projected_displacement(
    positions: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    direction_xy = np.asarray(direction[:2], dtype=np.float64)
    direction_xy /= max(float(np.linalg.norm(direction_xy)), 1.0e-12)
    return (positions[:, :2] - positions[0, :2]) @ direction_xy


def entry_speed_after_coulomb_travel(
    target_speed_m_s: float,
    friction: float,
    gravity_m_s2: float,
    distance_m: float,
) -> float:
    return math.sqrt(
        target_speed_m_s**2
        + 2.0 * friction * gravity_m_s2 * distance_m
    )


def coast_speed_for_distance(
    friction: float, gravity_m_s2: float, distance_m: float
) -> float:
    return math.sqrt(2.0 * friction * gravity_m_s2 * distance_m)


def climb_speed_for_distance(
    deceleration_m_s2: float, distance_m: float
) -> float:
    return math.sqrt(max(0.01, 2.0 * deceleration_m_s2 * distance_m))
