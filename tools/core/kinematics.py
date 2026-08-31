"""Shared kinematic bounds independent of a simulation backend."""

from __future__ import annotations

import math

import numpy as np


def energy_consistent_linear_speed_limit(
    configured_limit_m_s: float,
    initial_velocity_m_s: np.ndarray,
    positions_m: np.ndarray,
    gravity_m_s2: np.ndarray,
) -> float:
    """Keep the configured bound while allowing speed explained by gravity."""

    configured = float(configured_limit_m_s)
    initial_velocity = np.asarray(initial_velocity_m_s, dtype=np.float64)
    positions = np.asarray(positions_m, dtype=np.float64)
    gravity = np.asarray(gravity_m_s2, dtype=np.float64)
    if (
        not math.isfinite(configured)
        or configured <= 0.0
        or initial_velocity.shape != (3,)
        or positions.ndim != 2
        or positions.shape[1] != 3
        or not len(positions)
        or gravity.shape != (3,)
        or not np.all(np.isfinite(initial_velocity))
        or not np.all(np.isfinite(positions))
        or not np.all(np.isfinite(gravity))
    ):
        raise ValueError("linear-speed bound inputs are invalid")
    gravitational_energy_gain = max(
        0.0,
        float(np.max((positions - positions[0]) @ gravity)),
    )
    energy_speed = math.sqrt(
        float(initial_velocity @ initial_velocity)
        + 2.0 * gravitational_energy_gain
    )
    return max(configured, 1.15 * energy_speed)
