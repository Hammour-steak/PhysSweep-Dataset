#!/usr/bin/env python3
"""Shared geometry-aware PyBullet time-step selection."""

from __future__ import annotations

import math
from typing import Any


def simulation_hz_for_min_extent(engine: dict[str, Any], minimum_extent_m: float) -> int:
    rule = engine.get("adaptive_time_step", {})
    output_fps = int(engine["output_fps"])
    minimum_hz = int(engine["minimum_simulation_hz"])
    if not rule.get("enabled", False):
        return minimum_hz
    maximum_hz = int(rule["maximum_hz"])
    if minimum_hz > maximum_hz:
        raise ValueError("adaptive simulation minimum exceeds maximum")
    if minimum_extent_m <= 0.0:
        raise ValueError("minimum collision extent must be positive")
    travel_limit = (
        minimum_extent_m
        * float(rule["maximum_travel_fraction_of_min_extent_per_step"])
    )
    requested = float(rule["reference_speed_m_s"]) / max(travel_limit, 1.0e-8)
    selected = max(minimum_hz, min(maximum_hz, math.ceil(requested)))
    if rule.get("round_to_output_fps_multiple", True):
        selected = math.ceil(selected / output_fps) * output_fps
        selected = min(maximum_hz, selected)
    if selected % output_fps != 0:
        raise ValueError("adaptive simulation hz must be divisible by output fps")
    return selected


def simulation_hz_for_geometry(
    engine: dict[str, Any], size_m: list[float]
) -> int:
    return simulation_hz_for_min_extent(
        engine, min(float(value) for value in size_m)
    )
