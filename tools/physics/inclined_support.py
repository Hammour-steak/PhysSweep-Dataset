#!/usr/bin/env python3
"""Shared finite-inclined-plane geometry derived from support metadata."""

from __future__ import annotations

import math
from typing import Any


def inclined_plane_geometry(placement: dict[str, Any]) -> dict[str, Any]:
    if str(placement.get("support_shape")) != "inclined_ramp":
        raise ValueError("inclined support geometry requires support_shape=inclined_ramp")
    if str(placement.get("slope_axis", "y")) != "y":
        raise ValueError("inclined support v1 supports only slope_axis=y")
    if abs(float(placement.get("yaw_degrees", 0.0))) > 1.0e-8:
        raise ValueError("inclined support v1 requires zero yaw")

    target = [float(value) for value in placement["target_size"]]
    if len(target) < 2 or min(target[:2]) <= 0.0:
        raise ValueError("inclined support target_size x/y must be positive")
    rise = float(placement["slope_rise_m"])
    thickness = float(placement.get("thickness", 0.08))
    if rise <= 0.0 or thickness <= 0.0:
        raise ValueError("inclined support rise and thickness must be positive")
    center = [
        float(value)
        for value in placement.get(
            "position", [0.0, 0.0, float(placement["support_top_z"])]
        )
    ]
    center[2] = float(placement["support_top_z"])
    angle = math.atan2(rise, target[1])
    cosine = math.cos(angle)
    sine = math.sin(angle)
    tangent_u = [1.0, 0.0, 0.0]
    tangent_v = [0.0, cosine, sine]
    normal = [0.0, -sine, cosine]

    safe = placement["safe_surface_bounds"]
    bounds_u = [float(safe["x"][0]) - center[0], float(safe["x"][1]) - center[0]]
    bounds_v = [
        (float(safe["y"][0]) - center[1]) / cosine,
        (float(safe["y"][1]) - center[1]) / cosine,
    ]
    corners = []
    for u in bounds_u:
        for v in bounds_v:
            top = [
                center[axis] + u * tangent_u[axis] + v * tangent_v[axis]
                for axis in range(3)
            ]
            corners.append(top)
            corners.append([top[axis] - thickness * normal[axis] for axis in range(3)])
    aabb_min = [min(point[axis] for point in corners) for axis in range(3)]
    aabb_max = [max(point[axis] for point in corners) for axis in range(3)]
    return {
        "geometry_version": "finite_inclined_plane_y_v1",
        "point_m": center,
        "normal": normal,
        "tangent_u": tangent_u,
        "tangent_v": tangent_v,
        "bounds_u_m": bounds_u,
        "bounds_v_m": bounds_v,
        "thickness_m": thickness,
        "slope_angle_degrees": math.degrees(angle),
        "low_top_z_m": center[2] - rise / 2.0,
        "high_top_z_m": center[2] + rise / 2.0,
        "world_aabb_m": {"min": aabb_min, "max": aabb_max},
    }
