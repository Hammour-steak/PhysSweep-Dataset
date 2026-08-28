#!/usr/bin/env python3
"""Shared procedural table-support geometry derived from metadata."""

from __future__ import annotations

import math
from typing import Any


def table_structure_boxes(placement: dict[str, Any]) -> list[dict[str, Any]]:
    profile = placement.get("support_structure", {})
    profile_id = profile.get("profile_id")
    if not profile_id:
        return []
    if profile_id not in {
        "four_leg_end_frame_table_v1",
        "open_frame_counter_v1",
    }:
        raise ValueError(f"unsupported visible support structure: {profile_id}")
    target_x, target_y = [float(value) for value in placement["target_size"][:2]]
    support_top = float(placement["support_top_z"])
    thickness = float(placement["thickness"])
    position = [float(value) for value in placement.get("position", [0.0, 0.0, support_top])]
    yaw = math.radians(float(placement.get("yaw_degrees", 0.0)))
    leg_height = support_top - thickness
    leg_size = float(profile.get("leg_cross_section_m", 0.11))
    inset_x = float(profile.get("leg_inset_x_m", 0.26))
    inset_y = float(profile.get("leg_inset_y_m", 0.18))
    crossbar_height = float(profile.get("end_crossbar_height_m", 0.12))
    if min(leg_height, leg_size, inset_x, inset_y, crossbar_height) <= 0.0:
        raise ValueError("table support dimensions must be positive")

    x_offsets = (-target_x / 2.0 + inset_x, target_x / 2.0 - inset_x)
    y_offsets = (-target_y / 2.0 + inset_y, target_y / 2.0 - inset_y)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

    def world(local_x: float, local_y: float, z: float) -> list[float]:
        return [
            position[0] + local_x * cos_yaw - local_y * sin_yaw,
            position[1] + local_x * sin_yaw + local_y * cos_yaw,
            z,
        ]

    boxes: list[dict[str, Any]] = []
    for x_index, local_x in enumerate(x_offsets):
        for y_index, local_y in enumerate(y_offsets):
            boxes.append(
                {
                    "id": f"leg_{x_index}_{y_index}",
                    "role": "table_leg",
                    "center_m": world(local_x, local_y, leg_height / 2.0),
                    "dimensions_m": [leg_size, leg_size, leg_height],
                    "yaw_degrees": math.degrees(yaw),
                }
            )
        boxes.append(
            {
                "id": f"end_crossbar_{x_index}",
                "role": "table_end_crossbar",
                "center_m": world(
                    local_x,
                    0.0,
                    support_top - thickness - crossbar_height / 2.0,
                ),
                "dimensions_m": [leg_size, target_y - 2.0 * inset_y + leg_size, crossbar_height],
                "yaw_degrees": math.degrees(yaw),
            }
        )
    return boxes
