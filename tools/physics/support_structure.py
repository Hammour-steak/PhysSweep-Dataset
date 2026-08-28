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


def support_context_requirement(
    placement: dict[str, Any],
    camera_facing_direction_xy: tuple[float, float] | list[float],
    *,
    source_semantic_class: str | None = None,
) -> dict[str, Any]:
    """Return camera-visible support evidence without inventing structure.

    A declared table/counter frame needs visible legs. A finite slab without a
    frame instead needs its front support edge in view, which works for both a
    floor mat and an integral countertop without pretending either has legs.
    """

    direction_x, direction_y = [float(value) for value in camera_facing_direction_xy]
    direction_norm = math.hypot(direction_x, direction_y)
    if direction_norm <= 1.0e-8:
        raise ValueError("camera-facing support direction must be nonzero")
    direction_x /= direction_norm
    direction_y /= direction_norm

    structure_boxes = table_structure_boxes(placement)
    leg_boxes = [box for box in structure_boxes if box["role"] == "table_leg"]
    if leg_boxes:
        support_center_xy = [
            float(value)
            for value in placement.get("position", [0.0, 0.0])[:2]
        ]
        camera_facing_legs = sorted(
            leg_boxes,
            key=lambda box: (
                (float(box["center_m"][0]) - support_center_xy[0]) * direction_x
                + (float(box["center_m"][1]) - support_center_xy[1]) * direction_y
            ),
            reverse=True,
        )[:2]
        visibility_points: list[tuple[float, float, float]] = []
        for box in camera_facing_legs:
            center = [float(value) for value in box["center_m"]]
            dimensions = [float(value) for value in box["dimensions_m"]]
            visibility_points.append(
                (center[0], center[1], center[2] - 0.22 * dimensions[2])
            )
        return {
            "policy": "table_frame_two_visible_legs_v2",
            "required_visible_table_leg_ids": [box["id"] for box in camera_facing_legs],
            "required_visible_support_edge_ids": [],
            "visibility_points": visibility_points,
        }

    if bool(placement.get("show_table_legs", False)):
        raise ValueError("show_table_legs requires a declared support_structure")

    support_shape = str(placement.get("support_shape", "rectangular_slab"))
    if support_shape == "inclined_ramp":
        return {
            "policy": "inclined_support_edge_context_v1",
            "required_visible_table_leg_ids": [],
            "required_visible_support_edge_ids": [],
            "visibility_points": [],
        }
    if support_shape != "rectangular_slab":
        raise ValueError(f"unsupported support context shape: {support_shape}")

    slab = support_slab_aabb(placement)
    lower = slab["min"]
    upper = slab["max"]
    thickness = upper[2] - lower[2]
    if thickness <= 0.0:
        raise ValueError("support slab must have positive thickness")
    edge_offset_m = 0.003
    edge_z = lower[2] + 0.35 * thickness
    if abs(direction_x) >= abs(direction_y):
        face_x = upper[0] + edge_offset_m if direction_x >= 0.0 else lower[0] - edge_offset_m
        y_span = upper[1] - lower[1]
        visibility_points = [
            (face_x, lower[1] + fraction * y_span, edge_z)
            for fraction in (0.28, 0.72)
        ]
    else:
        face_y = upper[1] + edge_offset_m if direction_y >= 0.0 else lower[1] - edge_offset_m
        x_span = upper[0] - lower[0]
        visibility_points = [
            (lower[0] + fraction * x_span, face_y, edge_z)
            for fraction in (0.28, 0.72)
        ]
    semantic_class = str(source_semantic_class or "")
    policy = (
        "finite_floor_surface_front_edge_v1"
        if semantic_class == "floor" or float(placement["support_top_z"]) <= 0.20
        else "integral_slab_front_edge_v1"
    )
    return {
        "policy": policy,
        "required_visible_table_leg_ids": [],
        "required_visible_support_edge_ids": ["front_edge_0", "front_edge_1"],
        "visibility_points": visibility_points,
    }


def support_slab_aabb(placement: dict[str, Any]) -> dict[str, list[float]]:
    """Return the solid top slab that shields objects from the frame below."""
    yaw_degrees = float(placement.get("yaw_degrees", 0.0))
    if abs(yaw_degrees) > 1.0e-8:
        raise ValueError("support slab AABB currently requires zero yaw")
    target_x, target_y = [float(value) for value in placement["target_size"][:2]]
    support_top = float(placement["support_top_z"])
    thickness = float(placement["thickness"])
    position = [float(value) for value in placement.get("position", [0.0, 0.0, support_top])]
    if min(target_x, target_y, thickness) <= 0.0:
        raise ValueError("support slab dimensions must be positive")
    return {
        "min": [
            position[0] - target_x / 2.0,
            position[1] - target_y / 2.0,
            support_top - thickness,
        ],
        "max": [
            position[0] + target_x / 2.0,
            position[1] + target_y / 2.0,
            support_top,
        ],
    }


def box_aabb(box: dict[str, Any]) -> dict[str, list[float]]:
    if abs(float(box.get("yaw_degrees", 0.0))) > 1.0e-8:
        raise ValueError("support-structure AABB QA currently requires zero yaw")
    center = [float(value) for value in box["center_m"]]
    half = [0.5 * float(value) for value in box["dimensions_m"]]
    return {
        "min": [center[index] - half[index] for index in range(3)],
        "max": [center[index] + half[index] for index in range(3)],
    }
