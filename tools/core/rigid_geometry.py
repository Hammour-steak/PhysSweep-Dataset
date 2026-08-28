#!/usr/bin/env python3
"""Backend-neutral geometry helpers for the PhysSweep rigid pipeline."""

from __future__ import annotations

import math
from typing import Any


def finite_vector(value: Any, length: int, label: str) -> list[float]:
    """Return a fixed-length vector whose components are all finite."""

    result = [float(item) for item in value]
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"invalid {label}: {value}")
    return result


def positive_vector(value: Any, length: int, label: str) -> list[float]:
    """Return a fixed-length vector whose components are strictly positive."""

    result = [float(item) for item in value]
    if len(result) != length or not all(
        math.isfinite(item) and item > 0.0 for item in result
    ):
        raise ValueError(f"invalid {label}: {value}")
    return result


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def normalize(values: list[float] | tuple[float, ...]) -> list[float]:
    length = math.sqrt(sum(float(value) ** 2 for value in values))
    if length <= 1.0e-12:
        raise ValueError("cannot normalize a zero vector")
    return [float(value) / length for value in values]


def cross(left: list[float], right: list[float]) -> list[float]:
    return [
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    ]


def quaternion_wxyz_from_euler_degrees(euler_degrees: list[float]) -> list[float]:
    """Return a Blender-order quaternion for intrinsic XYZ Euler angles."""

    roll, pitch, yaw = [math.radians(float(value)) / 2.0 for value in euler_degrees]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def quaternion_xyzw_from_wxyz(value: list[float]) -> list[float]:
    if len(value) != 4:
        raise ValueError("quaternion must contain four values")
    return [float(value[1]), float(value[2]), float(value[3]), float(value[0])]


def quaternion_wxyz_from_basis(
    local_x: list[float], local_y: list[float], local_z: list[float]
) -> list[float]:
    matrix = [
        [local_x[0], local_y[0], local_z[0]],
        [local_x[1], local_y[1], local_z[1]],
        [local_x[2], local_y[2], local_z[2]],
    ]
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = [
            0.25 * scale,
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
        ]
    else:
        diagonal = [matrix[0][0], matrix[1][1], matrix[2][2]]
        index = max(range(3), key=diagonal.__getitem__)
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
            quaternion = [
                (matrix[2][1] - matrix[1][2]) / scale,
                0.25 * scale,
                (matrix[0][1] + matrix[1][0]) / scale,
                (matrix[0][2] + matrix[2][0]) / scale,
            ]
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
            quaternion = [
                (matrix[0][2] - matrix[2][0]) / scale,
                (matrix[0][1] + matrix[1][0]) / scale,
                0.25 * scale,
                (matrix[1][2] + matrix[2][1]) / scale,
            ]
        else:
            scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
            quaternion = [
                (matrix[1][0] - matrix[0][1]) / scale,
                (matrix[0][2] + matrix[2][0]) / scale,
                (matrix[1][2] + matrix[2][1]) / scale,
                0.25 * scale,
            ]
    return normalize(quaternion)


def euler_degrees_from_quaternion_wxyz(value: list[float]) -> list[float]:
    w, x, y, z = normalize(value)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_term = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_term)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def direction_unit(angle_degrees: float) -> list[float]:
    angle = math.radians(float(angle_degrees))
    return [math.cos(angle), math.sin(angle), 0.0]


def object_contact_offset_m(shape: str, size_m: list[float]) -> float:
    if len(size_m) != 3 or min(float(value) for value in size_m) <= 0.0:
        raise ValueError("object size must contain three positive values")
    if shape == "sphere":
        return float(size_m[0]) / 2.0
    if shape in {"cuboid", "cylinder"}:
        return float(size_m[2]) / 2.0
    raise ValueError(f"unsupported rigid shape: {shape}")


def _layout_for_support(support: dict[str, Any], motion: str, subtype: dict[str, Any]) -> str:
    placement = support.get("overrides", {}).get("placement", {})
    support_shape = str(placement.get("support_shape", "rectangular_slab"))
    subtype_label = str(subtype["label"])
    if support_shape == "inclined_ramp":
        return str(placement.get("structure_family", "straight_standard"))
    if support_shape == "tray_surface":
        return "tray_surface"
    if support_shape == "pedestal_block":
        return "raised_platform"
    if bool(placement.get("ground_surface", False)):
        return "floor_patch"
    if motion in {
        "drop_fall_1obj",
        "projectile_1obj",
        "arc_projectile_1obj",
        "edge_fall_1obj",
    }:
        return "wide_flat"
    if "short" in subtype_label:
        return "narrow_flat"
    if "long" in subtype_label or "fast" in subtype_label:
        return "wide_flat"
    return "standard_flat"


def _static_box(
    collider_id: str,
    size_m: list[float],
    position_m: list[float],
    *,
    role: str,
    material_role: str,
    rotation_euler_degrees: list[float] | None = None,
    visible: bool = True,
    occludes_camera: bool = False,
) -> dict[str, Any]:
    if len(size_m) != 3 or min(float(value) for value in size_m) <= 0.0:
        raise ValueError(f"invalid static box size: {collider_id}")
    if len(position_m) != 3:
        raise ValueError(f"invalid static box position: {collider_id}")
    return {
        "id": collider_id,
        "primitive": "box",
        "role": role,
        "material_role": material_role,
        "size_m": [round(float(value), 6) for value in size_m],
        "position_m": [round(float(value), 6) for value in position_m],
        "rotation_euler_degrees": [
            round(float(value), 6)
            for value in (rotation_euler_degrees or [0.0, 0.0, 0.0])
        ],
        "visible": bool(visible),
        "collision_enabled": True,
        "occludes_camera": bool(occludes_camera),
    }


def _support_transition_contract(
    motion: str,
    support_size_m: list[float],
    surface_top_z_m: float,
    colliders: list[dict[str, Any]],
    motion_direction: list[float],
) -> dict[str, Any] | None:
    if motion not in {"edge_fall_1obj", "ramp_to_flat_1obj"}:
        return None
    source = next(
        (collider for collider in colliders if collider["role"] == "primary_support"),
        None,
    )
    if source is None:
        raise ValueError("support transition has no primary support collider")
    destination_role_order = (
        ["environment_floor"]
        if motion == "edge_fall_1obj"
        else ["landing_surface", "environment_floor"]
    )
    destination = next(
        (
            collider
            for role in destination_role_order
            for collider in colliders
            if collider["role"] == role
        ),
        None,
    )
    if destination is None:
        raise ValueError("support transition requires a destination surface")
    destination_top_z = float(destination["position_m"][2]) + 0.5 * float(
        destination["size_m"][2]
    )
    direction = [float(motion_direction[0]), float(motion_direction[1])]
    norm = math.hypot(*direction)
    if norm <= 1.0e-8:
        raise ValueError("support transition direction must be nonzero")
    direction = [value / norm for value in direction]
    if motion == "ramp_to_flat_1obj":
        boundary_xy = [0.0, -support_size_m[1] / 2.0]
        boundary_z = destination_top_z
        transition_type = "incline_to_horizontal"
        intermediate_phase = "continuous_contact"
    else:
        half_size = [float(value) / 2.0 for value in support_size_m[:2]]
        distances = [
            half_size[axis] / abs(direction[axis])
            for axis in range(2)
            if abs(direction[axis]) > 1.0e-8
        ]
        if not distances:
            raise ValueError("support transition direction misses the support boundary")
        distance = min(distances)
        boundary_xy = [value * distance for value in direction]
        boundary_z = float(surface_top_z_m)
        transition_type = "raised_edge_to_floor"
        intermediate_phase = "airborne"
    return {
        "version": "physweep_support_transition_v1",
        "type": transition_type,
        "source_collider_id": str(source["id"]),
        "destination_collider_id": str(destination["id"]),
        "boundary_point_m": [
            round(float(boundary_xy[0]), 6),
            round(float(boundary_xy[1]), 6),
            round(boundary_z, 6),
        ],
        "outward_direction_xy": [
            round(float(direction[0]), 9),
            round(float(direction[1]), 9),
        ],
        "source_boundary_height_m": round(boundary_z, 6),
        "destination_surface_height_m": round(destination_top_z, 6),
        "height_drop_m": round(
            max(0.0, boundary_z - destination_top_z), 6
        ),
        "intermediate_phase": intermediate_phase,
        "required_contact_sequence": ["source", "destination"],
        "allow_source_recontact_after_destination": False,
    }


def _compile_support_geometry(
    support: dict[str, Any],
    motion: str,
    subtype: dict[str, Any],
    motion_direction: list[float] | None = None,
) -> dict[str, Any]:
    """Compile one semantic support record into explicit rigid colliders."""

    placement = dict(support.get("overrides", {}).get("placement", {}))
    support_shape = str(placement.get("support_shape", "rectangular_slab"))
    scene_class = str(support["scene_class"])
    ground_surface = bool(placement.get("ground_surface", False))
    structure_style = str(placement.get("structure_style", "none"))
    layout = _layout_for_support(support, motion, subtype)
    size = [float(value) for value in support["size"]]
    direction = normalize(motion_direction or [0.0, 1.0, 0.0])
    direction = normalize([direction[0], direction[1], 0.0])
    if layout == "wide_flat":
        size[0] *= 1.16
        size[1] *= 1.10
    elif layout == "narrow_flat":
        size[0] *= 0.92
        size[1] *= 0.78
    elif layout == "floor_patch":
        size[0] *= 1.10
        size[1] *= 1.10
    top_z = float(support["top_z"])
    thickness = float(placement.get("thickness", 0.08))
    if ground_surface:
        if scene_class != "ground_flat" or support_shape != "rectangular_slab":
            raise ValueError("ground_surface is reserved for ground_flat rectangular supports")
        if abs(top_z) > 1.0e-8:
            raise ValueError("ground_surface top_z must be zero")
        thickness = 0.10
    if support_shape == "pedestal_block":
        thickness = float(placement.get("thickness", top_z))
    if (
        scene_class == "ground_feature"
        and support_shape != "inclined_ramp"
        and abs(top_z - thickness) > 1.0e-8
    ):
        raise ValueError(
            "horizontal ground_feature supports must rest on the environment floor"
        )
    colliders: list[dict[str, Any]] = []
    visual_geometry: dict[str, Any] | None = None

    slope_angle = 0.0
    if support_shape == "inclined_ramp":
        if str(placement.get("slope_axis", "y")) != "y":
            raise ValueError("rigid ramp v1 supports only slope_axis=y")
        structure_family = str(placement.get("structure_family", "straight_standard"))
        rise = float(placement.get("slope_rise_m", 0.30))
        slope_angle = math.atan2(rise, size[1])
        slope_angle_degrees = math.degrees(slope_angle)
        if not 5.0 <= slope_angle_degrees <= 30.0:
            raise ValueError("inclined support angle must remain within 5 to 30 degrees")
        family_angle_ranges = {
            "straight_long_shallow": (8.0, 12.0),
            "straight_standard": (12.0, 18.0),
            "channel_medium": (12.0, 18.0),
            "straight_short_steep": (20.0, 30.0),
        }
        if structure_family in family_angle_ranges:
            minimum_angle, maximum_angle = family_angle_ranges[structure_family]
            if not minimum_angle <= slope_angle_degrees <= maximum_angle:
                raise ValueError(
                    f"{structure_family} angle must remain within "
                    f"{minimum_angle:g} to {maximum_angle:g} degrees"
                )
        anchor_to_floor = bool(placement.get("anchor_low_edge_to_floor", False))
        platform_top = placement.get("base_platform_top_z_m")
        if anchor_to_floor and platform_top is not None:
            raise ValueError("inclined support cannot use floor and platform anchors together")
        low_edge_offset = (
            rise * math.cos(slope_angle) / 2.0
            + thickness * math.sin(slope_angle) ** 2
            / (2.0 * math.cos(slope_angle))
        )
        if anchor_to_floor:
            if scene_class != "ground_feature":
                raise ValueError("floor-anchored ramps require scene_class=ground_feature")
            top_z = low_edge_offset
        elif platform_top is not None:
            if scene_class != "raised_feature":
                raise ValueError("platform-anchored ramps require scene_class=raised_feature")
            platform_top = float(platform_top)
            if platform_top <= 0.20:
                raise ValueError("raised ramp platform must remain visibly above the floor")
            top_z = platform_top + low_edge_offset
        center_z = top_z - thickness / (2.0 * math.cos(slope_angle))
        colliders.append(
            _static_box(
                "support",
                [size[0], size[1], thickness],
                [0.0, 0.0, center_z],
                role="primary_support",
                material_role="support_surface",
                rotation_euler_degrees=[math.degrees(slope_angle), 0.0, 0.0],
                occludes_camera=True,
            )
        )
        if platform_top is not None:
            margin = [float(value) for value in placement["base_platform_margin_m"]]
            platform_thickness = float(placement["base_platform_thickness_m"])
            if len(margin) != 2 or min(margin) <= 0.0 or platform_thickness <= 0.0:
                raise ValueError("raised ramp platform dimensions must be positive")
            platform_size = [size[0] + margin[0], size[1] + margin[1]]
            colliders.append(
                _static_box(
                    "ramp_base_platform",
                    [platform_size[0], platform_size[1], platform_thickness],
                    [0.0, 0.0, platform_top - platform_thickness / 2.0],
                    role="support_platform",
                    material_role="support_structure",
                    occludes_camera=True,
                )
            )
            leg_height = platform_top - platform_thickness
            leg_size = 0.10
            for x_sign in (-1.0, 1.0):
                for y_sign in (-1.0, 1.0):
                    colliders.append(
                        _static_box(
                            f"ramp_platform_leg_{int(x_sign):+d}_{int(y_sign):+d}",
                            [leg_size, leg_size, leg_height],
                            [
                                x_sign * (platform_size[0] / 2.0 - 0.20),
                                y_sign * (platform_size[1] / 2.0 - 0.16),
                                leg_height / 2.0,
                            ],
                            role="support_structure",
                            material_role="support_structure",
                            occludes_camera=True,
                        )
                    )
        support_base_z = float(platform_top) if platform_top is not None else 0.0
        visual_geometry = {
            "primitive": "solid_wedge",
            "size_xy_m": [round(size[0], 6), round(size[1], 6)],
            "base_z_m": round(top_z - rise / 2.0, 6),
            "high_top_z_m": round(top_z + rise / 2.0, 6),
            "slope_axis": "y",
        }
        colliders[0]["render_replaced_by_solid_wedge"] = True
        for name, y in (
            ("ramp_base_low", -size[1] * 0.38),
            ("ramp_base_high", size[1] * 0.38),
        ):
            ramp_bottom_z = (
                center_z
                + y * math.sin(slope_angle)
                - thickness * math.cos(slope_angle) / 2.0
            )
            height = ramp_bottom_z - support_base_z
            if height <= 0.025:
                continue
            colliders.append(
                _static_box(
                    name,
                    [size[0] * 0.72, 0.12, height],
                    [0.0, y, support_base_z + height / 2.0],
                    role="support_structure",
                    material_role="support_structure",
                    occludes_camera=True,
                )
            )
            colliders[-1]["render_replaced_by_solid_wedge"] = True
        rail_height = float(placement.get("side_rail_height_m", 0.0))
        rail_width = float(placement.get("side_rail_width_m", 0.0))
        has_side_rails = rail_height > 0.0 or rail_width > 0.0
        if has_side_rails:
            if min(rail_height, rail_width) <= 0.0:
                raise ValueError("inclined support side rails require positive height and width")
            if structure_family != "channel_medium":
                raise ValueError("inclined side rails are reserved for channel structures")
            if rail_width * 2.0 >= size[0] * 0.35:
                raise ValueError("inclined side rails leave insufficient usable width")
            normal = [0.0, -math.sin(slope_angle), math.cos(slope_angle)]
            rail_center_z = center_z + normal[2] * (thickness + rail_height) / 2.0
            rail_center_y = normal[1] * (thickness + rail_height) / 2.0
            for sign in (-1.0, 1.0):
                colliders.append(
                    _static_box(
                        f"ramp_side_rail_{int(sign):+d}",
                        [rail_width, size[1], rail_height],
                        [
                            sign * (size[0] / 2.0 - rail_width / 2.0),
                            rail_center_y,
                            rail_center_z,
                        ],
                        role="support_rail",
                        material_role="support_structure",
                        rotation_euler_degrees=[slope_angle_degrees, 0.0, 0.0],
                        occludes_camera=True,
                    )
                )
        elif structure_family == "channel_medium":
            raise ValueError("channel structures require inclined side rails")
        if motion == "ramp_to_flat_1obj":
            landing_length = float(placement["landing_length_m"])
            landing_top = (
                center_z
                - size[1] * math.sin(slope_angle) / 2.0
                + thickness * math.cos(slope_angle) / 2.0
            )
            if landing_top <= 0.025:
                if not anchor_to_floor:
                    raise ValueError("ramp-to-flat landing must remain above the floor")
            else:
                colliders.append(
                    _static_box(
                        "landing_surface",
                        [size[0], landing_length, landing_top],
                        [0.0, -size[1] / 2.0 - landing_length / 2.0, landing_top / 2.0],
                        role="landing_surface",
                        material_role="support_surface",
                        occludes_camera=True,
                    )
                )
    else:
        support_size = [8.0, 8.0, thickness] if ground_surface else [size[0], size[1], thickness]
        colliders.append(
            _static_box(
                "support",
                support_size,
                [0.0, 0.0, top_z - thickness / 2.0],
                role="primary_support",
                material_role="support_surface",
                occludes_camera=not ground_surface,
            )
        )
        if structure_style == "corridor":
            wall_height = float(placement["corridor_wall_height_m"])
            wall_thickness = float(placement["corridor_wall_thickness_m"])
            if wall_height < 0.6 or wall_thickness < 0.04:
                raise ValueError("corridor walls are too small for physical use")
            if wall_thickness * 2.0 >= size[1] * 0.25:
                raise ValueError("corridor walls leave insufficient clear width")
            for sign in (-1.0, 1.0):
                colliders.append(
                    _static_box(
                        f"corridor_wall_{int(sign):+d}",
                        [size[0], wall_thickness, wall_height],
                        [
                            0.0,
                            sign * (size[1] / 2.0 - wall_thickness / 2.0),
                            wall_height / 2.0,
                        ],
                        role="support_structure",
                        material_role="back_wall",
                        occludes_camera=True,
                    )
                )
        if motion == "wall_impact_1obj":
            wall_offset = 0.46
            wall_height = 0.48
            wall_yaw = math.degrees(math.atan2(direction[1], direction[0]))
            colliders.append(
                _static_box(
                    "impact_wall",
                    [0.06, 0.92, wall_height],
                    [
                        direction[0] * wall_offset,
                        direction[1] * wall_offset,
                        top_z + wall_height / 2.0,
                    ],
                    role="impact_wall",
                    material_role="support_structure",
                    rotation_euler_degrees=[0.0, 0.0, wall_yaw],
                    occludes_camera=True,
                )
            )

        rail_height = float(placement.get("rail_height_m", 0.0))
        rail_width = float(placement.get("rail_width_m", 0.0))
        if support_shape == "tray_surface":
            for sign in (-1.0, 1.0):
                colliders.append(
                    _static_box(
                        f"rail_y_{int(sign):+d}",
                        [size[0], rail_width, rail_height],
                        [0.0, sign * (size[1] / 2.0 - rail_width / 2.0), top_z + rail_height / 2.0],
                        role="support_rail",
                        material_role="support_structure",
                        occludes_camera=True,
                    )
                )
            if support_shape == "tray_surface":
                for sign in (-1.0, 1.0):
                    colliders.append(
                        _static_box(
                            f"rail_x_{int(sign):+d}",
                            [rail_width, size[1] - 2.0 * rail_width, rail_height],
                            [sign * (size[0] / 2.0 - rail_width / 2.0), 0.0, top_z + rail_height / 2.0],
                            role="support_rail",
                            material_role="support_structure",
                            occludes_camera=True,
                        )
                    )

        structural_height = top_z - thickness
        if bool(placement.get("show_table_legs", False)) and structural_height > 0.12:
            leg_size = 0.10
            for x_sign in (-1.0, 1.0):
                for y_sign in (-1.0, 1.0):
                    colliders.append(
                        _static_box(
                            f"table_leg_{int(x_sign):+d}_{int(y_sign):+d}",
                            [leg_size, leg_size, structural_height],
                            [
                                x_sign * (size[0] / 2.0 - 0.22),
                                y_sign * (size[1] / 2.0 - 0.16),
                                structural_height / 2.0,
                            ],
                            role="support_structure",
                            material_role="support_structure",
                            occludes_camera=True,
                        )
                    )
        elif structure_style == "cabinet" and structural_height > 0.12:
            colliders.append(
                _static_box(
                    "counter_cabinet",
                    [size[0] * 0.90, size[1] * 0.86, structural_height],
                    [0.0, 0.03, structural_height / 2.0],
                    role="support_structure",
                    material_role="support_structure",
                    occludes_camera=True,
                )
            )

    if not ground_surface:
        recovery_top = 0.0
        colliders.append(
            _static_box(
                "environment_floor",
                [20.0, 20.0, 0.10],
                [0.0, 0.0, recovery_top - 0.05],
                role="environment_floor",
                material_role="room_floor",
                visible=True,
            )
        )

    angle_degrees = math.degrees(slope_angle)
    normal = [0.0, -math.sin(slope_angle), math.cos(slope_angle)]
    tangent_uphill = [0.0, math.cos(slope_angle), math.sin(slope_angle)]
    safe_margin = {
        "tray_surface": 0.055,
        "inclined_ramp": 0.085,
    }.get(support_shape, 0.08)
    result = {
        "semantic_type": str(support["label"]),
        "scene_class": scene_class,
        "layout": layout,
        "structure_family": str(placement.get("structure_family", layout)),
        "structure_anchor": (
            "floor_flush_low_edge"
            if bool(placement.get("anchor_low_edge_to_floor", False))
            else (
                "raised_platform_flush_low_edge"
                if placement.get("base_platform_top_z_m") is not None
                else "free_standing"
            )
        ),
        "support_shape": support_shape,
        "size_m": [round(value, 6) for value in size],
        "surface_center_z_m": round(top_z, 6),
        "surface_frame": {
            "slope_angle_degrees": round(angle_degrees, 6),
            "normal": [round(value, 9) for value in normal],
            "tangent_uphill": [round(value, 9) for value in tangent_uphill],
            "tangent_cross": [1.0, 0.0, 0.0],
        },
        "safe_surface_bounds": {
            "x": [round(-size[0] / 2.0 + safe_margin, 6), round(size[0] / 2.0 - safe_margin, 6)],
            "y": [round(-size[1] / 2.0 + safe_margin, 6), round(size[1] / 2.0 - safe_margin, 6)],
        },
        "colliders": colliders,
    }
    transition_contract = _support_transition_contract(
        motion,
        size,
        top_z,
        colliders,
        motion_direction or [0.0, -1.0, 0.0],
    )
    if transition_contract is not None:
        result["transition_contract"] = transition_contract
    if "motion_axis" in placement:
        result["motion_axis"] = str(placement["motion_axis"])
    if structure_style == "corridor":
        side_walls = [
            collider
            for collider in colliders
            if collider["role"] == "support_structure"
            and collider["material_role"] == "back_wall"
        ]
        if len(side_walls) != 2:
            raise ValueError("corridor camera envelope requires two side walls")
        result["camera_envelope"] = {
            "type": "paired_parallel_walls",
            "motion_axis": str(placement["motion_axis"]),
            "collider_ids": [str(collider["id"]) for collider in side_walls],
            "clearance_m": round(float(placement["camera_clearance_m"]), 6),
        }
    if visual_geometry is not None:
        result["visual_geometry"] = visual_geometry
    if "landing_length_m" in placement:
        result["landing_length_m"] = round(
            float(placement["landing_length_m"]), 6
        )
    if "maximum_planar_trajectory_distance_m" in placement:
        maximum_distance = float(placement["maximum_planar_trajectory_distance_m"])
        if maximum_distance <= 0.0:
            raise ValueError("maximum trajectory distance must be positive")
        result["maximum_planar_trajectory_distance_m"] = round(
            maximum_distance, 6
        )
    return result


def _unsupported_pocketed_table(
    support: dict[str, Any],
    motion: str,
    subtype: dict[str, Any],
    motion_direction: list[float] | None = None,
) -> dict[str, Any]:
    del support, motion, subtype, motion_direction
    raise ValueError(
        "pocketed_table is catalogued but not active until its segmented collision "
        "and visual transform are calibrated"
    )


SUPPORT_BUILDERS = {
    "flat_surface": _compile_support_geometry,
    "inclined_ramp": _compile_support_geometry,
    "tray": _compile_support_geometry,
    "pedestal": _compile_support_geometry,
    "pocketed_table": _unsupported_pocketed_table,
}

SUPPORT_SHAPE_TO_TOPOLOGY = {
    "rectangular_slab": "flat_surface",
    "inclined_ramp": "inclined_ramp",
    "tray_surface": "tray",
    "pedestal_block": "pedestal",
    "pocketed_table": "pocketed_table",
}


def build_support_geometry(
    support: dict[str, Any],
    motion: str,
    subtype: dict[str, Any],
    motion_direction: list[float] | None = None,
) -> dict[str, Any]:
    """Dispatch a scene kit by topology, never by a concrete scene id."""

    placement = support.get("overrides", {}).get("placement", {})
    support_shape = str(placement.get("support_shape", "rectangular_slab"))
    topology = str(support.get("topology", SUPPORT_SHAPE_TO_TOPOLOGY.get(support_shape, "")))
    try:
        builder = SUPPORT_BUILDERS[topology]
    except KeyError as exc:
        raise ValueError(f"unsupported support topology: {topology or support_shape}") from exc
    geometry = builder(support, motion, subtype, motion_direction)
    geometry["topology"] = topology
    return geometry


def support_surface_height_m(support_geometry: dict[str, Any], x: float, y: float) -> float:
    del x
    frame = support_geometry["surface_frame"]
    angle = math.radians(float(frame["slope_angle_degrees"]))
    return float(support_geometry["surface_center_z_m"]) + float(y) * math.tan(angle)


def pose_on_support(
    support_geometry: dict[str, Any],
    shape: str,
    size_m: list[float],
    x: float,
    y: float,
    yaw_degrees: float,
    clearance_m: float,
    pose_profile: str = "support_normal",
    motion_direction: list[float] | None = None,
) -> dict[str, Any]:
    frame = support_geometry["surface_frame"]
    normal = [float(value) for value in frame["normal"]]
    if pose_profile == "support_normal":
        offset = object_contact_offset_m(shape, size_m) + float(clearance_m)
        euler = [float(frame["slope_angle_degrees"]), 0.0, float(yaw_degrees)]
        quaternion = quaternion_wxyz_from_euler_degrees(euler)
    elif pose_profile == "side_on_motion":
        if shape != "cylinder" or motion_direction is None:
            raise ValueError("side_on_motion requires a cylinder and motion direction")
        direction = [float(value) for value in motion_direction]
        tangent = normalize(
            [
                direction[index]
                - normal[index]
                * sum(direction[axis] * normal[axis] for axis in range(3))
                for index in range(3)
            ]
        )
        cylinder_axis = normalize(cross(normal, tangent))
        local_y = normal
        local_x = normalize(cross(local_y, cylinder_axis))
        quaternion = quaternion_wxyz_from_basis(local_x, local_y, cylinder_axis)
        euler = euler_degrees_from_quaternion_wxyz(quaternion)
        offset = float(size_m[0]) / 2.0 + float(clearance_m)
    else:
        raise ValueError(f"unsupported pose profile: {pose_profile}")
    contact_z = support_surface_height_m(support_geometry, x, y)
    contact = [float(x), float(y), contact_z]
    center = [contact[index] + normal[index] * offset for index in range(3)]
    return {
        "contact_point_m": [round(value, 6) for value in contact],
        "position_m": [round(value, 6) for value in center],
        "orientation_euler_degrees": [round(value, 6) for value in euler],
        "orientation_quaternion_wxyz": [
            round(value, 9) for value in quaternion
        ],
    }


def slope_tangent_velocity(support_geometry: dict[str, Any], speed_m_s: float, uphill: bool) -> list[float]:
    tangent = [float(value) for value in support_geometry["surface_frame"]["tangent_uphill"]]
    sign = 1.0 if uphill else -1.0
    return [round(sign * tangent[index] * float(speed_m_s), 6) for index in range(3)]


def validate_support_geometry(value: dict[str, Any]) -> None:
    ids = [str(record["id"]) for record in value["colliders"]]
    if len(ids) != len(set(ids)):
        raise ValueError("support collider ids must be unique")
    bounds = value["safe_surface_bounds"]
    if float(bounds["x"][0]) >= float(bounds["x"][1]):
        raise ValueError("support safe x bounds are empty")
    if float(bounds["y"][0]) >= float(bounds["y"][1]):
        raise ValueError("support safe y bounds are empty")
    for collider in value["colliders"]:
        if collider["primitive"] != "box":
            raise ValueError("rigid support v1 accepts only box colliders")
        if min(float(item) for item in collider["size_m"]) <= 0.0:
            raise ValueError(f"nonpositive collider dimension: {collider['id']}")
