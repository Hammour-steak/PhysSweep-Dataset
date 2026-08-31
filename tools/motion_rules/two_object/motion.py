"""Initial-state builders for the bounded two-object motion matrix."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from tools.core.rigid_geometry import (
    finite_vector,
    object_contact_offset_m,
    positive_vector,
    sphere_primitive_center_distance_m,
    upright_footprint_half_extents_m,
    upright_pair_center_distance_m,
)


_SHARED_FIELDS = {
    "schema_version",
    "contact_friction",
    "contact_restitution",
    "initial_support_clearance_m",
    "minimum_support_contact_fraction",
    "interaction_audit",
}
_AUDIT_FIELDS = {
    "minimum_initial_clearance_m",
    "minimum_pre_contact_closing_speed_m_s",
    "maximum_first_contact_time_s",
    "maximum_independent_rest_path_length_m",
    "minimum_pre_contact_arc_vertical_ascent_m",
}
_OBSERVATION_FIELDS = {
    "schema_version",
    "solver_profile_template_id",
    "focal_length_mm",
    "maximum_camera_view_azimuth_deviation_degrees",
    "minimum_camera_distance_m",
    "maximum_camera_distance_m",
    "maximum_camera_distance_above_minimum_m",
    "full_motion_envelope_margin_ndc",
    "preferred_full_motion_envelope_span_ndc",
    "minimum_per_object_median_span_ndc",
    "minimum_per_object_unoccluded_fraction",
    "minimum_pair_keyframe_projected_center_separation_to_radius_sum_ratio",
    "minimum_support_context_visible_fraction",
    "minimum_support_anchor_visible_fraction",
    "minimum_support_anchor_unoccluded_fraction",
}
_VIEW_FAMILY_FIELDS = {
    "id",
    "relative_azimuth_degrees",
    "preferred_elevation_degrees",
    "minimum_elevation_degrees",
    "maximum_elevation_degrees",
}
_COMMON_INTENT_FIELDS = {
    "id",
    "interaction_class",
    "kinematic_regime",
    "layout",
    "object_motions",
    "linear_velocity_m_s",
    "minimum_displacement_m",
}
_LAYOUT_FIELDS = {
    "planned_supported_contact": {"contact_time_s", "impact_offset_ratio"},
    "ballistic_airborne_contact": {
        "contact_time_s",
        "contact_elevation_degrees",
    },
    "ballistic_airborne_pair_contact": {
        "contact_time_s",
        "contact_center_height_above_support_m",
    },
    "parallel_supported_independent": {
        "initial_x_offset_m",
        "lateral_clearance_m",
    },
    "separated_airborne_supported_independent": {
        "airborne_surface_gap_m",
        "lateral_clearance_m",
    },
}
_LAYOUT_CONTRACT = {
    "planned_supported_contact": ("interacting", "supported_supported"),
    "ballistic_airborne_contact": ("interacting", "airborne_supported"),
    "ballistic_airborne_pair_contact": ("interacting", "airborne_airborne"),
    "parallel_supported_independent": ("independent", "supported_supported"),
    "separated_airborne_supported_independent": (
        "independent",
        "airborne_supported",
    ),
}
_NUMERICAL_EPSILON = 1.0e-8
_SUPPORT_EDGE_MARGIN_M = 0.02
_MAXIMUM_IMPACT_OFFSET_RATIO = 0.80


def _validate_contracts(
    shared: dict[str, Any],
    observation: dict[str, Any],
    view_family: dict[str, Any],
    intent: dict[str, Any],
) -> str:
    if set(shared) != _SHARED_FIELDS or (
        shared.get("schema_version")
        != "physweep_two_object_shared_physics_v2"
    ):
        raise ValueError("unsupported two-object shared physics")
    audit = shared.get("interaction_audit")
    if not isinstance(audit, dict) or set(audit) != _AUDIT_FIELDS:
        raise ValueError("two-object interaction audit fields are incomplete")
    positive_vector(
        [
            audit["minimum_initial_clearance_m"],
            audit["minimum_pre_contact_closing_speed_m_s"],
            audit["maximum_first_contact_time_s"],
            audit["maximum_independent_rest_path_length_m"],
            audit["minimum_pre_contact_arc_vertical_ascent_m"],
        ],
        5,
        "two-object interaction audit values",
    )
    if set(observation) != _OBSERVATION_FIELDS:
        raise ValueError("two-object observation fields are incomplete")
    if (
        observation.get("schema_version")
        != "physweep_two_object_camera_observation_v2"
    ):
        raise ValueError("unsupported two-object camera observation")
    (
        focal_length,
        camera_view_deviation,
        minimum_camera_distance,
        maximum_camera_distance,
        maximum_distance_above_minimum,
        envelope_margin,
        preferred_envelope_span,
        minimum_object_span,
        minimum_unoccluded_fraction,
        projected_separation_ratio,
        minimum_context_fraction,
        minimum_anchor_fraction,
        minimum_anchor_unoccluded_fraction,
    ) = finite_vector(
        [
            observation["focal_length_mm"],
            observation["maximum_camera_view_azimuth_deviation_degrees"],
            observation["minimum_camera_distance_m"],
            observation["maximum_camera_distance_m"],
            observation["maximum_camera_distance_above_minimum_m"],
            observation["full_motion_envelope_margin_ndc"],
            observation["preferred_full_motion_envelope_span_ndc"],
            observation["minimum_per_object_median_span_ndc"],
            observation["minimum_per_object_unoccluded_fraction"],
            observation[
                "minimum_pair_keyframe_projected_center_separation_to_radius_sum_ratio"
            ],
            observation["minimum_support_context_visible_fraction"],
            observation["minimum_support_anchor_visible_fraction"],
            observation["minimum_support_anchor_unoccluded_fraction"],
        ],
        13,
        "two-object observation values",
    )
    if not str(observation["solver_profile_template_id"]).strip():
        raise ValueError("two-object camera solver profile is empty")
    if focal_length <= 0.0:
        raise ValueError("two-object camera focal length must be positive")
    if not 0.0 < camera_view_deviation <= 45.0:
        raise ValueError("camera view deviation must lie in (0, 45] degrees")
    if set(view_family) != _VIEW_FAMILY_FIELDS:
        raise ValueError("two-object camera-view family is incomplete")
    relative_azimuth, preferred_elevation, minimum_elevation, maximum_elevation = (
        finite_vector(
            [
                view_family["relative_azimuth_degrees"],
                view_family["preferred_elevation_degrees"],
                view_family["minimum_elevation_degrees"],
                view_family["maximum_elevation_degrees"],
            ],
            4,
            "two-object camera-view family values",
        )
    )
    if not str(view_family["id"]).strip() or not -180.0 <= relative_azimuth <= 180.0:
        raise ValueError("two-object camera-view family azimuth is invalid")
    if not (
        0.0
        < minimum_elevation
        <= preferred_elevation
        <= maximum_elevation
        < 90.0
    ):
        raise ValueError("two-object camera elevations are invalid")
    if (
        minimum_camera_distance <= 0.0
        or minimum_camera_distance >= maximum_camera_distance
        or maximum_distance_above_minimum <= 0.0
        or maximum_distance_above_minimum >= maximum_camera_distance
    ):
        raise ValueError("two-object camera distance limits are invalid")
    if not 0.0 <= envelope_margin < 0.20:
        raise ValueError("full-motion envelope margin must lie in [0, 0.20)")
    if not 0.0 < preferred_envelope_span <= 1.0 - 2.0 * envelope_margin:
        raise ValueError("preferred full-motion envelope span is invalid")
    if not 0.0 < minimum_object_span <= preferred_envelope_span:
        raise ValueError("minimum per-object span is invalid")
    if not 0.0 < minimum_unoccluded_fraction <= 1.0:
        raise ValueError("minimum per-object unoccluded fraction is invalid")
    if not 0.0 < projected_separation_ratio <= 1.0:
        raise ValueError("projected separation ratio must lie in (0, 1]")
    if not (
        0.0 <= minimum_context_fraction <= 1.0
        and 0.0 <= minimum_anchor_fraction <= 1.0
        and 0.0 <= minimum_anchor_unoccluded_fraction <= 1.0
    ):
        raise ValueError("two-object support-context camera fractions are invalid")
    layout = str(intent.get("layout", ""))
    layout_fields = _LAYOUT_FIELDS.get(layout)
    if layout_fields is None or set(intent) != _COMMON_INTENT_FIELDS | layout_fields:
        raise ValueError("unsupported two-object motion intent fields")
    interaction_class = str(intent["interaction_class"])
    kinematic_regime = str(intent["kinematic_regime"])
    if (interaction_class, kinematic_regime) != _LAYOUT_CONTRACT[layout]:
        raise ValueError("two-object layout contradicts its interaction contract")
    return layout


def _validate_intent_kinematics(
    layout: str,
    object_motions: list[str],
    velocities: np.ndarray,
) -> None:
    """Reject motion labels that contradict the declared initial state."""

    if len(object_motions) != 2 or any(not motion for motion in object_motions):
        raise ValueError("two-object intent requires two object motions")
    horizontal_speeds = np.linalg.norm(velocities[:, :2], axis=1)
    vertical_speeds = velocities[:, 2]

    def require_supported_labels() -> None:
        if any(abs(speed) > _NUMERICAL_EPSILON for speed in vertical_speeds):
            raise ValueError("supported motion cannot start with vertical velocity")
        for motion, speed in zip(
            object_motions, horizontal_speeds, strict=True
        ):
            if motion not in {"rest", "roll_or_slide_1obj"}:
                raise ValueError("supported motion has an unsupported motion label")
            moving = float(speed) > _NUMERICAL_EPSILON
            if moving != (motion == "roll_or_slide_1obj"):
                raise ValueError("supported motion label contradicts its velocity")

    if layout in {"planned_supported_contact", "parallel_supported_independent"}:
        require_supported_labels()
        return
    if layout == "ballistic_airborne_contact":
        if object_motions[1] != "rest" or float(
            np.linalg.norm(velocities[1])
        ) > _NUMERICAL_EPSILON:
            raise ValueError("supported target must start at rest")
        airborne_motion = object_motions[0]
        if airborne_motion == "drop_fall_1obj":
            if vertical_speeds[0] > _NUMERICAL_EPSILON:
                raise ValueError("drop motion cannot start upward")
        elif airborne_motion == "arc_projectile_1obj":
            if (
                vertical_speeds[0] <= _NUMERICAL_EPSILON
                or horizontal_speeds[0] <= _NUMERICAL_EPSILON
            ):
                raise ValueError("arc projectile must start upward and forward")
        else:
            raise ValueError("unsupported airborne-contact motion label")
        return
    if layout == "ballistic_airborne_pair_contact":
        if object_motions != ["arc_projectile_1obj", "arc_projectile_1obj"]:
            raise ValueError("airborne pair must contain two arc projectiles")
        if bool(np.any(vertical_speeds <= _NUMERICAL_EPSILON)) or bool(
            np.any(horizontal_speeds <= _NUMERICAL_EPSILON)
        ):
            raise ValueError("airborne pair must start upward and forward")
        if float(np.linalg.norm(velocities[0, :2] - velocities[1, :2])) <= (
            _NUMERICAL_EPSILON
        ):
            raise ValueError("airborne pair must have horizontal closing motion")
        return
    if object_motions != ["drop_fall_1obj", "rest"] or bool(
        np.any(np.linalg.norm(velocities, axis=1) > _NUMERICAL_EPSILON)
    ):
        raise ValueError(
            "separated airborne-supported motion must start as drop and rest"
        )


def _object_geometry(
    objects: list[dict[str, Any]], shape_contract: dict[str, Any]
) -> list[dict[str, Any]]:
    if (
        set(shape_contract) != {"schema_version", "families"}
        or shape_contract.get("schema_version")
        != "physweep_two_object_shape_families_v1"
    ):
        raise ValueError("unsupported two-object shape-family contract")
    families = shape_contract.get("families")
    if not isinstance(families, list):
        raise ValueError("two-object shape families must be records")
    by_geometry = {
        str(record.get("geometry_type", "")): record
        for record in families
        if isinstance(record, dict)
    }
    if len(by_geometry) != len(families):
        raise ValueError("two-object geometry types must be unique")
    result = []
    for obj in objects:
        geometry = obj.get("geometry", {})
        shape = str(geometry.get("type", ""))
        family = by_geometry.get(shape)
        if family is None:
            raise ValueError(f"unsupported two-object geometry: {shape}")
        dimensions = positive_vector(
            geometry["size_m"], 3, f"{obj['object_id']} geometry dimensions"
        )
        result.append(
            {
                "shape": shape,
                "size_m": dimensions,
                "support_offset_m": object_contact_offset_m(shape, dimensions),
                "footprint_half_extents_m": upright_footprint_half_extents_m(
                    shape, dimensions
                ),
                "stable_pose_profile": str(family["stable_pose_profile"]),
                "supported_motion_mode": str(family["supported_motion_mode"]),
            }
        )
    return result


def _support_center(bounds: dict[str, Any]) -> np.ndarray:
    """Return the placement center of the reviewed safe support region."""

    return np.asarray(
        [
            0.5 * (float(bounds["x"][0]) + float(bounds["x"][1])),
            0.5 * (float(bounds["y"][0]) + float(bounds["y"][1])),
        ],
        dtype=np.float64,
    )


def _supported_positions_z(
    support: dict[str, Any], geometry: list[dict[str, Any]], clearance_m: float
) -> list[float]:
    surface_z = float(support["surface_center_z_m"])
    return [
        surface_z + float(record["support_offset_m"]) + clearance_m
        for record in geometry
    ]


def _planned_supported_contact(
    center: np.ndarray,
    geometry: list[dict[str, Any]],
    velocities: np.ndarray,
    intent: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    contact_time = positive_vector(
        [intent["contact_time_s"]], 1, "planned contact time"
    )[0]
    relative_approach = velocities[0, :2] - velocities[1, :2]
    approach_speed = float(np.linalg.norm(relative_approach))
    if approach_speed <= _NUMERICAL_EPSILON:
        raise ValueError("planned contact requires nonzero relative velocity")
    impact_offset = finite_vector(
        [intent["impact_offset_ratio"]], 1, "impact offset ratio"
    )[0]
    if abs(impact_offset) >= _MAXIMUM_IMPACT_OFFSET_RATIO:
        raise ValueError(
            "impact offset ratio exceeds the supported initial-state contract"
        )
    central_normal = relative_approach / approach_speed
    angle = math.asin(impact_offset)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    contact_normal = rotation @ central_normal
    center_distance = upright_pair_center_distance_m(
        str(geometry[0]["shape"]),
        list(geometry[0]["size_m"]),
        str(geometry[1]["shape"]),
        list(geometry[1]["size_m"]),
        contact_normal.tolist(),
    )
    initial_delta = (
        center_distance * contact_normal + relative_approach * contact_time
    )
    positions_xy = np.stack(
        [center - 0.5 * initial_delta, center + 0.5 * initial_delta]
    )
    return positions_xy, np.asarray(
        [contact_normal[0], contact_normal[1], 0.0], dtype=np.float64
    )


def _ballistic_airborne_contact(
    scene: dict[str, Any],
    center: np.ndarray,
    support_z: list[float],
    geometry: list[dict[str, Any]],
    velocities: np.ndarray,
    intent: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    contact_time = positive_vector(
        [intent["contact_time_s"]], 1, "ballistic contact time"
    )[0]
    elevation = math.radians(
        finite_vector(
            [intent["contact_elevation_degrees"]],
            1,
            "ballistic contact elevation",
        )[0]
    )
    if not 0.0 < elevation < math.pi / 2.0:
        raise ValueError("ballistic contact elevation must lie in (0, 90) degrees")
    contact_normal = np.asarray(
        [math.cos(elevation), 0.0, -math.sin(elevation)],
        dtype=np.float64,
    )
    object_b_initial = np.asarray(
        [center[0], center[1], support_z[1]], dtype=np.float64
    )
    object_b_contact = object_b_initial + velocities[1] * contact_time
    if str(geometry[0]["shape"]) != "sphere":
        raise ValueError("ballistic pair contact requires a sphere as object_a")
    center_distance = sphere_primitive_center_distance_m(
        list(geometry[0]["size_m"]),
        str(geometry[1]["shape"]),
        list(geometry[1]["size_m"]),
        contact_normal.tolist(),
    )
    object_a_contact = object_b_contact - center_distance * contact_normal
    gravity = np.asarray(
        finite_vector(
            scene["simulation"]["world"]["gravity_m_s2"],
            3,
            "simulation gravity",
        ),
        dtype=np.float64,
    )
    object_a_initial = (
        object_a_contact
        - velocities[0] * contact_time
        - 0.5 * gravity * contact_time**2
    )
    positions = np.stack([object_a_initial, object_b_initial])
    return positions, contact_normal


def _ballistic_airborne_pair_contact(
    scene: dict[str, Any],
    center: np.ndarray,
    geometry: list[dict[str, Any]],
    velocities: np.ndarray,
    intent: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if any(str(record["shape"]) != "sphere" for record in geometry):
        raise ValueError("airborne pair contact requires two spheres")
    contact_time = positive_vector(
        [intent["contact_time_s"]], 1, "airborne pair contact time"
    )[0]
    contact_height = positive_vector(
        [intent["contact_center_height_above_support_m"]],
        1,
        "airborne pair contact height",
    )[0]
    relative_approach = velocities[0] - velocities[1]
    approach_speed = float(np.linalg.norm(relative_approach))
    if approach_speed <= _NUMERICAL_EPSILON:
        raise ValueError("airborne pair contact requires nonzero relative velocity")
    contact_normal = relative_approach / approach_speed
    center_distance = sphere_primitive_center_distance_m(
        list(geometry[0]["size_m"]),
        "sphere",
        list(geometry[1]["size_m"]),
        contact_normal.tolist(),
    )
    support_surface_z = float(
        scene["simulation"]["support"]["surface_center_z_m"]
    )
    contact_center = np.asarray(
        [center[0], center[1], support_surface_z + contact_height],
        dtype=np.float64,
    )
    contact_positions = np.stack(
        [
            contact_center - 0.5 * center_distance * contact_normal,
            contact_center + 0.5 * center_distance * contact_normal,
        ]
    )
    gravity = np.asarray(
        finite_vector(
            scene["simulation"]["world"]["gravity_m_s2"],
            3,
            "simulation gravity",
        ),
        dtype=np.float64,
    )
    initial_positions = (
        contact_positions
        - velocities * contact_time
        - 0.5 * gravity[None, :] * contact_time**2
    )
    minimum_bottom = min(
        float(initial_positions[index, 2])
        - float(geometry[index]["support_offset_m"])
        for index in range(2)
    )
    if minimum_bottom <= support_surface_z:
        raise ValueError("airborne pair initial state intersects the support")
    return initial_positions, contact_normal


def _parallel_supported_independent(
    center: np.ndarray,
    support_z: list[float],
    geometry: list[dict[str, Any]],
    intent: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    offsets_x = finite_vector(
        intent["initial_x_offset_m"], 2, "independent x offsets"
    )
    clearance = positive_vector(
        [intent["lateral_clearance_m"]], 1, "independent lateral clearance"
    )[0]
    separation_y = sum(
        float(record["footprint_half_extents_m"][1]) for record in geometry
    ) + clearance
    positions = np.asarray(
        [
            [center[0] + offsets_x[0], center[1] - 0.5 * separation_y, support_z[0]],
            [center[0] + offsets_x[1], center[1] + 0.5 * separation_y, support_z[1]],
        ],
        dtype=np.float64,
    )
    return positions, np.asarray([1.0, 0.0, 0.0], dtype=np.float64)


def _separated_airborne_supported_independent(
    center: np.ndarray,
    support_z: list[float],
    geometry: list[dict[str, Any]],
    intent: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    height = positive_vector(
        [intent["airborne_surface_gap_m"]],
        1,
        "independent airborne height",
    )[0]
    clearance = positive_vector(
        [intent["lateral_clearance_m"]], 1, "independent lateral clearance"
    )[0]
    separation_y = sum(
        float(record["footprint_half_extents_m"][1]) for record in geometry
    ) + clearance
    positions = np.asarray(
        [
            [center[0], center[1] - 0.5 * separation_y, support_z[0] + height],
            [center[0], center[1] + 0.5 * separation_y, support_z[1]],
        ],
        dtype=np.float64,
    )
    return positions, np.asarray([0.0, 1.0, 0.0], dtype=np.float64)


def _inside_support(
    positions: np.ndarray,
    geometry: list[dict[str, Any]],
    bounds: dict[str, Any],
) -> None:
    for index, position in enumerate(positions):
        half_x, half_y = geometry[index]["footprint_half_extents_m"]
        if (
            position[0]
            < float(bounds["x"][0]) + half_x + _SUPPORT_EDGE_MARGIN_M
            or position[0]
            > float(bounds["x"][1]) - half_x - _SUPPORT_EDGE_MARGIN_M
            or position[1]
            < float(bounds["y"][0]) + half_y + _SUPPORT_EDGE_MARGIN_M
            or position[1]
            > float(bounds["y"][1]) - half_y - _SUPPORT_EDGE_MARGIN_M
        ):
            raise ValueError("host support is too small for the two-object layout")


def _trajectory_angle_degrees(velocities: np.ndarray) -> float | None:
    speeds = np.linalg.norm(velocities[:, :2], axis=1)
    if bool(np.any(speeds <= _NUMERICAL_EPSILON)):
        return None
    cosine = float(
        np.clip(
            velocities[0, :2] @ velocities[1, :2] / (speeds[0] * speeds[1]),
            -1.0,
            1.0,
        )
    )
    return round(math.degrees(math.acos(cosine)), 6)


def apply_two_object_motion(
    pair_scene: dict[str, Any],
    shape_contract: dict[str, Any],
    shared: dict[str, Any],
    observation: dict[str, Any],
    view_family: dict[str, Any],
    intent: dict[str, Any],
) -> dict[str, Any]:
    """Apply one declared initial-state intent without prescribing its outcome."""

    layout = _validate_contracts(shared, observation, view_family, intent)
    scene = copy.deepcopy(pair_scene)
    objects = scene.get("simulation", {}).get("objects", [])
    if len(objects) != 2 or any(not isinstance(obj, dict) for obj in objects):
        raise ValueError("two-object motion requires one ordered object pair")
    object_ids = [str(obj.get("object_id", "")) for obj in objects]
    if any(not object_id for object_id in object_ids) or len(set(object_ids)) != 2:
        raise ValueError("two-object motion ids must be unique")
    geometry = _object_geometry(objects, shape_contract)
    support = scene["simulation"]["support"]
    if abs(float(support["surface_frame"]["slope_angle_degrees"])) > 1.0e-8:
        raise ValueError("the initial two-object matrix requires a flat support")
    bounds = support["safe_surface_bounds"]
    center = _support_center(bounds)
    support_clearance = positive_vector(
        [shared["initial_support_clearance_m"]],
        1,
        "two-object initial support clearance",
    )[0]
    support_z = _supported_positions_z(support, geometry, support_clearance)
    velocities = np.asarray(
        [
            finite_vector(value, 3, f"{object_ids[index]} linear velocity")
            for index, value in enumerate(intent["linear_velocity_m_s"])
        ],
        dtype=np.float64,
    )
    if velocities.shape != (2, 3):
        raise ValueError("two-object intent requires two linear velocities")
    object_motions = [str(value) for value in intent["object_motions"]]
    _validate_intent_kinematics(layout, object_motions, velocities)

    if layout == "planned_supported_contact":
        positions_xy, approach_axis = _planned_supported_contact(
            center, geometry, velocities, intent
        )
        positions = np.column_stack([positions_xy, support_z])
    elif layout == "ballistic_airborne_contact":
        positions, approach_axis = _ballistic_airborne_contact(
            scene, center, support_z, geometry, velocities, intent
        )
    elif layout == "ballistic_airborne_pair_contact":
        positions, approach_axis = _ballistic_airborne_pair_contact(
            scene, center, geometry, velocities, intent
        )
    elif layout == "parallel_supported_independent":
        positions, approach_axis = _parallel_supported_independent(
            center, support_z, geometry, intent
        )
    else:
        positions, approach_axis = _separated_airborne_supported_independent(
            center, support_z, geometry, intent
        )
    if layout in {
        "planned_supported_contact",
        "ballistic_airborne_contact",
        "ballistic_airborne_pair_contact",
    }:
        contact_time = float(intent["contact_time_s"])
        planned_contact_positions = positions + velocities * contact_time
        if layout == "ballistic_airborne_contact":
            gravity = np.asarray(
                scene["simulation"]["world"]["gravity_m_s2"],
                dtype=np.float64,
            )
            planned_contact_positions[0] += 0.5 * gravity * contact_time**2
        elif layout == "ballistic_airborne_pair_contact":
            gravity = np.asarray(
                scene["simulation"]["world"]["gravity_m_s2"],
                dtype=np.float64,
            )
            planned_contact_positions += 0.5 * gravity[None, :] * contact_time**2
        path_xy = np.vstack(
            [positions[:, :2], planned_contact_positions[:, :2]]
        )
        envelope_center = 0.5 * (path_xy.min(axis=0) + path_xy.max(axis=0))
        translation = center - envelope_center
        positions[:, :2] += translation
        planned_contact_positions[:, :2] += translation
        _inside_support(positions, geometry, bounds)
        _inside_support(planned_contact_positions, geometry, bounds)
    else:
        _inside_support(positions, geometry, bounds)

    contact_friction, contact_restitution, minimum_support_fraction = finite_vector(
        [
            shared["contact_friction"],
            shared["contact_restitution"],
            shared["minimum_support_contact_fraction"],
        ],
        3,
        "two-object shared physics values",
    )
    if contact_friction < 0.0 or not 0.0 <= contact_restitution <= 1.0:
        raise ValueError("two-object contact parameters are invalid")
    if not 0.0 < minimum_support_fraction <= 1.0:
        raise ValueError("support contact fraction must lie in (0, 1]")
    minimum_displacements = finite_vector(
        intent["minimum_displacement_m"], 2, "object displacement thresholds"
    )
    if any(value < 0.0 for value in minimum_displacements):
        raise ValueError("object displacement thresholds must be nonnegative")
    for motion, minimum_displacement in zip(
        object_motions, minimum_displacements, strict=True
    ):
        if (motion == "rest") != (
            abs(minimum_displacement) <= _NUMERICAL_EPSILON
        ):
            raise ValueError(
                "motion label contradicts its minimum displacement threshold"
            )
    supported_by_regime = {
        "supported_supported": [True, True],
        "airborne_supported": [False, True],
        "airborne_airborne": [False, False],
    }
    supported = supported_by_regime[str(intent["kinematic_regime"])]

    interaction_class = str(intent["interaction_class"])
    contact_requirement = (
        "must_contact" if interaction_class == "interacting" else "must_not_contact"
    )
    for index, obj in enumerate(objects):
        material = copy.deepcopy(obj["material"])
        material["contact_friction"] = contact_friction
        material["contact_restitution"] = contact_restitution
        obj["material"] = material
        velocity = velocities[index]
        rolling = (
            supported[index]
            and geometry[index]["supported_motion_mode"] == "rolling"
        )
        angular = (
            np.asarray(
                [
                    -velocity[1] / float(geometry[index]["support_offset_m"]),
                    velocity[0] / float(geometry[index]["support_offset_m"]),
                    0.0,
                ],
                dtype=np.float64,
            )
            if rolling
            else np.zeros(3, dtype=np.float64)
        )
        obj["initial_state"] = {
            "pose_profile": (
                str(geometry[index]["stable_pose_profile"])
                if supported[index]
                else "airborne"
            ),
            "position_m": positions[index].tolist(),
            "orientation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "linear_velocity_m_s": velocity.tolist(),
            "angular_velocity_rad_s": angular.tolist(),
        }
        expected = {
            "motion_family": object_motions[index],
            "contact_mode": (
                "supported_pair_motion" if supported[index] else "airborne_pair_motion"
            ),
            "must_contact_primary_support": supported[index],
            "minimum_displacement_m": minimum_displacements[index],
            "minimum_support_contact_fraction": (
                minimum_support_fraction
                if supported[index] and interaction_class == "independent"
                else 0.0
            ),
        }
        relation_key = (
            "required_object_contact_id"
            if interaction_class == "interacting"
            else "forbidden_object_contact_id"
        )
        expected[relation_key] = object_ids[1 - index]
        obj["expected_motion"] = expected

    motion_id = str(intent["id"])
    scene["scene_id"] = f"{scene['scene_id']}__{motion_id}"
    scene["semantic_sampling"]["five_dimensions"]["motion"] = {
        "family": motion_id,
        "subtype": layout,
        "interaction_class": interaction_class,
        "kinematic_regime": str(intent["kinematic_regime"]),
        "object_motions": dict(zip(object_ids, object_motions)),
        "trajectory_angle_degrees": _trajectory_angle_degrees(velocities),
        "impact_offset_ratio": float(intent.get("impact_offset_ratio", 0.0)),
    }
    scene["simulation"]["interaction"] = {
        "type": (
            "pairwise_collision"
            if interaction_class == "interacting"
            else "pairwise_independent"
        ),
        "interaction_class": interaction_class,
        "motion_pattern": motion_id,
        "kinematic_regime": str(intent["kinematic_regime"]),
        "contact_requirement": contact_requirement,
        "object_ids": object_ids,
        "approach_axis_xyz": approach_axis.tolist(),
        "impact_offset_ratio": float(intent.get("impact_offset_ratio", 0.0)),
        "camera_view_family_id": str(view_family["id"]),
        "camera_relative_azimuth_degrees": float(
            view_family["relative_azimuth_degrees"]
        ),
        "preferred_camera_elevation_degrees": float(
            view_family["preferred_elevation_degrees"]
        ),
        "minimum_camera_elevation_degrees": float(
            view_family["minimum_elevation_degrees"]
        ),
        "maximum_camera_elevation_degrees": float(
            view_family["maximum_elevation_degrees"]
        ),
        **copy.deepcopy(shared["interaction_audit"]),
        **copy.deepcopy(observation),
    }
    envelope_margin = float(observation["full_motion_envelope_margin_ndc"])
    maximum_envelope_span = 1.0 - 2.0 * envelope_margin
    scene["camera_request"] = {
        "schema_version": "physweep_two_object_camera_request_v1",
        "profile": str(observation["solver_profile_template_id"]),
        "observation": {
            "version": "physweep_two_object_camera_observation_request_v1",
            "intent": "joint_full_motion_envelope",
            "focus_event": {"type": "fraction", "fraction": 1.0},
            "structure_context": "horizontal_surface",
            "preferred_object_span_ndc": float(
                observation["preferred_full_motion_envelope_span_ndc"]
            ),
            "minimum_median_object_span_ndc": float(
                observation["minimum_per_object_median_span_ndc"]
            ),
            "minimum_anchor_visible_fraction": float(
                observation["minimum_support_anchor_visible_fraction"]
            ),
            "minimum_anchor_unoccluded_fraction": float(
                observation["minimum_support_anchor_unoccluded_fraction"]
            ),
        },
        "focal_length_mm": float(observation["focal_length_mm"]),
        "minimum_full_trajectory_center_visible_fraction": 1.0,
        "minimum_primary_trajectory_center_visible_fraction": 1.0,
        "full_trajectory_camera_target_blend": 1.0,
        "minimum_initial_object_span_ndc": float(
            observation["minimum_per_object_median_span_ndc"]
        ),
        "minimum_initial_object_visible_fraction": 1.0,
        "initial_object_center_margin_ndc": envelope_margin,
        "initial_object_corner_margin_ndc": envelope_margin,
        "maximum_initial_object_span_ndc": maximum_envelope_span,
        "minimum_support_context_visible_fraction": float(
            observation["minimum_support_context_visible_fraction"]
        ),
        "minimum_primary_trajectory_unoccluded_fraction": 0.0,
        "minimum_full_trajectory_unoccluded_fraction": 0.0,
        "minimum_camera_distance_floor_m": float(
            observation["minimum_camera_distance_m"]
        ),
        "minimum_camera_distance_offset_m": 0.0,
        "minimum_camera_distance_support_diagonal_scale": 0.0,
        "preferred_camera_distance_offset_m": 0.20,
        "camera_distance_penalty_weight": 0.08,
        "maximum_camera_distance_above_minimum_m": float(
            observation["maximum_camera_distance_above_minimum_m"]
        ),
        "minimum_camera_elevation_degrees": float(
            view_family["minimum_elevation_degrees"]
        ),
        "maximum_camera_elevation_degrees": float(
            view_family["maximum_elevation_degrees"]
        ),
        "soft_maximum_focus_span_ndc": float(
            observation["preferred_full_motion_envelope_span_ndc"]
        ),
        "maximum_focus_span_ndc": maximum_envelope_span,
        "focus_span_penalty_weight": 0.12,
        "maximum_camera_distance_m": float(
            observation["maximum_camera_distance_m"]
        ),
        "allow_partial_exit": False,
    }
    return scene
