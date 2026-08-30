"""Initial-state builders for the bounded two-object motion matrix."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np

from tools.core.rigid_geometry import finite_vector, positive_vector


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
}
_OBSERVATION_FIELDS = {
    "schema_version",
    "maximum_camera_side_deviation_degrees",
    "preferred_camera_elevation_degrees",
    "minimum_camera_elevation_degrees",
    "maximum_camera_elevation_degrees",
    "maximum_camera_distance_m",
    "maximum_camera_distance_above_minimum_m",
    "full_motion_envelope_margin_ndc",
    "preferred_full_motion_envelope_span_ndc",
    "minimum_per_object_median_span_ndc",
    "minimum_per_object_unoccluded_fraction",
    "minimum_pair_keyframe_projected_center_separation_to_radius_sum_ratio",
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
    intent: dict[str, Any],
) -> str:
    if set(shared) != _SHARED_FIELDS or (
        shared.get("schema_version")
        != "physweep_two_object_shared_physics_v1"
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
        ],
        3,
        "two-object interaction audit values",
    )
    if set(observation) != _OBSERVATION_FIELDS:
        raise ValueError("two-object observation fields are incomplete")
    if (
        observation.get("schema_version")
        != "physweep_two_object_camera_observation_v1"
    ):
        raise ValueError("unsupported two-object camera observation")
    (
        camera_side_deviation,
        preferred_elevation,
        minimum_elevation,
        maximum_elevation,
        maximum_camera_distance,
        maximum_distance_above_minimum,
        envelope_margin,
        preferred_envelope_span,
        minimum_object_span,
        minimum_unoccluded_fraction,
        projected_separation_ratio,
    ) = finite_vector(
        [
            observation["maximum_camera_side_deviation_degrees"],
            observation["preferred_camera_elevation_degrees"],
            observation["minimum_camera_elevation_degrees"],
            observation["maximum_camera_elevation_degrees"],
            observation["maximum_camera_distance_m"],
            observation["maximum_camera_distance_above_minimum_m"],
            observation["full_motion_envelope_margin_ndc"],
            observation["preferred_full_motion_envelope_span_ndc"],
            observation["minimum_per_object_median_span_ndc"],
            observation["minimum_per_object_unoccluded_fraction"],
            observation[
                "minimum_pair_keyframe_projected_center_separation_to_radius_sum_ratio"
            ],
        ],
        11,
        "two-object observation values",
    )
    if not 0.0 < camera_side_deviation <= 45.0:
        raise ValueError("camera side deviation must lie in (0, 45] degrees")
    if not (
        0.0
        < minimum_elevation
        <= preferred_elevation
        <= maximum_elevation
        < 90.0
    ):
        raise ValueError("two-object camera elevations are invalid")
    if (
        maximum_camera_distance <= 0.0
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
    layout = str(intent.get("layout", ""))
    layout_fields = _LAYOUT_FIELDS.get(layout)
    if layout_fields is None or set(intent) != _COMMON_INTENT_FIELDS | layout_fields:
        raise ValueError("unsupported two-object motion intent fields")
    interaction_class = str(intent["interaction_class"])
    kinematic_regime = str(intent["kinematic_regime"])
    if (interaction_class, kinematic_regime) != _LAYOUT_CONTRACT[layout]:
        raise ValueError("two-object layout contradicts its interaction contract")
    return layout


def _sphere_radii(objects: list[dict[str, Any]]) -> list[float]:
    radii = []
    for obj in objects:
        geometry = obj.get("geometry", {})
        if geometry.get("type") != "sphere":
            raise ValueError("the initial two-object matrix requires sphere candidates")
        dimensions = positive_vector(
            geometry["size_m"], 3, f"{obj['object_id']} sphere dimensions"
        )
        if max(dimensions) - min(dimensions) > 1.0e-8:
            raise ValueError("sphere candidate dimensions must be isotropic")
        radii.append(0.5 * dimensions[0])
    return radii


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
    support: dict[str, Any], radii: list[float], clearance_m: float
) -> list[float]:
    surface_z = float(support["surface_center_z_m"])
    return [surface_z + radius + clearance_m for radius in radii]


def _planned_supported_contact(
    center: np.ndarray,
    radii: list[float],
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
    center_distance = radii[0] + radii[1]
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
    radii: list[float],
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
    object_a_contact = object_b_contact - sum(radii) * contact_normal
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


def _parallel_supported_independent(
    center: np.ndarray,
    support_z: list[float],
    radii: list[float],
    intent: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    offsets_x = finite_vector(
        intent["initial_x_offset_m"], 2, "independent x offsets"
    )
    clearance = positive_vector(
        [intent["lateral_clearance_m"]], 1, "independent lateral clearance"
    )[0]
    separation_y = sum(radii) + clearance
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
    radii: list[float],
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
    separation_y = sum(radii) + clearance
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
    radii: list[float],
    bounds: dict[str, Any],
) -> None:
    for index, position in enumerate(positions):
        radius = radii[index]
        if (
            position[0]
            < float(bounds["x"][0]) + radius + _SUPPORT_EDGE_MARGIN_M
            or position[0]
            > float(bounds["x"][1]) - radius - _SUPPORT_EDGE_MARGIN_M
            or position[1]
            < float(bounds["y"][0]) + radius + _SUPPORT_EDGE_MARGIN_M
            or position[1]
            > float(bounds["y"][1]) - radius - _SUPPORT_EDGE_MARGIN_M
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
    shared: dict[str, Any],
    observation: dict[str, Any],
    intent: dict[str, Any],
) -> dict[str, Any]:
    """Apply one declared initial-state intent without prescribing its outcome."""

    layout = _validate_contracts(shared, observation, intent)
    scene = copy.deepcopy(pair_scene)
    objects = scene.get("simulation", {}).get("objects", [])
    if len(objects) != 2 or any(not isinstance(obj, dict) for obj in objects):
        raise ValueError("two-object motion requires one ordered object pair")
    object_ids = [str(obj.get("object_id", "")) for obj in objects]
    if any(not object_id for object_id in object_ids) or len(set(object_ids)) != 2:
        raise ValueError("two-object motion ids must be unique")
    radii = _sphere_radii(objects)
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
    support_z = _supported_positions_z(support, radii, support_clearance)
    velocities = np.asarray(
        [
            finite_vector(value, 3, f"{object_ids[index]} linear velocity")
            for index, value in enumerate(intent["linear_velocity_m_s"])
        ],
        dtype=np.float64,
    )
    if velocities.shape != (2, 3):
        raise ValueError("two-object intent requires two linear velocities")

    if layout == "planned_supported_contact":
        positions_xy, approach_axis = _planned_supported_contact(
            center, radii, velocities, intent
        )
        positions = np.column_stack([positions_xy, support_z])
    elif layout == "ballistic_airborne_contact":
        positions, approach_axis = _ballistic_airborne_contact(
            scene, center, support_z, radii, velocities, intent
        )
    elif layout == "parallel_supported_independent":
        positions, approach_axis = _parallel_supported_independent(
            center, support_z, radii, intent
        )
    else:
        positions, approach_axis = _separated_airborne_supported_independent(
            center, support_z, radii, intent
        )
    if layout in {"planned_supported_contact", "ballistic_airborne_contact"}:
        contact_time = float(intent["contact_time_s"])
        planned_contact_positions = positions + velocities * contact_time
        if layout == "ballistic_airborne_contact":
            gravity = np.asarray(
                scene["simulation"]["world"]["gravity_m_s2"],
                dtype=np.float64,
            )
            planned_contact_positions[0] += 0.5 * gravity * contact_time**2
        path_xy = np.vstack(
            [positions[:, :2], planned_contact_positions[:, :2]]
        )
        envelope_center = 0.5 * (path_xy.min(axis=0) + path_xy.max(axis=0))
        translation = center - envelope_center
        positions[:, :2] += translation
        planned_contact_positions[:, :2] += translation
        _inside_support(positions, radii, bounds)
        _inside_support(planned_contact_positions, radii, bounds)
    else:
        _inside_support(positions, radii, bounds)

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
    object_motions = [str(value) for value in intent["object_motions"]]
    if len(object_motions) != 2 or any(not value for value in object_motions):
        raise ValueError("two-object intent requires two object motions")
    supported = (
        [True, True]
        if intent["kinematic_regime"] == "supported_supported"
        else [False, True]
    )

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
        angular = (
            np.asarray(
                [-velocity[1] / radii[index], velocity[0] / radii[index], 0.0],
                dtype=np.float64,
            )
            if supported[index]
            else np.zeros(3, dtype=np.float64)
        )
        obj["initial_state"] = {
            "pose_profile": "support_normal" if supported[index] else "airborne",
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
        **copy.deepcopy(shared["interaction_audit"]),
        **copy.deepcopy(observation),
    }
    return scene
