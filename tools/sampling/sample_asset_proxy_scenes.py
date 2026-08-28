#!/usr/bin/env python3
"""Sample and simulate deterministic scenes using only admitted asset proxies."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from tools.assets.physical_proxy_catalog import load_catalog, records_by_id
from tools.assets.scene_kit_compiler import validate_registry_counts
from tools.assets.static_support_proxy import compile_static_support_binding
from tools.assets.visual_environment_binding import choose_environment
from tools.core.hashing import relative_file_binding as file_binding
from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json
from tools.core.paths import resolve_project_path as project_path
from tools.dataset_contract.immutable_scene_contract import (
    freeze_metadata,
    write_simulation_record,
)
from tools.dataset_contract.object_identity_contract import attach_object_identity
from tools.motion_rules.one_object import asset_motion_group
from tools.physics.asset_proxy_simulation import (
    ASSET_AUDIT_VERSION,
    asset_motion_usefulness,
    local_bounds,
    quaternion,
    simulate_scene,
)
from tools.physics.physics_time_step import simulation_hz_for_min_extent


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HDRI_MANIFEST_PATH = PROJECT_ROOT / "assets/manifests/hdri_admission.json"
BACKEND_PATH = PROJECT_ROOT / "configs/pybullet_backend.json"
VISUAL_RULES_PATH = PROJECT_ROOT / "configs/visual_sampling.json"
CAMERA_RULES_PATH = PROJECT_ROOT / "configs/one_object_sampling_rules.json"
DEFAULT_REGISTRY_PATH = Path("configs/asset_proxy_registry.json")
DEFAULT_CATALOG_PATH = Path("assets/proxies/catalog.json")
DEFAULT_SEMANTIC_RULES_PATH = Path("configs/asset_semantic_scene_rules.json")
DEFAULT_COMPOSITION_RULES_PATH = Path("configs/asset_scene_composition.json")


def asset_camera_observation(
    camera_rules: dict[str, Any], profile: str
) -> dict[str, Any]:
    try:
        motion = camera_rules["asset_profile_motion_intents"][profile]
        observation = copy.deepcopy(camera_rules["motion_intents"][motion])
        context = camera_rules["structure_contexts"][
            observation["structure_context"]
        ]
    except KeyError as error:
        raise ValueError(
            f"asset motion profile has no camera observation mapping: {profile}"
        ) from error
    if observation["focus_event"]["type"] == "required_motion_collider":
        raise ValueError("asset camera observation cannot use an unresolved collider")
    if observation["focus_event"]["type"] == "transition_destination_contact":
        if profile != "edge_exit":
            raise ValueError(
                "asset transition destination is undefined for this motion profile"
            )
        observation["focus_event"]["type"] = "collider_contact"
        observation["focus_event"]["collider_id"] = "environment_floor"
    observation["minimum_anchor_visible_fraction"] = float(
        context["minimum_anchor_visible_fraction"]
    )
    return {"version": str(camera_rules["version"]), **observation}


def proxy_volume_fill_ratio(record: dict[str, Any]) -> float:
    """Return analytic collider volume divided by its compound AABB volume."""

    solid_volume = 0.0
    for collider in record["proxy"]["colliders"]:
        shape = str(collider["shape"])
        size = [float(value) for value in collider["size_m"]]
        if shape == "box":
            solid_volume += math.prod(size)
        elif shape == "sphere":
            radius = 0.5 * max(size)
            solid_volume += 4.0 * math.pi * radius**3 / 3.0
        elif shape == "cylinder":
            radius = 0.5 * max(size[:2])
            solid_volume += math.pi * radius**2 * size[2]
        else:
            raise ValueError(f"unsupported proxy shape: {shape}")
    low, high = local_bounds(record)
    bounds_volume = float(np.prod(high - low))
    if bounds_volume <= 0.0:
        raise ValueError(f"non-positive proxy bounds volume: {record['asset_id']}")
    return min(1.0, solid_volume / bounds_volume)


def choose_mass(record: dict[str, Any], rng: random.Random) -> float:
    low, high = [float(value) for value in record["proxy"]["mass_range_kg"]]
    return round(math.exp(rng.uniform(math.log(low), math.log(high))), 6)


def place_static_prop(
    record: dict[str, Any],
    surface: dict[str, Any],
    rng: random.Random,
    placement_rules: dict[str, Any],
) -> dict[str, Any] | None:
    low, high = local_bounds(record)
    extent = high - low
    sx, sy = [float(value) for value in surface["size_xy_m"]]
    yaw_degrees = rng.uniform(
        *[float(value) for value in placement_rules["static_prop_yaw_degrees"]]
    )
    yaw_radians = math.radians(yaw_degrees)
    signed_cosine = math.cos(yaw_radians)
    signed_sine = math.sin(yaw_radians)
    cosine = abs(signed_cosine)
    sine = abs(signed_sine)
    oriented_half_x = 0.5 * (
        cosine * float(extent[0]) + sine * float(extent[1])
    )
    oriented_half_y = 0.5 * (
        sine * float(extent[0]) + cosine * float(extent[1])
    )
    support_margin = float(
        placement_rules["static_prop_support_margin_m"]
    )
    if (
        oriented_half_x + support_margin >= 0.5 * sx
        or oriented_half_y + support_margin >= 0.5 * sy
    ):
        return None
    side = rng.choice([-1.0, 1.0])
    center_x, center_y = [float(value) for value in surface["center_xy_m"]]
    x_limit = 0.5 * sx - oriented_half_x - support_margin
    aabb_center_x = center_x + rng.uniform(-0.20, 0.20) * x_limit
    y_limit = 0.5 * sy - oriented_half_y - support_margin
    aabb_center_y = center_y + side * y_limit
    local_center_x = 0.5 * float(low[0] + high[0])
    local_center_y = 0.5 * float(low[1] + high[1])
    center_offset_x = (
        signed_cosine * local_center_x - signed_sine * local_center_y
    )
    center_offset_y = (
        signed_sine * local_center_x + signed_cosine * local_center_y
    )
    x = aabb_center_x - center_offset_x
    y = aabb_center_y - center_offset_y
    oriented_extent = [
        2.0 * oriented_half_x,
        2.0 * oriented_half_y,
        float(extent[2]),
    ]
    yaw_degrees = round(yaw_degrees, 4)
    if y_limit <= 0.0:
        raise ValueError(
            f"static prop has no valid support-edge placement: {record['asset_id']}"
        )
    binding = {
        "asset_id": record["asset_id"],
        "name": record["name"],
        "position_m": [round(x, 6), round(y, 6), float(surface["z_m"])],
        "world_aabb_center_xy_m": [
            round(aabb_center_x, 6),
            round(aabb_center_y, 6),
        ],
        "yaw_degrees": yaw_degrees,
        "world_aabb_extent_m": [
            round(float(value), 6) for value in oriented_extent
        ],
        "source_local_extent_m": [
            round(float(value), 6) for value in extent
        ],
        "placement_rule": "support_edge_oriented_aabb_v2",
    }
    variants = record["visual"].get("variant_object_names", [])
    if variants:
        binding["visual_object_names"] = [str(rng.choice(variants))]
    return binding


def motion_initial_state(
    profile: str,
    surface: dict[str, Any],
    bounds_low: np.ndarray,
    bounds_high: np.ndarray,
    prop: dict[str, Any] | None,
    rng: random.Random,
    rules: dict[str, Any],
    dynamic_friction: float,
    support_friction: float,
    gravity_m_s2: float,
    proxy_motion_class: str,
    physical_support_size_xy_m: list[float] | None = None,
    interaction_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    asset_motion_group(profile)
    interaction_policy = interaction_policy or {}
    placement = rules["placement"]
    profile_rules = rules["motion_profiles"][profile]
    sx, sy = [float(value) for value in surface["size_xy_m"]]
    cx, cy = [float(value) for value in surface["center_xy_m"]]
    half_x = max(abs(float(bounds_low[0])), abs(float(bounds_high[0])))
    half_y = max(abs(float(bounds_low[1])), abs(float(bounds_high[1])))
    half_z = max(abs(float(bounds_low[2])), abs(float(bounds_high[2])))
    edge_exit_direction: np.ndarray | None = None
    if profile == "edge_exit":
        directions = interaction_policy.get("clear_exit_directions_xy", [])
        if not directions:
            raise ValueError(
                "edge_exit requires an asset-reviewed clear_exit_directions_xy policy"
            )
        edge_exit_direction = np.asarray(rng.choice(directions), dtype=np.float64)
        direction_norm = float(np.linalg.norm(edge_exit_direction))
        if edge_exit_direction.shape != (2,) or direction_norm <= 1.0e-8:
            raise ValueError(
                f"invalid edge_exit direction: {edge_exit_direction.tolist()}"
            )
        edge_exit_direction /= direction_norm
        direction_degrees = math.degrees(
            math.atan2(edge_exit_direction[1], edge_exit_direction[0])
        )
        aligned_yaw = (
            direction_degrees
            if half_x <= half_y
            else direction_degrees - 90.0
        )
        yaw_candidates = [((aligned_yaw + 180.0) % 360.0) - 180.0]
    else:
        yaw_candidates = [
            rng.uniform(*[float(value) for value in placement["yaw_degrees"]])
        ]
    if prop and profile == "edge_exit":
        raise ValueError("edge_exit does not allow a static prop in its exit lane")
    if prop:
        yaw_candidates.extend(
            rng.uniform(*[float(value) for value in placement["yaw_degrees"]])
            for _ in range(int(placement["prop_lane_yaw_attempts"]) - 1)
        )
        yaw_candidates.append(0.0)
        prop_center_y = float(
            prop.get("world_aabb_center_xy_m", prop["position_m"][:2])[1]
        )
        prop_side = math.copysign(1.0, prop_center_y - cy or 1.0)
        prop_half_y = 0.5 * float(prop["world_aabb_extent_m"][1])
    minimum_clearance = float(placement["prop_lane_minimum_clearance_m"])
    selected_geometry = None
    best_geometry = None
    for yaw_candidate in yaw_candidates:
        yaw_radians = math.radians(yaw_candidate)
        cosine = abs(math.cos(yaw_radians))
        sine = abs(math.sin(yaw_radians))
        candidate_half_x = cosine * half_x + sine * half_y
        candidate_half_y = sine * half_x + cosine * half_y
        candidate_safe_x = max(
            float(placement["minimum_safe_half_extent_x_m"]),
            0.5 * sx
            - candidate_half_x
            - float(placement["support_margin_x_m"]),
        )
        candidate_safe_y = max(
            float(placement["minimum_safe_half_extent_y_m"]),
            0.5 * sy
            - candidate_half_y
            - float(placement["support_margin_y_m"]),
        )
        candidate_lane_y = cy
        candidate_clearance = None
        if prop:
            candidate_safe_y = (
                0.5 * sy
                - candidate_half_y
                - float(placement["prop_lane_support_margin_m"])
            )
            if candidate_safe_y > 0.0:
                candidate_lane_y = cy - prop_side * candidate_safe_y
                prop_inner_edge = prop_center_y - prop_side * prop_half_y
                dynamic_inner_edge = (
                    candidate_lane_y + prop_side * candidate_half_y
                )
                candidate_clearance = prop_side * (
                    prop_inner_edge - dynamic_inner_edge
                )
        geometry = (
            yaw_candidate,
            candidate_half_x,
            candidate_half_y,
            candidate_safe_x,
            candidate_safe_y,
            candidate_lane_y,
            candidate_clearance,
        )
        if best_geometry is None or (
            candidate_clearance
            if candidate_clearance is not None
            else float("-inf")
        ) > (
            best_geometry[-1]
            if best_geometry[-1] is not None
            else float("-inf")
        ):
            best_geometry = geometry
        if not prop or (
            candidate_clearance is not None
            and candidate_clearance >= minimum_clearance
        ):
            selected_geometry = geometry
            break
    if selected_geometry is None:
        assert best_geometry is not None
        (
            yaw,
            oriented_half_x,
            oriented_half_y,
            safe_x,
            safe_y,
            lane_y,
            prop_lane_clearance,
        ) = best_geometry
        best_clearance = (
            float(prop_lane_clearance)
            if prop_lane_clearance is not None
            else float("-inf")
        )
        raise ValueError(
            f"{profile} best prop lane clearance "
            f"{best_clearance:.4f} m is below "
            f"{minimum_clearance:.4f} m "
            f"(support_y={sy:.4f}, prop_half_y={prop_half_y:.4f}, "
            f"dynamic_half_y={oriented_half_y:.4f}, "
            f"prop_center_y={prop_center_y:.4f}, lane_y={lane_y:.4f})"
        )
    (
        yaw,
        oriented_half_x,
        oriented_half_y,
        safe_x,
        safe_y,
        lane_y,
        prop_lane_clearance,
    ) = selected_geometry
    support_clearance = float(
        profile_rules.get(
            "support_clearance_m", placement["support_clearance_m"]
        )
    )
    rest_z = (
        float(surface["z_m"])
        + half_z
        + support_clearance
    )
    angular = [0.0, 0.0, 0.0]
    calculation: dict[str, Any] | None = None

    def coast_adjusted_velocity(
        sampled_velocity_xy: list[float],
        position_xy: list[float],
    ) -> tuple[list[float], dict[str, Any]]:
        sampled_speed = math.hypot(*sampled_velocity_xy)
        if sampled_speed <= 1.0e-8:
            raise ValueError(f"{profile} sampled a zero launch velocity")
        direction = [value / sampled_speed for value in sampled_velocity_xy]
        candidates = []
        for coordinate, center, safe_extent, component in zip(
            position_xy,
            (cx, cy),
            (safe_x, safe_y),
            direction,
        ):
            if component > 1.0e-8:
                candidates.append((center + safe_extent - coordinate) / component)
            elif component < -1.0e-8:
                candidates.append((center - safe_extent - coordinate) / component)
        if not candidates:
            raise ValueError(f"{profile} has no finite coast direction")
        available_distance = min(value for value in candidates if value > 0.0)
        target_fraction = rng.uniform(
            *[
                float(value)
                for value in profile_rules[
                    "target_coast_fraction_of_available_distance"
                ]
            ]
        )
        maximum_target = min(
            float(profile_rules["maximum_coast_distance_m"]),
            available_distance * 0.82,
        )
        if maximum_target < 0.10:
            raise ValueError(
                f"{profile} has only {available_distance:.3f} m of safe travel"
            )
        target_distance = min(
            max(
                target_fraction * available_distance,
                min(
                    float(profile_rules["minimum_coast_distance_m"]),
                    maximum_target,
                ),
            ),
            maximum_target,
        )
        effective_friction = max(0.0, dynamic_friction * support_friction)
        if proxy_motion_class == "rolling_round":
            round_rules = rules["round_proxy_motion"]
            target_duration = rng.uniform(
                *[float(value) for value in round_rules["target_duration_s"]]
            )
            launch_speed = target_distance / target_duration
            launch_speed = max(
                float(round_rules["minimum_launch_speed_m_s"]),
                min(
                    launch_speed,
                    float(round_rules["maximum_launch_speed_m_s"]),
                ),
            )
            scale = launch_speed / sampled_speed
            velocity_xy = [value * scale for value in sampled_velocity_xy]
            return velocity_xy, {
                "method": "rolling_travel_time_target",
                "proxy_motion_class": proxy_motion_class,
                "effective_friction": round(effective_friction, 7),
                "available_safe_distance_m": round(available_distance, 7),
                "target_fraction_of_available_distance": round(
                    target_fraction, 7
                ),
                "target_coast_distance_m": round(target_distance, 7),
                "target_duration_s": round(target_duration, 7),
                "launch_speed_m_s": round(launch_speed, 7),
            }
        speed_margin = rng.uniform(
            *[
                float(value)
                for value in profile_rules["minimum_coast_speed_margin"]
            ]
        )
        maximum_launch_speed = float(profile_rules["maximum_launch_speed_m_s"])
        unconstrained_target_distance = target_distance
        if effective_friction > 1.0e-8:
            speed_limited_distance = (
                (maximum_launch_speed / speed_margin) ** 2
                / (2.0 * effective_friction * gravity_m_s2)
            )
            target_distance = min(target_distance, speed_limited_distance)
        minimum_coast_speed = math.sqrt(
            2.0 * effective_friction * gravity_m_s2 * target_distance
        )
        launch_speed = max(sampled_speed, minimum_coast_speed * speed_margin)
        if launch_speed > maximum_launch_speed + 1.0e-7:
            raise ValueError(
                f"{profile} requires {launch_speed:.3f} m/s, above the "
                f"{maximum_launch_speed:.3f} m/s physical limit"
            )
        scale = launch_speed / sampled_speed
        velocity_xy = [value * scale for value in sampled_velocity_xy]
        return velocity_xy, {
            "method": "coulomb_coast_to_visible_motion",
            "proxy_motion_class": proxy_motion_class,
            "effective_friction": round(effective_friction, 7),
            "available_safe_distance_m": round(available_distance, 7),
            "target_fraction_of_available_distance": round(target_fraction, 7),
            "unconstrained_target_coast_distance_m": round(
                unconstrained_target_distance, 7
            ),
            "target_coast_distance_m": round(target_distance, 7),
            "sampled_speed_m_s": round(sampled_speed, 7),
            "minimum_coast_speed_m_s": round(minimum_coast_speed, 7),
            "speed_margin": round(speed_margin, 7),
            "launch_speed_m_s": round(launch_speed, 7),
        }

    if profile == "vertical_drop":
        position = [
            cx + rng.uniform(*profile_rules["start_x_safe_fraction"]) * safe_x,
            lane_y,
            rest_z + rng.uniform(*profile_rules["drop_height_m"]),
        ]
        velocity = [0.0, 0.0, 0.0]
    elif profile == "resting_push":
        position = [
            cx + float(profile_rules["start_x_safe_fraction"]) * safe_x,
            lane_y,
            rest_z,
        ]
        velocity_xy, calculation = coast_adjusted_velocity(
            [rng.uniform(*profile_rules["velocity_x_m_s"]), 0.0],
            position[:2],
        )
        velocity = [*velocity_xy, 0.0]
    elif profile == "diagonal_push":
        position = [
            cx + float(profile_rules["start_x_safe_fraction"]) * safe_x,
            lane_y + float(profile_rules["start_y_safe_fraction"]) * safe_y,
            rest_z,
        ]
        velocity_xy, calculation = coast_adjusted_velocity(
            [
                rng.uniform(*profile_rules["velocity_x_m_s"]),
                rng.uniform(*profile_rules["velocity_y_m_s"]),
            ],
            position[:2],
        )
        velocity = [*velocity_xy, 0.0]
        angular = [
            0.0,
            0.0,
            rng.uniform(*profile_rules["angular_velocity_z_rad_s"]),
        ]
    elif profile == "edge_exit":
        assert edge_exit_direction is not None
        direction = edge_exit_direction
        lateral = np.asarray([-direction[1], direction[0]], dtype=np.float64)
        directional_safe_extent = min(
            safe_extent / abs(component)
            for safe_extent, component in zip((safe_x, safe_y), direction)
            if abs(component) > 1.0e-8
        )
        start_fraction = rng.uniform(
            *[float(value) for value in profile_rules["start_x_safe_fraction"]]
        )
        lateral_fraction = float(
            interaction_policy.get("edge_exit_lateral_center_fraction", 0.0)
        )
        lateral_safe_extent = min(
            safe_extent / abs(component)
            for safe_extent, component in zip((safe_x, safe_y), lateral)
            if abs(component) > 1.0e-8
        )
        position_xy = (
            np.asarray([cx, cy], dtype=np.float64)
            + direction * start_fraction * directional_safe_extent
            + lateral * lateral_fraction * lateral_safe_extent
        )
        position = [float(position_xy[0]), float(position_xy[1]), rest_z]
        projected_half_extent = (
            abs(direction[0]) * oriented_half_x
            + abs(direction[1]) * oriented_half_y
        )
        physical_size_xy_m = physical_support_size_xy_m or [sx, sy]
        support_edge_distance = min(
            half_extent / abs(component)
            for half_extent, component in zip(
                (
                    0.5 * float(physical_size_xy_m[0]),
                    0.5 * float(physical_size_xy_m[1]),
                ),
                direction,
            )
            if abs(component) > 1.0e-8
        )
        exit_center_distance = support_edge_distance + projected_half_extent
        start_center_distance = float(
            np.dot(position_xy - np.asarray([cx, cy]), direction)
        )
        coast_distance = max(0.01, exit_center_distance - start_center_distance)
        effective_friction = max(0.0, dynamic_friction * support_friction)
        tipping_barrier_height = 0.0
        if proxy_motion_class != "rolling_round":
            tipping_barrier_height = max(
                0.0,
                math.hypot(projected_half_extent, half_z) - half_z,
            )
        tipping_energy_multiplier = float(
            profile_rules["non_round_tipping_energy_multiplier"]
        )
        minimum_specific_energy = gravity_m_s2 * (
            effective_friction * coast_distance
            + tipping_energy_multiplier * tipping_barrier_height
        )
        minimum_coast_speed = math.sqrt(2.0 * minimum_specific_energy)
        speed_margin = rng.uniform(
            *[float(value) for value in profile_rules["minimum_coast_speed_margin"]]
        )
        launch_speed = minimum_coast_speed * speed_margin
        maximum_launch_speed = float(profile_rules["maximum_launch_speed_m_s"])
        if launch_speed > maximum_launch_speed + 1.0e-7:
            raise ValueError(
                f"{profile} requires {launch_speed:.3f} m/s, above the "
                f"{maximum_launch_speed:.3f} m/s physical limit"
            )
        velocity = [
            float(direction[0]) * launch_speed,
            float(direction[1]) * launch_speed,
            0.0,
        ]
        calculation = {
            "method": "coulomb_coast_to_clear_support",
            "clear_exit_direction_xy": [
                round(float(direction[0]), 7),
                round(float(direction[1]), 7),
            ],
            "start_safe_fraction": round(start_fraction, 7),
            "proxy_motion_class": proxy_motion_class,
            "safe_support_size_xy_m": [round(sx, 7), round(sy, 7)],
            "physical_support_size_xy_m": [
                round(float(value), 7) for value in physical_size_xy_m
            ],
            "projected_half_extent_m": round(projected_half_extent, 7),
            "coast_distance_m": round(coast_distance, 7),
            "effective_friction": round(effective_friction, 7),
            "tipping_barrier_height_m": round(tipping_barrier_height, 7),
            "tipping_energy_multiplier": round(tipping_energy_multiplier, 7),
            "minimum_specific_energy_j_kg": round(minimum_specific_energy, 7),
            "minimum_coast_speed_m_s": round(minimum_coast_speed, 7),
            "speed_margin": round(speed_margin, 7),
            "launch_speed_m_s": round(launch_speed, 7),
        }
    elif profile == "workbench_clear_zone_drop":
        position = [
            cx,
            cy + rng.uniform(*profile_rules["start_y_safe_fraction"]) * safe_y,
            rest_z + rng.uniform(*profile_rules["drop_height_m"]),
        ]
        velocity = [0.0, 0.0, 0.0]
    elif profile == "workbench_long_axis_push":
        position = [
            cx,
            cy + float(profile_rules["start_y_safe_fraction"]) * safe_y,
            rest_z,
        ]
        velocity_xy, calculation = coast_adjusted_velocity(
            [0.0, rng.uniform(*profile_rules["velocity_y_m_s"])],
            position[:2],
        )
        velocity = [*velocity_xy, 0.0]
    else:
        raise ValueError(profile)
    result = {
        "position_m": [round(float(value), 7) for value in position],
        "orientation_quaternion_xyzw": [round(float(value), 8) for value in quaternion([0.0, 0.0, yaw])],
        "linear_velocity_m_s": [round(float(value), 7) for value in velocity],
        "angular_velocity_rad_s": [round(float(value), 7) for value in angular],
    }
    if calculation is not None:
        if prop_lane_clearance is not None:
            calculation["prop_lane_clearance_m"] = round(
                prop_lane_clearance, 7
            )
        result["calculation"] = calculation
    return result


def assignments(
    dynamic: list[dict[str, Any]],
    supports: list[dict[str, Any]],
    props: list[dict[str, Any]],
    count: int,
    rng: random.Random,
    excluded_support_categories: set[str],
    profiles: list[str] | None = None,
    edge_exit_minimum_proxy_volume_fill_ratio: float = 0.0,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, str]]:
    regular_supports = [
        record
        for record in supports
        if str(record["semantic_category"]) not in excluded_support_categories
    ]
    if not regular_supports:
        raise ValueError("generic asset sampling requires a non-game support")
    profiles = profiles or ["vertical_drop", "resting_push", "diagonal_push", "edge_exit"]
    edge_safe_supports = [
        record
        for record in regular_supports
        if record["proxy"].get("interaction_policy", {}).get(
            "clear_exit_directions_xy"
        )
    ]
    if "edge_exit" in profiles and not edge_safe_supports:
        raise ValueError(
            "edge_exit sampling requires at least one support with a reviewed "
            "clear_exit_directions_xy policy"
        )
    edge_safe_dynamics = [
        record
        for record in dynamic
        if proxy_volume_fill_ratio(record)
        >= edge_exit_minimum_proxy_volume_fill_ratio
    ]
    if "edge_exit" in profiles and not edge_safe_dynamics:
        raise ValueError(
            "edge_exit sampling requires at least one compact dynamic proxy"
        )
    result: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None, str]] = []
    regular_dynamic_index = 0
    edge_dynamic_index = 0
    regular_support_index = 0
    edge_support_index = 0
    for index in range(count):
        profile = profiles[index % len(profiles)]
        if profile == "edge_exit":
            dynamic_record = edge_safe_dynamics[
                edge_dynamic_index % len(edge_safe_dynamics)
            ]
            support_record = edge_safe_supports[
                edge_support_index % len(edge_safe_supports)
            ]
            edge_dynamic_index += 1
            edge_support_index += 1
        else:
            dynamic_record = dynamic[regular_dynamic_index % len(dynamic)]
            support_record = regular_supports[
                regular_support_index % len(regular_supports)
            ]
            regular_dynamic_index += 1
            regular_support_index += 1
        result.append((dynamic_record, support_record, None, profile))
    def transverse_capacity(index: int) -> float:
        dynamic_record, support_record, _, _ = result[index]
        dynamic_y = float(dynamic_record["visual"]["canonical_extent_m"][1])
        support_y = float(support_record["proxy"]["usable_surfaces"][0]["size_xy_m"][1])
        return support_y - dynamic_y

    def prop_compatible(prop_record: dict[str, Any], support_record: dict[str, Any]) -> bool:
        prop_category = str(prop_record["semantic_category"])
        support_category = str(support_record["semantic_category"])
        if "game_table" in support_category:
            return False
        if "kitchen" in support_category:
            return prop_category in {"prop_tray", "prop_tableware"}
        if "office" in support_category or "lab_bench" in support_category:
            return prop_category in {"prop_lamp", "prop_books"}
        return True

    props_by_width = sorted(
        props,
        key=lambda record: float(record["visual"]["canonical_extent_m"][1]),
        reverse=True,
    )
    unused_slots = set(range(count))
    for prop_record in props_by_width:
        compatible_slots = [
            index
            for index in unused_slots
            if prop_compatible(prop_record, result[index][1])
        ]
        if not compatible_slots:
            raise ValueError(f"no semantically compatible support slot for {prop_record['asset_id']}")
        index = max(compatible_slots, key=transverse_capacity)
        dynamic_record, support_record, _, profile = result[index]
        result[index] = (dynamic_record, support_record, prop_record, profile)
        unused_slots.remove(index)
    rng.shuffle(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument(
        "--semantic-rules",
        type=Path,
        default=DEFAULT_SEMANTIC_RULES_PATH,
    )
    parser.add_argument(
        "--composition-rules",
        type=Path,
        default=DEFAULT_COMPOSITION_RULES_PATH,
    )
    parser.add_argument("--visual-rules", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 360])
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--support-id")
    parser.add_argument("--dynamic-id")
    parser.add_argument("--static-prop-id")
    parser.add_argument("--profiles", nargs="+")
    parser.add_argument("--no-static-props", action="store_true")
    parser.add_argument("--scene-id-prefix", default="assetonly")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    registry_path = project_path(root, args.registry)
    semantic_rules_path = project_path(root, args.semantic_rules)
    composition_rules_path = project_path(root, args.composition_rules)
    visual_rules_path = (
        args.visual_rules.resolve()
        if args.visual_rules
        else root / VISUAL_RULES_PATH.relative_to(PROJECT_ROOT)
    )
    camera_rules_path = root / CAMERA_RULES_PATH.relative_to(PROJECT_ROOT)
    catalog_path = project_path(root, args.catalog)
    backend_path = root / BACKEND_PATH.relative_to(PROJECT_ROOT)
    output = args.output.resolve()
    registry = load_json(registry_path)
    validate_registry_counts(registry)
    proxy_manifest, proxy_records = load_catalog(
        root, catalog_path, require_runtime_validation=True
    )
    physical_by_id = records_by_id(proxy_records)
    semantic_rules = load_json(semantic_rules_path)
    composition_rules = load_json(composition_rules_path)
    visual_rules = load_json(visual_rules_path)
    camera_rules = load_json(camera_rules_path)["camera_observation"]
    composition_by_id = {
        record["asset_id"]: record for record in composition_rules["records"]
    }
    excluded_support_categories = set(
        semantic_rules["generic_one_object"]["excluded_support_categories"]
    )
    hdri_records = load_json(root / HDRI_MANIFEST_PATH.relative_to(PROJECT_ROOT))["records"]
    backend = load_json(backend_path)
    enabled = [record for record in registry["records"] if record["admission"].get("sampling_enabled", False)]
    dynamic = sorted((record for record in enabled if record["proxy"]["kind"] == "dynamic_rigid"), key=lambda item: item["asset_id"])
    if args.dynamic_id:
        dynamic = [record for record in dynamic if record["asset_id"] == args.dynamic_id]
        if not dynamic:
            raise ValueError(f"unknown enabled dynamic asset: {args.dynamic_id}")
    support_statuses = {"ready_generic"}
    if args.support_id:
        support_statuses.add("ready_specialized")
    supports = sorted(
        (
            record
            for record in enabled
            if record["proxy"]["kind"] == "support_compound"
            and composition_by_id[record["asset_id"]]["sampling_status"] in support_statuses
            and record["asset_id"] in physical_by_id
            and physical_by_id[record["asset_id"]]["admission"]["sampling_ready"]
            and (not args.support_id or record["asset_id"] == args.support_id)
        ),
        key=lambda item: item["asset_id"],
    )
    if args.support_id and not supports:
        raise ValueError(f"unknown enabled reviewed support asset: {args.support_id}")
    if args.support_id and args.profiles:
        allowed_profiles = set(
            composition_by_id[args.support_id]["scene_fit"].get("allowed", [])
        )
        profile_semantics = {
            "vertical_drop": "drop_to_surface",
            "resting_push": "slide_push",
            "diagonal_push": "slide_push",
            "edge_exit": "edge_exit",
            "workbench_clear_zone_drop": "workbench_clear_zone_drop",
            "workbench_long_axis_push": "workbench_long_axis_push",
        }
        unsupported_profiles = sorted(
            profile
            for profile in args.profiles
            if profile_semantics.get(profile) not in allowed_profiles
        )
        if unsupported_profiles:
            raise ValueError(
                f"profiles are not admitted for {args.support_id}: {unsupported_profiles}"
            )
    props = sorted(
        (
            record
            for record in enabled
            if record["proxy"]["kind"] == "static_compound"
            and composition_by_id[record["asset_id"]]["sampling_status"] == "ready_static"
        ),
        key=lambda item: item["asset_id"],
    )
    if args.static_prop_id:
        props = [record for record in props if record["asset_id"] == args.static_prop_id]
        if not props:
            raise ValueError(f"unknown enabled reviewed static prop: {args.static_prop_id}")
    if args.static_prop_id and args.no_static_props:
        raise ValueError("--static-prop-id conflicts with --no-static-props")
    if args.no_static_props:
        props = []
    if args.count < len(dynamic):
        raise ValueError(f"count must be at least {len(dynamic)} to cover every dynamic asset")
    rng = random.Random(args.seed)
    jobs = assignments(
        dynamic,
        supports,
        props,
        args.count,
        rng,
        excluded_support_categories,
        args.profiles,
        float(
            backend["asset_proxy_rules"]["motion_profiles"]["edge_exit"][
                "minimum_proxy_volume_fill_ratio"
            ]
        ),
    )
    records = []
    for index, (dynamic_record, support_record, prop_record, profile) in enumerate(jobs, start=1):
        slug = re.sub(r"[^a-z0-9]+", "_", dynamic_record["name"].lower()).strip("_")[:24]
        scene_id = f"{args.scene_id_prefix}_{index:03d}_{profile}_{slug}"
        support_proxy_record = physical_by_id[support_record["asset_id"]]
        support_binding = compile_static_support_binding(
            support_proxy_record,
            usage_id="curated_support",
        )
        surface = copy.deepcopy(
            support_binding["target_support_frame"]["safe_surface"]
        )
        prop_binding = (
            place_static_prop(
                prop_record,
                surface,
                rng,
                backend["asset_proxy_rules"]["placement"],
            )
            if prop_record
            else None
        )
        bounds_low, bounds_high = local_bounds(dynamic_record)
        collider_shapes = {
            str(collider["shape"])
            for collider in dynamic_record["proxy"]["colliders"]
        }
        proxy_motion_class = (
            "rolling_round" if collider_shapes == {"sphere"} else "sliding"
        )
        scene_dir = output / "scenes" / scene_id
        physics_dir = scene_dir / "physics"
        physics_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = physics_dir / "trajectory.npz"
        audit_path = physics_dir / "audit.json"
        simulation_record_path = physics_dir / "simulation_record.json"
        metadata_path = scene_dir / "metadata.json"
        minimum_proxy_extent_m = float(np.min(bounds_high - bounds_low))
        simulation_hz = simulation_hz_for_min_extent(
            backend["engine"], minimum_proxy_extent_m
        )
        frame_count = int(round(args.duration * args.fps)) + 1
        maximum_attempts = int(
            backend["asset_proxy_rules"]["sampling"][
                "maximum_attempts_per_scene"
            ]
        )
        rejected_attempts: list[dict[str, Any]] = []
        trajectory: dict[str, np.ndarray] | None = None
        audit: dict[str, Any] | None = None
        initial: dict[str, Any] | None = None
        mass: float | None = None
        for attempt_index in range(1, maximum_attempts + 1):
            try:
                initial = motion_initial_state(
                    profile,
                    surface,
                    bounds_low,
                    bounds_high,
                    prop_binding,
                    rng,
                    backend["asset_proxy_rules"],
                    float(
                        dynamic_record["proxy"]
                        .get("material", {})
                        .get("friction", 0.45)
                    ),
                    float(
                        backend["asset_proxy_rules"]["contact"]["support"][
                            "lateral_friction"
                        ]
                    ),
                    abs(float(backend["engine"]["gravity_m_s2"][2])),
                    proxy_motion_class,
                    physical_support_size_xy_m=support_binding[
                        "target_support_frame"
                    ]["size_xy_m"],
                    interaction_policy=support_binding["usage_contract"],
                )
            except ValueError as error:
                rejected_attempts.append(
                    {
                        "attempt_index": attempt_index,
                        "initialization_error": str(error),
                    }
                )
                continue
            mass = choose_mass(dynamic_record, rng)
            expected_motion = asset_motion_usefulness(backend, profile)
            trajectory, audit = simulate_scene(
                root,
                dynamic_record,
                support_binding,
                prop_record,
                prop_binding,
                initial,
                mass,
                args.duration,
                args.fps,
                profile,
                backend,
                expected_motion,
            )
            if audit["passed"]:
                break
            rejected_attempts.append(
                {
                    "attempt_index": attempt_index,
                    "failed_checks": [
                        name
                        for name, passed in audit["checks"].items()
                        if not passed
                    ],
                }
            )
        if trajectory is None or audit is None or initial is None or mass is None:
            raise ValueError(
                f"{scene_id} produced no simulatable initial state after "
                f"{maximum_attempts} attempts"
            )
        environment = choose_environment(
            root,
            support_record,
            hdri_records,
            visual_rules,
            rng,
        )
        metadata = {
            "schema_version": "physweep_asset_proxy_scene_v3",
            "scene_id": scene_id,
            "seed": args.seed,
            "dynamic_asset_name": str(dynamic_record["name"]),
            "registry": {
                "path": str(registry_path.relative_to(root)),
                "sha256": sha256(registry_path),
            },
            "semantic_rules": {
                "path": str(semantic_rules_path.relative_to(root)),
                "sha256": sha256(semantic_rules_path),
            },
            "composition_rules": {
                "path": str(composition_rules_path.relative_to(root)),
                "sha256": sha256(composition_rules_path),
            },
            "visual_rules": {
                "path": str(visual_rules_path.relative_to(root)),
                "sha256": sha256(visual_rules_path),
            },
            "camera_rules": {
                "path": str(camera_rules_path.relative_to(root)),
                "sha256": sha256(camera_rules_path),
            },
            "physical_proxy_catalog": {
                "path": str(catalog_path.relative_to(root)),
                "sha256": sha256(catalog_path),
                "records_sha256": proxy_manifest["records_sha256"],
            },
            "assets": {
                "dynamic_asset_id": dynamic_record["asset_id"],
                "support_asset_id": support_record["asset_id"],
                "static_prop_asset_id": prop_binding["asset_id"] if prop_binding else None,
            },
            "sampling": {
                "accepted": bool(audit["passed"]),
                "attempt_count": attempt_index,
                "maximum_attempts": maximum_attempts,
                "rejected_attempts": rejected_attempts,
            },
            "physics": {
                "backend": "pybullet_exact_static_support_v1",
                "audit_version": ASSET_AUDIT_VERSION,
                "backend_config": {
                    "path": str(backend_path.relative_to(root)),
                    "sha256": sha256(backend_path),
                },
                "duration_s": args.duration,
                "output_fps": args.fps,
                "simulation_hz": simulation_hz,
                "frame_count": frame_count,
                "motion_profile": profile,
                "expected_motion": expected_motion,
                "mass_kg": mass,
                "initial_state": initial,
                "support_surface": surface,
                "static_support_binding": support_binding,
                "static_prop": prop_binding,
                "trajectory_path": str(trajectory_path.relative_to(root)),
                "audit_path": str(audit_path.relative_to(root)),
                "simulation_record_path": str(
                    simulation_record_path.relative_to(root)
                ),
            },
            "camera_request": {
                "observation": asset_camera_observation(camera_rules, profile)
            },
            "render": {
                "evidence_contract": "physweep_specialized_render_evidence_v2",
                "resolution": args.resolution,
                "samples": args.samples,
                "video_path": str((output / "videos" / f"{scene_id}.mp4").relative_to(root)),
                "inspection_frame_dir": str((output / "frames" / scene_id).relative_to(root)),
                "environment": environment,
            },
            "implementation": {
                "generator": file_binding(root, Path(__file__)),
                "renderer": file_binding(
                    root, root / "tools/rendering/render_asset_proxy_scene.py"
                ),
                "render_evidence": file_binding(
                    root, root / "tools/rendering/specialized_render_evidence.py"
                ),
            },
        }
        attach_object_identity(
            metadata,
            trajectory_path=str(trajectory_path.relative_to(root)),
            mask_path=str((output / "masks" / scene_id).relative_to(root)),
        )
        metadata = freeze_metadata(metadata_path, metadata)
        frozen_physics = metadata["physics"]
        if int(audit["simulation_hz"]) != int(frozen_physics["simulation_hz"]):
            raise RuntimeError("simulation frequency differs from frozen metadata")
        if int(trajectory["time_s"].shape[0]) != int(
            frozen_physics["frame_count"]
        ):
            raise RuntimeError("trajectory length differs from frozen metadata")
        np.savez_compressed(trajectory_path, **trajectory)
        write_json(audit_path, audit)
        write_simulation_record(
            root=root,
            metadata_path=metadata_path,
            metadata=metadata,
            trajectory_path=trajectory_path,
            audit_path=audit_path,
            record_path=simulation_record_path,
        )
        records.append(
            {
                "scene_id": scene_id,
                "metadata_path": str(metadata_path.relative_to(root)),
                "dynamic_asset": dynamic_record["name"],
                "dynamic_asset_id": dynamic_record["asset_id"],
                "support_asset": support_record["name"],
                "support_asset_id": support_record["asset_id"],
                "static_prop_asset": prop_binding["name"] if prop_binding else None,
                "static_prop_asset_id": prop_binding["asset_id"] if prop_binding else None,
                "motion_profile": profile,
                "attempt_count": attempt_index,
                "rejected_attempt_count": len(rejected_attempts),
                "audit_passed": audit["passed"],
                "failed_checks": [name for name, passed in audit["checks"].items() if not passed],
            }
        )
    counts = {
        "dynamic_assets": len({record["dynamic_asset_id"] for record in records}),
        "support_assets": len({record["support_asset_id"] for record in records}),
        "static_prop_assets": len({record["static_prop_asset_id"] for record in records if record["static_prop_asset_id"]}),
        "rejected_attempts": sum(
            record["rejected_attempt_count"] for record in records
        ),
        "motion_profiles": dict(Counter(record["motion_profile"] for record in records)),
    }
    manifest = {
        "schema_version": "physweep_asset_proxy_manifest_v1",
        "dataset_id": f"asset_only_random{args.count}_seed{args.seed}",
        "seed": args.seed,
        "output_root": str(output),
        "sample_count": len(records),
        "passed_count": sum(record["audit_passed"] for record in records),
        "counts": counts,
        "records": records,
    }
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    print(manifest_path)
    print(json.dumps({"passed": manifest["passed_count"], "total": len(records), **counts}, indent=2))
    if manifest["passed_count"] != len(records):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
