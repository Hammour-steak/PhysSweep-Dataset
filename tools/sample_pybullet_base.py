#!/usr/bin/env python3
"""Sample deterministic one-object PhysSweep base scenes for PyBullet."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from rigid_geometry import (
    build_support_geometry,
    clamp,
    cross,
    direction_unit,
    normalize,
    object_contact_offset_m,
    pose_on_support,
    support_surface_height_m,
    validate_support_geometry,
)
from scene_kit_compiler import load_sampling_bundle
from physics_time_step import simulation_hz_for_geometry
from static_support_proxy import compile_static_support_binding
from environment_collision import compile_environment_binding
from camera_geometry import camera_corridor_admits_inclined_surface
from motion_rules import MotionDerivationContext, MotionPlan, derive_motion
from motion_rules.ballistic import (
    bounce_observation_contract as grouped_bounce_observation_contract,
)
try:
    from object_identity_contract import attach_object_identity
except ModuleNotFoundError:  # import when the script is loaded as tools.* in tests
    from tools.object_identity_contract import attach_object_identity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = PROJECT_ROOT / "configs/one_object_sampling_bundle.json"
DEFAULT_OUTPUT = "physweep_pybullet_base_v1"


class IncompatibleSupportVisual(ValueError):
    """The selected visual support cannot safely host the sampled object."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_active_rules(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    return load_sampling_bundle(root, root / BUNDLE_PATH.relative_to(PROJECT_ROOT))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sampling_manifest_rule_sources(
    root: Path, bundle_path: Path, bundle: dict[str, Any]
) -> dict[str, str]:
    """Separate the sampling entry point from the compiled camera rules."""

    root = root.resolve()
    bundle_path = bundle_path.resolve()
    rules_path = (root / str(bundle["base_rules"])).resolve()
    return {
        "sampling_bundle_path": str(bundle_path.relative_to(root)),
        "sampling_bundle_sha256": sha256(bundle_path),
        "rules_path": str(rules_path.relative_to(root)),
        "rules_sha256": sha256(rules_path),
    }


def log_uniform(rng: random.Random, lower: float, upper: float) -> float:
    if lower <= 0.0 or upper < lower:
        raise ValueError("log-uniform interval must be positive and ordered")
    return math.exp(rng.uniform(math.log(lower), math.log(upper)))


def repeated_shuffled(values: list[Any], count: int, rng: random.Random) -> list[Any]:
    result = [copy.deepcopy(values[index % len(values)]) for index in range(count)]
    rng.shuffle(result)
    return result


def coverage_cycle_by_group(
    groups: list[str], values: list[Any], rng: random.Random
) -> list[Any]:
    if not values:
        raise ValueError("coverage cycle values cannot be empty")
    cycles: dict[str, list[Any]] = {}
    result = []
    for group in groups:
        if not cycles.get(group):
            cycle = copy.deepcopy(values)
            rng.shuffle(cycle)
            cycles[group] = cycle
        result.append(cycles[group].pop())
    return result


def object_supports_motion(obj: dict[str, Any], motion: str) -> bool:
    return motion not in set(obj.get("excluded_motion_families", []))


def balanced_objects_for_motions(
    motions: list[str],
    objects: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    object_order = copy.deepcopy(objects)
    rng.shuffle(object_order)
    usage: Counter[str] = Counter()
    pair_usage: Counter[tuple[str, str]] = Counter()
    motion_cursor: Counter[str] = Counter()
    selected = []
    for motion in motions:
        compatible = [
            obj for obj in object_order if object_supports_motion(obj, motion)
        ]
        if not compatible:
            raise ValueError(f"motion has no compatible object profile: {motion}")
        minimum_pair_usage = min(
            pair_usage[(motion, str(obj["label"]))] for obj in compatible
        )
        pair_least_used = [
            obj
            for obj in compatible
            if pair_usage[(motion, str(obj["label"]))] == minimum_pair_usage
        ]
        minimum_usage = min(usage[str(obj["label"])] for obj in pair_least_used)
        least_used = [
            obj
            for obj in pair_least_used
            if usage[str(obj["label"])] == minimum_usage
        ]
        obj = least_used[motion_cursor[motion] % len(least_used)]
        motion_cursor[motion] += 1
        usage[str(obj["label"])] += 1
        pair_usage[(motion, str(obj["label"]))] += 1
        selected.append(copy.deepcopy(obj))
    return selected


def balanced_motion_object_pairs(
    motions: list[str],
    objects: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[tuple[str, dict[str, Any]]]:
    motion_order = list(motions)
    object_order = copy.deepcopy(objects)
    rng.shuffle(motion_order)
    rng.shuffle(object_order)
    step = max(1, len(motion_order) - 1)
    while math.gcd(step, len(object_order)) != 1:
        step -= 1
    pair_usage: Counter[tuple[str, str]] = Counter()
    pairs = []
    for index in range(count):
        motion_index = index % len(motion_order)
        round_index = index // len(motion_order)
        object_index = (motion_index + round_index * step) % len(object_order)
        motion = motion_order[motion_index]
        obj = object_order[object_index]
        if not object_supports_motion(obj, motion):
            compatible = [
                candidate
                for candidate in object_order
                if object_supports_motion(candidate, motion)
            ]
            minimum_usage = min(
                pair_usage[(motion, str(candidate["label"]))]
                for candidate in compatible
            )
            obj = next(
                candidate
                for candidate in compatible
                if pair_usage[(motion, str(candidate["label"]))] == minimum_usage
            )
        pair_usage[(motion, str(obj["label"]))] += 1
        pairs.append((motion, copy.deepcopy(obj)))
    return pairs


def balanced_visual_variants(
    objects: list[dict[str, Any]],
    target_mesh_fraction: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not 0.0 <= target_mesh_fraction <= 1.0:
        raise ValueError("target mesh fraction must be between zero and one")
    mesh_capable: dict[str, list[int]] = {}
    for index, obj in enumerate(objects):
        if any(variant["type"] == "mesh" for variant in obj["visual_variants"]):
            mesh_capable.setdefault(str(obj["label"]), []).append(index)
    eligible_count = sum(len(indices) for indices in mesh_capable.values())
    target_mesh_count = min(
        eligible_count, int(round(len(objects) * target_mesh_fraction))
    )
    labels = list(mesh_capable)
    rng.shuffle(labels)
    for indices in mesh_capable.values():
        rng.shuffle(indices)
    mesh_indices: set[int] = set()
    while len(mesh_indices) < target_mesh_count:
        progressed = False
        for label in labels:
            if mesh_capable[label]:
                mesh_indices.add(mesh_capable[label].pop())
                progressed = True
                if len(mesh_indices) == target_mesh_count:
                    break
        if not progressed:
            break
    selected = []
    for index, obj in enumerate(objects):
        visual_type = "mesh" if index in mesh_indices else "primitive"
        candidates = [
            variant
            for variant in obj["visual_variants"]
            if variant["type"] == visual_type
        ]
        if not candidates:
            raise ValueError(
                f"object {obj['label']} lacks required {visual_type} visual variant"
            )
        selected.append(copy.deepcopy(rng.choice(candidates)))
    return selected


def balanced_scene_visual_types(
    count: int, target_mesh_fraction: float, rng: random.Random
) -> list[str]:
    if not 0.0 <= target_mesh_fraction <= 1.0:
        raise ValueError("scene target mesh fraction must be between zero and one")
    mesh_count = int(round(count * target_mesh_fraction))
    types = ["mesh_backdrop"] * mesh_count
    types.extend(["procedural_room"] * (count - mesh_count))
    rng.shuffle(types)
    return types


def scene_visual_profile_admits_support(
    profile: dict[str, Any], support: dict[str, Any]
) -> bool:
    composition = profile.get("composition")
    if isinstance(composition, dict):
        if str(composition.get("review_status")) != "approved":
            return False
        support_id = str(support["label"])
        return any(
            support_id in {str(value) for value in binding["support_ids"]}
            for binding in composition.get("bindings", [])
        )
    if str(support["theme"]) not in {str(value) for value in profile["themes"]}:
        return False
    allowed_classes = {
        str(value)
        for value in profile.get(
            "scene_classes",
            ["ground_flat", "raised_flat", "ground_feature", "raised_feature"],
        )
    }
    if str(support["scene_class"]) not in allowed_classes:
        return False
    support_ids = profile.get("support_ids")
    return support_ids is None or str(support["label"]) in {
        str(value) for value in support_ids
    }


def scene_visual_profile_admits_camera(
    profile: dict[str, Any],
    motion: str,
    motion_subtype: str,
    trajectory_extent: str,
    support: dict[str, Any],
    camera_rules: dict[str, Any],
    camera_profile: dict[str, Any] | None = None,
) -> bool:
    composition = profile.get("composition")
    if isinstance(composition, dict):
        support_id = str(support["label"])
        compatible = any(
            support_id in {str(value) for value in binding["support_ids"]}
            and motion in {str(value) for value in binding["motion_families"]}
            and motion_subtype
            in {
                str(value)
                for value in binding.get("motion_subtypes", [motion_subtype])
            }
            and trajectory_extent
            in {
                str(value)
                for value in binding.get(
                    "trajectory_extents", [trajectory_extent]
                )
            }
            for binding in composition.get("bindings", [])
        )
        if not compatible:
            return False
        structure_context = str(
            camera_rules["motion_intents"][motion]["structure_context"]
        )
        if structure_context in {"inclined_surface", "ramp_and_landing"}:
            if camera_profile is None:
                return False
            camera_contract = composition["camera"]
            if not camera_corridor_admits_inclined_surface(
                base_azimuth_degrees=float(
                    camera_profile["overrides"]["view_rule"]["azimuth_degrees"]
                ),
                maximum_deviation_degrees=float(
                    camera_contract["maximum_local_azimuth_deviation_degrees"]
                ),
                minimum_side_readability=float(
                    camera_rules["minimum_inclined_surface_side_readability"]
                ),
            ):
                return False
    if str(profile["visual_type"]) != "mesh_backdrop":
        return True
    motion_range = camera_rules["motion_intents"][motion][
        "elevation_range_degrees"
    ]
    motion_minimum = float(motion_range[0])
    motion_maximum = float(motion_range[1])
    context = profile.get("camera_context", {})
    effective_minimum = max(
        motion_minimum,
        float(context.get("minimum_elevation_degrees", motion_minimum)),
    )
    effective_maximum = min(
        motion_maximum,
        float(context.get("maximum_elevation_degrees", motion_maximum)),
    )
    return effective_minimum <= effective_maximum


def compatibility_rule(rules: dict[str, Any], motion: str) -> dict[str, Any]:
    try:
        return rules["architecture"]["compatibility"]["motions"][motion]
    except KeyError as exc:
        raise ValueError(f"motion lacks a compatibility rule: {motion}") from exc


def constrained_trajectory_extent(
    motion: str,
    extent: dict[str, Any],
    axes: dict[str, Any],
    rules: dict[str, Any],
) -> dict[str, Any]:
    maximum = compatibility_rule(rules, motion).get("maximum_trajectory_extent")
    if maximum is None:
        return copy.deepcopy(extent)
    order = [str(record["label"]) for record in axes["trajectory_extent_axis"]]
    if str(maximum) not in order:
        raise ValueError(f"unknown maximum trajectory extent for {motion}: {maximum}")
    if order.index(str(extent["label"])) <= order.index(str(maximum)):
        return copy.deepcopy(extent)
    return copy.deepcopy(
        next(
            record
            for record in axes["trajectory_extent_axis"]
            if str(record["label"]) == str(maximum)
        )
    )


def scene_class_allowed(scene_class: str, motion: str, rules: dict[str, Any]) -> bool:
    return scene_class in set(compatibility_rule(rules, motion)["scene_classes"])


def support_allowed(
    support: dict[str, Any],
    motion: str,
    rules: dict[str, Any],
    scene_class: str | None = None,
) -> bool:
    support_scene_class = str(support["scene_class"])
    if scene_class is not None and support_scene_class != scene_class:
        return False
    rule = compatibility_rule(rules, motion)
    return (
        support_scene_class in set(rule["scene_classes"])
        and str(support["topology"]) in set(rule["topologies"])
    )


def weighted_scene_class_cycle(
    records: list[dict[str, Any]],
    supports: list[dict[str, Any]],
    motion: str,
    count: int,
    rng: random.Random,
    rules: dict[str, Any],
) -> list[str]:
    admitted = [
        record
        for record in records
        if scene_class_allowed(str(record["label"]), motion, rules)
        and any(
            support_allowed(support, motion, rules, str(record["label"]))
            for support in supports
        )
    ]
    if not admitted:
        raise ValueError(f"no scene class is admitted for {motion}")
    total_weight = sum(float(record["weight"]) for record in admitted)
    raw = [count * float(record["weight"]) / total_weight for record in admitted]
    allocated = [math.floor(value) for value in raw]
    remaining = count - sum(allocated)
    fractional = [value - floor for value, floor in zip(raw, allocated)]
    for _ in range(remaining):
        total = sum(fractional)
        if total <= 0.0:
            candidates = [index for index, value in enumerate(allocated) if value == 0]
            selected = rng.choice(candidates)
        else:
            threshold = rng.random() * total
            cumulative = 0.0
            selected = len(fractional) - 1
            for index, value in enumerate(fractional):
                cumulative += value
                if threshold <= cumulative:
                    selected = index
                    break
        allocated[selected] += 1
        fractional[selected] = 0.0
    cycle = [
        str(record["label"])
        for record, copies in zip(admitted, allocated)
        for _ in range(copies)
    ]
    rng.shuffle(cycle)
    return cycle


def support_compatibility_group(motion: str, rules: dict[str, Any]) -> str:
    return str(compatibility_rule(rules, motion)["group"])


def support_mesh_scale_ratio(
    support: dict[str, Any], profile: dict[str, Any]
) -> float:
    """Predict the renderer's non-uniform scale ratio from declared bounds."""

    source_x, source_y, _ = [float(value) for value in profile["source_bbox_size"]]
    angle = math.radians(float(profile["alignment_yaw_degrees"]))
    cosine = abs(math.cos(angle))
    sine = abs(math.sin(angle))
    aligned_x = cosine * source_x + sine * source_y
    aligned_y = sine * source_x + cosine * source_y
    if "colliders" in support:
        primary = [
            record
            for record in support["colliders"]
            if str(record["role"]) == "primary_support"
        ]
        if len(primary) != 1 or str(primary[0]["primitive"]) != "box":
            return math.inf
        target_x, target_y = [float(value) for value in primary[0]["size_m"][:2]]
        target_support_plane = float(support["surface_center_z_m"])
    else:
        target_x, target_y = [float(value) for value in support["size"][:2]]
        target_support_plane = float(support["top_z"])
    scales = [
        target_x / max(aligned_x, 1.0e-8),
        target_y / max(aligned_y, 1.0e-8),
        target_support_plane
        / max(float(profile["source_support_plane_z_from_bottom"]), 1.0e-8),
    ]
    return max(scales) / max(min(scales), 1.0e-8)


def bind_exact_static_support(
    architecture: dict[str, Any],
    support_geometry: dict[str, Any],
    support_visual_profile: dict[str, Any],
) -> dict[str, Any]:
    """Freeze one exact support binding before placement and simulation."""

    if str(support_visual_profile["visual_type"]) != "mesh_support":
        raise ValueError("exact support binding requires a mesh-support profile")
    if str(support_geometry["topology"]) != "flat_surface":
        raise ValueError("exact support binding currently requires a flat surface")
    records = {
        str(record["asset_id"]): record
        for record in architecture["physical_proxy_catalog"]["active_records"]
    }
    asset_id = str(support_visual_profile["asset_id"])
    if asset_id not in records:
        raise ValueError(f"support visual lacks an active physical proxy: {asset_id}")
    record = records[asset_id]
    if not bool(record["admission"]["sampling_ready"]):
        raise ValueError(f"support physical proxy is not sampling-ready: {asset_id}")
    primary = [
        collider
        for collider in support_geometry["colliders"]
        if str(collider["role"]) == "primary_support"
    ]
    if len(primary) != 1 or str(primary[0]["primitive"]) != "box":
        raise ValueError("exact support binding requires one primary support box")
    if any(
        abs(float(value)) > 1.0e-8
        for value in primary[0]["rotation_euler_degrees"]
    ):
        raise ValueError("exact support binding requires an axis-aligned surface")
    usage_id = f"generic_support:{support_geometry['semantic_type']}"
    binding = compile_static_support_binding(
        record,
        usage_id=usage_id,
        target_size_xy_m=[float(value) for value in primary[0]["size_m"][:2]],
        target_center_xy_m=[
            float(primary[0]["position_m"][0]),
            float(primary[0]["position_m"][1]),
        ],
        target_support_plane_z_m=float(
            support_geometry["surface_center_z_m"]
        ),
    )
    for collider in support_geometry["colliders"]:
        if str(collider["role"]) in {"primary_support", "support_structure"}:
            collider["collision_enabled"] = False
            collider["camera_proxy_enabled"] = True
            collider["replaced_by_static_support_binding"] = str(
                binding["binding_sha256"]
            )
    safe = binding["target_support_frame"]["safe_surface"]
    center = [float(value) for value in safe["center_xy_m"]]
    size = [float(value) for value in safe["size_xy_m"]]
    support_geometry["safe_surface_bounds"] = {
        "x": [
            round(center[0] - size[0] / 2.0, 6),
            round(center[0] + size[0] / 2.0, 6),
        ],
        "y": [
            round(center[1] - size[1] / 2.0, 6),
            round(center[1] + size[1] / 2.0, 6),
        ],
    }
    support_geometry["exact_static_binding"] = binding
    support_geometry["collision_authority"] = "exact_static_proxy"
    return binding


def scaled_size(
    record: dict[str, Any],
    scale_bin: dict[str, Any],
    rng: random.Random,
    readability: dict[str, Any],
) -> tuple[list[float], float, dict[str, Any]]:
    lower, upper = [float(value) for value in scale_bin["range"]]
    sampled_scale = rng.uniform(lower, upper)
    source_size = [float(value) for value in record["size"]]
    shape = str(record["shape"])
    semantic_category = str(record["semantic_category"])
    semantic_policy = readability.get("semantic_category_overrides", {}).get(
        semantic_category, {}
    )
    characteristic_rule = str(
        semantic_policy.get(
            "characteristic_rule",
            readability[f"{shape}_characteristic"],
        )
    )
    if characteristic_rule == "diameter" and shape in {"sphere", "cylinder"}:
        characteristic_extent = min(source_size[0], source_size[1])
    elif characteristic_rule == "second_largest_extent" and shape == "cuboid":
        characteristic_extent = sorted(source_size, reverse=True)[1]
    elif characteristic_rule == "largest_extent" and shape == "cuboid":
        characteristic_extent = max(source_size)
    else:
        raise ValueError(
            f"unsupported readability rule for {shape}: {characteristic_rule}"
        )
    minimum_extent = float(
        semantic_policy.get(
            "minimum_characteristic_extent_m",
            readability["minimum_characteristic_extent_m"],
        )
    )
    readability_scale = min(
        float(readability["maximum_readability_uplift_scale"]),
        minimum_extent / max(characteristic_extent, 1.0e-8),
    )
    effective_scale = max(sampled_scale, readability_scale)
    size_m = [round(value * effective_scale, 6) for value in source_size]
    return size_m, effective_scale, {
        "characteristic_rule": characteristic_rule,
        "policy_scope": (
            f"semantic_category:{semantic_category}" if semantic_policy else "default"
        ),
        "source_characteristic_extent_m": round(characteristic_extent, 6),
        "minimum_characteristic_extent_m": minimum_extent,
        "sampled_scale": round(sampled_scale, 6),
        "readability_floor_scale": round(readability_scale, 6),
        "effective_scale": round(effective_scale, 6),
        "adjusted": effective_scale > sampled_scale + 1.0e-9,
    }


def safe_half_extents(
    support: dict[str, Any],
    shape: str,
    size_m: list[float],
    pose_profile: str,
    direction: list[float],
) -> tuple[float, float, float, float]:
    bounds = support["safe_surface_bounds"]
    if shape == "sphere":
        object_half_x = object_half_y = float(size_m[0]) / 2.0
    elif shape == "cylinder" and pose_profile == "side_on_motion":
        normal = [float(value) for value in support["surface_frame"]["normal"]]
        tangent = normalize(
            [
                direction[index]
                - normal[index]
                * sum(direction[axis] * normal[axis] for axis in range(3))
                for index in range(3)
            ]
        )
        cylinder_axis = normalize(cross(normal, tangent))
        radius = max(float(size_m[0]), float(size_m[1])) / 2.0
        half_length = float(size_m[2]) / 2.0
        object_half_x = abs(cylinder_axis[0]) * half_length + radius
        object_half_y = abs(cylinder_axis[1]) * half_length + radius
    else:
        object_half_x = float(size_m[0]) / 2.0
        object_half_y = float(size_m[1]) / 2.0
    center_x = 0.5 * (float(bounds["x"][0]) + float(bounds["x"][1]))
    center_y = 0.5 * (float(bounds["y"][0]) + float(bounds["y"][1]))
    half_x = 0.5 * (float(bounds["x"][1]) - float(bounds["x"][0])) - object_half_x
    half_y = 0.5 * (float(bounds["y"][1]) - float(bounds["y"][0])) - object_half_y
    if min(half_x, half_y) <= 0.015:
        raise IncompatibleSupportVisual(
            "object does not fit in support safe bounds"
        )
    return center_x, center_y, half_x, half_y


def ray_limit(half_x: float, half_y: float, direction: list[float]) -> float:
    limits = []
    if abs(direction[0]) > 1.0e-8:
        limits.append(half_x / abs(direction[0]))
    if abs(direction[1]) > 1.0e-8:
        limits.append(half_y / abs(direction[1]))
    if not limits:
        raise ValueError("motion direction has no horizontal component")
    return min(limits)


def zone_offsets(zone: dict[str, Any], half_x: float, half_y: float, direction: list[float]) -> tuple[float, float]:
    lateral = [-direction[1], direction[0]]
    span = min(half_x, half_y)
    along = float(zone["along"]) * span * 0.55
    across = float(zone["lateral"]) * span * 0.55
    return (
        direction[0] * along + lateral[0] * across,
        direction[1] * along + lateral[1] * across,
    )


def clamp_to_safe(value: float, half_extent: float) -> float:
    return clamp(value, -half_extent, half_extent)


def extent_fraction(label: str) -> float:
    return {"short": 0.42, "medium": 0.60, "long": 0.76}[label]


def restitution_for_motion(
    rng: random.Random, backend: dict[str, Any], motion: str, object_nominal: float
) -> float:
    rules = backend["base_parameter_rules"]["restitution_by_motion"]
    lower, upper = rules.get(motion, rules["default"])
    sampled = rng.uniform(float(lower), float(upper))
    if motion == "bounce_1obj":
        return sampled
    if motion == "wall_impact_1obj":
        return min(sampled, max(0.08, float(object_nominal)))
    return min(sampled, max(0.04, float(object_nominal)))


def bounce_observation_contract(
    backend: dict[str, Any], shape: str, size_m: list[float]
) -> dict[str, Any]:
    return grouped_bounce_observation_contract(backend, shape, size_m)


def material_record(asset: dict[str, Any]) -> dict[str, Any]:
    return {
        "asset_source": "poly_haven_curated_v2",
        "asset_id": str(asset["asset_id"]),
        "path": str(asset["path"]),
        "map_count": int(asset["map_count"]),
        "maps": str(asset["maps"]),
        "curation_grade": {
            "surface": str(asset["surface_grade"]),
            "object": str(asset["object_grade"]),
        },
    }


def choose_material(
    rng: random.Random,
    assets: dict[str, dict[str, Any]],
    preferred: list[str],
    fallback: list[str],
    excluded: set[str] | None = None,
    allowed_object_grades: set[str] | None = None,
) -> dict[str, Any]:
    excluded = excluded or set()
    def admitted(asset_id: str) -> bool:
        if asset_id not in assets or asset_id in excluded:
            return False
        return (
            allowed_object_grades is None
            or str(assets[asset_id]["object_grade"]) in allowed_object_grades
        )

    candidates = [item for item in preferred if admitted(item)]
    if not candidates:
        candidates = [item for item in fallback if admitted(item)]
    if not candidates:
        raise ValueError("no admitted material remains after exclusions")
    return material_record(assets[rng.choice(candidates)])


def choose_appearance(
    rng: random.Random,
    materials: dict[str, dict[str, Any]],
    hdri_records: list[dict[str, Any]],
    object_hint: str,
    support_label: str,
    support_theme: str,
    environment_category: str,
    surface_family: str,
    contrast: dict[str, Any],
    visual_rules: dict[str, Any],
    object_color: list[float],
    visual_rules_sha256: str,
) -> dict[str, Any]:
    family_surface = visual_rules["surface_material_pools_by_family"].get(
        surface_family, []
    )
    semantic_surface = visual_rules["support_material_pools"].get(
        support_label,
        visual_rules["support_material_pools_by_theme"][support_theme],
    )
    compatible_surface = [item for item in family_surface if item in semantic_surface]
    all_surface = compatible_surface or semantic_surface
    support_material = choose_material(
        rng,
        materials,
        all_surface,
        semantic_surface,
    )
    excluded = {support_material["asset_id"]}
    object_pool = visual_rules["object_material_pools"].get(
        object_hint,
        visual_rules["fallback_object_material_pools_by_value"]["medium"],
    )
    object_material = choose_material(
        rng,
        materials,
        object_pool,
        visual_rules["fallback_object_material_pools_by_value"]["medium"],
        excluded,
        set(
            visual_rules["material_role_admission"]["dynamic_object"][
                "allowed_object_grades"
            ]
        ),
    )
    excluded.add(object_material["asset_id"])
    wall_source = visual_rules["wall_pools_by_environment"].get(
        environment_category
    )
    if not wall_source:
        wall_source = (
            visual_rules["wall_accent_pools_by_theme"][support_theme]
            if rng.random() < float(visual_rules["wall_accent_probability"])
            else visual_rules["wall_primary_pools_by_theme"][support_theme]
        )
    wall_material = choose_material(
        rng,
        materials,
        wall_source,
        visual_rules["wall_fallback_pool"],
        excluded,
    )
    excluded.add(wall_material["asset_id"])
    room_floor_source = visual_rules["room_floor_pools_by_environment"].get(
        environment_category, visual_rules["room_floor_pool"]
    )
    room_floor = choose_material(
        rng,
        materials,
        room_floor_source,
        visual_rules["room_floor_pool"],
        excluded,
    )
    structure_source = visual_rules.get(
        "structure_material_pools_by_support", {}
    ).get(support_label)
    if not structure_source:
        structure_source = visual_rules[
            "structure_material_pools_by_environment"
        ].get(
            environment_category,
            visual_rules["structure_material_pools_by_theme"][support_theme],
        )
    structure = choose_material(
        rng,
        materials,
        structure_source,
        visual_rules["fallback_object_material_pools_by_value"]["medium"],
        {support_material["asset_id"], room_floor["asset_id"]},
    )

    roles = visual_rules["hdri_roles_by_environment"].get(
        environment_category,
        visual_rules["hdri_roles_by_theme"].get(
            support_theme, ["studio_soft", "indoor_neutral"]
        ),
    )
    hdri_candidates = [
        record
        for record in hdri_records
        if record["role"] in roles and record["tier"] in {"primary", "secondary"}
    ]
    if not hdri_candidates:
        raise ValueError(f"no HDRI candidate for theme {support_theme}")
    weights = [float(record.get("sample_weight", 1.0)) for record in hdri_candidates]
    hdri = rng.choices(hdri_candidates, weights=weights, k=1)[0]
    scales = visual_rules["texture_scale_ranges"]
    hdri_strength_range = visual_rules[
        "hdri_strength_ranges_by_environment"
    ].get(environment_category, [0.26, 0.40])
    return {
        "visual_rules_version": str(visual_rules["version"]),
        "visual_rules_sha256": visual_rules_sha256,
        "surface_family": surface_family,
        "environment_category": environment_category,
        "contrast_policy": str(contrast["label"]),
        "materials": {
            "dynamic_object": {
                "record": object_material,
                "texture_scale": round(rng.uniform(*scales["dynamic_object"]), 4),
                "semantic_color_srgb": [float(value) for value in object_color],
                "semantic_color_mix": {
                    "plastic": 0.28,
                    "painted_metal": 0.24,
                    "rubber": 0.14,
                    "cardboard": 0.10,
                    "painted_wood": 0.08,
                    "stone": 0.06
                }.get(object_hint, 0.10),
            },
            "support_surface": {
                "record": support_material,
                "texture_scale": round(rng.uniform(*scales["support_surface"]), 4),
            },
            "support_structure": {
                "record": structure,
                "texture_scale": round(rng.uniform(*scales["support_structure"]), 4),
            },
            "room_floor": {
                "record": room_floor,
                "texture_scale": round(rng.uniform(*scales["room_floor"]), 4),
            },
            "back_wall": {
                "record": wall_material,
                "texture_scale": round(rng.uniform(*scales["back_wall"]), 4),
            },
        },
        "hdri": {
            "name": str(hdri["name"]),
            "path": str(hdri["source_path"]),
            "sha256": str(hdri["sha256"]),
            "role": str(hdri["role"]),
            "tier": str(hdri["tier"]),
            "strength": round(rng.uniform(*hdri_strength_range), 4),
            "rotation_degrees": round(rng.uniform(0.0, 360.0), 4),
        },
    }


def derive_initial_condition(
    rng: random.Random,
    backend: dict[str, Any],
    motion: str,
    subtype: dict[str, Any],
    trajectory_extent: dict[str, Any],
    zone: dict[str, Any],
    direction_record: dict[str, Any],
    shape: str,
    size_m: list[float],
    pose_profile: str,
    support: dict[str, Any],
    sampled_friction: float,
    restitution: float,
) -> dict[str, Any]:
    clearance = float(backend["contact"]["clearance_m"])
    yaw = (
        0.0
        if pose_profile == "side_on_motion"
        or support["support_shape"] in {"inclined_ramp", "tray_surface"}
        else rng.uniform(-25.0, 25.0)
    )
    direction = direction_unit(float(direction_record["angle_degrees"]))
    center_x, center_y, half_x, half_y = safe_half_extents(
        support, shape, size_m, pose_profile, direction
    )
    zone_x, zone_y = zone_offsets(zone, half_x, half_y, direction)
    limit = ray_limit(half_x, half_y, direction)
    trajectory_fraction = extent_fraction(str(trajectory_extent["label"]))
    desired_distance = clamp(2.0 * limit * trajectory_fraction, 0.32, 1.65)
    start_offset = min(limit * 0.78, desired_distance * 0.48)
    start_x = center_x + clamp_to_safe(
        -direction[0] * start_offset + zone_x, half_x
    )
    start_y = center_y + clamp_to_safe(
        -direction[1] * start_offset + zone_y, half_y
    )
    pose = pose_on_support(
        support,
        shape,
        size_m,
        start_x,
        start_y,
        yaw,
        clearance,
        pose_profile,
        direction,
    )
    try:
        minimum_active_duration_s = float(
            backend["quality"]["minimum_active_duration_s_by_motion"][motion]
        )
    except KeyError as error:
        raise ValueError(
            f"motion lacks a useful-duration contract: {motion}"
        ) from error
    expected: dict[str, Any] = {
        "motion_family": motion,
        "must_remain_finite": True,
        "minimum_displacement_m": 0.10,
        "minimum_active_duration_s": minimum_active_duration_s,
        "active_speed_threshold_m_s": float(
            backend["quality"]["active_speed_threshold_m_s"]
        ),
        "allow_support_exit_after_primary_motion": bool(
            backend["quality"]["allow_exit_after_primary_motion"]
        ),
    }
    context = MotionDerivationContext(
        rng=rng,
        backend=backend,
        motion=motion,
        subtype=subtype,
        trajectory_extent=trajectory_extent,
        shape=shape,
        size_m=size_m,
        pose_profile=pose_profile,
        support=support,
        sampled_friction=sampled_friction,
        restitution=restitution,
        clearance=clearance,
        yaw=yaw,
        direction=direction,
        center_x=center_x,
        center_y=center_y,
        half_x=half_x,
        half_y=half_y,
        zone_x=zone_x,
        zone_y=zone_y,
        limit=limit,
        desired_distance=desired_distance,
        trajectory_extent_fraction=trajectory_fraction,
    )
    plan = MotionPlan(
        pose=pose,
        linear_velocity_m_s=[0.0, 0.0, 0.0],
        angular_velocity_rad_s=[0.0, 0.0, 0.0],
        effective_contact_friction=sampled_friction,
        expected_motion=expected,
    )
    return derive_motion(context, plan).as_metadata()


def compile_camera_observation(
    camera_rules: dict[str, Any],
    motion: str,
    expected_motion: dict[str, Any],
) -> dict[str, Any]:
    try:
        observation = copy.deepcopy(camera_rules["motion_intents"][motion])
    except KeyError as error:
        raise ValueError(f"motion has no camera observation intent: {motion}") from error
    structure_context = str(observation["structure_context"])
    try:
        context_rules = camera_rules["structure_contexts"][structure_context]
    except KeyError as error:
        raise ValueError(
            f"camera observation uses unknown structure context: {structure_context}"
        ) from error
    focus_event = observation["focus_event"]
    if focus_event["type"] == "required_motion_collider":
        collider_id = expected_motion.get("required_collider_contact_id")
        if not collider_id:
            raise ValueError(
                f"camera observation for {motion} requires a motion collider"
            )
        focus_event["type"] = "collider_contact"
        focus_event["collider_id"] = str(collider_id)
    observation["minimum_anchor_visible_fraction"] = float(
        context_rules["minimum_anchor_visible_fraction"]
    )
    return {
        "version": str(camera_rules["version"]),
        **observation,
    }


def build_scene(
    rng: random.Random,
    seed: int,
    index: int,
    axes: dict[str, Any],
    architecture: dict[str, Any],
    camera_rules: dict[str, Any],
    backend: dict[str, Any],
    materials: dict[str, dict[str, Any]],
    hdri_records: list[dict[str, Any]],
    visual_rules: dict[str, Any],
    selection: dict[str, Any],
    support: dict[str, Any],
    provenance: dict[str, str],
) -> dict[str, Any]:
    motion = str(selection["motion"])
    subtype = copy.deepcopy(selection["subtype"])
    extent = selection["trajectory_extent"]
    obj = selection["object"]
    visual_profile = selection["visual_profile"]
    shape = str(obj["shape"])
    pose_profile = str(obj.get("pose_profile", "support_normal"))
    size_m, object_scale, scale_diagnostics = scaled_size(
        obj,
        selection["object_scale"],
        rng,
        backend["base_parameter_rules"]["object_scale_readability"],
    )
    effective_direction = copy.deepcopy(selection["direction"])
    if motion == "ramp_to_flat_1obj":
        effective_direction = {"label": "downhill", "angle_degrees": -90.0}
    motion_direction = direction_unit(float(effective_direction["angle_degrees"]))
    support_geometry = build_support_geometry(
        support, motion, subtype, motion_direction
    )
    validate_support_geometry(support_geometry)
    analytic_support_geometry = copy.deepcopy(support_geometry)
    support_visual_profile = copy.deepcopy(selection["support_visual_profile"])
    requested_support_visual_profile = copy.deepcopy(support_visual_profile)
    scene_composition = selection["scene_visual_profile"].get("composition")
    if (
        isinstance(scene_composition, dict)
        and str(scene_composition.get("composition_mode")) == "integrated_ground"
        and support_visual_profile["visual_type"] == "mesh_support"
    ):
        support_visual_profile = {
            "id": "procedural_support_proxy",
            "visual_type": "procedural_proxy",
            "support_ids": [str(support["label"])],
            "pre_freeze_rejection": {
                "requested_profile": str(requested_support_visual_profile["id"]),
                "reason": "source_environment_owns_action_surface_visual",
            },
        }
    if (
        support_visual_profile["visual_type"] == "mesh_support"
        and support_mesh_scale_ratio(support_geometry, support_visual_profile)
        > float(support_visual_profile["maximum_axis_scale_ratio"])
    ):
        support_visual_profile = {
            "id": "procedural_support_proxy",
            "visual_type": "procedural_proxy",
            "support_ids": [str(support["label"])],
            "pre_freeze_rejection": {
                "requested_profile": str(
                    requested_support_visual_profile["id"]
                ),
                "reason": "exact_support_axis_scale_ratio_exceeds_policy",
            },
        }
    if support_visual_profile["visual_type"] == "mesh_support":
        bind_exact_static_support(
            architecture,
            support_geometry,
            support_visual_profile,
        )
        try:
            safe_half_extents(
                support_geometry,
                shape,
                size_m,
                pose_profile,
                motion_direction,
            )
        except IncompatibleSupportVisual:
            support_geometry = analytic_support_geometry
            support_geometry["collision_authority"] = "analytic_scene_kit"
            support_visual_profile = {
                "id": "procedural_support_proxy",
                "visual_type": "procedural_proxy",
                "support_ids": [str(support["label"])],
                "pre_freeze_rejection": {
                    "requested_profile": str(
                        requested_support_visual_profile["id"]
                    ),
                    "reason": "exact_support_safe_surface_too_small_for_object",
                },
            }
    else:
        support_geometry["collision_authority"] = "analytic_scene_kit"

    sampled_friction = clamp(float(obj["friction"]) * rng.uniform(0.82, 1.08), 0.03, 0.95)
    mass_kg = log_uniform(rng, float(obj["mass"][0]), float(obj["mass"][1]))
    restitution = restitution_for_motion(rng, backend, motion, float(obj["restitution"]))
    initial = derive_initial_condition(
        rng,
        backend,
        motion,
        subtype,
        extent,
        selection["initial_zone"],
        effective_direction,
        shape,
        size_m,
        pose_profile,
        support_geometry,
        sampled_friction,
        restitution,
    )
    camera_observation = compile_camera_observation(
        camera_rules, motion, initial["expected_motion"]
    )
    appearance = choose_appearance(
        rng,
        materials,
        hdri_records,
        str(visual_profile["material_hint"]),
        str(support["label"]),
        str(support["theme"]),
        str(selection["scene_visual_profile"]["environment_category"]),
        str(selection["surface_family"]),
        selection["contrast"],
        visual_rules,
        list(visual_profile["color"]),
        provenance["visual_sampling_sha256"],
    )
    appearance["scene_visual"] = copy.deepcopy(selection["scene_visual_profile"])
    appearance["support_visual"] = copy.deepcopy(support_visual_profile)
    scene_id = (
        f"physweeprigid_{index:06d}_{motion.removesuffix('_1obj')}_"
        f"{subtype['label']}_{obj['label']}_{support['label']}"
    )
    contact = backend["contact"]
    rolling = float(contact["rolling_friction_by_shape"][shape])
    spinning = float(contact["spinning_friction_by_shape"][shape])
    simulation_hz = simulation_hz_for_geometry(backend["engine"], size_m)
    frame_count = int(round(float(backend["engine"]["duration_s"]) * int(backend["engine"]["output_fps"]))) + 1
    planar_size_m = max(float(size_m[0]), float(size_m[1]))
    minimum_initial_object_span_ndc = float(
        backend["quality"]["minimum_initial_object_span_ndc"]
    )
    maximum_initial_object_span_ndc = min(
        float(backend["quality"]["maximum_initial_object_span_cap_ndc"]),
        float(backend["quality"]["maximum_initial_object_span_ndc"])
        * math.sqrt(
            max(
                1.0,
                planar_size_m
                / float(backend["quality"]["camera_object_span_reference_m"]),
            )
        ),
    )
    maximum_camera_distance_above_minimum_m = float(
        camera_observation.get(
            "maximum_camera_distance_above_minimum_m",
            backend["quality"]["maximum_camera_distance_above_minimum_m"],
        )
    )
    elevation_range = camera_observation["elevation_range_degrees"]
    initial_speed = math.sqrt(
        sum(float(value) ** 2 for value in initial["linear_velocity_m_s"])
    )
    initial_height_above_floor = max(0.0, float(initial["pose"]["position_m"][2]))
    gravity = abs(float(backend["engine"]["gravity_m_s2"][2]))
    energy_consistent_speed = math.sqrt(
        initial_speed * initial_speed
        + 2.0 * gravity * initial_height_above_floor
    )
    maximum_admissible_linear_speed = max(
        float(backend["quality"]["maximum_linear_speed_m_s"]),
        1.15 * energy_consistent_speed,
    )
    metadata = {
        "schema_version": "physweep_pybullet_rigid_metadata_v1",
        "scene_id": scene_id,
        "dataset_id": "physweep",
        "dataset_stage": "one_object_base_candidate",
        "seed": seed,
        "sample_index": index,
        "semantic_sampling": {
            "sampling_bundle_version": provenance["bundle_version"],
            "matrix_version": provenance["matrix_version"],
            "five_dimensions": {
                "motion": {
                    "family": motion,
                    "subtype": str(subtype["label"]),
                    "direction": str(effective_direction["label"]),
                    "direction_angle_degrees": float(effective_direction["angle_degrees"]),
                    "trajectory_extent": str(extent["label"]),
                    "initial_position_zone": str(selection["initial_zone"]["label"]),
                },
                "foreground_object": {
                    "object_type": str(obj["label"]),
                    "semantic_category": str(obj["semantic_category"]),
                    "shape": shape,
                    "pose_profile": pose_profile,
                    "material_hint": str(visual_profile["material_hint"]),
                    "visual_type": str(visual_profile["type"]),
                    "visual_variant_id": str(visual_profile["id"]),
                    "visual_asset_id": visual_profile.get("asset_id"),
                    "scale_bin": str(selection["object_scale"]["label"]),
                    "uniform_scale": round(object_scale, 6),
                    "scale_readability": scale_diagnostics,
                },
                "support_interaction": {
                    "scene_class": str(support["scene_class"]),
                    "support_type": str(support["label"]),
                    "support_layout": str(support_geometry["layout"]),
                    "support_shape": str(support_geometry["support_shape"]),
                    "scene_theme": str(support["theme"]),
                    "scene_visual_profile": str(
                        selection["scene_visual_profile"]["id"]
                    ),
                    "scene_visual_type": str(
                        selection["scene_visual_profile"]["visual_type"]
                    ),
                    "support_visual_profile": str(
                        support_visual_profile["id"]
                    ),
                    "support_visual_type": str(
                        support_visual_profile["visual_type"]
                    ),
                    "collision_authority": str(
                        support_geometry["collision_authority"]
                    ),
                },
                "camera_observation": {
                    "camera_profile": str(selection["camera"]["label"]),
                    "observation_intent": str(camera_observation["intent"]),
                    "structure_context": str(
                        camera_observation["structure_context"]
                    ),
                },
                "appearance_lighting": {
                    "surface_family": str(selection["surface_family"]),
                    "environment_category": str(
                        selection["scene_visual_profile"][
                            "environment_category"
                        ]
                    ),
                    "contrast_policy": str(selection["contrast"]["label"]),
                    "hdri_role": str(appearance["hdri"]["role"]),
                },
            },
        },
        "simulation": {
            "backend": {
                "id": str(backend["backend_id"]),
                "config_version": str(backend["version"]),
                "config_sha256": provenance["backend_sha256"],
            },
            "time": {
                "duration_s": float(backend["engine"]["duration_s"]),
                "output_fps": int(backend["engine"]["output_fps"]),
                "simulation_hz": simulation_hz,
                "frame_count": frame_count,
            },
            "world": {"gravity_m_s2": list(backend["engine"]["gravity_m_s2"])},
            "solver": {
                "iterations": int(backend["engine"]["solver_iterations"]),
                "deterministic_overlapping_pairs": bool(backend["engine"]["deterministic_overlapping_pairs"]),
                "restitution_velocity_threshold_m_s": float(backend["engine"]["restitution_velocity_threshold_m_s"]),
                "contact_breaking_threshold_m": float(backend["engine"]["contact_breaking_threshold_m"]),
            },
            "support": {
                **support_geometry,
                "dynamics": {
                    "lateral_friction": float(contact["support_lateral_friction"]),
                    "restitution": float(contact["support_restitution"]),
                },
            },
            "objects": [
                {
                    "object_id": "object_a",
                    "body_model": "rigid_body",
                    "semantic_type": str(obj["label"]),
                    "geometry": {"type": shape, "size_m": size_m},
                    "visual_profile": copy.deepcopy(visual_profile),
                    "collision_profile": copy.deepcopy(obj["collision_profile"]),
                    "material": {
                        "mass_kg": round(mass_kg, 6),
                        "contact_friction": initial["effective_contact_friction"],
                        "source_nominal_contact_friction": round(sampled_friction, 6),
                        "contact_restitution": round(restitution, 6),
                        "linear_damping": float(contact["linear_damping"]),
                        "angular_damping": float(contact["angular_damping"]),
                        "rolling_friction": rolling,
                        "spinning_friction": spinning,
                    },
                    "initial_state": {
                        "pose_profile": pose_profile,
                        **initial["pose"],
                        "linear_velocity_m_s": initial["linear_velocity_m_s"],
                        "angular_velocity_rad_s": initial["angular_velocity_rad_s"],
                    },
                    "expected_motion": initial["expected_motion"],
                }
            ],
        },
        "appearance": appearance,
        "camera_request": {
            "profile": str(selection["camera"]["label"]),
            "observation": camera_observation,
            "focal_length_mm": min(44.0, float(selection["camera"]["focal_length_mm"])),
            "minimum_full_trajectory_center_visible_fraction": float(
                camera_observation.get(
                    "minimum_full_trajectory_center_visible_fraction",
                    backend["quality"]["minimum_full_trajectory_center_visible_fraction"],
                )
            ),
            "minimum_primary_trajectory_center_visible_fraction": float(
                backend["quality"]["minimum_primary_trajectory_center_visible_fraction"]
            ),
            "full_trajectory_camera_target_blend": float(
                camera_observation.get(
                    "full_trajectory_camera_target_blend",
                    backend["quality"]["full_trajectory_camera_target_blend"],
                )
            ),
            "minimum_initial_object_span_ndc": round(
                minimum_initial_object_span_ndc, 6
            ),
            "minimum_initial_object_visible_fraction": float(
                backend["quality"]["minimum_initial_object_visible_fraction"]
            ),
            "initial_object_center_margin_ndc": float(
                backend["quality"]["initial_object_center_margin_ndc"]
            ),
            "maximum_initial_object_span_ndc": round(
                maximum_initial_object_span_ndc, 6
            ),
            "minimum_support_context_visible_fraction": float(
                backend["quality"]["minimum_support_context_visible_fraction"]
            ),
            "minimum_primary_trajectory_unoccluded_fraction": float(
                backend["quality"]["minimum_primary_trajectory_unoccluded_fraction"]
            ),
            "minimum_full_trajectory_unoccluded_fraction": float(
                backend["quality"]["minimum_full_trajectory_unoccluded_fraction"]
            ),
            "minimum_camera_distance_floor_m": float(
                backend["quality"]["minimum_camera_distance_floor_m"]
            ),
            "minimum_camera_distance_offset_m": float(
                backend["quality"]["minimum_camera_distance_offset_m"]
            ),
            "minimum_camera_distance_support_diagonal_scale": float(
                backend["quality"]["minimum_camera_distance_support_diagonal_scale"]
            ),
            "preferred_camera_distance_offset_m": float(
                backend["quality"]["preferred_camera_distance_offset_m"]
            ),
            "camera_distance_penalty_weight": float(
                backend["quality"]["camera_distance_penalty_weight"]
            ),
            "maximum_camera_distance_above_minimum_m": float(
                maximum_camera_distance_above_minimum_m
            ),
            "minimum_camera_elevation_degrees": float(elevation_range[0]),
            "maximum_camera_elevation_degrees": float(elevation_range[1]),
            "soft_maximum_focus_span_ndc": float(
                backend["quality"]["soft_maximum_focus_span_ndc"]
            ),
            "maximum_focus_span_ndc": float(
                backend["quality"]["maximum_focus_span_ndc"]
            ),
            "focus_span_penalty_weight": float(
                backend["quality"]["focus_span_penalty_weight"]
            ),
            "maximum_camera_distance_m": float(
                backend["quality"]["maximum_camera_distance_m"]
            ),
            "allow_partial_exit": True,
        },
        "render_request": {
            "resolution": [1280, 720],
            "samples": 16,
            "engine": "BLENDER_EEVEE",
            "fps": int(backend["engine"]["output_fps"]),
        },
        "qa": {
            "status": "sampled_pending_simulation",
            "limits": {
                "maximum_trajectory_penetration_m": float(
                    backend["quality"]["maximum_trajectory_penetration_m"]
                ),
                "maximum_trajectory_penetration_fraction_of_min_extent": float(
                    backend["quality"][
                        "maximum_trajectory_penetration_fraction_of_min_extent"
                    ]
                ),
                "maximum_initial_penetration_m": float(
                    backend["quality"]["maximum_initial_penetration_m"]
                ),
                "maximum_linear_speed_m_s": round(
                    maximum_admissible_linear_speed, 6
                ),
                "maximum_angular_speed_rad_s": float(
                    backend["quality"]["maximum_angular_speed_rad_s"]
                ),
                "maximum_rotational_surface_speed_m_s": float(
                    backend["quality"]["maximum_rotational_surface_speed_m_s"]
                ),
                "rolling_coupling_ratio_range": list(
                    backend["quality"]["rolling_coupling_ratio_range"]
                ),
            },
        },
    }
    metadata["environment_binding"] = compile_environment_binding(
        metadata, axes["camera_axis"]
    )
    return metadata


def build_batch(
    rules: dict[str, Any],
    backend: dict[str, Any],
    materials: dict[str, dict[str, Any]],
    hdri_records: list[dict[str, Any]],
    visual_rules: dict[str, Any],
    seed: int,
    count: int,
    provenance: dict[str, str] | None = None,
    motion_sequence: list[str] | None = None,
    support_sequence: list[str] | None = None,
    object_sequence: list[str] | None = None,
) -> list[dict[str, Any]]:
    if provenance is None:
        bundle = load_json(BUNDLE_PATH)
        provenance = {
            "bundle_version": str(rules["architecture"]["bundle_version"]),
            "matrix_version": str(rules["version"]),
            "backend_sha256": sha256(PROJECT_ROOT / bundle["backend"]),
            "visual_sampling_sha256": sha256(
                PROJECT_ROOT / bundle["visual_sampling"]
            ),
        }
    rng = random.Random(seed)
    axes = rules["axes"]
    if motion_sequence is None:
        motion_object_pairs = balanced_motion_object_pairs(
            axes["motion_axis"], axes["object_axis"], count, rng
        )
        motions = [motion for motion, _ in motion_object_pairs]
        objects = [obj for _, obj in motion_object_pairs]
    else:
        motions = [str(motion) for motion in motion_sequence]
        if len(motions) != count:
            raise ValueError("motion sequence length must match sample count")
        unknown = set(motions) - set(axes["motion_axis"])
        if unknown:
            raise ValueError(f"motion sequence contains unknown motions: {sorted(unknown)}")
        objects = balanced_objects_for_motions(motions, axes["object_axis"], rng)
    if object_sequence is not None:
        object_labels = [str(label) for label in object_sequence]
        if len(object_labels) != count:
            raise ValueError("object sequence length must match sample count")
        object_by_label = {
            str(record["label"]): record for record in axes["object_axis"]
        }
        unknown = set(object_labels) - set(object_by_label)
        if unknown:
            raise ValueError(
                f"object sequence contains unknown profiles: {sorted(unknown)}"
            )
        objects = [copy.deepcopy(object_by_label[label]) for label in object_labels]
        for motion, obj in zip(motions, objects):
            if not object_supports_motion(obj, motion):
                raise ValueError(
                    f"object {obj['label']} is incompatible with motion {motion}"
                )
    explicit_supports: list[dict[str, Any]] | None = None
    if support_sequence is not None:
        support_labels = [str(label) for label in support_sequence]
        if len(support_labels) != count:
            raise ValueError("support sequence length must match sample count")
        support_by_label = {
            str(record["label"]): record for record in axes["support_axis"]
        }
        unknown = set(support_labels) - set(support_by_label)
        if unknown:
            raise ValueError(
                f"support sequence contains unknown supports: {sorted(unknown)}"
            )
        explicit_supports = [
            copy.deepcopy(support_by_label[label]) for label in support_labels
        ]
        for motion, support in zip(motions, explicit_supports):
            if not support_allowed(
                support, motion, rules, str(support["scene_class"])
            ):
                raise ValueError(
                    f"support {support['label']} is incompatible with motion {motion}"
                )
    visual_profiles = balanced_visual_variants(
        objects,
        float(rules["architecture"]["visual_sampling"]["target_mesh_fraction"]),
        rng,
    )
    camera_rng = random.Random((seed << 1) ^ 0x5A17C3)
    cameras = coverage_cycle_by_group(motions, axes["camera_axis"], camera_rng)
    directions = repeated_shuffled(axes["motion_direction_axis"], count, rng)
    scales = repeated_shuffled(axes["object_scale_axis"], count, rng)
    extents = repeated_shuffled(axes["trajectory_extent_axis"], count, rng)
    zones = repeated_shuffled(axes["initial_position_zone_axis"], count, rng)
    contrasts = repeated_shuffled(axes["surface_contrast_axis"], count, rng)
    surface_families = repeated_shuffled(
        list(visual_rules["surface_material_pools_by_family"]), count, rng
    )
    motion_counts = Counter(motions)
    scene_class_cycles = {
        str(motion): weighted_scene_class_cycle(
            axes["scene_class_axis"],
            axes["support_axis"],
            str(motion),
            motion_counts[str(motion)],
            rng,
            rules,
        )
        for motion in axes["motion_axis"]
    }
    support_cycles: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for motion in axes["motion_axis"]:
        for scene_class in scene_class_cycles[str(motion)]:
            key = (support_compatibility_group(str(motion), rules), scene_class)
            candidates = [
                copy.deepcopy(record)
                for record in axes["support_axis"]
                if support_allowed(record, str(motion), rules, scene_class)
            ]
            if not candidates:
                raise ValueError(f"no support is admitted for {motion} in {scene_class}")
            if key in support_cycles:
                existing = {str(record["label"]) for record in support_cycles[key]}
                current = {str(record["label"]) for record in candidates}
                if existing != current:
                    raise ValueError(f"inconsistent support compatibility group: {key}")
                continue
            rng.shuffle(candidates)
            support_cycles[key] = candidates
    scene_class_seen: Counter[str] = Counter()
    support_seen: Counter[tuple[str, str]] = Counter()
    subtype_seen: Counter[str] = Counter()
    scene_visual_rng = random.Random(seed ^ 0x5CE1E)
    scene_visual_types = balanced_scene_visual_types(
        len(motions),
        float(
            rules["architecture"]["scene_visual_profiles"]["sampling"][
                "target_mesh_fraction"
            ]
        ),
        scene_visual_rng,
    )
    visual_profiles_by_support_and_type: dict[
        tuple[str, str], list[dict[str, Any]]
    ] = {}
    support_records: dict[str, dict[str, Any]] = {}
    for support in axes["support_axis"]:
        support_id = str(support["label"])
        if support_id in support_records:
            raise ValueError(f"duplicate support axis record: {support_id}")
        support_records[support_id] = support
        for visual_type in ("procedural_room", "mesh_backdrop"):
            profiles = [
                copy.deepcopy(profile)
                for profile in rules["architecture"]["scene_visual_profiles"][
                    "profiles"
                ]
                if str(profile["visual_type"]) == visual_type
                and scene_visual_profile_admits_support(profile, support)
            ]
            if not profiles and visual_type == "procedural_room":
                raise ValueError(
                    f"no {visual_type} scene visual profile admits support={support_id} "
                    f"theme={support['theme']} scene_class={support['scene_class']}"
                )
            scene_visual_rng.shuffle(profiles)
            visual_profiles_by_support_and_type[(support_id, visual_type)] = profiles
    scene_visual_seen: Counter[tuple[str, str]] = Counter()
    scene_visual_global_seen: Counter[str] = Counter()
    environment_category_seen: Counter[str] = Counter()
    support_visual_rng = random.Random(seed ^ 0x5A7707)
    support_mesh_profiles_by_id: dict[str, list[dict[str, Any]]] = {}
    for profile in rules["architecture"]["support_mesh_profiles"]["profiles"]:
        for support_id in profile["support_ids"]:
            support_mesh_profiles_by_id.setdefault(str(support_id), []).append(
                copy.deepcopy(profile)
            )
    support_visual_cycles: dict[str, list[str]] = {}
    mesh_fraction = Fraction(
        float(
            rules["architecture"]["support_mesh_profiles"]["sampling"][
                "target_mesh_fraction_per_support"
            ]
        )
    ).limit_denominator(20)
    for support_id, profiles in support_mesh_profiles_by_id.items():
        support_visual_rng.shuffle(profiles)
        cycle = ["mesh_support"] * mesh_fraction.numerator
        cycle.extend(
            ["procedural_proxy"]
            * (mesh_fraction.denominator - mesh_fraction.numerator)
        )
        support_visual_rng.shuffle(cycle)
        support_visual_cycles[support_id] = cycle
    support_visual_seen: Counter[str] = Counter()
    support_mesh_seen: Counter[str] = Counter()
    scenes = []
    for offset, motion in enumerate(motions):
        subtypes = axes["motion_subtype_axis"][motion]
        subtype = copy.deepcopy(subtypes[subtype_seen[motion] % len(subtypes)])
        subtype_seen[motion] += 1
        if explicit_supports is None:
            scene_class = scene_class_cycles[motion][scene_class_seen[motion]]
            scene_class_seen[motion] += 1
            support_key = (support_compatibility_group(motion, rules), scene_class)
            candidates = support_cycles[support_key]
            support = copy.deepcopy(
                candidates[support_seen[support_key] % len(candidates)]
            )
            support_seen[support_key] += 1
        else:
            support = copy.deepcopy(explicit_supports[offset])
        scene_visual_type = scene_visual_types[offset]
        support_id = str(support["label"])
        visual_key = (support_id, scene_visual_type)
        scene_visual_profiles = [
            profile
            for profile in visual_profiles_by_support_and_type[visual_key]
            if scene_visual_profile_admits_camera(
                profile,
                motion,
                str(subtype["label"]),
                str(extents[offset]["label"]),
                support,
                rules["camera_observation"],
                camera_profile=cameras[offset],
            )
        ]
        if not scene_visual_profiles and scene_visual_type == "mesh_backdrop":
            scene_visual_type = "procedural_room"
            visual_key = (support_id, scene_visual_type)
            scene_visual_profiles = [
                profile
                for profile in visual_profiles_by_support_and_type[visual_key]
                if scene_visual_profile_admits_camera(
                    profile,
                    motion,
                    str(subtype["label"]),
                    str(extents[offset]["label"]),
                    support,
                    rules["camera_observation"],
                    camera_profile=cameras[offset],
                )
            ]
        if not scene_visual_profiles:
            raise ValueError(
                "no scene visual profile has camera constraints compatible with "
                f"motion={motion} support={support_id} visual_type={scene_visual_type}"
            )
        unseen_profiles = [
            profile
            for profile in scene_visual_profiles
            if scene_visual_global_seen[str(profile["id"])] == 0
        ]
        profile_pool = unseen_profiles or scene_visual_profiles
        minimum_category_usage = min(
            environment_category_seen[str(profile["environment_category"])]
            for profile in profile_pool
        )
        least_used_categories = {
            str(profile["environment_category"])
            for profile in profile_pool
            if environment_category_seen[
                str(profile["environment_category"])
            ]
            == minimum_category_usage
        }
        category_balanced_profiles = [
            profile
            for profile in profile_pool
            if str(profile["environment_category"]) in least_used_categories
        ]
        minimum_usage = min(
            scene_visual_global_seen[str(profile["id"])]
            for profile in category_balanced_profiles
        )
        least_used_profiles = [
            profile
            for profile in category_balanced_profiles
            if scene_visual_global_seen[str(profile["id"])] == minimum_usage
        ]
        scene_visual_profile = copy.deepcopy(
            least_used_profiles[
                scene_visual_seen[visual_key] % len(least_used_profiles)
            ]
        )
        scene_visual_seen[visual_key] += 1
        scene_visual_global_seen[str(scene_visual_profile["id"])] += 1
        environment_category_seen[
            str(scene_visual_profile["environment_category"])
        ] += 1
        support_id = str(support["label"])
        if support_id in support_mesh_profiles_by_id:
            cycle = support_visual_cycles[support_id]
            support_visual_type = cycle[
                support_visual_seen[support_id] % len(cycle)
            ]
            support_visual_seen[support_id] += 1
        else:
            support_visual_type = "procedural_proxy"
        if support_visual_type == "mesh_support":
            policy = rules["architecture"]["support_mesh_profiles"]["policy"]
            profiles = [
                profile
                for profile in support_mesh_profiles_by_id[support_id]
                if support_mesh_scale_ratio(support, profile)
                <= float(policy["maximum_axis_scale_ratio"])
            ]
            if profiles:
                support_visual_profile = copy.deepcopy(
                    profiles[support_mesh_seen[support_id] % len(profiles)]
                )
                support_visual_profile.update(copy.deepcopy(policy))
                support_mesh_seen[support_id] += 1
            else:
                support_visual_type = "procedural_proxy"
        if support_visual_type == "procedural_proxy":
            support_visual_profile = {
                "id": "procedural_support_proxy",
                "visual_type": "procedural_proxy",
                "support_ids": [support_id],
            }
        selection = {
            "motion": motion,
            "subtype": subtype,
            "object": objects[offset],
            "visual_profile": visual_profiles[offset],
            "camera": cameras[offset],
            "direction": directions[offset],
            "object_scale": scales[offset],
            "trajectory_extent": constrained_trajectory_extent(
                motion, extents[offset], axes, rules
            ),
            "initial_zone": zones[offset],
            "contrast": contrasts[offset],
            "surface_family": surface_families[offset],
            "scene_visual_profile": scene_visual_profile,
            "support_visual_profile": support_visual_profile,
        }
        scene = build_scene(
            rng,
            seed,
            offset + 1,
            axes,
            rules["architecture"],
            rules["camera_observation"],
            backend,
            materials,
            hdri_records,
            visual_rules,
            selection,
            support,
            provenance,
        )
        attach_object_identity(scene)
        scenes.append(scene)
    return scenes


def manifest_counts(scenes: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    dimensions = [scene["semantic_sampling"]["five_dimensions"] for scene in scenes]
    return {
        "motion": dict(Counter(item["motion"]["family"] for item in dimensions)),
        "object": dict(Counter(item["foreground_object"]["object_type"] for item in dimensions)),
        "visual_type": dict(Counter(item["foreground_object"]["visual_type"] for item in dimensions)),
        "scene_class": dict(Counter(item["support_interaction"]["scene_class"] for item in dimensions)),
        "support": dict(Counter(item["support_interaction"]["support_type"] for item in dimensions)),
        "scene_visual": dict(
            Counter(
                item["support_interaction"]["scene_visual_profile"]
                for item in dimensions
            )
        ),
        "scene_visual_type": dict(
            Counter(
                item["support_interaction"]["scene_visual_type"]
                for item in dimensions
            )
        ),
        "support_visual": dict(
            Counter(
                item["support_interaction"]["support_visual_profile"]
                for item in dimensions
            )
        ),
        "support_visual_type": dict(
            Counter(
                item["support_interaction"]["support_visual_type"]
                for item in dimensions
            )
        ),
        "camera": dict(
            Counter(
                item["camera_observation"]["camera_profile"]
                for item in dimensions
            )
        ),
        "surface_family": dict(Counter(item["appearance_lighting"]["surface_family"] for item in dimensions)),
        "environment_category": dict(
            Counter(
                item["appearance_lighting"]["environment_category"]
                for item in dimensions
            )
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output-dataset", default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--resolution", nargs=2, type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument(
        "--motions",
        nargs="+",
        help="Optional exact motion sequence supplied by the decoupled outer sampler",
    )
    parser.add_argument(
        "--supports",
        nargs="+",
        help="Optional exact support sequence for structure-level validation",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        help="Optional exact object-profile sequence for asset-level validation",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    root = args.root.resolve()
    output_root = root / "datasets" / args.output_dataset
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    bundle_path = (
        args.bundle.resolve()
        if args.bundle
        else root / BUNDLE_PATH.relative_to(PROJECT_ROOT)
    )
    bundle = load_json(bundle_path)
    rules = load_sampling_bundle(root, bundle_path)
    dependency_paths = {
        key: root / str(bundle[key])
        for key in (
            "base_rules",
            "object_profiles",
            "object_visual_preflight",
            "object_visual_preflight_report",
            "object_visual_curation",
            "object_visual_repairs",
            "scene_kits",
            "scene_visual_profiles",
            "scene_mesh_profiles",
            "environment_composition",
            "support_mesh_profiles",
            "asset_proxy_registry",
            "physical_proxy_catalog",
            "compatibility",
            "visual_sampling",
            "backend",
            "backend_capabilities",
            "material_manifest",
            "hdri_manifest",
        )
    }
    implementation_paths = {
        key: root / str(path)
        for key, path in bundle["implementation"].items()
    }
    backend = copy.deepcopy(load_json(dependency_paths["backend"]))
    if args.duration is not None:
        backend["engine"]["duration_s"] = float(args.duration)
    if args.fps is not None:
        backend["engine"]["output_fps"] = int(args.fps)
    if float(backend["engine"]["duration_s"]) <= 0.0:
        raise SystemExit("--duration must be positive")
    if int(backend["engine"]["output_fps"]) <= 0:
        raise SystemExit("--fps must be positive")
    material_manifest = load_json(dependency_paths["material_manifest"])
    hdri_manifest = load_json(dependency_paths["hdri_manifest"])
    visual_rules = load_json(dependency_paths["visual_sampling"])
    materials = {str(record["asset_id"]): record for record in material_manifest["assets"]}
    scenes = build_batch(
        rules,
        backend,
        materials,
        list(hdri_manifest["records"]),
        visual_rules,
        args.seed,
        args.count,
        {
            "bundle_version": str(bundle["version"]),
            "matrix_version": str(rules["version"]),
            "backend_sha256": sha256(dependency_paths["backend"]),
            "visual_sampling_sha256": sha256(
                dependency_paths["visual_sampling"]
            ),
        },
        motion_sequence=args.motions,
        support_sequence=args.supports,
        object_sequence=args.objects,
    )
    resolution = args.resolution or [1280, 720]
    render_samples = args.samples if args.samples is not None else 16
    if min(resolution) <= 0 or render_samples <= 0:
        raise SystemExit("render resolution and samples must be positive")
    for scene in scenes:
        scene["render_request"]["resolution"] = list(resolution)
        scene["render_request"]["samples"] = int(render_samples)
    samples = []
    for scene in scenes:
        metadata_path = output_root / "scenes" / scene["scene_id"] / "metadata.json"
        write_json(metadata_path, scene)
        samples.append(
            {
                "scene_id": scene["scene_id"],
                "metadata_path": str(metadata_path.relative_to(root)),
                "metadata_sha256": sha256(metadata_path),
            }
        )
    rule_sources = sampling_manifest_rule_sources(root, bundle_path, bundle)
    manifest = {
        "schema_version": "physweep_pybullet_base_manifest_v1",
        "dataset_id": args.output_dataset,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "sample_count": len(samples),
        "production_spec": {
            "duration_s": float(backend["engine"]["duration_s"]),
            "output_fps": int(backend["engine"]["output_fps"]),
            "frame_count": int(
                round(
                    float(backend["engine"]["duration_s"])
                    * int(backend["engine"]["output_fps"])
                )
            )
            + 1,
            "resolution": list(resolution),
            "samples": int(render_samples),
        },
        **rule_sources,
        "compiled_from": {
            key: {
                "path": str(path.relative_to(root)),
                "sha256": sha256(path),
            }
            for key, path in dependency_paths.items()
        },
        "implementation": {
            key: {
                "path": str(path.relative_to(root)),
                "sha256": sha256(path),
            }
            for key, path in implementation_paths.items()
        },
        "backend_path": str(dependency_paths["backend"].relative_to(root)),
        "backend_sha256": sha256(dependency_paths["backend"]),
        "coverage": manifest_counts(scenes),
        "samples": samples,
        "status": "sampled_pending_simulation",
    }
    write_json(output_root / "manifest.json", manifest)
    print(f"dataset: {output_root}")
    print(f"samples: {len(samples)}")
    print(json.dumps(manifest["coverage"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
