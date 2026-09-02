#!/usr/bin/env python3
"""Build the declared two-object coverage matrix from released 1obj metadata."""

from __future__ import annotations

import argparse
import copy
import hashlib
import math
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Iterator, Sequence

from tools.core.camera_geometry import (
    pair_camera_geometry_eligible,
)
from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json, write_json
from tools.core.rigid_geometry import pose_on_support
from tools.motion_rules.two_object.motion import (
    apply_two_object_motion,
    interaction_approach_axis,
)
from tools.sampling.sample_two_object_base import (
    DATASET_ID,
    DEFAULT_MATRIX,
    _validated_intents,
    build_two_object_scene,
    camera_view_family,
    compatible_scale_pair_ids,
    compatible_shape_pair_ids,
)
from tools.scene_rules.two_object import (
    DEFAULT_TWO_OBJECT_SCENE_RULES,
    allowed_camera_view_families,
    allowed_scene_classes,
    load_two_object_scene_rules,
    resolved_two_object_scene_rules,
)
from tools.sampling.two_object_sources import declared_within, released_source_pool


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _rank(seed: int, *parts: object) -> str:
    value = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _axis_counts(cells: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    fields = {
        "interaction_class": "interaction_class",
        "motion": "motion_id",
        "ordered_shape_pair": "shape_pair_id",
        "ordered_scale_pair": "scale_pair_id",
        "scene_class": "scene_class",
        "scene_rule": "scene_rule_id",
        "camera_view_family": "camera_view_family_id",
        "source_family_pair": "source_family_pair_id",
        "visual_environment": "visual_environment_category",
    }
    return {
        label: dict(sorted(Counter(str(cell[field]) for cell in cells).items()))
        for label, field in fields.items()
        if cells and all(field in cell for cell in cells)
    }


def _balanced_cell_order(
    cells: Sequence[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Balance axes within each interaction class, then preserve their full mix."""

    classes = ("interacting", "independent")
    if {str(cell["interaction_class"]) for cell in cells} != set(classes):
        raise ValueError("coverage cells require both interaction classes")

    def balanced_class_order(
        class_cells: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        remaining = [copy.deepcopy(cell) for cell in class_cells]
        axis_fields = (
            "motion_id",
            "shape_pair_id",
            "scale_pair_id",
            "scene_class",
        )
        levels = {
            field: sorted({str(cell[field]) for cell in remaining})
            for field in axis_fields
        }
        counts = {field: Counter() for field in axis_fields}
        ordered = []
        while remaining:
            def score(cell: dict[str, Any]) -> tuple[int, int, int, str]:
                ranges = []
                for field in axis_fields:
                    values = [counts[field][level] for level in levels[field]]
                    values[levels[field].index(str(cell[field]))] += 1
                    ranges.append(max(values) - min(values))
                return (
                    ranges[0],
                    max(ranges[1:]),
                    sum(ranges[1:]),
                    _rank(seed, "coverage-cell", cell["cell_id"]),
                )

            selected = min(remaining, key=score)
            remaining.remove(selected)
            ordered.append(selected)
            for field in axis_fields:
                counts[field][str(selected[field])] += 1
        return ordered

    by_class = {
        interaction_class: balanced_class_order(
            [
                cell
                for cell in cells
                if str(cell["interaction_class"]) == interaction_class
            ]
        )
        for interaction_class in classes
    }
    interacting_fraction = len(by_class["interacting"]) / len(cells)
    ordered = []
    class_indices = Counter()
    for prefix_size in range(1, len(cells) + 1):
        required_interacting = math.ceil(interacting_fraction * prefix_size)
        interaction_class = (
            "interacting"
            if class_indices["interacting"] < required_interacting
            else "independent"
        )
        if class_indices[interaction_class] >= len(by_class[interaction_class]):
            interaction_class = (
                "independent"
                if interaction_class == "interacting"
                else "interacting"
            )
        ordered.append(by_class[interaction_class][class_indices[interaction_class]])
        class_indices[interaction_class] += 1
    return ordered


def _assign_camera_view_families(
    cells: Sequence[dict[str, Any]],
    view_families: Sequence[dict[str, Any]],
    scene_rules: dict[str, Any],
    intents_by_id: dict[str, dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    """Balance views globally and cover compatible views in every stratum."""

    family_ids = [str(record["id"]) for record in view_families]
    declared_family_ids = set(family_ids)
    rule_family_ids = {
        str(family_id)
        for rule in scene_rules["physical_rules"]
        for family_id in rule["allowed_camera_view_families"]
    }
    if rule_family_ids != declared_family_ids:
        raise ValueError(
            "two-object scene and sampling camera families differ"
        )
    declared_motion_ids = set(intents_by_id)
    for rule in scene_rules["physical_rules"]:
        overrides = rule["camera_view_family_overrides"]
        if not overrides:
            continue
        compatible_motion_ids = {
            motion_id
            for motion_id, intent in intents_by_id.items()
            if str(intent["kinematic_regime"])
            in rule["allowed_kinematic_regimes"]
            and str(intent["interaction_class"])
            in rule["allowed_interaction_classes"]
        }
        if set(overrides) != compatible_motion_ids or not set(
            overrides
        ).issubset(declared_motion_ids):
            raise ValueError(
                "two-object camera overrides differ from compatible motions"
            )
    axis_fields = (
        "motion_id",
        "shape_pair_id",
        "scale_pair_id",
        "scene_class",
    )
    compatible_by_cell: dict[str, tuple[str, ...]] = {}
    compatible_by_axis: dict[str, dict[str, set[str]]] = {
        field: {} for field in axis_fields
    }
    for cell in cells:
        intent = intents_by_id[str(cell["motion_id"])]
        compatible = tuple(
            family_id
            for family_id in family_ids
            if any(
                str(rule["scene_class"]) == str(cell["scene_class"])
                and str(intent["kinematic_regime"])
                in rule["allowed_kinematic_regimes"]
                and str(intent["interaction_class"])
                in rule["allowed_interaction_classes"]
                and family_id
                in allowed_camera_view_families(
                    rule, str(cell["motion_id"])
                )
                for rule in scene_rules["physical_rules"]
            )
        )
        if not compatible:
            raise ValueError("two-object cell has no compatible camera family")
        compatible_by_cell[str(cell["cell_id"])] = compatible
        for field in axis_fields:
            compatible_by_axis[field].setdefault(
                str(cell[field]), set()
            ).update(compatible)
    global_use: Counter[str] = Counter()
    conditional_use: dict[str, dict[str, Counter[str]]] = {
        field: {} for field in axis_fields
    }
    assigned = []
    for original in cells:
        cell = copy.deepcopy(original)
        compatible_family_ids = compatible_by_cell[str(cell["cell_id"])]
        for field in axis_fields:
            conditional_use[field].setdefault(str(cell[field]), Counter())

        def score(family_id: str) -> tuple[int, int, int, int, int, str]:
            conditional_ranges = []
            current_conditional_use = []
            uncovered_penalty = 0
            for field in axis_fields:
                counts = conditional_use[field][str(cell[field])]
                axis_family_ids = compatible_by_axis[field][str(cell[field])]
                has_uncovered_family = any(
                    counts[candidate] == 0 for candidate in axis_family_ids
                )
                uncovered_penalty += int(
                    has_uncovered_family and counts[family_id] > 0
                )
                hypothetical = [
                    counts[candidate] + int(candidate == family_id)
                    for candidate in axis_family_ids
                ]
                conditional_ranges.append(max(hypothetical) - min(hypothetical))
                current_conditional_use.append(counts[family_id])
            return (
                uncovered_penalty,
                global_use[family_id],
                max(conditional_ranges),
                sum(conditional_ranges),
                sum(current_conditional_use),
                _rank(seed, "camera-view-family", cell["cell_id"], family_id),
            )

        family_id = min(compatible_family_ids, key=score)
        global_use[family_id] += 1
        for field in axis_fields:
            conditional_use[field][str(cell[field])][family_id] += 1
        cell["camera_view_family_id"] = family_id
        cell["cell_id"] = "__".join([cell["cell_id"], family_id])
        assigned.append(cell)
    return assigned


def coverage_cells(
    matrix: dict[str, Any],
    limit: int | None = None,
    *,
    scene_rules: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Return a balanced prefix of the complete declared Cartesian matrix."""

    resolved_scene_rules = resolved_two_object_scene_rules(scene_rules)
    intents = _validated_intents(matrix, resolved_scene_rules)
    coverage = matrix["coverage_plan"]
    declared_scene_classes = allowed_scene_classes(resolved_scene_rules)
    replicates = int(coverage["replicates_per_cell"])
    cells = []
    for intent in intents:
        motion_id = str(intent["id"])
        interaction_class = str(intent["interaction_class"])
        kinematic_regime = str(intent["kinematic_regime"])
        compatible_scene_classes = tuple(
            scene_class
            for scene_class in declared_scene_classes
            if any(
                str(rule["scene_class"]) == scene_class
                and kinematic_regime in set(rule["allowed_kinematic_regimes"])
                and interaction_class
                in set(rule["allowed_interaction_classes"])
                and bool(allowed_camera_view_families(rule, motion_id))
                for rule in resolved_scene_rules["physical_rules"]
            )
        )
        if not compatible_scene_classes:
            raise ValueError(
                f"two-object motion has no compatible scene class: {motion_id}"
            )
        allowed_pairs = set(compatible_shape_pair_ids(matrix, motion_id))
        for shape_pair, scale_pair, scene_class, replicate_index in product(
            coverage["role_ordered_shape_pairs"],
            coverage["role_ordered_scale_pairs"],
            compatible_scene_classes,
            range(replicates),
        ):
            shape_pair_id = str(shape_pair["id"])
            if (
                shape_pair_id not in allowed_pairs
                or str(scale_pair["id"])
                not in compatible_scale_pair_ids(
                    matrix, motion_id, str(scene_class)
                )
            ):
                continue
            cell_id = "__".join(
                [
                    motion_id,
                    shape_pair_id,
                    str(scale_pair["id"]),
                    str(scene_class),
                    f"r{replicate_index:02d}",
                ]
            )
            cells.append(
                {
                    "cell_id": cell_id,
                    "motion_id": motion_id,
                    "interaction_class": interaction_class,
                    "shape_pair_id": shape_pair_id,
                    "object_a_shape": str(shape_pair["object_a"]),
                    "object_b_shape": str(shape_pair["object_b"]),
                    "scale_pair_id": str(scale_pair["id"]),
                    "object_a_scale_bin": str(scale_pair["object_a"]),
                    "object_b_scale_bin": str(scale_pair["object_b"]),
                    "scene_class": str(scene_class),
                    "replicate_index": replicate_index,
                }
            )
    full_count = len(cells)
    interacting_count = sum(
        cell["interaction_class"] == "interacting" for cell in cells
    )
    if (
        interacting_count / full_count
        < float(coverage["minimum_interacting_fraction"])
    ):
        raise ValueError("two-object coverage does not meet its interaction mix")
    if limit is not None and (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= full_count
    ):
        raise ValueError("two-object coverage limit is outside the full matrix")
    ordered = _assign_camera_view_families(
        _balanced_cell_order(cells, int(coverage["seed"])),
        coverage["camera_view_families"],
        resolved_scene_rules,
        {str(intent["id"]): intent for intent in intents},
        int(coverage["seed"]),
    )
    return ordered if limit is None else ordered[:limit], full_count


def _resolved_within(root: Path, value: Path) -> Path:
    resolved = (value if value.is_absolute() else root / value).resolve()
    resolved.relative_to(root)
    return resolved


def _source_dynamics_profile(source: dict[str, Any]) -> tuple[float, float]:
    """Validate and retain the source facts used by pair eligibility."""

    metadata = source.get("metadata")
    if not isinstance(metadata, dict) or not isinstance(
        metadata.get("simulation"), dict
    ):
        return math.nan, math.nan
    objects = metadata["simulation"].get("objects")
    if not isinstance(objects, list) or len(objects) != 1:
        raise ValueError("two-object source must contain exactly one object")
    mass = float(objects[0].get("material", {}).get("mass_kg", 0.0))
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("two-object source mass must be finite and positive")
    size = objects[0].get("geometry", {}).get("size_m")
    if (
        not isinstance(size, list)
        or len(size) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in size
        )
    ):
        raise ValueError("two-object source size must be three positive values")
    dimensions = list(map(float, size))
    return mass, max(dimensions) / min(dimensions)


def _dynamics_profiles_eligible(
    profiles: Sequence[tuple[float, float]],
    matrix: dict[str, Any],
    cell: dict[str, Any],
) -> bool:
    """Apply the pair rule to already validated source facts."""

    has_profiles = [not math.isnan(profile[0]) for profile in profiles]
    if not any(has_profiles):
        return True
    if not all(has_profiles):
        raise ValueError("two-object source metadata is incomplete")
    interaction_class = str(cell.get("interaction_class", ""))
    if interaction_class not in {"interacting", "independent"}:
        raise ValueError("two-object coverage cell has no interaction class")
    if interaction_class == "independent":
        return True
    masses = [profile[0] for profile in profiles]
    aspect_ratios = [profile[1] for profile in profiles]
    maximum_ratio = float(
        matrix["candidate_pool"]["pair_eligibility"][
            "maximum_interacting_mass_ratio"
        ]
    )
    maximum_aspect_ratio = float(
        matrix["candidate_pool"]["pair_eligibility"][
            "maximum_interacting_geometry_aspect_ratio"
        ]
    )
    return (
        max(masses) / min(masses) <= maximum_ratio + 1.0e-12
        and max(aspect_ratios) <= maximum_aspect_ratio + 1.0e-12
    )


def _pair_source_scene(
    host: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any] | None:
    """Build the one metadata-shaped pair used by every cheap source check."""

    metadata = [
        host.get("metadata"),
        left.get("metadata"),
        right.get("metadata"),
    ]
    has_simulation = [
        isinstance(record, dict) and isinstance(record.get("simulation"), dict)
        for record in metadata
    ]
    if not any(has_simulation):
        return None
    if not all(has_simulation):
        raise ValueError("two-object source metadata is incomplete")
    host_simulation = metadata[0]["simulation"]
    source_objects = [
        source_metadata["simulation"]["objects"][0]
        for source_metadata in metadata[1:]
    ]
    return {
        "scene_id": "two_object_layout_check",
        "semantic_sampling": {"five_dimensions": {}},
        "simulation": {
            "world": copy.deepcopy(host_simulation["world"]),
            "support": copy.deepcopy(host_simulation["support"]),
            "objects": [
                {
                    "object_id": object_id,
                    "geometry": copy.deepcopy(source_object["geometry"]),
                    "material": copy.deepcopy(source_object["material"]),
                }
                for object_id, source_object in zip(
                    ("object_a", "object_b"), source_objects, strict=True
                )
            ],
        },
    }


def _pair_layout_fits_host(
    host: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    matrix: dict[str, Any],
    cell: dict[str, Any],
) -> bool:
    pair_scene = _pair_source_scene(host, left, right)
    if pair_scene is None:
        return True
    intent = next(
        value
        for value in matrix["motion_intents"]
        if str(value["id"]) == str(cell["motion_id"])
    )
    try:
        apply_two_object_motion(
            pair_scene,
            matrix["shape_families"],
            matrix["shared_physics"],
            matrix["pair_observation"],
            camera_view_family(
                matrix, str(cell["camera_view_family_id"])
            ),
            intent,
        )
    except ValueError as error:
        if str(error) == "host support is too small for the two-object layout":
            return False
        raise
    return True


def _camera_plane_readability(
    rule: dict[str, Any], cell: dict[str, Any]
) -> tuple[tuple[int, int], float] | None:
    readability = rule.get(
        "object_camera_plane_readability_by_view_family", {}
    ).get(str(cell["camera_view_family_id"]))
    if readability is None:
        return None
    axes = tuple(map(int, readability["geometry_size_axes"]))
    return (axes[0], axes[1]), float(readability["minimum_extent_m"])


def _source_camera_plane_extent(
    source: dict[str, Any], rule: dict[str, Any], cell: dict[str, Any]
) -> float:
    """Return the declared local camera-plane extent for one source."""

    readability = _camera_plane_readability(rule, cell)
    if readability is None:
        return math.inf
    axes, _ = readability
    simulation = source.get("metadata", {}).get("simulation")
    if not isinstance(simulation, dict):
        return math.inf
    objects = simulation["objects"]
    if len(objects) != 1:
        raise ValueError("two-object source must contain exactly one object")
    size = list(map(float, objects[0]["geometry"]["size_m"]))
    return max(size[axis] for axis in axes)


def _source_meets_rule_camera_extent(
    source: dict[str, Any], rule: dict[str, Any], cell: dict[str, Any]
) -> bool:
    """Apply cheap ramp readability gates before copying a full host."""

    readability = _camera_plane_readability(rule, cell)
    if readability is None:
        return True
    _, minimum_extent = readability
    return (
        _source_camera_plane_extent(source, rule, cell) + 1.0e-12
        >= minimum_extent
    )


def _source_geometry_signature(source: dict[str, Any]) -> tuple[Any, ...]:
    """Return the exact geometry facts used by layout and camera checks."""

    simulation = source.get("metadata", {}).get("simulation")
    if not isinstance(simulation, dict):
        source_id = str(source.get("source", {}).get("scene_id", ""))
        if not source_id:
            raise ValueError("two-object source metadata is incomplete")
        return "unbound_source", source_id
    objects = simulation.get("objects")
    if not isinstance(objects, list) or len(objects) != 1:
        raise ValueError("two-object source must contain exactly one object")
    geometry = objects[0].get("geometry", {})
    size = geometry.get("size_m")
    if not isinstance(size, list) or len(size) != 3:
        raise ValueError("two-object source size must contain three values")
    return str(geometry.get("type", "")), *map(float, size)


def _pair_sources_meet_rule_camera_geometry(
    host: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    matrix: dict[str, Any],
    cell: dict[str, Any],
    rule: dict[str, Any],
) -> bool:
    """Check supported-contact geometry and any scene-specific size floor."""

    readability = _camera_plane_readability(rule, cell)
    minimum_extent = 0.0
    if (
        readability is not None
        and str(cell["camera_view_family_id"])
        in rule.get("pair_camera_geometry_view_families", ())
    ):
        _, minimum_extent = readability
    lightweight_scene = _pair_source_scene(host, left, right)
    if lightweight_scene is None:
        return True
    motion = next(
        value
        for value in matrix["motion_intents"]
        if str(value["id"]) == str(cell["motion_id"])
    )
    if (
        str(motion["interaction_class"]) == "independent"
        or str(motion["layout"]) != "planned_supported_contact"
    ):
        return True
    support = lightweight_scene["simulation"]["support"]
    for obj in lightweight_scene["simulation"]["objects"]:
        geometry = obj["geometry"]
        pose = pose_on_support(
            support,
            str(geometry["type"]),
            list(map(float, geometry["size_m"])),
            0.0,
            0.0,
            0.0,
            0.0,
            pose_profile="support_normal",
        )
        obj["initial_state"] = {
            "orientation_quaternion_wxyz": pose[
                "orientation_quaternion_wxyz"
            ]
        }
    family = camera_view_family(
        matrix, str(cell["camera_view_family_id"])
    )
    observation = matrix["pair_observation"]
    lightweight_scene["simulation"]["interaction"] = {
        "approach_axis_xyz": interaction_approach_axis(
            motion, support
        ).tolist(),
        "camera_relative_azimuth_degrees": float(
            family["relative_azimuth_degrees"]
        ),
        "maximum_camera_view_azimuth_deviation_degrees": float(
            observation["maximum_camera_view_azimuth_deviation_degrees"]
        ),
        "minimum_camera_elevation_degrees": float(
            family["minimum_elevation_degrees"]
        ),
        "preferred_camera_elevation_degrees": float(
            family["preferred_elevation_degrees"]
        ),
        "maximum_camera_elevation_degrees": float(
            family["maximum_elevation_degrees"]
        ),
        "minimum_pair_keyframe_projected_center_separation_to_radius_sum_ratio": float(
            observation[
                "minimum_pair_keyframe_projected_center_separation_to_radius_sum_ratio"
            ]
        ),
    }
    return pair_camera_geometry_eligible(
        lightweight_scene,
        float(minimum_extent),
    )


def select_coverage_sources(
    cells: Sequence[dict[str, Any]],
    objects: Sequence[dict[str, Any]],
    hosts: Sequence[dict[str, Any]],
    matrix: dict[str, Any],
    scene_rules: dict[str, Any] | None = None,
    *,
    require_all_profiles: bool = True,
) -> list[dict[str, Any]]:
    """Assign compatible sources while balancing profiles and source reuse."""

    resolved_scene_rules = resolved_two_object_scene_rules(scene_rules)
    plan = matrix["coverage_plan"]
    seed = int(plan["seed"])
    intent_regimes = {
        str(intent["id"]): str(intent["kinematic_regime"])
        for intent in matrix["motion_intents"]
    }
    intent_interaction_classes = {
        str(intent["id"]): str(intent["interaction_class"])
        for intent in matrix["motion_intents"]
    }
    intent_layouts = {
        str(intent["id"]): str(intent["layout"])
        for intent in matrix["motion_intents"]
    }
    scene_rule_regimes = {
        str(rule["id"]): set(map(str, rule["allowed_kinematic_regimes"]))
        for rule in resolved_scene_rules["physical_rules"]
    }
    scene_rule_interaction_classes = {
        str(rule["id"]): set(map(str, rule["allowed_interaction_classes"]))
        for rule in resolved_scene_rules["physical_rules"]
    }
    scene_rules_by_id = {
        str(rule["id"]): rule for rule in resolved_scene_rules["physical_rules"]
    }
    scene_rule_classes = {
        str(rule["id"]): str(rule["scene_class"])
        for rule in resolved_scene_rules["physical_rules"]
    }
    unknown_host_rules = sorted(
        {
            str(record.get("scene_rule_id", ""))
            for record in hosts
        }.difference(scene_rule_regimes)
    )
    if unknown_host_rules:
        raise ValueError(
            f"two-object hosts name unknown scene rules: {unknown_host_rules}"
        )
    if any(
        str(record.get("scene_class", ""))
        != scene_rule_classes[str(record["scene_rule_id"])]
        for record in hosts
    ):
        raise ValueError("two-object host scene class contradicts its scene rule")
    maximum_reuse = int(
        plan["selection_policy"]["maximum_object_source_reuse"]
    )
    maximum_host_reuse = int(
        plan["selection_policy"]["maximum_host_source_reuse"]
    )
    objects_by_family_shape_scale: dict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = {}
    object_dynamics_profiles: dict[str, tuple[float, float]] = {}
    object_geometry_signatures: dict[str, tuple[Any, ...]] = {}
    for record in objects:
        source_id = str(record["source"]["scene_id"])
        if source_id in object_dynamics_profiles:
            raise ValueError("two-object source pool contains duplicate ids")
        object_dynamics_profiles[source_id] = _source_dynamics_profile(record)
        object_geometry_signatures[source_id] = _source_geometry_signature(record)
        key = (
            str(record["source_family"]),
            str(record["shape_family_id"]),
            str(record["scale_bin"]),
        )
        objects_by_family_shape_scale.setdefault(key, []).append(record)
    hosts_by_scene: dict[str, list[dict[str, Any]]] = {}
    for record in hosts:
        hosts_by_scene.setdefault(str(record["scene_class"]), []).append(record)

    object_use: Counter[str] = Counter()
    object_profile_use: Counter[str] = Counter()
    source_family_pair_use: Counter[str] = Counter()
    camera_source_family_pair_use: Counter[tuple[str, str]] = Counter()
    source_family_role_use: Counter[tuple[str, str]] = Counter()
    host_profile_use: Counter[str] = Counter()
    host_type_use: Counter[str] = Counter()
    motion_scene_rule_use: Counter[tuple[str, str]] = Counter()
    environment_use: Counter[str] = Counter()
    scene_environment_use: Counter[tuple[str, str]] = Counter()
    camera_scene_environment_use: Counter[tuple[str, str, str]] = Counter()
    source_pair_scene_environment_use: Counter[tuple[str, str, str]] = Counter()
    used_pairs: set[tuple[str, str]] = set()
    host_use: Counter[str] = Counter()
    source_rule_extent_cache: dict[tuple[str, str, str], bool] = {}
    pair_camera_geometry_cache: dict[tuple[Any, ...], bool] = {}
    pair_layout_cache: dict[tuple[Any, ...], bool] = {}

    def source_meets_rule_camera_extent(
        source: dict[str, Any], rule: dict[str, Any], cell: dict[str, Any]
    ) -> bool:
        key = (
            str(source["source"]["scene_id"]),
            str(rule["id"]),
            str(cell["camera_view_family_id"]),
        )
        if key not in source_rule_extent_cache:
            source_rule_extent_cache[key] = _source_meets_rule_camera_extent(
                source, rule, cell
            )
        return source_rule_extent_cache[key]

    def pair_meets_rule_camera_geometry(
        host: dict[str, Any],
        left: dict[str, Any],
        right: dict[str, Any],
        cell: dict[str, Any],
        rule: dict[str, Any],
    ) -> bool:
        simulation = host.get("metadata", {}).get("simulation")
        if not isinstance(simulation, dict):
            return _pair_sources_meet_rule_camera_geometry(
                host, left, right, matrix, cell, rule
            )
        support = simulation["support"]
        frame = support["surface_frame"]
        key = (
            object_geometry_signatures[str(left["source"]["scene_id"])],
            object_geometry_signatures[str(right["source"]["scene_id"])],
            str(cell["motion_id"]),
            str(cell["camera_view_family_id"]),
            str(rule["id"]),
            float(frame["slope_angle_degrees"]),
            *map(float, frame["tangent_cross"]),
            *map(float, frame["tangent_uphill"]),
            *map(float, frame["normal"]),
        )
        if key not in pair_camera_geometry_cache:
            pair_camera_geometry_cache[key] = (
                _pair_sources_meet_rule_camera_geometry(
                    host,
                    left,
                    right,
                    matrix,
                    cell,
                    rule,
                )
            )
        return pair_camera_geometry_cache[key]

    def pair_layout_fits_host(
        host: dict[str, Any],
        left: dict[str, Any],
        right: dict[str, Any],
        cell: dict[str, Any],
    ) -> bool:
        simulation = host.get("metadata", {}).get("simulation")
        if not isinstance(simulation, dict):
            return _pair_layout_fits_host(
                host,
                left,
                right,
                matrix,
                cell,
            )
        support = simulation["support"]
        bounds = support["safe_surface_bounds"]
        frame = support["surface_frame"]
        physical_host_signature = (
            str(support["support_shape"]),
            float(bounds["x"][1]) - float(bounds["x"][0]),
            float(bounds["y"][1]) - float(bounds["y"][0]),
            *map(float, frame["tangent_cross"]),
            *map(float, frame["tangent_uphill"]),
            *map(float, frame["normal"]),
            *map(float, simulation["world"]["gravity_m_s2"]),
        )
        key = (
            object_geometry_signatures[str(left["source"]["scene_id"])],
            object_geometry_signatures[str(right["source"]["scene_id"])],
            str(cell["motion_id"]),
            str(cell["camera_view_family_id"]),
            str(host["scene_rule_id"]),
            *physical_host_signature,
        )
        if key not in pair_layout_cache:
            pair_layout_cache[key] = _pair_layout_fits_host(
                host,
                left,
                right,
                matrix,
                cell,
            )
        return pair_layout_cache[key]

    original_cell_order = {
        str(cell["cell_id"]): index for index, cell in enumerate(cells)
    }

    def requires_pair_camera_geometry(cell: dict[str, Any]) -> bool:
        motion_id = str(cell["motion_id"])
        return (
            intent_interaction_classes[motion_id] == "interacting"
            and intent_layouts[motion_id] == "planned_supported_contact"
        )

    assignment_cells = sorted(
        cells,
        key=lambda cell: (
            not requires_pair_camera_geometry(cell),
            original_cell_order[str(cell["cell_id"])],
        ),
    )
    selected = []
    selected_cell_order: dict[str, int] = {}
    for cell in assignment_cells:
        camera_view_family_id = str(cell["camera_view_family_id"])
        scene_class = str(cell["scene_class"])
        strict_pair_geometry = requires_pair_camera_geometry(cell)

        def object_key(record: dict[str, Any], role: str) -> tuple[int, int, str]:
            source_id = str(record["source"]["scene_id"])
            profile_id = str(record["visual_profile_id"])
            return (
                object_use[source_id],
                object_profile_use[profile_id],
                _rank(seed, cell["cell_id"], role, source_id),
            )

        cell_camera_ranking_rule = next(
            (
                scene_rules_by_id[str(record["scene_rule_id"])]
                for record in hosts_by_scene.get(scene_class, [])
                if intent_regimes[str(cell["motion_id"])]
                in scene_rule_regimes[str(record["scene_rule_id"])]
                and intent_interaction_classes[str(cell["motion_id"])]
                in scene_rule_interaction_classes[
                    str(record["scene_rule_id"])
                ]
                and camera_view_family_id
                in allowed_camera_view_families(
                    scene_rules_by_id[str(record["scene_rule_id"])],
                    str(cell["motion_id"]),
                )
                and camera_view_family_id
                in scene_rules_by_id[str(record["scene_rule_id"])].get(
                    "object_camera_plane_readability_by_view_family", {}
                )
            ),
            None,
        )

        def source_family_camera_rank(
            source_family: str, role: str
        ) -> float:
            if strict_pair_geometry or cell_camera_ranking_rule is None:
                return 1.0
            pool = objects_by_family_shape_scale.get(
                (
                    source_family,
                    str(cell[f"{role}_shape"]),
                    str(cell[f"{role}_scale_bin"]),
                ),
                [],
            )
            return min(
                (
                    -_source_camera_plane_extent(
                        record, cell_camera_ranking_rule, cell
                    )
                    for record in pool
                    if source_meets_rule_camera_extent(
                        record, cell_camera_ranking_rule, cell
                    )
                ),
                default=math.inf,
            )

        declared_source_pairs = sorted(
            plan["role_ordered_source_family_pairs"],
            key=lambda record: (
                max(
                    source_family_camera_rank(
                        str(record["object_a"]), "object_a"
                    ),
                    source_family_camera_rank(
                        str(record["object_b"]), "object_b"
                    ),
                ),
                camera_source_family_pair_use[
                    (camera_view_family_id, str(record["id"]))
                ],
                source_family_pair_use[str(record["id"])],
                source_family_role_use[("object_a", str(record["object_a"]))]
                + source_family_role_use[("object_b", str(record["object_b"]))],
                _rank(seed, cell["cell_id"], "source-family", record["id"]),
            ),
        )
        pair = None
        selected_source_pair = None
        host = None
        for source_pair in declared_source_pairs:
            source_pair_id = str(source_pair["id"])
            left_family = str(source_pair["object_a"])
            right_family = str(source_pair["object_b"])
            left_pool = objects_by_family_shape_scale.get(
                (
                    left_family,
                    str(cell["object_a_shape"]),
                    str(cell["object_a_scale_bin"]),
                ),
                [],
            )
            right_pool = objects_by_family_shape_scale.get(
                (
                    right_family,
                    str(cell["object_b_shape"]),
                    str(cell["object_b_scale_bin"]),
                ),
                [],
            )
            candidate_hosts = [
                record
                for record in hosts_by_scene.get(scene_class, [])
                if intent_regimes[str(cell["motion_id"])]
                in scene_rule_regimes[str(record["scene_rule_id"])]
                and intent_interaction_classes[str(cell["motion_id"])]
                in scene_rule_interaction_classes[
                    str(record["scene_rule_id"])
                ]
                and camera_view_family_id
                in allowed_camera_view_families(
                    scene_rules_by_id[str(record["scene_rule_id"])],
                    str(cell["motion_id"]),
                )
                and host_use[str(record["source"]["scene_id"])]
                < maximum_host_reuse
            ]
            candidate_hosts.sort(
                key=lambda record: (
                    motion_scene_rule_use[
                        (str(cell["motion_id"]), str(record["scene_rule_id"]))
                    ]
                    > 0,
                    host_use[str(record["source"]["scene_id"])],
                    scene_environment_use[
                        (scene_class, str(record["environment_category"]))
                    ],
                    camera_scene_environment_use[
                        (
                            camera_view_family_id,
                            scene_class,
                            str(record["environment_category"]),
                        )
                    ],
                    source_pair_scene_environment_use[
                        (
                            source_pair_id,
                            scene_class,
                            str(record["environment_category"]),
                        )
                    ],
                    motion_scene_rule_use[
                        (str(cell["motion_id"]), str(record["scene_rule_id"]))
                    ],
                    environment_use[str(record["environment_category"])],
                    host_profile_use[str(record["visual_profile_id"])],
                    host_type_use[str(record["visual_type"])],
                    _rank(
                        seed,
                        cell["cell_id"],
                        "host",
                        record["source"]["scene_id"],
                    ),
                )
            )
            candidate_rules = [
                scene_rules_by_id[str(record["scene_rule_id"])]
                for record in candidate_hosts
            ]
            left_pool = [
                record
                for record in left_pool
                if any(
                    source_meets_rule_camera_extent(record, rule, cell)
                    for rule in candidate_rules
                )
            ]
            right_pool = [
                record
                for record in right_pool
                if any(
                    source_meets_rule_camera_extent(record, rule, cell)
                    for rule in candidate_rules
                )
            ]

            def ranked_object_key(
                record: dict[str, Any], role: str
            ) -> tuple[Any, ...]:
                base_key = object_key(record, role)
                if cell_camera_ranking_rule is None:
                    return base_key
                extent_rank = -_source_camera_plane_extent(
                    record, cell_camera_ranking_rule, cell
                )
                if strict_pair_geometry:
                    return base_key[:2] + (extent_rank, base_key[2])
                return (
                    extent_rank,
                    *base_key,
                )

            ordered_left_pool = sorted(
                (
                    record
                    for record in left_pool
                    if object_use[str(record["source"]["scene_id"])]
                    < maximum_reuse
                ),
                key=lambda value: ranked_object_key(value, "a"),
            )
            ordered_right_pool = sorted(
                (
                    record
                    for record in right_pool
                    if object_use[str(record["source"]["scene_id"])]
                    < maximum_reuse
                ),
                key=lambda value: ranked_object_key(value, "b"),
            )

            def eligible_object_pairs() -> Iterator[
                tuple[dict[str, Any], dict[str, Any], tuple[str, str]]
            ]:
                for left in ordered_left_pool:
                    left_id = str(left["source"]["scene_id"])
                    for right in ordered_right_pool:
                        right_id = str(right["source"]["scene_id"])
                        unordered_pair = tuple(sorted((left_id, right_id)))
                        if (
                            left_id == right_id
                            or unordered_pair in used_pairs
                            or not _dynamics_profiles_eligible(
                                (
                                    object_dynamics_profiles[left_id],
                                    object_dynamics_profiles[right_id],
                                ),
                                matrix,
                                cell,
                            )
                        ):
                            continue
                        yield left, right, unordered_pair

            candidate_object_pair_cache = []
            candidate_object_pair_iterator = eligible_object_pairs()
            for candidate_host in candidate_hosts:
                candidate_rule = scene_rules_by_id[
                    str(candidate_host["scene_rule_id"])
                ]
                candidate_host_id = str(candidate_host["source"]["scene_id"])
                pair_index = 0
                while True:
                    if pair_index == len(candidate_object_pair_cache):
                        try:
                            candidate_object_pair_cache.append(
                                next(candidate_object_pair_iterator)
                            )
                        except StopIteration:
                            break
                    left, right, unordered_pair = (
                        candidate_object_pair_cache[pair_index]
                    )
                    pair_index += 1
                    left_id = str(left["source"]["scene_id"])
                    right_id = str(right["source"]["scene_id"])
                    if (
                        left_id == candidate_host_id
                        or right_id == candidate_host_id
                        or not source_meets_rule_camera_extent(
                            left, candidate_rule, cell
                        )
                        or not source_meets_rule_camera_extent(
                            right, candidate_rule, cell
                        )
                    ):
                        continue
                    if not pair_layout_fits_host(
                        candidate_host,
                        left,
                        right,
                        cell,
                    ):
                        continue
                    if not pair_meets_rule_camera_geometry(
                        candidate_host,
                        left,
                        right,
                        cell,
                        candidate_rule,
                    ):
                        continue
                    pair = (left, right, unordered_pair)
                    selected_source_pair = source_pair
                    host = candidate_host
                    break
                if pair is not None:
                    break
            if pair is not None:
                break
        if pair is None:
            raise ValueError(
                f"cannot satisfy two-object source cell: {cell['cell_id']}"
            )
        if selected_source_pair is None or host is None:
            raise AssertionError("selected source pair or host is missing")
        left, right, unordered_pair = pair
        selected_cell = copy.deepcopy(cell)
        selected_cell.update(
            {
                "source_family_pair_id": str(selected_source_pair["id"]),
                "object_a_source_family": str(selected_source_pair["object_a"]),
                "object_b_source_family": str(selected_source_pair["object_b"]),
            }
        )
        source_pair_id = str(selected_source_pair["id"])
        environment_category = str(host["environment_category"])
        scene_rule_id = str(host["scene_rule_id"])
        selected_cell["scene_rule_id"] = scene_rule_id
        selected_cell["visual_environment_category"] = environment_category
        selected_cell["cell_id"] = "__".join(
            [str(selected_cell["cell_id"]), scene_rule_id, environment_category]
        )
        selected_cell_order[str(selected_cell["cell_id"])] = original_cell_order[
            str(cell["cell_id"])
        ]
        used_pairs.add(unordered_pair)
        source_family_pair_use[str(selected_source_pair["id"])] += 1
        camera_source_family_pair_use[
            (camera_view_family_id, str(selected_source_pair["id"]))
        ] += 1
        source_family_role_use[
            ("object_a", str(selected_source_pair["object_a"]))
        ] += 1
        source_family_role_use[
            ("object_b", str(selected_source_pair["object_b"]))
        ] += 1
        host_use[str(host["source"]["scene_id"])] += 1
        for record in (left, right):
            object_use[str(record["source"]["scene_id"])] += 1
            object_profile_use[str(record["visual_profile_id"])] += 1
        host_profile_use[str(host["visual_profile_id"])] += 1
        host_type_use[str(host["visual_type"])] += 1
        motion_scene_rule_use[(str(cell["motion_id"]), scene_rule_id)] += 1
        environment_use[environment_category] += 1
        scene_environment_use[(scene_class, environment_category)] += 1
        camera_scene_environment_use[
            (camera_view_family_id, scene_class, environment_category)
        ] += 1
        source_pair_scene_environment_use[
            (source_pair_id, scene_class, environment_category)
        ] += 1
        selected.append(
            {
                "cell": selected_cell,
                "host": host,
                "objects": [left, right],
            }
        )
    selected.sort(
        key=lambda record: selected_cell_order[str(record["cell"]["cell_id"])]
    )

    eligible_object_profiles = {str(value["visual_profile_id"]) for value in objects}
    eligible_host_profiles = {str(value["visual_profile_id"]) for value in hosts}
    object_profile_policy = str(
        plan["selection_policy"]["object_visual_profile_coverage"]
    )
    if (
        require_all_profiles
        and object_profile_policy == "all_eligible"
        and set(object_profile_use) != eligible_object_profiles
    ):
        raise ValueError("coverage selection misses eligible object visual profiles")
    if require_all_profiles and set(host_profile_use) != eligible_host_profiles:
        raise ValueError("coverage selection misses eligible host visual profiles")
    return selected


def _validate_complete_scene_coverage(
    matrix: dict[str, Any],
    scene_rules: dict[str, Any],
    selections: Sequence[dict[str, Any]],
) -> None:
    """Require a full selection to materialize every declared scene pairing."""

    selected_cells = [selection["cell"] for selection in selections]

    declared_rules = {str(rule["id"]) for rule in scene_rules["physical_rules"]}
    selected_rules = {str(cell["scene_rule_id"]) for cell in selected_cells}
    if selected_rules != declared_rules:
        raise ValueError(
            "complete two-object coverage misses physical scene rules: "
            f"{sorted(declared_rules - selected_rules)}"
        )
    declared_pairs = {
        (str(rule_id), str(category["id"]))
        for category in scene_rules["visual_environment_coverage"]["categories"]
        for rule_id in category["allowed_scene_rules"]
    }
    selected_pairs = {
        (str(cell["scene_rule_id"]), str(cell["visual_environment_category"]))
        for cell in selected_cells
    }
    if selected_pairs != declared_pairs:
        raise ValueError(
            "complete two-object coverage misses scene/environment pairs: "
            f"{sorted(declared_pairs - selected_pairs)}"
        )
    motion_regimes = {
        str(intent["id"]): str(intent["kinematic_regime"])
        for intent in matrix["motion_intents"]
    }
    motion_interaction_classes = {
        str(intent["id"]): str(intent["interaction_class"])
        for intent in matrix["motion_intents"]
    }
    declared_motion_rules = {
        (motion_id, str(rule["id"]))
        for motion_id, regime in motion_regimes.items()
        for rule in scene_rules["physical_rules"]
        if regime in rule["allowed_kinematic_regimes"]
        and motion_interaction_classes[motion_id]
        in rule["allowed_interaction_classes"]
        and bool(allowed_camera_view_families(rule, motion_id))
    }
    selected_motion_rules = {
        (str(cell["motion_id"]), str(cell["scene_rule_id"]))
        for cell in selected_cells
    }
    if selected_motion_rules != declared_motion_rules:
        raise ValueError(
            "complete two-object coverage misses motion/scene-rule pairs: "
            f"{sorted(declared_motion_rules - selected_motion_rules)}"
        )
    declared_camera_rules = {
        (str(camera_family), str(rule["id"]))
        for rule in scene_rules["physical_rules"]
        for camera_family in rule["allowed_camera_view_families"]
    }
    selected_camera_rules = {
        (str(cell["camera_view_family_id"]), str(cell["scene_rule_id"]))
        for cell in selected_cells
    }
    if selected_camera_rules != declared_camera_rules:
        raise ValueError(
            "complete two-object coverage misses camera/scene-rule pairs: "
            f"{sorted(declared_camera_rules - selected_camera_rules)}"
        )


def build_coverage_scenes(
    selections: Sequence[dict[str, Any]],
    matrix: dict[str, Any],
    scene_rules: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compile selected sources and attach the minimal per-scene coverage fact."""

    resolved_scene_rules = resolved_two_object_scene_rules(scene_rules)
    scenes = []
    role_ids = [str(role["object_id"]) for role in matrix["objects"]["roles"]]
    for sample_index, selection in enumerate(selections, start=1):
        cell = selection["cell"]
        host = copy.deepcopy(selection["host"]["metadata"])
        stub = "_".join(
            [
                f"physweep2scene_{sample_index:06d}",
                str(cell["shape_pair_id"]),
                str(cell["scale_pair_id"]),
                str(cell["scene_rule_id"]),
                str(cell["visual_environment_category"]),
            ]
        )
        host["scene_id"] = stub
        scene = build_two_object_scene(
            host,
            matrix,
            str(cell["motion_id"]),
            [record["metadata"] for record in selection["objects"]],
            sample_index=sample_index,
            camera_view_family_id=str(cell["camera_view_family_id"]),
            scene_rules=resolved_scene_rules,
            scene_rule_id=str(cell["scene_rule_id"]),
        )
        scene["two_object_sampling"] = {
            "schema_version": "physweep_two_object_coverage_cell_v4",
            "cell": copy.deepcopy(cell),
            "sources": {
                "host": copy.deepcopy(selection["host"]["source"]),
                "objects": [
                    {
                        "object_id": object_id,
                        **copy.deepcopy(record["source"]),
                    }
                    for object_id, record in zip(role_ids, selection["objects"])
                ],
            },
        }
        scenes.append(scene)
    return scenes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--released-base-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--scene-rules", type=Path, default=DEFAULT_TWO_OBJECT_SCENE_RULES
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_root = args.source_root.resolve()
    released_path = _resolved_within(root, args.released_base_manifest)
    source_path = declared_within(source_root, args.source_manifest)
    matrix_path = _resolved_within(root, args.matrix)
    scene_rules_path = _resolved_within(root, args.scene_rules)
    output_dir = _resolved_within(root, args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("two-object coverage output directory must be empty")
    matrix = read_json(matrix_path)
    scene_rules = load_two_object_scene_rules(scene_rules_path)
    cells, full_cell_count = coverage_cells(
        matrix, args.limit, scene_rules=scene_rules
    )
    objects, hosts = released_source_pool(
        root=root,
        released_base_manifest_path=released_path,
        source_root=source_root,
        source_manifest_path=source_path,
        matrix=matrix,
        scene_rules=scene_rules,
    )
    selections = select_coverage_sources(
        cells,
        objects,
        hosts,
        matrix,
        scene_rules,
        require_all_profiles=len(cells) == full_cell_count,
    )
    if len(cells) == full_cell_count:
        _validate_complete_scene_coverage(matrix, scene_rules, selections)
    selected_cells = [selection["cell"] for selection in selections]
    scenes = build_coverage_scenes(selections, matrix, scene_rules)
    samples = []
    for scene in scenes:
        metadata_path = output_dir / "scenes" / scene["scene_id"] / "metadata.json"
        write_json(metadata_path, scene)
        samples.append(
            {
                "scene_id": scene["scene_id"],
                "metadata_path": metadata_path.relative_to(root).as_posix(),
                "metadata_sha256": sha256(metadata_path),
                "coverage_cell_id": scene["two_object_sampling"]["cell"][
                    "cell_id"
                ],
            }
        )
    manifest = {
        "schema_version": "physweep_pybullet_base_manifest_v1",
        "dataset_id": DATASET_ID,
        "sample_count": len(samples),
        "matrix": {
            "path": matrix_path.relative_to(root).as_posix(),
            "sha256": sha256(matrix_path),
        },
        "scene_rules": {
            "path": scene_rules_path.relative_to(root).as_posix(),
            "sha256": sha256(scene_rules_path),
        },
        "source_release": {
            "source_project_root": (
                source_root.relative_to(root).as_posix()
                if source_root.is_relative_to(root)
                else str(source_root)
            ),
            "released_base_manifest": {
                "path": released_path.relative_to(root).as_posix(),
                "sha256": sha256(released_path),
            },
            "generation_manifest": {
                "path": source_path.relative_to(source_root).as_posix(),
                "sha256": sha256(source_path),
            },
        },
        "coverage": {
            "schema_version": "physweep_two_object_coverage_selection_v7",
            "full_cell_count": full_cell_count,
            "selected_cell_count": len(cells),
            "complete_cartesian_product": len(cells) == full_cell_count,
            "axis_counts": _axis_counts(selected_cells),
        },
        "samples": samples,
        "status": "sampled_pending_simulation",
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
