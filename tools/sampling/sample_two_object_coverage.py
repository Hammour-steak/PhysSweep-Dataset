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
from typing import Any, Sequence

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json, write_json
from tools.sampling.sample_two_object_base import (
    DATASET_ID,
    DEFAULT_MATRIX,
    _validated_intents,
    build_two_object_scene,
    compatible_shape_pair_ids,
)
from tools.scene_rules.two_object import (
    DEFAULT_TWO_OBJECT_SCENE_RULES,
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
    seed: int,
) -> list[dict[str, Any]]:
    """Balance views globally and within every existing coverage stratum."""

    family_ids = [str(record["id"]) for record in view_families]
    axis_fields = (
        "motion_id",
        "shape_pair_id",
        "scale_pair_id",
        "scene_class",
    )
    global_use: Counter[str] = Counter()
    conditional_use: dict[str, dict[str, Counter[str]]] = {
        field: {} for field in axis_fields
    }
    assigned = []
    for original in cells:
        cell = copy.deepcopy(original)
        for field in axis_fields:
            conditional_use[field].setdefault(str(cell[field]), Counter())

        def score(family_id: str) -> tuple[int, int, int, int, str]:
            conditional_ranges = []
            current_conditional_use = []
            for field in axis_fields:
                counts = conditional_use[field][str(cell[field])]
                hypothetical = [
                    counts[candidate] + int(candidate == family_id)
                    for candidate in family_ids
                ]
                conditional_ranges.append(max(hypothetical) - min(hypothetical))
                current_conditional_use.append(counts[family_id])
            return (
                global_use[family_id],
                max(conditional_ranges),
                sum(conditional_ranges),
                sum(current_conditional_use),
                _rank(seed, "camera-view-family", cell["cell_id"], family_id),
            )

        family_id = min(family_ids, key=score)
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
    scene_classes = allowed_scene_classes(resolved_scene_rules)
    replicates = int(coverage["replicates_per_cell"])
    cells = []
    for intent in intents:
        motion_id = str(intent["id"])
        interaction_class = str(intent["interaction_class"])
        allowed_pairs = set(compatible_shape_pair_ids(matrix, motion_id))
        for shape_pair, scale_pair, scene_class, replicate_index in product(
            coverage["role_ordered_shape_pairs"],
            coverage["role_ordered_scale_pairs"],
            scene_classes,
            range(replicates),
        ):
            shape_pair_id = str(shape_pair["id"])
            if shape_pair_id not in allowed_pairs:
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
        <= float(coverage["minimum_interacting_fraction"])
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
        int(coverage["seed"]),
    )
    return ordered if limit is None else ordered[:limit], full_count


def _resolved_within(root: Path, value: Path) -> Path:
    resolved = (value if value.is_absolute() else root / value).resolve()
    resolved.relative_to(root)
    return resolved


def _pair_dynamics_eligible(
    left: dict[str, Any],
    right: dict[str, Any],
    matrix: dict[str, Any],
    cell: dict[str, Any],
) -> bool:
    """Reject only interacting pairs with a declared extreme mass mismatch."""

    metadata = [left.get("metadata"), right.get("metadata")]
    has_simulation = [
        isinstance(record, dict) and isinstance(record.get("simulation"), dict)
        for record in metadata
    ]
    if not any(has_simulation):
        return True
    if not all(has_simulation):
        raise ValueError("two-object source metadata is incomplete")
    masses = []
    aspect_ratios = []
    for record in metadata:
        objects = record["simulation"].get("objects")
        if not isinstance(objects, list) or len(objects) != 1:
            raise ValueError("two-object source must contain exactly one object")
        mass = float(objects[0].get("material", {}).get("mass_kg", 0.0))
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError("two-object source mass must be finite and positive")
        masses.append(mass)
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
        aspect_ratios.append(max(map(float, size)) / min(map(float, size)))
    interaction_class = str(cell.get("interaction_class", ""))
    if interaction_class not in {"interacting", "independent"}:
        raise ValueError("two-object coverage cell has no interaction class")
    if interaction_class == "independent":
        return True
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


def _pair_layout_fits_host(
    host: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
    matrix: dict[str, Any],
    cell: dict[str, Any],
    scene_rules: dict[str, Any] | None = None,
) -> bool:
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
        return True
    if not all(has_simulation):
        raise ValueError("two-object source metadata is incomplete")
    declared_scene_rule_id = host.get("scene_rule_id")
    try:
        build_two_object_scene(
            metadata[0],
            matrix,
            str(cell["motion_id"]),
            metadata[1:],
            camera_view_family_id=str(cell["camera_view_family_id"]),
            scene_rules=scene_rules,
            scene_rule_id=(
                None
                if declared_scene_rule_id is None
                else str(declared_scene_rule_id)
            ),
        )
    except ValueError as error:
        if str(error) == "host support is too small for the two-object layout":
            return False
        raise
    return True


def select_coverage_sources(
    cells: Sequence[dict[str, Any]],
    objects: Sequence[dict[str, Any]],
    hosts: Sequence[dict[str, Any]],
    matrix: dict[str, Any],
    scene_rules: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign compatible sources while balancing profiles and source reuse."""

    resolved_scene_rules = resolved_two_object_scene_rules(scene_rules)
    plan = matrix["coverage_plan"]
    seed = int(plan["seed"])
    intent_regimes = {
        str(intent["id"]): str(intent["kinematic_regime"])
        for intent in matrix["motion_intents"]
    }
    scene_rule_regimes = {
        str(rule["id"]): set(map(str, rule["allowed_kinematic_regimes"]))
        for rule in resolved_scene_rules["physical_rules"]
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
    for record in objects:
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
    scene_rule_use: Counter[str] = Counter()
    motion_scene_rule_use: Counter[tuple[str, str]] = Counter()
    camera_scene_rule_use: Counter[tuple[str, str]] = Counter()
    scene_rule_environment_use: Counter[tuple[str, str]] = Counter()
    environment_use: Counter[str] = Counter()
    scene_environment_use: Counter[tuple[str, str]] = Counter()
    camera_scene_environment_use: Counter[tuple[str, str, str]] = Counter()
    source_pair_scene_environment_use: Counter[tuple[str, str, str]] = Counter()
    used_pairs: set[tuple[str, str]] = set()
    host_use: Counter[str] = Counter()
    selected = []
    for cell in cells:
        camera_view_family_id = str(cell["camera_view_family_id"])
        scene_class = str(cell["scene_class"])

        def object_key(record: dict[str, Any], role: str) -> tuple[int, int, str]:
            source_id = str(record["source"]["scene_id"])
            profile_id = str(record["visual_profile_id"])
            return (
                object_use[source_id],
                object_profile_use[profile_id],
                _rank(seed, cell["cell_id"], role, source_id),
            )

        declared_source_pairs = sorted(
            plan["role_ordered_source_family_pairs"],
            key=lambda record: (
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
            for left in sorted(
                left_pool, key=lambda value: object_key(value, "a")
            ):
                left_id = str(left["source"]["scene_id"])
                if object_use[left_id] >= maximum_reuse:
                    continue
                for right in sorted(
                    right_pool, key=lambda value: object_key(value, "b")
                ):
                    right_id = str(right["source"]["scene_id"])
                    if left_id == right_id or object_use[right_id] >= maximum_reuse:
                        continue
                    unordered_pair = tuple(sorted((left_id, right_id)))
                    if unordered_pair in used_pairs:
                        continue
                    if not _pair_dynamics_eligible(left, right, matrix, cell):
                        continue
                    excluded_host_ids = {left_id, right_id}
                    candidate_host = next(
                        (
                            record
                            for record in candidate_hosts
                            if str(record["source"]["scene_id"])
                            not in excluded_host_ids
                            and _pair_layout_fits_host(
                                record,
                                left,
                                right,
                                matrix,
                                cell,
                                resolved_scene_rules,
                            )
                        ),
                        None,
                    )
                    if candidate_host is None:
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
        scene_rule_use[scene_rule_id] += 1
        motion_scene_rule_use[(str(cell["motion_id"]), scene_rule_id)] += 1
        camera_scene_rule_use[(camera_view_family_id, scene_rule_id)] += 1
        scene_rule_environment_use[(scene_rule_id, environment_category)] += 1
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

    eligible_object_profiles = {str(value["visual_profile_id"]) for value in objects}
    eligible_host_profiles = {str(value["visual_profile_id"]) for value in hosts}
    object_profile_policy = str(
        plan["selection_policy"]["object_visual_profile_coverage"]
    )
    if (
        object_profile_policy == "all_eligible"
        and set(object_profile_use) != eligible_object_profiles
    ):
        raise ValueError("coverage selection misses eligible object visual profiles")
    if set(host_profile_use) != eligible_host_profiles:
        raise ValueError("coverage selection misses eligible host visual profiles")
    selected_objects = [
        record for selection in selected for record in selection["objects"]
    ]
    selected_scene_classes = sorted(
        {str(record["scene_class"]) for record in hosts}
    )
    environment_categories_by_scene = {
        scene_class: sorted(
            {
                str(record["environment_category"])
                for record in hosts
                if str(record["scene_class"]) == scene_class
            }
        )
        for scene_class in selected_scene_classes
    }
    selected_camera_families = sorted(
        {str(cell["camera_view_family_id"]) for cell in cells}
    )
    audit = {
        "unique_object_source_count": len(object_use),
        "maximum_object_source_reuse": max(object_use.values()),
        "unique_source_pair_count": len(used_pairs),
        "unique_host_count": len(host_use),
        "maximum_host_source_reuse": max(host_use.values()),
        "eligible_object_visual_profile_count": len(eligible_object_profiles),
        "selected_object_visual_profile_count": len(object_profile_use),
        "selected_source_family_pair_counts": dict(
            sorted(source_family_pair_use.items())
        ),
        "selected_camera_source_family_pair_counts": {
            camera_family_id: {
                str(source_pair["id"]): camera_source_family_pair_use[
                    (camera_family_id, str(source_pair["id"]))
                ]
                for source_pair in plan["role_ordered_source_family_pairs"]
            }
            for camera_family_id in sorted(
                {str(cell["camera_view_family_id"]) for cell in cells}
            )
        },
        "selected_object_source_family_counts": dict(
            sorted(
                Counter(
                    record["source_family"] for record in selected_objects
                ).items()
            )
        ),
        "selected_object_shape_counts": dict(
            sorted(
                Counter(
                    str(record["shape_family_id"])
                    for record in selected_objects
                ).items()
            )
        ),
        "selected_object_visual_profile_counts_by_shape": {
            shape: len(
                {
                    str(record["visual_profile_id"])
                    for record in selected_objects
                    if str(record["shape_family_id"]) == shape
                }
            )
            for shape in sorted(
                {str(record["shape_family_id"]) for record in selected_objects}
            )
        },
        "selected_host_visual_profile_count": len(host_profile_use),
        "selected_host_visual_type_counts": dict(sorted(host_type_use.items())),
        "selected_scene_rule_counts": dict(sorted(scene_rule_use.items())),
        "selected_motion_scene_rule_counts": {
            motion_id: {
                rule_id: motion_scene_rule_use[(motion_id, rule_id)]
                for rule_id in sorted(scene_rule_use)
            }
            for motion_id in sorted({str(cell["motion_id"]) for cell in cells})
        },
        "selected_camera_scene_rule_counts": {
            camera_family: {
                rule_id: camera_scene_rule_use[(camera_family, rule_id)]
                for rule_id in sorted(scene_rule_use)
            }
            for camera_family in selected_camera_families
        },
        "selected_scene_rule_environment_counts": {
            rule_id: {
                category: count
                for (selected_rule_id, category), count in sorted(
                    scene_rule_environment_use.items()
                )
                if selected_rule_id == rule_id
            }
            for rule_id in sorted(scene_rule_use)
        },
        "selected_host_environment_category_counts": dict(
            sorted(environment_use.items())
        ),
        "selected_scene_environment_category_counts": {
            scene_class: {
                category: scene_environment_use[(scene_class, category)]
                for category in environment_categories_by_scene[scene_class]
            }
            for scene_class in selected_scene_classes
        },
        "selected_camera_scene_environment_category_counts": {
            camera_family: {
                scene_class: {
                    category: camera_scene_environment_use[
                        (camera_family, scene_class, category)
                    ]
                    for category in environment_categories_by_scene[scene_class]
                }
                for scene_class in selected_scene_classes
            }
            for camera_family in selected_camera_families
        },
        "selected_source_family_pair_scene_environment_category_counts": {
            source_pair_id: {
                scene_class: {
                    category: source_pair_scene_environment_use[
                        (source_pair_id, scene_class, category)
                    ]
                    for category in environment_categories_by_scene[scene_class]
                }
                for scene_class in selected_scene_classes
            }
            for source_pair_id in sorted(source_family_pair_use)
        },
    }
    return selected, audit


def _validate_complete_scene_coverage(
    matrix: dict[str, Any],
    scene_rules: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    """Require a complete matrix to materialize every declared scene pairing."""

    declared_rules = {str(rule["id"]) for rule in scene_rules["physical_rules"]}
    selected_rules = set(audit["selected_scene_rule_counts"])
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
        (str(rule_id), str(category))
        for rule_id, counts in audit[
            "selected_scene_rule_environment_counts"
        ].items()
        for category, count in counts.items()
        if int(count) > 0
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
    declared_motion_rules = {
        (motion_id, str(rule["id"]))
        for motion_id, regime in motion_regimes.items()
        for rule in scene_rules["physical_rules"]
        if regime in rule["allowed_kinematic_regimes"]
    }
    selected_motion_rules = {
        (str(motion_id), str(rule_id))
        for motion_id, counts in audit["selected_motion_scene_rule_counts"].items()
        for rule_id, count in counts.items()
        if int(count) > 0
    }
    if selected_motion_rules != declared_motion_rules:
        raise ValueError(
            "complete two-object coverage misses motion/scene-rule pairs: "
            f"{sorted(declared_motion_rules - selected_motion_rules)}"
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
    objects, hosts, source_audit = released_source_pool(
        root=root,
        released_base_manifest_path=released_path,
        source_root=source_root,
        source_manifest_path=source_path,
        matrix=matrix,
        scene_rules=scene_rules,
    )
    selections, selection_audit = select_coverage_sources(
        cells, objects, hosts, matrix, scene_rules
    )
    if len(cells) == full_cell_count:
        _validate_complete_scene_coverage(matrix, scene_rules, selection_audit)
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
            "schema_version": "physweep_two_object_coverage_selection_v6",
            "full_cell_count": full_cell_count,
            "selected_cell_count": len(cells),
            "complete_cartesian_product": len(cells) == full_cell_count,
            "axis_counts": _axis_counts(selected_cells),
            "source_pool_audit": source_audit,
            "selection_audit": selection_audit,
        },
        "samples": samples,
        "status": "sampled_pending_simulation",
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
