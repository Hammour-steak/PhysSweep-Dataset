#!/usr/bin/env python3
"""Sample one-object scenes from decoupled motion and environment rules."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scene_kit_compiler import validate_registry_counts
from physical_proxy_catalog import (
    load_catalog,
    validate_curated_registry_bindings,
)
from sample_asset_proxy_scenes import proxy_volume_fill_ratio
from sample_pybullet_base import manifest_counts as generic_manifest_counts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT_ROOT / "configs/one_object_sampling_matrix.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def matrix_dependency_paths(
    root: Path, matrix: dict[str, Any]
) -> dict[str, Path]:
    required = {
        "generic_sampling_bundle",
        "asset_proxy_registry",
        "physical_proxy_catalog",
        "asset_scene_composition",
        "asset_semantic_scene_rules",
        "visual_sampling",
        "physics_backend",
        "backend_capabilities",
        "production_video",
        "environment_collision_proxies",
        "environment_composition",
        "passive_pinball_backend",
        "specialized_scene_backends",
    }
    declared = set(matrix.get("dependencies", {}))
    if declared != required:
        raise ValueError(
            "sampling matrix dependency set is invalid: "
            f"missing={sorted(required - declared)}, "
            f"extra={sorted(declared - required)}"
        )
    paths = {
        key: root / str(matrix["dependencies"][key]) for key in sorted(required)
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"sampling dependency is missing: {key}")
    return paths


def matrix_implementation_paths(
    root: Path, matrix: dict[str, Any]
) -> dict[str, Path]:
    required = {
        "matrix_sampler",
        "proxy_catalog",
        "asset_proxy_sampler",
        "billiards_generator",
        "billiards_renderer",
        "passive_pinball_generator",
        "passive_pinball_renderer",
        "specialized_backend_registry",
        "motion_rule_package",
        "motion_rule_contracts",
        "motion_rule_common",
        "motion_rule_registry",
        "motion_rule_planar",
        "motion_rule_ballistic",
        "motion_rule_incline",
        "motion_rule_transition",
    }
    declared = set(matrix.get("implementation", {}))
    if declared != required:
        raise ValueError(
            "sampling matrix implementation set is invalid: "
            f"missing={sorted(required - declared)}, "
            f"extra={sorted(declared - required)}"
        )
    paths = {
        key: root / str(matrix["implementation"][key]) for key in sorted(required)
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"sampling implementation is missing: {key}")
    return paths


def allocate_axis_counts(
    records: list[dict[str, Any]], count: int, axis_name: str
) -> dict[str, int]:
    if count < 0:
        raise ValueError(f"{axis_name} count cannot be negative")
    if not records:
        raise ValueError(f"{axis_name} axis cannot be empty")
    record_ids = [str(record["id"]) for record in records]
    if any(not record_id for record_id in record_ids):
        raise ValueError(f"{axis_name} axis has an empty id")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError(f"{axis_name} axis has duplicate ids")
    if any(float(record["weight"]) < 0.0 for record in records):
        raise ValueError(f"{axis_name} weights cannot be negative")
    if any(int(record.get("minimum_per_batch", 0)) < 0 for record in records):
        raise ValueError(f"{axis_name} minimum cannot be negative")
    minimum_total = sum(int(record.get("minimum_per_batch", 0)) for record in records)
    if count < minimum_total:
        raise ValueError(
            f"count {count} cannot satisfy {axis_name} minimum total {minimum_total}"
        )
    weight_total = sum(float(record["weight"]) for record in records)
    if not math.isclose(weight_total, 1.0, abs_tol=1.0e-9):
        raise ValueError(f"{axis_name} weights must sum to one")
    raw = [count * float(record["weight"]) / weight_total for record in records]
    floors = [math.floor(value) for value in raw]
    allocation = {
        str(record["id"]): value for record, value in zip(records, floors)
    }
    remainder = count - sum(floors)
    order = sorted(
        range(len(records)),
        key=lambda index: (-(raw[index] - floors[index]), index),
    )
    for index in order[:remainder]:
        allocation[str(records[index]["id"])] += 1
    for record in records:
        record_id = str(record["id"])
        minimum = int(record.get("minimum_per_batch", 0))
        while allocation[record_id] < minimum:
            donors = sorted(
                (
                    other
                    for other in records
                    if allocation[str(other["id"])]
                    > int(other.get("minimum_per_batch", 0))
                ),
                key=lambda other: (
                    -(
                        allocation[str(other["id"])]
                        - int(other.get("minimum_per_batch", 0))
                    ),
                    str(other["id"]),
                ),
            )
            if not donors:
                raise ValueError(f"{axis_name} minimum allocation has no donor")
            donor_id = str(donors[0]["id"])
            allocation[donor_id] -= 1
            allocation[record_id] += 1
    return allocation


def repeated_cycle(values: list[str], count: int, rng: random.Random) -> list[str]:
    if not values:
        raise ValueError("matrix cycle cannot be empty")
    result: list[str] = []
    while len(result) < count:
        cycle = list(values)
        rng.shuffle(cycle)
        result.extend(cycle)
    return result[:count]


def least_used_choices(
    values: list[str],
    count: int,
    rng: random.Random,
    usage: Counter[str],
) -> list[str]:
    if not values:
        raise ValueError("matrix choice pool cannot be empty")
    result = []
    for _ in range(count):
        minimum = min(usage[value] for value in values)
        candidates = [value for value in values if usage[value] == minimum]
        selected = str(rng.choice(candidates))
        usage[selected] += 1
        result.append(selected)
    return result


def compatible_quota_feasible(
    motion_counts: Counter[str],
    environments: list[dict[str, Any]],
    quotas: dict[str, int],
) -> bool:
    required_flow = sum(quotas.get(str(environment["id"]), 0) for environment in environments)
    if required_flow == 0:
        return True
    if required_flow > sum(motion_counts.values()):
        return False

    source = "__source__"
    sink = "__sink__"
    adjacency: dict[str, list[str]] = defaultdict(list)
    capacity: dict[tuple[str, str], int] = {}

    def add_edge(start: str, end: str, value: int) -> None:
        adjacency[start].append(end)
        adjacency[end].append(start)
        capacity[(start, end)] = value
        capacity[(end, start)] = 0

    for environment in environments:
        environment_id = str(environment["id"])
        quota = int(quotas.get(environment_id, 0))
        if quota <= 0:
            continue
        environment_node = f"environment:{environment_id}"
        add_edge(source, environment_node, quota)
        for motion in sorted(environment["motion_bindings"]):
            if motion_counts[motion] > 0:
                add_edge(environment_node, f"motion:{motion}", quota)
    for motion, available in sorted(motion_counts.items()):
        if available > 0:
            add_edge(f"motion:{motion}", sink, int(available))

    total_flow = 0
    while total_flow < required_flow:
        visited: set[str] = set()

        def augment(node: str, flow: int) -> int:
            if node == sink:
                return flow
            visited.add(node)
            for neighbor in adjacency[node]:
                residual = capacity[(node, neighbor)]
                if residual <= 0 or neighbor in visited:
                    continue
                pushed = augment(neighbor, min(flow, residual))
                if pushed:
                    capacity[(node, neighbor)] -= pushed
                    capacity[(neighbor, node)] += pushed
                    return pushed
            return 0

        pushed = augment(source, required_flow - total_flow)
        if not pushed:
            return False
        total_flow += pushed
    return True


def assign_environment_ids(
    motions: list[str],
    environments: list[dict[str, Any]],
    allocations: dict[str, int],
    declared_motion_ids: set[str],
    rng: random.Random,
) -> list[str]:
    if not set(motions) <= declared_motion_ids:
        raise ValueError("motion schedule contains an undeclared motion")
    assignment: list[str | None] = [None] * len(motions)
    usage: Counter[tuple[str, str]] = Counter()
    motion_counts = Counter(motions)
    specialized = sorted(
        (
            environment
            for environment in environments
            if set(environment["motion_bindings"]) != declared_motion_ids
        ),
        key=lambda environment: (
            len(environment["motion_bindings"]),
            str(environment["id"]),
        ),
    )
    catch_all = [
        environment
        for environment in environments
        if set(environment["motion_bindings"]) == declared_motion_ids
    ]
    if len(catch_all) != 1:
        raise ValueError("decoupled matrix requires exactly one catch-all environment")
    catch_all_environment = catch_all[0]
    reserve_fraction = float(
        catch_all_environment.get("minimum_per_motion_fraction_of_batch", 0.0)
    )
    if not 0.0 <= reserve_fraction <= 1.0:
        raise ValueError("catch-all motion reserve fraction must be between zero and one")
    reserve_per_motion = math.floor(len(motions) * reserve_fraction)
    reserved_motion_counts = {
        motion: reserve_per_motion for motion in sorted(declared_motion_ids)
    }
    if sum(reserved_motion_counts.values()) > allocations[str(catch_all_environment["id"])]:
        raise ValueError("catch-all environment quota cannot satisfy motion reserves")
    for motion, reserved in reserved_motion_counts.items():
        if motion_counts[motion] < reserved:
            raise ValueError(f"motion allocation cannot satisfy catch-all reserve: {motion}")
        motion_counts[motion] -= reserved
    remaining_quotas = {
        str(environment["id"]): int(allocations[str(environment["id"])])
        for environment in specialized
    }
    if not compatible_quota_feasible(motion_counts, specialized, remaining_quotas):
        raise ValueError("compatible environment quotas have no feasible assignment")
    for environment in specialized:
        environment_id = str(environment["id"])
        compatible = set(environment["motion_bindings"])
        for _ in range(allocations[environment_id]):
            candidate_motions = [
                motion
                for motion in sorted(compatible)
                if motion_counts[motion] > 0
            ]
            tie_break = {
                motion: rng.random()
                for motion in candidate_motions
            }
            candidate_motions.sort(
                key=lambda motion: (
                    usage[(environment_id, motion)],
                    tie_break[motion],
                    motion,
                )
            )
            selected_motion = None
            for motion in candidate_motions:
                motion_counts[motion] -= 1
                remaining_quotas[environment_id] -= 1
                if compatible_quota_feasible(
                    motion_counts, specialized, remaining_quotas
                ):
                    selected_motion = motion
                    break
                motion_counts[motion] += 1
                remaining_quotas[environment_id] += 1
            if selected_motion is None:
                raise ValueError(
                    f"cannot satisfy compatible environment quota: {environment_id}"
                )
            candidates = [
                index
                for index, motion in enumerate(motions)
                if assignment[index] is None and motion == selected_motion
            ]
            selected_index = rng.choice(candidates)
            assignment[selected_index] = environment_id
            usage[(environment_id, selected_motion)] += 1
    catch_all_id = str(catch_all_environment["id"])
    unassigned = [index for index, value in enumerate(assignment) if value is None]
    if len(unassigned) != allocations[catch_all_id]:
        raise ValueError("catch-all environment quota does not match unassigned motions")
    for index in unassigned:
        assignment[index] = catch_all_id
    return [str(value) for value in assignment]


def build_schedule(
    matrix: dict[str, Any],
    count: int,
    seed: int,
    profile_dynamic_eligibility: dict[str, set[str]] | None = None,
) -> list[dict[str, Any]]:
    profile_dynamic_eligibility = profile_dynamic_eligibility or {}
    rng = random.Random(seed)
    motion_allocations = allocate_axis_counts(
        matrix["motion_intents"], count, "motion"
    )
    environment_allocations = allocate_axis_counts(
        matrix["environments"], count, "environment"
    )
    motions: list[str] = []
    for motion in matrix["motion_intents"]:
        motions.extend(
            [str(motion["id"])] * motion_allocations[str(motion["id"])]
        )
    rng.shuffle(motions)
    environment_ids = assign_environment_ids(
        motions,
        matrix["environments"],
        environment_allocations,
        set(motion_allocations),
        rng,
    )
    environments_by_id = {
        str(environment["id"]): environment
        for environment in matrix["environments"]
    }
    slots: list[dict[str, Any]] = []
    dynamic_usage: Counter[str] = Counter()
    profile_usage: Counter[tuple[str, str]] = Counter()
    for environment_id, environment in environments_by_id.items():
        selected_motion_records = [
            (index, motion)
            for index, (motion, selected_environment) in enumerate(
                zip(motions, environment_ids)
            )
            if selected_environment == environment_id
        ]
        selected_motions = [motion for _, motion in selected_motion_records]
        environment_count = len(selected_motions)
        profiles = []
        for motion in selected_motions:
            profile_pool = [
                str(value)
                for value in environment["motion_bindings"][motion]
            ]
            minimum = min(
                profile_usage[(environment_id, profile)]
                for profile in profile_pool
            )
            candidates = [
                profile
                for profile in profile_pool
                if profile_usage[(environment_id, profile)] == minimum
            ]
            profile = str(rng.choice(candidates))
            profile_usage[(environment_id, profile)] += 1
            profiles.append(profile)
        support_prop_pairs = environment.get("support_prop_pairs")
        support_dynamic_entries = environment.get("support_dynamic_entries")
        if support_prop_pairs:
            pair_values = repeated_cycle(
                [str(index) for index in range(len(support_prop_pairs))],
                environment_count,
                rng,
            )
            supports = [
                str(support_prop_pairs[int(index)]["support_asset_id"])
                for index in pair_values
            ]
            static_props = [
                str(support_prop_pairs[int(index)]["static_prop_asset_id"])
                for index in pair_values
            ]
            dynamic_pool_ids = [
                str(support_prop_pairs[int(index)]["dynamic_pool_id"])
                for index in pair_values
            ]
        elif support_dynamic_entries:
            entry_usage: Counter[int] = Counter()
            entry_values = []
            for profile in profiles:
                candidates = [
                    index
                    for index, entry in enumerate(support_dynamic_entries)
                    if profile in set(entry["profiles"])
                ]
                if not candidates:
                    raise ValueError(
                        f"no support entry admits profile {profile} in "
                        f"{environment_id}"
                    )
                minimum = min(entry_usage[index] for index in candidates)
                least_used = [
                    index for index in candidates if entry_usage[index] == minimum
                ]
                selected = int(rng.choice(least_used))
                entry_usage[selected] += 1
                entry_values.append(str(selected))
            supports = [
                str(support_dynamic_entries[int(index)]["support_asset_id"])
                for index in entry_values
            ]
            static_props = ["none"] * environment_count
            dynamic_pool_ids = [
                str(support_dynamic_entries[int(index)]["dynamic_pool_id"])
                for index in entry_values
            ]
        else:
            supports = repeated_cycle(
                [
                    str(value)
                    for value in environment.get("support_asset_ids", ["none"])
                ],
                environment_count,
                rng,
            )
            static_props = ["none"] * environment_count
            dynamic_pool_ids = ["none"] * environment_count
        if any(value != "none" for value in dynamic_pool_ids):
            pools = environment["dynamic_pools"]
            dynamics = []
            for pool_id, profile in zip(dynamic_pool_ids, profiles):
                pool = [str(value) for value in pools[pool_id]]
                eligible = profile_dynamic_eligibility.get(profile)
                if eligible is not None:
                    pool = [value for value in pool if value in eligible]
                if not pool:
                    raise ValueError(
                        f"dynamic pool {pool_id} has no asset eligible for "
                        f"profile {profile}"
                    )
                dynamics.extend(
                    least_used_choices(
                        pool,
                        1,
                        rng,
                        dynamic_usage,
                    )
                )
        else:
            dynamic_values = [
                str(value)
                for value in environment.get("dynamic_asset_ids", ["none"])
            ]
            if dynamic_values == ["none"]:
                dynamics = repeated_cycle(dynamic_values, environment_count, rng)
            else:
                dynamics = []
                for profile in profiles:
                    pool = dynamic_values
                    eligible = profile_dynamic_eligibility.get(profile)
                    if eligible is not None:
                        pool = [value for value in pool if value in eligible]
                    if not pool:
                        raise ValueError(
                            f"environment {environment_id} has no asset eligible "
                            f"for profile {profile}"
                        )
                    dynamics.extend(
                        least_used_choices(pool, 1, rng, dynamic_usage)
                    )
        for (
            (motion_slot_index, motion),
            profile,
            support_id,
            static_prop_id,
            dynamic_id,
        ) in zip(
            selected_motion_records, profiles, supports, static_props, dynamics
        ):
            slots.append(
                {
                    "_motion_slot_index": motion_slot_index,
                    "motion_intent": motion,
                    "environment_id": environment_id,
                    "generator": str(environment["generator"]),
                    "profile": profile,
                    "support_asset_id": None if support_id == "none" else support_id,
                    "static_prop_asset_id": (
                        None if static_prop_id == "none" else static_prop_id
                    ),
                    "dynamic_asset_id": None if dynamic_id == "none" else dynamic_id,
                }
            )
    slots.sort(key=lambda slot: int(slot.pop("_motion_slot_index")))
    for index, slot in enumerate(slots, start=1):
        slot["index"] = index
        slot["scene_id"] = (
            f"physweep1scene_{index:06d}_{slot['motion_intent']}_"
            f"{slot['environment_id']}"
        )
        slot["seed"] = rng.randrange(1, 2**31 - 1)
    return slots


def validate_matrix(root: Path, matrix: dict[str, Any]) -> None:
    dependencies = matrix_dependency_paths(root, matrix)
    matrix_implementation_paths(root, matrix)
    registry = load_json(dependencies["asset_proxy_registry"])
    validate_registry_counts(registry)
    _, physical_proxy_records = load_catalog(
        root, dependencies["physical_proxy_catalog"]
    )
    validate_curated_registry_bindings(physical_proxy_records, registry)
    composition = load_json(dependencies["asset_scene_composition"])
    semantic_rules = load_json(dependencies["asset_semantic_scene_rules"])
    backend = load_json(dependencies["physics_backend"])
    capabilities = load_json(dependencies["backend_capabilities"])
    if capabilities["active_backend"] != backend["backend_id"]:
        raise ValueError("capability id does not match physics backend")
    specialized = capabilities["specialized_scopes"]
    asset_profiles = set(backend["asset_proxy_rules"]["motion_profiles"])
    if set(specialized["asset_proxy_single_object"]["profiles"]) != asset_profiles:
        raise ValueError("asset-proxy capabilities do not match backend profiles")
    all_semantic_profiles = [
        str(profile)
        for family in semantic_rules["specialized_scene_families"].values()
        for profile in family["profiles"]
    ]
    if len(all_semantic_profiles) != len(set(all_semantic_profiles)):
        raise ValueError("specialized semantic profile belongs to multiple families")
    semantic_family_by_profile = {
        str(profile): str(family_id)
        for family_id, family in semantic_rules["specialized_scene_families"].items()
        for profile in family["profiles"]
    }
    semantic_rule_by_profile = {
        str(profile): family
        for family in semantic_rules["specialized_scene_families"].values()
        for profile in family["profiles"]
    }
    billiards_profiles = set(backend["billiards_rules"]["initial_states"])
    if set(specialized["billiards"]["profiles"]) != billiards_profiles:
        raise ValueError("billiards capabilities do not match backend profiles")
    semantic_body_counts = {
        int(family["dynamic_object_count"])
        for family in semantic_rules["specialized_scene_families"].values()
    }
    if set(specialized["billiards"]["dynamic_body_counts"]) != semantic_body_counts:
        raise ValueError("billiards capabilities do not match semantic body counts")
    pinball_backend = load_json(dependencies["passive_pinball_backend"])
    pinball_profiles = set(pinball_backend["profiles"])
    if set(specialized["passive_pinball"]["profiles"]) != pinball_profiles:
        raise ValueError("passive-pinball capabilities do not match backend profiles")
    pinball_family = semantic_rules["specialized_scene_families"].get(
        "passive_pinball_single_ball", {}
    )
    if (
        set(pinball_family.get("profiles", [])) != pinball_profiles
        or int(pinball_family.get("dynamic_object_count", 0)) != 1
        or bool(pinball_family.get("active_mechanisms_supported", True))
    ):
        raise ValueError("passive-pinball semantic contract is inconsistent")
    registry_by_id = {record["asset_id"]: record for record in registry["records"]}
    composition_by_id = {
        record["asset_id"]: record for record in composition["records"]
    }
    motion_ids = [str(motion["id"]) for motion in matrix["motion_intents"]]
    if len(motion_ids) != len(set(motion_ids)):
        raise ValueError("duplicate motion intent id")
    environment_ids = [
        str(environment["id"]) for environment in matrix["environments"]
    ]
    if len(environment_ids) != len(set(environment_ids)):
        raise ValueError("duplicate environment id")
    allocate_axis_counts(matrix["motion_intents"], 40, "motion")
    allocate_axis_counts(matrix["environments"], 40, "environment")
    generic_bundle = load_json(dependencies["generic_sampling_bundle"])
    generic_rules = load_json(root / str(generic_bundle["base_rules"]))
    generic_motion_ids = set(generic_rules["axes"]["motion_axis"])
    if set(motion_ids) != generic_motion_ids:
        raise ValueError("outer motion intents do not match the generic motion axis")
    declared_profiles = {
        profile
        for family in semantic_rules["specialized_scene_families"].values()
        for profile in family["profiles"]
    }
    if int(matrix["policy"]["dynamic_object_count"]) != 1:
        raise ValueError("one-object sampling matrix must declare one dynamic object")
    catch_all_count = 0
    for environment in matrix["environments"]:
        bindings = environment.get("motion_bindings", {})
        if not bindings or any(not profiles for profiles in bindings.values()):
            raise ValueError(
                f"environment has an empty motion binding: {environment['id']}"
            )
        if not set(bindings) <= set(motion_ids):
            raise ValueError(
                f"environment binds an unknown motion: {environment['id']}"
            )
        if set(bindings) == set(motion_ids):
            catch_all_count += 1
        specialized_support_axes = sum(
            bool(environment.get(key))
            for key in (
                "support_asset_ids",
                "support_prop_pairs",
                "support_dynamic_entries",
            )
        )
        if specialized_support_axes > 1:
            raise ValueError(
                f"environment mixes support-axis declarations: {environment['id']}"
            )
        dynamic_pools = environment.get("dynamic_pools", {})
        if any(not values for values in dynamic_pools.values()):
            raise ValueError(
                f"environment has an empty dynamic pool: {environment['id']}"
            )
        generator = str(environment["generator"])
        profiles = {
            str(profile)
            for values in bindings.values()
            for profile in values
        }
        if generator == "pybullet_base":
            if (
                set(bindings) != set(motion_ids)
                or profiles != {"five_dimensional_matrix"}
            ):
                raise ValueError("generic environment must bind every motion")
        elif generator == "asset_proxy":
            if not profiles <= asset_profiles:
                raise ValueError(
                    f"asset environment has unsupported profiles: {environment['id']}"
                )
        elif generator == "billiards":
            if not profiles <= declared_profiles or not profiles <= billiards_profiles:
                raise ValueError(
                    f"billiards environment has undeclared profiles: "
                    f"{environment['id']}"
                )
            if any(
                int(semantic_rule_by_profile[profile]["dynamic_object_count"]) != 1
                for profile in profiles
            ):
                raise ValueError("multi-object billiards profile leaked into one-object matrix")
        elif generator == "passive_pinball":
            if profiles != pinball_profiles:
                raise ValueError(
                    "passive-pinball environment must bind every declared profile"
                )
            if specialized_support_axes:
                raise ValueError(
                    "passive-pinball fixture must not be represented as an asset axis"
                )
        else:
            raise ValueError(f"unknown environment generator: {generator}")
        for asset_id in environment.get("support_asset_ids", []):
            if asset_id not in registry_by_id or asset_id not in composition_by_id:
                raise ValueError(f"unknown support asset in matrix: {asset_id}")
            record = registry_by_id[asset_id]
            if not record["admission"].get("sampling_enabled", False):
                raise ValueError(f"disabled support asset in matrix: {asset_id}")
            if record["proxy"]["kind"] != "support_compound":
                raise ValueError(f"non-support asset in support axis: {asset_id}")
            if composition_by_id[asset_id]["sampling_status"] not in {
                "ready_generic",
                "ready_specialized",
            }:
                raise ValueError(f"unreviewed support asset in matrix: {asset_id}")
            if generator == "billiards":
                semantic_families = {
                    semantic_family_by_profile[profile] for profile in profiles
                }
                semantic_categories = {
                    str(semantic_rule_by_profile[profile]["support_category"])
                    for profile in profiles
                }
                if record["semantic_category"] not in semantic_categories:
                    raise ValueError(
                        f"billiards support has incompatible semantics: {asset_id}"
                    )
                allowed = set(
                    composition_by_id[asset_id]["scene_fit"]["allowed"]
                )
                if not semantic_families <= allowed:
                    raise ValueError(
                        f"billiards support lacks declared scene admission: {asset_id}"
                    )
        for pair in environment.get("support_prop_pairs", []):
            support_id = str(pair["support_asset_id"])
            prop_id = str(pair["static_prop_asset_id"])
            pool_id = str(pair["dynamic_pool_id"])
            if pool_id not in dynamic_pools:
                raise ValueError(
                    f"unknown dynamic pool for support/prop pair: {pool_id}"
                )
            if support_id not in registry_by_id or support_id not in composition_by_id:
                raise ValueError(f"unknown paired support asset in matrix: {support_id}")
            support = registry_by_id[support_id]
            if (
                not support["admission"].get("sampling_enabled", False)
                or support["proxy"]["kind"] != "support_compound"
            ):
                raise ValueError(f"unreviewed paired support asset in matrix: {support_id}")
            if composition_by_id[support_id]["sampling_status"] != "ready_generic":
                raise ValueError(f"non-generic paired support asset in matrix: {support_id}")
            if prop_id not in registry_by_id or prop_id not in composition_by_id:
                raise ValueError(f"unknown static prop asset in matrix: {prop_id}")
            prop = registry_by_id[prop_id]
            if (
                not prop["admission"].get("sampling_enabled", False)
                or prop["proxy"]["kind"] != "static_compound"
                or composition_by_id[prop_id]["sampling_status"] != "ready_static"
            ):
                raise ValueError(f"unreviewed static prop asset in matrix: {prop_id}")
        for entry in environment.get("support_dynamic_entries", []):
            support_id = str(entry["support_asset_id"])
            pool_id = str(entry["dynamic_pool_id"])
            admitted_profiles = {str(value) for value in entry.get("profiles", [])}
            if not admitted_profiles or not admitted_profiles <= profiles:
                raise ValueError(
                    f"support entry has invalid profile admission: {support_id}"
                )
            if pool_id not in dynamic_pools:
                raise ValueError(
                    f"unknown dynamic pool for support entry: {pool_id}"
                )
            if support_id not in registry_by_id or support_id not in composition_by_id:
                raise ValueError(f"unknown pooled support asset in matrix: {support_id}")
            support = registry_by_id[support_id]
            if (
                not support["admission"].get("sampling_enabled", False)
                or support["proxy"]["kind"] != "support_compound"
            ):
                raise ValueError(f"unreviewed pooled support asset in matrix: {support_id}")
            if composition_by_id[support_id]["sampling_status"] != "ready_generic":
                raise ValueError(f"non-generic pooled support asset in matrix: {support_id}")
            clear_exit_directions = (
                support.get("proxy", {})
                .get("interaction_policy", {})
                .get("clear_exit_directions_xy", [])
            )
            if "edge_exit" in admitted_profiles and not clear_exit_directions:
                raise ValueError(
                    f"edge-exit support lacks reviewed exit directions: {support_id}"
                )
        if environment.get("support_dynamic_entries"):
            covered_profiles = {
                str(profile)
                for entry in environment["support_dynamic_entries"]
                for profile in entry["profiles"]
            }
            if covered_profiles != profiles:
                raise ValueError(
                    f"support entries do not cover environment profiles: "
                    f"{environment['id']}"
                )
        declared_dynamic_ids = list(environment.get("dynamic_asset_ids", []))
        declared_dynamic_ids.extend(
            asset_id
            for values in dynamic_pools.values()
            for asset_id in values
        )
        for asset_id in declared_dynamic_ids:
            if asset_id not in registry_by_id:
                raise ValueError(f"unknown dynamic asset in matrix: {asset_id}")
            record = registry_by_id[asset_id]
            if (
                not record["admission"].get("sampling_enabled", False)
                or record["proxy"]["kind"] != "dynamic_rigid"
            ):
                raise ValueError(f"unreviewed dynamic asset in matrix: {asset_id}")
    if catch_all_count != 1:
        raise ValueError("sampling matrix must define exactly one catch-all environment")


def run(command: list[str], root: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{tail}")


def generic_retry_seed(base_seed: int, slot_index: int, attempt_number: int) -> int:
    """Return a deterministic, slot-specific seed for a replacement candidate."""
    if attempt_number < 2:
        raise ValueError("generic retry attempt numbers start at 2")
    return int(
        base_seed
        + 1_000_003 * (attempt_number - 1)
        + 10_007 * (slot_index + 1)
    )


def sample_generic_candidate_batch(
    *,
    root: Path,
    bundle_path: Path,
    output_dataset: str,
    motions: list[str],
    seed: int,
    duration_s: float,
    output_fps: int,
    resolution: list[int],
    render_samples: int,
    physics_workers: int,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    """Sample, simulate, and audit one deterministic generic candidate batch."""
    run(
        [
            sys.executable,
            str(root / "tools/sample_pybullet_base.py"),
            "--root",
            str(root),
            "--bundle",
            str(bundle_path),
            "--output-dataset",
            output_dataset,
            "--count",
            str(len(motions)),
            "--seed",
            str(seed),
            "--duration",
            str(duration_s),
            "--fps",
            str(output_fps),
            "--resolution",
            *[str(value) for value in resolution],
            "--samples",
            str(render_samples),
            "--motions",
            *motions,
        ],
        root,
    )
    manifest_path = root / "datasets" / output_dataset / "manifest.json"
    manifest = load_json(manifest_path)
    if len(manifest["samples"]) != len(motions):
        raise RuntimeError("generic candidate batch returned an unexpected sample count")
    simulation_output_root = manifest_path.parent / "physics"
    run(
        [
            sys.executable,
            str(root / "tools/run_pybullet_batch.py"),
            "--root",
            str(root),
            "--manifest",
            str(manifest_path),
            "--output-root",
            str(simulation_output_root),
            "--workers",
            str(physics_workers),
            "--allow-audit-rejections",
        ],
        root,
    )
    simulation_manifest_path = simulation_output_root / "manifest.json"
    simulation_manifest = load_json(simulation_manifest_path)
    if int(simulation_manifest["sample_count"]) != len(motions):
        raise RuntimeError("generic simulation batch returned an unexpected sample count")
    return manifest_path, manifest, simulation_manifest_path, simulation_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--resolution", nargs=2, type=int)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--physics-workers", type=int, default=8)
    parser.add_argument("--generic-max-attempts", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.generic_max_attempts <= 0:
        raise SystemExit("--generic-max-attempts must be positive")
    root = args.root.resolve()
    matrix_path = args.matrix.resolve()
    matrix = load_json(matrix_path)
    validate_matrix(root, matrix)
    dependency_paths = matrix_dependency_paths(root, matrix)
    implementation_paths = matrix_implementation_paths(root, matrix)
    production_spec = load_json(dependency_paths["production_video"])
    duration_s = float(production_spec["duration_s"])
    output_fps = int(production_spec["output_fps"])
    resolution = list(args.resolution or production_spec["resolution"])
    render_samples = int(
        args.samples if args.samples is not None else production_spec["samples"]
    )
    expected_frame_count = int(round(duration_s * output_fps)) + 1
    if expected_frame_count != int(production_spec["frame_count"]):
        raise ValueError("production video frame count is inconsistent")
    if duration_s <= 0.0 or output_fps <= 0 or min(resolution) <= 0 or render_samples <= 0:
        raise ValueError("production video values must be positive")
    registry = load_json(dependency_paths["asset_proxy_registry"])
    backend = load_json(dependency_paths["physics_backend"])
    edge_rules = backend["asset_proxy_rules"]["motion_profiles"]["edge_exit"]
    minimum_edge_fill = float(edge_rules["minimum_proxy_volume_fill_ratio"])
    edge_eligible_dynamic_ids = {
        str(record["asset_id"])
        for record in registry["records"]
        if record["proxy"]["kind"] == "dynamic_rigid"
        and record["admission"].get("sampling_enabled", False)
        and proxy_volume_fill_ratio(record) >= minimum_edge_fill
    }
    schedule = build_schedule(
        matrix,
        args.count,
        args.seed,
        profile_dynamic_eligibility={"edge_exit": edge_eligible_dynamic_ids},
    )
    dataset_root = (root / "datasets" / args.output_dataset).resolve()
    datasets_root = (root / "datasets").resolve()
    if datasets_root not in dataset_root.parents:
        raise ValueError("output dataset must remain under the project datasets directory")
    if dataset_root.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite: {dataset_root}")
        shutil.rmtree(dataset_root)
    dataset_root.mkdir(parents=True)

    records: list[dict[str, Any]] = []
    generic_slots = [slot for slot in schedule if slot["generator"] == "pybullet_base"]
    if generic_slots:
        candidate_root = f"{args.output_dataset}/generic_matrix/candidates"
        (
            initial_manifest_path,
            initial_manifest,
            initial_simulation_path,
            initial_simulation,
        ) = sample_generic_candidate_batch(
            root=root,
            bundle_path=dependency_paths["generic_sampling_bundle"],
            output_dataset=f"{candidate_root}/initial",
            motions=[str(slot["motion_intent"]) for slot in generic_slots],
            seed=args.seed,
            duration_s=duration_s,
            output_fps=output_fps,
            resolution=resolution,
            render_samples=render_samples,
            physics_workers=args.physics_workers,
        )
        initial_simulation_by_scene = {
            str(record["scene_id"]): record
            for record in initial_simulation["records"]
        }
        accepted_samples: list[dict[str, Any]] = []
        accepted_simulations: list[dict[str, Any]] = []
        candidate_attempts: list[dict[str, Any]] = []
        rejected_check_counts: Counter[str] = Counter()
        candidate_wall_time_s = 0.0
        for slot, initial_child in zip(generic_slots, initial_manifest["samples"]):
            child = initial_child
            simulation_record = initial_simulation_by_scene[str(child["scene_id"])]
            candidate_manifest_path = initial_manifest_path
            candidate_simulation_path = initial_simulation_path
            candidate_seed = args.seed
            attempt_number = 1
            while True:
                accepted = bool(
                    simulation_record.get("ok")
                    and simulation_record.get("audit_passed")
                )
                failed_checks = list(simulation_record.get("failed_checks", []))
                rejected_check_counts.update([] if accepted else failed_checks)
                candidate_wall_time_s += float(simulation_record.get("wall_time_s", 0.0))
                candidate_attempts.append(
                    {
                        "slot_index": int(slot["index"]),
                        "matrix_scene_id": str(slot["scene_id"]),
                        "motion_intent": str(slot["motion_intent"]),
                        "attempt_number": attempt_number,
                        "seed": candidate_seed,
                        "accepted": accepted,
                        "candidate_scene_id": str(child["scene_id"]),
                        "metadata_path": str(child["metadata_path"]),
                        "metadata_sha256": str(child["metadata_sha256"]),
                        "candidate_manifest_path": str(
                            candidate_manifest_path.relative_to(root)
                        ),
                        "candidate_simulation_manifest_path": str(
                            candidate_simulation_path.relative_to(root)
                        ),
                        "audit_path": simulation_record.get("audit_path"),
                        "audit_sha256": simulation_record.get("audit_sha256"),
                        "failed_checks": failed_checks,
                        "error": simulation_record.get("error"),
                    }
                )
                if accepted:
                    break
                if attempt_number >= args.generic_max_attempts:
                    raise RuntimeError(
                        "generic slot exhausted candidate attempts: "
                        f"index={slot['index']} motion={slot['motion_intent']} "
                        f"failed_checks={failed_checks}"
                    )
                attempt_number += 1
                candidate_seed = generic_retry_seed(
                    args.seed, int(slot["index"]), attempt_number
                )
                retry_dataset = (
                    f"{candidate_root}/slot_{int(slot['index']):06d}/"
                    f"attempt_{attempt_number:02d}"
                )
                (
                    candidate_manifest_path,
                    retry_manifest,
                    candidate_simulation_path,
                    retry_simulation,
                ) = sample_generic_candidate_batch(
                    root=root,
                    bundle_path=dependency_paths["generic_sampling_bundle"],
                    output_dataset=retry_dataset,
                    motions=[str(slot["motion_intent"])],
                    seed=candidate_seed,
                    duration_s=duration_s,
                    output_fps=output_fps,
                    resolution=resolution,
                    render_samples=render_samples,
                    physics_workers=1,
                )
                child = retry_manifest["samples"][0]
                simulation_record = retry_simulation["records"][0]
            child = copy.deepcopy(child)
            trajectory_path = Path(str(simulation_record["trajectory_path"])).resolve()
            simulation_record_path = trajectory_path.with_name(
                "simulation_record.json"
            )
            if not simulation_record_path.exists():
                raise RuntimeError(
                    f"accepted simulation record is missing: {simulation_record_path}"
                )
            child["simulation_record_path"] = str(
                simulation_record_path.relative_to(root)
            )
            child["trajectory_path"] = str(trajectory_path.relative_to(root))
            accepted_samples.append(child)
            accepted_simulations.append(simulation_record)

        generic_root = dataset_root / "generic_matrix"
        attempts_manifest_path = generic_root / "candidate_attempts.json"
        rejected_candidate_count = sum(
            not record["accepted"] for record in candidate_attempts
        )
        write_json(
            attempts_manifest_path,
            {
                "schema_version": "physweep_generic_candidate_attempts_v1",
                "dataset_id": f"{args.output_dataset}_generic_candidates",
                "seed": args.seed,
                "slot_count": len(generic_slots),
                "candidate_count": len(candidate_attempts),
                "accepted_candidate_count": len(accepted_samples),
                "rejected_candidate_count": rejected_candidate_count,
                "first_attempt_pass_count": sum(
                    record["accepted"] and record["attempt_number"] == 1
                    for record in candidate_attempts
                ),
                "max_attempts_per_slot": args.generic_max_attempts,
                "failed_check_counts": dict(rejected_check_counts),
                "records": candidate_attempts,
            },
        )
        child_manifest = copy.deepcopy(initial_manifest)
        child_manifest_path = generic_root / "manifest.json"
        child_manifest.update(
            {
                "dataset_id": f"{args.output_dataset}/generic_matrix",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "sample_count": len(accepted_samples),
                "coverage": generic_manifest_counts(
                    [
                        load_json(root / str(sample["metadata_path"]))
                        for sample in accepted_samples
                    ]
                ),
                "samples": accepted_samples,
                "status": "simulated_accepted",
                "acceptance": {
                    "candidate_attempts_manifest_path": str(
                        attempts_manifest_path.relative_to(root)
                    ),
                    "candidate_count": len(candidate_attempts),
                    "rejected_candidate_count": rejected_candidate_count,
                    "max_attempts_per_slot": args.generic_max_attempts,
                },
            }
        )
        write_json(child_manifest_path, child_manifest)
        simulation_manifest_path = generic_root / "simulation_manifest.json"
        write_json(
            simulation_manifest_path,
            {
                "schema_version": "physweep_pybullet_batch_record_v1",
                "dataset_id": child_manifest["dataset_id"],
                "source_manifest": str(child_manifest_path),
                "sample_count": len(accepted_simulations),
                "passed_count": len(accepted_simulations),
                "rejected_count": 0,
                "error_count": 0,
                "pass_rate": 1.0,
                "failed_check_counts": {},
                "wall_time_s": round(candidate_wall_time_s, 6),
                "workers": max(1, args.physics_workers),
                "candidate_count": len(candidate_attempts),
                "rejected_candidate_count": rejected_candidate_count,
                "candidate_failed_check_counts": dict(rejected_check_counts),
                "candidate_attempts_manifest_path": str(
                    attempts_manifest_path.relative_to(root)
                ),
                "records": accepted_simulations,
            },
        )
        for slot, child in zip(generic_slots, accepted_samples):
            metadata_path = root / str(child["metadata_path"])
            if sha256(metadata_path) != str(child["metadata_sha256"]):
                raise RuntimeError("generic child metadata hash mismatch")
            metadata = load_json(metadata_path)
            generated_motion = str(
                metadata["simulation"]["objects"][0]["expected_motion"][
                    "motion_family"
                ]
            )
            if generated_motion != slot["motion_intent"]:
                raise RuntimeError(
                    "generic child motion does not match the outer schedule: "
                    f"{generated_motion} != {slot['motion_intent']}"
                )
            records.append(
                {
                    **slot,
                    "pipeline": "generic_pybullet",
                    "metadata_path": child["metadata_path"],
                    "metadata_sha256": child["metadata_sha256"],
                    "status": "simulated_accepted",
                }
            )

    asset_records: list[dict[str, Any]] = []
    billiards_records: list[dict[str, Any]] = []
    passive_pinball_records: list[dict[str, Any]] = []
    for slot in schedule:
        if slot["generator"] == "pybullet_base":
            continue
        scene_dir = dataset_root / "specialized" / slot["scene_id"]
        if slot["generator"] == "billiards":
            run(
                [
                    sys.executable,
                    str(implementation_paths["billiards_generator"]),
                    "--root",
                    str(root),
                    "--output",
                    str(scene_dir),
                    "--profile",
                    slot["profile"],
                    "--support-id",
                    slot["support_asset_id"],
                    "--registry",
                    str(dependency_paths["asset_proxy_registry"]),
                    "--catalog",
                    str(dependency_paths["physical_proxy_catalog"]),
                    "--semantic-rules",
                    str(dependency_paths["asset_semantic_scene_rules"]),
                    "--composition-rules",
                    str(dependency_paths["asset_scene_composition"]),
                    "--backend",
                    str(dependency_paths["physics_backend"]),
                    "--visual-rules",
                    str(dependency_paths["visual_sampling"]),
                    "--scene-id",
                    slot["scene_id"],
                    "--seed",
                    str(slot["seed"]),
                    "--resolution",
                    *[str(value) for value in resolution],
                    "--samples",
                    str(render_samples),
                    "--duration",
                    str(duration_s),
                    "--fps",
                    str(output_fps),
                ],
                root,
            )
            metadata_path = scene_dir / "metadata.json"
            metadata = load_json(metadata_path)
            if metadata["semantics"]["profile"] != slot["profile"]:
                raise RuntimeError("billiards profile does not match the schedule")
            if (
                metadata["assets"]["support_asset_id"]
                != slot["support_asset_id"]
            ):
                raise RuntimeError("billiards support does not match the schedule")
            if (
                float(metadata["physics"]["duration_s"]) != duration_s
                or int(metadata["physics"]["output_fps"]) != output_fps
                or list(metadata["render"]["resolution"]) != resolution
                or int(metadata["render"]["samples"]) != render_samples
            ):
                raise RuntimeError("billiards production contract mismatch")
            record = {
                **slot,
                "pipeline": "billiards",
                "metadata_path": str(metadata_path.relative_to(root)),
                "metadata_sha256": sha256(metadata_path),
                "status": "simulated_accepted",
            }
            records.append(record)
            billiards_records.append(record)
            continue
        if slot["generator"] == "passive_pinball":
            run(
                [
                    sys.executable,
                    str(implementation_paths["passive_pinball_generator"]),
                    "--root",
                    str(root),
                    "--config",
                    str(dependency_paths["passive_pinball_backend"]),
                    "--output",
                    str(scene_dir),
                    "--profile",
                    slot["profile"],
                    "--scene-id",
                    slot["scene_id"],
                    "--seed",
                    str(slot["seed"]),
                    "--resolution",
                    *[str(value) for value in resolution],
                    "--samples",
                    str(render_samples),
                ],
                root,
            )
            metadata_path = scene_dir / "metadata.json"
            metadata = load_json(metadata_path)
            if metadata["semantics"]["profile"] != slot["profile"]:
                raise RuntimeError(
                    "passive-pinball profile does not match the schedule"
                )
            if (
                float(metadata["simulation"]["time"]["duration_s"]) != duration_s
                or int(metadata["simulation"]["time"]["output_fps"])
                != output_fps
                or list(metadata["render"]["resolution"]) != resolution
                or int(metadata["render"]["samples"]) != render_samples
            ):
                raise RuntimeError("passive-pinball production contract mismatch")
            record = {
                **slot,
                "pipeline": "passive_pinball",
                "metadata_path": str(metadata_path.relative_to(root)),
                "metadata_sha256": sha256(metadata_path),
                "status": "simulated_accepted",
            }
            records.append(record)
            passive_pinball_records.append(record)
            continue
        if slot["generator"] != "asset_proxy":
            raise ValueError(f"unknown scene generator: {slot['generator']}")
        asset_command = [
            sys.executable,
            str(root / "tools/sample_asset_proxy_scenes.py"),
            "--root",
            str(root),
            "--registry",
            str(dependency_paths["asset_proxy_registry"]),
            "--catalog",
            str(dependency_paths["physical_proxy_catalog"]),
            "--semantic-rules",
            str(dependency_paths["asset_semantic_scene_rules"]),
            "--composition-rules",
            str(dependency_paths["asset_scene_composition"]),
            "--visual-rules",
            str(dependency_paths["visual_sampling"]),
            "--output",
            str(scene_dir),
            "--count",
            "1",
            "--seed",
            str(slot["seed"]),
            "--support-id",
            slot["support_asset_id"],
            "--dynamic-id",
            slot["dynamic_asset_id"],
            "--profiles",
            slot["profile"],
            "--scene-id-prefix",
            slot["scene_id"],
            "--duration",
            str(duration_s),
            "--fps",
            str(output_fps),
            "--resolution",
            *[str(value) for value in resolution],
            "--samples",
            str(render_samples),
        ]
        if slot["static_prop_asset_id"]:
            asset_command.extend(
                ["--static-prop-id", slot["static_prop_asset_id"]]
            )
        else:
            asset_command.append("--no-static-props")
        run(asset_command, root)
        child_manifest = load_json(scene_dir / "manifest.json")
        if (
            int(child_manifest["sample_count"]) != 1
            or int(child_manifest["passed_count"]) != 1
        ):
            raise RuntimeError("asset branch did not return one accepted sample")
        child = child_manifest["records"][0]
        metadata_path = root / child["metadata_path"]
        metadata = load_json(metadata_path)
        expected_assets = {
            "dynamic_asset_id": slot["dynamic_asset_id"],
            "support_asset_id": slot["support_asset_id"],
            "static_prop_asset_id": slot["static_prop_asset_id"],
        }
        if metadata["assets"] != expected_assets:
            raise RuntimeError("asset branch does not match scheduled assets")
        if metadata["physics"]["motion_profile"] != slot["profile"]:
            raise RuntimeError("asset branch profile does not match the schedule")
        if (
            float(metadata["physics"]["duration_s"]) != duration_s
            or int(metadata["physics"]["output_fps"]) != output_fps
            or list(metadata["render"]["resolution"]) != resolution
            or int(metadata["render"]["samples"]) != render_samples
        ):
            raise RuntimeError("asset branch production contract mismatch")
        record = {
            **slot,
            "pipeline": "asset_proxy",
            "child_scene_id": child["scene_id"],
            "metadata_path": child["metadata_path"],
            "metadata_sha256": sha256(metadata_path),
            "status": "simulated_accepted",
        }
        records.append(record)
        asset_records.append({**child, "matrix_scene_id": slot["scene_id"]})

    records.sort(key=lambda record: int(record["index"]))
    if asset_records:
        write_json(
            dataset_root / "asset_proxy_manifest.json",
            {
                "schema_version": "physweep_decoupled_asset_branch_manifest_v1",
                "dataset_id": f"{args.output_dataset}_asset_branches",
                "seed": args.seed,
                "output_root": str(dataset_root),
                "sample_count": len(asset_records),
                "passed_count": len(asset_records),
                "records": asset_records,
            },
        )
    manifest = {
        "schema_version": "physweep_one_object_decoupled_manifest_v4",
        "dataset_id": args.output_dataset,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "sample_count": len(records),
        "production_spec": {
            "duration_s": duration_s,
            "output_fps": output_fps,
            "frame_count": expected_frame_count,
            "resolution": resolution,
            "samples": render_samples,
            "source": {
                "path": str(dependency_paths["production_video"].relative_to(root)),
                "sha256": sha256(dependency_paths["production_video"]),
            },
        },
        "sampling_matrix": {
            "path": str(matrix_path.relative_to(root)),
            "sha256": sha256(matrix_path),
            "version": matrix["version"],
        },
        "dependencies": {
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
        "motion_counts": dict(Counter(record["motion_intent"] for record in records)),
        "environment_counts": dict(
            Counter(record["environment_id"] for record in records)
        ),
        "profile_counts": dict(Counter(record["profile"] for record in records)),
        "generic_manifest_path": (
            f"datasets/{args.output_dataset}/generic_matrix/manifest.json"
            if generic_slots
            else None
        ),
        "generic_simulation_manifest_path": (
            f"datasets/{args.output_dataset}/generic_matrix/simulation_manifest.json"
            if generic_slots
            else None
        ),
        "asset_proxy_manifest_path": (
            str((dataset_root / "asset_proxy_manifest.json").relative_to(root))
            if asset_records
            else None
        ),
        "billiards_metadata_paths": [
            record["metadata_path"] for record in billiards_records
        ],
        "passive_pinball_metadata_paths": [
            record["metadata_path"] for record in passive_pinball_records
        ],
        "records": records,
    }
    write_json(dataset_root / "manifest.json", manifest)
    print(dataset_root / "manifest.json")
    print(
        json.dumps(
            {
                "motions": manifest["motion_counts"],
                "environments": manifest["environment_counts"],
                "profiles": manifest["profile_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
