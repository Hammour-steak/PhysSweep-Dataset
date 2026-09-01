#!/usr/bin/env python3
"""Replace failed 2obj camera selections without changing coverage cells."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from tools.core.hashing import sha256_file
from tools.core.json_io import read_json, write_json
from tools.sampling.sample_two_object_coverage import (
    _axis_counts,
    _dynamics_profiles_eligible,
    _pair_layout_fits_host,
    _pair_sources_meet_rule_camera_geometry,
    _rank,
    _source_camera_plane_extent,
    _source_dynamics_profile,
    _source_meets_rule_camera_extent,
    _validate_complete_scene_coverage,
    build_coverage_scenes,
)
from tools.sampling.two_object_sources import released_source_pool
from tools.scene_rules.two_object import (
    allowed_camera_view_families,
    load_two_object_scene_rules,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else root / path).resolve()
    resolved.relative_to(root)
    return resolved


def _source_id(record: dict[str, Any]) -> str:
    return str(record["source"]["scene_id"])


def selection_signature(
    selection: dict[str, Any],
) -> tuple[str, str, str, str]:
    objects = selection["objects"]
    return (
        str(selection["cell"]["cell_id"]),
        _source_id(selection["host"]),
        _source_id(objects[0]),
        _source_id(objects[1]),
    )


def _replace_cell_camera_family(
    cell: dict[str, Any], camera_family_id: str
) -> dict[str, Any]:
    result = copy.deepcopy(cell)
    previous = str(result["camera_view_family_id"])
    parts = str(result["cell_id"]).split("__")
    matching = [index for index, value in enumerate(parts) if value == previous]
    if len(matching) != 1:
        raise ValueError("two-object cell id has no unique camera-family token")
    parts[matching[0]] = str(camera_family_id)
    result["camera_view_family_id"] = str(camera_family_id)
    result["cell_id"] = "__".join(parts)
    return result


def camera_failure_mode(error: str) -> str:
    """Classify a strict solver rejection for generic replacement ranking."""

    if "visually overlaps" in error:
        return "pair_overlap"
    if "per-object visibility" in error:
        return "per_object_visibility"
    return "generic"


def _replacement_camera_priority(
    source: dict[str, Any],
    rule: dict[str, Any],
    cell: dict[str, Any],
    failure_mode: str,
) -> tuple[float, float]:
    extent = _source_camera_plane_extent(source, rule, cell)
    if extent == float("inf"):
        return 0.0, 0.0
    if failure_mode == "per_object_visibility":
        geometry = source["metadata"]["simulation"]["objects"][0][
            "geometry"
        ]
        robust_extent = sorted(map(float, geometry["size_m"]))[1]
        return -robust_extent, -extent
    if failure_mode == "pair_overlap":
        return extent, 0.0
    return 0.0, 0.0


def replace_failed_selections(
    selections: Sequence[dict[str, Any]],
    failed_scene_ids: set[str],
    objects: Sequence[dict[str, Any]],
    hosts: Sequence[dict[str, Any]],
    matrix: dict[str, Any],
    scene_rules: dict[str, Any],
    *,
    attempt: int,
    failure_modes: dict[str, str] | None = None,
    previously_rejected: set[tuple[str, str, str, str]] = frozenset(),
) -> list[dict[str, Any]]:
    """Freeze passing rows and deterministically replace rejected assignments."""

    if not failed_scene_ids:
        raise ValueError("two-object replacement requires at least one failure")
    if attempt < 1:
        raise ValueError("two-object replacement attempt must be positive")
    by_scene_id = {str(value["scene_id"]): value for value in selections}
    if len(by_scene_id) != len(selections):
        raise ValueError("two-object base selections contain duplicate scene ids")
    unknown = failed_scene_ids.difference(by_scene_id)
    if unknown:
        raise ValueError(f"camera failures are absent from the base: {sorted(unknown)}")
    failure_modes = failure_modes or {}
    if not set(failure_modes).issubset(failed_scene_ids):
        raise ValueError("camera failure modes contain a passing scene")

    plan = matrix["coverage_plan"]
    seed = int(plan["seed"])
    maximum_object_reuse = int(
        plan["selection_policy"]["maximum_object_source_reuse"]
    )
    maximum_host_reuse = int(
        plan["selection_policy"]["maximum_host_source_reuse"]
    )
    rules_by_id = {
        str(rule["id"]): rule for rule in scene_rules["physical_rules"]
    }
    source_pairs = list(plan["role_ordered_source_family_pairs"])
    source_pair_ids = [str(record["id"]) for record in source_pairs]
    if len(source_pair_ids) != len(set(source_pair_ids)):
        raise ValueError("two-object source-family pair ids repeat")
    intents = {
        str(record["id"]): record for record in matrix["motion_intents"]
    }
    object_profiles = {_source_id(value): _source_dynamics_profile(value) for value in objects}

    retained = [
        value
        for value in selections
        if str(value["scene_id"]) not in failed_scene_ids
    ]
    object_use: Counter[str] = Counter(
        _source_id(obj) for value in retained for obj in value["objects"]
    )
    object_profile_use: Counter[str] = Counter(
        str(obj["visual_profile_id"])
        for value in retained
        for obj in value["objects"]
    )
    host_use: Counter[str] = Counter(_source_id(value["host"]) for value in retained)
    host_profile_use: Counter[str] = Counter(
        str(value["host"]["visual_profile_id"]) for value in retained
    )
    host_type_use: Counter[str] = Counter(
        str(value["host"]["visual_type"]) for value in retained
    )
    source_pair_use: Counter[str] = Counter(
        str(value["cell"]["source_family_pair_id"]) for value in retained
    )
    used_pairs = {
        tuple(sorted(_source_id(obj) for obj in value["objects"]))
        for value in retained
    }
    rejected = set(previously_rejected)
    rejected.update(
        selection_signature(by_scene_id[scene_id]) for scene_id in failed_scene_ids
    )
    rejected_object_pairs = {
        (cell_id, left_id, right_id)
        for cell_id, _host_id, left_id, right_id in rejected
    }

    replacements: dict[str, dict[str, Any]] = {}
    for original in selections:
        scene_id = str(original["scene_id"])
        if scene_id not in failed_scene_ids:
            continue
        original_cell = original["cell"]
        rule = rules_by_id[str(original_cell["scene_rule_id"])]
        motion = intents[str(original_cell["motion_id"])]
        camera_families = allowed_camera_view_families(
            rule, str(original_cell["motion_id"])
        )
        current_camera_family = str(
            original_cell["camera_view_family_id"]
        )
        if current_camera_family not in camera_families:
            raise ValueError("two-object cell camera family is no longer allowed")
        if attempt > 1 and len(camera_families) > 1:
            next_index = (
                camera_families.index(current_camera_family) + 1
            ) % len(camera_families)
            cell = _replace_cell_camera_family(
                original_cell, camera_families[next_index]
            )
        else:
            cell = copy.deepcopy(original_cell)
        cell_id = str(cell["cell_id"])
        failure_mode = failure_modes.get(scene_id, "generic")

        candidate_hosts = [
            value
            for value in hosts
            if str(value["scene_class"]) == str(cell["scene_class"])
            and str(value["scene_rule_id"]) == str(cell["scene_rule_id"])
            and str(value["environment_category"])
            == str(cell["visual_environment_category"])
            and host_use[_source_id(value)] < maximum_host_reuse
        ]
        candidate_hosts.sort(
            key=lambda value: (
                host_use[_source_id(value)],
                host_profile_use[str(value["visual_profile_id"])],
                host_type_use[str(value["visual_type"])],
                _rank(
                    seed,
                    "replacement",
                    attempt,
                    cell_id,
                    "host",
                    _source_id(value),
                ),
            )
        )
        selected = None
        original_source_pair_id = str(cell["source_family_pair_id"])
        ordered_source_pairs = sorted(
            source_pairs,
            key=lambda record: (
                attempt > 1 and str(record["id"]) == original_source_pair_id,
                attempt == 1 and str(record["id"]) != original_source_pair_id,
                source_pair_use[str(record["id"])],
                _rank(
                    seed,
                    "replacement-source-family",
                    attempt,
                    cell_id,
                    str(record["id"]),
                ),
            ),
        )
        for source_pair in ordered_source_pairs:
            object_pools = []
            for role, source_family in (
                ("object_a", str(source_pair["object_a"])),
                ("object_b", str(source_pair["object_b"])),
            ):
                pool = [
                    value
                    for value in objects
                    if str(value["source_family"]) == source_family
                    and str(value["shape_family_id"])
                    == str(cell[f"{role}_shape"])
                    and str(value["scale_bin"])
                    == str(cell[f"{role}_scale_bin"])
                    and object_use[_source_id(value)] < maximum_object_reuse
                    and _source_meets_rule_camera_extent(value, rule, cell)
                ]
                pool.sort(
                    key=lambda value: (
                        _replacement_camera_priority(
                            value, rule, cell, failure_mode
                        ),
                        object_use[_source_id(value)],
                        object_profile_use[str(value["visual_profile_id"])],
                        _rank(
                            seed,
                            "replacement",
                            attempt,
                            cell_id,
                            role,
                            _source_id(value),
                        ),
                    )
                )
                if not pool:
                    object_pools = []
                    break
                object_pools.append(pool)
            if len(object_pools) != 2:
                continue

            for allow_rejected_object_pair in (False, True):
                for host in candidate_hosts:
                    host_id = _source_id(host)
                    for left in object_pools[0]:
                        left_id = _source_id(left)
                        if left_id == host_id:
                            continue
                        for right in object_pools[1]:
                            right_id = _source_id(right)
                            pair = tuple(sorted((left_id, right_id)))
                            signature = (cell_id, host_id, left_id, right_id)
                            object_pair = (cell_id, left_id, right_id)
                            if (
                                left_id == right_id
                                or right_id == host_id
                                or pair in used_pairs
                                or signature in rejected
                                or (
                                    not allow_rejected_object_pair
                                    and object_pair in rejected_object_pairs
                                )
                                or not _dynamics_profiles_eligible(
                                    (
                                        object_profiles[left_id],
                                        object_profiles[right_id],
                                    ),
                                    matrix,
                                    cell,
                                )
                                or not _pair_layout_fits_host(
                                    host, left, right, matrix, cell
                                )
                                or not _pair_sources_meet_rule_camera_geometry(
                                    host, left, right, matrix, cell, rule
                                )
                            ):
                                continue
                            selected_cell = copy.deepcopy(cell)
                            selected_cell.update(
                                {
                                    "source_family_pair_id": str(
                                        source_pair["id"]
                                    ),
                                    "object_a_source_family": str(
                                        source_pair["object_a"]
                                    ),
                                    "object_b_source_family": str(
                                        source_pair["object_b"]
                                    ),
                                }
                            )
                            selected = {
                                "scene_id": scene_id,
                                "cell": selected_cell,
                                "host": host,
                                "objects": [left, right],
                            }
                            used_pairs.add(pair)
                            source_pair_use[str(source_pair["id"])] += 1
                            host_use[host_id] += 1
                            host_profile_use[str(host["visual_profile_id"])] += 1
                            host_type_use[str(host["visual_type"])] += 1
                            for obj in (left, right):
                                object_use[_source_id(obj)] += 1
                                object_profile_use[
                                    str(obj["visual_profile_id"])
                                ] += 1
                            break
                        if selected is not None:
                            break
                    if selected is not None:
                        break
                if selected is not None:
                    break
            if selected is not None:
                break
        if selected is None:
            raise RuntimeError(f"no deterministic replacement for {cell_id}")
        replacements[scene_id] = selected

    result = [
        replacements.get(str(value["scene_id"]), value) for value in selections
    ]
    if max(object_use.values()) > maximum_object_reuse:
        raise AssertionError("replacement exceeded object-source reuse")
    if max(host_use.values()) > maximum_host_reuse:
        raise AssertionError("replacement exceeded host-source reuse")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--camera-failure-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    base_path = _resolve(root, args.base_manifest)
    failure_path = _resolve(root, args.camera_failure_manifest)
    output_dir = _resolve(root, args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("two-object replacement output must be empty")

    base = read_json(base_path)
    failures = read_json(failure_path)
    failure_records = failures.get("failures")
    if (
        base.get("schema_version") != "physweep_pybullet_base_manifest_v1"
        or not isinstance(base.get("samples"), list)
        or int(base.get("sample_count", -1)) != len(base["samples"])
        or failures.get("schema_version")
        != "physweep_pybullet_binding_failure_manifest_v1"
        or not isinstance(failure_records, list)
        or int(failures.get("failed_count", -1)) != len(failure_records)
    ):
        raise ValueError("two-object replacement input manifest is invalid")

    matrix_path = _resolve(root, base["matrix"]["path"])
    rules_path = _resolve(root, base["scene_rules"]["path"])
    if (
        sha256_file(matrix_path) != str(base["matrix"]["sha256"])
        or sha256_file(rules_path) != str(base["scene_rules"]["sha256"])
    ):
        raise ValueError("two-object replacement rules changed after sampling")
    matrix = read_json(matrix_path)
    scene_rules = load_two_object_scene_rules(rules_path)

    source_release = base["source_release"]
    source_root = Path(source_release["source_project_root"]).resolve()
    released_path = _resolve(
        root, source_release["released_base_manifest"]["path"]
    )
    source_manifest_path = (
        source_root / source_release["generation_manifest"]["path"]
    ).resolve()
    if (
        sha256_file(released_path)
        != str(source_release["released_base_manifest"]["sha256"])
        or sha256_file(source_manifest_path)
        != str(source_release["generation_manifest"]["sha256"])
    ):
        raise ValueError("two-object replacement source release changed")
    objects, hosts = released_source_pool(
        root=root,
        released_base_manifest_path=released_path,
        source_root=source_root,
        source_manifest_path=source_manifest_path,
        matrix=matrix,
        scene_rules=scene_rules,
    )
    objects_by_id = {_source_id(value): value for value in objects}
    hosts_by_id = {_source_id(value): value for value in hosts}

    prior_selections = []
    prior_metadata = {}
    for sample in base["samples"]:
        metadata_path = _resolve(root, sample["metadata_path"])
        if sha256_file(metadata_path) != str(sample["metadata_sha256"]):
            raise ValueError("two-object base metadata changed after sampling")
        metadata = read_json(metadata_path)
        scene_id = str(sample["scene_id"])
        if str(metadata.get("scene_id")) != scene_id:
            raise ValueError("two-object base sample identity is invalid")
        sampling = metadata["two_object_sampling"]
        source = sampling["sources"]
        object_ids = [str(value["scene_id"]) for value in source["objects"]]
        host_id = str(source["host"]["scene_id"])
        if host_id not in hosts_by_id or any(
            value not in objects_by_id for value in object_ids
        ):
            raise ValueError("two-object base source is no longer eligible")
        prior_selections.append(
            {
                "scene_id": scene_id,
                "cell": copy.deepcopy(sampling["cell"]),
                "host": hosts_by_id[host_id],
                "objects": [objects_by_id[value] for value in object_ids],
            }
        )
        prior_metadata[scene_id] = metadata

    failed_scene_ids = {str(value["scene_id"]) for value in failure_records}
    if len(failed_scene_ids) != len(failure_records):
        raise ValueError("two-object camera failure manifest repeats a scene")
    failure_modes = {
        str(value["scene_id"]): camera_failure_mode(str(value.get("error", "")))
        for value in failure_records
    }
    prior_replacement = base.get("replacement", {})
    attempt = int(prior_replacement.get("attempt", 0)) + 1
    previous_rejected = {
        (
            str(value["cell_id"]),
            str(value["host_scene_id"]),
            str(value["object_scene_ids"][0]),
            str(value["object_scene_ids"][1]),
        )
        for value in prior_replacement.get("rejected_assignments", [])
    }
    selections = replace_failed_selections(
        prior_selections,
        failed_scene_ids,
        objects,
        hosts,
        matrix,
        scene_rules,
        attempt=attempt,
        failure_modes=failure_modes,
        previously_rejected=previous_rejected,
    )
    _validate_complete_scene_coverage(matrix, scene_rules, selections)
    scenes = build_coverage_scenes(selections, matrix, scene_rules)
    samples = []
    for scene in scenes:
        scene_id = str(scene["scene_id"])
        if scene_id not in failed_scene_ids and scene != prior_metadata[scene_id]:
            raise AssertionError(f"passing two-object scene changed: {scene_id}")
        metadata_path = output_dir / "scenes" / scene_id / "metadata.json"
        write_json(metadata_path, scene)
        samples.append(
            {
                "scene_id": scene_id,
                "metadata_path": metadata_path.relative_to(root).as_posix(),
                "metadata_sha256": sha256_file(metadata_path),
                "coverage_cell_id": scene["two_object_sampling"]["cell"][
                    "cell_id"
                ],
            }
        )

    rejected_assignments = [
        {
            "cell_id": signature[0],
            "host_scene_id": signature[1],
            "object_scene_ids": [signature[2], signature[3]],
        }
        for signature in sorted(
            previous_rejected
            | {
                selection_signature(value)
                for value in prior_selections
                if str(value["scene_id"]) in failed_scene_ids
            }
        )
    ]
    selected_cells = [value["cell"] for value in selections]
    manifest = {
        "schema_version": "physweep_pybullet_base_manifest_v1",
        "dataset_id": str(base["dataset_id"]),
        "sample_count": len(samples),
        "matrix": copy.deepcopy(base["matrix"]),
        "scene_rules": copy.deepcopy(base["scene_rules"]),
        "source_release": copy.deepcopy(source_release),
        "coverage": {
            "schema_version": "physweep_two_object_coverage_selection_v7",
            "full_cell_count": int(base["coverage"]["full_cell_count"]),
            "selected_cell_count": len(samples),
            "complete_cartesian_product": True,
            "axis_counts": _axis_counts(selected_cells),
        },
        "replacement": {
            "attempt": attempt,
            "selection_policy": "camera_failure_mode_extent_v5",
            "source_base_manifest": base_path.relative_to(root).as_posix(),
            "source_base_manifest_sha256": sha256_file(base_path),
            "camera_failure_manifest": failure_path.relative_to(root).as_posix(),
            "camera_failure_manifest_sha256": sha256_file(failure_path),
            "replaced_count": len(failed_scene_ids),
            "rejected_assignments": rejected_assignments,
        },
        "samples": samples,
        "status": "sampled_pending_simulation",
    }
    write_json(output_dir / "manifest.json", manifest)
    print(output_dir / "manifest.json")


if __name__ == "__main__":
    main()
