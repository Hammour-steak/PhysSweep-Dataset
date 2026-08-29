#!/usr/bin/env python3
"""Select and build the declared two-object coverage matrix from released 1obj metadata."""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json, write_json
from tools.sampling.sample_two_object_base import (
    DATASET_ID,
    DEFAULT_MATRIX,
    _validated_intents,
    build_two_object_scene,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_IDENTITY_RELATIONS = {"same_visual_profile", "different_visual_profile"}


def _rank(seed: int, *parts: object) -> str:
    value = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _axis_counts(cells: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    fields = {
        "motion": "motion_id",
        "ordered_scale_pair": "scale_pair_id",
        "visual_identity_relation": "visual_identity_relation",
        "scene_class": "scene_class",
    }
    return {
        label: dict(sorted(Counter(str(cell[field]) for cell in cells).items()))
        for label, field in fields.items()
    }


def _balanced_cell_order(
    cells: Sequence[dict[str, Any]], seed: int
) -> list[dict[str, Any]]:
    """Order a Cartesian matrix so every prefix balances all declared axes."""

    remaining = [copy.deepcopy(cell) for cell in cells]
    axis_fields = (
        "motion_id",
        "scale_pair_id",
        "visual_identity_relation",
        "scene_class",
    )
    levels = {
        field: sorted({str(cell[field]) for cell in remaining})
        for field in axis_fields
    }
    counts = {field: Counter() for field in axis_fields}
    ordered = []
    while remaining:
        def score(cell: dict[str, Any]) -> tuple[int, int, str]:
            ranges = []
            for field in axis_fields:
                values = [counts[field][level] for level in levels[field]]
                values[levels[field].index(str(cell[field]))] += 1
                ranges.append(max(values) - min(values))
            return (
                sum(ranges),
                max(ranges),
                _rank(seed, "coverage-cell", cell["cell_id"]),
            )

        selected = min(remaining, key=score)
        remaining.remove(selected)
        ordered.append(selected)
        for field in axis_fields:
            counts[field][str(selected[field])] += 1
    return ordered


def coverage_cells(
    matrix: dict[str, Any], limit: int | None = None
) -> tuple[list[dict[str, Any]], int]:
    """Return a balanced prefix of the complete declared Cartesian matrix."""

    intents = _validated_intents(matrix)
    coverage = matrix["coverage_plan"]
    scene_classes = matrix["scene_compatibility"]["allowed_scene_classes"]
    cells = []
    for intent, scale_pair, relation, scene_class, replicate_index in product(
        intents,
        coverage["role_ordered_scale_pairs"],
        coverage["visual_identity_relations"],
        scene_classes,
        range(int(coverage["replicates_per_cell"])),
    ):
        cell_id = "__".join(
            [
                str(intent["id"]),
                str(scale_pair["id"]),
                str(relation),
                str(scene_class),
                f"r{replicate_index:02d}",
            ]
        )
        cells.append(
            {
                "cell_id": cell_id,
                "motion_id": str(intent["id"]),
                "scale_pair_id": str(scale_pair["id"]),
                "object_a_scale_bin": str(scale_pair["object_a"]),
                "object_b_scale_bin": str(scale_pair["object_b"]),
                "visual_identity_relation": str(relation),
                "scene_class": str(scene_class),
                "replicate_index": replicate_index,
            }
        )
    full_count = len(cells)
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= full_count
    ):
        raise ValueError("two-object coverage limit is outside the full matrix")
    ordered = _balanced_cell_order(cells, int(coverage["seed"]))
    return ordered if limit is None else ordered[:limit], full_count


def _resolved_within(root: Path, value: Path) -> Path:
    resolved = (value if value.is_absolute() else root / value).resolve()
    resolved.relative_to(root)
    return resolved


def _declared_within(root: Path, value: Path) -> Path:
    """Keep a declared relative path lexical while allowing reviewed symlinks."""

    candidate = value if value.is_absolute() else root / value
    absolute = Path(os.path.abspath(candidate))
    absolute.relative_to(root)
    return absolute


def _source_reference(
    source_root: Path,
    record: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scene_id": str(metadata["scene_id"]),
        "path": _declared_within(source_root, Path(str(record["path"])))
        .relative_to(source_root)
        .as_posix(),
        "sha256": str(record["metadata_sha256"]),
    }


def released_source_pool(
    *,
    root: Path,
    released_base_manifest_path: Path,
    source_root: Path,
    source_manifest_path: Path,
    matrix: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load eligible objects and hosts from the generation manifest named by 1obj."""

    _validated_intents(matrix)
    source_contract = matrix["candidate_pool"]["source_release"]
    eligibility = matrix["candidate_pool"]["object_eligibility"]
    host_eligibility = matrix["candidate_pool"]["host_eligibility"]
    released_path = _resolved_within(root, released_base_manifest_path)
    generation_path = _declared_within(source_root, source_manifest_path)
    released = read_json(released_path)
    generation = read_json(generation_path)
    if released.get("schema_version") != source_contract[
        "released_base_manifest_schema_version"
    ]:
        raise ValueError("released 1obj base manifest has the wrong schema")
    provenance = released.get("provenance", {}).get(
        "source_generation_release_metadata"
    )
    if not isinstance(provenance, dict) or (
        provenance.get("schema_version")
        != source_contract["generation_manifest_schema_version"]
        or provenance.get("manifest_sha256") != sha256(generation_path)
    ):
        raise ValueError("released 1obj base does not name this generation manifest")
    records = generation.get("records")
    if (
        generation.get("schema_version")
        != source_contract["generation_manifest_schema_version"]
        or generation.get("dataset_id") != released.get("dataset_id")
        or not isinstance(records, list)
        or int(generation.get("sample_count", -1)) != len(records)
        or int(generation.get("group_count", -1))
        != int(released.get("sample_count", -2))
    ):
        raise ValueError("generation manifest contradicts the released 1obj base")

    base_records = [
        record
        for record in records
        if str(record.get("kind")) == source_contract["sample_kind"]
    ]
    base_ids = [str(record.get("scene_id", "")) for record in base_records]
    if (
        len(base_records) != int(generation["group_count"])
        or any(not value for value in base_ids)
        or len(base_ids) != len(set(base_ids))
    ):
        raise ValueError("generation manifest has invalid canonical base records")

    objects = []
    hosts = []
    rejected_host_role_count = 0
    rejected_bounded_camera_host_count = 0
    allowed_scenes = set(matrix["scene_compatibility"]["allowed_scene_classes"])
    expected_schema = source_contract["generation_metadata_schema_version"]
    allowed_scale_bins = set(eligibility["scale_bins"])
    for record in base_records:
        if record.get("source_schema_version") != expected_schema:
            continue
        metadata_path = _declared_within(
            source_root, Path(str(record["path"]))
        )
        if sha256(metadata_path) != str(record["metadata_sha256"]):
            raise ValueError(
                f"source metadata changed after release: {record['scene_id']}"
            )
        metadata = read_json(metadata_path)
        if (
            metadata.get("schema_version") != expected_schema
            or str(metadata.get("scene_id", "")) != str(record["scene_id"])
        ):
            raise ValueError(
                f"source metadata identity is invalid: {record['scene_id']}"
            )
        simulation_objects = metadata.get("simulation", {}).get("objects")
        if not isinstance(simulation_objects, list) or len(simulation_objects) != 1:
            raise ValueError("eligible 1obj metadata must contain one object")
        obj = simulation_objects[0]
        support = metadata["simulation"]["support"]
        scene_class = str(support.get("scene_class", ""))
        if scene_class in allowed_scenes:
            roles = {
                str(collider.get("role", ""))
                for collider in support.get("colliders", [])
                if isinstance(collider, dict)
            }
            required_roles = set(host_eligibility["required_collider_roles"])
            allowed_roles = set(host_eligibility["allowed_collider_roles"])
            if support.get("support_shape") == host_eligibility["support_shape"]:
                motion_neutral = required_roles.issubset(roles) and roles.issubset(
                    allowed_roles
                )
                camera_unbounded = support.get("camera_envelope") is None
                if motion_neutral and camera_unbounded:
                    scene_visual = metadata["appearance"]["scene_visual"]
                    visual_id = str(scene_visual.get("id", ""))
                    visual_type = str(scene_visual.get("visual_type", ""))
                    if not visual_id or not visual_type:
                        raise ValueError(
                            "eligible two-object host lacks visual identity"
                        )
                    hosts.append(
                        {
                            "metadata": metadata,
                            "source": _source_reference(
                                source_root, record, metadata
                            ),
                            "scene_class": scene_class,
                            "visual_profile_id": visual_id,
                            "visual_type": visual_type,
                        }
                    )
                elif not motion_neutral:
                    rejected_host_role_count += 1
                else:
                    rejected_bounded_camera_host_count += 1
        geometry = obj.get("geometry", {})
        if (
            obj.get("body_model") != eligibility["body_model"]
            or geometry.get("type") != eligibility["geometry_type"]
        ):
            continue
        size = np.asarray(geometry.get("size_m"), dtype=np.float64)
        if (
            size.shape != (3,)
            or not np.isfinite(size).all()
            or bool(np.any(size <= 0.0))
            or (
                eligibility["require_isotropic_geometry"]
                and not np.allclose(size, size[0], atol=1.0e-8, rtol=0.0)
            )
        ):
            raise ValueError("eligible sphere geometry must be finite and isotropic")
        foreground = metadata.get("semantic_sampling", {}).get(
            "five_dimensions", {}
        ).get("foreground_object", {})
        scale_bin = str(foreground.get("scale_bin", ""))
        visual_profile_id = str(obj.get("visual_profile", {}).get("id", ""))
        if scale_bin not in allowed_scale_bins or not visual_profile_id:
            raise ValueError("eligible sphere lacks scale or visual identity")
        objects.append(
            {
                "metadata": metadata,
                "source": _source_reference(source_root, record, metadata),
                "scale_bin": scale_bin,
                "visual_profile_id": visual_profile_id,
            }
        )
    if not objects or not hosts:
        raise ValueError("released 1obj metadata yields no eligible 2obj sources")
    audit = {
        "released_base_count": len(base_records),
        "eligible_object_count": len(objects),
        "eligible_host_count": len(hosts),
        "rejected_motion_specific_host_count": rejected_host_role_count,
        "rejected_bounded_camera_host_count": (
            rejected_bounded_camera_host_count
        ),
        "object_scale_bin_counts": dict(
            sorted(Counter(record["scale_bin"] for record in objects).items())
        ),
        "object_visual_profile_count": len(
            {record["visual_profile_id"] for record in objects}
        ),
        "host_scene_class_counts": dict(
            sorted(Counter(record["scene_class"] for record in hosts).items())
        ),
        "host_visual_profile_count": len(
            {record["visual_profile_id"] for record in hosts}
        ),
    }
    return objects, hosts, audit


def select_coverage_sources(
    cells: Sequence[dict[str, Any]],
    objects: Sequence[dict[str, Any]],
    hosts: Sequence[dict[str, Any]],
    matrix: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign sources without replacement while balancing every visual profile."""

    plan = matrix["coverage_plan"]
    seed = int(plan["seed"])
    maximum_reuse = int(
        plan["selection_policy"]["maximum_object_source_reuse"]
    )
    objects_by_scale: dict[str, list[dict[str, Any]]] = {}
    for record in objects:
        objects_by_scale.setdefault(str(record["scale_bin"]), []).append(record)
    hosts_by_scene: dict[str, list[dict[str, Any]]] = {}
    for record in hosts:
        hosts_by_scene.setdefault(str(record["scene_class"]), []).append(record)

    object_use: Counter[str] = Counter()
    object_profile_use: Counter[str] = Counter()
    host_profile_use: Counter[str] = Counter()
    host_type_use: Counter[str] = Counter()
    used_pairs: set[tuple[str, str]] = set()
    used_hosts: set[str] = set()
    selected = []
    for cell in cells:
        left_pool = objects_by_scale.get(str(cell["object_a_scale_bin"]), [])
        right_pool = objects_by_scale.get(str(cell["object_b_scale_bin"]), [])
        relation = str(cell["visual_identity_relation"])
        if relation not in _IDENTITY_RELATIONS:
            raise ValueError(f"unsupported visual identity relation: {relation}")

        def object_key(record: dict[str, Any], role: str) -> tuple[int, int, str]:
            source_id = str(record["source"]["scene_id"])
            profile_id = str(record["visual_profile_id"])
            return (
                object_use[source_id],
                object_profile_use[profile_id],
                _rank(seed, cell["cell_id"], role, source_id),
            )

        pair = None
        for left in sorted(left_pool, key=lambda value: object_key(value, "a")):
            left_id = str(left["source"]["scene_id"])
            if object_use[left_id] >= maximum_reuse:
                continue
            for right in sorted(
                right_pool, key=lambda value: object_key(value, "b")
            ):
                right_id = str(right["source"]["scene_id"])
                if left_id == right_id or object_use[right_id] >= maximum_reuse:
                    continue
                same_profile = (
                    left["visual_profile_id"] == right["visual_profile_id"]
                )
                if same_profile != (relation == "same_visual_profile"):
                    continue
                unordered_pair = tuple(sorted((left_id, right_id)))
                if unordered_pair in used_pairs:
                    continue
                pair = (left, right, unordered_pair)
                break
            if pair is not None:
                break
        if pair is None:
            raise ValueError(f"cannot satisfy two-object source cell: {cell['cell_id']}")
        left, right, unordered_pair = pair
        excluded_host_ids = {
            str(left["source"]["scene_id"]),
            str(right["source"]["scene_id"]),
        }
        eligible_hosts = [
            record
            for record in hosts_by_scene.get(str(cell["scene_class"]), [])
            if str(record["source"]["scene_id"]) not in used_hosts
            and str(record["source"]["scene_id"]) not in excluded_host_ids
        ]
        if not eligible_hosts:
            raise ValueError(f"cannot satisfy two-object host cell: {cell['cell_id']}")
        host = min(
            eligible_hosts,
            key=lambda record: (
                host_profile_use[str(record["visual_profile_id"])],
                host_type_use[str(record["visual_type"])],
                _rank(
                    seed,
                    cell["cell_id"],
                    "host",
                    record["source"]["scene_id"],
                ),
            ),
        )
        used_pairs.add(unordered_pair)
        used_hosts.add(str(host["source"]["scene_id"]))
        for record in (left, right):
            object_use[str(record["source"]["scene_id"])] += 1
            object_profile_use[str(record["visual_profile_id"])] += 1
        host_profile_use[str(host["visual_profile_id"])] += 1
        host_type_use[str(host["visual_type"])] += 1
        selected.append(
            {
                "cell": copy.deepcopy(cell),
                "host": host,
                "objects": [left, right],
            }
        )

    eligible_object_profiles = {str(value["visual_profile_id"]) for value in objects}
    eligible_host_profiles = {str(value["visual_profile_id"]) for value in hosts}
    if set(object_profile_use) != eligible_object_profiles:
        raise ValueError("coverage selection misses eligible object visual profiles")
    if set(host_profile_use) != eligible_host_profiles:
        raise ValueError("coverage selection misses eligible host visual profiles")
    audit = {
        "unique_object_source_count": len(object_use),
        "maximum_object_source_reuse": max(object_use.values()),
        "unique_source_pair_count": len(used_pairs),
        "unique_host_count": len(used_hosts),
        "selected_object_visual_profile_count": len(object_profile_use),
        "selected_host_visual_profile_count": len(host_profile_use),
        "selected_host_visual_type_counts": dict(sorted(host_type_use.items())),
    }
    return selected, audit


def build_coverage_scenes(
    selections: Sequence[dict[str, Any]], matrix: dict[str, Any]
) -> list[dict[str, Any]]:
    """Compile selected sources and attach the minimal per-scene coverage fact."""

    scenes = []
    role_ids = [str(role["object_id"]) for role in matrix["object_pair"]["roles"]]
    for sample_index, selection in enumerate(selections, start=1):
        cell = selection["cell"]
        host = copy.deepcopy(selection["host"]["metadata"])
        stub = "_".join(
            [
                f"physweep2scene_{sample_index:06d}",
                str(cell["scale_pair_id"]),
                str(cell["visual_identity_relation"]),
                str(cell["scene_class"]),
            ]
        )
        host["scene_id"] = stub
        scene = build_two_object_scene(
            host,
            matrix,
            str(cell["motion_id"]),
            [record["metadata"] for record in selection["objects"]],
            sample_index=sample_index,
        )
        scene["two_object_sampling"] = {
            "schema_version": "physweep_two_object_coverage_cell_v1",
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
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_root = args.source_root.resolve()
    released_path = _resolved_within(root, args.released_base_manifest)
    source_path = _declared_within(source_root, args.source_manifest)
    matrix_path = _resolved_within(root, args.matrix)
    output_dir = _resolved_within(root, args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("two-object coverage output directory must be empty")
    matrix = read_json(matrix_path)
    cells, full_cell_count = coverage_cells(matrix, args.limit)
    objects, hosts, source_audit = released_source_pool(
        root=root,
        released_base_manifest_path=released_path,
        source_root=source_root,
        source_manifest_path=source_path,
        matrix=matrix,
    )
    selections, selection_audit = select_coverage_sources(
        cells, objects, hosts, matrix
    )
    scenes = build_coverage_scenes(selections, matrix)
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
            "schema_version": "physweep_two_object_coverage_selection_v1",
            "full_cell_count": full_cell_count,
            "selected_cell_count": len(cells),
            "complete_cartesian_product": len(cells) == full_cell_count,
            "axis_counts": _axis_counts(cells),
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
