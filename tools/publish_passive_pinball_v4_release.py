#!/usr/bin/env python3
"""Publish v4 by replacing 32 complete v3 generic groups with passive pinball."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from publish_sweep_release import (
        load_json,
        resolve,
        root_relative,
        sha256,
        validate_groups,
        validate_source_artifacts,
        write_json,
    )
    from sample_pybullet_base import manifest_counts as generic_manifest_counts
except ModuleNotFoundError:
    from tools.publish_sweep_release import (
        load_json,
        resolve,
        root_relative,
        sha256,
        validate_groups,
        validate_source_artifacts,
        write_json,
    )
    from tools.sample_pybullet_base import (
        manifest_counts as generic_manifest_counts,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_PIPELINES = Counter(
    {"generic_pybullet": 2240, "asset_proxy": 928, "billiards": 32}
)
EXPECTED_V4_PIPELINES = Counter(
    {
        "generic_pybullet": 2208,
        "asset_proxy": 928,
        "billiards": 32,
        "passive_pinball": 32,
    }
)


def project_path(root: Path, value: str | Path) -> Path:
    return resolve(root, Path(value))


def verified_manifest(
    root: Path, path_value: str | Path, expected_hash: str | None = None
) -> tuple[Path, dict[str, Any]]:
    path = project_path(root, path_value)
    path.relative_to(root)
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_hash is not None and sha256(path) != expected_hash:
        raise ValueError(f"manifest hash mismatch: {path}")
    manifest = load_json(path)
    records = manifest.get("records")
    if records is not None and int(manifest.get("sample_count", len(records))) != len(
        records
    ):
        raise ValueError(f"manifest count is inconsistent: {path}")
    return path, manifest


def replacement_index(
    source_base: dict[str, Any], replacement: dict[str, Any]
) -> tuple[dict[int, dict[str, Any]], set[str], set[str]]:
    source_by_index = {
        int(record["index"]): record for record in source_base["records"]
    }
    if len(source_by_index) != len(source_base["records"]):
        raise ValueError("source base manifest contains duplicate indices")
    records = replacement["records"]
    if int(replacement["sample_count"]) != len(records) or len(records) != 32:
        raise ValueError("v4 requires exactly 32 replacement records")
    replacements: dict[int, dict[str, Any]] = {}
    old_parents: set[str] = set()
    new_parents: set[str] = set()
    for record in records:
        index = int(record["index"])
        original = source_by_index.get(index)
        if original is None or index in replacements:
            raise ValueError(f"invalid or duplicate replacement index: {index}")
        if (
            original.get("pipeline") != "generic_pybullet"
            or original.get("motion_intent") != "drop_fall_1obj"
            or record.get("pipeline") != "passive_pinball"
            or record.get("motion_intent") != original.get("motion_intent")
            or record.get("replaces_scene_id") != original.get("scene_id")
            or record.get("replaces_metadata_path") != original.get("metadata_path")
            or record.get("replaces_metadata_sha256")
            != original.get("metadata_sha256")
            or record.get("replaces_pipeline") != original.get("pipeline")
        ):
            raise ValueError(f"replacement changes its v4-preserved slot: {index}")
        replacements[index] = record
        old_parents.add(str(original["metadata_path"]))
        new_parents.add(str(record["metadata_path"]))
    if len(old_parents) != 32 or len(new_parents) != 32:
        raise ValueError("replacement parent metadata paths are not unique")
    return replacements, old_parents, new_parents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--replacement-manifest", type=Path, required=True)
    parser.add_argument("--pinball-metadata-manifest", type=Path, required=True)
    parser.add_argument("--pinball-physics-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_release_path = project_path(root, args.source_release)
    source_release_path.relative_to(root / "datasets")
    source_release = load_json(source_release_path)
    if source_release.get("schema_version") != "physweep_one_object_sweep_release_v1":
        raise ValueError("v4 publisher requires the published v3 release")
    source_base_path, source_base = verified_manifest(
        root,
        source_release["base_manifest"],
        str(source_release["base_manifest_sha256"]),
    )
    source_metadata_path, source_metadata = verified_manifest(
        root,
        source_release["metadata_manifest"],
        str(source_release["metadata_manifest_sha256"]),
    )
    source_physics_path, source_physics = verified_manifest(
        root,
        source_release["physics_manifest"],
        str(source_release["physics_manifest_sha256"]),
    )
    if (
        int(source_release["base_group_count"]) != 3200
        or int(source_release["sample_count"]) != 41600
        or int(source_base["sample_count"]) != 3200
        or Counter(record["pipeline"] for record in source_base["records"])
        != EXPECTED_SOURCE_PIPELINES
    ):
        raise ValueError("source v3 release does not have the frozen 3200-group layout")

    replacement_path, replacement = verified_manifest(
        root, args.replacement_manifest
    )
    if replacement.get("schema_version") != (
        "physweep_passive_pinball_v4_replacement_manifest_v1"
    ):
        raise ValueError("unsupported passive-pinball replacement manifest")
    if (
        project_path(root, replacement["source_release"]) != source_release_path
        or str(replacement["source_release_sha256"]) != sha256(source_release_path)
        or project_path(root, replacement["source_base_manifest"])
        != source_base_path
        or str(replacement["source_base_manifest_sha256"])
        != sha256(source_base_path)
    ):
        raise ValueError("replacement source provenance differs from v3 release")
    replacements, old_parents, new_parents = replacement_index(
        source_base, replacement
    )
    for record in replacements.values():
        metadata_path = project_path(root, record["metadata_path"])
        if sha256(metadata_path) != str(record["metadata_sha256"]):
            raise ValueError(f"replacement metadata hash mismatch: {metadata_path}")
        metadata = load_json(metadata_path)
        if (
            metadata.get("schema_version") != "physweep_passive_pinball_scene_v1"
            or metadata.get("scene_id") != record.get("scene_id")
        ):
            raise ValueError(f"replacement metadata identity mismatch: {metadata_path}")

    pinball_metadata_path, pinball_metadata = verified_manifest(
        root, args.pinball_metadata_manifest
    )
    pinball_physics_path, pinball_physics = verified_manifest(
        root, args.pinball_physics_manifest
    )
    pinball_records = list(pinball_metadata["records"])
    pinball_physics_records = list(pinball_physics["records"])
    if (
        len(pinball_records) != 416
        or len(pinball_physics_records) != 416
        or validate_groups(pinball_records) != 32
        or {str(record["parent"]) for record in pinball_records} != new_parents
        or any(
            record.get("source_schema_version")
            != "physweep_passive_pinball_scene_v1"
            for record in pinball_records
        )
        or any(
            not record.get("ok")
            or not record.get("audit_passed")
            or record.get("failed_checks")
            for record in pinball_physics_records
        )
    ):
        raise ValueError("pinball sweep inputs are not 32 complete accepted groups")
    validate_source_artifacts(root, pinball_records, pinball_physics_records)

    retained_metadata = [
        record
        for record in source_metadata["records"]
        if str(record["parent"]) not in old_parents
    ]
    retained_ids = {str(record["scene_id"]) for record in retained_metadata}
    retained_physics = [
        record
        for record in source_physics["records"]
        if str(record["scene_id"]) in retained_ids
    ]
    metadata_records = retained_metadata + pinball_records
    physics_records = retained_physics + pinball_physics_records
    if (
        len(metadata_records) != 41600
        or len(physics_records) != 41600
        or validate_groups(metadata_records) != 3200
        or {str(record["scene_id"]) for record in metadata_records}
        != {str(record["scene_id"]) for record in physics_records}
    ):
        raise ValueError("v4 sweep merge does not preserve 3200 complete groups")

    merged_base_records = []
    for original in source_base["records"]:
        merged_base_records.append(
            copy.deepcopy(replacements.get(int(original["index"]), original))
        )
    merged_base_records.sort(key=lambda record: int(record["index"]))
    pipeline_counts = Counter(record["pipeline"] for record in merged_base_records)
    if pipeline_counts != EXPECTED_V4_PIPELINES:
        raise ValueError(f"v4 base pipeline distribution is wrong: {pipeline_counts}")

    source_generic_path, source_generic = verified_manifest(
        root, source_base["generic_manifest_path"]
    )
    generic_samples = [
        sample
        for sample in source_generic["samples"]
        if str(sample["metadata_path"]) not in old_parents
    ]
    if len(generic_samples) != EXPECTED_V4_PIPELINES["generic_pybullet"]:
        raise ValueError("v4 generic child manifest did not remove exactly 32 samples")
    for sample in generic_samples:
        path = project_path(root, sample["metadata_path"])
        if sha256(path) != str(sample["metadata_sha256"]):
            raise ValueError(f"retained generic metadata hash mismatch: {path}")

    source_asset_path, source_asset = verified_manifest(
        root, source_base["asset_proxy_manifest_path"]
    )
    output_dir = project_path(root, args.output_dir)
    output_dir.relative_to(root / "datasets")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    )
    publish_dir = Path(temporary_output.name)

    def published_relative(path: Path) -> str:
        return root_relative(root, output_dir / path.relative_to(publish_dir))

    generic_output = publish_dir / "generic_manifest.json"
    generic_release = {
        **source_generic,
        "dataset_id": "physweep_one_object_v4_generic",
        "sample_count": len(generic_samples),
        "samples": generic_samples,
        "coverage": generic_manifest_counts(
            [
                load_json(project_path(root, sample["metadata_path"]))
                for sample in generic_samples
            ]
        ),
        "v4_source": {
            "path": root_relative(root, source_generic_path),
            "sha256": sha256(source_generic_path),
            "removed_group_count": 32,
        },
    }
    write_json(generic_output, generic_release)

    asset_output = publish_dir / "asset_proxy_manifest.json"
    write_json(
        asset_output,
        {
            **source_asset,
            "dataset_id": "physweep_one_object_v4_asset_proxy",
            "v4_source": {
                "path": root_relative(root, source_asset_path),
                "sha256": sha256(source_asset_path),
                "record_policy": "byte_equivalent_source_records",
            },
        },
    )

    matrix_path = root / "configs/one_object_sampling_matrix.json"
    merged_base = {
        **source_base,
        "schema_version": "physweep_one_object_decoupled_manifest_v4",
        "dataset_id": "physweep_one_object_v4",
        "sample_count": 3200,
        "motion_counts": dict(Counter(r["motion_intent"] for r in merged_base_records)),
        "environment_counts": dict(
            Counter(r["environment_id"] for r in merged_base_records)
        ),
        "profile_counts": dict(Counter(r["profile"] for r in merged_base_records)),
        "generic_manifest_path": published_relative(generic_output),
        "asset_proxy_manifest_path": published_relative(asset_output),
        "passive_pinball_metadata_paths": [
            str(record["metadata_path"])
            for record in replacement["records"]
        ],
        "records": merged_base_records,
        "v4_extension": {
            "mode": "deterministic_whole_group_replacement",
            "source_base_manifest": root_relative(root, source_base_path),
            "source_base_manifest_sha256": sha256(source_base_path),
            "replacement_manifest": root_relative(root, replacement_path),
            "replacement_manifest_sha256": sha256(replacement_path),
            "sampling_matrix": {
                "path": root_relative(root, matrix_path),
                "sha256": sha256(matrix_path),
                "version": load_json(matrix_path)["version"],
            },
            "pipeline_counts": dict(pipeline_counts),
        },
    }
    base_output = publish_dir / "base_manifest.json"
    write_json(base_output, merged_base)

    sources = [
        {
            "kind": "v3_retained_groups",
            "release": root_relative(root, source_release_path),
            "release_sha256": sha256(source_release_path),
            "metadata_manifest": root_relative(root, source_metadata_path),
            "metadata_manifest_sha256": sha256(source_metadata_path),
            "physics_manifest": root_relative(root, source_physics_path),
            "physics_manifest_sha256": sha256(source_physics_path),
            "group_count": 3168,
            "sample_count": 41184,
        },
        {
            "kind": "passive_pinball_replacement_groups",
            "metadata_manifest": root_relative(root, pinball_metadata_path),
            "metadata_manifest_sha256": sha256(pinball_metadata_path),
            "physics_manifest": root_relative(root, pinball_physics_path),
            "physics_manifest_sha256": sha256(pinball_physics_path),
            "group_count": 32,
            "sample_count": 416,
        },
    ]
    metadata_records.sort(key=lambda record: (str(record["parent"]), str(record["scene_id"])))
    physics_records.sort(key=lambda record: str(record["scene_id"]))
    metadata_output = publish_dir / "metadata_manifest.json"
    physics_output = publish_dir / "physics_manifest.json"
    write_json(
        metadata_output,
        {
            "schema_version": "physweep_release_metadata_manifest_v2",
            "dataset_id": "physweep_one_object_v4",
            "sample_count": 41600,
            "group_count": 3200,
            "group_size": 13,
            "sources": sources,
            "records": metadata_records,
        },
    )
    write_json(
        physics_output,
        {
            "schema_version": "physweep_release_physics_manifest_v2",
            "dataset_id": "physweep_one_object_v4",
            "sample_count": 41600,
            "passed_count": 41600,
            "rejected_count": 0,
            "error_count": 0,
            "pass_rate": 1.0,
            "group_count": 3200,
            "group_size": 13,
            "sources": sources,
            "records": physics_records,
        },
    )
    release = {
        "schema_version": "physweep_one_object_sweep_release_v2",
        "dataset_id": "physweep_one_object_v4",
        "base_group_count": 3200,
        "samples_per_group": 13,
        "sample_count": 41600,
        "base_count": 3200,
        "derived_count": 38400,
        "axes": ["mass_kg", "contact_friction", "contact_restitution"],
        "levels_per_axis": 5,
        "pipeline_group_counts": dict(pipeline_counts),
        "base_manifest": published_relative(base_output),
        "base_manifest_sha256": sha256(base_output),
        "metadata_manifest": published_relative(metadata_output),
        "metadata_manifest_sha256": sha256(metadata_output),
        "physics_manifest": published_relative(physics_output),
        "physics_manifest_sha256": sha256(physics_output),
        "source_release": root_relative(root, source_release_path),
        "source_release_sha256": sha256(source_release_path),
        "replacement_manifest": root_relative(root, replacement_path),
        "replacement_manifest_sha256": sha256(replacement_path),
        "validation": {
            "unique_scene_ids": True,
            "exact_group_size": True,
            "canonical_base_per_group": 1,
            "derived_records_per_axis": 4,
            "all_physics_records_passed": True,
            "whole_group_replacement": True,
            "source_v3_immutable": True,
        },
    }
    write_json(publish_dir / "manifest.json", release)
    publish_dir.replace(output_dir)
    temporary_output.cleanup()
    print(json.dumps({"release": root_relative(root, output_dir / "manifest.json"), "groups": 3200, "samples": 41600, "pipeline_group_counts": dict(pipeline_counts)}, indent=2))


if __name__ == "__main__":
    main()
