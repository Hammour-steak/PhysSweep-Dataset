#!/usr/bin/env python3
"""Publish a verified whole-group specialized extension without mutating its source."""

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
        validate_groups,
        validate_source_artifacts,
        write_json,
    )
    from specialized_backend_registry import specialized_by_pipeline
    from specialized_release_extension import (
        index_replacements,
        load_extension_spec,
        sha256,
    )
except ModuleNotFoundError:
    from tools.publish_sweep_release import (
        load_json,
        resolve,
        root_relative,
        validate_groups,
        validate_source_artifacts,
        write_json,
    )
    from tools.specialized_backend_registry import specialized_by_pipeline
    from tools.specialized_release_extension import (
        index_replacements,
        load_extension_spec,
        sha256,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generic_manifest_counts(metadata: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        from sample_pybullet_base import manifest_counts
    except ModuleNotFoundError:
        from tools.sample_pybullet_base import manifest_counts
    return manifest_counts(metadata)


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


def file_binding(root: Path, path: Path | str) -> dict[str, str]:
    resolved = project_path(root, path)
    resolved.relative_to(root)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": root_relative(root, resolved), "sha256": sha256(resolved)}


def specialized_renderer_binding(
    root: Path, replacement: dict[str, Any]
) -> dict[str, str]:
    pipeline = str(replacement["pipeline"])
    record = specialized_by_pipeline(root).get(pipeline)
    if record is None:
        raise ValueError(f"specialized registry lacks pipeline: {pipeline}")
    if str(record["source_schema_version"]) != str(
        replacement["scene_schema_version"]
    ):
        raise ValueError(f"specialized renderer schema mismatch: {pipeline}")
    return file_binding(root, record["renderer_script"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--replacement-manifest", type=Path, required=True)
    parser.add_argument("--specialized-metadata-manifest", type=Path, required=True)
    parser.add_argument("--specialized-physics-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_root = args.source_root.resolve()
    source_release_path = project_path(source_root, args.source_release)
    source_release_path.relative_to(source_root / "datasets")
    source_release = load_json(source_release_path)

    replacement_path, replacement_manifest = verified_manifest(
        root, args.replacement_manifest
    )
    if replacement_manifest.get("schema_version") != (
        "physweep_specialized_replacement_manifest_v1"
    ):
        raise ValueError("unsupported specialized replacement manifest")
    spec_binding = replacement_manifest["extension_spec"]
    spec_path = project_path(root, spec_binding["path"])
    if sha256(spec_path) != str(spec_binding["sha256"]):
        raise ValueError("specialized extension spec hash mismatch")
    spec = load_extension_spec(root, spec_path)
    if str(spec["extension_id"]) != str(spec_binding["extension_id"]):
        raise ValueError("specialized extension id mismatch")
    source_contract = spec["source_release"]
    target_contract = spec["target_release"]
    group = spec["group_contract"]
    replacement = spec["replacement"]
    expected_source_pipelines = Counter(source_contract["pipeline_group_counts"])
    expected_target_pipelines = Counter(target_contract["pipeline_group_counts"])
    group_count = int(group["base_group_count"])
    sample_count = int(group["sample_count"])
    replacement_count = int(group["replacement_group_count"])

    if (
        source_release.get("schema_version") != source_contract["schema_version"]
        or source_release.get("dataset_id") != source_contract["dataset_id"]
        or int(source_release["base_group_count"]) != group_count
        or int(source_release["sample_count"]) != sample_count
        or Counter(source_release["pipeline_group_counts"])
        != expected_source_pipelines
    ):
        raise ValueError("source release differs from the extension contract")
    source_base_path, source_base = verified_manifest(
        source_root,
        source_release["base_manifest"],
        str(source_release["base_manifest_sha256"]),
    )
    source_metadata_path, source_metadata = verified_manifest(
        source_root,
        source_release["metadata_manifest"],
        str(source_release["metadata_manifest_sha256"]),
    )
    source_physics_path, source_physics = verified_manifest(
        source_root,
        source_release["physics_manifest"],
        str(source_release["physics_manifest_sha256"]),
    )
    if (
        source_base.get("schema_version") != source_contract["base_manifest_schema"]
        or int(source_base["sample_count"]) != group_count
        or Counter(record["pipeline"] for record in source_base["records"])
        != expected_source_pipelines
        or int(source_metadata["sample_count"]) != sample_count
        or int(source_physics["sample_count"]) != sample_count
        or int(source_physics["passed_count"]) != sample_count
        or int(source_physics["rejected_count"]) != 0
        or int(source_physics["error_count"]) != 0
        or validate_groups(source_metadata["records"]) != group_count
        or any(
            not record.get("ok")
            or not record.get("audit_passed")
            or record.get("failed_checks")
            for record in source_physics["records"]
        )
    ):
        raise ValueError("source release does not contain complete accepted groups")
    if (
        project_path(source_root, replacement_manifest["source_release"])
        != source_release_path
        or str(replacement_manifest["source_release_sha256"])
        != sha256(source_release_path)
        or project_path(source_root, replacement_manifest["source_base_manifest"])
        != source_base_path
        or str(replacement_manifest["source_base_manifest_sha256"])
        != sha256(source_base_path)
        or Path(str(replacement_manifest["source_project_root"])).resolve()
        != source_root
    ):
        raise ValueError("replacement source provenance differs from the release")
    replacements, old_parents, new_parents = index_replacements(
        source_base, replacement_manifest, spec
    )
    for record in replacements.values():
        metadata_path = project_path(root, record["metadata_path"])
        if sha256(metadata_path) != str(record["metadata_sha256"]):
            raise ValueError(f"replacement metadata hash mismatch: {metadata_path}")
        metadata = load_json(metadata_path)
        if (
            metadata.get("schema_version") != replacement["scene_schema_version"]
            or metadata.get("scene_id") != record.get("scene_id")
            or metadata.get("semantics", {}).get("profile") != record.get("profile")
        ):
            raise ValueError(f"replacement metadata identity mismatch: {metadata_path}")

    specialized_metadata_path, specialized_metadata = verified_manifest(
        root, args.specialized_metadata_manifest
    )
    specialized_physics_path, specialized_physics = verified_manifest(
        root, args.specialized_physics_manifest
    )
    specialized_records = list(specialized_metadata["records"])
    specialized_physics_records = list(specialized_physics["records"])
    replacement_sample_count = replacement_count * int(group["samples_per_group"])
    if (
        len(specialized_records) != replacement_sample_count
        or len(specialized_physics_records) != replacement_sample_count
        or validate_groups(specialized_records) != replacement_count
        or {str(record["parent"]) for record in specialized_records} != new_parents
        or any(
            record.get("source_schema_version")
            != replacement["scene_schema_version"]
            for record in specialized_records
        )
        or any(
            not record.get("ok")
            or not record.get("audit_passed")
            or record.get("failed_checks")
            for record in specialized_physics_records
        )
    ):
        raise ValueError("specialized sweep inputs are not complete accepted groups")
    validate_source_artifacts(root, specialized_records, specialized_physics_records)

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
    metadata_records = retained_metadata + specialized_records
    physics_records = retained_physics + specialized_physics_records
    if (
        len(metadata_records) != sample_count
        or len(physics_records) != sample_count
        or validate_groups(metadata_records) != group_count
        or {str(record["scene_id"]) for record in metadata_records}
        != {str(record["scene_id"]) for record in physics_records}
    ):
        raise ValueError("extension merge does not preserve complete sweep groups")

    merged_base_records = [
        copy.deepcopy(replacements.get(int(original["index"]), original))
        for original in source_base["records"]
    ]
    merged_base_records.sort(key=lambda record: int(record["index"]))
    pipeline_counts = Counter(record["pipeline"] for record in merged_base_records)
    if pipeline_counts != expected_target_pipelines:
        raise ValueError(f"target base pipeline distribution is wrong: {pipeline_counts}")

    source_generic_path, source_generic = verified_manifest(
        source_root, source_base["generic_manifest_path"]
    )
    generic_samples = [
        sample
        for sample in source_generic["samples"]
        if str(sample["metadata_path"]) not in old_parents
    ]
    if len(generic_samples) != expected_target_pipelines["generic_pybullet"]:
        raise ValueError("generic child manifest removed the wrong number of samples")
    for sample in generic_samples:
        path = project_path(source_root, sample["metadata_path"])
        if sha256(path) != str(sample["metadata_sha256"]):
            raise ValueError(f"retained generic metadata hash mismatch: {path}")
    source_asset_path, source_asset = verified_manifest(
        source_root, source_base["asset_proxy_manifest_path"]
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
        "dataset_id": f"{target_contract['dataset_id']}_generic",
        "sample_count": len(generic_samples),
        "samples": generic_samples,
        "coverage": generic_manifest_counts(
            [
                load_json(project_path(source_root, sample["metadata_path"]))
                for sample in generic_samples
            ]
        ),
        "extension_source": {
            "source_project_root": str(source_root),
            "path": root_relative(source_root, source_generic_path),
            "sha256": sha256(source_generic_path),
            "removed_group_count": replacement_count,
        },
    }
    write_json(generic_output, generic_release)

    asset_output = publish_dir / "asset_proxy_manifest.json"
    write_json(
        asset_output,
        {
            **source_asset,
            "dataset_id": f"{target_contract['dataset_id']}_asset_proxy",
            "extension_source": {
                "source_project_root": str(source_root),
                "path": root_relative(source_root, source_asset_path),
                "sha256": sha256(source_asset_path),
                "record_policy": "byte_equivalent_source_records",
            },
        },
    )

    matrix_path = root / "configs/one_object_sampling_matrix.json"
    extension_bindings = {
        "extension_spec": file_binding(root, spec_path),
        "replacement_preparer": file_binding(
            root, "tools/prepare_specialized_release_replacements.py"
        ),
        "release_publisher": file_binding(root, Path(__file__)),
        "backend_config": file_binding(root, replacement["backend_config"]),
        "generator": file_binding(root, replacement["generator_script"]),
        "renderer": specialized_renderer_binding(root, replacement),
        "specialized_registry": file_binding(
            root, "configs/specialized_scene_backends.json"
        ),
        "physics_sweep": file_binding(root, "configs/physics_sweep.json"),
    }
    merged_base = {
        **source_base,
        "schema_version": target_contract["base_manifest_schema"],
        "dataset_id": target_contract["dataset_id"],
        "sample_count": group_count,
        "motion_counts": dict(
            Counter(record["motion_intent"] for record in merged_base_records)
        ),
        "environment_counts": dict(
            Counter(record["environment_id"] for record in merged_base_records)
        ),
        "profile_counts": dict(
            Counter(record["profile"] for record in merged_base_records)
        ),
        "generic_manifest_path": published_relative(generic_output),
        "asset_proxy_manifest_path": published_relative(asset_output),
        f"{replacement['pipeline']}_metadata_paths": [
            str(record["metadata_path"]) for record in replacement_manifest["records"]
        ],
        "records": merged_base_records,
        f"{spec['extension_id']}_extension": {
            "mode": "deterministic_whole_group_replacement",
            "source_project_root": str(source_root),
            "source_base_manifest": root_relative(source_root, source_base_path),
            "source_base_manifest_sha256": sha256(source_base_path),
            "replacement_manifest": root_relative(root, replacement_path),
            "replacement_manifest_sha256": sha256(replacement_path),
            "sampling_matrix": {
                "path": root_relative(root, matrix_path),
                "sha256": sha256(matrix_path),
                "version": load_json(matrix_path)["version"],
            },
            "bindings": extension_bindings,
            "pipeline_counts": dict(pipeline_counts),
        },
    }
    base_output = publish_dir / "base_manifest.json"
    write_json(base_output, merged_base)

    sources = [
        {
            "kind": "retained_source_groups",
            "source_project_root": str(source_root),
            "release": root_relative(source_root, source_release_path),
            "release_sha256": sha256(source_release_path),
            "metadata_manifest": root_relative(source_root, source_metadata_path),
            "metadata_manifest_sha256": sha256(source_metadata_path),
            "physics_manifest": root_relative(source_root, source_physics_path),
            "physics_manifest_sha256": sha256(source_physics_path),
            "group_count": group_count - replacement_count,
            "sample_count": sample_count - replacement_sample_count,
        },
        {
            "kind": f"{replacement['pipeline']}_replacement_groups",
            "source_project_root": str(root),
            "metadata_manifest": root_relative(root, specialized_metadata_path),
            "metadata_manifest_sha256": sha256(specialized_metadata_path),
            "physics_manifest": root_relative(root, specialized_physics_path),
            "physics_manifest_sha256": sha256(specialized_physics_path),
            "group_count": replacement_count,
            "sample_count": replacement_sample_count,
        },
    ]
    metadata_records.sort(
        key=lambda record: (str(record["parent"]), str(record["scene_id"]))
    )
    physics_records.sort(key=lambda record: str(record["scene_id"]))
    metadata_output = publish_dir / "metadata_manifest.json"
    physics_output = publish_dir / "physics_manifest.json"
    write_json(
        metadata_output,
        {
            "schema_version": "physweep_release_metadata_manifest_v2",
            "dataset_id": target_contract["dataset_id"],
            "sample_count": sample_count,
            "group_count": group_count,
            "group_size": int(group["samples_per_group"]),
            "sources": sources,
            "records": metadata_records,
        },
    )
    write_json(
        physics_output,
        {
            "schema_version": "physweep_release_physics_manifest_v2",
            "dataset_id": target_contract["dataset_id"],
            "sample_count": sample_count,
            "passed_count": sample_count,
            "rejected_count": 0,
            "error_count": 0,
            "pass_rate": 1.0,
            "group_count": group_count,
            "group_size": int(group["samples_per_group"]),
            "sources": sources,
            "records": physics_records,
        },
    )
    release = {
        "schema_version": target_contract["schema_version"],
        "dataset_id": target_contract["dataset_id"],
        "base_group_count": group_count,
        "samples_per_group": int(group["samples_per_group"]),
        "sample_count": sample_count,
        "base_count": group_count,
        "derived_count": sample_count - group_count,
        "axes": ["mass_kg", "contact_friction", "contact_restitution"],
        "levels_per_axis": 5,
        "pipeline_group_counts": dict(pipeline_counts),
        "base_manifest": published_relative(base_output),
        "base_manifest_sha256": sha256(base_output),
        "metadata_manifest": published_relative(metadata_output),
        "metadata_manifest_sha256": sha256(metadata_output),
        "physics_manifest": published_relative(physics_output),
        "physics_manifest_sha256": sha256(physics_output),
        "source_project_root": str(source_root),
        "source_release": root_relative(source_root, source_release_path),
        "source_release_sha256": sha256(source_release_path),
        "extension_spec": root_relative(root, spec_path),
        "extension_spec_sha256": sha256(spec_path),
        "replacement_manifest": root_relative(root, replacement_path),
        "replacement_manifest_sha256": sha256(replacement_path),
        "validation": {
            "unique_scene_ids": True,
            "exact_group_size": True,
            "canonical_base_per_group": 1,
            "derived_records_per_axis": 4,
            "all_physics_records_passed": True,
            "whole_group_replacement": True,
            "source_release_immutable": True,
        },
    }
    write_json(publish_dir / "manifest.json", release)
    publish_dir.replace(output_dir)
    temporary_output.cleanup()
    print(
        json.dumps(
            {
                "release": root_relative(root, output_dir / "manifest.json"),
                "groups": group_count,
                "samples": sample_count,
                "pipeline_group_counts": dict(pipeline_counts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
