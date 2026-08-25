#!/usr/bin/env python3
"""Publish audited sweep shards as one group-complete release manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sample_pybullet_base import manifest_counts as generic_manifest_counts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AXES = ("mass_kg", "contact_friction", "contact_restitution")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_groups(records: list[dict[str, Any]]) -> int:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["parent"])].append(record)
    for parent, group in groups.items():
        if len(group) != 13:
            raise ValueError(f"group does not contain 13 records: {parent}")
        if sum(record["kind"] == "base" for record in group) != 1:
            raise ValueError(f"group does not contain one canonical base: {parent}")
        axis_counts = Counter(
            record.get("axis") for record in group if record["kind"] == "sweep"
        )
        if axis_counts != Counter({axis: 4 for axis in AXES}):
            raise ValueError(f"group has an invalid axis layout: {parent}")
        derived = [record for record in group if record["kind"] == "sweep"]
        for axis in AXES:
            levels = {
                int(record["level_index"])
                for record in derived
                if record["axis"] == axis
            }
            if levels != {0, 1, 3, 4}:
                raise ValueError(f"group has invalid sweep levels: {parent}/{axis}")
        targets = {
            (str(record["target_object_id"]), int(record["target_object_index"]))
            for record in derived
        }
        if len(targets) != 1 or next(iter(targets))[1] != 0:
            raise ValueError(f"group does not target one object: {parent}")
    return len(groups)


def validate_source_artifacts(
    root: Path,
    metadata_records: list[dict[str, Any]],
    physics_records: list[dict[str, Any]],
) -> None:
    physics_by_id = {str(record["scene_id"]): record for record in physics_records}
    if len(physics_by_id) != len(physics_records):
        raise ValueError("physics manifest contains duplicate scene ids")
    for metadata in metadata_records:
        scene_id = str(metadata["scene_id"])
        physics = physics_by_id.get(scene_id)
        if physics is None:
            raise ValueError(f"physics record is missing: {scene_id}")
        metadata_path = resolve(root, Path(metadata["path"]))
        metadata_path.relative_to(root)
        metadata_hash = sha256(metadata_path)
        if (
            resolve(root, Path(physics["metadata_path"])) != metadata_path
            or str(metadata["metadata_sha256"]) != metadata_hash
            or str(physics["metadata_sha256"]) != metadata_hash
        ):
            raise ValueError(f"metadata provenance mismatch: {scene_id}")
        if str(metadata["source_schema_version"]) != str(
            physics["source_schema_version"]
        ):
            raise ValueError(f"source schema mismatch: {scene_id}")
        for path_key, hash_key in (
            ("resolved_scene_path", "resolved_scene_sha256"),
            ("trajectory_path", "trajectory_sha256"),
            ("audit_path", "audit_sha256"),
        ):
            artifact = resolve(root, Path(physics[path_key]))
            artifact.relative_to(root)
            if sha256(artifact) != str(physics[hash_key]):
                raise ValueError(f"physics artifact hash mismatch: {scene_id}/{path_key}")


def merge_base_records(
    base_records: list[dict[str, Any]],
    replacement_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = {int(record["index"]): record for record in base_records}
    if len(merged) != len(base_records):
        raise ValueError("base manifest contains duplicate indices")
    replacement_indices = [int(record["index"]) for record in replacement_records]
    if len(replacement_indices) != len(set(replacement_indices)):
        raise ValueError("replacement manifest contains duplicate indices")
    for replacement in replacement_records:
        index = int(replacement["index"])
        if index not in merged:
            raise ValueError(f"replacement refers to an unknown base index: {index}")
        if replacement["replaces_metadata_path"] != merged[index]["metadata_path"]:
            raise ValueError(f"replacement provenance does not match base index: {index}")
        merged[index] = replacement
    return [merged[index] for index in sorted(merged)]


def select_source_records(
    metadata: dict[str, Any],
    physics: dict[str, Any],
    replaced_parents: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted_physics = [
        record
        for record in physics["records"]
        if record.get("ok") and record.get("audit_passed")
    ]
    accepted_ids = {str(record["scene_id"]) for record in accepted_physics}
    selected_metadata = [
        record
        for record in metadata["records"]
        if str(record["scene_id"]) in accepted_ids
        and str(record["parent"]) not in replaced_parents
    ]
    selected_ids = {str(record["scene_id"]) for record in selected_metadata}
    selected_physics = [
        record
        for record in accepted_physics
        if str(record["scene_id"]) in selected_ids
    ]
    if len(selected_metadata) != len(selected_ids):
        raise ValueError("source metadata contains duplicate accepted scene ids")
    if len(selected_metadata) != len(selected_physics):
        raise ValueError("accepted physics records do not match metadata")
    return selected_metadata, selected_physics


def merge_generic_samples(
    source_samples: list[dict[str, Any]],
    replacement_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = {str(sample["metadata_path"]): sample for sample in source_samples}
    if len(merged) != len(source_samples):
        raise ValueError("generic manifest contains duplicate metadata paths")
    for replacement in replacement_records:
        replaced_path = str(replacement["replaces_metadata_path"])
        if replaced_path not in merged:
            raise ValueError(
                "generic replacement provenance does not match the source manifest: "
                f"{replaced_path}"
            )
        del merged[replaced_path]
        sample = {
            "scene_id": str(replacement["candidate_scene_id"]),
            "metadata_path": str(replacement["metadata_path"]),
            "metadata_sha256": str(replacement["metadata_sha256"]),
            "simulation_record_path": str(replacement["simulation_record_path"]),
            "trajectory_path": str(replacement["trajectory_path"]),
        }
        if sample["metadata_path"] in merged:
            raise ValueError("generic replacement metadata path is already present")
        merged[sample["metadata_path"]] = sample
    samples = sorted(merged.values(), key=lambda sample: str(sample["scene_id"]))
    scene_ids = [str(sample["scene_id"]) for sample in samples]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("generic release contains duplicate scene ids")
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--metadata-manifest", type=Path, action="append", required=True)
    parser.add_argument("--physics-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path)
    parser.add_argument("--replacement-base-manifest", type=Path)
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> None:
    args = parse_args()
    if len(args.metadata_manifest) != len(args.physics_manifest):
        raise ValueError("metadata and physics manifests must be paired")
    if (args.base_manifest is None) != (args.replacement_base_manifest is None):
        raise ValueError(
            "--base-manifest and --replacement-base-manifest must be paired"
        )
    root = args.root.resolve()
    replacement_path = None
    replacement = None
    replaced_parents: set[str] = set()
    if args.replacement_base_manifest:
        replacement_path = resolve(root, args.replacement_base_manifest)
        replacement = load_json(replacement_path)
        replaced_parents = {
            str(record["replaces_metadata_path"])
            for record in replacement["records"]
        }
    output_dir = resolve(root, args.output_dir)
    datasets_root = (root / "datasets").resolve()
    if datasets_root not in output_dir.parents:
        raise ValueError("release output must remain under datasets")
    if output_dir.exists():
        raise FileExistsError(f"release output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    metadata_records: list[dict[str, Any]] = []
    physics_records: list[dict[str, Any]] = []
    sources = []
    for metadata_arg, physics_arg in zip(
        args.metadata_manifest, args.physics_manifest
    ):
        metadata_path = resolve(root, metadata_arg)
        physics_path = resolve(root, physics_arg)
        metadata = load_json(metadata_path)
        physics = load_json(physics_path)
        if any(
            not record.get("ok") or not record["audit_passed"]
            for record in physics["records"]
        ):
            raise ValueError("physics manifest contains a failed record")
        selected_metadata, selected_physics = select_source_records(
            metadata, physics, replaced_parents
        )
        validate_source_artifacts(root, selected_metadata, selected_physics)
        metadata_records.extend(selected_metadata)
        physics_records.extend(selected_physics)
        sources.append(
            {
                "metadata_manifest": str(metadata_path.relative_to(root)),
                "metadata_manifest_sha256": sha256(metadata_path),
                "physics_manifest": str(physics_path.relative_to(root)),
                "physics_manifest_sha256": sha256(physics_path),
                "sample_count": len(selected_metadata),
            }
        )

    metadata_ids = [str(record["scene_id"]) for record in metadata_records]
    physics_ids = [str(record["scene_id"]) for record in physics_records]
    if len(metadata_ids) != len(set(metadata_ids)):
        raise ValueError("release contains duplicate metadata scene ids")
    if len(physics_ids) != len(set(physics_ids)):
        raise ValueError("release contains duplicate physics scene ids")
    if set(metadata_ids) != set(physics_ids):
        raise ValueError("release metadata and physics scene ids differ")
    group_count = validate_groups(metadata_records)
    sample_count = len(metadata_records)
    if sample_count != group_count * 13:
        raise ValueError("release sample count does not match group count")

    metadata_records.sort(key=lambda record: (record["parent"], record["scene_id"]))
    physics_records.sort(key=lambda record: record["scene_id"])
    metadata_output = output_dir / "metadata_manifest.json"
    physics_output = output_dir / "physics_manifest.json"
    write_json(
        metadata_output,
        {
            "schema_version": "physweep_release_metadata_manifest_v1",
            "dataset_id": output_dir.parent.name,
            "sample_count": sample_count,
            "group_count": group_count,
            "group_size": 13,
            "sources": sources,
            "records": metadata_records,
        },
    )
    write_json(
        physics_output,
        {
            "schema_version": "physweep_release_physics_manifest_v1",
            "dataset_id": output_dir.parent.name,
            "sample_count": sample_count,
            "passed_count": sample_count,
            "rejected_count": 0,
            "error_count": 0,
            "pass_rate": 1.0,
            "group_count": group_count,
            "group_size": 13,
            "sources": sources,
            "records": physics_records,
        },
    )
    release = {
        "schema_version": "physweep_one_object_sweep_release_v1",
        "dataset_id": output_dir.parent.name,
        "base_group_count": group_count,
        "samples_per_group": 13,
        "sample_count": sample_count,
        "base_count": group_count,
        "derived_count": sample_count - group_count,
        "axes": list(AXES),
        "levels_per_axis": 5,
        "metadata_manifest": str(metadata_output.relative_to(root)),
        "metadata_manifest_sha256": sha256(metadata_output),
        "physics_manifest": str(physics_output.relative_to(root)),
        "physics_manifest_sha256": sha256(physics_output),
        "validation": {
            "unique_scene_ids": True,
            "exact_group_size": True,
            "canonical_base_per_group": 1,
            "derived_records_per_axis": 4,
            "all_physics_records_passed": True,
        },
    }
    if replacement_path is not None and replacement is not None:
        release.update(
            {
                "replacement_base_manifest": str(replacement_path.relative_to(root)),
                "replacement_base_manifest_sha256": sha256(replacement_path),
                "replacement_group_count": int(replacement["sample_count"]),
            }
        )
        if args.base_manifest:
            base_path = resolve(root, args.base_manifest)
            base = load_json(base_path)
            if int(replacement.get("sample_count", -1)) != len(
                replacement["records"]
            ):
                raise ValueError("replacement sample count is inconsistent")
            source_base_path = resolve(
                root, Path(replacement["source_base_manifest"])
            )
            if (
                source_base_path != base_path
                or sha256(source_base_path)
                != str(replacement["source_base_manifest_sha256"])
            ):
                raise ValueError("replacement source base provenance does not match")
            base_by_index = {
                int(record["index"]): record for record in base["records"]
            }
            slot_contract = tuple(replacement["slot_contract"])
            for record in replacement["records"]:
                original = base_by_index.get(int(record["index"]))
                if original is None or any(
                    record.get(field) != original.get(field)
                    for field in slot_contract
                ):
                    raise ValueError("replacement changes its frozen base slot")
                metadata_path = resolve(root, Path(record["metadata_path"]))
                metadata_path.relative_to(root)
                if sha256(metadata_path) != str(record["metadata_sha256"]):
                    raise ValueError("replacement metadata hash mismatch")
            merged_base_records = merge_base_records(
                list(base["records"]), list(replacement["records"])
            )
            if len(merged_base_records) != group_count:
                raise ValueError("published base count does not match sweep groups")
            base_output = output_dir / "base_manifest.json"
            merged_base = {
                **base,
                "dataset_id": f"{base['dataset_id']}_release",
                "sample_count": len(merged_base_records),
                "motion_counts": dict(
                    Counter(record["motion_intent"] for record in merged_base_records)
                ),
                "environment_counts": dict(
                    Counter(record["environment_id"] for record in merged_base_records)
                ),
                "profile_counts": dict(
                    Counter(record["profile"] for record in merged_base_records)
                ),
                "records": merged_base_records,
                "release_sources": {
                    "base_manifest": str(base_path.relative_to(root)),
                    "base_manifest_sha256": sha256(base_path),
                    "replacement_base_manifest": str(
                        replacement_path.relative_to(root)
                    ),
                    "replacement_base_manifest_sha256": sha256(replacement_path),
                },
            }
            generic_replacements = [
                record
                for record in replacement["records"]
                if record["pipeline"] == "generic_pybullet"
            ]
            if generic_replacements:
                source_generic_manifest_path = resolve(
                    root, Path(base["generic_manifest_path"])
                )
                source_generic_manifest = load_json(source_generic_manifest_path)
                generic_samples = merge_generic_samples(
                    list(source_generic_manifest["samples"]), generic_replacements
                )
                for sample in generic_samples:
                    metadata_path = resolve(root, Path(sample["metadata_path"]))
                    if sha256(metadata_path) != str(sample["metadata_sha256"]):
                        raise ValueError(
                            f"generic release metadata hash mismatch: {metadata_path}"
                        )
                generic_output = output_dir / "generic_manifest.json"
                generic_release = {
                    **source_generic_manifest,
                    "dataset_id": f"{source_generic_manifest['dataset_id']}_release",
                    "sample_count": len(generic_samples),
                    "coverage": generic_manifest_counts(
                        [
                            load_json(resolve(root, Path(sample["metadata_path"])))
                            for sample in generic_samples
                        ]
                    ),
                    "samples": generic_samples,
                    "release_sources": {
                        "generic_manifest": str(
                            source_generic_manifest_path.relative_to(root)
                        ),
                        "generic_manifest_sha256": sha256(
                            source_generic_manifest_path
                        ),
                        "replacement_base_manifest": str(
                            replacement_path.relative_to(root)
                        ),
                        "replacement_base_manifest_sha256": sha256(replacement_path),
                    },
                }
                acceptance = dict(generic_release.get("acceptance", {}))
                acceptance.update(
                    {
                        "camera_replacement_count": len(generic_replacements),
                        "camera_replacement_manifest_path": str(
                            replacement_path.relative_to(root)
                        ),
                    }
                )
                generic_release["acceptance"] = acceptance
                write_json(generic_output, generic_release)
                merged_base["generic_manifest_path"] = str(
                    generic_output.relative_to(root)
                )
            source_asset_manifest_path = resolve(
                root, Path(base["asset_proxy_manifest_path"])
            )
            source_asset_manifest = load_json(source_asset_manifest_path)
            asset_replacements = [
                record
                for record in replacement["records"]
                if record["pipeline"] == "asset_proxy"
            ]
            replaced_outer_scene_ids = {
                str(record["replaces_scene_id"])
                for record in asset_replacements
            }
            asset_records = [
                record
                for record in source_asset_manifest["records"]
                if str(record["matrix_scene_id"]) not in replaced_outer_scene_ids
            ]
            for record in asset_replacements:
                metadata_path = resolve(root, Path(record["metadata_path"]))
                child_manifest = load_json(metadata_path.parents[2] / "manifest.json")
                if (
                    int(child_manifest["sample_count"]) != 1
                    or int(child_manifest["passed_count"]) != 1
                ):
                    raise ValueError("replacement asset child manifest is invalid")
                asset_records.append(
                    {
                        **child_manifest["records"][0],
                        "matrix_scene_id": record["scene_id"],
                    }
                )
            asset_records.sort(key=lambda record: record["matrix_scene_id"])
            asset_output = output_dir / "asset_proxy_manifest.json"
            write_json(
                asset_output,
                {
                    **source_asset_manifest,
                    "dataset_id": f"{source_asset_manifest['dataset_id']}_release",
                    "output_root": str(output_dir.relative_to(root)),
                    "sample_count": len(asset_records),
                    "passed_count": len(asset_records),
                    "records": asset_records,
                },
            )
            merged_base["asset_proxy_manifest_path"] = str(
                asset_output.relative_to(root)
            )
            write_json(base_output, merged_base)
            release.update(
                {
                    "base_manifest": str(base_output.relative_to(root)),
                    "base_manifest_sha256": sha256(base_output),
                }
            )
    write_json(output_dir / "manifest.json", release)
    print(f"release manifest: {output_dir / 'manifest.json'}")
    print(f"groups={group_count} samples={sample_count}")


if __name__ == "__main__":
    main()
