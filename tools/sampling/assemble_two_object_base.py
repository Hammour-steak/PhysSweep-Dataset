#!/usr/bin/env python3
"""Merge generic and specialized 2obj samples into one hash-bound base manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tools.core.hashing import relative_file_binding, sha256_file
from tools.core.json_io import read_json, write_json_atomic_sorted
from tools.core.paths import resolve_project_path_within_root
from tools.dataset_contract.object_identity_contract import validate_object_identity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "physweep_two_object_base_manifest_v1"
GENERIC_MANIFEST_SCHEMA = "physweep_pybullet_base_manifest_v1"
SPECIALIZED_MANIFEST_SCHEMA = "physweep_two_object_specialized_base_manifest_v1"
SPECIALIZED_FAMILIES = {
    "billiards": "physweep_billiards_scene_v4",
    "passive_pinball": "physweep_passive_pinball_scene_v1",
    "marble_run": "physweep_marble_run_scene_v1",
}


def _samples(document: dict[str, Any], expected_schema: str) -> list[dict[str, Any]]:
    if document.get("schema_version") != expected_schema:
        raise ValueError(f"unsupported two-object source manifest: {document.get('schema_version')}")
    if document.get("dataset_id") != "physweep_two_object":
        raise ValueError("two-object source manifest has the wrong dataset id")
    samples = document.get("samples")
    if not isinstance(samples, list) or int(document.get("sample_count", -1)) != len(samples):
        raise ValueError("two-object source manifest count differs")
    return samples


def assemble_base_manifest(
    root: Path,
    generic_manifest_path: Path,
    specialized_manifest_path: Path,
) -> dict[str, Any]:
    """Validate and merge both sampling branches without copying metadata."""

    root = root.resolve()
    source_paths = {
        "generic": resolve_project_path_within_root(root, generic_manifest_path),
        "specialized": resolve_project_path_within_root(root, specialized_manifest_path),
    }
    source_documents = {name: read_json(path) for name, path in source_paths.items()}
    source_samples = {
        "generic": _samples(source_documents["generic"], GENERIC_MANIFEST_SCHEMA),
        "specialized": _samples(
            source_documents["specialized"], SPECIALIZED_MANIFEST_SCHEMA
        ),
    }
    records: list[dict[str, Any]] = []
    for source_name in ("generic", "specialized"):
        for sample in source_samples[source_name]:
            metadata_path = resolve_project_path_within_root(
                root, Path(str(sample.get("metadata_path", "")))
            )
            if not metadata_path.is_file():
                raise FileNotFoundError(metadata_path)
            digest = sha256_file(metadata_path)
            if digest != str(sample.get("metadata_sha256", "")):
                raise ValueError(f"source metadata hash differs: {metadata_path}")
            metadata = read_json(metadata_path)
            scene_id = str(sample.get("scene_id", ""))
            if not scene_id or str(metadata.get("scene_id", "")) != scene_id:
                raise ValueError(f"source scene id differs: {metadata_path}")
            identity = validate_object_identity(metadata)
            if int(identity["dynamic_object_count"]) != 2:
                raise ValueError(f"base scene does not contain two objects: {scene_id}")
            schema = str(metadata.get("schema_version", ""))
            if source_name == "generic":
                if schema != "physweep_pybullet_rigid_metadata_v1":
                    raise ValueError("generic two-object source has the wrong schema")
                family = "generic"
            else:
                family = str(sample.get("family", ""))
                if SPECIALIZED_FAMILIES.get(family) != schema:
                    raise ValueError(f"specialized family/schema mismatch: {scene_id}")
            records.append(
                {
                    "scene_id": scene_id,
                    "family": family,
                    "source_schema_version": schema,
                    "metadata_path": metadata_path.relative_to(root).as_posix(),
                    "metadata_sha256": digest,
                }
            )
    scene_ids = [record["scene_id"] for record in records]
    metadata_paths = [record["metadata_path"] for record in records]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("two-object base sources contain duplicate scene ids")
    if len(metadata_paths) != len(set(metadata_paths)):
        raise ValueError("two-object base sources contain duplicate metadata paths")
    records.sort(key=lambda record: str(record["scene_id"]))
    family_counts = {
        family: sum(record["family"] == family for record in records)
        for family in ("generic", *SPECIALIZED_FAMILIES)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "physweep_two_object",
        "object_count": 2,
        "sample_count": len(records),
        "family_counts": family_counts,
        "sources": {
            name: relative_file_binding(root, path)
            for name, path in source_paths.items()
        },
        "records": records,
        "status": "sampled_pending_simulation",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--generic-manifest", type=Path, required=True)
    parser.add_argument("--specialized-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = resolve_project_path_within_root(root, args.output)
    if output.exists():
        raise FileExistsError(output)
    manifest = assemble_base_manifest(
        root, args.generic_manifest, args.specialized_manifest
    )
    write_json_atomic_sorted(output, manifest)
    print(output)


if __name__ == "__main__":
    main()
