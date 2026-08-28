#!/usr/bin/env python3
"""Generate deterministic whole-slot replacements from an extension spec."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json_atomic as write_json
from tools.core.paths import resolve_project_path as project_path
from tools.release.specialized_release_extension import (
    load_extension_spec,
    project_root_reference,
    select_replacement_slots,
    stable_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def root_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def module_name(root: Path, script: Path) -> str:
    relative = script.resolve().relative_to(root.resolve())
    if relative.suffix != ".py" or relative.name == "__init__.py":
        raise ValueError(f"generator is not an executable Python module: {script}")
    return ".".join(relative.with_suffix("").parts)


def validate_binding(root: Path, binding: dict[str, Any], label: str) -> None:
    path = Path(str(binding["path"]))
    declared = path if path.is_absolute() else root / path
    declared.absolute().relative_to(root)
    resolved = declared.resolve()
    if not resolved.is_file() or sha256(resolved) != str(binding["sha256"]):
        raise ValueError(f"replacement {label} binding changed: {resolved}")


def validate_existing_candidate(
    root: Path,
    candidate_dir: Path,
    scene_id: str,
    profile: str,
    seed: int,
    spec: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    metadata_path = candidate_dir / "metadata.json"
    trajectory_path = candidate_dir / "trajectory.npz"
    audit_path = candidate_dir / "audit.json"
    record_path = candidate_dir / "simulation_record.json"
    for path in (metadata_path, trajectory_path, audit_path, record_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = load_json(metadata_path)
    audit = load_json(audit_path)
    record = load_json(record_path)
    replacement = spec["replacement"]
    if (
        metadata.get("schema_version") != replacement["scene_schema_version"]
        or str(metadata.get("scene_id")) != scene_id
        or int(metadata.get("seed")) != seed
        or str(metadata.get("semantics", {}).get("profile")) != profile
        or int(metadata.get("semantics", {}).get("dynamic_object_count", -1)) != 1
        or int(metadata.get("semantics", {}).get("active_mechanism_count", -1)) != 0
        or not audit.get("passed")
        or str(record.get("scene_id")) != scene_id
        or sha256(metadata_path) != str(record["metadata"]["sha256"])
        or sha256(trajectory_path) != str(record["trajectory"]["sha256"])
        or sha256(audit_path) != str(record["audit"]["sha256"])
    ):
        raise ValueError(f"existing specialized candidate is not reusable: {scene_id}")
    validate_binding(root, metadata["physics"]["backend_config"], "backend")
    candidate_binding = metadata["physics"].get("candidate_config")
    if candidate_binding is not None:
        validate_binding(root, candidate_binding, "candidate")
    for label, binding in metadata["implementation"].items():
        validate_binding(root, binding, f"implementation {label}")
    return metadata_path, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_root = args.source_root.resolve()
    spec_path = project_path(root, args.spec)
    spec = load_extension_spec(root, spec_path)
    group = spec["group_contract"]
    replacement = spec["replacement"]
    source_contract = spec["source_release"]
    release_path = project_path(source_root, args.source_release)
    release_path.relative_to(source_root / "datasets")
    release = load_json(release_path)
    if (
        release.get("schema_version") != source_contract["schema_version"]
        or release.get("dataset_id") != source_contract["dataset_id"]
        or Counter(release["pipeline_group_counts"])
        != Counter(source_contract["pipeline_group_counts"])
    ):
        raise ValueError("source release differs from the extension contract")
    base_path = project_path(source_root, release["base_manifest"])
    if sha256(base_path) != str(release["base_manifest_sha256"]):
        raise ValueError("source release base manifest hash mismatch")
    base = load_json(base_path)
    if (
        base.get("schema_version") != source_contract["base_manifest_schema"]
        or int(base["sample_count"]) != int(group["base_group_count"])
        or int(base["sample_count"]) != len(base["records"])
        or Counter(record["pipeline"] for record in base["records"])
        != Counter(source_contract["pipeline_group_counts"])
    ):
        raise ValueError("source base manifest differs from the extension contract")
    output_root = project_path(root, args.output_root)
    output_root.relative_to(root / "datasets")
    manifest_path = output_root / "manifest.json"
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"replacement output exists; use --resume: {output_root}")
    if args.resume and manifest_path.exists():
        raise FileExistsError(f"replacement manifest is already complete: {manifest_path}")
    output_root.mkdir(parents=True, exist_ok=True)

    count = int(group["replacement_group_count"])
    chosen = select_replacement_slots(
        base["records"],
        count,
        int(args.seed),
        namespace=str(replacement["selection_namespace"]),
        source_pipeline=str(replacement["source_pipeline"]),
        motion_intent=str(replacement["source_motion_intent"]),
    )
    production = base["production_spec"]
    generator = project_path(root, replacement["generator_script"])
    backend_config = project_path(root, replacement["backend_config"])
    profiles = [str(value) for value in replacement["profiles"]]
    records = []
    for ordinal, original in enumerate(chosen):
        index = int(original["index"])
        original_metadata_path = project_path(source_root, original["metadata_path"])
        if sha256(original_metadata_path) != str(original["metadata_sha256"]):
            raise ValueError(f"source slot metadata hash mismatch: {original_metadata_path}")
        profile = profiles[ordinal % len(profiles)]
        scene_id = str(replacement["scene_id_template"]).format(index=index)
        candidate_seed = stable_seed(
            ":".join(
                (
                    str(replacement["candidate_seed_namespace"]),
                    str(args.seed),
                    str(index),
                    profile,
                )
            )
        )
        candidate_dir = output_root / "base" / scene_id
        if candidate_dir.exists():
            if not args.resume:
                raise FileExistsError(candidate_dir)
            metadata_path, metadata = validate_existing_candidate(
                root, candidate_dir, scene_id, profile, candidate_seed, spec
            )
        else:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    module_name(root, generator),
                    "--root",
                    str(root),
                    "--config",
                    str(backend_config),
                    "--output",
                    str(candidate_dir),
                    "--seed",
                    str(candidate_seed),
                    "--profile",
                    profile,
                    "--scene-id",
                    scene_id,
                    "--resolution",
                    *[str(value) for value in production["resolution"]],
                    "--samples",
                    str(production["samples"]),
                ],
                cwd=root,
                check=True,
            )
            metadata_path, metadata = validate_existing_candidate(
                root, candidate_dir, scene_id, profile, candidate_seed, spec
            )
        time_binding = metadata["simulation"]["time"]
        if (
            float(time_binding["duration_s"]) != float(production["duration_s"])
            or int(time_binding["output_fps"]) != int(production["output_fps"])
            or int(time_binding["frame_count"]) != int(production["frame_count"])
            or list(metadata["render"]["resolution"]) != list(production["resolution"])
            or int(metadata["render"]["samples"]) != int(production["samples"])
        ):
            raise ValueError(f"candidate production contract mismatch: {scene_id}")
        records.append(
            {
                "index": index,
                "scene_id": scene_id,
                "seed": candidate_seed,
                "motion_intent": replacement["source_motion_intent"],
                "environment_id": replacement["environment_id"],
                "generator": replacement["generator"],
                "profile": profile,
                "pipeline": replacement["pipeline"],
                "dynamic_asset_id": None,
                "support_asset_id": None,
                "static_prop_asset_id": None,
                "metadata_path": root_relative(root, metadata_path),
                "metadata_sha256": sha256(metadata_path),
                "status": "simulated_accepted",
                "replaces_scene_id": str(original["scene_id"]),
                "replaces_metadata_path": str(original["metadata_path"]),
                "replaces_metadata_sha256": str(original["metadata_sha256"]),
                "replaces_pipeline": str(original["pipeline"]),
            }
        )
    expected_profiles = Counter(
        {profile: count // len(profiles) for profile in profiles}
    )
    if Counter(record["profile"] for record in records) != expected_profiles:
        raise RuntimeError("specialized profile allocation is not balanced")
    manifest = {
        "schema_version": "physweep_specialized_replacement_manifest_v1",
        "dataset_id": f"{spec['target_release']['dataset_id']}_{replacement['pipeline']}_replacements",
        "extension_spec": {
            "path": root_relative(root, spec_path),
            "sha256": sha256(spec_path),
            "extension_id": spec["extension_id"],
        },
        "sample_count": len(records),
        "source_project_root": project_root_reference(root, source_root),
        "source_release": root_relative(source_root, release_path),
        "source_release_sha256": sha256(release_path),
        "source_base_manifest": root_relative(source_root, base_path),
        "source_base_manifest_sha256": sha256(base_path),
        "selection_policy": {
            "version": replacement["selection_namespace"],
            "seed": int(args.seed),
            "source_pipeline": replacement["source_pipeline"],
            "source_motion_intent": replacement["source_motion_intent"],
            "ranking": "sha256_ascending",
            "preserved_slot_fields": group["preserved_slot_fields"],
            "whole_group_replacement": True,
        },
        "profile_counts": dict(Counter(record["profile"] for record in records)),
        "records": records,
    }
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": root_relative(root, manifest_path),
                "sample_count": len(records),
                "profile_counts": manifest["profile_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
