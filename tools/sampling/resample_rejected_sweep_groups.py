#!/usr/bin/env python3
"""Materialize deterministic replacements for rejected asset-proxy sweep groups."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.sampling.sample_asset_proxy_scenes import proxy_volume_fill_ratio
from tools.sampling.sample_one_object_scene_matrix import (
    MATRIX_PATH,
    build_schedule,
    load_json,
    matrix_dependency_paths,
    sha256,
    validate_matrix,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SLOT_KEYS = ("generator", "motion_intent", "environment_id", "profile")


def asset_signature(record: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    return (
        record.get("dynamic_asset_id"),
        record.get("support_asset_id"),
        record.get("static_prop_asset_id"),
    )


def select_replacement_slots(
    originals: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    occupied_signatures: set[tuple[str | None, str | None, str | None]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used_candidate_indices: set[int] = set()
    for original in sorted(originals, key=lambda record: int(record["index"])):
        compatible = [
            candidate
            for candidate in candidates
            if int(candidate["index"]) not in used_candidate_indices
            and all(candidate[key] == original[key] for key in SLOT_KEYS)
            and candidate.get("dynamic_asset_id") != original.get("dynamic_asset_id")
        ]
        novel = [
            candidate
            for candidate in compatible
            if asset_signature(candidate) not in occupied_signatures
        ]
        pool = novel or compatible
        if not pool:
            raise RuntimeError(
                "no deterministic replacement candidate for slot "
                f"{original['index']} ({original['motion_intent']}, "
                f"{original['environment_id']}, {original['profile']})"
            )
        candidate = pool[0]
        used_candidate_indices.add(int(candidate["index"]))
        occupied_signatures.add(asset_signature(candidate))
        selected.append((original, candidate))
    return selected


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
        raise RuntimeError(f"command failed: {' '.join(command)}\n{tail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--matrix", type=Path, default=MATRIX_PATH)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--rejected-groups", type=Path, required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    matrix_path = resolve(root, args.matrix)
    base_manifest_path = resolve(root, args.base_manifest)
    rejected_groups_path = resolve(root, args.rejected_groups)
    matrix = load_json(matrix_path)
    validate_matrix(root, matrix)
    dependencies = matrix_dependency_paths(root, matrix)
    base_manifest = load_json(base_manifest_path)
    rejected = load_json(rejected_groups_path)

    base_records = list(base_manifest["records"])
    by_metadata_path = {
        str(record["metadata_path"]): record for record in base_records
    }
    originals = []
    for group in rejected["groups"]:
        parent = str(group["parent"])
        if parent not in by_metadata_path:
            raise ValueError(f"rejected parent is absent from base manifest: {parent}")
        original = by_metadata_path[parent]
        if original["generator"] != "asset_proxy":
            raise ValueError(
                "this replacement tool accepts asset-proxy groups only: "
                f"{original['scene_id']}"
            )
        originals.append(original)

    registry = load_json(dependencies["asset_proxy_registry"])
    backend = load_json(dependencies["physics_backend"])
    edge_rules = backend["asset_proxy_rules"]["motion_profiles"]["edge_exit"]
    minimum_edge_fill = float(edge_rules["minimum_proxy_volume_fill_ratio"])
    edge_eligible = {
        str(record["asset_id"])
        for record in registry["records"]
        if record["proxy"]["kind"] == "dynamic_rigid"
        and record["admission"].get("sampling_enabled", False)
        and proxy_volume_fill_ratio(record) >= minimum_edge_fill
    }
    candidates = build_schedule(
        matrix,
        int(base_manifest["sample_count"]),
        args.seed,
        profile_dynamic_eligibility={"edge_exit": edge_eligible},
    )
    selected = select_replacement_slots(
        originals,
        candidates,
        {asset_signature(record) for record in base_records},
    )

    output_root = (root / "datasets" / args.output_dataset).resolve()
    datasets_root = (root / "datasets").resolve()
    if datasets_root not in output_root.parents:
        raise ValueError("output dataset must remain under datasets")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    production = load_json(dependencies["production_video"])
    records = []
    for attempt_index, (original, candidate) in enumerate(selected, start=1):
        replacement_scene_id = (
            f"{original['scene_id']}__replacement_{attempt_index:02d}"
        )
        scene_dir = output_root / "specialized" / replacement_scene_id
        command = [
            sys.executable,
            "-m",
            "tools.sampling.sample_asset_proxy_scenes",
            "--root",
            str(root),
            "--registry",
            str(dependencies["asset_proxy_registry"]),
            "--catalog",
            str(dependencies["physical_proxy_catalog"]),
            "--semantic-rules",
            str(dependencies["asset_semantic_scene_rules"]),
            "--composition-rules",
            str(dependencies["asset_scene_composition"]),
            "--visual-rules",
            str(dependencies["visual_sampling"]),
            "--output",
            str(scene_dir),
            "--count",
            "1",
            "--seed",
            str(candidate["seed"]),
            "--support-id",
            str(candidate["support_asset_id"]),
            "--dynamic-id",
            str(candidate["dynamic_asset_id"]),
            "--profiles",
            str(candidate["profile"]),
            "--scene-id-prefix",
            replacement_scene_id,
            "--duration",
            str(production["duration_s"]),
            "--fps",
            str(production["output_fps"]),
            "--resolution",
            *[str(value) for value in production["resolution"]],
            "--samples",
            str(production["samples"]),
        ]
        if candidate.get("static_prop_asset_id"):
            command.extend(
                ["--static-prop-id", str(candidate["static_prop_asset_id"])]
            )
        else:
            command.append("--no-static-props")
        run(command, root)
        child_manifest = load_json(scene_dir / "manifest.json")
        if (
            int(child_manifest["sample_count"]) != 1
            or int(child_manifest["passed_count"]) != 1
        ):
            raise RuntimeError(f"replacement base failed: {replacement_scene_id}")
        child = child_manifest["records"][0]
        metadata_path = root / str(child["metadata_path"])
        records.append(
            {
                **candidate,
                "index": original["index"],
                "scene_id": replacement_scene_id,
                "pipeline": "asset_proxy",
                "child_scene_id": child["scene_id"],
                "metadata_path": child["metadata_path"],
                "metadata_sha256": sha256(metadata_path),
                "status": "simulated_accepted",
                "replaces_scene_id": original["scene_id"],
                "replaces_metadata_path": original["metadata_path"],
                "replacement_seed": args.seed,
            }
        )

    records.sort(key=lambda record: int(record["index"]))
    write_json(
        output_root / "manifest.json",
        {
            "schema_version": "physweep_one_object_replacement_manifest_v1",
            "dataset_id": args.output_dataset,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "sample_count": len(records),
            "source_base_manifest": str(base_manifest_path.relative_to(root)),
            "source_base_manifest_sha256": sha256(base_manifest_path),
            "rejected_groups": str(rejected_groups_path.relative_to(root)),
            "rejected_groups_sha256": sha256(rejected_groups_path),
            "slot_contract": list(SLOT_KEYS),
            "records": records,
        },
    )
    print(f"replacement base manifest: {output_root / 'manifest.json'}")
    print(f"replacements={len(records)}")


if __name__ == "__main__":
    main()
