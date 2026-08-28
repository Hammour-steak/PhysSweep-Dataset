#!/usr/bin/env python3
"""Prepare immutable branch manifests for a staged PhysSweep render."""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json
from tools.core.paths import (
    project_relative_path as root_relative,
    resolve_project_path as project_path,
)
from tools.physics.specialized_backend_registry import specialized_by_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def validated_source_records(
    manifest: dict[str, Any], expected_pipelines: set[str]
) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != "physweep_one_object_decoupled_manifest_v5":
        raise ValueError("formal render preparation requires the active manifest schema")
    records = list(manifest["records"])
    if int(manifest["sample_count"]) != len(records):
        raise ValueError("source manifest sample count is inconsistent")
    scene_ids = [str(record["scene_id"]) for record in records]
    indices = [int(record["index"]) for record in records]
    if len(scene_ids) != len(set(scene_ids)) or len(indices) != len(set(indices)):
        raise ValueError("source manifest contains duplicate scene ids or indices")
    pipelines = {str(record["pipeline"]) for record in records}
    if not pipelines <= expected_pipelines:
        raise ValueError(
            f"source manifest contains unknown pipelines: "
            f"{pipelines - expected_pipelines}"
        )
    return records


def choose_one(
    records: list[dict[str, Any]],
    key: str,
    value: str,
    seed: str,
) -> dict[str, Any]:
    candidates = sorted(
        (record for record in records if str(record[key]) == value),
        key=lambda record: str(record["scene_id"]),
    )
    if not candidates:
        raise ValueError(f"no candidate for {key}={value}")
    return candidates[random.Random(seed).randrange(len(candidates))]


def choose_unselected(
    records: list[dict[str, Any]],
    key: str,
    value: str,
    seed: str,
    selected_ids: set[str],
) -> dict[str, Any]:
    candidates = [
        record
        for record in records
        if str(record[key]) == value
        and str(record["scene_id"]) not in selected_ids
    ]
    if not candidates:
        raise ValueError(f"no unselected candidate for {key}={value}")
    candidates.sort(key=lambda record: str(record["scene_id"]))
    return candidates[random.Random(seed).randrange(len(candidates))]


def pilot_selection(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = list(manifest["records"])
    generic = [record for record in records if record["pipeline"] == "generic_pybullet"]
    assets = [record for record in records if record["pipeline"] == "asset_proxy"]
    billiards = [record for record in records if record["pipeline"] == "billiards"]
    pinball = [
        record for record in records if record["pipeline"] == "passive_pinball"
    ]
    dataset_id = str(manifest["dataset_id"])

    chosen: list[dict[str, Any]] = []
    for motion in sorted({str(record["motion_intent"]) for record in generic}):
        chosen.append(
            choose_one(generic, "motion_intent", motion, f"{dataset_id}:generic:{motion}")
        )
    for profile in sorted({str(record["profile"]) for record in assets}):
        chosen.append(
            choose_one(assets, "profile", profile, f"{dataset_id}:asset:{profile}")
        )
    for profile in sorted({str(record["profile"]) for record in billiards}):
        chosen.append(
            choose_one(
                billiards,
                "profile",
                profile,
                f"{dataset_id}:billiards:{profile}",
            )
        )

    if pinball:
        chosen.append(
            choose_one(
                pinball,
                "pipeline",
                "passive_pinball",
                f"{dataset_id}:passive-pinball",
            )
        )

    chosen_ids = {str(record["scene_id"]) for record in chosen}
    if not pinball:
        remaining_generic = [
            record for record in generic if str(record["scene_id"]) not in chosen_ids
        ]
        extra = random.Random(f"{dataset_id}:generic:extra").choice(
            remaining_generic
        )
        chosen.append(extra)
    chosen.sort(key=lambda record: int(record["index"]))
    if len(chosen) != 20:
        raise ValueError(f"pilot selection must contain 20 records, got {len(chosen)}")
    return chosen


def pilot40_selection(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = list(manifest["records"])
    chosen = pilot_selection(manifest)
    selected_ids = {str(record["scene_id"]) for record in chosen}
    dataset_id = str(manifest["dataset_id"])

    groups = (
        (
            [record for record in records if record["pipeline"] == "generic_pybullet"],
            "motion_intent",
            "generic",
        ),
        (
            [record for record in records if record["pipeline"] == "asset_proxy"],
            "profile",
            "asset",
        ),
        (
            [record for record in records if record["pipeline"] == "billiards"],
            "profile",
            "billiards",
        ),
    )
    # pilot_selection already keeps one example for each billiards profile.
    # Do not add a second coverage layer for billiards; use generic scenes to
    # fill the fixed-size visual review set instead.
    for pool, key, label in groups[:2]:
        for value in sorted({str(record[key]) for record in pool}):
            extra = choose_unselected(
                pool,
                key,
                value,
                f"{dataset_id}:{label}:extra:{value}",
                selected_ids,
            )
            chosen.append(extra)
            selected_ids.add(str(extra["scene_id"]))

    pinball = [
        record for record in records if record["pipeline"] == "passive_pinball"
    ]
    selected_pinball_profiles = {
        str(record["profile"])
        for record in chosen
        if record["pipeline"] == "passive_pinball"
    }
    for profile in sorted({str(record["profile"]) for record in pinball}):
        if profile in selected_pinball_profiles:
            continue
        extra = choose_unselected(
            pinball,
            "profile",
            profile,
            f"{dataset_id}:passive-pinball:extra:{profile}",
            selected_ids,
        )
        chosen.append(extra)
        selected_ids.add(str(extra["scene_id"]))

    while len(chosen) < 40:
        remaining_generic = [
            record
            for record in records
            if record["pipeline"] == "generic_pybullet"
            and str(record["scene_id"]) not in selected_ids
        ]
        if not remaining_generic:
            raise ValueError("not enough generic records to fill pilot40 selection")
        remaining_generic.sort(key=lambda record: str(record["scene_id"]))
        extra = random.Random(
            f"{dataset_id}:generic:extra40:{len(chosen)}"
        ).choice(remaining_generic)
        chosen.append(extra)
        selected_ids.add(str(extra["scene_id"]))
    if len(chosen) != 40:
        raise ValueError(f"pilot40 selection must contain 40 records, got {len(chosen)}")
    chosen.sort(key=lambda record: int(record["index"]))
    return chosen


def choose_balanced(
    records: list[dict[str, Any]],
    key: str,
    count: int,
    seed: str,
    selected_ids: set[str],
) -> list[dict[str, Any]]:
    """Choose deterministically while keeping categorical coverage balanced."""
    values = sorted({str(record[key]) for record in records})
    if not values:
        raise ValueError(f"cannot balance an empty record pool by {key}")
    chosen: list[dict[str, Any]] = []
    round_index = 0
    while len(chosen) < count:
        progressed = False
        for value in values:
            if len(chosen) == count:
                break
            candidates = sorted(
                (
                    record
                    for record in records
                    if str(record[key]) == value
                    and str(record["scene_id"]) not in selected_ids
                ),
                key=lambda record: str(record["scene_id"]),
            )
            if not candidates:
                continue
            record = candidates[
                random.Random(f"{seed}:{round_index}:{value}").randrange(
                    len(candidates)
                )
            ]
            chosen.append(record)
            selected_ids.add(str(record["scene_id"]))
            progressed = True
        if not progressed:
            raise ValueError(f"not enough records to select {count} balanced by {key}")
        round_index += 1
    return chosen


def stress60_selection(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Select a deterministic 60-scene physics and visual stress review."""
    records = list(manifest["records"])
    generic = [record for record in records if record["pipeline"] == "generic_pybullet"]
    assets = [record for record in records if record["pipeline"] == "asset_proxy"]
    billiards = [record for record in records if record["pipeline"] == "billiards"]
    pinball = [
        record for record in records if record["pipeline"] == "passive_pinball"
    ]
    transition_motions = {"edge_fall_1obj", "ramp_to_flat_1obj"}
    ordinary = [
        record
        for record in generic
        if str(record["motion_intent"]) not in transition_motions
    ]
    transitions = [
        record
        for record in generic
        if str(record["motion_intent"]) in transition_motions
    ]
    dataset_id = str(manifest["dataset_id"])
    selected_ids: set[str] = set()
    chosen = choose_balanced(
        ordinary,
        "motion_intent",
        28 if pinball else 30,
        f"{dataset_id}:stress60:ordinary",
        selected_ids,
    )
    chosen.extend(
        choose_balanced(
            transitions,
            "motion_intent",
            20,
            f"{dataset_id}:stress60:transition",
            selected_ids,
        )
    )
    chosen.extend(
        choose_balanced(
            assets,
            "profile",
            8,
            f"{dataset_id}:stress60:asset",
            selected_ids,
        )
    )
    chosen.extend(
        choose_balanced(
            billiards,
            "profile",
            2,
            f"{dataset_id}:stress60:billiards",
            selected_ids,
        )
    )
    if pinball:
        chosen.extend(
            choose_balanced(
                pinball,
                "profile",
                2,
                f"{dataset_id}:stress60:passive-pinball",
                selected_ids,
            )
        )
    if len(chosen) != 60:
        raise ValueError(f"stress60 selection must contain 60 records, got {len(chosen)}")
    chosen.sort(key=lambda record: int(record["index"]))
    return chosen


def review100_selection(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Select a deterministic 100-scene formal distribution review."""
    records = list(manifest["records"])
    generic = [
        record for record in records if record["pipeline"] == "generic_pybullet"
    ]
    assets = [record for record in records if record["pipeline"] == "asset_proxy"]
    billiards = [record for record in records if record["pipeline"] == "billiards"]
    pinball = [
        record for record in records if record["pipeline"] == "passive_pinball"
    ]
    dataset_id = str(manifest["dataset_id"])
    selected_ids: set[str] = set()

    chosen = choose_balanced(
        generic,
        "motion_intent",
        68 if pinball else 70,
        f"{dataset_id}:review100:generic",
        selected_ids,
    )
    chosen.extend(
        choose_balanced(
            assets,
            "environment_id",
            14,
            f"{dataset_id}:review100:asset-environment",
            selected_ids,
        )
    )
    chosen.extend(
        choose_balanced(
            assets,
            "profile",
            14,
            f"{dataset_id}:review100:asset-profile",
            selected_ids,
        )
    )
    chosen.extend(
        choose_balanced(
            billiards,
            "profile",
            2,
            f"{dataset_id}:review100:billiards",
            selected_ids,
        )
    )
    if pinball:
        chosen.extend(
            choose_balanced(
                pinball,
                "profile",
                2,
                f"{dataset_id}:review100:passive-pinball",
                selected_ids,
            )
        )
    if len(chosen) != 100:
        raise ValueError(
            f"review100 selection must contain 100 records, got {len(chosen)}"
        )
    chosen.sort(key=lambda record: int(record["index"]))
    return chosen


def update_counts(manifest: dict[str, Any], records: list[dict[str, Any]]) -> None:
    manifest["sample_count"] = len(records)
    manifest["motion_counts"] = dict(
        sorted(Counter(str(record["motion_intent"]) for record in records).items())
    )
    manifest["environment_counts"] = dict(
        sorted(Counter(str(record["environment_id"]) for record in records).items())
    )
    manifest["profile_counts"] = dict(
        sorted(Counter(str(record["profile"]) for record in records).items())
    )
    manifest["records"] = records


def stage_render_record(
    root: Path,
    source_record: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    record = copy.deepcopy(source_record)
    source_path = project_path(root, record["metadata_path"])
    source_path.relative_to(root)
    source_hash = sha256(source_path)
    declared_hash = record.get("metadata_sha256")
    if declared_hash is not None and str(declared_hash) != source_hash:
        raise ValueError(f"source metadata hash mismatch: {source_path}")
    scene_id = str(record["scene_id"])
    record["metadata_path"] = root_relative(root, source_path)
    record["metadata_sha256"] = source_hash
    record["render_output"] = {
        "video_path": root_relative(root, output_root / "videos" / f"{scene_id}.mp4"),
        "inspection_frame_dir": root_relative(root, output_root / "frames" / scene_id),
    }
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--selection",
        choices=("pilot20", "pilot40", "stress60", "review100", "all"),
        default="pilot20",
    )
    parser.add_argument(
        "--pipeline",
        help=(
            "Restrict an all-record plan to one pipeline. This stages a "
            "versioned replacement branch without restaging retained data."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_records(
    manifest: dict[str, Any], selection: str, pipeline: str | None = None
) -> list[dict[str, Any]]:
    if pipeline is not None and selection != "all":
        raise ValueError("a pipeline filter requires --selection all")
    selectors = {
        "pilot20": pilot_selection,
        "pilot40": pilot40_selection,
        "stress60": stress60_selection,
        "review100": review100_selection,
    }
    if selection == "all":
        records = list(manifest["records"])
    elif selection in selectors:
        records = selectors[selection](manifest)
    else:
        raise ValueError(f"unknown render selection: {selection}")
    if pipeline is not None:
        records = [record for record in records if record["pipeline"] == pipeline]
        if not records:
            raise ValueError(f"source manifest contains no {pipeline} records")
    return records


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_manifest_path = project_path(root, args.manifest)
    source_manifest_path.relative_to(root / "datasets")
    source_manifest = load_json(source_manifest_path)
    specialized = specialized_by_pipeline(root)
    pipeline_order = ["generic_pybullet", *specialized]
    validated_source_records(source_manifest, set(pipeline_order))
    if args.pipeline is not None and args.pipeline not in pipeline_order:
        raise ValueError(f"unknown pipeline filter: {args.pipeline}")
    output_root = project_path(root, args.output_root)
    if (root / "outputs").resolve() not in output_root.parents:
        raise ValueError("formal render output must remain under root/outputs")

    records = select_records(source_manifest, args.selection, args.pipeline)
    selection_id = args.selection
    if args.pipeline is not None:
        selection_id = f"{selection_id}__{args.pipeline}"
    selected_by_pipeline = {
        pipeline: [record for record in records if record["pipeline"] == pipeline]
        for pipeline in pipeline_order
    }

    source_generic_path = project_path(root, source_manifest["generic_manifest_path"])
    source_generic_path.relative_to(root / "datasets")
    source_generic = load_json(source_generic_path)
    generic_source_samples = list(source_generic["samples"])
    if int(source_generic["sample_count"]) != len(generic_source_samples):
        raise ValueError("generic source manifest sample count is inconsistent")
    generic_paths = [str(sample["metadata_path"]) for sample in generic_source_samples]
    if len(generic_paths) != len(set(generic_paths)):
        raise ValueError("generic source manifest contains duplicate metadata paths")
    selected_generic_paths = {
        str(record["metadata_path"])
        for record in selected_by_pipeline["generic_pybullet"]
    }
    generic_samples = [
        sample
        for sample in generic_source_samples
        if str(sample["metadata_path"]) in selected_generic_paths
    ]
    if len(generic_samples) != len(selected_generic_paths):
        raise ValueError("generic render selection does not match source manifest")
    for sample in generic_samples:
        metadata_path = project_path(root, sample["metadata_path"])
        metadata_path.relative_to(root / "datasets")
        metadata = load_json(metadata_path)
        if (
            sha256(metadata_path) != str(sample["metadata_sha256"])
            or str(metadata["scene_id"]) != str(sample["scene_id"])
        ):
            raise ValueError(f"generic source provenance mismatch: {metadata_path}")
    generic_source = copy.deepcopy(source_generic)
    generic_source["sample_count"] = len(generic_samples)
    generic_source["samples"] = generic_samples
    generic_source_path = output_root / "manifests" / "generic_source_manifest.json"

    source_asset_path = project_path(root, source_manifest["asset_proxy_manifest_path"])
    source_asset_path.relative_to(root / "datasets")
    source_asset = load_json(source_asset_path)
    selected_child_ids = {
        str(record["child_scene_id"])
        for record in selected_by_pipeline["asset_proxy"]
    }
    source_asset_records = {
        str(record["scene_id"]): record for record in source_asset["records"]
    }
    if (
        int(source_asset["sample_count"]) != len(source_asset["records"])
        or len(source_asset_records) != len(source_asset["records"])
    ):
        raise ValueError("asset source manifest contains duplicate or missing records")
    asset_root = output_root / "asset"
    staged_asset_records = []
    for scene_id in sorted(selected_child_ids):
        child = source_asset_records.get(scene_id)
        if child is None:
            raise ValueError(f"asset child is missing from source manifest: {scene_id}")
        outer_scene_ids = {
            str(record["scene_id"])
            for record in selected_by_pipeline["asset_proxy"]
            if str(record["child_scene_id"]) == scene_id
        }
        if outer_scene_ids != {str(child["matrix_scene_id"])}:
            raise ValueError(f"asset child matrix provenance mismatch: {scene_id}")
        staged_asset_records.append(
            stage_render_record(root, child, asset_root)
        )
    asset_manifest = copy.deepcopy(source_asset)
    asset_manifest["dataset_id"] = f"{source_manifest['dataset_id']}__{selection_id}"
    asset_manifest["output_root"] = str(asset_root)
    asset_manifest["sample_count"] = len(staged_asset_records)
    asset_manifest["passed_count"] = len(staged_asset_records)
    asset_manifest["records"] = staged_asset_records
    asset_manifest_path = asset_root / "asset_render_manifest.json"

    staged_specialized: dict[str, tuple[Path, dict[str, Any]]] = {}
    for pipeline, backend in specialized.items():
        if pipeline == "asset_proxy":
            continue
        branch = str(backend["sweep_branch"])
        branch_root = output_root / branch
        branch_records = [
            stage_render_record(root, outer_record, branch_root)
            for outer_record in selected_by_pipeline[pipeline]
        ]
        branch_manifest = {
            "schema_version": f"physweep_{branch}_staged_manifest_v1",
            "dataset_id": f"{source_manifest['dataset_id']}__{selection_id}",
            "source_manifest": str(source_manifest_path),
            "output_root": root_relative(root, branch_root),
            "sample_count": len(branch_records),
            "records": branch_records,
        }
        manifest_path = branch_root / f"{branch}_manifest.json"
        staged_specialized[pipeline] = (
            manifest_path,
            branch_manifest,
        )

    staged_outer = copy.deepcopy(source_manifest)
    staged_outer.pop("generic_simulation_manifest_path", None)
    for backend in specialized.values():
        staged_outer.pop(f"{backend['sweep_branch']}_metadata_paths", None)
    staged_outer["dataset_id"] = f"{source_manifest['dataset_id']}__{selection_id}"
    staged_outer["source_manifest"] = str(source_manifest_path)
    staged_outer["selection"] = args.selection
    staged_outer["pipeline_filter"] = args.pipeline
    update_counts(staged_outer, records)
    staged_outer["generic_manifest_path"] = root_relative(root, generic_source_path)
    staged_outer["asset_proxy_manifest_path"] = root_relative(root, asset_manifest_path)
    outer_manifest_path = output_root / "staged_manifest.json"

    plan = {
        "schema_version": "physweep_staged_render_plan_v1",
        "selection": args.selection,
        "pipeline_filter": args.pipeline,
        "sample_count": len(records),
        "pipeline_counts": {
            pipeline: len(values) for pipeline, values in selected_by_pipeline.items()
        },
        "source_manifest": str(source_manifest_path),
        "staged_manifest": str(outer_manifest_path),
        "generic_source_manifest": str(generic_source_path),
        "generic_bound_manifest": str(output_root / "generic" / "bound_manifest.json"),
        "generic_render_manifest": str(output_root / "generic" / "render_manifest.json"),
        "asset_render_input_manifest": str(asset_manifest_path),
        "asset_render_manifest": str(asset_root / "render_manifest.json"),
    }
    for pipeline, backend in specialized.items():
        if pipeline == "asset_proxy":
            continue
        branch = str(backend["sweep_branch"])
        plan[f"{branch}_render_input_manifest"] = str(
            staged_specialized[pipeline][0]
        )
        plan[f"{branch}_render_manifest"] = str(
            output_root / branch / str(backend["render_manifest_name"])
        )
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    write_json(generic_source_path, generic_source)
    write_json(asset_manifest_path, asset_manifest)
    for manifest_path, manifest in staged_specialized.values():
        write_json(manifest_path, manifest)
    write_json(outer_manifest_path, staged_outer)
    write_json(output_root / "render_plan.json", plan)
    print(json.dumps(plan, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
