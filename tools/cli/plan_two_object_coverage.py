"""Check generic 2obj coverage and real source assignments without generation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.cli.dataset_generation import CODE_ROOT, generation_code_sha256
from tools.core.hashing import sha256_file
from tools.core.json_io import read_json, write_json
from tools.sampling.sample_two_object_coverage import (
    _validate_complete_scene_coverage, coverage_cells, coverage_summary,
    select_coverage_sources, source_capacity_bounds, source_selection_audit,
)
from tools.sampling.two_object_sources import released_source_pool
from tools.scene_rules.two_object import load_two_object_scene_rules


def selection_records(selections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep source identities and hashes, without compiling scene metadata."""
    return [
        {"cell": value["cell"], "host": value["host"]["source"],
         "objects": [obj["source"] for obj in value["objects"]]}
        for value in selections
    ]


def selection_sha256(selections: list[dict[str, Any]]) -> str:
    payload = json.dumps(selection_records(selections), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_input_bindings(bindings: dict[str, dict[str, str]], code_sha256: str) -> None:
    for name, binding in bindings.items():
        if sha256_file(Path(binding["path"])) != binding["sha256"]:
            raise ValueError(f"coverage planning input changed during assignment: {name}")
    if generation_code_sha256() != code_sha256:
        raise ValueError("coverage planning code changed during assignment")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--released-base-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=CODE_ROOT / "configs/two_object_sampling_matrix.json")
    parser.add_argument("--scene-rules", type=Path, default=CODE_ROOT / "configs/two_object_scene_rules.json")
    parser.add_argument("--generic-limit", type=int, required=True)
    parser.add_argument("--replicates-per-cell", type=int, required=True)
    parser.add_argument("--maximum-object-source-reuse", type=int, required=True)
    parser.add_argument("--maximum-host-source-reuse", type=int, required=True)
    parser.add_argument("--verify-repeatability", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("coverage planning output must be a new directory")
    initial_code = generation_code_sha256()
    bindings = {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in {
            "input_matrix": args.matrix, "scene_rules": args.scene_rules,
            "released_base_manifest": args.released_base_manifest,
            "source_manifest": args.source_root / args.source_manifest,
        }.items()
    }
    matrix = copy.deepcopy(read_json(args.matrix))
    matrix["coverage_plan"]["replicates_per_cell"] = args.replicates_per_cell
    policy = matrix["coverage_plan"]["selection_policy"]
    policy["maximum_object_source_reuse"] = args.maximum_object_source_reuse
    policy["maximum_host_source_reuse"] = args.maximum_host_source_reuse
    rules = load_two_object_scene_rules(args.scene_rules)
    cells, full_count = coverage_cells(matrix, args.generic_limit, scene_rules=rules)
    summary = coverage_summary(cells, full_count, matrix)
    objects, hosts = released_source_pool(
        root=args.root.resolve(), released_base_manifest_path=args.released_base_manifest.resolve(),
        source_root=args.source_root.resolve(), source_manifest_path=args.source_manifest,
        matrix=matrix, scene_rules=rules,
    )
    capacity = source_capacity_bounds(cells, objects, hosts, matrix)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "matrix.json", matrix)
    write_json(output / "capacity.json", capacity)
    if capacity["deficits"]:
        raise ValueError(f"insufficient declared source capacity: {capacity['deficits']}")

    def progress(count: int) -> None:
        if count % 100 == 0 or count == len(cells):
            print(f"Assigned {count}/{len(cells)} source pairs", flush=True)

    selections = select_coverage_sources(
        cells, objects, hosts, matrix, rules,
        require_all_profiles=summary["complete_logical_coverage"], progress=progress,
    )
    if summary["complete_logical_coverage"]:
        _validate_complete_scene_coverage(matrix, rules, selections)
    selected_hash = selection_sha256(selections)
    if args.verify_repeatability:
        repeated = select_coverage_sources(
            cells, list(reversed(objects)), list(reversed(hosts)), matrix, rules,
            require_all_profiles=summary["complete_logical_coverage"], progress=progress,
        )
        if selection_sha256(repeated) != selected_hash:
            raise ValueError("source assignment changed when candidate input order was reversed")
    verify_input_bindings(bindings, initial_code)
    write_json(output / "source_assignments.json", selection_records(selections))
    bindings.update({
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in {
            "matrix": output / "matrix.json",
            "source_assignments": output / "source_assignments.json",
        }.items()
    })
    report = {
        "schema_version": "physweep_two_object_coverage_readiness_v1",
        "status": "source_assignment_verified_pending_physics_and_render",
        "generator_code_sha256": initial_code, "bindings": bindings,
        "coverage": coverage_summary([value["cell"] for value in selections], full_count, matrix),
        "source_reuse": source_selection_audit(selections, matrix),
        "selection_sha256": selected_hash, "reversed_input_repeatability_verified": args.verify_repeatability,
        "scope": "Generic combinatorial coverage and source assignment only; no scenes, physics, cameras or renders generated.",
    }
    write_json(output / "readiness.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
