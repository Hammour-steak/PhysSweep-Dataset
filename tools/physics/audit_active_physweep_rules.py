#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = ROOT / "configs/one_object_sampling_bundle.json"
MATRIX_PATH = ROOT / "configs/one_object_sampling_matrix.json"


def load_unique_json(path: Path) -> Any:
    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        counts = Counter(key for key, _ in pairs)
        duplicates = sorted(key for key, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate JSON keys in {path}: {duplicates}")
        return dict(pairs)

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def collect_named_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"id", "asset_id", "label"} and isinstance(child, str):
                found.add(child)
            found.update(collect_named_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(collect_named_ids(child))
    return found


def duplicate_ids(value: Any, location: str = "root") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            errors.extend(duplicate_ids(child, f"{location}.{key}"))
    elif isinstance(value, list):
        for id_key in ("id", "asset_id", "label"):
            ids = [item[id_key] for item in value if isinstance(item, dict) and id_key in item]
            duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
            if duplicates:
                errors.append(f"duplicate {id_key} at {location}: {duplicates}")
        for index, child in enumerate(value):
            errors.extend(duplicate_ids(child, f"{location}[{index}]"))
    return errors


def unique_paths(paths: list[Path]) -> list[Path]:
    return list(dict.fromkeys(path.resolve() for path in paths))


def main() -> None:
    bundle_path = BUNDLE_PATH
    matrix_path = MATRIX_PATH
    bundle = load_unique_json(bundle_path)
    matrix = load_unique_json(matrix_path)
    bundle_dependencies = [
        ROOT / value
        for key, value in bundle.items()
        if key not in {"version", "implementation", "policy"} and isinstance(value, str)
    ]
    matrix_dependencies = [ROOT / value for value in matrix["dependencies"].values()]
    config_paths = unique_paths(
        [bundle_path, matrix_path, *bundle_dependencies, *matrix_dependencies]
    )
    implementation_paths = unique_paths(
        [
            *(ROOT / value for value in bundle["implementation"].values()),
            *(ROOT / value for value in matrix["implementation"].values()),
        ]
    )
    required_paths = config_paths + implementation_paths
    missing = [str(path.relative_to(ROOT)) for path in required_paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing active paths: {missing}")

    configs = {path: load_unique_json(path) for path in config_paths}
    active_release_literals = {
        str(path.relative_to(ROOT)).replace("\\", "/") for path in config_paths
    }
    release_path_pattern = re.compile(
        r"^(?:configs/(?:one_object_sampling_(?:bundle|matrix)|asset_proxy_registry)\.json|"
        r"assets/proxies/catalog\.json)$"
    )
    errors = [
        error
        for path, value in configs.items()
        for error in duplicate_ids(value, str(path.relative_to(ROOT)))
    ]

    concrete_sources = (
        ROOT / bundle["object_profiles"],
        ROOT / "configs/scene_kits.json",
        ROOT / bundle["asset_proxy_registry"],
    )
    concrete_ids = set().union(*(collect_named_ids(load_unique_json(path)) for path in concrete_sources))
    allowed_behavior_literals = {
        "cuboid", "sphere", "cylinder", "inclined_ramp", "rectangular_slab",
        "ground_flat", "raised_flat", "ground_feature", "raised_feature",
        "lab_bench",
    }
    concrete_ids -= allowed_behavior_literals

    for path in implementation_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        string_literals = {
            node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        retired_release_paths = sorted(
            literal
            for literal in string_literals
            if release_path_pattern.fullmatch(literal)
            and literal not in active_release_literals
        )
        if retired_release_paths:
            errors.append(
                f"retired release paths in {path.relative_to(ROOT)}: "
                f"{retired_release_paths}"
            )
        leaked = sorted(string_literals & concrete_ids)
        if leaked:
            errors.append(f"concrete ids in {path.relative_to(ROOT)}: {leaked}")
        function_names = [
            node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        repeated_functions = sorted(name for name, count in Counter(function_names).items() if count > 1)
        if repeated_functions:
            errors.append(f"duplicate functions in {path.relative_to(ROOT)}: {repeated_functions}")

    source_text = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in implementation_paths
    )
    backend = load_unique_json(ROOT / "configs/pybullet_backend.json")
    unused_quality = [key for key in backend["quality"] if key not in source_text]
    if unused_quality:
        errors.append(f"unreferenced backend quality keys: {unused_quality}")

    active_text = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in required_paths)
    stale_terms = [
        term
        for term in (
            "MPM",
            "trajectory_context_occlusion_camera_v3",
            "object_visual_admission",
            "exclude_profile_from_sampling",
        )
        if term in active_text
    ]
    if stale_terms:
        errors.append(f"stale active terms: {stale_terms}")

    if errors:
        raise SystemExit("\n".join(errors))
    print(
        json.dumps(
            {
                "active_entries": [
                    str(bundle_path.relative_to(ROOT)),
                    str(matrix_path.relative_to(ROOT)),
                ],
                "active_configs": len(config_paths),
                "active_implementations": len(implementation_paths),
                "backend_quality_keys": len(backend["quality"]),
                "status": "clean",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
