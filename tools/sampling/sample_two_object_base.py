#!/usr/bin/env python3
"""Build one deterministic 2obj collision from reviewed 1obj object sources."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json, write_json
from tools.dataset_contract.object_identity_contract import attach_object_identity
from tools.motion_rules.two_object import apply_two_sphere_collision
from tools.sampling.object_pair import compile_object_pair_scene


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "two_object_sampling.json"


def build_two_object_reference(
    host_template: dict[str, Any],
    config: dict[str, Any],
    object_templates: Sequence[dict[str, Any]] | None = None,
    sample_index: int = 1,
) -> dict[str, Any]:
    """Compose an object pair, then apply the independent collision rule."""

    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise ValueError("sample index must be an integer")
    if sample_index < 1:
        raise ValueError("sample index must be positive")
    if set(config) != {"schema_version", "object_pair", "motion_rule"} or (
        config.get("schema_version") != "physweep_two_object_reference_v1"
    ):
        raise ValueError("unsupported two-object reference configuration")
    pair_config = config.get("object_pair")
    if not isinstance(pair_config, dict) or (
        set(pair_config) != {"schema_version", "roles"}
        or pair_config.get("schema_version") != "physweep_object_pair_v1"
    ):
        raise ValueError("unsupported object-pair configuration")
    motion_rule = config.get("motion_rule")
    if not isinstance(motion_rule, dict):
        raise ValueError("two-object reference lacks a motion rule")
    sources = (
        tuple(object_templates)
        if object_templates is not None
        else (host_template, host_template)
    )
    pair_scene = compile_object_pair_scene(
        host_template,
        sources,
        pair_config.get("roles", []),
    )
    scene = apply_two_sphere_collision(pair_scene, motion_rule)
    scene["dataset_stage"] = "two_object_base_candidate"
    scene["sample_index"] = sample_index
    attach_object_identity(scene)
    return scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--object-a-template", type=Path)
    parser.add_argument("--object-b-template", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sample-index", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    host_path = args.template.resolve()
    object_paths = [
        (args.object_a_template or host_path).resolve(),
        (args.object_b_template or host_path).resolve(),
    ]
    config_path = args.config.resolve()
    output_path = args.output.resolve()
    config = read_json(config_path)
    scene = build_two_object_reference(
        read_json(host_path),
        config,
        [read_json(path) for path in object_paths],
        sample_index=args.sample_index,
    )
    write_json(output_path, scene)
    object_ids = [
        str(role["object_id"]) for role in config["object_pair"]["roles"]
    ]
    object_sources = [
        {
            "object_id": object_id,
            "path": str(path.relative_to(root)),
            "sha256": sha256(path),
        }
        for object_id, path in zip(object_ids, object_paths)
    ]
    manifest = {
        "schema_version": "physweep_pybullet_base_manifest_v1",
        "dataset_id": "physweep_two_object",
        "sample_count": 1,
        "config": {
            "path": str(config_path.relative_to(root)),
            "sha256": sha256(config_path),
        },
        "host_template": {
            "path": str(host_path.relative_to(root)),
            "sha256": sha256(host_path),
        },
        "object_sources": object_sources,
        "samples": [
            {
                "scene_id": scene["scene_id"],
                "metadata_path": str(output_path.relative_to(root)),
                "metadata_sha256": sha256(output_path),
            }
        ],
        "status": "sampled_pending_simulation",
    }
    manifest_path = output_path.parent / "manifest.json"
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
