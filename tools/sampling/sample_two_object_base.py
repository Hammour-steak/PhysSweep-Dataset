#!/usr/bin/env python3
"""Build the bounded 2obj motion matrix from reviewed 1obj object sources."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json, write_json
from tools.dataset_contract.object_identity_contract import attach_object_identity
from tools.motion_rules.two_object import apply_two_object_motion
from tools.sampling.object_pair import compile_object_pair_scene


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ID = "physweep_two_object"
DEFAULT_MATRIX = PROJECT_ROOT / "configs" / "two_object_sampling_matrix.json"
_MATRIX_FIELDS = {
    "schema_version",
    "object_pair",
    "shared_physics",
    "pair_observation",
    "motion_intents",
    "policy",
}
_POLICY_FIELDS = {
    "post_contact_outcome_is_not_preclassified",
    "one_factor_sweep_keeps_both_initial_states_fixed",
    "camera_and_scene_expansion_deferred",
    "initial_contact_is_deferred",
    "airborne_airborne_is_deferred",
}


def _validated_intents(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    if set(matrix) != _MATRIX_FIELDS or (
        matrix.get("schema_version") != "physweep_two_object_sampling_matrix_v1"
    ):
        raise ValueError("unsupported two-object sampling matrix")
    pair = matrix.get("object_pair")
    if not isinstance(pair, dict) or set(pair) != {"schema_version", "roles"}:
        raise ValueError("two-object matrix lacks an object-pair contract")
    if pair.get("schema_version") != "physweep_object_pair_v1":
        raise ValueError("unsupported object-pair contract")
    policy = matrix.get("policy")
    if not isinstance(policy, dict) or set(policy) != _POLICY_FIELDS:
        raise ValueError("two-object sampling policy is incomplete")
    if not all(bool(policy[field]) for field in _POLICY_FIELDS):
        raise ValueError("two-object sampling policy may not be weakened")
    intents = matrix.get("motion_intents")
    if not isinstance(intents, list) or not intents:
        raise ValueError("two-object matrix contains no motion intents")
    if any(not isinstance(intent, dict) for intent in intents):
        raise ValueError("two-object motion intents must be records")
    ids = [str(intent.get("id", "")) for intent in intents]
    if any(not motion_id for motion_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("two-object motion intent ids must be unique")
    return intents


def build_two_object_scene(
    host_template: dict[str, Any],
    matrix: dict[str, Any],
    motion_id: str,
    object_templates: Sequence[dict[str, Any]] | None = None,
    sample_index: int | None = None,
) -> dict[str, Any]:
    """Compose two sources and apply one declared initial-state intent."""

    intents = _validated_intents(matrix)
    matches = [intent for intent in intents if str(intent["id"]) == motion_id]
    if len(matches) != 1:
        raise ValueError(f"unknown two-object motion intent: {motion_id}")
    declared_index = intents.index(matches[0]) + 1
    resolved_index = declared_index if sample_index is None else sample_index
    if isinstance(resolved_index, bool) or not isinstance(resolved_index, int):
        raise ValueError("sample index must be an integer")
    if resolved_index < 1:
        raise ValueError("sample index must be positive")
    sources = (
        tuple(object_templates)
        if object_templates is not None
        else (host_template, host_template)
    )
    pair_scene = compile_object_pair_scene(
        host_template,
        sources,
        matrix["object_pair"]["roles"],
    )
    scene = apply_two_object_motion(
        pair_scene,
        matrix["shared_physics"],
        matrix["pair_observation"],
        matches[0],
    )
    scene["dataset_id"] = DATASET_ID
    scene["dataset_stage"] = "two_object_base_candidate"
    scene["sample_index"] = resolved_index
    attach_object_identity(scene)
    return scene


def build_two_object_matrix(
    host_template: dict[str, Any],
    matrix: dict[str, Any],
    object_templates: Sequence[dict[str, Any]] | None = None,
    motion_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Build all or a declared subset of matrix rows in stable matrix order."""

    intents = _validated_intents(matrix)
    declared_ids = [str(intent["id"]) for intent in intents]
    requested_ids = (
        declared_ids if motion_ids is None else [str(value) for value in motion_ids]
    )
    if len(requested_ids) != len(set(requested_ids)):
        raise ValueError("two-object motion intents may not be repeated")
    selected = set(requested_ids)
    unknown = sorted(selected.difference(declared_ids))
    if unknown:
        raise ValueError(f"unknown two-object motion intents: {unknown}")
    if not selected:
        raise ValueError("at least one two-object motion intent is required")
    return [
        build_two_object_scene(
            host_template,
            matrix,
            motion_id,
            object_templates,
        )
        for motion_id in declared_ids
        if motion_id in selected
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--object-a-template", type=Path)
    parser.add_argument("--object-b-template", type=Path)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--motion-id", action="append")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    host_path = args.template.resolve()
    object_paths = [
        (args.object_a_template or host_path).resolve(),
        (args.object_b_template or host_path).resolve(),
    ]
    matrix_path = args.matrix.resolve()
    output_dir = args.output_dir.resolve()
    try:
        host_relative = host_path.relative_to(root)
        object_relatives = [path.relative_to(root) for path in object_paths]
        matrix_relative = matrix_path.relative_to(root)
        output_dir.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "two-object matrix inputs and output must remain under --root"
        ) from error
    matrix = read_json(matrix_path)
    scenes = build_two_object_matrix(
        read_json(host_path),
        matrix,
        [read_json(path) for path in object_paths],
        args.motion_id,
    )
    samples = []
    for scene in scenes:
        metadata_path = output_dir / "scenes" / scene["scene_id"] / "metadata.json"
        write_json(metadata_path, scene)
        samples.append(
            {
                "scene_id": scene["scene_id"],
                "metadata_path": str(metadata_path.relative_to(root)),
                "metadata_sha256": sha256(metadata_path),
            }
        )
    object_ids = [
        str(role["object_id"]) for role in matrix["object_pair"]["roles"]
    ]
    object_sources = [
        {
            "object_id": object_id,
            "path": str(relative),
            "sha256": sha256(path),
        }
        for object_id, path, relative in zip(
            object_ids, object_paths, object_relatives
        )
    ]
    manifest = {
        "schema_version": "physweep_pybullet_base_manifest_v1",
        "dataset_id": DATASET_ID,
        "sample_count": len(samples),
        "matrix": {
            "path": str(matrix_relative),
            "sha256": sha256(matrix_path),
        },
        "host_template": {
            "path": str(host_relative),
            "sha256": sha256(host_path),
        },
        "object_sources": object_sources,
        "samples": samples,
        "status": "sampled_pending_simulation",
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
