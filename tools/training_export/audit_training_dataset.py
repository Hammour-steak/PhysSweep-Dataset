#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from tools.core.hashing import sha256_file as _sha256
from tools.core.sweep_values import SWEEP_AXES, sweep_group_size
from tools.dataset_contract.gt_scene_input import (
    MODEL_SCENE_SCHEMA,
    inspect_model_scene_condition,
)
from tools.dataset_contract.schema import iter_jsonl, validate_manifest


def _controls(record: dict) -> dict[str, float]:
    dynamic_object = record["conditioning"]["physics"]["object"]
    return {
        "mass_kg": float(dynamic_object["mass_kg"]),
        "contact_friction": float(dynamic_object["friction"]),
        "contact_restitution": float(dynamic_object["restitution"]),
    }


def _initial_state(record: dict) -> tuple[float, ...]:
    physics = record["conditioning"]["physics"]
    state = physics["object"]["initial_state"]
    return tuple(
        float(value)
        for vector in (
            state["linear_velocity_camera_m_s"],
            state["angular_velocity_camera_rad_s"],
            physics["world"]["gravity_camera_m_s2"],
        )
        for value in vector
    )


def _secondary_dynamics(record: dict) -> tuple[float, ...]:
    dynamic_object = record["conditioning"]["physics"]["object"]
    return tuple(
        float(dynamic_object[key])
        for key in (
            "rolling_friction",
            "spinning_friction",
            "linear_damping",
            "angular_damping",
        )
    )


def _inertia_per_mass(record: dict) -> tuple[float, ...]:
    dynamic_object = record["conditioning"]["physics"]["object"]
    mass = float(dynamic_object["mass_kg"])
    return tuple(
        float(value) / mass
        for row in dynamic_object["inertia_tensor_camera_kg_m2"]
        for value in row
    )


def _audit_groups(
    records: list[dict],
    ) -> tuple[list[str], dict[str, str], dict[str, dict[str, int]]]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["base_scene_id"]].append(record)
    errors = []
    bindings = {}
    target_videos = set()
    base_level_counts = {axis: defaultdict(int) for axis in SWEEP_AXES}
    common_fields = (
        "first_frame",
        "scene",
        "text",
    )
    for scene_id, items in grouped.items():
        bases = [item for item in items if item["sweep"]["mode"] == "base"]
        if len(bases) != 1:
            errors.append(f"base group does not contain one center sample: {scene_id}")
            continue
        base = bases[0]
        if len(items) != sweep_group_size(1):
            errors.append(
                "base group does not contain "
                f"{sweep_group_size(1)} samples: {scene_id}"
            )
        reference = base["conditioning"]
        reference_state = _initial_state(base)
        reference_secondary_dynamics = _secondary_dynamics(base)
        reference_inertia_per_mass = _inertia_per_mass(base)
        for item in items:
            for field in common_fields:
                if item["conditioning"].get(field) != reference.get(field):
                    errors.append(f"{field} changes inside base group: {scene_id}")
                    break
            if _initial_state(item) != reference_state:
                errors.append(f"initial physical state changes inside base group: {scene_id}")
            if _secondary_dynamics(item) != reference_secondary_dynamics:
                errors.append(f"secondary dynamics change inside base group: {scene_id}")
            if not all(
                math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-10)
                for actual, expected in zip(
                    _inertia_per_mass(item), reference_inertia_per_mass
                )
            ):
                errors.append(f"inertia is not mass-scaled inside base group: {scene_id}")
            video = item["target"]["video"]
            if video in target_videos:
                errors.append(f"target video is reused: {video}")
            target_videos.add(video)
        base_controls = _controls(base)
        for axis in SWEEP_AXES:
            base_level = int(base["sweep"]["base_level_indices"][axis])
            base_level_counts[axis][str(base_level)] += 1
            if base_level != 2:
                errors.append(
                    f"{axis} base level must be the middle index 2: "
                    f"{scene_id} {base_level}"
                )
            variants = [
                item
                for item in items
                if item["sweep"]["mode"] == "one_factor"
                and item["sweep"]["axis"] == axis
            ]
            levels = sorted(int(item["sweep"]["level_index"]) for item in variants)
            expected_levels = sorted(set(range(5)) - {base_level})
            if levels != expected_levels:
                errors.append(
                    f"{axis} levels do not exclude declared base level "
                    f"{base_level}: {scene_id} {levels}"
                )
                continue
            indexed = [
                (int(item["sweep"]["level_index"]), item) for item in variants
            ]
            indexed.append((base_level, base))
            ordered = [item for _, item in sorted(indexed)]
            values = [_controls(item)[axis] for item in ordered]
            if len(set(values)) != 5 or any(left >= right for left, right in zip(values, values[1:])):
                errors.append(
                    f"{axis} values are not five distinct increasing levels: "
                    f"{scene_id} {values}"
                )
            for variant in variants:
                controls = _controls(variant)
                for fixed_axis in SWEEP_AXES:
                    if fixed_axis == axis:
                        continue
                    if not math.isclose(
                        controls[fixed_axis],
                        base_controls[fixed_axis],
                        rel_tol=1e-9,
                        abs_tol=1e-9,
                    ):
                        errors.append(
                            f"{axis} sweep changes {fixed_axis}: {variant['sample_id']}"
                        )
        bindings[reference["scene"]] = scene_id
    return errors, bindings, {
        axis: dict(sorted(counts.items())) for axis, counts in base_level_counts.items()
    }


def audit(dataset_root: Path, project_root: Path, forbid_approximations: bool) -> dict:
    project_root = project_root.resolve()
    dataset_root = dataset_root.resolve()
    summary_path = dataset_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("path_base") != "physweep_project_root":
        raise ValueError(
            "dataset summary must declare project-root-relative published paths"
        )
    manifest_path = project_root / summary["manifest"]
    errors = []
    if _sha256(manifest_path) != summary["manifest_sha256"]:
        errors.append("manifest hash mismatch")
    validation = validate_manifest(manifest_path, project_root, check_files=True)
    records = list(iter_jsonl(manifest_path))
    group_errors, scene_bindings, base_level_distribution = _audit_groups(records)
    errors.extend(group_errors)

    referenced_scenes = {
        record["conditioning"]["scene"] for record in records
    }
    scene_index_path = project_root / summary["scene_condition"]["index"]
    if _sha256(scene_index_path) != summary["scene_condition"]["index_sha256"]:
        errors.append("scene index hash mismatch")
    scene_index = json.loads(scene_index_path.read_text(encoding="utf-8"))
    if scene_index.get("schema") != "physweep.model_scene_condition_index.v1":
        errors.append("unexpected scene index schema")
    if summary["scene_condition"].get("schema") != MODEL_SCENE_SCHEMA:
        errors.append("unexpected scene condition schema in summary")
    indexed_scenes = {record["path"] for record in scene_index["records"]}
    if indexed_scenes != referenced_scenes:
        errors.append("scene index and training manifest reference different scenes")

    approximations = []
    for record in scene_index["records"]:
        path = project_root / record["path"]
        if _sha256(path) != record["sha256"]:
            errors.append(f"scene hash mismatch: {record['scene_id']}")
            continue
        inspection = inspect_model_scene_condition(path)
        if inspection["schema"] != MODEL_SCENE_SCHEMA:
            errors.append(f"scene schema mismatch: {record['scene_id']}")
        if inspection["point_count"] != summary["scene_condition"]["point_count"]:
            errors.append(f"point count mismatch: {record['scene_id']}")
        expected_object = int(summary["scene_condition"]["object_points"])
        expected_environment = int(summary["scene_condition"]["environment_points"])
        if inspection["object_point_count"] != expected_object:
            errors.append(f"object point quota mismatch: {record['scene_id']}")
        if inspection["environment_point_count"] != expected_environment:
            errors.append(f"environment point quota mismatch: {record['scene_id']}")
        binding = scene_bindings.get(record["path"])
        if binding is None:
            errors.append(f"scene has no training binding: {record['scene_id']}")
        elif inspection["scene_id"] != binding:
            errors.append(f"scene id binding mismatch: {record['scene_id']}")
    if forbid_approximations and approximations:
        errors.append("scene conditions contain geometry approximations")
    return {
        "schema": "physweep.training_dataset_audit.v2",
        "passed": not errors,
        "dataset_root": str(dataset_root),
        **validation,
        "scene_count": len(indexed_scenes),
        "group_contract": {
            "samples_per_base": sweep_group_size(1),
            "base_samples": 1,
            "nonbase_samples_per_axis": 4,
            "shared_first_frame_and_scene": True,
            "one_factor_isolation": True,
            "base_level_policy": "middle_index_2_required",
            "base_level_distribution": base_level_distribution,
        },
        "approximation_scene_count": len(approximations),
        "approximations": approximations,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit compiled PhysSweep training data")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--forbid-approximations", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.dataset_root
    if not root.is_absolute():
        root = args.project_root / root
    result = audit(root, args.project_root, args.forbid_approximations)
    output = (
        (args.project_root / args.output).resolve()
        if args.output is not None and not args.output.is_absolute()
        else args.output.resolve()
        if args.output is not None
        else root.resolve() / "audit.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
