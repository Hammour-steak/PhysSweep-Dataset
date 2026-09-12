#!/usr/bin/env python3
"""Dispatch a resolved PhysSweep scene to its reviewed PyBullet adapter."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from tools.core.hashing import sha256_file as sha256
from tools.core.rigid_geometry import generic_collision_proxy
from tools.core.json_io import write_json_atomic as write_json
from tools.physics.generate_billiards_scene import simulate as simulate_billiards
from tools.physics.generate_marble_run_scene import simulate as simulate_marble_run
from tools.physics.generate_passive_pinball_scene import simulate as simulate_passive_pinball
from tools.physics.resolved_simulation_scene import (
    compile_resolved_scene, TWO_OBJECT_MATERIAL_EXTRAS,
)
from tools.physics.asset_proxy_simulation import simulate_scene as simulate_asset_proxy
from tools.physics.simulate_pybullet_rigid import simulate as simulate_generic_rigid
from tools.physics.two_object_specialized_simulation import (
    simulate_two_object_specialized,
)
from tools.physics.three_object_specialized_simulation import simulate_three_object_specialized
from tools.core.integration_configuration import generic_integration_configuration, integration_execution_matches
from tools.physics.contact_configuration import contact_processing_execution_matches
from tools.physics.solver_configuration import (
    TWO_OBJECT_ADAPTERS, SPECIALIZED_SPHERE_ADAPTERS, resolved_solver_configuration, solver_execution_matches,
)

DISPATCH_RECORD_VERSION = "physweep_dispatched_simulation_record_v1"
TRAJECTORY_LAYOUT_VERSION = "physweep_object_trajectory_v2"


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.stem}.tmp-{os.getpid()}-{time.time_ns()}.npz"
    )
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _xyzw_to_wxyz(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[..., [3, 0, 1, 2]]


def _generic(scene: dict[str, Any], *, root: Path = Path(__file__).resolve().parents[2]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    source = copy.deepcopy(scene["source_metadata"])
    source_objects = source["simulation"]["objects"]
    if len(source_objects) != len(scene["objects"]):
        raise ValueError("resolved and generic source object counts differ")
    for source_object, resolved_object in zip(source_objects, scene["objects"]):
        if source_object["object_id"] != resolved_object["object_id"]:
            raise ValueError("resolved and generic source object order differs")
        if resolved_object["collision_proxy"] != generic_collision_proxy(source_object):
            raise ValueError("resolved generic collision proxy differs from execution geometry")
        source_object["material"].update(copy.deepcopy(resolved_object["material"]))
    trajectory, audit = simulate_generic_rigid(source, root=root)
    objects = scene["objects"]
    position = np.stack(
        [trajectory[f"{obj['object_id']}__position_m"] for obj in objects], axis=1
    )
    orientation = np.stack(
        [trajectory[f"{obj['object_id']}__quaternion_wxyz"] for obj in objects], axis=1
    )
    linear = np.stack(
        [trajectory[f"{obj['object_id']}__linear_velocity_m_s"] for obj in objects],
        axis=1,
    )
    angular = np.stack(
        [trajectory[f"{obj['object_id']}__angular_velocity_rad_s"] for obj in objects],
        axis=1,
    )
    contact = np.stack(
        [trajectory[f"{obj['object_id']}__all_contact_count"] for obj in objects],
        axis=1,
    )
    runtime_material = np.stack(
        [trajectory[f"{obj['object_id']}__runtime_dynamics"][:3] for obj in objects],
        axis=0,
    )
    inertia = np.stack(
        [
            trajectory[f"{obj['object_id']}__runtime_inertia_diagonal_kg_m2"]
            for obj in objects
        ],
        axis=0,
    )
    normalized = {
        "time_s": trajectory["time_s"],
        "position_m": position,
        "quaternion_wxyz": orientation,
        "linear_velocity_m_s": linear,
        "angular_velocity_rad_s": angular,
        "contact_count": contact,
        "runtime_material": runtime_material,
        "inertia_diagonal_kg_m2": inertia,
    }
    render_suffixes = {
        "aabb_min_m",
        "aabb_max_m",
        "primary_support_contact_count",
        "all_contact_count",
        "minimum_contact_distance_m",
    }
    for key, value in trajectory.items():
        if "__" not in key:
            continue
        suffix = key.split("__", 1)[1]
        if (
            suffix in render_suffixes
            or suffix.startswith("collider_contact_count__")
            or suffix.startswith("object_contact_count__")
        ):
            normalized[f"adapter__{key}"] = value
    return normalized, audit


def _asset(scene: dict[str, Any], root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if len(scene["objects"]) != 1:
        raise ValueError("asset proxy v3 adapter currently supports one dynamic object")
    obj = scene["objects"][0]
    payload = scene["adapter_payload"]
    dynamic = copy.deepcopy(payload["dynamic_record"])
    dynamic.setdefault("proxy", {}).setdefault("material", {})["friction"] = obj[
        "material"
    ]["contact_friction"]
    dynamic["proxy"]["material"]["restitution"] = obj["material"][
        "contact_restitution"
    ]
    arrays, audit = simulate_asset_proxy(
        root,
        dynamic,
        payload["static_support_binding"],
        payload["static_prop_record"],
        payload["static_prop_binding"],
        obj["initial_state"],
        float(obj["material"]["mass_kg"]),
        float(scene["time"]["duration_s"]),
        int(scene["time"]["output_fps"]),
        payload["motion_profile"],
        payload["backend"],
        payload["expected_motion"],
    )
    contact = (
        np.asarray(arrays["support_contact"], dtype=np.int32)
        + np.asarray(arrays["ground_contact"], dtype=np.int32)
        + np.asarray(arrays["prop_contact"], dtype=np.int32)
    )[:, None]
    normalized = {
        "time_s": arrays["time_s"],
        "position_m": arrays["position_m"][:, None, :],
        "quaternion_wxyz": _xyzw_to_wxyz(arrays["quaternion_xyzw"])[:, None, :],
        "linear_velocity_m_s": arrays["linear_velocity_m_s"][:, None, :],
        "angular_velocity_rad_s": arrays["angular_velocity_rad_s"][:, None, :],
        "contact_count": contact,
        "runtime_material": np.asarray(arrays["runtime_dynamics"][:3], dtype=np.float64)[
            None, :
        ],
        "inertia_diagonal_kg_m2": arrays["runtime_inertia_diagonal_kg_m2"][None, :],
    }
    for key, value in arrays.items():
        normalized[f"adapter__{key}"] = value
    return normalized, audit


def _billiards(scene: dict[str, Any], root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    payload = scene["adapter_payload"]
    backend = copy.deepcopy(payload["backend"])
    objects = scene["objects"]
    materials = [copy.deepcopy(obj["material"]) for obj in objects]
    initial = [
        {
            "object_id": obj["object_id"],
            "position_m": obj["initial_state"]["position_m"],
            "velocity_m_s": obj["initial_state"]["linear_velocity_m_s"],
        }
        for obj in objects
    ]
    arrays, audit, simulated_initial = simulate_billiards(
        root,
        payload["static_support_binding"],
        float(scene["time"]["duration_s"]),
        int(scene["time"]["output_fps"]),
        payload["profile"],
        backend,
        initial=initial,
        object_materials=materials,
    )
    if simulated_initial != initial:
        raise RuntimeError("billiards adapter changed the initial state")
    normalized = {
        "time_s": arrays["time_s"],
        "position_m": arrays["position_m"],
        "quaternion_wxyz": _xyzw_to_wxyz(arrays["quaternion_xyzw"]),
        "linear_velocity_m_s": arrays["linear_velocity_m_s"],
        "angular_velocity_rad_s": arrays["angular_velocity_rad_s"],
        "contact_count": arrays["contact_count"],
        "runtime_material": arrays["runtime_material"],
        "inertia_diagonal_kg_m2": arrays["runtime_inertia_diagonal_kg_m2"],
    }
    for key, value in arrays.items():
        normalized[f"adapter__{key}"] = value
    return normalized, audit


def _single_sphere_fixture(
    scene: dict[str, Any],
    root: Path,
    *,
    label: str,
    simulator: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if len(scene["objects"]) != 1:
        raise ValueError(f"{label} adapter requires exactly one object")
    source = copy.deepcopy(scene["source_metadata"])
    source_object = source["simulation"]["objects"][0]
    resolved_object = scene["objects"][0]
    if source_object["object_id"] != resolved_object["object_id"]:
        raise ValueError(f"{label} source and resolved object ids differ")
    source_object["material"].update(copy.deepcopy(resolved_object["material"]))
    arrays, audit = simulator(root, source)
    normalized = {
        "time_s": arrays["time_s"],
        "position_m": arrays["position_m"],
        "quaternion_wxyz": _xyzw_to_wxyz(arrays["quaternion_xyzw"]),
        "linear_velocity_m_s": arrays["linear_velocity_m_s"],
        "angular_velocity_rad_s": arrays["angular_velocity_rad_s"],
        "contact_count": arrays["contact_count"],
        "runtime_material": arrays["runtime_material"],
        "inertia_diagonal_kg_m2": arrays["runtime_inertia_diagonal_kg_m2"],
    }
    for key, value in arrays.items():
        normalized[f"adapter__{key}"] = value
    return normalized, audit


def _passive_pinball(
    scene: dict[str, Any], root: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    return _single_sphere_fixture(
        scene,
        root,
        label="passive-pinball",
        simulator=simulate_passive_pinball,
    )


def _marble_run(
    scene: dict[str, Any], root: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    return _single_sphere_fixture(
        scene,
        root,
        label="marble-run",
        simulator=simulate_marble_run,
    )


def _is_two_object_sweep(scene: dict[str, Any]) -> bool:
    return (
        scene.get("variant", {}).get("kind") == "sweep"
        and len(scene.get("objects", [])) == 2
        and scene["backend_binding"]["adapter_id"] in {
            "generic_rigid_v1", "billiards_two_object_v1",
            "passive_pinball_two_object_v1", "marble_run_two_object_v1",
        }
    )


def _adapter_hard_results(
    scene: dict[str, Any], adapter_audit: dict[str, Any]
) -> list[bool]:
    adapter_id = scene["backend_binding"]["adapter_id"]
    records = adapter_audit.get("checks", {})
    if adapter_id in {'generic_rigid_v1', 'billiards_three_object_v1', 'passive_pinball_three_object_v1', 'marble_run_three_object_v1'} and len(scene['objects']) == 3:
        from tools.motion_rules.three_object.interaction import audit_hard_results
        return audit_hard_results(scene['objects'], adapter_audit, scene['variant']['kind'] == 'sweep')
    if _is_two_object_sweep(scene):
        # Base admission owns trajectory quality. Derived outcomes are retained
        # as raw diagnostics; only execution/metadata integrity can stop a sweep.
        if adapter_id == "generic_rigid_v1":
            if not isinstance(records, list):
                raise ValueError("generic rigid adapter returned invalid audit checks")
            by_id = {record["id"]: record for record in records}
            if len(by_id) != len(records):
                raise ValueError("generic rigid adapter returned duplicate audit checks")
            integrity_checks = (
                "finite_state", "initial_position_matches_metadata",
                "initial_linear_velocity_matches_metadata",
                "pybullet_dynamics_match_metadata",
                "pybullet_support_dynamics_match_metadata",
                "runtime_inertia_is_finite_and_positive",
                "collision_proxy_matches_definition",
            )
            return [
                bool(by_id.get(f"{obj['object_id']}__{name}", {}).get("passed"))
                for obj in scene["objects"] for name in integrity_checks
            ]
        if not isinstance(records, dict):
            raise ValueError(f"{adapter_id} adapter returned invalid audit checks")
        return [bool(records.get(name)) for name in ("finite_trajectory", "two_dynamic_objects")]
    if adapter_id == "generic_rigid_v1":
        return [
            bool(record.get("passed")) or record.get("severity") == "advisory"
            for record in records
        ]
    if not isinstance(records, dict):
        raise ValueError(f"{adapter_id} adapter returned invalid audit checks")
    if adapter_id in {
        "billiards_two_object_v1",
        "passive_pinball_two_object_v1",
        "marble_run_two_object_v1",
    }:
        advisories = adapter_audit.get("advisories", []) if scene["variant"]["kind"] == "sweep" else []
        return [bool(passed) or name in advisories for name, passed in records.items()]
    if adapter_id in {"passive_pinball_v1", "marble_run_v1"}:
        return [bool(passed) for passed in records.values()]
    hard_exact = {
        "finite_trajectory",
        "initial_penetration_within_limit",
        "no_initial_prop_overlap",
        "no_unplanned_static_prop_contact",
        "penetration_within_limit",
        "linear_speed_within_limit",
        "rotational_surface_speed_within_limit",
        "world_bounds_valid",
        "drop_lateral_drift_within_limit",
        "balls_remain_on_or_above_bed",
        "balls_remain_inside_rails",
        "no_unforced_speed_gain",
        "vertical_motion_matches_initial_contact_clearance",
    }
    hard_tokens = (
        "runtime_",
        "parameter_match",
        "proxy_",
        "inertia",
        "penetration",
        "unforced_speed_gain",
        "vertical_motion_matches",
        "remain_on_or_above",
        "remain_inside",
    )
    return [
        bool(passed)
        for name, passed in records.items()
        if name in hard_exact or any(token in str(name) for token in hard_tokens)
    ]


def _common_audit(
    scene: dict[str, Any], trajectory: dict[str, np.ndarray], adapter_audit: dict[str, Any]
) -> dict[str, Any]:
    object_count = len(scene["objects"])
    frame_count = int(scene["time"]["frame_count"])
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, value: Any, expected: Any) -> None:
        checks.append({"id": name, "passed": bool(passed), "value": value, "expected": expected})

    adapter = scene["backend_binding"]["adapter_id"]
    if adapter == "generic_rigid_v1" or adapter in SPECIALIZED_SPHERE_ADAPTERS:
        expected_solver = resolved_solver_configuration(adapter, scene.get("source_metadata", {}), scene)
        execution = adapter_audit.get("solver_execution")
        check("runtime_solver_configuration_exact", solver_execution_matches(expected_solver, execution), execution, expected_solver)

    if adapter == "generic_rigid_v1":
        integration = generic_integration_configuration(scene["source_metadata"])
        observed = adapter_audit.get("integration_execution")
        matches = (scene["time"]["simulation_hz"] == integration["simulation_hz"]
                   and integration_execution_matches(integration, observed, adapter_audit.get("solver_execution")))
        check("runtime_integration_configuration_exact", matches, observed, integration)

    if adapter in SPECIALIZED_SPHERE_ADAPTERS:
        execution = adapter_audit.get("contact_processing_execution")
        check("runtime_contact_processing_exact", contact_processing_execution_matches(scene, execution),
              execution, "applied threshold for every dynamic object matches resolved configuration")

    position = np.asarray(trajectory["position_m"], dtype=np.float64)
    orientation = np.asarray(trajectory["quaternion_wxyz"], dtype=np.float64)
    linear = np.asarray(trajectory["linear_velocity_m_s"], dtype=np.float64)
    angular = np.asarray(trajectory["angular_velocity_rad_s"], dtype=np.float64)
    time_s = np.asarray(trajectory["time_s"], dtype=np.float64)
    contact = np.asarray(trajectory["contact_count"])
    expected_shape = (frame_count, object_count, 3)
    expected_time = np.arange(frame_count, dtype=np.float64) / float(
        scene["time"]["output_fps"]
    )
    time_error = (
        float(np.max(np.abs(time_s - expected_time)))
        if time_s.shape == expected_time.shape
        else float("inf")
    )
    check("time_axis_exact", time_error <= 1.0e-9, time_error, 1.0e-9)
    check("position_shape", position.shape == expected_shape, list(position.shape), list(expected_shape))
    check("linear_velocity_shape", linear.shape == expected_shape, list(linear.shape), list(expected_shape))
    check("angular_velocity_shape", angular.shape == expected_shape, list(angular.shape), list(expected_shape))
    check(
        "orientation_shape",
        orientation.shape == (frame_count, object_count, 4),
        list(orientation.shape),
        [frame_count, object_count, 4],
    )
    check(
        "finite_state",
        all(
            np.isfinite(value).all()
            for value in (time_s, position, orientation, linear, angular)
        ),
        None,
        True,
    )
    check(
        "contact_count_shape",
        contact.shape == (frame_count, object_count),
        list(contact.shape),
        [frame_count, object_count],
    )
    check(
        "contact_count_valid",
        np.isfinite(contact).all()
        and bool(np.all(contact >= 0))
        and bool(np.all(contact == np.floor(contact))),
        None,
        "finite non-negative integers",
    )
    quaternion_norm_error = (
        float(np.max(np.abs(np.linalg.norm(orientation, axis=2) - 1.0)))
        if orientation.shape == (frame_count, object_count, 4)
        else float("inf")
    )
    check(
        "orientation_unit_norm",
        quaternion_norm_error <= 1.0e-6,
        quaternion_norm_error,
        1.0e-6,
    )
    if position.shape == expected_shape:
        expected_position = np.asarray(
            [obj["initial_state"]["position_m"] for obj in scene["objects"]],
            dtype=np.float64,
        )
        expected_linear = np.asarray(
            [obj["initial_state"]["linear_velocity_m_s"] for obj in scene["objects"]],
            dtype=np.float64,
        )
        expected_angular = np.asarray(
            [obj["initial_state"]["angular_velocity_rad_s"] for obj in scene["objects"]],
            dtype=np.float64,
        )
        expected_orientation = []
        for obj in scene["objects"]:
            initial = obj["initial_state"]
            if "orientation_quaternion_wxyz" in initial:
                expected_orientation.append(initial["orientation_quaternion_wxyz"])
            else:
                x, y, z, w = initial["orientation_quaternion_xyzw"]
                expected_orientation.append([w, x, y, z])
        expected_orientation_array = np.asarray(expected_orientation, dtype=np.float64)
        position_error = float(np.max(np.abs(position[0] - expected_position)))
        velocity_error = float(np.max(np.abs(linear[0] - expected_linear)))
        angular_error = float(np.max(np.abs(angular[0] - expected_angular)))
        orientation_error = float(
            np.max(
                np.minimum(
                    np.linalg.norm(orientation[0] - expected_orientation_array, axis=1),
                    np.linalg.norm(orientation[0] + expected_orientation_array, axis=1),
                )
            )
        )
        check("initial_position_exact", position_error <= 1.0e-7, position_error, 1.0e-7)
        check("initial_velocity_exact", velocity_error <= 1.0e-7, velocity_error, 1.0e-7)
        check("initial_angular_velocity_exact", angular_error <= 1.0e-7, angular_error, 1.0e-7)
        check("initial_orientation_exact", orientation_error <= 1.0e-7, orientation_error, 1.0e-7)
    runtime_material = np.asarray(trajectory["runtime_material"], dtype=np.float64)
    expected_material = np.asarray(
        [
            [
                obj["material"]["mass_kg"],
                obj["material"]["contact_friction"],
                obj["material"]["contact_restitution"],
            ]
            for obj in scene["objects"]
        ],
        dtype=np.float64,
    )
    material_error = (
        float(np.max(np.abs(runtime_material - expected_material)))
        if runtime_material.shape == expected_material.shape
        else float("inf")
    )
    check(
        "runtime_material_exact",
        material_error <= 1.0e-7,
        material_error,
        1.0e-7,
    )
    if adapter in SPECIALIZED_SPHERE_ADAPTERS:
        expected_extras = np.asarray([
            [obj["material"][key] for key in TWO_OBJECT_MATERIAL_EXTRAS]
            for obj in scene["objects"]
        ], dtype=np.float64)
        runtime_extras = np.asarray(trajectory.get("runtime_material_extras", []), dtype=np.float64)
        extra_error = (
            float(np.max(np.abs(runtime_extras - expected_extras)))
            if runtime_extras.shape == expected_extras.shape else float("inf")
        )
        check("runtime_material_extras_exact", extra_error <= 1.0e-7, extra_error, 1.0e-7)
    inertia = np.asarray(trajectory["inertia_diagonal_kg_m2"], dtype=np.float64)
    check(
        "runtime_inertia_valid",
        inertia.shape == (object_count, 3)
        and np.isfinite(inertia).all()
        and bool(np.all(inertia > 0.0)),
        {"shape": list(inertia.shape), "minimum": float(np.min(inertia))},
        {"shape": [object_count, 3], "minimum_exclusive": 0.0},
    )

    hard_results = _adapter_hard_results(scene, adapter_audit)
    check(
        "adapter_hard_invariants",
        bool(hard_results) and all(hard_results),
        {"checked": len(hard_results), "failed": hard_results.count(False)},
        {"checked_minimum": 1, "failed": 0},
    )
    return {
        "schema_version": "physweep_dispatch_audit_v1",
        "scene_id": scene["scene_id"],
        "adapter_id": scene["backend_binding"]["adapter_id"],
        "passed": all(record["passed"] for record in checks),
        "checks": checks,
        "adapter_audit": adapter_audit,
        "adapter_audit_passed": bool(adapter_audit.get("passed", False)),
        "adapter_audit_policy": (
            "base_quality_rules_diagnostic_for_two_object_sweep_v1"
            if _is_two_object_sweep(scene) else "diagnostic_for_sweep_semantics"
        ),
    }


def dispatch_simulation(
    metadata_path: Path, output_dir: Path, root: Path
) -> dict[str, Any]:
    """Compile, simulate, normalize, audit, and write one immutable result."""
    started = time.perf_counter()
    metadata_path = metadata_path.resolve()
    metadata_payload = metadata_path.read_bytes()
    metadata_sha256 = hashlib.sha256(metadata_payload).hexdigest()
    metadata = json.loads(metadata_payload)
    scene = compile_resolved_scene(metadata, root, metadata_path)
    if scene["source_metadata_sha256"] != metadata_sha256:
        raise RuntimeError("metadata changed while it was being loaded")
    adapter_id = scene["backend_binding"]["adapter_id"]
    if adapter_id == "generic_rigid_v1":
        trajectory, adapter_audit = _generic(scene, root=root)
    elif adapter_id == "asset_proxy_v3":
        trajectory, adapter_audit = _asset(scene, root)
    elif adapter_id == "billiards_v4":
        trajectory, adapter_audit = _billiards(scene, root)
    elif adapter_id == "passive_pinball_v1":
        trajectory, adapter_audit = _passive_pinball(scene, root)
    elif adapter_id == "marble_run_v1":
        trajectory, adapter_audit = _marble_run(scene, root)
    elif adapter_id in {"billiards_three_object_v1", "passive_pinball_three_object_v1", "marble_run_three_object_v1"}:
        trajectory, adapter_audit = simulate_three_object_specialized(scene, root)
    elif adapter_id in TWO_OBJECT_ADAPTERS:
        trajectory, adapter_audit = simulate_two_object_specialized(scene, root)
    else:
        raise ValueError(f"unsupported adapter: {adapter_id}")
    audit = _common_audit(scene, trajectory, adapter_audit)
    if sha256(metadata_path) != metadata_sha256:
        raise RuntimeError("metadata changed during simulation")
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectory.npz"
    object_ids = np.asarray([obj["object_id"] for obj in scene["objects"]])
    write_npz(
        trajectory_path,
        schema_version=np.asarray(TRAJECTORY_LAYOUT_VERSION),
        object_ids=object_ids,
        **trajectory,
    )
    audit_path = output_dir / "trajectory_audit.json"
    write_json(audit_path, audit)
    resolved_path = output_dir / "resolved_scene.json"
    serializable_scene = copy.deepcopy(scene)
    serializable_scene.pop("source_metadata", None)
    write_json(resolved_path, serializable_scene)
    record = {
        "schema_version": DISPATCH_RECORD_VERSION,
        "scene_id": scene["scene_id"],
        "source_schema_version": scene["source_schema_version"],
        "adapter_id": adapter_id,
        "metadata_path": str(metadata_path),
        "metadata_sha256": metadata_sha256,
        "resolved_scene_path": str(resolved_path),
        "resolved_scene_sha256": sha256(resolved_path),
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": sha256(trajectory_path),
        "audit_path": str(audit_path),
        "audit_sha256": sha256(audit_path),
        "audit_passed": bool(audit["passed"]),
        "adapter_audit_passed": bool(audit["adapter_audit_passed"]),
        "failed_checks": [record["id"] for record in audit["checks"] if not record["passed"]],
        "wall_time_s": round(time.perf_counter() - started, 6),
    }
    write_json(output_dir / "simulation_record.json", record)
    return record
