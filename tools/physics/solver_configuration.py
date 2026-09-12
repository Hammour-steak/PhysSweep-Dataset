"""One solver contract for simulation, execution evidence and release metadata."""

from __future__ import annotations

import math
from typing import Any, Mapping


TWO_OBJECT_ADAPTERS = frozenset({
    "billiards_two_object_v1", "passive_pinball_two_object_v1",
    "marble_run_two_object_v1",
})
SPECIALIZED_SPHERE_ADAPTERS = TWO_OBJECT_ADAPTERS | {
    'billiards_three_object_v1', 'passive_pinball_three_object_v1',
    'marble_run_three_object_v1',
}
SOLVER_ARGUMENTS = {
    "solver_iterations": "numSolverIterations",
    "deterministic_overlapping_pairs": "deterministicOverlappingPairs",
    "restitution_velocity_threshold_m_s": "restitutionVelocityThreshold",
    "enable_cone_friction": "enableConeFriction",
    "use_split_impulse": "useSplitImpulse",
    "contact_breaking_threshold_m": "contactBreakingThreshold",
    "solver_residual_threshold": "solverResidualThreshold",
}
BOOLEAN_FIELDS = frozenset({
    "deterministic_overlapping_pairs", "enable_cone_friction", "use_split_impulse",
})


def normalize_solver(values: Mapping[str, Any], *, generic: bool = False) -> dict[str, Any]:
    values = dict(values)
    if "iterations" in values:
        if "solver_iterations" in values:
            raise ValueError("both solver iteration field names are present")
        values["solver_iterations"] = values.pop("iterations")
    values.setdefault("deterministic_overlapping_pairs", True)
    if generic:
        values.setdefault("enable_cone_friction", True)
        values.setdefault("use_split_impulse", True)
        # Preserve the generic solver's historical disabled early-exit setting.
        values.setdefault("solver_residual_threshold", 0.0)
    required = set(SOLVER_ARGUMENTS) - {"contact_breaking_threshold_m", "solver_residual_threshold"}
    if not required.issubset(values):
        raise ValueError("resolved solver contract is incomplete")
    result = {key: values[key] for key in SOLVER_ARGUMENTS if key in values}
    iterations = result["solver_iterations"]
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("solver_iterations must be a positive integer")
    for key in BOOLEAN_FIELDS:
        if not isinstance(result[key], bool):
            raise ValueError(f"{key} must be boolean")
    for key in ("restitution_velocity_threshold_m_s", "contact_breaking_threshold_m", "solver_residual_threshold"):
        if key in result:
            value = result[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
            result[key] = float(value)
    return result


def resolved_solver_configuration(
    adapter_id: str, source: Mapping[str, Any], scene: Mapping[str, Any],
) -> dict[str, Any]:
    if adapter_id == "generic_rigid_v1":
        return normalize_solver(source["simulation"]["solver"], generic=True)
    if adapter_id not in SPECIALIZED_SPHERE_ADAPTERS:
        raise ValueError(f"unsupported shared solver adapter: {adapter_id}")
    backend = scene["adapter_payload"]["backend"]
    if adapter_id in {"billiards_two_object_v1", "billiards_three_object_v1"}:
        return normalize_solver(backend["billiards_rules"]["engine"])
    declared = normalize_solver(source["physics"]["engine"])
    # Backend constants and candidate engine values must describe the same run.
    # Marble backends explicitly inherit these fields and contain no constants.
    backend_physics = backend.get("physics", {})
    for key, value in declared.items():
        if key in backend_physics and backend_physics[key] != value:
            raise ValueError(f"conflicting source/backend solver configuration: {key}")
    return declared


def solver_arguments(configuration: Mapping[str, Any]) -> dict[str, Any]:
    return {
        argument: int(configuration[key]) if key in BOOLEAN_FIELDS else configuration[key]
        for key, argument in SOLVER_ARGUMENTS.items() if key in configuration
    }


def apply_solver_configuration(
    pb: Any, configuration: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = solver_arguments(configuration)
    pb.setPhysicsEngineParameter(**parameters)
    reported = pb.getPhysicsEngineParameters()
    if reported["numSolverIterations"] != parameters["numSolverIterations"]:
        raise ValueError("PyBullet solver iterations differ from applied configuration")
    return {
        "pybullet_arguments": parameters,
        "reported_parameters": reported,
        "evidence": "setPhysicsEngineParameter arguments; available values read back with getPhysicsEngineParameters",
    }


def solver_execution_matches(configuration: Mapping[str, Any], execution: Any) -> bool:
    if not isinstance(execution, dict):
        return False
    arguments = execution.get("pybullet_arguments", {})
    reported = execution.get("reported_parameters", {})
    return (
        all(arguments.get(key) == value for key, value in solver_arguments(configuration).items())
        and reported.get("numSolverIterations") == configuration["solver_iterations"]
    )
