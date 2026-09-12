"""Compatibility entry and two-object admission over shared sphere execution."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
from tools.physics.specialized_sphere_simulation import (
    _pybullet, _configure_world, _create_spheres,
    _billiards_fixture, _pinball_fixture, _marble_fixture,
    simulate_specialized_spheres,
)

def simulate_two_object_specialized(
    scene: dict[str, Any], root: Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run one two-sphere scene and audit its declared interaction semantics."""

    if len(scene["objects"]) != 2:
        raise ValueError("specialized two-object simulation requires two objects")
    adapter = str(scene["backend_binding"]["adapter_id"])
    fixture_builder = {
        "billiards_two_object_v1": _billiards_fixture,
        "passive_pinball_two_object_v1": _pinball_fixture,
        "marble_run_two_object_v1": _marble_fixture,
    }.get(adapter)
    if fixture_builder is None:
        raise ValueError(f"unsupported specialized two-object adapter: {adapter}")
    arrays, execution = simulate_specialized_spheres(
        scene, root, fixture_builder=fixture_builder, pybullet_factory=_pybullet,
        configure_world=_configure_world, create_spheres=_create_spheres,
    )
    simulation_hz = int(scene["time"]["simulation_hz"])
    positions = arrays["position_m"]; orientations = arrays["adapter__quaternion_xyzw"]
    linear = arrays["linear_velocity_m_s"]; angular = arrays["angular_velocity_rad_s"]
    bodies = scene["objects"]
    solver_execution = execution["solver_execution"]
    contact_execution = execution["contact_processing_execution"]
    first_pair_step = execution["first_pair_step"]; first_rail_step = execution["first_rail_step"]
    minimum_contact_distance = execution["minimum_contact_distance"]
    maximum_speed = execution["maximum_speed"]
    path_lengths = execution["path_lengths"]; touched = execution["touched"]

    payload = scene["adapter_payload"]
    quality = payload.get("quality", payload.get("two_object_quality"))
    if not isinstance(quality, dict):
        raise ValueError("specialized two-object quality contract is missing")
    first_pair_time = (
        None if first_pair_step is None else first_pair_step / float(simulation_hz)
    )
    checks = {
        "finite_trajectory": all(
            np.isfinite(value).all()
            for value in (positions, orientations, linear, angular)
        ),
        "two_dynamic_objects": len(bodies) == 2,
        "pair_contact_observed": first_pair_step is not None,
        "first_pair_contact_within_limit": first_pair_time is not None
        and first_pair_time <= float(quality["maximum_first_pair_contact_time_s"]),
        "maximum_penetration": -minimum_contact_distance
        <= float(quality["maximum_penetration_m"]),
    }
    if adapter == "billiards_two_object_v1":
        checks["rail_contact_before_pair_contact_absent"] = not bool(
            quality.get("rail_contact_before_pair_contact_is_forbidden", False)
        ) or first_rail_step is None or (
            first_pair_step is not None and first_rail_step >= first_pair_step
        )
    elif adapter == "passive_pinball_two_object_v1":
        checks["minimum_path_length_per_object"] = bool(
            np.all(path_lengths >= float(quality["minimum_path_length_per_object_m"]))
        )
        checks["minimum_distinct_fixture_contacts_per_object"] = all(
            len(records)
            >= int(quality["minimum_distinct_fixture_contacts_per_object"])
            for records in touched
        )
    else:
        required = set(str(value) for value in quality["required_track_contact_ids"])
        checks["minimum_path_length_per_object"] = bool(
            np.all(path_lengths >= float(quality["minimum_path_length_per_object_m"]))
        )
        checks["required_track_contacts_per_object"] = all(
            required <= records for records in touched
        )
    # Contact, travel and fixture-order requirements admit bases only.
    outcome_checks = {
        "pair_contact_observed", "first_pair_contact_within_limit",
        "rail_contact_before_pair_contact_absent", "minimum_path_length_per_object",
        "minimum_distinct_fixture_contacts_per_object", "required_track_contacts_per_object",
    }
    advisories = [
        name for name, passed in checks.items()
        if not passed and name in outcome_checks and scene["variant"]["kind"] == "sweep"
    ]
    audit = {
        "schema_version": "physweep_two_object_specialized_audit_v1",
        "solver_execution": solver_execution,
        "contact_processing_execution": contact_execution,
        "contact_processing_evidence": "pybullet_changeDynamics_arguments",
        "passed": all(passed or name in advisories for name, passed in checks.items()),
        "checks": checks,
        "advisories": advisories,
        "contact_count_sampling": "maximum_over_preceding_output_interval; frame_zero_initial_state",
        "runtime_material_extras_fields": [
            "rolling_friction", "spinning_friction", "linear_damping", "angular_damping",
        ],
        "runtime_material_extras_evidence": {
            "rolling_friction": "pybullet_getDynamicsInfo",
            "spinning_friction": "pybullet_getDynamicsInfo",
            "linear_damping": "pybullet_changeDynamics_argument",
            "angular_damping": "pybullet_changeDynamics_argument",
        },
        "metrics": {
            "first_pair_contact_time_s": (
                None if first_pair_time is None else round(first_pair_time, 9)
            ),
            "maximum_penetration_m": round(-minimum_contact_distance, 9),
            "path_length_m": [round(float(value), 9) for value in path_lengths],
            "maximum_linear_speed_m_s": round(maximum_speed, 9),
            "fixture_contacts": [sorted(records) for records in touched],
        },
    }
    return arrays, audit
