"""Pure resolution of nominal and effective integration time grids."""
from __future__ import annotations
import math
from typing import Any, Mapping


def generic_integration_configuration(source: Mapping[str, Any]) -> dict[str, Any]:
    simulation = source["simulation"]
    time = simulation["time"]
    nominal = time["simulation_hz"]
    fps = time["output_fps"]
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (nominal, fps)) or nominal % fps:
        raise ValueError("integration frequencies must be positive integers and simulation_hz divisible by output_fps")
    exact = simulation.get("support", {}).get("exact_static_binding")
    ballistic = any(obj.get("expected_motion", {}).get("contact_mode") == "ballistic_then_contact"
                    for obj in simulation["objects"])
    factor = 2 if exact is not None and ballistic else 1
    actual = nominal * factor
    return {"nominal_simulation_hz": nominal, "substep_factor": factor,
            "simulation_hz": actual, "steps_per_frame": actual // fps,
            "fixed_time_step_s": 1.0 / actual}




def integration_execution_matches(expected: Mapping[str, Any], execution: Any, solver: Any) -> bool:
    if execution != expected or not isinstance(solver, dict):
        return False
    reported = solver.get("reported_parameters", {})
    value = reported.get("fixedTimeStep")
    return (isinstance(value, (int, float)) and math.isfinite(value)
            and math.isclose(value, expected["fixed_time_step_s"], rel_tol=1.e-12, abs_tol=0.)
            and reported.get("numSubSteps") == 0)
