"""Keep intervention outcomes while rejecting corrupt two-object sweep data."""
from copy import deepcopy
import unittest

import numpy as np

from tools.physics.pybullet_backend_dispatcher import _common_audit, _adapter_hard_results


def fixture(adapter="generic_rigid_v1"):
    objects = [{
        "object_id": name,
        "initial_state": {"position_m": [float(i), 0., 0.],
            "linear_velocity_m_s": [0., 0., 0.],
            "angular_velocity_rad_s": [0., 0., 0.],
            "orientation_quaternion_wxyz": [1., 0., 0., 0.]},
        "material": {"mass_kg": 1., "contact_friction": .2, "contact_restitution": .5},
    } for i, name in enumerate(("object_a", "object_b"))]
    scene = {"scene_id": "pair", "objects": objects,
        "backend_binding": {"adapter_id": adapter}, "variant": {"kind": "sweep"},
        "time": {"frame_count": 3, "output_fps": 24, "simulation_hz": 240}}
    engine = {"solver_iterations": 180, "deterministic_overlapping_pairs": True,
              "restitution_velocity_threshold_m_s": .02,
              "enable_cone_friction": True, "use_split_impulse": True}
    scene["source_metadata"] = {"simulation": {"solver": dict(engine)}, "physics": {"engine": dict(engine)}}
    scene["source_metadata"]["simulation"].update(time=dict(scene["time"]), objects=deepcopy(objects), support={})
    scene["adapter_payload"] = {"backend": {"physics": dict(engine), "billiards_rules": {"engine": dict(engine), "ball_dynamics": {}}}}
    trajectory = {
        "time_s": np.arange(3) / 24,
        "position_m": np.tile([o["initial_state"]["position_m"] for o in objects], (3, 1, 1)),
        "quaternion_wxyz": np.tile([1., 0., 0., 0.], (3, 2, 1)),
        "linear_velocity_m_s": np.zeros((3, 2, 3)),
        "angular_velocity_rad_s": np.zeros((3, 2, 3)),
        "contact_count": np.zeros((3, 2), dtype=int),
        "runtime_material": np.tile([1., .2, .5], (2, 1)),
        "inertia_diagonal_kg_m2": np.ones((2, 3)),
    }
    # Large finite responses are not an execution/metadata mismatch.
    trajectory["linear_velocity_m_s"][1:, 0, 0] = 100.
    trajectory["angular_velocity_rad_s"][1:, 1, 0] = 500.
    if adapter != "generic_rigid_v1":
        for obj in objects:
            obj["material"].update(rolling_friction=.0025, spinning_friction=.0006,
                                   linear_damping=.025, angular_damping=.01)
        trajectory["runtime_material_extras"] = np.tile([.0025,.0006,.025,.01], (2,1))
    if adapter == "generic_rigid_v1":
        checks = [{"id": f"{o['object_id']}__{name}", "passed": True}
            for o in objects for name in (
                "finite_state", "initial_position_matches_metadata",
                "initial_linear_velocity_matches_metadata", "pybullet_dynamics_match_metadata",
                "pybullet_support_dynamics_match_metadata", "runtime_inertia_is_finite_and_positive",
                "collision_proxy_matches_definition")]
        checks += [{"id": name, "passed": False} for name in (
            "object_a__bounded_linear_speed", "object_b__bounded_angular_speed",
            "object_b__bounded_penetration", "required_pair_collision")]
    else:
        checks = {"finite_trajectory": True, "two_dynamic_objects": True,
            "maximum_penetration": False, "pair_contact_observed": False}
    return scene, trajectory, {"passed": False, "checks": checks, "advisories": [],
        "contact_processing_execution": [{"object_id": o["object_id"], "contactProcessingThreshold": 0.0} for o in objects],
        "solver_execution": {"pybullet_arguments": {
            "numSolverIterations": 180, "deterministicOverlappingPairs": 1,
            "restitutionVelocityThreshold": .02, "enableConeFriction": 1, "useSplitImpulse": 1,
            **({"solverResidualThreshold": 0.0} if adapter == "generic_rigid_v1" else {})},
            "reported_parameters": {"numSolverIterations": 180, "fixedTimeStep": 1/240, "numSubSteps": 0}},
        "integration_execution": {"nominal_simulation_hz": 240, "substep_factor": 1,
            "simulation_hz": 240, "steps_per_frame": 10, "fixed_time_step_s": 1/240}}


class TwoObjectSweepAdmissionTests(unittest.TestCase):
    def test_quality_failures_are_retained_for_sweep_and_still_reject_base(self):
        for adapter in ("generic_rigid_v1", "billiards_two_object_v1",
                        "passive_pinball_two_object_v1", "marble_run_two_object_v1"):
            with self.subTest(adapter=adapter):
                scene, trajectory, raw = fixture(adapter)
                before = deepcopy(raw)
                audit = _common_audit(scene, trajectory, raw)
                self.assertTrue(audit["passed"])
                self.assertFalse(audit["adapter_audit_passed"])
                self.assertEqual(raw, before)
                self.assertEqual(audit["adapter_audit"], before)
                self.assertEqual(audit["adapter_audit_policy"],
                    "base_quality_rules_diagnostic_for_two_object_sweep_v1")
                scene["variant"]["kind"] = "base"
                self.assertFalse(_common_audit(scene, trajectory, raw)["passed"])

    def test_sweep_still_rejects_nonfinite_or_misbound_trajectory(self):
        for key, index, value in (
            ("position_m", (1, 0, 0), float("nan")),
            ("position_m", (0, 1, 0), 2.),
            ("runtime_material", (0, 0), 2.),
            ("inertia_diagonal_kg_m2", (1, 0), 0.),
            ("quaternion_wxyz", (1, 0, 0), 2.),
        ):
            with self.subTest(key=key, index=index):
                scene, trajectory, raw = fixture()
                trajectory[key][index] = value
                self.assertFalse(_common_audit(scene, trajectory, raw)["passed"])

    def test_proxy_mismatch_or_missing_integrity_check_cannot_be_waived(self):
        for missing in (False, True):
            scene, trajectory, raw = fixture()
            target = next(r for r in raw["checks"]
                if r["id"] == "object_b__collision_proxy_matches_definition")
            if missing:
                raw["checks"].remove(target)
            else:
                target.update(passed=False, severity="advisory")
            self.assertFalse(_common_audit(scene, trajectory, raw)["passed"])

    def test_one_object_sweep_keeps_existing_adapter_gates(self):
        scene, _, raw = fixture()
        scene["objects"] = scene["objects"][:1]
        self.assertFalse(all(_adapter_hard_results(scene, raw)))


if __name__ == "__main__":
    unittest.main()
