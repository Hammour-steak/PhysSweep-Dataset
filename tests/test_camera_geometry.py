from __future__ import annotations

import copy
import math
import unittest

import numpy as np

from tools.core.camera_geometry import pair_camera_geometry_eligible
from tools.motion_rules.two_object.motion import interaction_approach_axis


class CameraGeometryTests(unittest.TestCase):
    def test_interaction_approach_axis_uses_the_support_motion_frame(self) -> None:
        intent = {
            "interaction_class": "interacting",
            "layout": "planned_supported_contact",
            "linear_velocity_m_s": [[0.7, 0.0, 0.0], [0.0, 0.0, 0.0]],
            "impact_offset_ratio": 0.0,
        }
        flat = {
            "support_shape": "rectangular_slab",
            "surface_frame": {
                "tangent_cross": [1.0, 0.0, 0.0],
                "tangent_uphill": [0.0, 1.0, 0.0],
                "normal": [0.0, 0.0, 1.0],
            },
        }
        self.assertTrue(
            np.allclose(
                interaction_approach_axis(intent, flat),
                [1.0, 0.0, 0.0],
                atol=1.0e-12,
                rtol=0.0,
            )
        )
        slope = math.radians(9.0)
        incline = copy.deepcopy(flat)
        incline["support_shape"] = "inclined_ramp"
        incline["surface_frame"] = {
            "tangent_cross": [1.0, 0.0, 0.0],
            "tangent_uphill": [0.0, math.cos(slope), math.sin(slope)],
            "normal": [0.0, -math.sin(slope), math.cos(slope)],
        }
        self.assertTrue(
            np.allclose(
                interaction_approach_axis(intent, incline),
                incline["surface_frame"]["tangent_uphill"],
                atol=1.0e-12,
                rtol=0.0,
            )
        )

    def test_inclined_pair_eligibility_uses_physical_geometry(self) -> None:
        slope = math.radians(9.0)
        orientation = [math.cos(0.5 * slope), math.sin(0.5 * slope), 0.0, 0.0]
        support = {
            "support_shape": "inclined_ramp",
            "surface_frame": {
                "tangent_cross": [1.0, 0.0, 0.0],
                "tangent_uphill": [0.0, math.cos(slope), math.sin(slope)],
                "normal": [0.0, -math.sin(slope), math.cos(slope)],
            },
        }
        interaction = {
            "approach_axis_xyz": support["surface_frame"]["tangent_uphill"],
            "camera_relative_azimuth_degrees": 90.0,
            "maximum_camera_view_azimuth_deviation_degrees": 8.0,
            "minimum_camera_elevation_degrees": 22.0,
            "preferred_camera_elevation_degrees": 28.0,
            "maximum_camera_elevation_degrees": 34.0,
            "minimum_pair_keyframe_projected_center_separation_to_radius_sum_ratio": 0.75,
        }

        def primitive(shape: str, size: list[float]) -> dict:
            return {
                "geometry": {"type": shape, "size_m": size},
                "initial_state": {
                    "orientation_quaternion_wxyz": orientation,
                },
            }

        readable = {
            "simulation": {
                "support": support,
                "interaction": interaction,
                "objects": [
                    primitive("sphere", [0.20, 0.20, 0.20]),
                    primitive("sphere", [0.22, 0.22, 0.22]),
                ],
            }
        }
        self.assertTrue(pair_camera_geometry_eligible(readable, 0.17))
        self.assertTrue(pair_camera_geometry_eligible(readable, 0.0))
        flat = copy.deepcopy(readable)
        flat["simulation"]["support"]["support_shape"] = "rectangular_slab"
        flat["simulation"]["support"]["surface_frame"] = {
            "tangent_uphill": [0.0, 1.0, 0.0],
            "tangent_cross": [1.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
        }
        self.assertTrue(pair_camera_geometry_eligible(flat, 0.0))

        too_small = copy.deepcopy(readable)
        too_small["simulation"]["objects"][0] = primitive(
            "sphere", [0.15, 0.15, 0.15]
        )
        self.assertFalse(pair_camera_geometry_eligible(too_small, 0.17))

        overlap = copy.deepcopy(readable)
        overlap["simulation"]["objects"] = [
            primitive("cylinder", [0.073, 0.073, 0.30]),
            primitive("cylinder", [0.073, 0.073, 0.30]),
        ]
        self.assertFalse(pair_camera_geometry_eligible(overlap, 0.17))
        self.assertFalse(pair_camera_geometry_eligible(overlap, 0.0))

        invalid_approach = copy.deepcopy(readable)
        invalid_approach["simulation"]["interaction"][
            "approach_axis_xyz"
        ] = [0.0, 0.0, 0.0]
        with self.assertRaisesRegex(ValueError, "finite and nonzero"):
            pair_camera_geometry_eligible(invalid_approach, 0.17)


if __name__ == "__main__":
    unittest.main()
