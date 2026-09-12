import copy
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from tests.three_object_fixtures import source
from tools.core.hashing import sha256_json
from tools.core.primitive_support import initial_bounds, initial_extents
from tools.core.rigid_dynamics import expected_object_inertia, principal_inertia
from tools.rendering.three_object_primitives import (
    blocked_by_compound,
    blocked_by_primitive,
    compound_surface_samples,
    pybullet_compound_ray_world,
)
from tools.rendering.three_object_camera import audit_three_objects
from tools.sampling.sample_three_object_base import build_three_object_scene
from tools.sampling.three_object_coverage import build_plan
from tools.sampling.three_object_sampling_request import load_pilot_rules, validate_rules
from tools.scene_rules.three_object import validate_initial_layout


ASSETS = [
    "sketchfab_046b94a7775e496aac26ad8c699496e6",
    "sketchfab_1551616939cc4da7a9d731dfddd4090f",
    "sketchfab_beb69bc08a8f487ab8c5207fb155cbf2",
]


def collision_profile():
    colliders = [
        {
            "shape": "cylinder",
            "size_m": [0.12, 0.12, 0.08],
            "position_m": [0.0, 0.0, -0.02],
            "rotation_euler_degrees": [0.0, 0.0, 0.0],
        },
        {
            "shape": "cylinder",
            "size_m": [0.06, 0.06, 0.06],
            "position_m": [0.0, 0.0, 0.03],
            "rotation_euler_degrees": [0.0, 0.0, 0.0],
        },
    ]
    bare = {"type": "compound", "colliders": colliders}
    return {
        **bare,
        "inertia_reference": {
            "schema_version": "physweep_compound_inertia_reference_v1",
            "method": "released_base_runtime_v1",
            "reference_mass_kg": 0.5,
            "diagonal_kg_m2": [0.001, 0.001, 0.0005],
            "collision_profile_sha256": sha256_json(bare),
        },
    }


def compound_row(index=10, asset=ASSETS[0], scale="small"):
    row = source(index)
    row["source"]["source_family"] = "asset"
    obj = row["metadata"]["simulation"]["objects"][0]
    obj["geometry"] = {"type": "cylinder", "size_m": [0.12, 0.12, 0.12]}
    obj["collision_profile"] = collision_profile()
    obj["visual_profile"]["id"] = asset
    row["metadata"]["semantic_sampling"]["five_dimensions"]["foreground_object"]["scale_bin"] = scale
    return row


def fixture_pool():
    objects = []
    for index in range(3):
        row = source(index)
        row["source"]["source_family"] = "generic"
        row["metadata"]["semantic_sampling"]["five_dimensions"]["foreground_object"]["scale_bin"] = "medium"
        objects.append(row)
    for index, (asset, scale) in enumerate(zip(ASSETS, ("small", "small", "large")), 10):
        objects.append(compound_row(index, asset, scale))
    hosts = []
    for host in ("ground_flat", "raised_flat"):
        for category in ("home_office", "lab_studio"):
            row = source()
            row["source"]["source_family"] = "generic"
            row["metadata"]["simulation"]["support"]["scene_class"] = host
            row["metadata"]["appearance"]["scene_visual"]["environment_category"] = category
            hosts.append(row)
    return {"objects": objects, "hosts": hosts, "summary": {}}


class CompoundAssetTests(unittest.TestCase):
    def test_exact_initial_bounds_and_support_offset(self):
        obj = compound_row()["metadata"]["simulation"]["objects"][0]
        low, high = initial_bounds(obj)
        np.testing.assert_allclose(low, [-0.06, -0.06, -0.06])
        np.testing.assert_allclose(high, [0.06, 0.06, 0.06])
        np.testing.assert_allclose(initial_extents(obj), [0.06, 0.06, 0.06])

        asymmetric = copy.deepcopy(obj)
        asymmetric["collision_profile"]["colliders"][1]["position_m"][2] = 0.0305
        asymmetric["collision_profile"]["colliders"][1]["size_m"][2] = 0.061
        asymmetric["geometry"]["size_m"][2] = 0.121
        asymmetric["collision_profile"]["inertia_reference"]["collision_profile_sha256"] = sha256_json({
            "type": "compound",
            "colliders": asymmetric["collision_profile"]["colliders"],
        })
        low, high = initial_bounds(asymmetric)
        self.assertAlmostEqual(low[2], -0.06)
        self.assertAlmostEqual(high[2], 0.061)

        rules = load_pilot_rules(matrix_path=Path("three_object_d5h_sampling_matrix.json"))
        asymmetric_row = compound_row()
        asymmetric_row["metadata"]["simulation"]["objects"][0] = asymmetric
        scene = build_three_object_scene(
            host_source=source(),
            object_sources=[source(0), source(1), asymmetric_row],
            rules=rules,
            scene_id="compound_support_offset",
            template_id="pair_control",
            role_order=["P", "Q", "R"],
        )
        target = scene["simulation"]["objects"][2]
        self.assertAlmostEqual(target["initial_state"]["position_m"][2], 0.06)
        validate_initial_layout(scene, rules["scene"], 0.001)

    def test_inertia_reference_scales_and_binds_colliders(self):
        obj = compound_row()["metadata"]["simulation"]["objects"][0]
        np.testing.assert_allclose(expected_object_inertia(obj), [0.001, 0.001, 0.0005])
        changed = copy.deepcopy(obj)
        changed["material"]["mass_kg"] = 2.0
        np.testing.assert_allclose(expected_object_inertia(changed), [0.004, 0.004, 0.002])
        changed["collision_profile"]["colliders"][0]["size_m"][0] += 0.001
        with self.assertRaisesRegex(ValueError, "does not match"):
            expected_object_inertia(changed)
        primitive = source()["metadata"]["simulation"]["objects"][0]
        np.testing.assert_allclose(
            expected_object_inertia(primitive),
            principal_inertia("sphere", np.asarray(primitive["geometry"]["size_m"]), primitive["material"]["mass_kg"]),
        )

    def test_exact_compound_surface_and_rays_do_not_use_envelope(self):
        colliders = collision_profile()["colliders"]
        camera = np.asarray([4.0, 4.0, 4.0])
        points, weights = compound_surface_samples(colliders, np.zeros(3), [1, 0, 0, 0], camera)
        self.assertTrue(np.all(weights > 0))
        radial = np.linalg.norm(points[:, :2], axis=1)
        self.assertTrue(np.all(radial[points[:, 2] > 0.020001] <= 0.03000001))
        self.assertTrue(np.any(np.isclose(points[:, 2], 0.02) & (radial > 0.031)))

        camera = np.asarray([-3.0, 0.05, 0.05])
        ends = np.asarray([[3.0, 0.05, 0.05], [3.0, -0.01, 0.05]])
        exact = blocked_by_compound(camera, ends, np.zeros(3), [1, 0, 0, 0], colliders)
        envelope = blocked_by_primitive(camera, ends, np.zeros(3), [1, 0, 0, 0], "cylinder", [0.12, 0.12, 0.12])
        np.testing.assert_array_equal(exact, [False, True])
        np.testing.assert_array_equal(envelope, [True, True])
        try:
            import pybullet  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("PyBullet is exercised by the server CI runtime")
        obj = compound_row()["metadata"]["simulation"]["objects"][0]
        with pybullet_compound_ray_world([obj]) as actual_blocked:
            actual = actual_blocked(
                obj["object_id"],
                [-3.0, 0.0, 0.0],
                [[3.0, 0.0, 0.0], [3.0, 0.2, 0.2]],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            )
        np.testing.assert_array_equal(actual, [True, False])

    def test_compound_frame_check_ignores_empty_world_aabb_corners(self):
        objects = [copy.deepcopy(source(index)["metadata"]["simulation"]["objects"][0]) for index in range(3)]
        for index, obj in enumerate(objects):
            obj["object_id"] = f"object_{chr(ord('a') + index)}"
        objects[2] = copy.deepcopy(compound_row()["metadata"]["simulation"]["objects"][0])
        objects[2]["object_id"] = "object_c"
        centers = {
            "object_a": np.asarray([-1.0, 0.0, 0.1]),
            "object_b": np.asarray([1.0, 0.0, 0.1]),
            "object_c": np.asarray([0.0, 0.0, 0.0]),
        }
        trajectory = {}
        for obj in objects:
            object_id = obj["object_id"]
            positions = np.repeat(centers[object_id][None, :], 2, axis=0)
            trajectory[object_id + "__position_m"] = positions
            radius = float(obj["geometry"]["size_m"][0]) / 2
            trajectory[object_id + "__aabb_min_m"] = positions - radius
            trajectory[object_id + "__aabb_max_m"] = positions + radius
        # These bounds represent the conservative empty-corner failure mode.
        # The compound child union itself remains small and centered.
        trajectory["object_c__aabb_min_m"] = np.repeat([[-4.0, -4.0, -4.0]], 2, axis=0)
        trajectory["object_c__aabb_max_m"] = np.repeat([[4.0, 4.0, 4.0]], 2, axis=0)
        trajectory["object_c__quaternion_wxyz"] = np.repeat([[1.0, 0.0, 0.0, 0.0]], 2, axis=0)
        camera = {
            "position_m": [0.0, -8.0, 4.0],
            "target_m": [0.0, 0.0, 0.0],
            "focal_length_mm": 35.0,
            "sensor_width_mm": 36.0,
            "clip_start_m": 0.05,
            "clip_end_m": 100.0,
        }
        audit = audit_three_objects(
            {"simulation": {"objects": objects}},
            trajectory,
            camera,
            contract={
                "camera_rules": {
                    "frame_margin_fraction": 0.04,
                    "minimum_object_extent_fraction_of_short_side": 0.0,
                    "maximum_object_extent_fraction_of_short_side": 10.0,
                    "maximum_safe_frame_violation_fraction": 1.0,
                    "maximum_out_of_frame_fraction": 1.0,
                    "maximum_frame_overflow_fraction": 10.0,
                    "event_frame_margin_fraction": 0.0,
                    "event_frame_padding": 0,
                    "maximum_occluded_fraction": 1.0,
                }
            },
            resolution=[1280, 720],
            blockers=[],
            compound_occlusion=lambda _object_id, _camera, points, _center, _q: np.zeros(len(points), dtype=bool),
        )
        self.assertEqual(audit["objects"]["object_c"]["out_of_frame_fraction"], 0.0)
        self.assertTrue(audit["objects"]["object_c"]["passed"])

    def test_three_assets_have_two_deterministic_cells(self):
        rules = load_pilot_rules(matrix_path=Path("three_object_d5h_sampling_matrix.json"))
        plan = build_plan(fixture_pool(), rules)
        self.assertEqual(plan["required_asset_ids"], ASSETS)
        counts = Counter((cell["required_visual_asset_by_role"]["R"], cell["template"]) for cell in plan["cells"])
        self.assertEqual(len(counts), 6)
        self.assertEqual(set(counts.values()), {1})
        changed = fixture_pool()
        changed["objects"].reverse()
        changed["hosts"].reverse()
        self.assertEqual(plan, build_plan(changed, rules))
        invalid = copy.deepcopy(rules["matrix"])
        invalid["required_asset_ids"] = invalid["required_asset_ids"][:-1]
        with self.assertRaisesRegex(ValueError, "three explicit assets"):
            validate_rules(invalid, rules["motion"], rules["scene"], rules["sweep"], rules["dataset"])


if __name__ == "__main__":
    unittest.main()
