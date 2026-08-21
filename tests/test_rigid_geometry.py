from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from rigid_geometry import (  # noqa: E402
    build_support_geometry,
    pose_on_support,
    support_surface_height_m,
    validate_support_geometry,
)
from sample_pybullet_base import load_active_rules  # noqa: E402


class RigidGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.axes = load_active_rules(ROOT)["axes"]

    def test_every_declared_support_compiles_to_explicit_colliders(self) -> None:
        slope_subtype = self.axes["motion_subtype_axis"]["slope_slide_down_1obj"][0]
        flat_subtype = self.axes["motion_subtype_axis"]["slide_push_1obj"][0]
        for support in self.axes["support_axis"]:
            support_shape = support.get("overrides", {}).get("placement", {}).get(
                "support_shape", "rectangular_slab"
            )
            motion = (
                "slope_slide_down_1obj"
                if support_shape == "inclined_ramp"
                else "slide_push_1obj"
            )
            subtype = slope_subtype if support_shape == "inclined_ramp" else flat_subtype
            with self.subTest(support=support["label"]):
                geometry = build_support_geometry(support, motion, subtype)
                validate_support_geometry(geometry)
                ids = {record["id"] for record in geometry["colliders"]}
                self.assertIn("support", ids)
                if support["scene_class"] == "ground_flat":
                    self.assertNotIn("environment_floor", ids)
                else:
                    self.assertIn("environment_floor", ids)
                self.assertTrue(all(record["primitive"] == "box" for record in geometry["colliders"]))

    def test_every_object_shape_can_be_placed_on_every_compatible_support(self) -> None:
        slope_subtype = self.axes["motion_subtype_axis"]["slope_slide_down_1obj"][0]
        flat_subtype = self.axes["motion_subtype_axis"]["slide_push_1obj"][0]
        for support in self.axes["support_axis"]:
            support_shape = support.get("overrides", {}).get("placement", {}).get(
                "support_shape", "rectangular_slab"
            )
            motion = (
                "slope_slide_down_1obj"
                if support_shape == "inclined_ramp"
                else "slide_push_1obj"
            )
            subtype = slope_subtype if support_shape == "inclined_ramp" else flat_subtype
            geometry = build_support_geometry(support, motion, subtype)
            for obj in self.axes["object_axis"]:
                with self.subTest(support=support["label"], object=obj["label"]):
                    pose = pose_on_support(
                        geometry,
                        obj["shape"],
                        obj["size"],
                        0.0,
                        0.0,
                        0.0,
                        0.003,
                        obj.get("pose_profile", "support_normal"),
                        [1.0, 0.0, 0.0],
                    )
                    surface_z = support_surface_height_m(geometry, 0.0, 0.0)
                    self.assertGreater(pose["position_m"][2], surface_z)

    def test_ramp_surface_height_increases_uphill(self) -> None:
        support = next(
            item
            for item in self.axes["support_axis"]
            if item["label"] == "ground_ramp_long_shallow"
        )
        subtype = self.axes["motion_subtype_axis"]["slope_slide_down_1obj"][0]
        geometry = build_support_geometry(support, "slope_slide_down_1obj", subtype)
        low = support_surface_height_m(geometry, 0.0, -0.4)
        high = support_surface_height_m(geometry, 0.0, 0.4)
        self.assertGreater(high, low)

    def test_ramps_declare_one_metadata_driven_solid_wedge(self) -> None:
        subtype = self.axes["motion_subtype_axis"]["slope_slide_down_1obj"][0]
        for support in self.axes["support_axis"]:
            placement = support.get("overrides", {}).get("placement", {})
            if placement.get("support_shape") != "inclined_ramp":
                continue
            geometry = build_support_geometry(
                support, "slope_slide_down_1obj", subtype
            )
            visual = geometry["visual_geometry"]
            with self.subTest(support=support["label"]):
                self.assertEqual(visual["primitive"], "solid_wedge")
                self.assertEqual(visual["slope_axis"], "y")
                self.assertGreater(visual["high_top_z_m"], visual["base_z_m"])
                replaced = [
                    record
                    for record in geometry["colliders"]
                    if record.get("render_replaced_by_solid_wedge", False)
                ]
                self.assertTrue(replaced)
                self.assertEqual(replaced[0]["role"], "primary_support")

    def test_ground_flat_is_one_flush_full_floor(self) -> None:
        subtype = self.axes["motion_subtype_axis"]["slide_push_1obj"][0]
        for support in self.axes["support_axis"]:
            if support["scene_class"] != "ground_flat":
                continue
            geometry = build_support_geometry(support, "slide_push_1obj", subtype)
            primary = next(item for item in geometry["colliders"] if item["id"] == "support")
            with self.subTest(support=support["label"]):
                self.assertEqual(geometry["surface_center_z_m"], 0.0)
                self.assertEqual(primary["size_m"][:2], [8.0, 8.0])
                self.assertEqual(primary["position_m"][2], -0.05)

    def test_track_semantics_are_not_declared(self) -> None:
        labels = {str(item["label"]) for item in self.axes["support_axis"]}
        shapes = {
            str(
                item.get("overrides", {})
                .get("placement", {})
                .get("support_shape", "rectangular_slab")
            )
            for item in self.axes["support_axis"]
        }
        self.assertTrue({"floor_track", "lab_track"}.isdisjoint(labels))
        self.assertNotIn("narrow_track", shapes)

    def test_side_on_cylinders_rest_on_their_radius(self) -> None:
        support_record = next(
            item for item in self.axes["support_axis"] if item["label"] == "wood_tabletop"
        )
        subtype = self.axes["motion_subtype_axis"]["roll_or_slide_1obj"][0]
        support = build_support_geometry(support_record, "roll_or_slide_1obj", subtype)
        obj = next(
            (
                item
                for item in self.axes["object_axis"]
                if item["shape"] == "cylinder" and item["pose_profile"] == "side_on"
            ),
            None,
        )
        if obj is None:
            self.skipTest("active asset-only library has no side-on cylinder profile")
        pose = pose_on_support(
            support,
            obj["shape"],
            obj["size"],
            0.0,
            0.0,
            0.0,
            0.003,
            obj["pose_profile"],
            [1.0, 0.0, 0.0],
        )
        expected_height = support["surface_center_z_m"] + obj["size"][0] / 2.0 + 0.003
        self.assertAlmostEqual(pose["position_m"][2], expected_height, places=6)
        self.assertAlmostEqual(sum(value * value for value in pose["orientation_quaternion_wxyz"]), 1.0, places=6)

    def test_wall_impact_compiles_an_explicit_wall_collider(self) -> None:
        support_record = next(
            item for item in self.axes["support_axis"] if item["label"] == "wood_tabletop"
        )
        subtype = self.axes["motion_subtype_axis"]["wall_impact_1obj"][0]
        geometry = build_support_geometry(
            support_record, "wall_impact_1obj", subtype, [1.0, 0.0, 0.0]
        )
        wall = next(item for item in geometry["colliders"] if item["id"] == "impact_wall")
        self.assertEqual(wall["role"], "impact_wall")
        self.assertGreater(wall["position_m"][0], 0.0)

    def test_ramp_transition_compiles_a_contiguous_landing_surface(self) -> None:
        support_record = next(
            item
            for item in self.axes["support_axis"]
            if item["label"] == "raised_ramp_standard"
        )
        subtype = self.axes["motion_subtype_axis"]["ramp_to_flat_1obj"][0]
        geometry = build_support_geometry(
            support_record, "ramp_to_flat_1obj", subtype, [0.0, -1.0, 0.0]
        )
        landing = next(
            item for item in geometry["colliders"] if item["id"] == "landing_surface"
        )
        ramp = next(item for item in geometry["colliders"] if item["id"] == "support")
        angle = math.radians(float(ramp["rotation_euler_degrees"][0]))
        ramp_low_edge_z = (
            float(ramp["position_m"][2])
            - float(ramp["size_m"][1]) * math.sin(angle) / 2.0
            + float(ramp["size_m"][2]) * math.cos(angle) / 2.0
        )
        landing_top_z = landing["position_m"][2] + landing["size_m"][2] / 2.0
        self.assertAlmostEqual(landing_top_z, ramp_low_edge_z, places=5)
        self.assertEqual(landing["role"], "landing_surface")

    def test_ground_ramp_transition_uses_the_flush_environment_floor(self) -> None:
        support = next(
            item
            for item in self.axes["support_axis"]
            if item["label"] == "ground_ramp_standard"
        )
        subtype = self.axes["motion_subtype_axis"]["ramp_to_flat_1obj"][0]
        geometry = build_support_geometry(
            support, "ramp_to_flat_1obj", subtype, [0.0, -1.0, 0.0]
        )
        ids = {item["id"] for item in geometry["colliders"]}
        self.assertIn("environment_floor", ids)
        self.assertNotIn("landing_surface", ids)

    def test_ramp_geometry_is_owned_by_the_scene_kit(self) -> None:
        support = next(
            item
            for item in self.axes["support_axis"]
            if item["label"] == "ground_ramp_short_steep"
        )
        subtypes = self.axes["motion_subtype_axis"]["slope_slide_down_1obj"]
        angles = {
            build_support_geometry(support, "slope_slide_down_1obj", subtype)[
                "surface_frame"
            ]["slope_angle_degrees"]
            for subtype in subtypes
        }
        self.assertEqual(len(angles), 1)

    def test_ramp_families_have_distinct_physical_angles(self) -> None:
        subtype = self.axes["motion_subtype_axis"]["slope_slide_down_1obj"][0]
        angles = {}
        for family in (
            "straight_long_shallow",
            "straight_standard",
            "straight_short_steep",
        ):
            support = next(
                item
                for item in self.axes["support_axis"]
                if item.get("overrides", {})
                .get("placement", {})
                .get("structure_family") == family
            )
            angles[family] = build_support_geometry(
                support, "slope_slide_down_1obj", subtype
            )["surface_frame"]["slope_angle_degrees"]
        self.assertLess(angles["straight_long_shallow"], angles["straight_standard"])
        self.assertLess(angles["straight_standard"], angles["straight_short_steep"])

    def test_channel_ramps_compile_two_inclined_side_rails(self) -> None:
        subtype = self.axes["motion_subtype_axis"]["slope_slide_down_1obj"][0]
        channels = [
            item
            for item in self.axes["support_axis"]
            if item.get("overrides", {})
            .get("placement", {})
            .get("structure_family") == "channel_medium"
        ]
        self.assertEqual(len(channels), 2)
        for support in channels:
            geometry = build_support_geometry(
                support, "slope_slide_down_1obj", subtype
            )
            rails = [
                item
                for item in geometry["colliders"]
                if item["role"] == "support_rail"
            ]
            with self.subTest(support=support["label"]):
                self.assertEqual(len(rails), 2)
                self.assertTrue(
                    all(
                        rail["rotation_euler_degrees"][0]
                        == geometry["surface_frame"]["slope_angle_degrees"]
                        for rail in rails
                    )
                )

    def test_ground_ramp_low_edges_are_flush_with_the_floor(self) -> None:
        subtype = self.axes["motion_subtype_axis"]["slope_slide_down_1obj"][0]
        supports = [
            item
            for item in self.axes["support_axis"]
            if item.get("overrides", {})
            .get("placement", {})
            .get("anchor_low_edge_to_floor", False)
        ]
        self.assertEqual(len(supports), 4)
        for support in supports:
            geometry = build_support_geometry(
                support, "slope_slide_down_1obj", subtype
            )
            ramp = next(item for item in geometry["colliders"] if item["id"] == "support")
            angle = math.radians(float(ramp["rotation_euler_degrees"][0]))
            low_edge_z = (
                float(ramp["position_m"][2])
                - float(ramp["size_m"][1]) * math.sin(angle) / 2.0
                + float(ramp["size_m"][2]) * math.cos(angle) / 2.0
            )
            with self.subTest(support=support["label"]):
                self.assertAlmostEqual(low_edge_z, 0.0, places=5)
                self.assertEqual(geometry["structure_anchor"], "floor_flush_low_edge")

    def test_raised_ramps_are_flush_with_a_legged_platform(self) -> None:
        subtype = self.axes["motion_subtype_axis"]["slope_slide_down_1obj"][0]
        supports = [
            item
            for item in self.axes["support_axis"]
            if item.get("overrides", {})
            .get("placement", {})
            .get("base_platform_top_z_m") is not None
        ]
        self.assertEqual(len(supports), 4)
        for support in supports:
            geometry = build_support_geometry(
                support, "slope_slide_down_1obj", subtype
            )
            colliders = geometry["colliders"]
            ramp = next(item for item in colliders if item["id"] == "support")
            platform = next(
                item for item in colliders if item["id"] == "ramp_base_platform"
            )
            legs = [item for item in colliders if item["id"].startswith("ramp_platform_leg_")]
            angle = math.radians(float(ramp["rotation_euler_degrees"][0]))
            low_edge_z = (
                float(ramp["position_m"][2])
                - float(ramp["size_m"][1]) * math.sin(angle) / 2.0
                + float(ramp["size_m"][2]) * math.cos(angle) / 2.0
            )
            platform_top_z = platform["position_m"][2] + platform["size_m"][2] / 2.0
            with self.subTest(support=support["label"]):
                self.assertAlmostEqual(low_edge_z, platform_top_z, places=5)
                self.assertEqual(len(legs), 4)
                self.assertEqual(
                    geometry["structure_anchor"], "raised_platform_flush_low_edge"
                )


if __name__ == "__main__":
    unittest.main()
