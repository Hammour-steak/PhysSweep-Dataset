from __future__ import annotations

import hashlib
import json
import math
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

from tools.assets.environment_collision import (  # noqa: E402
    binding_sha256,
    compile_environment_binding,
    dynamic_back_wall_clearance_m,
    dynamic_motion_lane,
    validate_environment_binding,
)
from tools.sampling.sample_one_object_scene_matrix import (  # noqa: E402
    MATRIX_PATH,
    matrix_dependency_paths,
)
from tools.sampling.sample_pybullet_base import BUNDLE_PATH  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authoritative_floor_band_face_count(
    path: Path,
    floor_z_m: float,
    half_band_m: float,
    minimum_abs_normal_z: float,
) -> int:
    vertices = []
    faces = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("v "):
            vertices.append([float(value) for value in line.split()[1:4]])
        elif line.startswith("f "):
            faces.append([int(value) - 1 for value in line.split()[1:4]])
    values = np.asarray(vertices, dtype=np.float64)
    triangles = values[np.asarray(faces, dtype=np.int64)]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    horizontal = np.abs(normals[:, 2]) / np.maximum(lengths, 1.0e-12)
    centroid_z = triangles[:, :, 2].mean(axis=1)
    return int(
        np.count_nonzero(
            (horizontal >= minimum_abs_normal_z)
            & (np.abs(centroid_z - floor_z_m) <= half_band_m)
        )
    )


class VisualEnvironmentCollisionV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profiles = load_json(ROOT / "configs/scene_mesh_profiles.json")
        cls.proxies = load_json(
            ROOT / "configs/visual_environment_collision_proxies.json"
        )
        cls.registry = load_json(ROOT / "configs/asset_proxy_registry.json")
        cls.bundle = load_json(BUNDLE_PATH)
        cls.matrix = load_json(MATRIX_PATH)
        cls.compositions = load_json(
            ROOT / "configs/visual_environment_composition.json"
        )
        cls.rules = load_json(ROOT / "configs/one_object_sampling_rules.json")

    def test_environment_collision_rejects_undeclared_object_counts(self) -> None:
        metadata = {
            "simulation": {
                "objects": [{"object_id": "a"}, {"object_id": "b"}],
                "time": {"duration_s": 4.0},
            }
        }
        with self.assertRaisesRegex(ValueError, "supports dynamic object counts"):
            dynamic_back_wall_clearance_m(metadata, [1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "supports dynamic object counts"):
            dynamic_motion_lane(metadata)

    def test_every_admitted_environment_has_one_hashed_static_proxy(self) -> None:
        profiles = {
            str(record["asset"]["asset_id"]): record
            for record in self.profiles["profiles"]
        }
        proxies = {
            str(record["asset_id"]): record for record in self.proxies["records"]
        }
        self.assertEqual(len(profiles), 20)
        self.assertEqual(set(profiles), set(proxies))
        self.assertTrue(self.profiles["policy"]["environment_proxy_is_always_loaded"])
        for asset_id, profile in profiles.items():
            attached = profile["asset"]["collision_proxy"]
            source = proxies[asset_id]["proxy"]
            self.assertEqual(attached["representation"], "static_concave_mesh")
            self.assertEqual(attached["path"], source["path"])
            self.assertEqual(attached["sha256"], source["sha256"])
            path = ROOT / str(attached["path"])
            self.assertTrue(path.is_file())
            self.assertEqual(sha256(path), str(attached["sha256"]))
            self.assertLessEqual(int(attached["face_count"]), 81000)
            transform = proxies[asset_id]["transform_contract"]
            self.assertEqual(
                authoritative_floor_band_face_count(
                    path,
                    float(transform["authoritative_floor_z_m"]),
                    float(transform["authoritative_floor_exclusion_half_band_m"]),
                    float(transform["authoritative_floor_minimum_abs_normal_z"]),
                ),
                0,
            )

    def test_unified_registry_declares_static_environment_role(self) -> None:
        registry = {
            str(record["asset_id"]): record for record in self.registry["records"]
        }
        for proxy in self.proxies["records"]:
            record = registry[str(proxy["asset_id"])]
            self.assertEqual(record["asset_role"], "static_environment")
            self.assertEqual(record["admission"]["status"], "physics_ready")
            self.assertEqual(record["proxy"]["kind"], "static_environment_mesh")
            self.assertTrue(record["proxy"]["colliders"])

    def test_release_chain_references_environment_collision_contract(self) -> None:
        self.assertEqual(
            self.bundle["scene_mesh_profiles"], "configs/scene_mesh_profiles.json"
        )
        self.assertEqual(
            self.bundle["environment_composition"],
            "configs/visual_environment_composition.json",
        )
        self.assertEqual(
            self.bundle["environment_collision_proxies"],
            "configs/visual_environment_collision_proxies.json",
        )
        self.assertEqual(
            self.bundle["asset_proxy_registry"], "configs/asset_proxy_registry.json"
        )
        self.assertEqual(
            self.bundle["implementation"]["environment_collision"],
            "tools/assets/environment_collision.py",
        )
        self.assertEqual(
            self.matrix["dependencies"]["generic_sampling_bundle"],
            "configs/one_object_sampling_bundle.json",
        )
        self.assertEqual(
            self.matrix["dependencies"]["environment_composition"],
            self.bundle["environment_composition"],
        )
        self.assertEqual(
            self.matrix["dependencies"]["environment_collision_proxies"],
            self.bundle["environment_collision_proxies"],
        )

    def test_matrix_sampler_accepts_the_active_environment_contract(self) -> None:
        self.assertEqual(
            MATRIX_PATH,
            ROOT / "configs/one_object_sampling_matrix.json",
        )
        dependencies = matrix_dependency_paths(ROOT, self.matrix)
        self.assertEqual(
            dependencies["environment_collision_proxies"],
            ROOT / self.bundle["environment_collision_proxies"],
        )

    def test_visual_and_collision_mesh_share_one_frozen_world_pose(self) -> None:
        camera_axis = self.rules["axes"]["camera_axis"]
        compositions = {
            str(record["profile_id"]): record
            for record in self.compositions["records"]
        }
        for profile in self.profiles["profiles"]:
            composition = compositions[str(profile["id"])]
            if composition["review_status"] != "approved":
                continue
            profile = {**profile, "composition": composition}
            metadata = {
                "appearance": {"scene_visual": profile},
                "camera_request": {"profile": "front_left_oblique"},
                "simulation": {
                    "support": {
                        "scene_class": "ground_flat",
                        "dynamics": {
                            "lateral_friction": 0.7,
                            "restitution": 0.1,
                        },
                    },
                    "objects": [
                        {"initial_state": {"position_m": [0.2, -0.1, 0.4]}}
                    ],
                },
            }
            binding = compile_environment_binding(metadata, camera_axis)
            metadata["environment_binding"] = binding
            validate_environment_binding(metadata)
            visual = next(
                record
                for record in binding["visual_objects"]
                if record["primitive"] == "mesh"
            )
            collision = next(
                record
                for record in binding["colliders"]
                if record["primitive"] == "static_concave_mesh"
            )
            self.assertEqual(visual["position_m"], collision["position_m"])
            self.assertEqual(
                visual["rotation_euler_degrees"],
                collision["rotation_euler_degrees"],
            )
            camera_azimuth = next(
                float(record["overrides"]["view_rule"]["azimuth_degrees"])
                for record in camera_axis
                if record["label"] == "front_left_oblique"
            )
            reviewed_frame_world_yaw = (
                camera_azimuth
                - float(composition["camera"]["preferred_local_azimuth_degrees"])
            )
            expected_asset_world_yaw = (
                reviewed_frame_world_yaw
                + float(profile["asset"].get("review_yaw_degrees", 0.0))
            )
            self.assertAlmostEqual(
                float(visual["rotation_euler_degrees"][2]),
                expected_asset_world_yaw,
            )
            self.assertAlmostEqual(
                float(binding["placement"]["reviewed_frame_world_yaw_degrees"]),
                reviewed_frame_world_yaw,
            )
            self.assertAlmostEqual(
                float(binding["placement"]["asset_world_yaw_degrees"]),
                expected_asset_world_yaw,
            )
            self.assertEqual(
                visual["transform_contract"], collision["transform_contract"]
            )
            yaw = math.radians(float(visual["rotation_euler_degrees"][2]))
            anchor = composition["action_surface"]["anchor_local_m"]
            mapped_anchor = [
                float(visual["position_m"][0])
                + math.cos(yaw) * float(anchor[0])
                - math.sin(yaw) * float(anchor[1]),
                float(visual["position_m"][1])
                + math.sin(yaw) * float(anchor[0])
                + math.cos(yaw) * float(anchor[1]),
                float(visual["position_m"][2]) + float(anchor[2]),
            ]
            self.assertTrue(
                np.allclose(mapped_anchor, [0.2, -0.1, 0.0], atol=1.0e-9)
            )
            self.assertEqual(binding_sha256(binding), binding["binding_sha256"])

    def test_room_wall_moves_beyond_a_toward_wall_motion_envelope(self) -> None:
        metadata = {
            "appearance": {
                "scene_visual": {
                    "id": "test_room",
                    "visual_type": "procedural_room",
                    "back_wall_distance_m": 2.0,
                    "wall_enabled": True,
                    "decor": [],
                    "set_pieces": [],
                }
            },
            "camera_request": {"profile": "rear_oblique"},
            "simulation": {
                "time": {"duration_s": 4.0},
                "support": {
                    "scene_class": "ground_feature",
                    "surface_frame": {"slope_angle_degrees": 14.0},
                    "dynamics": {
                        "lateral_friction": 0.7,
                        "restitution": 0.1,
                    },
                },
                "objects": [
                    {
                        "geometry": {"size_m": [0.2, 0.2, 0.2]},
                        "initial_state": {
                            "position_m": [0.0, 0.0, 0.2],
                            "linear_velocity_m_s": [0.0, -0.8, 0.0],
                        },
                        "expected_motion": {
                            "minimum_downhill_displacement_m": 0.4
                        },
                    }
                ],
            },
        }
        camera_axis = self.rules["axes"]["camera_axis"]
        binding = compile_environment_binding(metadata, camera_axis)
        outward = binding["placement"]["outward_direction_xy"]
        expected = dynamic_back_wall_clearance_m(metadata, outward)
        self.assertGreater(expected, 6.0)
        self.assertAlmostEqual(
            binding["placement"]["dynamic_back_wall_clearance_m"], expected
        )
        self.assertAlmostEqual(
            binding["placement"]["back_wall_distance_m"], expected
        )

    def test_procedural_set_piece_moves_outside_the_motion_capsule(self) -> None:
        metadata = {
            "appearance": {
                "scene_visual": {
                    "id": "test_room",
                    "visual_type": "procedural_room",
                    "back_wall_distance_m": 3.0,
                    "wall_enabled": False,
                    "decor": [],
                    "set_pieces": [
                        {
                            "id": "blocking_piece",
                            "size_m": [1.0, 0.6, 1.0],
                            "offset_lateral_outward_z": [0.0, -1.0, 0.5],
                            "material_role": "support_structure",
                        }
                    ],
                }
            },
            "camera_request": {"profile": "rear_oblique"},
            "simulation": {
                "time": {"duration_s": 4.0},
                "support": {
                    "scene_class": "ground_flat",
                    "surface_frame": {"slope_angle_degrees": 0.0},
                    "dynamics": {
                        "lateral_friction": 0.7,
                        "restitution": 0.1,
                    },
                },
                "objects": [
                    {
                        "geometry": {"size_m": [0.2, 0.2, 0.2]},
                        "initial_state": {
                            "position_m": [0.0, 0.0, 0.2],
                            "linear_velocity_m_s": [0.0, -1.0, 0.0],
                        },
                    }
                ],
            },
        }
        self.assertIsNotNone(dynamic_motion_lane(metadata))
        binding = compile_environment_binding(
            metadata, self.rules["axes"]["camera_axis"]
        )
        piece = next(
            record
            for record in binding["colliders"]
            if record["id"] == "environment_piece_blocking_piece"
        )
        self.assertGreater(piece["dynamic_lane_shift_m"], 0.0)

    def test_only_human_reviewed_integrated_environments_are_admitted(self) -> None:
        records = self.compositions["records"]
        self.assertEqual(
            self.compositions["policy"]["integrated_environment_scene_classes"],
            ["ground_flat"],
        )
        self.assertEqual(
            self.compositions["policy"]["portable_ground_feature_visual_policy"],
            "forbid_in_integrated_environment_v1",
        )
        self.assertEqual(len(records), 20)
        self.assertEqual(
            sum(record["review_status"] == "approved" for record in records),
            8,
        )
        self.assertEqual(
            sum(record["review_status"] == "paused" for record in records),
            12,
        )
        for record in records:
            if record["review_status"] == "approved":
                self.assertEqual(record["composition_mode"], "integrated_ground")
                self.assertTrue(
                    all(
                        support_id
                        not in {"ground_ramp_short_steep", "ground_channel_ramp"}
                        for binding in record["bindings"]
                        for support_id in binding["support_ids"]
                    )
                )
                self.assertGreater(
                    record["action_surface"]["audited_clear_radius_m"]
                    - record["action_surface"]["admitted_action_radius_m"],
                    0.119,
                )
                camera = record["camera"]
                self.assertLessEqual(
                    camera["maximum_local_azimuth_deviation_degrees"], 12.0
                )
                self.assertLessEqual(
                    camera["minimum_elevation_degrees"],
                    camera["preferred_elevation_degrees"],
                )
                self.assertLessEqual(
                    camera["preferred_elevation_degrees"],
                    camera["maximum_elevation_degrees"],
                )
                self.assertGreaterEqual(camera["maximum_distance_m"], 3.65)
                self.assertLessEqual(
                    camera["minimum_distance_m"], camera["maximum_distance_m"]
                )
                self.assertGreaterEqual(camera["target_depth_offset_m"], 0.0)
                self.assertGreaterEqual(camera["focal_length_cap_mm"], 20.0)
            else:
                self.assertTrue(record["reason"])

    def test_dining_modern_uses_reviewed_drop_only_action_patch(self) -> None:
        record = next(
            value
            for value in self.compositions["records"]
            if value["profile_id"] == "mesh_env_dining_modern"
        )
        self.assertEqual(record["review_status"], "approved")
        self.assertEqual(
            record["action_surface"],
            {
                "anchor_local_m": [-0.001026, -2.088147, 0.599033],
                "audited_clear_radius_m": 0.68,
                "admitted_action_radius_m": 0.52,
            },
        )
        self.assertEqual(record["camera"]["reviewed_view"], "south")
        self.assertEqual(
            record["bindings"],
            [
                {
                    "support_ids": ["wood_floor"],
                    "motion_families": ["drop_fall_1obj"],
                }
            ],
        )

    def test_kitchen_modern_uses_reviewed_main_floor_drop_patch(self) -> None:
        record = next(
            value
            for value in self.compositions["records"]
            if value["profile_id"] == "mesh_env_kitchen_modern"
        )
        profile = next(
            value
            for value in self.profiles["profiles"]
            if value["id"] == "mesh_env_kitchen_modern"
        )
        proxy = next(
            value
            for value in self.proxies["records"]
            if value["asset_id"] == "env_kitchen_modern_9843a830"
        )
        self.assertEqual(record["review_status"], "approved")
        self.assertEqual(
            record["action_surface"],
            {
                "anchor_local_m": [-0.804596, -1.335845, 0.064868],
                "audited_clear_radius_m": 0.84,
                "admitted_action_radius_m": 0.68,
            },
        )
        self.assertEqual(record["camera"]["reviewed_view"], "south")
        self.assertEqual(
            record["bindings"],
            [
                {
                    "support_ids": ["concrete_floor_mat"],
                    "motion_families": ["drop_fall_1obj"],
                }
            ],
        )
        self.assertEqual(
            profile["asset"]["floor_alignment"]["method"],
            "reviewed_horizontal_floor",
        )
        self.assertAlmostEqual(
            proxy["transform_contract"]["authoritative_floor_z_m"],
            0.06486835,
            places=8,
        )
        self.assertEqual(
            proxy["proxy"]["authoritative_floor_face_count_removed"],
            368,
        )

if __name__ == "__main__":
    unittest.main()
