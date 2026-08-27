from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.base_release_schema import (
    BASE_SAMPLE_SCHEMA,
    MASK_MANIFEST_SCHEMA,
    TRAJECTORY_FIELDS,
    TRAJECTORY_SCHEMA,
    build_base_metadata,
    build_mask_manifest,
    canonical_trajectory,
    _dynamic_material_extras,
    sha256,
    validate_base_metadata,
    write_deterministic_npz,
)


class BaseReleaseSchemaTests(unittest.TestCase):
    def test_adapter_specific_material_defaults_do_not_cross_pipelines(self) -> None:
        backend = {
            "asset_proxy_rules": {
                "contact": {
                    "dynamic_defaults": {
                        "rolling_friction": 0.0008,
                        "spinning_friction": 0.0003,
                        "linear_damping": 0.025,
                        "angular_damping": 0.035,
                    }
                }
            },
            "billiards_rules": {
                "ball_dynamics": {
                    "rolling_friction": 0.0025,
                    "spinning_friction": 0.0006,
                    "linear_damping": 0.025,
                    "angular_damping": 0.01,
                }
            },
        }
        resolved = {
            "backend_binding": {"adapter_id": "billiards_v4"},
            "adapter_payload": {"backend": backend},
        }
        self.assertEqual(
            _dynamic_material_extras({}, resolved, "cue_ball"),
            backend["billiards_rules"]["ball_dynamics"],
        )

    def source_trajectory(self, path: Path) -> None:
        time_s = np.asarray([0.0, 1.0], dtype=np.float64)
        position = np.zeros((2, 1, 3), dtype=np.float64)
        quaternion = np.zeros((2, 1, 4), dtype=np.float64)
        quaternion[:, :, 0] = 1.0
        np.savez_compressed(
            path,
            schema_version=np.asarray("physweep_object_trajectory_v2"),
            object_ids=np.asarray(["ball"]),
            time_s=time_s,
            position_m=position,
            quaternion_wxyz=quaternion,
            linear_velocity_m_s=np.zeros_like(position),
            angular_velocity_rad_s=np.zeros_like(position),
            contact_count=np.zeros((2, 1), dtype=np.int64),
            runtime_material=np.asarray([[0.2, 0.3, 0.4]], dtype=np.float64),
            inertia_diagonal_kg_m2=np.asarray(
                [[0.001, 0.001, 0.001]], dtype=np.float64
            ),
            adapter__position_m=position,
            adapter__quaternion_xyzw=quaternion[:, :, [1, 2, 3, 0]],
        )

    def test_trajectory_is_minimal_and_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.npz"
            self.source_trajectory(source)
            arrays, info = canonical_trajectory(source)
            self.assertEqual(tuple(arrays), TRAJECTORY_FIELDS)
            self.assertEqual(str(arrays["schema_version"]), TRAJECTORY_SCHEMA)
            self.assertEqual(info["object_ids"], ["ball"])
            first = root / "first.npz"
            second = root / "second.npz"
            write_deterministic_npz(first, arrays)
            write_deterministic_npz(second, arrays)
            self.assertEqual(sha256(first), sha256(second))
            with np.load(first, allow_pickle=False) as archive:
                self.assertEqual(tuple(archive.files), TRAJECTORY_FIELDS)
                self.assertNotIn("runtime_material", archive.files)
                self.assertFalse(any(key.startswith("adapter__") for key in archive.files))

    def test_base_metadata_uses_real_object_axis_and_final_camera(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.npz"
            self.source_trajectory(source_path)
            _, trajectory = canonical_trajectory(source_path)
            source = {
                "schema_version": "source_v1",
                "scene_id": "scene__base",
                "seed": 7,
                "semantics": {
                    "scene_family": "test_fixture",
                    "profile": "drop",
                    "description": "One ball drops.",
                    "dynamic_object_count": 1,
                },
                "simulation": {
                    "objects": [
                        {
                            "object_id": "ball",
                            "visual": {"shape": "sphere", "radius_m": 0.1},
                            "material": {
                                "rolling_friction": 0.01,
                                "spinning_friction": 0.02,
                                "linear_damping": 0.03,
                                "angular_damping": 0.04,
                            },
                        }
                    ]
                },
                "render": {"resolution": [1280, 720], "samples": 16},
                "object_identity": {
                    "objects": [
                        {
                            "object_id": "ball",
                            "role": "dynamic",
                            "semantic_label": "ball",
                            "asset_id": None,
                            "mask_instance_id": 1,
                            "trajectory_key": "ball",
                            "mask_key": "ball",
                        }
                    ],
                    "text": {"caption": "The ball drops."},
                },
            }
            resolved = {
                "scene_id": "scene__base",
                "backend_binding": {
                    "backend_id": "pybullet_rigid",
                    "adapter_id": "test_v1",
                },
                "time": {
                    "duration_s": 1.0,
                    "output_fps": 1,
                    "simulation_hz": 10,
                    "frame_count": 2,
                },
                "world": {"gravity_m_s2": [0.0, 0.0, -9.81]},
                "objects": [
                    {
                        "object_id": "ball",
                        "object_index": 0,
                        "collision_proxy": {"type": "sphere", "radius_m": 0.1},
                        "material": {
                            "mass_kg": 0.2,
                            "contact_friction": 0.3,
                            "contact_restitution": 0.4,
                        },
                        "initial_state": {
                            "position_m": [0.0, 0.0, 1.0],
                            "orientation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
                            "linear_velocity_m_s": [0.0, 0.0, 0.0],
                            "angular_velocity_rad_s": [0.0, 0.0, 0.0],
                        },
                    }
                ],
            }
            render_record = {
                "scene_id": "scene__base",
                "camera": {
                    "seed": 7,
                    "mode": "front",
                    "position_m": [2.0, -2.0, 1.5],
                    "target_m": [0.0, 0.0, 0.5],
                    "focal_length_mm": 50.0,
                    "sensor_width_mm": 36.0,
                    "diagnostics": {"attempts": 100},
                },
            }
            metadata = build_base_metadata(
                family="test",
                group_id="logical_scene",
                source=source,
                source_metadata_sha256="a" * 64,
                resolved_scene=resolved,
                render_record=render_record,
                render_metadata=None,
                trajectory_info=trajectory,
                trajectory_sha256="b" * 64,
                video_sha256="c" * 64,
            )
            metadata["artifacts"]["masks"] = {
                "manifest_sha256": "d" * 64,
            }
            summary = validate_base_metadata(metadata)
            self.assertEqual(metadata["schema_version"], BASE_SAMPLE_SCHEMA)
            mask_binding = metadata["artifacts"].pop("masks")
            with self.assertRaisesRegex(ValueError, "mask binding"):
                validate_base_metadata(metadata)
            metadata["artifacts"]["masks"] = mask_binding
            self.assertNotIn("kind", metadata)
            metadata["kind"] = "base"
            with self.assertRaisesRegex(ValueError, "base fields"):
                validate_base_metadata(metadata)
            del metadata["kind"]
            self.assertEqual(summary["object_ids"], ["ball"])
            obj = metadata["physics"]["objects"][0]
            self.assertEqual(obj["array_index"], 0)
            self.assertTrue(obj["object_valid"])
            self.assertEqual(obj["initial_state"]["quaternion_wxyz"], [1.0, 0.0, 0.0, 0.0])
            self.assertEqual(obj["inertia_diagonal_kg_m2"], [0.001, 0.001, 0.001])
            self.assertEqual(
                obj["material"],
                {
                    "mass_kg": 0.2,
                    "contact_friction": 0.3,
                    "contact_restitution": 0.4,
                    "rolling_friction": 0.01,
                    "spinning_friction": 0.02,
                    "linear_damping": 0.03,
                    "angular_damping": 0.04,
                },
            )
            linear_damping = obj["material"].pop("linear_damping")
            with self.assertRaisesRegex(ValueError, "dynamic material"):
                validate_base_metadata(metadata)
            obj["material"]["linear_damping"] = linear_damping
            self.assertNotIn("trajectory_key", json.dumps(metadata))
            self.assertNotIn("diagnostics", metadata["visual"]["camera"])
            self.assertEqual(
                set(metadata["visual"]["camera"]),
                {
                    "position_m",
                    "target_m",
                    "focal_length_mm",
                    "sensor_width_mm",
                    "clip_start_m",
                    "clip_end_m",
                },
            )
            self.assertEqual(metadata["visual"]["camera"]["clip_start_m"], 0.03)
            self.assertEqual(metadata["visual"]["camera"]["clip_end_m"], 100.0)
            self.assertNotIn("resolution", metadata["visual"])
            self.assertEqual(metadata["visual"]["render_samples"], 16)
            self.assertNotIn("frame_count", metadata["physics"]["time"])
            self.assertNotIn("dynamic_object_count", metadata["semantics"])
            self.assertNotIn("scene_family", metadata["semantics"])
            metadata["artifacts"]["trajectory"]["path"] = "../trajectory.npz"
            with self.assertRaisesRegex(ValueError, "trajectory binding"):
                validate_base_metadata(metadata)
            del metadata["artifacts"]["trajectory"]["path"]
            camera_without_sensor = dict(render_record)
            camera_without_sensor["camera"] = {
                key: value
                for key, value in render_record["camera"].items()
                if key != "sensor_width_mm"
            }
            with self.assertRaisesRegex(ValueError, "sensor_width_mm"):
                build_base_metadata(
                    family="test",
                    group_id="logical_scene",
                    source=source,
                    source_metadata_sha256="a" * 64,
                    resolved_scene=resolved,
                    render_record=camera_without_sensor,
                    render_metadata=None,
                    trajectory_info=trajectory,
                    trajectory_sha256="b" * 64,
                    video_sha256="c" * 64,
                )

    def test_mask_manifest_uses_object_axis_and_ordered_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask_dir = root / "ball"
            mask_dir.mkdir()
            for index in (1, 2):
                (mask_dir / f"frame_{index:04d}.png").write_bytes(
                    f"frame-{index}".encode("ascii")
                )
            manifest = build_mask_manifest(
                scene_id="scene__base",
                mask_root=root,
                objects=[{"object_id": "ball", "mask_instance_id": 1}],
            )
            self.assertEqual(manifest["schema_version"], MASK_MANIFEST_SCHEMA)
            self.assertEqual(manifest["frame_count"], 2)
            self.assertEqual(len(manifest["objects"][0]["frame_sha256"]), 2)
            self.assertNotIn("filename", json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "invalid mask object id"):
                build_mask_manifest(
                    scene_id="scene__base",
                    mask_root=root,
                    objects=[{"object_id": "../ball", "mask_instance_id": 1}],
                )
            with self.assertRaisesRegex(ValueError, "invalid mask object id"):
                build_mask_manifest(
                    scene_id="scene__base",
                    mask_root=root,
                    objects=[{"object_id": None, "mask_instance_id": 1}],
                )


if __name__ == "__main__":
    unittest.main()
