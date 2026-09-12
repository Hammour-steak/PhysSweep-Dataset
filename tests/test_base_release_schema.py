from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.core.hashing import sha256_json
from tools.release.base_release_schema import (
    BASE_SAMPLE_SCHEMA,
    SAMPLE_METADATA_FIELDS,
    SWEEP_SAMPLE_SCHEMA,
    TRAJECTORY_FIELDS,
    TRAJECTORY_SCHEMA,
    build_base_metadata,
    build_sweep_metadata,
    canonical_trajectory,
    _compact_generic_object_annotations,
    _clean_text,
    _dynamic_material_extras,
    _fixture_descriptor,
    _compact_objects,
    _solver_contract,
    build_fixture_payload,
    sha256,
    validate_base_metadata,
    validate_sweep_metadata,
    write_deterministic_npz,
)


class BaseReleaseSchemaTests(unittest.TestCase):
    def test_two_object_billiards_reuses_fixture_dynamics_and_solver_contracts(self):
        dynamics = {"rolling_friction": 0.01, "spinning_friction": 0.02, "linear_damping": 0.03, "angular_damping": 0.04}
        engine = {"solver_iterations": 100, "restitution_velocity_threshold_m_s": 0.02, "enable_cone_friction": True, "use_split_impulse": True}
        payload = {"static_support_binding": {"representation": "mesh"}, "backend": {
            "billiards_rules": {"ball_dynamics": dynamics, "engine": engine, "support_dynamics": {"lateral_friction": 0.3, "restitution": 0.4}}}}
        source = {"object_identity": {"objects": [{"object_id": name, "semantic_label": "ball"} for name in ("object_a", "object_b")]}}
        scene = {"backend_binding": {"adapter_id": "billiards_two_object_v1"}, "adapter_payload": payload,
                 "objects": [{"object_id": name, "collision_proxy": {"type": "sphere", "size_m": [0.1]*3},
                              "material": {"mass_kg": 0.2, "contact_friction": 0.3, "contact_restitution": 0.4},
                              "initial_state": {"position_m": [index,0,1], "orientation_quaternion_xyzw": [0,0,0,1],
                                                "linear_velocity_m_s": [0,0,0], "angular_velocity_rad_s": [0,0,0]}}
                             for index,name in enumerate(("object_a", "object_b"))]}
        for adapter in ("billiards_v4", "billiards_two_object_v1"):
            self.assertEqual(build_fixture_payload(adapter, source, scene)["physical"], {
                "static_support_binding": payload["static_support_binding"],
                "support_dynamics": payload["backend"]["billiards_rules"]["support_dynamics"]})
            self.assertEqual(_solver_contract(adapter, source, scene), {**engine, "deterministic_overlapping_pairs": True})
        self.assertEqual(_dynamic_material_extras(source, scene, "object_a"), dynamics)
        objects, _ = _compact_objects("billiards", source, scene, {
            "object_ids": ["object_a", "object_b"], "runtime_material": [[0.2,0.3,0.4]]*2,
            "inertia_diagonal_kg_m2": [[0.001]*3]*2}, "table", {
                "table": {"cue_ball": "unused", "object_ball_2": "red", "object_ball_1": "yellow"}})
        self.assertEqual([obj["visual"]["material_template"]["source_object_name"] for obj in objects], ["yellow", "red"])
        dynamics["contact_processing_threshold_m"] = 0.02
        objects, _ = _compact_objects("billiards", source, scene, {
            "object_ids": ["object_a", "object_b"], "runtime_material": [[0.2,0.3,0.4]]*2,
            "inertia_diagonal_kg_m2": [[0.001]*3]*2}, "table", {
                "table": {"cue_ball": "unused", "object_ball_2": "red", "object_ball_1": "yellow"}})
        self.assertEqual([obj["material"]["contact_processing_threshold_m"] for obj in objects], [.02, .02])

    def test_fixture_binding_comes_from_metadata_without_a_redundant_render_hash(self):
        for family in ("passive_pinball", "marble_run"):
            fixture = {"representation": "analytic", "colliders": [{"id": "board"}]}
            source = {"semantics": {"profile": family}, "physics": {"fixture": fixture}}
            expected = sha256_json(fixture)
            with self.subTest(family=family):
                identity, digest = _fixture_descriptor(source, {})
                self.assertEqual(identity, {"id": family, "representation": "analytic"})
                self.assertEqual(digest, expected)
                self.assertEqual(_fixture_descriptor(source, {"fixture_sha256": expected}), (identity, digest))
                fixture["colliders"][0]["id"] = "changed_board"
                self.assertNotEqual(_fixture_descriptor(source, {})[1], digest)
        with self.assertRaisesRegex(ValueError, "source binding hash is incomplete"):
            _fixture_descriptor({"semantics": {"profile": "missing"}, "simulation": {
                "support": {"collision_authority": "analytic"}}}, {})

    def test_text_binding_supports_two_objects_with_the_same_label(self) -> None:
        source = {
            "object_identity": {
                "text": {
                    "caption": (
                        "the physassets 1 barrel and the physassets 1 barrel "
                        "move separately."
                    ),
                    "object_mentions": [
                        {
                            "object_id": "object_a",
                            "text": "the physassets 1 barrel",
                        },
                        {
                            "object_id": "object_b",
                            "text": "the physassets 1 barrel",
                        },
                    ],
                }
            }
        }
        text = _clean_text(
            "generic",
            source,
            {"object_a": "barrel", "object_b": "barrel"},
        )
        self.assertEqual(
            text["caption"], "the barrel and the barrel move separately."
        )
        self.assertEqual(
            [record["char_span"] for record in text["object_mentions"]],
            [[0, 10], [15, 25]],
        )

    def test_generic_semantics_bind_each_object_explicitly(self) -> None:
        source = {
            "semantic_sampling": {
                "five_dimensions": {
                    "foreground_objects": [
                        {
                            "object_id": "object_a",
                            "semantic_category": "box",
                            "scale_bin": "small",
                            "uniform_scale": 0.8,
                        },
                        {
                            "object_id": "object_b",
                            "semantic_category": "ball",
                            "scale_bin": "medium",
                            "uniform_scale": 1.0,
                        },
                    ]
                }
            }
        }
        annotations = _compact_generic_object_annotations(
            source, ["object_a", "object_b"]
        )
        self.assertEqual(list(annotations), ["object_a", "object_b"])
        self.assertEqual(annotations["object_b"]["semantic_category"], "ball")
        with self.assertRaisesRegex(ValueError, "order differs"):
            _compact_generic_object_annotations(source, ["object_b", "object_a"])

    def test_generic_multi_object_semantics_reject_singular_1obj_field(self) -> None:
        source = {
            "semantic_sampling": {
                "five_dimensions": {
                    "foreground_object": {
                        "semantic_category": "box",
                        "scale_bin": "small",
                        "uniform_scale": 0.8,
                    }
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "requires foreground_objects"):
            _compact_generic_object_annotations(source, ["object_a", "object_b"])

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
                            "visual": {
                                "shape": "sphere",
                                "radius_m": 0.1,
                                "color_rgba": [0.8, 0.2, 0.1, 1.0],
                                "unused_generation_field": "drop-me",
                            },
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
                "physics": {
                    "fixture": {"representation": "analytic"},
                    "engine": {
                        "solver_iterations": 50,
                        "deterministic_overlapping_pairs": True,
                        "restitution_velocity_threshold_m_s": 0.02,
                        "enable_cone_friction": True,
                        "use_split_impulse": True,
                        "unused_generation_field": "drop-me",
                    },
                },
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
                    "text": {
                        "caption": "The ball drops.",
                        "object_mentions": [
                            {"object_id": "ball", "text": "The ball"}
                        ],
                    },
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
                "world": {
                    "gravity_m_s2": [0.0, 0.0, -9.81],
                    "unused_generation_field": "drop-me",
                },
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
                "fixture_sha256": "e" * 64,
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
                fixture_sha256="f" * 64,
            )
            summary = validate_base_metadata(metadata)
            self.assertEqual(metadata["schema_version"], BASE_SAMPLE_SCHEMA)
            self.assertEqual(set(metadata), SAMPLE_METADATA_FIELDS["base"])
            sweep_metadata = build_sweep_metadata(
                sweep={
                    "target_object_id": "ball",
                    "parameter": "mass_kg",
                    "level_index": 0,
                    "value": 0.2,
                },
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
                fixture_sha256="f" * 64,
            )
            sweep_summary = validate_sweep_metadata(sweep_metadata)
            self.assertEqual(sweep_metadata["schema_version"], SWEEP_SAMPLE_SCHEMA)
            self.assertEqual(set(sweep_metadata), SAMPLE_METADATA_FIELDS["sweep"])
            self.assertEqual(sweep_metadata["sample_kind"], "sweep")
            self.assertEqual(sweep_summary["group_id"], "logical_scene")
            sweep_metadata["sweep"]["value"] = 0.3
            with self.assertRaisesRegex(ValueError, "differs from object material"):
                validate_sweep_metadata(sweep_metadata)
            self.assertEqual(metadata["sample_kind"], "base")
            metadata["kind"] = "base"
            with self.assertRaisesRegex(ValueError, "base fields"):
                validate_base_metadata(metadata)
            del metadata["kind"]
            self.assertEqual(summary["object_ids"], ["ball"])
            obj = metadata["physics"]["objects"][0]
            self.assertNotIn("array_index", obj)
            self.assertNotIn("semantic_label", obj)
            self.assertEqual(
                metadata["semantics"]["objects"],
                [{"object_id": "ball", "semantic_label": "ball"}],
            )
            self.assertTrue(obj["object_valid"])
            self.assertEqual(
                obj["visual"],
                {"base_color_srgb_rgba": [0.8, 0.2, 0.1, 1.0]},
            )
            self.assertEqual(
                metadata["physics"]["world"],
                {"gravity_m_s2": [0.0, 0.0, -9.81]},
            )
            self.assertEqual(
                set(metadata["physics"]["solver"]),
                {
                    "solver_iterations",
                    "deterministic_overlapping_pairs",
                    "restitution_velocity_threshold_m_s",
                    "enable_cone_friction",
                    "use_split_impulse",
                },
            )
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
                    "contact_processing_threshold_m": 0.0,
                },
            )
            original_adapter = metadata["physics"]["backend"]["adapter_id"]
            obj["material"]["contact_processing_threshold_m"] = .02
            with self.assertRaisesRegex(ValueError, "dynamic material"):
                validate_base_metadata(metadata)
            for adapter in ("billiards_v4", "billiards_two_object_v1"):
                metadata["physics"]["backend"]["adapter_id"] = adapter
                validate_base_metadata(metadata)
            metadata["physics"]["backend"]["adapter_id"] = original_adapter
            obj["material"]["contact_processing_threshold_m"] = 0.0
            linear_damping = obj["material"].pop("linear_damping")
            with self.assertRaisesRegex(ValueError, "dynamic material"):
                validate_base_metadata(metadata)
            obj["material"]["linear_damping"] = linear_damping
            obj["unused_generation_field"] = "must-not-pass"
            with self.assertRaisesRegex(ValueError, "object fields"):
                validate_base_metadata(metadata)
            del obj["unused_generation_field"]
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
                    fixture_sha256="f" * 64,
                )


if __name__ == "__main__":
    unittest.main()
