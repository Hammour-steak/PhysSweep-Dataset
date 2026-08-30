from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from tools.core.hashing import sha256_file
from tools.sampling.object_collection import compile_object_collection_scene
from tools.sampling.sample_two_object_base import _validated_intents
from tools.sampling.sample_two_object_coverage import (
    _axis_counts,
    coverage_cells,
    released_source_pool,
    select_coverage_sources,
)
from tools.sampling.two_object_sources import _asset_object_template


ROOT = Path(__file__).resolve().parents[1]


def load_matrix() -> dict:
    return json.loads(
        (ROOT / "configs/two_object_sampling_matrix.json").read_text(
            encoding="utf-8"
        )
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


class TwoObjectCoverageTests(unittest.TestCase):
    def test_asset_source_adapter_accepts_only_exact_single_primitive(self) -> None:
        matrix = load_matrix()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            registry_path = root / "registry.json"
            visual_path = root / "assets/phone.glb"
            visual_path.parent.mkdir(parents=True)
            visual_path.write_bytes(b"unit asset mesh")
            asset_id = "asset_phone"
            registry = {
                "records": [
                    {
                        "asset_id": asset_id,
                        "proxy": {
                            "colliders": [
                                {
                                    "shape": "box",
                                    "size_m": [0.16, 0.08, 0.01],
                                    "position_m": [0.0, 0.0, 0.0],
                                    "rotation_euler_degrees": [0.0, 0.0, 0.0],
                                }
                            ]
                        },
                        "visual": {
                            "path": "assets/phone.glb",
                            "sha256": sha256_file(visual_path),
                            "canonical_extent_m": [0.16, 0.08, 0.01],
                            "alignment_euler_degrees": [0.0, 0.0, 0.0],
                        },
                    }
                ]
            }
            write_json(registry_path, registry)
            generation = {
                "scene_id": "asset_source",
                "registry": {
                    "path": registry_path.relative_to(root).as_posix(),
                    "sha256": sha256_file(registry_path),
                },
                "assets": {"dynamic_asset_id": asset_id},
            }
            material = {
                "mass_kg": 0.2,
                "contact_friction": 0.3,
                "contact_restitution": 0.1,
                "linear_damping": 0.02,
                "angular_damping": 0.03,
                "rolling_friction": 0.0,
                "spinning_friction": 0.0,
            }
            released = {
                "physics": {"objects": [{"asset_id": asset_id, "material": material}]},
                "semantics": {
                    "objects": [
                        {"object_id": "object_a", "semantic_label": "smartphone"}
                    ]
                },
            }
            template, reason = _asset_object_template(
                source_root=root,
                runtime_root=root,
                generation_metadata=generation,
                release_metadata=released,
                eligibility=matrix["candidate_pool"]["object_eligibility"],
                registry_cache={},
                visual_hash_cache={},
            )
            self.assertEqual(reason, "eligible")
            obj = template["simulation"]["objects"][0]
            self.assertEqual(obj["geometry"]["type"], "cuboid")
            self.assertEqual(obj["visual_profile"]["type"], "mesh")
            self.assertEqual(
                template["semantic_sampling"]["five_dimensions"][
                    "foreground_object"
                ]["scale_bin"],
                "small",
            )
            visual_path.write_bytes(b"tampered unit asset mesh")
            with self.assertRaisesRegex(ValueError, "visual hash mismatch"):
                _asset_object_template(
                    source_root=root,
                    runtime_root=root,
                    generation_metadata=generation,
                    release_metadata=released,
                    eligibility=matrix["candidate_pool"]["object_eligibility"],
                    registry_cache={},
                    visual_hash_cache={},
                )
            visual_path.write_bytes(b"unit asset mesh")
            registry["records"][0]["proxy"]["colliders"].append(
                copy.deepcopy(registry["records"][0]["proxy"]["colliders"][0])
            )
            write_json(registry_path, registry)
            generation["registry"]["sha256"] = sha256_file(registry_path)
            rejected, reason = _asset_object_template(
                source_root=root,
                runtime_root=root,
                generation_metadata=generation,
                release_metadata=released,
                eligibility=matrix["candidate_pool"]["object_eligibility"],
                registry_cache={},
                visual_hash_cache={},
            )
            self.assertIsNone(rejected)
            self.assertEqual(reason, "compound_proxy")

    def test_matrix_declares_complete_ordered_cartesian_coverage(self) -> None:
        matrix = load_matrix()
        cells, full_count = coverage_cells(matrix)
        self.assertEqual(full_count, 1134)
        self.assertEqual(len(cells), 1134)
        counts = _axis_counts(cells)
        self.assertEqual(set(counts["motion"].values()), {54, 162})
        self.assertEqual(
            set(counts["ordered_shape_pair"].values()), {108, 126, 144, 162}
        )
        self.assertEqual(set(counts["ordered_scale_pair"].values()), {126})
        self.assertEqual(set(counts["scene_class"].values()), {567})
        self.assertEqual(set(counts["camera_view_family"].values()), {189})
        family_ids = set(counts["camera_view_family"])
        for axis in (
            "motion_id",
            "shape_pair_id",
            "scale_pair_id",
            "scene_class",
        ):
            conditional = defaultdict(Counter)
            for cell in cells:
                conditional[str(cell[axis])][
                    str(cell["camera_view_family_id"])
                ] += 1
            self.assertTrue(
                all(
                    set(axis_counts) == family_ids
                    for axis_counts in conditional.values()
                ),
                axis,
            )
            maximum_spread = max(
                max(counts.values()) - min(counts.values())
                for counts in conditional.values()
            )
            self.assertLessEqual(maximum_spread, 3, axis)

    def test_balanced_smoke_prefix_covers_every_axis(self) -> None:
        cells, full_count = coverage_cells(load_matrix(), 72)
        self.assertEqual(full_count, 1134)
        counts = _axis_counts(cells)
        self.assertLessEqual(
            max(counts["motion"].values()) - min(counts["motion"].values()),
            2,
        )
        self.assertEqual(set(counts["ordered_shape_pair"].values()), {8})
        self.assertEqual(set(counts["ordered_scale_pair"].values()), {8})
        self.assertEqual(set(counts["scene_class"].values()), {36})
        self.assertEqual(set(counts["camera_view_family"].values()), {12})

    def test_matrix_rejects_missing_scale_cell_and_weakened_uniqueness(self) -> None:
        matrix = load_matrix()
        missing = copy.deepcopy(matrix)
        missing["coverage_plan"]["role_ordered_scale_pairs"].pop()
        with self.assertRaisesRegex(ValueError, "full product"):
            _validated_intents(missing)
        missing_shape = copy.deepcopy(matrix)
        missing_shape["coverage_plan"]["role_ordered_shape_pairs"].pop()
        with self.assertRaisesRegex(ValueError, "full product"):
            _validated_intents(missing_shape)
        unknown_shape_pair = copy.deepcopy(matrix)
        unknown_shape_pair["shape_motion_compatibility"]["pair_sets"][
            "all_shape_pairs"
        ].append("unknown_to_sphere")
        with self.assertRaisesRegex(ValueError, "shape-pair sets"):
            _validated_intents(unknown_shape_pair)
        unsupported_host = copy.deepcopy(matrix)
        unsupported_host["candidate_pool"]["host_eligibility"][
            "allowed_visual_types"
        ].append("mesh_environment")
        with self.assertRaisesRegex(ValueError, "reviewed procedural"):
            _validated_intents(unsupported_host)
        weakened = copy.deepcopy(matrix)
        weakened["coverage_plan"]["selection_policy"][
            "source_pair_uniqueness"
        ] = "allow_reuse"
        with self.assertRaisesRegex(ValueError, "may not be weakened"):
            _validated_intents(weakened)
        excessive_reuse = copy.deepcopy(matrix)
        excessive_reuse["coverage_plan"]["selection_policy"][
            "maximum_host_source_reuse"
        ] = 3
        with self.assertRaisesRegex(ValueError, "may not be weakened"):
            _validated_intents(excessive_reuse)
        active_host = copy.deepcopy(matrix)
        active_host["candidate_pool"]["host_eligibility"][
            "allowed_collider_roles"
        ].append("impact_wall")
        with self.assertRaisesRegex(ValueError, "motion-neutral"):
            _validated_intents(active_host)
        bounded_host = copy.deepcopy(matrix)
        bounded_host["candidate_pool"]["host_eligibility"][
            "camera_envelope_policy"
        ] = "allow_bounded"
        with self.assertRaisesRegex(ValueError, "host-eligibility"):
            _validated_intents(bounded_host)

    def test_source_selection_obeys_scale_and_uniqueness(self) -> None:
        matrix = load_matrix()
        cells, _ = coverage_cells(matrix, 18)
        objects = []
        for source_family in ("generic", "asset"):
            for shape in ("sphere", "cuboid", "cylinder"):
                for scale in ("small", "medium", "large"):
                    for profile_index in range(3):
                        for source_index in range(4):
                            source_id = (
                                f"object_{source_family}_{shape}_{scale}_"
                                f"{profile_index}_{source_index}"
                            )
                            objects.append(
                                {
                                    "metadata": {},
                                    "source": {"scene_id": source_id},
                                    "source_family": source_family,
                                    "shape_family_id": shape,
                                    "scale_bin": scale,
                                    "visual_profile_id": (
                                        f"{source_family}_{shape}_profile_"
                                        f"{profile_index}"
                                    ),
                                }
                            )
        hosts = []
        for scene_class in ("ground_flat", "raised_flat"):
            for profile_index in range(3):
                for source_index in range(6):
                    source_id = (
                        f"host_{scene_class}_{profile_index}_{source_index}"
                    )
                    hosts.append(
                        {
                            "metadata": {},
                            "source": {"scene_id": source_id},
                            "scene_class": scene_class,
                            "visual_profile_id": f"host_profile_{profile_index}",
                            "visual_type": "procedural_room",
                        }
                    )
        selected, audit = select_coverage_sources(
            cells, objects, hosts, matrix
        )
        self.assertEqual(len(selected), 18)
        self.assertEqual(audit["unique_source_pair_count"], 18)
        self.assertEqual(audit["unique_host_count"], 18)
        self.assertLessEqual(audit["maximum_object_source_reuse"], 2)
        self.assertLessEqual(audit["maximum_host_source_reuse"], 2)
        self.assertEqual(
            set(audit["selected_source_family_pair_counts"]),
            {
                "generic_to_generic",
                "generic_to_asset",
                "asset_to_generic",
                "asset_to_asset",
            },
        )
        source_pair_counts = audit["selected_source_family_pair_counts"].values()
        self.assertLessEqual(max(source_pair_counts) - min(source_pair_counts), 1)
        self.assertTrue(
            all(
                max(counts.values()) - min(counts.values()) <= 1
                for counts in audit[
                    "selected_camera_source_family_pair_counts"
                ].values()
            ),
            audit["selected_camera_source_family_pair_counts"],
        )
        for selection in selected:
            cell = selection["cell"]
            left, right = selection["objects"]
            self.assertEqual(left["scale_bin"], cell["object_a_scale_bin"])
            self.assertEqual(right["scale_bin"], cell["object_b_scale_bin"])
            self.assertEqual(
                left["shape_family_id"], cell["object_a_shape"]
            )
            self.assertEqual(
                right["shape_family_id"], cell["object_b_shape"]
            )
            self.assertEqual(
                selection["host"]["scene_class"], cell["scene_class"]
            )

    def test_source_selection_allows_matching_size_and_appearance(self) -> None:
        cell = {
            "cell_id": "same_size_and_appearance",
            "object_a_shape": "sphere",
            "object_b_shape": "sphere",
            "object_a_scale_bin": "small",
            "object_b_scale_bin": "small",
            "scene_class": "ground_flat",
            "camera_view_family_id": "side_left_mid",
        }
        objects = [
            {
                "metadata": {},
                "source": {"scene_id": f"object_{index}"},
                "source_family": "generic",
                "shape_family_id": "sphere",
                "scale_bin": "small",
                "visual_profile_id": "shared_profile",
            }
            for index in range(2)
        ]
        hosts = [
            {
                "metadata": {},
                "source": {"scene_id": "host"},
                "scene_class": "ground_flat",
                "visual_profile_id": "host_profile",
                "visual_type": "procedural_room",
            }
        ]
        selected, _ = select_coverage_sources(
            [cell], objects, hosts, load_matrix()
        )
        left, right = selected[0]["objects"]
        self.assertEqual(left["scale_bin"], right["scale_bin"])
        self.assertEqual(
            left["visual_profile_id"], right["visual_profile_id"]
        )

    def test_object_collection_preserves_appearance_for_three_objects(self) -> None:
        def candidate(index: int) -> dict:
            material = {
                "record": {
                    "asset_source": "unit",
                    "asset_id": f"material_{index}",
                    "path": f"materials/{index}",
                },
                "texture_scale": 1.0 + index,
                "semantic_color_mix": 0.05 * index,
            }
            return {
                "schema_version": "physweep_pybullet_rigid_metadata_v1",
                "scene_id": f"source_{index}",
                "simulation": {
                    "objects": [
                        {
                            "object_id": "object_a",
                            "body_model": "rigid_body",
                            "semantic_type": f"source object {index}",
                            "visual_profile": {
                                "id": f"profile_{index}",
                                "material_policy": "source_or_bound_fallback",
                            },
                        }
                    ]
                },
                "semantic_sampling": {
                    "five_dimensions": {
                        "foreground_object": {
                            "semantic_category": "sphere",
                            "scale_bin": "medium",
                            "uniform_scale": 1.0,
                        }
                    }
                },
                "appearance": {"materials": {"dynamic_object": material}},
            }

        sources = [candidate(index) for index in range(3)]
        original_sources = copy.deepcopy(sources)
        host = copy.deepcopy(sources[0])
        roles = [{"object_id": f"object_{index}"} for index in range(3)]
        scene = compile_object_collection_scene(host, sources, roles)
        self.assertEqual(sources, original_sources)
        for index, role in enumerate(roles):
            object_id = role["object_id"]
            self.assertEqual(
                scene["simulation"]["objects"][index]["visual_profile"],
                sources[index]["simulation"]["objects"][0]["visual_profile"],
            )
            self.assertEqual(
                scene["appearance"]["materials"]["dynamic_objects"][object_id],
                sources[index]["appearance"]["materials"]["dynamic_object"],
            )

        missing_generic_material = copy.deepcopy(sources[1])
        missing_generic_material["appearance"]["materials"] = {}
        with self.assertRaisesRegex(
            ValueError, "lacks a dynamic appearance material"
        ):
            compile_object_collection_scene(
                host,
                [sources[0], missing_generic_material],
                roles[:2],
            )

        asset_adapter = copy.deepcopy(missing_generic_material)
        asset_adapter["simulation"]["objects"][0]["visual_profile"].update(
            {"type": "mesh", "material_policy": "source_or_bound_fallback"}
        )
        fallback_scene = compile_object_collection_scene(
            host,
            [sources[0], asset_adapter],
            roles[:2],
        )
        self.assertEqual(
            fallback_scene["appearance"]["materials"]["dynamic_objects"][
                "object_1"
            ],
            host["appearance"]["materials"]["dynamic_object"],
        )

    def test_released_base_manifest_pins_generation_manifest(self) -> None:
        matrix = load_matrix()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source_root = root / "source"
            records = []
            for index, scale in enumerate(("small", "large")):
                scene_id = f"source_{index}"
                metadata_path = source_root / f"metadata/{scene_id}.json"
                metadata = {
                    "schema_version": "physweep_pybullet_rigid_metadata_v1",
                    "scene_id": scene_id,
                    "simulation": {
                        "objects": [
                            {
                                "body_model": "rigid_body",
                                "geometry": {
                                    "type": "sphere",
                                    "size_m": [0.2, 0.2, 0.2],
                                },
                                "collision_profile": {
                                    "type": "sphere",
                                    "size_m": [0.2, 0.2, 0.2],
                                },
                                "initial_state": {
                                    "pose_profile": "support_normal"
                                },
                                "visual_profile": {"id": f"profile_{index}"},
                            }
                        ],
                        "support": {
                            "scene_class": "ground_flat",
                            "support_shape": "rectangular_slab",
                            "colliders": [
                                {"id": "support", "role": "primary_support"}
                            ],
                        },
                    },
                    "semantic_sampling": {
                        "five_dimensions": {
                            "foreground_object": {"scale_bin": scale}
                        }
                    },
                    "appearance": {
                        "scene_visual": {
                            "id": f"host_{index}",
                            "visual_type": "procedural_room",
                        }
                    },
                }
                write_json(metadata_path, metadata)
                records.append(
                    {
                        "scene_id": scene_id,
                        "path": metadata_path.relative_to(source_root).as_posix(),
                        "metadata_sha256": sha256_file(metadata_path),
                        "source_schema_version": metadata["schema_version"],
                        "kind": "base",
                    }
                )
            source_manifest_path = source_root / "release/metadata_manifest.json"
            write_json(
                source_manifest_path,
                {
                    "schema_version": "physweep_release_metadata_manifest_v2",
                    "dataset_id": "unit_one_object",
                    "sample_count": 2,
                    "group_count": 2,
                    "records": records,
                },
            )
            released_path = root / "outputs/one_object/base/manifest.json"
            pipelines = {}
            for family, family_records in (
                ("generic", records),
                ("asset", []),
            ):
                branch_path = released_path.parent / family / "manifest.json"
                compact_records = []
                for record in family_records:
                    scene_id = record["scene_id"]
                    compact_path = branch_path.parent / scene_id / "metadata.json"
                    compact = {
                        "schema_version": "physweep_base_sample_v11",
                        "scene_id": scene_id,
                        "family": family,
                        "lineage": {
                            "source_generation_metadata_sha256": record[
                                "metadata_sha256"
                            ]
                        },
                    }
                    write_json(compact_path, compact)
                    compact_records.append(
                        {
                            "scene_id": scene_id,
                            "metadata_sha256": sha256_file(compact_path),
                        }
                    )
                write_json(
                    branch_path,
                    {
                        "schema_version": "physweep_base_pipeline_view_v12",
                        "pipeline": family,
                        "sample_count": len(compact_records),
                        "records": compact_records,
                    },
                )
                pipelines[family] = {
                    "manifest": branch_path.relative_to(
                        released_path.parent
                    ).as_posix(),
                    "manifest_sha256": sha256_file(branch_path),
                }
            write_json(
                released_path,
                {
                    "schema_version": "physweep_base_release_view_v14",
                    "dataset_id": "unit_one_object",
                    "sample_count": 2,
                    "pipelines": pipelines,
                    "provenance": {
                        "source_generation_release_metadata": {
                            "schema_version": (
                                "physweep_release_metadata_manifest_v2"
                            ),
                            "manifest_sha256": sha256_file(
                                source_manifest_path
                            ),
                        }
                    },
                },
            )
            objects, hosts, audit = released_source_pool(
                root=root,
                released_base_manifest_path=released_path,
                source_root=source_root,
                source_manifest_path=source_manifest_path,
                matrix=matrix,
            )
            self.assertEqual(len(objects), 2)
            self.assertEqual(len(hosts), 2)
            self.assertEqual(audit["released_base_count"], 2)
            tampered = json.loads(
                source_manifest_path.read_text(encoding="utf-8")
            )
            tampered["dataset_id"] = "changed"
            write_json(source_manifest_path, tampered)
            with self.assertRaisesRegex(ValueError, "does not name"):
                released_source_pool(
                    root=root,
                    released_base_manifest_path=released_path,
                    source_root=source_root,
                    source_manifest_path=source_manifest_path,
                    matrix=matrix,
                )


if __name__ == "__main__":
    unittest.main()
