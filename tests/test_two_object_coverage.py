from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.core.hashing import sha256_file
from tools.sampling.sample_two_object_base import _validated_intents
from tools.sampling.sample_two_object_coverage import (
    _axis_counts,
    coverage_cells,
    released_source_pool,
    select_coverage_sources,
)


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
    def test_matrix_declares_complete_ordered_cartesian_coverage(self) -> None:
        matrix = load_matrix()
        cells, full_count = coverage_cells(matrix)
        self.assertEqual(full_count, 324)
        self.assertEqual(len(cells), 324)
        counts = _axis_counts(cells)
        self.assertEqual(set(counts["motion"].values()), {36})
        self.assertEqual(set(counts["ordered_scale_pair"].values()), {36})
        self.assertEqual(
            set(counts["visual_identity_relation"].values()), {162}
        )
        self.assertEqual(set(counts["scene_class"].values()), {162})

    def test_balanced_smoke_prefix_covers_every_axis(self) -> None:
        cells, full_count = coverage_cells(load_matrix(), 72)
        self.assertEqual(full_count, 324)
        counts = _axis_counts(cells)
        self.assertEqual(set(counts["motion"].values()), {8})
        self.assertEqual(set(counts["ordered_scale_pair"].values()), {8})
        self.assertEqual(
            set(counts["visual_identity_relation"].values()), {36}
        )
        self.assertEqual(set(counts["scene_class"].values()), {36})

    def test_matrix_rejects_missing_scale_cell_and_weakened_uniqueness(self) -> None:
        matrix = load_matrix()
        missing = copy.deepcopy(matrix)
        missing["coverage_plan"]["role_ordered_scale_pairs"].pop()
        with self.assertRaisesRegex(ValueError, "full product"):
            _validated_intents(missing)
        weakened = copy.deepcopy(matrix)
        weakened["coverage_plan"]["selection_policy"][
            "source_pair_uniqueness"
        ] = "allow_reuse"
        with self.assertRaisesRegex(ValueError, "may not be weakened"):
            _validated_intents(weakened)
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

    def test_source_selection_obeys_scale_identity_and_uniqueness(self) -> None:
        matrix = load_matrix()
        cells, _ = coverage_cells(matrix, 18)
        objects = []
        for scale in ("small", "medium", "large"):
            for profile_index in range(3):
                for source_index in range(4):
                    source_id = (
                        f"object_{scale}_{profile_index}_{source_index}"
                    )
                    objects.append(
                        {
                            "metadata": {},
                            "source": {"scene_id": source_id},
                            "scale_bin": scale,
                            "visual_profile_id": f"profile_{profile_index}",
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
        for selection in selected:
            cell = selection["cell"]
            left, right = selection["objects"]
            self.assertEqual(left["scale_bin"], cell["object_a_scale_bin"])
            self.assertEqual(right["scale_bin"], cell["object_b_scale_bin"])
            self.assertEqual(
                left["visual_profile_id"] == right["visual_profile_id"],
                cell["visual_identity_relation"] == "same_visual_profile",
            )
            self.assertEqual(
                selection["host"]["scene_class"], cell["scene_class"]
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
            write_json(
                released_path,
                {
                    "schema_version": "physweep_base_release_view_v14",
                    "dataset_id": "unit_one_object",
                    "sample_count": 2,
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
