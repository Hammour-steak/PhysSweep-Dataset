from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from tools.sampling.resample_two_object_camera_failures import (
    _replacement_camera_priority,
    camera_failure_mode,
    replace_failed_selections,
    selection_signature,
)


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


class TwoObjectReplacementTests(unittest.TestCase):
    def test_failure_mode_prioritizes_camera_plane_extent(self) -> None:
        rules = load_json("configs/two_object_scene_rules.json")
        rule = next(
            value
            for value in rules["physical_rules"]
            if value["id"] == "inclined_flat_slab"
        )
        cell = {"camera_view_family_id": "side_left_mid"}

        def source(extent: float) -> dict:
            return {
                "metadata": {
                    "simulation": {
                        "objects": [
                            {"geometry": {"size_m": [0.2, extent, extent]}}
                        ]
                    }
                }
            }

        small, large = source(0.2), source(0.3)
        self.assertEqual(
            camera_failure_mode("joint camera visually overlaps the two objects"),
            "pair_overlap",
        )
        self.assertEqual(
            camera_failure_mode("joint camera violates per-object visibility"),
            "per_object_visibility",
        )
        self.assertLess(
            _replacement_camera_priority(
                large, rule, cell, "per_object_visibility"
            ),
            _replacement_camera_priority(
                small, rule, cell, "per_object_visibility"
            ),
        )
        self.assertLess(
            _replacement_camera_priority(small, rule, cell, "pair_overlap"),
            _replacement_camera_priority(large, rule, cell, "pair_overlap"),
        )

    def test_replacement_freezes_passing_rows_and_preserves_cell(self) -> None:
        matrix = load_json("configs/two_object_sampling_matrix.json")
        rules = load_json("configs/two_object_scene_rules.json")
        cell = {
            "cell_id": "replacement_cell__side_left_mid",
            "motion_id": "surface_single_independent_2obj",
            "interaction_class": "independent",
            "shape_pair_id": "sphere_to_sphere",
            "object_a_shape": "sphere",
            "object_b_shape": "sphere",
            "scale_pair_id": "small_to_small",
            "object_a_scale_bin": "small",
            "object_b_scale_bin": "small",
            "scene_class": "ground_flat",
            "replicate_index": 0,
            "camera_view_family_id": "side_left_mid",
            "source_family_pair_id": "generic_to_generic",
            "object_a_source_family": "generic",
            "object_b_source_family": "generic",
            "scene_rule_id": "ground_patch_flat",
            "visual_environment_category": "minimal",
        }

        def obj(index: int, source_family: str = "generic") -> dict:
            return {
                "metadata": {},
                "source": {"scene_id": f"object_{index}"},
                "source_family": source_family,
                "shape_family_id": "sphere",
                "scale_bin": "small",
                "visual_profile_id": f"object_profile_{index}",
            }

        def host(index: int) -> dict:
            return {
                "metadata": {},
                "source": {"scene_id": f"host_{index}"},
                "scene_rule_id": "ground_patch_flat",
                "scene_class": "ground_flat",
                "visual_profile_id": f"host_profile_{index}",
                "visual_type": "procedural_room",
                "environment_category": "minimal",
            }

        objects = [obj(index) for index in range(1, 7)] + [
            obj(index, "asset") for index in range(7, 11)
        ]
        hosts = [host(index) for index in range(1, 4)]
        failed = {
            "scene_id": "failed",
            "cell": copy.deepcopy(cell),
            "host": hosts[0],
            "objects": objects[:2],
        }
        passing_cell = copy.deepcopy(cell)
        passing_cell["cell_id"] = "passing_cell"
        passing = {
            "scene_id": "passing",
            "cell": passing_cell,
            "host": hosts[1],
            "objects": objects[2:4],
        }
        result = replace_failed_selections(
            [failed, passing],
            {"failed"},
            objects,
            hosts,
            matrix,
            rules,
            attempt=1,
        )
        self.assertIs(result[1], passing)
        self.assertEqual(result[0]["cell"], cell)
        self.assertNotEqual(selection_signature(result[0]), selection_signature(failed))
        self.assertNotEqual(
            tuple(source["source"]["scene_id"] for source in result[0]["objects"]),
            tuple(source["source"]["scene_id"] for source in failed["objects"]),
        )

        later = replace_failed_selections(
            [failed, passing],
            {"failed"},
            objects,
            hosts,
            matrix,
            rules,
            attempt=2,
        )[0]
        self.assertNotEqual(
            later["cell"]["source_family_pair_id"], "generic_to_generic"
        )
        self.assertNotEqual(
            later["cell"]["camera_view_family_id"], "side_left_mid"
        )
        self.assertEqual(
            [source["source_family"] for source in later["objects"]],
            [
                later["cell"]["object_a_source_family"],
                later["cell"]["object_b_source_family"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
