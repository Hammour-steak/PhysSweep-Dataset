from __future__ import annotations

import unittest

from tools.dataset_contract.object_identity_contract import (
    OBJECT_IDENTITY_SCHEMA_VERSION,
    attach_object_identity,
    require_simulation_objects,
    require_single_simulation_object,
    validate_object_identity,
)


class ObjectIdentityContractTests(unittest.TestCase):
    def test_adapter_object_count_capability_is_explicit(self) -> None:
        metadata = {
            "simulation": {
                "objects": [
                    {"object_id": "object_a"},
                    {"object_id": "object_b"},
                ]
            }
        }
        self.assertEqual(
            [
                record["object_id"]
                for record in require_simulation_objects(metadata, (1, 2), "test")
            ],
            ["object_a", "object_b"],
        )
        with self.assertRaisesRegex(ValueError, "supports dynamic object counts"):
            require_single_simulation_object(metadata, "one_object_test")

    def test_generic_object_maps_text_trajectory_mask_and_sweep(self) -> None:
        metadata = {
            "semantic_sampling": {
                "five_dimensions": {"motion": {"family": "slide_push"}}
            },
            "simulation": {
                "support": {"label": "wood_tabletop"},
                "objects": [
                    {
                        "object_id": "object_a",
                        "semantic_type": "ball",
                        "visual_profile": {"asset_id": "ball_asset"},
                    }
                ],
            },
            "sweep": {"target_object_id": "object_a"},
        }
        attach_object_identity(metadata)
        self.assertEqual(
            metadata["object_identity"]["schema_version"],
            OBJECT_IDENTITY_SCHEMA_VERSION,
        )
        result = validate_object_identity(
            metadata,
            trajectory_keys={
                "object_a__position_m",
                "object_a__quaternion_wxyz",
            },
        )
        self.assertEqual(result["object_ids"], ["object_a"])
        self.assertEqual(result["dynamic_object_ids"], ["object_a"])
        self.assertEqual(
            metadata["object_identity"]["instance_masks"]["objects"]["object_a"][
                "instance_id"
            ],
            1,
        )

    def test_billiards_initial_states_get_distinct_ids(self) -> None:
        metadata = {
            "physics": {
                "initial_states": [
                    {"object_id": "cue_ball", "semantic_type": "cue ball"},
                    {"object_id": "object_ball", "semantic_type": "object ball"},
                ]
            }
        }
        attach_object_identity(metadata)
        result = validate_object_identity(
            metadata,
            trajectory_keys={"position_m", "quaternion_xyzw"},
        )
        self.assertEqual(result["object_ids"], ["cue_ball", "object_ball"])
        self.assertEqual(result["dynamic_object_count"], 2)
        self.assertIn("the cue ball", metadata["object_identity"]["text"]["caption"])
        self.assertEqual(
            metadata["object_identity"]["trajectory"]["layout"],
            "frame_object_channel",
        )

    def test_asset_scene_uses_dynamic_asset_as_the_joined_object(self) -> None:
        metadata = {
            "assets": {
                "dynamic_asset_id": "physassets_123",
                "support_asset_id": "raised_ramp",
            },
            "semantics": {"motion": "drop_fall"},
        }
        attach_object_identity(metadata)
        result = validate_object_identity(metadata)
        self.assertEqual(result["object_ids"], ["object_a"])
        self.assertEqual(
            metadata["object_identity"]["objects"][0]["asset_id"],
            "physassets_123",
        )
        self.assertEqual(
            metadata["object_identity"]["instance_masks"]["encoding"],
            "rgba_alpha_antialiased_silhouette_mask",
        )

    def test_target_object_must_exist(self) -> None:
        metadata = {
            "simulation": {
                "objects": [{"object_id": "object_a", "semantic_type": "ball"}]
            },
            "sweep": {"target_object_id": "missing"},
        }
        with self.assertRaises(ValueError):
            attach_object_identity(metadata)

    def test_two_object_caption_names_roles_motion_support_and_environment(self) -> None:
        metadata = {
            "semantic_sampling": {
                "five_dimensions": {
                    "motion": {"family": "air_drop_hit_supported_2obj"},
                    "appearance_lighting": {
                        "environment_category": "lab_studio"
                    },
                }
            },
            "simulation": {
                "support": {"semantic_type": "lab_bench"},
                "interaction": {
                    "motion_pattern": "air_drop_hit_supported_2obj"
                },
                "objects": [
                    {
                        "object_id": "object_a",
                        "semantic_type": "physassets_10575_tennis_ball",
                    },
                    {
                        "object_id": "object_b",
                        "semantic_type": "physassets_16301_drink_box",
                    },
                ],
            },
        }
        attach_object_identity(metadata)
        self.assertEqual(
            metadata["object_identity"]["text"]["caption"],
            (
                "In a laboratory studio, the tennis ball falls under gravity "
                "and collides with the drink carton, which starts at rest on "
                "the laboratory bench."
            ),
        )
        self.assertNotIn("physassets", str(metadata["object_identity"]))

    def test_two_identical_labels_remain_two_identity_mentions(self) -> None:
        metadata = {
            "semantic_sampling": {
                "five_dimensions": {
                    "motion": {"family": "surface_dual_independent_2obj"}
                }
            },
            "simulation": {
                "support": {"semantic_type": "wood_floor"},
                "interaction": {
                    "motion_pattern": "surface_dual_independent_2obj"
                },
                "objects": [
                    {"object_id": "object_a", "semantic_type": "barrel"},
                    {"object_id": "object_b", "semantic_type": "barrel"},
                ],
            },
        }
        attach_object_identity(metadata)
        caption = metadata["object_identity"]["text"]["caption"]
        self.assertEqual(caption.count("the barrel"), 2)
        self.assertEqual(
            [
                record["object_id"]
                for record in metadata["object_identity"]["text"][
                    "object_mentions"
                ]
            ],
            ["object_a", "object_b"],
        )


if __name__ == "__main__":
    unittest.main()
