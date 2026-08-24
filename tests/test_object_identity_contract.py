from __future__ import annotations

import unittest

from tools.object_identity_contract import (
    OBJECT_IDENTITY_SCHEMA_VERSION,
    attach_object_identity,
    validate_object_identity,
)


class ObjectIdentityContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
