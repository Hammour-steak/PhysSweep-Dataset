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
            "semantics": {"profile": "three_ball_collision"},
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

    def test_every_one_object_motion_has_an_explicit_event_caption(self) -> None:
        expected_fragments = {
            "drop_fall_1obj": "falls under gravity onto",
            "edge_fall_1obj": "falls from its edge",
            "projectile_1obj": "launched horizontally through the air",
            "arc_projectile_1obj": "launched upward and forward",
            "slide_push_1obj": "short initial push",
            "roll_or_slide_1obj": "rolls or slides",
            "slope_slide_down_1obj": "moves downhill",
            "slope_slide_up_1obj": "launched uphill",
            "wall_impact_1obj": "strikes a fixed wall",
            "ramp_to_flat_1obj": "continues onto flat ground",
            "bounce_1obj": "bounces on",
            "vertical_drop": "falls under gravity onto",
            "resting_push": "after an initial push",
            "diagonal_push": "moves diagonally",
            "edge_exit": "leaves its edge",
            "workbench_clear_zone_drop": "clear area of the workbench",
            "workbench_long_axis_push": "long axis of the workbench",
            "single_ball_free_roll": "without touching a rail",
            "single_ball_rail_rebound": "strikes a rail and rebounds",
            "dense_pinfield_descent": "dense passive peg field",
            "offset_pinfield_descent": "offset start",
            "early_release_chain": "starts upstream",
            "late_release_chain": "starts farther downstream",
        }
        for family, fragment in expected_fragments.items():
            with self.subTest(family=family):
                metadata = {
                    "semantic_sampling": {
                        "five_dimensions": {
                            "motion": {"family": family},
                            "appearance_lighting": {
                                "environment_category": "home_office"
                            },
                        }
                    },
                    "simulation": {
                        "support": {"semantic_type": "wood_tabletop"},
                        "objects": [
                            {
                                "object_id": "object_a",
                                "semantic_type": "physassets_7_book",
                            }
                        ],
                    },
                }
                attach_object_identity(metadata)
                caption = metadata["object_identity"]["text"]["caption"]
                self.assertIn(fragment, caption)
                self.assertIn("In an indoor room", caption)
                self.assertIn("the book", caption)
                self.assertNotIn("scenario", caption)
                self.assertNotIn("_1obj", caption)

    def test_unknown_one_object_motion_is_rejected(self) -> None:
        metadata = {
            "semantics": {"profile": "unreviewed_motion"},
            "simulation": {
                "objects": [
                    {"object_id": "object_a", "semantic_type": "ball"}
                ]
            },
        }
        with self.assertRaisesRegex(ValueError, "explicit caption"):
            attach_object_identity(metadata)

    def test_published_one_object_metadata_recompiles_the_same_event_contract(self) -> None:
        metadata = {
            "sample_kind": "base",
            "semantics": {
                "objects": [
                    {"object_id": "object_a", "semantic_label": "soccer ball"}
                ],
                "motion": {"family": "bounce_1obj"},
                "appearance": {"environment_category": "home_office"},
            },
            "physics": {
                "fixture": {"id": "wood_floor"},
                "objects": [{"object_id": "object_a", "object_valid": True}],
            },
        }
        attach_object_identity(metadata)
        self.assertEqual(
            metadata["object_identity"]["text"]["caption"],
            (
                "In an indoor room, the soccer ball falls under gravity and "
                "bounces on the wooden floor."
            ),
        )

    def test_published_caption_preserves_the_canonical_surface_label(self) -> None:
        metadata = {
            "sample_kind": "base",
            "semantics": {
                "objects": [
                    {
                        "object_id": "object_a",
                        "semantic_label": "four-way lug wrench",
                    }
                ],
                "profile": "diagonal_push",
            },
            "physics": {
                "fixture": {"id": "support_surface"},
                "objects": [{"object_id": "object_a", "object_valid": True}],
            },
        }
        attach_object_identity(metadata)
        caption = metadata["object_identity"]["text"]["caption"]
        self.assertEqual(
            caption,
            (
                "the four-way lug wrench moves diagonally across the support "
                "surface after an initial push."
            ),
        )
        self.assertIn("the four-way lug wrench", caption)

    def test_published_asset_caption_hides_an_internal_fixture_id(self) -> None:
        metadata = {
            "semantics": {
                "objects": [
                    {"object_id": "object_a", "semantic_label": "glass bottle"}
                ],
                "profile": "resting_push",
            },
            "physics": {
                "fixture": {
                    "id": "sketchfab_bg_8a5b41d6445c4f1fbefb2e4abfeebb0d"
                },
                "objects": [{"object_id": "object_a", "object_valid": True}],
            },
        }
        attach_object_identity(metadata)
        caption = metadata["object_identity"]["text"]["caption"]
        self.assertEqual(
            caption,
            "the glass bottle moves across the support surface after an initial push.",
        )
        self.assertNotIn("sketchfab", caption)

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

    def test_every_generic_two_object_motion_has_an_explicit_caption(self) -> None:
        expected_fragments = {
            "surface_hit_rest_2obj": "which starts at rest",
            "surface_glancing_hit_rest_2obj": "an offset path",
            "surface_head_on_2obj": "move toward each other",
            "surface_glancing_opposed_2obj": "opposed, offset paths",
            "surface_crossing_2obj": "crossing paths",
            "surface_catch_up_2obj": "catches up",
            "air_drop_hit_supported_2obj": "falls under gravity",
            "air_projectile_hit_supported_2obj": "launched upward and forward",
            "air_air_collision_2obj": "airborne paths",
            "surface_single_independent_2obj": "remains at rest",
            "surface_dual_independent_2obj": "without contacting each other",
            "air_supported_independent_2obj": "they do not contact",
        }
        for family, fragment in expected_fragments.items():
            with self.subTest(family=family):
                metadata = {
                    "semantic_sampling": {
                        "five_dimensions": {"motion": {"family": family}}
                    },
                    "simulation": {
                        "support": {"semantic_type": "wood_floor"},
                        "interaction": {"motion_pattern": family},
                        "objects": [
                            {"object_id": "object_a", "semantic_type": "ball"},
                            {"object_id": "object_b", "semantic_type": "box"},
                        ],
                    },
                }
                attach_object_identity(metadata)
                caption = metadata["object_identity"]["text"]["caption"]
                self.assertIn(fragment, caption)
                self.assertNotIn("_2obj", caption)

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
