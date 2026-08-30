from __future__ import annotations

import unittest

from tools.training_export.one_object_prompt_contract import (
    PROMPT_TEMPLATE_VERSION,
    build_training_prompt,
)


class TrainingPromptContractTests(unittest.TestCase):
    def test_training_prompt_wraps_the_dataset_event_caption(self) -> None:
        metadata = {
            "semantic_sampling": {
                "five_dimensions": {
                    "motion": {"family": "bounce_1obj"},
                    "appearance_lighting": {
                        "environment_category": "home_office"
                    },
                }
            },
            "simulation": {
                "support": {"semantic_type": "wood_floor"},
                "objects": [
                    {
                        "object_id": "object_a",
                        "semantic_type": "physassets_9_soccer_ball",
                    }
                ],
            },
        }
        prompt = build_training_prompt(metadata, "object_a")
        self.assertEqual(PROMPT_TEMPLATE_VERSION, "physweep_initial_event_prompt_v3")
        self.assertIn(
            "the soccer ball falls under gravity and bounces on the wooden floor.",
            prompt,
        )
        self.assertIn("In an indoor room", prompt)
        self.assertIn("Exactly one dynamic object is present", prompt)
        self.assertNotIn("scenario", prompt)
        self.assertNotIn("physassets", prompt)


if __name__ == "__main__":
    unittest.main()
