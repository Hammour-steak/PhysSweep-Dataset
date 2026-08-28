from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

from tools.rendering.appearance_adaptation import choose_adaptation


def stats(luminance: float, classification: str) -> dict:
    return {
        "mean_linear_luminance": luminance,
        "light_pixel_fraction": 0.0,
        "classification": classification,
    }


class AppearanceAdaptationTests(unittest.TestCase):
    def test_two_light_material_groups_reduce_ambient_illumination(self) -> None:
        result = choose_adaptation(stats(0.56, "light"), stats(0.62, "light"))
        self.assertEqual(result["decision"], "both_light")
        self.assertLess(result["exposure_delta_ev"], 0.0)
        self.assertLess(result["world_strength_scale"], 1.0)
        self.assertLess(result["fill_light_scale"], 1.0)

    def test_balanced_materials_do_not_change_lighting(self) -> None:
        result = choose_adaptation(stats(0.18, "medium"), stats(0.31, "medium"))
        self.assertEqual(result["decision"], "balanced")
        self.assertEqual(result["exposure_delta_ev"], 0.0)
        self.assertEqual(result["world_strength_scale"], 1.0)
        self.assertEqual(result["fill_light_scale"], 1.0)

    def test_dark_material_groups_receive_only_a_small_lift(self) -> None:
        result = choose_adaptation(stats(0.07, "dark"), stats(0.10, "dark"))
        self.assertEqual(result["decision"], "both_dark")
        self.assertLessEqual(result["exposure_delta_ev"], 0.12)
        self.assertLessEqual(result["fill_light_scale"], 1.10)

    def test_very_light_support_is_restrained_even_with_darker_object(self) -> None:
        result = choose_adaptation(stats(0.13, "medium"), stats(0.96, "light"))
        self.assertEqual(result["decision"], "very_light_support")
        self.assertEqual(result["exposure_delta_ev"], -0.52)
        self.assertLess(result["world_strength_scale"], 0.65)


if __name__ == "__main__":
    unittest.main()
