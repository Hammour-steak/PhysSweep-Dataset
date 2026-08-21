from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from bind_pybullet_visuals import shadow_readable_lighting


def metadata(size_m: list[float]) -> dict:
    return {
        "simulation": {
            "objects": [
                {
                    "geometry": {
                        "size_m": size_m,
                    }
                }
            ]
        }
    }


class ShadowReadableLightingTests(unittest.TestCase):
    def test_small_flat_object_uses_smaller_key(self) -> None:
        result = shadow_readable_lighting(metadata([0.26, 0.069, 0.033]))
        self.assertAlmostEqual(result["key_size_m"], 1.17)
        self.assertLess(result["key_energy_w"], 560.0)
        self.assertLess(result["fill_energy_w"], 55.0)
        self.assertAlmostEqual(result["contact_shadow_bias_m"], 0.00165)
        self.assertEqual(result["contact_shadow_distance_m"], 0.52)

    def test_key_size_has_stable_bounds(self) -> None:
        small = shadow_readable_lighting(metadata([0.05, 0.05, 0.05]))
        large = shadow_readable_lighting(metadata([0.8, 0.6, 0.4]))
        self.assertEqual(small["key_size_m"], 0.95)
        self.assertEqual(large["key_size_m"], 1.60)

    def test_energy_density_is_constant_until_cap(self) -> None:
        result = shadow_readable_lighting(metadata([0.20, 0.10, 0.05]))
        expected = 160.0 * result["key_size_m"] ** 2
        self.assertAlmostEqual(result["key_energy_w"], expected)


if __name__ == "__main__":
    unittest.main()
