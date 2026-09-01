import unittest
from unittest.mock import patch

from tools.rendering.bind_pybullet_visuals import bind_scene_outcome


class BindPybulletVisualsTests(unittest.TestCase):
    @patch("tools.rendering.bind_pybullet_visuals.bind_scene")
    def test_binding_outcome_preserves_success(self, bind_scene) -> None:
        sample = {"scene_id": "scene_ok", "metadata_sha256": "a" * 64}
        bind_scene.return_value = sample

        outcome = bind_scene_outcome("scene_ok", "unused")

        self.assertEqual(outcome, {"ok": True, "sample": sample})

    @patch("tools.rendering.bind_pybullet_visuals.bind_scene")
    def test_binding_outcome_bounds_failure_without_hiding_it(
        self, bind_scene
    ) -> None:
        bind_scene.side_effect = ValueError("camera failure " * 1000)

        outcome = bind_scene_outcome("scene_bad", "unused")

        self.assertFalse(outcome["ok"])
        self.assertEqual(outcome["scene_id"], "scene_bad")
        self.assertEqual(outcome["error_type"], "ValueError")
        self.assertLessEqual(len(outcome["error"]), 2400)
        self.assertTrue(outcome["error"].endswith("..."))


if __name__ == "__main__":
    unittest.main()
