import unittest

from tools.render_sweep_release import STAGE_NAMES, stage_span


class RenderSweepReleaseTests(unittest.TestCase):
    def test_default_runs_all_stages(self) -> None:
        self.assertEqual(stage_span(None, None), (0, len(STAGE_NAMES)))

    def test_explicit_start_defaults_to_one_stage(self) -> None:
        index = STAGE_NAMES.index("render_asset_sweeps")
        self.assertEqual(stage_span("render_asset_sweeps", None), (index, index + 1))

    def test_explicit_range_is_inclusive(self) -> None:
        first = STAGE_NAMES.index("render_asset_sweeps")
        last = STAGE_NAMES.index("render_billiards_sweeps")
        self.assertEqual(
            stage_span("render_asset_sweeps", "render_billiards_sweeps"),
            (first, last + 1),
        )

    def test_rejects_reversed_range(self) -> None:
        with self.assertRaises(ValueError):
            stage_span("render_generic_sweeps", "render_asset_sweeps")


if __name__ == "__main__":
    unittest.main()
