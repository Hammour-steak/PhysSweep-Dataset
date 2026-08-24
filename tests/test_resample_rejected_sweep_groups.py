from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from resample_rejected_sweep_groups import select_replacement_slots  # noqa: E402


class ReplacementSelectionTests(unittest.TestCase):
    def test_selection_preserves_slot_and_changes_dynamic_asset(self) -> None:
        original = {
            "index": 7,
            "generator": "asset_proxy",
            "motion_intent": "slide_push_1obj",
            "environment_id": "curated_support_asset",
            "profile": "diagonal_push",
            "dynamic_asset_id": "thin_phone",
            "support_asset_id": "desk_a",
            "static_prop_asset_id": None,
        }
        incompatible = {**original, "index": 10, "profile": "resting_push"}
        compatible = {
            **original,
            "index": 11,
            "dynamic_asset_id": "box",
            "support_asset_id": "desk_b",
        }

        selected = select_replacement_slots(
            [original], [incompatible, compatible], set()
        )

        self.assertEqual(selected, [(original, compatible)])

    def test_selection_prefers_novel_asset_signature(self) -> None:
        original = {
            "index": 7,
            "generator": "asset_proxy",
            "motion_intent": "slide_push_1obj",
            "environment_id": "curated_support_asset",
            "profile": "diagonal_push",
            "dynamic_asset_id": "thin_phone",
            "support_asset_id": "desk_a",
            "static_prop_asset_id": None,
        }
        duplicate = {
            **original,
            "index": 10,
            "dynamic_asset_id": "box",
            "support_asset_id": "desk_b",
        }
        novel = {
            **original,
            "index": 11,
            "dynamic_asset_id": "can",
            "support_asset_id": "desk_c",
        }

        selected = select_replacement_slots(
            [original],
            [duplicate, novel],
            {("box", "desk_b", None)},
        )

        self.assertEqual(selected, [(original, novel)])


if __name__ == "__main__":
    unittest.main()
