from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_GENERATORS = (
    "core/rigid_geometry.py",
    "sampling/sample_pybullet_base.py",
    "physics/rigid_trajectory.py",
    "physics/simulate_pybullet_rigid.py",
    "rendering/bind_pybullet_visuals.py",
    "rendering/render_pybullet_rigid.py",
    "assets/scene_kit_compiler.py",
    "sampling/sample_asset_proxy_scenes.py",
    "physics/generate_billiards_scene.py",
    "sampling/sample_one_object_scene_matrix.py",
)


class GeneralRuleTests(unittest.TestCase):
    def test_active_code_has_no_concrete_scene_id_patch(self) -> None:
        for name in ACTIVE_GENERATORS:
            source = (ROOT / "tools" / name).read_text(encoding="utf-8")
            with self.subTest(file=name):
                self.assertIsNone(re.search(r"physweeprigid_\d{6}", source))
                self.assertIsNone(re.search(r"(?:if|elif)[^\n]*scene_id", source))

    def test_backend_declares_strict_contract_invariants(self) -> None:
        backend = json.loads(
            (ROOT / "configs/pybullet_backend.json").read_text(encoding="utf-8")
        )
        self.assertTrue(all(backend["invariants"].values()))
        capabilities = json.loads(
            (ROOT / "configs/backend_capabilities.json").read_text(encoding="utf-8")
        )
        self.assertTrue(capabilities["policy"]["unsupported_combinations_are_rejected"])
        self.assertTrue(
            capabilities["policy"]["unsupported_combinations_are_never_substituted"]
        )
        self.assertEqual(
            capabilities["specialized_scopes"]["billiards"][
                "dynamic_body_counts"
            ],
            [1, 3],
        )


if __name__ == "__main__":
    unittest.main()
