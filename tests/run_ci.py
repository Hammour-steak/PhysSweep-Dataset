"""Run every repository test that does not require the external asset library."""

from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ASSET_MODULES = {
    "test_asset_ingestion_contract",
    "test_decoupled_sampling_matrix",
    "test_physical_proxy_catalog",
    "test_pybullet_sampler",
    "test_pybullet_simulation",
    "test_rigid_geometry",
    "test_sampling_architecture",
    "test_visual_environment_collision_v1",
}


def suite() -> unittest.TestSuite:
    loader = unittest.TestLoader()
    result = unittest.TestSuite()
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        if path.stem in EXTERNAL_ASSET_MODULES:
            continue
        module = importlib.import_module(f"tests.{path.stem}")
        result.addTests(loader.loadTestsFromModule(module))
    return result


if __name__ == "__main__":
    print(
        "External asset tests skipped in CI: "
        + ", ".join(sorted(EXTERNAL_ASSET_MODULES))
    )
    outcome = unittest.TextTestRunner(verbosity=2).run(suite())
    sys.exit(0 if outcome.wasSuccessful() else 1)
