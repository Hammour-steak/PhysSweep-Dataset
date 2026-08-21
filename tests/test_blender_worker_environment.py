from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from blender_worker_environment import isolated_blender_environment  # noqa: E402


class BlenderWorkerEnvironmentTests(unittest.TestCase):
    def test_environment_selects_egl_before_restricting_cuda(self) -> None:
        selector = Path("/tmp/libphysweep_egl_device.so")
        inherited = {
            "CUDA_VISIBLE_DEVICES": "7",
            "HIP_VISIBLE_DEVICES": "7",
            "LD_PRELOAD": "/tmp/existing.so",
        }
        with mock.patch.dict(os.environ, inherited, clear=False):
            with isolated_blender_environment(1, selector) as (
                environment,
                marker,
            ):
                self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)
                self.assertNotIn("HIP_VISIBLE_DEVICES", environment)
                self.assertEqual(environment["PHYSWEEP_EGL_CUDA_DEVICE"], "1")
                self.assertEqual(
                    environment["LD_PRELOAD"],
                    f"{selector}:/tmp/existing.so",
                )
                self.assertEqual(
                    marker, "PhysSweep EGL selector: CUDA device 1 "
                )

    def test_isolated_runtime_directories_are_removed(self) -> None:
        with isolated_blender_environment(0, Path("/tmp/selector.so")) as (
            environment,
            _,
        ):
            runtime_root = Path(environment["HOME"]).parent
            self.assertTrue(runtime_root.is_dir())
            for key in (
                "HOME",
                "XDG_CACHE_HOME",
                "XDG_CONFIG_HOME",
                "XDG_DATA_HOME",
                "TMPDIR",
            ):
                self.assertTrue(Path(environment[key]).is_dir())
        self.assertFalse(runtime_root.exists())


if __name__ == "__main__":
    unittest.main()
