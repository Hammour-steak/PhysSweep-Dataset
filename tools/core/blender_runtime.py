"""Small compatibility helpers shared by Blender entry points."""

from __future__ import annotations

import sys


def blender_argv() -> list[str]:
    """Return arguments after Blender's ``--`` separator."""

    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1 :]
    return sys.argv[1:]


def patch_numpy_for_blender_gltf() -> None:
    """Restore the NumPy alias expected by Blender's bundled glTF importer."""

    import numpy as np  # pylint: disable=import-outside-toplevel

    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]
