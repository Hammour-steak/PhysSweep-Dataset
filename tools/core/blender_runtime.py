"""Small compatibility helpers shared by Blender entry points."""

from __future__ import annotations

import sys
from typing import Any, Iterable


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


def clear_blender_scene(data_collections: Iterable[str] = ()) -> None:
    """Delete scene objects and optionally purge named orphan data collections."""

    import bpy  # pylint: disable=import-outside-toplevel

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for name in data_collections:
        collection = getattr(bpy.data, name)
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def blender_world_bounds(objects: Iterable[Any]) -> tuple[Any, Any]:
    """Return world-space bounds for Blender objects from their bound boxes."""

    import mathutils  # pylint: disable=import-outside-toplevel

    low = mathutils.Vector((float("inf"),) * 3)
    high = mathutils.Vector((float("-inf"),) * 3)
    found = False
    for obj in objects:
        for corner in obj.bound_box:
            found = True
            point = obj.matrix_world @ mathutils.Vector(corner)
            for axis in range(3):
                low[axis] = min(low[axis], point[axis])
                high[axis] = max(high[axis], point[axis])
    if not found:
        raise ValueError("cannot measure bounds from empty Blender geometry")
    return low, high
