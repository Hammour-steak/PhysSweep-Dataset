"""Small Blender-only scene helpers shared by render entry points."""

from __future__ import annotations

from typing import Any


def look_at(camera: Any, target: tuple[float, float, float]) -> None:
    """Aim a Blender camera at a world-space target."""

    import mathutils  # pylint: disable=import-outside-toplevel

    direction = mathutils.Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
