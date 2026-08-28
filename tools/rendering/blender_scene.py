"""Small Blender-only scene and CLI helpers shared by render entry points."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from tools.core.blender_runtime import blender_argv, look_at


def parse_scene_render_args(
    description: str | None,
    *,
    project_root: Path | None = None,
    include_masks: bool = False,
    include_mask_output: bool = False,
) -> argparse.Namespace:
    """Parse the common metadata/video arguments used by Blender scene renderers."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--metadata", type=Path, required=True)
    if project_root is not None:
        parser.add_argument("--root", type=Path, default=project_root)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--inspection-frame-dir", type=Path)
    if include_masks:
        parser.add_argument("--mask-only", action="store_true")
    if include_masks or include_mask_output:
        parser.add_argument("--instance-mask-dir", type=Path)
    values = blender_argv() if "--" in sys.argv else []
    return parser.parse_args(values)


def add_bound_lights(metadata: dict[str, Any]) -> None:
    """Create metadata-bound lights aimed at the declared camera target."""

    import bpy  # pylint: disable=import-outside-toplevel

    target = metadata["camera"]["target_m"]
    for binding in metadata["render"]["lights"]:
        light_data = bpy.data.lights.new(str(binding["id"]), str(binding["type"]))
        light_data.energy = float(binding["energy_w"])
        light_data.size = float(binding["size_m"])
        light_data.color = tuple(float(value) for value in binding["color_rgb"])
        light = bpy.data.objects.new(str(binding["id"]), light_data)
        bpy.context.collection.objects.link(light)
        light.location = tuple(float(value) for value in binding["position_m"])
        look_at(light, target)
