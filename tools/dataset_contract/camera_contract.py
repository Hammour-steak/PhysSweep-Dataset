"""Camera clipping values shared by validation, rendering, and publication."""

from __future__ import annotations

import math
from typing import Any, Mapping


def camera_clipping_range(
    camera: Mapping[str, Any], *, require_explicit: bool = False
) -> tuple[float, float]:
    values = []
    for name, default in (("clip_start_m", 0.03), ("clip_end_m", 100.0)):
        if require_explicit and name not in camera:
            raise ValueError("final generic camera has no clipping range")
        value = camera.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("camera clipping range must contain finite numbers")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("camera clipping range must contain finite numbers")
        values.append(value)
    near, far = values
    if near <= 0.0 or far <= near:
        raise ValueError("camera clipping range must satisfy 0 < near < far")
    return near, far
