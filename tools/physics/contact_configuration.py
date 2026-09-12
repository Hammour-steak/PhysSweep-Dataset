"""Resolve the contact-processing threshold used by simulation and export."""

from __future__ import annotations

import math
from typing import Any, Mapping


def contact_processing_threshold(scene: Mapping[str, Any]) -> float:
    adapter = scene["backend_binding"]["adapter_id"]
    value = 0.0
    if adapter in {"billiards_v4", "billiards_two_object_v1", "billiards_three_object_v1"}:
        value = scene["adapter_payload"]["backend"]["billiards_rules"]["ball_dynamics"].get(
            "contact_processing_threshold_m", 0.0
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("contact_processing_threshold_m must be finite and nonnegative")
    return float(value)


def contact_processing_execution_matches(scene: Mapping[str, Any], execution: Any) -> bool:
    expected = [
        {"object_id": obj["object_id"], "contactProcessingThreshold": contact_processing_threshold(scene)}
        for obj in scene["objects"]
    ]
    return isinstance(execution, list) and execution == expected
