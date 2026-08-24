#!/usr/bin/env python3
"""Shared accessors for canonical and adapter-specific trajectory channels."""

from __future__ import annotations

from typing import Any

import numpy as np


def object_trajectory_view(
    metadata: dict[str, Any], trajectory: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Map the canonical [frame, object, ...] layout to named object channels."""
    object_ids = [
        str(record["object_id"])
        for record in metadata["simulation"]["objects"]
    ]
    if all(f"{object_id}__position_m" in trajectory for object_id in object_ids):
        return trajectory
    stored_ids = [str(value) for value in np.asarray(trajectory["object_ids"])]
    if stored_ids != object_ids:
        raise ValueError("trajectory object ids do not match metadata")
    result = dict(trajectory)
    canonical_fields = (
        "position_m",
        "quaternion_wxyz",
        "linear_velocity_m_s",
        "angular_velocity_rad_s",
    )
    for object_index, object_id in enumerate(object_ids):
        for field in canonical_fields:
            values = np.asarray(trajectory[field])
            result[f"{object_id}__{field}"] = values[:, object_index]
    for key, value in trajectory.items():
        if key.startswith("adapter__"):
            result[key[len("adapter__") :]] = value
    return result
