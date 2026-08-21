"""Coordinate transforms used by the published dataset contract."""

from __future__ import annotations

import numpy as np


def transform_world_vector_to_camera(
    vector: list[float] | np.ndarray,
    camera: dict,
    *,
    axial: bool = False,
) -> np.ndarray:
    position = np.asarray(camera["position_m"], dtype=np.float64)
    target = np.asarray(camera["target_m"], dtype=np.float64)
    forward = target - position
    forward /= max(float(np.linalg.norm(forward)), 1.0e-12)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
    right /= max(float(np.linalg.norm(right)), 1.0e-12)
    up = np.cross(right, forward)
    up /= max(float(np.linalg.norm(up)), 1.0e-12)
    rotation = np.stack([right, up, forward], axis=0)
    transformed = rotation @ np.asarray(vector, dtype=np.float64)
    if axial:
        transformed *= np.linalg.det(rotation)
    return transformed
