"""Numerical integrity shared by canonical trajectory build, verify and resume."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def validate_trajectory_arrays(
    arrays: Mapping[str, Any], *,
    time: Mapping[str, Any] | None = None,
    objects: Sequence[Mapping[str, Any]] | None = None,
    require_sign_continuity: bool = True,
) -> None:
    ids = np.asarray(arrays["object_ids"])
    times = np.asarray(arrays["time_s"])
    if ids.ndim != 1 or ids.dtype.kind != "U" or not len(ids):
        raise ValueError("trajectory object axis must be nonempty strings")
    if len(set(ids.tolist())) != len(ids) or any(not item for item in ids.tolist()):
        raise ValueError("trajectory object ids must be nonempty and unique")
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("trajectory time axis must contain at least two frames")
    frames, count = len(times), len(ids)
    shapes = {
        "time_s": (frames,), "position_m": (frames, count, 3),
        "quaternion_wxyz": (frames, count, 4),
        "linear_velocity_m_s": (frames, count, 3),
        "angular_velocity_rad_s": (frames, count, 3),
        "contact_count": (frames, count),
    }
    for name, shape in shapes.items():
        value = np.asarray(arrays[name])
        if value.shape != shape:
            raise ValueError(f"trajectory {name} shape differs: {value.shape} != {shape}")
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"trajectory {name} must contain finite real numbers")
    if abs(float(times[0])) > 1.e-9 or not np.all(times[1:] > times[:-1]):
        raise ValueError("trajectory time axis must increase from zero")
    contact = np.asarray(arrays["contact_count"])
    if contact.dtype.kind not in "iu" or np.any(contact < 0) or np.any(contact > np.iinfo(np.int32).max):
        raise ValueError("trajectory contact_count must contain nonnegative int32-representable integers")
    quaternion = np.asarray(arrays["quaternion_wxyz"])
    if not np.allclose(np.linalg.norm(quaternion, axis=2), 1., rtol=0., atol=1.e-6):
        raise ValueError("trajectory quaternion norm differs from one")
    if require_sign_continuity and np.any(np.sum(quaternion[1:] * quaternion[:-1], axis=2) < -1.e-12):
        raise ValueError("trajectory quaternion sign continuity failed")
    if time is not None:
        fps, duration = time["output_fps"], time["duration_s"]
        if (isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0
                or isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not np.isfinite(duration) or duration <= 0):
            raise ValueError("trajectory metadata time is invalid")
        expected_frames = round(duration * fps) + 1
        if frames != expected_frames or time.get("frame_count", expected_frames) != expected_frames:
            raise ValueError("trajectory frame count differs from metadata time")
        if not np.allclose(times, np.arange(frames) / fps, rtol=0., atol=1.e-9):
            raise ValueError("trajectory time axis differs from metadata fps")
    if objects is not None:
        if ids.tolist() != [record["object_id"] for record in objects]:
            raise ValueError("trajectory object axis differs from metadata")
        for name in ("position_m", "linear_velocity_m_s", "angular_velocity_rad_s"):
            expected = np.asarray([record["initial_state"][name] for record in objects])
            if expected.shape != (count, 3) or not np.isfinite(expected).all() or not np.allclose(
                arrays[name][0], expected, rtol=0., atol=1.e-7,
            ):
                raise ValueError(f"trajectory initial {name} differs from metadata")
        expected = np.asarray([record["initial_state"]["quaternion_wxyz"] for record in objects])
        if expected.shape != (count, 4) or not np.isfinite(expected).all():
            raise ValueError("trajectory metadata initial orientation is invalid")
        distance = np.minimum(np.linalg.norm(quaternion[0] - expected, axis=1),
                              np.linalg.norm(quaternion[0] + expected, axis=1))
        if not np.all(distance <= 1.e-7):
            raise ValueError("trajectory initial orientation differs from metadata")
