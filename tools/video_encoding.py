#!/usr/bin/env python3
"""Shared deterministic video encoding policy for PhysSweep renderers."""

from __future__ import annotations

from typing import Any


PROFILE_VERSION = "physweep_h264_perceptually_lossless_long_gop_v1"


def configure_h264_output(
    scene: Any,
    *,
    fps: int,
    frame_count: int,
) -> dict[str, Any]:
    if fps <= 0 or frame_count <= 0:
        raise ValueError("fps and frame_count must be positive")
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "PERC_LOSSLESS"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.ffmpeg.gopsize = frame_count
    return {
        "profile_version": PROFILE_VERSION,
        "container": "MPEG4",
        "codec": "H264",
        "constant_rate_factor": "PERC_LOSSLESS",
        "preset": "GOOD",
        "gop_size_frames": frame_count,
        "fps": fps,
    }
