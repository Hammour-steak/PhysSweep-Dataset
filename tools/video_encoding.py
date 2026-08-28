#!/usr/bin/env python3
"""Shared deterministic video encoding policy for PhysSweep renderers."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
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


def normalize_h264_container(video_path: Path, *, ffmpeg: str = "ffmpeg") -> None:
    """Strip volatile MP4 metadata and non-visual SEI without re-encoding frames."""
    video_path = video_path.resolve()
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise FileNotFoundError(f"rendered video is missing: {video_path}")
    temporary = video_path.with_name(
        f".{video_path.stem}.normalized-{os.getpid()}-{time.time_ns()}"
        f"{video_path.suffix}"
    )
    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0",
        "-map_metadata",
        "-1",
        "-codec",
        "copy",
        "-bsf:v",
        "filter_units=remove_types=6",
        "-y",
        str(temporary),
    ]
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "failed to normalize rendered video container: "
                f"{completed.stderr.strip()}"
            )
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("video container normalization produced no output")
        os.replace(temporary, video_path)
    finally:
        temporary.unlink(missing_ok=True)
