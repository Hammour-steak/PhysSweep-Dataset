#!/usr/bin/env python3
"""Shared deterministic video encoding policy for PhysSweep renderers."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any


PROFILE_VERSION = "physweep_h264_perceptually_lossless_long_gop_v1"


def require_render_finished(result: Any, *, label: str) -> None:
    """Reject Blender operators that returned control after cancellation."""
    if set(result) != {"FINISHED"}:
        raise RuntimeError(f"{label} did not finish: {sorted(result)}")


def decoded_video_frame_count(
    video_path: Path,
    *,
    ffprobe: str = "ffprobe",
) -> int:
    """Return the number of frames that ffprobe can decode from the first video stream."""
    video_path = video_path.resolve()
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise FileNotFoundError(f"rendered video is missing: {video_path}")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    values = [value.strip() for value in completed.stdout.splitlines() if value.strip()]
    if completed.returncode != 0 or len(values) != 1:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"failed to count decoded video frames: {video_path}: {detail}")
    try:
        frame_count = int(values[0])
    except ValueError as exc:
        raise RuntimeError(
            f"ffprobe returned an invalid decoded frame count for {video_path}: {values[0]!r}"
        ) from exc
    if frame_count <= 0:
        raise RuntimeError(f"decoded video contains no frames: {video_path}")
    return frame_count


def require_video_frame_count(
    video_path: Path,
    expected_frame_count: int,
    *,
    ffprobe: str = "ffprobe",
) -> int:
    if expected_frame_count <= 0:
        raise ValueError("expected video frame count must be positive")
    observed = decoded_video_frame_count(video_path, ffprobe=ffprobe)
    if observed != expected_frame_count:
        raise ValueError(
            f"decoded video frame count differs: {video_path}: "
            f"expected={expected_frame_count} observed={observed}"
        )
    return observed


def video_has_expected_frame_count(
    video_path: Path,
    expected_frame_count: int,
    *,
    ffprobe: str = "ffprobe",
) -> bool:
    try:
        require_video_frame_count(
            video_path,
            expected_frame_count,
            ffprobe=ffprobe,
        )
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return False
    return True


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


def normalize_h264_container(
    video_path: Path,
    *,
    expected_frame_count: int,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> None:
    """Strip volatile MP4 metadata and non-visual SEI without re-encoding frames."""
    video_path = video_path.resolve()
    require_video_frame_count(
        video_path,
        expected_frame_count,
        ffprobe=ffprobe,
    )
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
        require_video_frame_count(
            temporary,
            expected_frame_count,
            ffprobe=ffprobe,
        )
        os.replace(temporary, video_path)
    finally:
        temporary.unlink(missing_ok=True)
