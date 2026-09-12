#!/usr/bin/env python3
"""Shared deterministic video encoding policy for PhysSweep renderers."""

from __future__ import annotations

import json
import os
import subprocess
import time
from fractions import Fraction
from pathlib import Path
from typing import Any


PROFILE_VERSION = "physweep_h264_perceptually_lossless_long_gop_v1"


def require_render_finished(result: Any, *, label: str) -> None:
    """Reject Blender operators that returned control after cancellation."""
    if set(result) != {"FINISHED"}:
        raise RuntimeError(f"{label} did not finish: {sorted(result)}")


def _decoded_video(video_path: Path, *, ffprobe: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Use one decoder path for encoding completion and release verification."""
    video_path = video_path.resolve()
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise FileNotFoundError(f"rendered video is missing: {video_path}")
    completed = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,avg_frame_rate,time_base,nb_read_frames:frame=width,height,best_effort_timestamp",
         "-of", "json", str(video_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    if completed.returncode != 0 or completed.stderr.strip():
        raise RuntimeError(f"failed to decode video: {video_path}: {completed.stderr.strip()}")
    try:
        probe = json.loads(completed.stdout)
        stream, = probe["streams"]
        frames = probe["frames"]
        if not frames or int(stream["nb_read_frames"]) != len(frames):
            raise ValueError("decoded frame records differ from the positive frame count")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid decoded video probe: {video_path}: {error}") from error
    return stream, frames


def decoded_video_frame_count(video_path: Path, *, ffprobe: str = "ffprobe") -> int:
    """Return the number of decoded frames from the first video stream."""
    return len(_decoded_video(video_path, ffprobe=ffprobe)[1])


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


def require_video_contract(
    video_path: Path,
    expected_frame_count: int,
    *,
    resolution: list[int] | tuple[int, int],
    fps: int,
    ffprobe: str = "ffprobe",
) -> int:
    """Decode once and verify geometry and the inclusive-endpoint time grid."""
    if (isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0
        or expected_frame_count <= 0 or len(resolution) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in resolution)):
        raise ValueError("invalid expected video contract")
    stream, frames = _decoded_video(video_path, ffprobe=ffprobe)
    observed = len(frames)
    try:
        time_base = Fraction(stream["time_base"])
        frame_rate = Fraction(stream["avg_frame_rate"])
        timestamps = [int(frame["best_effort_timestamp"]) * time_base for frame in frames]
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(f"invalid decoded video probe: {video_path}: {error}") from error
    if observed != expected_frame_count:
        raise ValueError(f"decoded video frame count differs: {video_path}: expected={expected_frame_count} observed={observed}")
    expected_size = tuple(resolution)
    if any((item.get("width"), item.get("height")) != expected_size for item in [stream, *frames]):
        raise ValueError(f"video dimensions differ: {video_path}: expected={expected_size}")
    if frame_rate != fps:
        raise ValueError(f"video frame rate differs: {video_path}: expected={fps} observed={frame_rate}")
    # Permit only timestamp rounding to the container's clock. Reject coarse clocks
    # that could hide a missing interval, as well as nonzero starts and VFR gaps.
    if (time_base <= 0 or time_base > Fraction(1, fps * 100)
        or any(abs(t - Fraction(index, fps)) > time_base for index, t in enumerate(timestamps))
        or any(b <= a for a, b in zip(timestamps, timestamps[1:]))):
        raise ValueError(f"video frame timestamps differ from output time grid: {video_path}")
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
