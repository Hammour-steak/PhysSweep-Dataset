from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.rendering.video_encoding import (
    decoded_video_frame_count,
    normalize_h264_container,
    require_render_finished,
    require_video_frame_count,
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_video(path: Path, date: str) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=24:d=0.1",
            "-frames:v",
            "2",
            "-codec:v",
            "libx264",
            "-metadata",
            f"date={date}",
            "-y",
            str(path),
        ],
        check=True,
    )


class VideoEncodingTests(unittest.TestCase):
    def test_render_operator_must_report_finished(self) -> None:
        require_render_finished({"FINISHED"}, label="test render")
        with self.assertRaisesRegex(RuntimeError, "test render did not finish"):
            require_render_finished({"CANCELLED"}, label="test render")

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "ffmpeg/ffprobe are unavailable",
    )
    def test_normalization_removes_volatile_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.mp4"
            second = root / "second.mp4"
            make_video(first, "2026/08/25 14:36:10")
            make_video(second, "2026/08/27 18:43:36")
            self.assertNotEqual(file_sha256(first), file_sha256(second))

            normalize_h264_container(first, expected_frame_count=2)
            normalize_h264_container(second, expected_frame_count=2)

            self.assertEqual(file_sha256(first), file_sha256(second))
            self.assertEqual(decoded_video_frame_count(first), 2)
            self.assertFalse(list(root.glob(".*.normalized-*.mp4")))

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "ffmpeg/ffprobe are unavailable",
    )
    def test_frame_validation_rejects_a_cleanly_decodable_short_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "short.mp4"
            make_video(video, "2026/08/29 00:00:00")
            with self.assertRaisesRegex(
                ValueError,
                "expected=3 observed=2",
            ):
                require_video_frame_count(video, 3)

    def test_normalization_requires_rendered_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                FileNotFoundError, "rendered video is missing"
            ):
                normalize_h264_container(
                    Path(directory) / "missing.mp4",
                    expected_frame_count=2,
                )


if __name__ == "__main__":
    unittest.main()
