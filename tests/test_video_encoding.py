from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.video_encoding import normalize_h264_container


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
    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is unavailable")
    def test_normalization_removes_volatile_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.mp4"
            second = root / "second.mp4"
            make_video(first, "2026/08/25 14:36:10")
            make_video(second, "2026/08/27 18:43:36")
            self.assertNotEqual(file_sha256(first), file_sha256(second))

            normalize_h264_container(first)
            normalize_h264_container(second)

            self.assertEqual(file_sha256(first), file_sha256(second))
            self.assertFalse(list(root.glob(".*.normalized-*.mp4")))

    def test_normalization_requires_rendered_video(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                FileNotFoundError, "rendered video is missing"
            ):
                normalize_h264_container(Path(directory) / "missing.mp4")


if __name__ == "__main__":
    unittest.main()
