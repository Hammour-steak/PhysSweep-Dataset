#!/usr/bin/env python3
"""Extract deterministic review frames from a directory of rendered videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_frames(
    input_dir: Path,
    output_dir: Path,
    pattern: str,
    frame_indices: list[int],
) -> list[dict[str, object]]:
    import cv2

    videos = sorted(input_dir.glob(pattern))
    if not videos:
        raise FileNotFoundError(f"no videos match {pattern} below {input_dir}")
    if any(index < 0 for index in frame_indices):
        raise ValueError("frame indices must be nonnegative")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for video in videos:
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"cannot open video: {video}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        written = []
        for frame_index in frame_indices:
            if frame_index >= frame_count:
                raise ValueError(
                    f"frame {frame_index} is outside {video.name} ({frame_count} frames)"
                )
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"cannot decode frame {frame_index} from {video}")
            output_path = output_dir / f"{video.stem}__frame_{frame_index:03d}.png"
            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"cannot write review image: {output_path}")
            written.append(str(output_path))
        capture.release()
        records.append(
            {
                "video_path": str(video),
                "frame_count": frame_count,
                "frames": written,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--frame-indices", type=int, nargs="+", required=True)
    args = parser.parse_args()
    records = extract_frames(
        args.input_dir.resolve(),
        args.output_dir.resolve(),
        str(args.pattern),
        [int(index) for index in args.frame_indices],
    )
    print(json.dumps({"video_count": len(records), "records": records}, indent=2))


if __name__ == "__main__":
    main()
