#!/usr/bin/env python3
"""Build a compact labeled contact sheet from review images."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="**/*.png")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--label-height", type=int, default=28)
    args = parser.parse_args()

    input_root = args.input_dir.resolve()
    paths = sorted(input_root.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no images match {args.pattern} below {args.input_dir}")
    if min(args.columns, args.thumb_width, args.label_height) <= 0:
        raise SystemExit("columns and cell dimensions must be positive")

    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise SystemExit(f"cannot decode {paths[0]}")
    thumb_height = max(1, round(first.shape[0] * args.thumb_width / first.shape[1]))
    rows = math.ceil(len(paths) / args.columns)
    cell_height = thumb_height + args.label_height
    sheet = np.full(
        (rows * cell_height, args.columns * args.thumb_width, 3),
        24,
        dtype=np.uint8,
    )

    for index, path in enumerate(paths):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"cannot decode {path}")
        image = cv2.resize(
            image,
            (args.thumb_width, thumb_height),
            interpolation=cv2.INTER_AREA,
        )
        row, column = divmod(index, args.columns)
        x0 = column * args.thumb_width
        y0 = row * cell_height
        sheet[y0 : y0 + thumb_height, x0 : x0 + args.thumb_width] = image
        relative = path.relative_to(input_root)
        label = (
            path.stem
            if len(relative.parts) == 1
            else f"{relative.parts[0]}/{path.stem}"
        )[:46]
        cv2.putText(
            sheet,
            label,
            (x0 + 6, y0 + thumb_height + args.label_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output.resolve()), sheet):
        raise SystemExit(f"cannot write {args.output}")
    print(f"images={len(paths)} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
