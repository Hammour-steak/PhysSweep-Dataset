#!/usr/bin/env python3
"""Create a deterministic image contact sheet with Blender's image API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


def blender_argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--gutter", type=int, default=4)
    args = parser.parse_args(blender_argv())
    paths = sorted(args.input_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"no images match {args.pattern} below {args.input_dir}")
    if args.columns < 1 or args.gutter < 0:
        raise SystemExit("columns and gutter are invalid")
    images = [bpy.data.images.load(str(path), check_existing=False) for path in paths]
    width, height = images[0].size
    if any(tuple(image.size) != (width, height) for image in images):
        raise SystemExit("all source images must have the same dimensions")
    rows = (len(images) + args.columns - 1) // args.columns
    canvas_width = args.columns * width + (args.columns + 1) * args.gutter
    canvas_height = rows * height + (rows + 1) * args.gutter
    canvas = bpy.data.images.new("physweep_contact_sheet", width=canvas_width, height=canvas_height, alpha=True)
    pixels = [0.025, 0.025, 0.025, 1.0] * (canvas_width * canvas_height)
    for index, image in enumerate(images):
        source = list(image.pixels[:])
        column = index % args.columns
        row = rows - 1 - index // args.columns
        offset_x = args.gutter + column * (width + args.gutter)
        offset_y = args.gutter + row * (height + args.gutter)
        for y in range(height):
            source_start = y * width * 4
            target_start = ((offset_y + y) * canvas_width + offset_x) * 4
            pixels[target_start : target_start + width * 4] = source[source_start : source_start + width * 4]
    canvas.pixels.foreach_set(pixels)
    canvas.filepath_raw = str(args.output.resolve())
    canvas.file_format = "PNG"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save()
    print(f"images={len(images)} output={args.output}")


if __name__ == "__main__":
    main()
