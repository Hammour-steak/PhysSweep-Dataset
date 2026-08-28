#!/usr/bin/env python3
"""Create compact labeled contact sheets for a PhysAssets classification JSONL."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args()

    groups: dict[str, list[dict]] = defaultdict(list)
    for line in args.input.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            groups[str(row.get("decision", "unknown"))].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    tile_w, tile_h = 220, 260
    page_size = args.columns * args.rows

    for decision, rows in sorted(groups.items()):
        rows.sort(key=lambda row: int(row["sample_id"]))
        for page_index in range(0, len(rows), page_size):
            page_rows = rows[page_index : page_index + page_size]
            canvas = Image.new(
                "RGB",
                (args.columns * tile_w, args.rows * tile_h),
                (32, 32, 32),
            )
            draw = ImageDraw.Draw(canvas)
            for index, row in enumerate(page_rows):
                x = (index % args.columns) * tile_w
                y = (index // args.columns) * tile_h
                image = Image.open(row["representative_image"]).convert("RGB")
                image.thumbnail((tile_w - 12, 205), Image.Resampling.LANCZOS)
                canvas.paste(
                    image,
                    (x + (tile_w - image.width) // 2, y + 4),
                )
                label = (
                    f'{row["sample_id"]} | {row.get("object_name", "")}\n'
                    f'{row.get("material", "")} | '
                    f'{float(row.get("confidence", 0)):.2f}'
                )
                draw.multiline_text(
                    (x + 6, y + 210),
                    label,
                    fill=(245, 245, 245),
                    font=font,
                    spacing=3,
                )
            output = args.output_dir / (
                f"{decision}_{page_index // page_size + 1:03d}.jpg"
            )
            canvas.save(output, quality=92)
            print(output)


if __name__ == "__main__":
    main()
