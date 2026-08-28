#!/usr/bin/env python3
"""Build up-to-date side and top review sheets for the PhysAssets core pool."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tools.core.json_io import read_jsonl

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=8)
    args = parser.parse_args()

    rows = read_jsonl(args.selection_dir / "core_assets.jsonl")
    overlay_dir = args.selection_dir / "proxy_overlays"
    font = ImageFont.load_default()
    tile_w, tile_h = 300, 270

    for view in ("side", "top"):
        review_rows = []
        for row in rows:
            sid = str(row["sample_id"])
            matches = sorted(overlay_dir.glob(f"{sid}_*_{view}.png"))
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {view} overlay for {sid}, found {len(matches)}")
            review_rows.append({
                "sample_id": sid,
                "category": row["category"],
                "quality_score": row["quality_score"],
                "representative_image": str(matches[0]),
            })

        index_path = args.selection_dir / f"core_{view}_review.jsonl"
        index_path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in review_rows),
            encoding="utf-8",
        )
        canvas = Image.new(
            "RGB",
            (args.columns * tile_w, math.ceil(len(review_rows) / args.columns) * tile_h),
            (30, 30, 30),
        )
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(review_rows):
            x = (index % args.columns) * tile_w
            y = (index // args.columns) * tile_h
            image = Image.open(row["representative_image"]).convert("RGB")
            image.thumbnail((tile_w - 12, 225), Image.Resampling.LANCZOS)
            canvas.paste(image, (x + (tile_w - image.width) // 2, y + 4))
            draw.text(
                (x + 6, y + 232),
                f'{row["sample_id"]} | {row["category"]} | {row["quality_score"]:.3f}',
                fill=(245, 245, 245),
                font=font,
            )
        canvas.save(args.selection_dir / f"core_{view}_overview.jpg", quality=92)
        print(index_path)


if __name__ == "__main__":
    main()
