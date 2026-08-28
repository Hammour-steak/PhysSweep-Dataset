#!/usr/bin/env python3
"""Classify PhysAssets renders for rigid-body simulation suitability."""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageOps


VIEW_NAMES = ("000.png", "006.png", "012.png", "018.png")
OUTPUT_LOCK = threading.Lock()

SYSTEM_PROMPT = """You curate isolated 3D assets for rigid-body physics videos.
The four panels always show different views of the SAME asset; never count the
panels as multiple objects. Judge geometry and physical role, not aesthetics,
text, graphics, realism, or artistic subject matter.

Classify as direct when it is a compact passive object that can immediately be
used in generic drop, slide, roll, and collision scenes with one simple rigid or
compound proxy. Typical direct assets are balls, bottles, cans, bowls, cups,
boxes, books, stones, weights, simple containers, and compact hand tools.
Classify as specialized when it is rigid or can be approximated as rigid, but
needs a dedicated proxy or semantic rule. Examples include vehicles, bicycles,
furniture used as a moving object, musical instruments, complex machinery,
figurines, statues, animals as models, thin complex tools, and objects with
fixed-but-complex assemblies.
Classify as support when it is rigid but mainly useful as a static environment
or support, such as a table, shelf, cabinet, large furniture, wall, or building.
Exclude cloth, flexible bags, plants, ordinary food, characters intended to
articulate, liquids, obvious soft/deforming items, disconnected object sets,
whole multi-object scenes, mechanisms that require moving joints, and broken or
unrecognizable meshes. Also exclude assets that include a large visible ground
plane or unrelated support geometry baked into the mesh. A material hint is
only a hint: paper boxes may be direct while paper bags are excluded.
Return strict JSON only."""

USER_PROMPT = """The four panels are views of one PhysAssets sample.
Material hint: {material}
Return:
{{
  "object_name": "short English noun",
  "decision": "direct|specialized|support|exclude|uncertain",
  "confidence": 0.0,
  "semantic_family": "container|sports|tool|household|furniture|vehicle|instrument|decorative|animal|character|food|plant|clothing|architecture|mechanism|natural|other",
  "structure": "single|simple_compound|articulated|multi_object|scene|unknown",
  "size_role": "handheld|small_movable|medium_movable|large_support|unknown",
  "proxy_difficulty": "easy|moderate|hard",
  "has_visible_base_or_ground": false,
  "has_flexible_parts": false,
  "has_moving_parts": false,
  "is_complete_mesh": true,
  "reason": "short concrete reason"
}}
Use direct conservatively. Use specialized rather than direct when a dedicated
object rule would be needed."""


def make_collage(sample_dir: Path) -> str:
    canvas = Image.new("RGB", (512, 512), "white")
    for index, name in enumerate(VIEW_NAMES):
        image = Image.open(sample_dir / name).convert("RGB")
        image.thumbnail((248, 248), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (256, 256), "white")
        tile.paste(image, ((256 - image.width) // 2, (256 - image.height) // 2))
        tile = ImageOps.expand(tile, border=1, fill=(190, 190, 190))
        x = (index % 2) * 256
        y = (index // 2) * 256
        canvas.paste(tile.crop((0, 0, 256, 256)), (x, y))
    buffer = io.BytesIO()
    canvas.save(buffer, format="JPEG", quality=88, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def parse_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0]
    value = json.loads(text)
    required = {
        "object_name",
        "decision",
        "confidence",
        "semantic_family",
        "structure",
        "size_role",
        "proxy_difficulty",
        "has_visible_base_or_ground",
        "has_flexible_parts",
        "has_moving_parts",
        "is_complete_mesh",
        "reason",
    }
    if not isinstance(value, dict) or not required.issubset(value):
        raise ValueError("response is missing required fields")
    if value["decision"] not in {
        "direct",
        "specialized",
        "support",
        "exclude",
        "uncertain",
    }:
        raise ValueError("invalid decision")
    return value


def classify(row: dict, api_url: str, model: str, timeout: int) -> dict:
    sample_dir = Path(row["representative_image"]).parent
    image_url = make_collage(sample_dir)
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 160,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                    {
                        "type": "text",
                        "text": USER_PROMPT.format(material=row["material"]),
                    },
                ],
            },
        ],
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    last_error = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = json.load(response)
            content = body["choices"][0]["message"]["content"]
            result = parse_response(content)
            return {**row, **result, "classifier_status": "ok"}
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(1 + attempt * 2)
    return {
        **row,
        "object_name": "",
        "decision": "uncertain",
        "confidence": 0,
        "semantic_family": "other",
        "structure": "unknown",
        "size_role": "unknown",
        "proxy_difficulty": "hard",
        "has_visible_base_or_ground": False,
        "has_flexible_parts": False,
        "has_moving_parts": False,
        "is_complete_mesh": False,
        "reason": last_error[:300],
        "classifier_status": "error",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8010/v1/chat/completions",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "PHYSWEEP_VISION_MODEL", "checkpoints/Qwen3-VL-8B-Instruct"
        ),
    )
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    with args.index.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.shuffle_seed is not None:
        random.Random(args.shuffle_seed).shuffle(rows)
    completed: set[str] = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            try:
                completed.add(str(json.loads(line)["sample_id"]))
            except (KeyError, json.JSONDecodeError):
                continue
    pending = [row for row in rows if row["sample_id"] not in completed]
    if args.limit is not None:
        pending = pending[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()
    done = 0
    with args.output.open("a", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    classify,
                    row,
                    args.api_url,
                    args.model,
                    args.timeout,
                ): row["sample_id"]
                for row in pending
            }
            for future in as_completed(futures):
                result = future.result()
                with OUTPUT_LOCK:
                    output.write(json.dumps(result, ensure_ascii=True) + "\n")
                    output.flush()
                done += 1
                if done % 25 == 0 or done == len(pending):
                    elapsed = max(time.monotonic() - start, 0.001)
                    print(
                        f"completed={done}/{len(pending)} "
                        f"rate={done / elapsed:.2f}/s",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
