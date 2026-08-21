#!/usr/bin/env python3
"""Score multi-view visual quality for admitted PhysAssets proxy candidates."""

from __future__ import annotations

import argparse
import json
import math
import struct
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


VIEW_IDS = ("000", "005", "010", "015", "020", "030", "035", "039")
GLB_JSON_CHUNK = 0x4E4F534A


def glb_stats(path: Path) -> dict:
    result = {"material_count": 0, "texture_count": 0, "image_count": 0}
    try:
        with path.open("rb") as handle:
            magic, version, length = struct.unpack("<4sII", handle.read(12))
            if magic != b"glTF" or version != 2:
                return result
            while handle.tell() < length:
                chunk_length, chunk_type = struct.unpack("<II", handle.read(8))
                payload = handle.read(chunk_length)
                if chunk_type != GLB_JSON_CHUNK:
                    continue
                document = json.loads(payload.rstrip(b"\x00 \t\r\n"))
                result = {
                    "material_count": len(document.get("materials", [])),
                    "texture_count": len(document.get("textures", [])),
                    "image_count": len(document.get("images", [])),
                }
                break
    except Exception:
        result["glb_metadata_error"] = True
    return result


def dhash(gray: np.ndarray) -> str:
    image = Image.fromarray(np.uint8(np.clip(gray * 255.0, 0, 255)), mode="L")
    small = np.asarray(image.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.float32)
    bits = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def image_metrics(path: Path) -> dict:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    height, width = gray.shape
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - background[None, None, :], axis=2)
    mask = distance > 0.035
    mask = ndimage.binary_opening(mask, iterations=1)
    mask = ndimage.binary_closing(mask, iterations=1)
    if mask.sum() < 64:
        return {"valid": False, "occupancy": float(mask.mean()), "dhash": dhash(gray)}
    labels, count = ndimage.label(mask)
    component_sizes = np.bincount(labels.ravel())[1:]
    threshold = max(16, int(component_sizes.max() * 0.01))
    significant_ids = np.flatnonzero(component_sizes >= threshold) + 1
    mask = np.isin(labels, significant_ids)
    occupancy = float(mask.mean())
    primary_id = int(np.argmax(component_sizes) + 1)
    primary_y, primary_x = np.where(labels == primary_id)
    primary_bbox_area = int(
        (primary_x.max() - primary_x.min() + 1)
        * (primary_y.max() - primary_y.min() + 1)
    )
    silhouette_fill = float(component_sizes[primary_id - 1] / max(1, primary_bbox_area))
    clipped = bool(
        primary_x.min() <= 1
        or primary_y.min() <= 1
        or primary_x.max() >= width - 2
        or primary_y.max() >= height - 2
    )
    significant_components = int(len(significant_ids))
    core = ndimage.binary_erosion(mask, iterations=3)
    if core.sum() < 64:
        core = mask
    laplacian = ndimage.laplace(gray)
    residual = gray - ndimage.gaussian_filter(gray, sigma=1.4)
    values = gray[core]
    histogram, _ = np.histogram(values, bins=32, range=(0.0, 1.0), density=False)
    probabilities = histogram / max(1, histogram.sum())
    probabilities = probabilities[probabilities > 0]
    entropy = float(-(probabilities * np.log2(probabilities)).sum() / 5.0)
    rg = rgb[..., 0][core] - rgb[..., 1][core]
    yb = 0.5 * (rgb[..., 0][core] + rgb[..., 1][core]) - rgb[..., 2][core]
    colorfulness = float(math.sqrt(float(rg.var() + yb.var())) + 0.3 * math.sqrt(float(rg.mean() ** 2 + yb.mean() ** 2)))
    return {
        "valid": True,
        "occupancy": occupancy,
        "clipped": clipped,
        "significant_components": significant_components,
        "silhouette_fill": silhouette_fill,
        "sharpness": float(np.mean(laplacian[core] ** 2)),
        "texture_detail": float(np.mean(residual[core] ** 2)),
        "luminance_entropy": entropy,
        "colorfulness": colorfulness,
        "mean_luminance": float(values.mean()),
        "dhash": dhash(gray),
    }


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0


def score_one(entry: dict, project_root: Path, extracted_root: Path, audit: dict[str, dict]) -> dict:
    sample_id = str(entry["sample_id"])
    proxy = json.loads(Path(entry["proxy_json"]).read_text(encoding="utf-8"))
    views = []
    for view_id in VIEW_IDS:
        path = extracted_root / sample_id / f"{view_id}.png"
        if path.exists():
            views.append(image_metrics(path))
    valid = [item for item in views if item.get("valid")]
    source = Path(proxy["source_glb"])
    if not source.is_absolute():
        source = project_root / source
    mesh_audit = audit.get(sample_id, {})
    return {
        "sample_id": sample_id,
        "objaverse_uid": str(entry["objaverse_uid"]),
        "name": str(entry["name"]),
        "method": str(entry["method"]),
        "proxy_json": str(entry["proxy_json"]),
        "source_glb": str(source),
        "proxy_to_visual_hull_volume_ratio": float(entry["proxy_to_visual_hull_volume_ratio"]),
        "view_count": len(views),
        "valid_view_count": len(valid),
        "median_occupancy": median([item["occupancy"] for item in valid]),
        "clipped_view_count": sum(bool(item.get("clipped")) for item in valid),
        "max_significant_components": max((int(item.get("significant_components", 0)) for item in valid), default=0),
        "median_silhouette_fill": median([item["silhouette_fill"] for item in valid]),
        "median_sharpness": median([item["sharpness"] for item in valid]),
        "median_texture_detail": median([item["texture_detail"] for item in valid]),
        "median_luminance_entropy": median([item["luminance_entropy"] for item in valid]),
        "median_colorfulness": median([item["colorfulness"] for item in valid]),
        "median_luminance": median([item["mean_luminance"] for item in valid]),
        "front_view_dhash": valid[0]["dhash"] if valid else "",
        "vertex_count": int(mesh_audit.get("vertex_count", 0)),
        "face_count": int(mesh_audit.get("face_count", 0)),
        "component_count": int(mesh_audit.get("component_count", 0)),
        "largest_component_fraction": float(mesh_audit.get("largest_component_fraction", 0.0)),
        **glb_stats(source),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passed-index", type=Path, required=True)
    parser.add_argument("--mesh-audit", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    entries = [json.loads(line) for line in args.passed_index.read_text(encoding="utf-8").splitlines() if line.strip()]
    audit = {str(row["sample_id"]): row for row in (json.loads(line) for line in args.mesh_audit.read_text(encoding="utf-8").splitlines() if line.strip())}
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score_one, entry, args.project_root, args.extracted_root, audit): entry for entry in entries}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if index % 100 == 0 or index == len(entries):
                print(f"scored {index}/{len(entries)}", flush=True)
    results.sort(key=lambda item: int(item["sample_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(item, ensure_ascii=True) + "\n" for item in results), encoding="utf-8")


if __name__ == "__main__":
    main()
