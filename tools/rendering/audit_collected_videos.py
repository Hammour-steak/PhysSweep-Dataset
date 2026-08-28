#!/usr/bin/env python3
"""Verify video specification and sampled-frame visual health for a collected batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.rendering.appearance_adaptation import frame_statistics_within_fixed_limits
from tools.rendering.video_encoding import PROFILE_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_file(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def integrity_error(path: Path, declared_hash: str) -> str | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return f"missing or empty file: {path}"
    actual_hash = sha256(path)
    if actual_hash != declared_hash:
        return f"sha256 mismatch: {path}"
    return None


def sampled_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"cannot decode frame {index}")
    return frame


def frame_statistics(frame: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    return {
        "mean_luma": round(float(gray.mean()), 4),
        "mean_luminance": round(float(gray.mean() / 255.0), 7),
        "luma_std": round(float(gray.std()), 4),
        "mean_gradient": round(float(gradient.mean()), 4),
        "highlight_fraction_above_0_90": round(
            float(np.mean(gray.astype(np.float32) / 255.0 > 0.90)),
            7,
        ),
        "clipped_dark_fraction": round(float(np.mean(gray <= 2)), 7),
        "clipped_light_fraction": round(float(np.mean(gray >= 253)), 7),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--frame-count", type=int, default=97)
    parser.add_argument("--review-frames", type=int, nargs="+", default=[0, 32, 64, 95])
    parser.add_argument(
        "--visual-rules",
        type=Path,
        default=PROJECT_ROOT / "configs/visual_sampling.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest.resolve())
    if manifest.get("schema_version") not in {
        "physweep_decoupled_collected_renders_v3",
        "physweep_decoupled_collected_renders_v4",
    }:
        raise ValueError("video audit requires a v3 or v4 collected-render manifest")
    inspection_limits = load_json(args.visual_rules.resolve())[
        "asset_proxy_render"
    ]["inspection_limits"]
    records = []
    errors = []
    if manifest.get("source_manifest") and manifest.get("source_manifest_sha256"):
        error = integrity_error(
            project_file(str(manifest["source_manifest"])),
            str(manifest["source_manifest_sha256"]),
        )
        if error:
            errors.append(error)
    for source in manifest["records"]:
        video = project_file(str(source["video_path"]))
        integrity_ok = True
        integrity_targets = [
            (video, str(source["sha256"])),
        ]
        for path_key, hash_key in (
            ("metadata_path", "metadata_sha256"),
            (
                "effective_render_metadata_path",
                "effective_render_metadata_sha256",
            ),
        ):
            if source.get(path_key) or source.get(hash_key):
                if not source.get(path_key) or not source.get(hash_key):
                    errors.append(
                        f"incomplete integrity pair for {source['scene_id']}: {path_key}"
                    )
                    integrity_ok = False
                    continue
                integrity_targets.append(
                    (
                        project_file(str(source[path_key])),
                        str(source[hash_key]),
                    )
                )
        provenance = source.get("render_provenance") or {}
        if provenance.get("trajectory_path") or provenance.get("trajectory_sha256"):
            if not provenance.get("trajectory_path") or not provenance.get(
                "trajectory_sha256"
            ):
                errors.append(
                    f"incomplete trajectory provenance: {source['scene_id']}"
                )
                integrity_ok = False
            else:
                integrity_targets.append(
                    (
                        project_file(str(provenance["trajectory_path"])),
                        str(provenance["trajectory_sha256"]),
                    )
                )
        for path, declared_hash in integrity_targets:
            error = integrity_error(path, declared_hash)
            if error:
                errors.append(f"{source['scene_id']}: {error}")
                integrity_ok = False
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            errors.append(f"cannot open {video}")
            continue
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        frames = [sampled_frame(capture, index) for index in args.review_frames]
        capture.release()
        statistics = [frame_statistics(frame) for frame in frames]
        first_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
        temporal_differences = [
            np.abs(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                - first_gray
            )
            for frame in frames[1:]
        ]
        temporal_mad = [
            round(float(np.mean(difference)), 4)
            for difference in temporal_differences
        ]
        temporal_p99_9 = [
            round(float(np.percentile(difference, 99.9)), 4)
            for difference in temporal_differences
        ]
        temporal_changed_fraction = [
            round(float(np.mean(difference > 8.0)), 7)
            for difference in temporal_differences
        ]
        spec_ok = (
            width == args.width
            and height == args.height
            and abs(fps - args.fps) <= 0.01
            and frame_count == args.frame_count
        )
        encoding = source.get("video_encoding") or {}
        encoding_ok = (
            encoding.get("profile_version") == PROFILE_VERSION
            and encoding.get("constant_rate_factor") == "PERC_LOSSLESS"
            and int(encoding.get("gop_size_frames", 0)) >= frame_count
        )
        visual_ok = all(
            frame_statistics_within_fixed_limits(item)
            and item["mean_luminance"]
            <= float(inspection_limits["maximum_mean_luminance"])
            and item["highlight_fraction_above_0_90"]
            <= float(
                inspection_limits["maximum_highlight_fraction_above_0_90"]
            )
            for item in statistics
        )
        motion_visible = (
            max(temporal_mad) >= 0.25
            or (
                max(temporal_p99_9) >= 8.0
                and max(temporal_changed_fraction) >= 0.0001
            )
        )
        if not spec_ok:
            errors.append(f"video spec failed: {video.name}")
        if not encoding_ok:
            errors.append(f"video encoding profile failed: {video.name}")
        if not visual_ok:
            errors.append(f"sampled visual statistics failed: {video.name}")
        if not motion_visible:
            errors.append(f"motion not visible in sampled frames: {video.name}")
        records.append(
            {
                "scene_id": source["scene_id"],
                "video": video.name,
                "motion_intent": source["motion_intent"],
                "environment_id": source["environment_id"],
                "profile": source["profile"],
                "width": width,
                "height": height,
                "fps": round(fps, 4),
                "frame_count": frame_count,
                "integrity_ok": integrity_ok,
                "spec_ok": spec_ok,
                "encoding_ok": encoding_ok,
                "video_encoding": encoding,
                "visual_statistics_ok": visual_ok,
                "motion_visible_in_review_frames": motion_visible,
                "temporal_mad_from_first": temporal_mad,
                "temporal_p99_9_from_first": temporal_p99_9,
                "temporal_changed_fraction_gt_8_from_first": temporal_changed_fraction,
                "frames": [
                    {"frame": index, **item}
                    for index, item in zip(args.review_frames, statistics)
                ],
            }
        )
    report = {
        "schema_version": "physweep_decoupled_rendered_batch_review_v3",
        "dataset_id": manifest["dataset_id"],
        "video_count": len(records),
        "integrity_passed": sum(record["integrity_ok"] for record in records),
        "spec_passed": sum(record["spec_ok"] for record in records),
        "encoding_passed": sum(record["encoding_ok"] for record in records),
        "visual_statistics_passed": sum(
            record["visual_statistics_ok"] for record in records
        ),
        "visible_motion_passed": sum(
            record["motion_visible_in_review_frames"] for record in records
        ),
        "motions": dict(
            Counter(record["motion_intent"] for record in records)
        ),
        "environments": dict(
            Counter(record["environment_id"] for record in records)
        ),
        "profiles": dict(Counter(record["profile"] for record in records)),
        "lowest_temporal_mad": sorted(
            (
                {
                    "video": record["video"],
                    "max_mad": max(record["temporal_mad_from_first"]),
                }
                for record in records
            ),
            key=lambda item: item["max_mad"],
        )[:8],
        "errors": errors,
        "records": records,
    }
    write_json(args.output.resolve(), report)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "records"},
            indent=2,
        )
    )
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
