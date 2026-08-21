#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def read_frame(video: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {index}: {video}")
    return frame


def label(
    frame: np.ndarray,
    title: str,
    subtitle: str,
    width: int = 400,
) -> np.ndarray:
    height = round(frame.shape[0] * width / frame.shape[1])
    image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height + 54, width, 3), 30, dtype=np.uint8)
    canvas[:height] = image
    cv2.putText(
        canvas,
        title[:54],
        (6, height + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        subtitle[:62],
        (6, height + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (185, 210, 225),
        1,
        cv2.LINE_AA,
    )
    return canvas


def sheet(rows: list[list[np.ndarray]], output: Path) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.unlink(missing_ok=True)
        return False
    cv2.imwrite(str(output), np.vstack([np.hstack(row) for row in rows]))
    return True


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path.cwd() / path


def deterministic_key(dataset_id: str, reason: str, scene_id: str) -> str:
    return hashlib.sha256(
        f"{dataset_id}:{reason}:{scene_id}".encode("utf-8")
    ).hexdigest()


def metadata_risk(record: dict) -> dict:
    path_value = record.get("effective_render_metadata_path")
    if not path_value:
        return {
            "full_trajectory_center_visible_fraction": None,
            "removed_optional_decor_count": 0,
        }
    try:
        metadata = json.loads(project_path(path_value).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "full_trajectory_center_visible_fraction": None,
            "removed_optional_decor_count": 0,
        }
    visualization = metadata.get("visualization") or {}
    diagnostics = (visualization.get("camera") or {}).get("diagnostics") or {}
    guard = (visualization.get("environment") or {}).get(
        "dynamic_visibility_guard"
    ) or {}
    return {
        "full_trajectory_center_visible_fraction": diagnostics.get(
            "full_trajectory_center_visible_fraction"
        ),
        "removed_optional_decor_count": len(
            guard.get("removed_optional_decor") or []
        ),
    }


def review_candidate(record: dict, audit_record: dict) -> dict:
    frame_statistics = audit_record["frames"]
    rendered_adaptation = (
        ((record.get("render_provenance") or {}).get("lighting_adaptation") or {}).get(
            "rendered_frame_exposure"
        )
        or {}
    )
    return {
        "record": record,
        "audit": audit_record,
        "metrics": {
            "minimum_luma_std": min(
                float(frame["luma_std"]) for frame in frame_statistics
            ),
            "mean_luma": float(
                np.mean([float(frame["mean_luma"]) for frame in frame_statistics])
            ),
            "maximum_highlight_fraction": max(
                float(frame["highlight_fraction_above_0_90"])
                for frame in frame_statistics
            ),
            "maximum_temporal_mad": max(
                [float(value) for value in audit_record["temporal_mad_from_first"]]
                or [0.0]
            ),
            "rendered_exposure_correction_ev": float(
                rendered_adaptation.get("cumulative_correction_ev", 0.0)
            ),
            **metadata_risk(record),
        },
        "reasons": [],
    }


def stratified_selection(manifest: dict, audit: dict, count: int = 40) -> list[dict]:
    audit_by_scene = {record["scene_id"]: record for record in audit["records"]}
    candidates = [
        review_candidate(record, audit_by_scene[record["scene_id"]])
        for record in manifest["records"]
    ]
    dataset_id = str(manifest.get("dataset_id", "physweep"))
    selected: list[dict] = []
    by_scene: dict[str, dict] = {}

    def add(candidate: dict, reason: str) -> None:
        scene_id = str(candidate["record"]["scene_id"])
        if scene_id not in by_scene:
            selected.append(candidate)
            by_scene[scene_id] = candidate
        if reason not in by_scene[scene_id]["reasons"]:
            by_scene[scene_id]["reasons"].append(reason)

    def deterministic_first(group: list[dict], reason: str) -> dict:
        return min(
            group,
            key=lambda item: deterministic_key(
                dataset_id,
                reason,
                str(item["record"]["scene_id"]),
            ),
        )

    for field, prefix in (
        ("motion_intent", "motion"),
        ("environment_id", "environment"),
    ):
        values = sorted({str(item["record"][field]) for item in candidates})
        for value in values:
            group = [item for item in candidates if str(item["record"][field]) == value]
            add(deterministic_first(group, f"{prefix}:{value}"), f"{prefix}:{value}")

    corrected = sorted(
        [
            item
            for item in candidates
            if item["metrics"]["rendered_exposure_correction_ev"] != 0.0
        ],
        key=lambda item: -abs(item["metrics"]["rendered_exposure_correction_ev"]),
    )
    for item in corrected[:4]:
        add(item, "rendered_exposure_correction")

    rankings = (
        ("lowest_motion_pixels", "maximum_temporal_mad", False, 4),
        ("lowest_luma_contrast", "minimum_luma_std", False, 4),
        ("darkest_frames", "mean_luma", False, 2),
        ("brightest_frames", "mean_luma", True, 2),
        ("highest_highlight_fraction", "maximum_highlight_fraction", True, 2),
    )
    for reason, metric, reverse, limit in rankings:
        ordered = sorted(
            candidates,
            key=lambda item: item["metrics"][metric],
            reverse=reverse,
        )
        for item in ordered[:limit]:
            add(item, reason)

    camera_risk = [
        item
        for item in candidates
        if item["metrics"]["full_trajectory_center_visible_fraction"] is not None
    ]
    camera_risk.sort(
        key=lambda item: item["metrics"]["full_trajectory_center_visible_fraction"]
    )
    for item in camera_risk[:4]:
        add(item, "lowest_full_trajectory_visibility")

    decor_risk = sorted(
        [
            item
            for item in candidates
            if item["metrics"]["removed_optional_decor_count"] > 0
        ],
        key=lambda item: -item["metrics"]["removed_optional_decor_count"],
    )
    for item in decor_risk[:4]:
        add(item, "optional_decor_removed")

    remaining = [
        item
        for item in candidates
        if item["record"]["scene_id"] not in by_scene
    ]
    remaining.sort(
        key=lambda item: deterministic_key(
            dataset_id,
            "deterministic_fill",
            str(item["record"]["scene_id"]),
        )
    )
    for item in remaining:
        if len(selected) >= count:
            break
        add(item, "deterministic_fill")
    return selected[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    audit = json.loads(args.audit.read_text())
    by_name = {Path(row["video_path"]).name: row for row in manifest["records"]}

    flagged_names = [error.split(": ", 1)[1] for error in audit["errors"]
                     if error.startswith("motion not visible")]
    metrics = []
    flagged_rows = []
    for name in flagged_names:
        video = Path(by_name[name]["video_path"])
        frames = [read_frame(video, index) for index in (0, 8, 16, 24)]
        base = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.int16)
        diffs = [np.abs(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.int16) - base)
                 for frame in frames[1:]]
        metrics.append({
            "video": name,
            "max_p99_9": round(max(float(np.percentile(diff, 99.9)) for diff in diffs), 3),
            "max_changed_fraction_gt_8": round(max(float(np.mean(diff > 8)) for diff in diffs), 7),
        })
        flagged_rows.append([label(frame, f"{name[:30]} f{index}", "flagged motion")
                             for frame, index in zip(frames, (0, 8, 16, 24))])
    sheet(flagged_rows, args.output_dir / "flagged_motion_review.jpg")

    chosen = stratified_selection(manifest, audit, 40)
    review_rows: list[list[np.ndarray]] = []
    for selection_index, item in enumerate(chosen, start=1):
        row = item["record"]
        video = project_path(row["video_path"])
        review_indices = (0, 48, 96)
        frames = [read_frame(video, index) for index in review_indices]
        reason = ",".join(item["reasons"])
        review_rows.append(
            [
                label(
                    frame,
                    f"#{selection_index:02d} {row['motion_intent']} f{frame_index}",
                    f"{row['environment_id']} | {reason}",
                )
                for frame, frame_index in zip(frames, review_indices)
            ]
        )
    for stale in args.output_dir.glob("stratified_motion_review_*.jpg"):
        stale.unlink()
    page_paths = []
    for page_index in range(0, len(review_rows), 10):
        path = args.output_dir / f"stratified_motion_review_{page_index // 10 + 1:02d}.jpg"
        sheet(review_rows[page_index : page_index + 10], path)
        page_paths.append(path)
    (args.output_dir / "stratified_selection.json").write_text(
        json.dumps(
            [
                {
                    "scene_id": item["record"]["scene_id"],
                    "motion_intent": item["record"]["motion_intent"],
                    "environment_id": item["record"]["environment_id"],
                    "video_path": item["record"]["video_path"],
                    "selection_reasons": item["reasons"],
                    "risk_metrics": item["metrics"],
                }
                for item in chosen
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "local_motion_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "flagged": len(metrics),
                "selected": len(chosen),
                "motion_coverage": dict(
                    Counter(item["record"]["motion_intent"] for item in chosen)
                ),
                "environment_coverage": dict(
                    Counter(item["record"]["environment_id"] for item in chosen)
                ),
                "review_pages": [str(path) for path in page_paths],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
