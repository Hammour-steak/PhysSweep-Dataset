#!/usr/bin/env python3
"""Render a bound PhysSweep PyBullet manifest with parallel Blender workers."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json_atomic as write_json
from tools.core.paths import resolve_project_path as project_path
from tools.rendering.blender_worker_environment import (
    build_egl_device_selector,
    isolated_blender_environment,
)
from tools.rendering.video_encoding import video_has_expected_frame_count

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def select_release_records(
    samples: list[dict[str, Any]],
    selection_manifest: dict[str, Any],
    source_schema_version: str | None,
) -> list[dict[str, Any]]:
    """Select the exact scene identities declared by a release manifest."""
    records = selection_manifest.get("records")
    if not isinstance(records, list) or int(
        selection_manifest.get("sample_count", -1)
    ) != len(records):
        raise ValueError("render selection manifest count is inconsistent")
    record_ids = [str(record.get("scene_id", "")) for record in records]
    if "" in record_ids or len(record_ids) != len(set(record_ids)):
        raise ValueError("render selection manifest scene ids are invalid")
    if source_schema_version is not None:
        records = [
            record
            for record in records
            if str(record.get("source_schema_version")) == source_schema_version
        ]
    if not records:
        raise ValueError("render selection manifest selected no records")
    requested = {str(record["scene_id"]): record for record in records}
    sample_ids = [str(sample.get("scene_id", "")) for sample in samples]
    if "" in sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("bound render manifest scene ids are invalid")
    available = set(sample_ids)
    missing = set(requested) - available
    if missing:
        raise ValueError(
            f"bound render manifest lacks {len(missing)} selected release scenes"
        )
    selected = []
    for sample in samples:
        scene_id = str(sample["scene_id"])
        record = requested.get(scene_id)
        if record is None:
            continue
        if str(sample.get("kind")) != str(record.get("kind")):
            raise ValueError(f"release and bound sweep kinds differ: {scene_id}")
        selected.append(sample)
    if len(selected) != len(requested):
        raise RuntimeError("release scene selection is incomplete")
    return selected


def validate_release_source_bindings(
    root: Path,
    samples: list[dict[str, Any]],
    selection_manifest: dict[str, Any],
    source_schema_version: str | None,
) -> None:
    """Verify each bound render input points to the selected release artifact."""
    records = selection_manifest["records"]
    if source_schema_version is not None:
        records = [
            record
            for record in records
            if str(record.get("source_schema_version")) == source_schema_version
        ]
    by_scene_id = {str(record["scene_id"]): record for record in records}
    for sample in samples:
        scene_id = str(sample["scene_id"])
        record = by_scene_id.get(scene_id)
        if record is None:
            raise ValueError(f"selected release record is missing: {scene_id}")
        metadata_path = project_path(root, str(sample["metadata_path"]))
        metadata_path.relative_to(root)
        if sha256(metadata_path) != str(sample["metadata_sha256"]):
            raise ValueError(f"render metadata hash mismatch: {metadata_path}")
        metadata = load_json(metadata_path)
        if str(metadata.get("scene_id")) != scene_id:
            raise ValueError(f"bound metadata scene id differs: {metadata_path}")
        source = metadata.get("source_metadata")
        if not isinstance(source, dict):
            raise ValueError(f"bound metadata lacks source binding: {metadata_path}")
        source_path = project_path(root, str(source.get("path", "")))
        release_path = project_path(root, str(record.get("path", "")))
        source_path.relative_to(root)
        release_path.relative_to(root)
        expected_hash = str(record.get("metadata_sha256", ""))
        if (
            source_path != release_path
            or str(source.get("sha256", "")) != expected_hash
            or not source_path.is_file()
            or sha256(source_path) != expected_hash
        ):
            raise ValueError(
                f"bound metadata source differs from release: {metadata_path}"
            )
        if str(load_json(source_path).get("scene_id")) != scene_id:
            raise ValueError(
                f"release source metadata scene id differs: {source_path}"
            )


def select_sweep_kind(
    root: Path,
    samples: list[dict[str, Any]],
    sweep_kind: str | None,
) -> list[dict[str, Any]]:
    """Select base or derived records using the immutable metadata contract."""
    if sweep_kind is None:
        return samples
    if sweep_kind not in {"base", "sweep"}:
        raise ValueError(f"unsupported sweep kind: {sweep_kind}")
    selected = []
    for sample in samples:
        metadata_path = project_path(root, str(sample["metadata_path"]))
        metadata_path.relative_to(root)
        if sha256(metadata_path) != str(sample["metadata_sha256"]):
            raise ValueError(f"render metadata hash mismatch: {metadata_path}")
        metadata_kind = str(load_json(metadata_path).get("sweep", {}).get("kind"))
        record_kind = str(sample.get("kind"))
        if metadata_kind not in {"base", "sweep"} or record_kind != metadata_kind:
            raise ValueError(
                f"render manifest and metadata sweep kinds differ: {metadata_path}"
            )
        if metadata_kind == sweep_kind:
            selected.append(sample)
    return selected


def result_manifest_name(sweep_kind: str | None) -> str:
    return {
        None: "render_manifest.json",
        "base": "base_render_manifest.json",
        "sweep": "derived_render_manifest.json",
    }[sweep_kind]


def reusable_render_record(
    root: Path,
    output_root: Path,
    sample: dict[str, Any],
    metadata_path: Path,
    metadata: dict[str, Any],
    record: dict[str, Any],
    first_frame_only: bool,
    gpu: int,
    script: Path,
) -> bool:
    render = metadata["visualization"]["render"]
    frame_dir = project_path(root, render["inspection_frame_dir"])
    if output_root not in frame_dir.parents:
        raise ValueError("inspection frames must remain below the render output root")
    expected_scope = "first_frame_only" if first_frame_only else "full_animation"
    trajectory = metadata["trajectory"]
    trajectory_path = project_path(root, trajectory["path"])
    trajectory_path.relative_to(root)
    inspection_frames = [
        project_path(root, value) for value in record.get("inspection_frames", [])
    ]
    expected_inspection_frames = (
        [int(render["frame_start"])]
        if first_frame_only
        else [int(value) for value in render["inspection_frames"]]
    )
    expected_inspection_paths = [
        frame_dir / f"frame_{frame:04d}.png" for frame in expected_inspection_frames
    ]
    log_path = output_root / "logs" / f"{sample['scene_id']}.log"
    egl_marker = f"PhysSweep EGL selector: CUDA device {gpu} "
    egl_verified = bool(record.get("egl_device_verified")) or (
        log_path.is_file()
        and egl_marker in log_path.read_text(encoding="utf-8", errors="replace")
    )
    reusable = (
        str(record.get("scene_id")) == str(sample["scene_id"])
        and record.get("implementation")
        == {"path": str(script), "sha256": sha256(script)}
        and record.get("render_scope") == expected_scope
        and project_path(root, str(record.get("metadata_path"))) == metadata_path
        and str(record.get("metadata_sha256")) == sha256(metadata_path)
        and str(sample.get("metadata_sha256")) == sha256(metadata_path)
        and project_path(root, str(record.get("trajectory_path"))) == trajectory_path
        and str(record.get("trajectory_sha256")) == sha256(trajectory_path)
        and str(trajectory["sha256"]) == sha256(trajectory_path)
        and inspection_frames == expected_inspection_paths
        and all(
            frame.parent == frame_dir
            and frame.is_file()
            and frame.stat().st_size > 0
            for frame in inspection_frames
        )
        and egl_verified
    )
    if not reusable:
        return False
    if first_frame_only:
        return record.get("video_path") is None and record.get("video_sha256") is None
    video_path = project_path(root, render["video_path"])
    if output_root not in video_path.parents:
        raise ValueError("video must remain below the render output root")
    expected_frame_count = int(render["frame_end"]) - int(render["frame_start"]) + 1
    if not (
        project_path(root, str(record.get("video_path"))) == video_path
        and video_path.is_file()
        and video_path.stat().st_size > 0
        and str(record.get("video_sha256")) == sha256(video_path)
        and video_has_expected_frame_count(video_path, expected_frame_count)
    ):
        return False
    mask_output = record.get("instance_mask_output")
    validation = mask_output.get("validation") if isinstance(mask_output, dict) else None
    if not isinstance(validation, dict):
        return False
    for object_id, object_record in mask_output.get("objects", {}).items():
        report = validation.get("objects", {}).get(str(object_id))
        object_dir = project_path(root, object_record["directory"])
        if output_root not in object_dir.parents:
            raise ValueError("instance masks must remain below the render output root")
        masks = list(object_dir.glob("frame_*.png"))
        expected_masks = [
            object_dir / f"frame_{frame:04d}.png"
            for frame in range(
                int(render["frame_start"]), int(render["frame_end"]) + 1
            )
        ]
        if (
            not isinstance(report, dict)
            or int(report.get("frame_count", -1)) != expected_frame_count
            or len(masks) != expected_frame_count
            or any(
                not mask.is_file() or mask.stat().st_size == 0
                for mask in expected_masks
            )
        ):
            return False
    return bool(mask_output.get("objects"))


def worker(
    root: str,
    blender: str,
    script: str,
    output_root: str,
    sample: dict[str, Any],
    gpu: int,
    first_frame_only: bool,
    selector_path: str,
    resume: bool,
) -> dict[str, Any]:
    project_root = Path(root)
    metadata_path = project_path(project_root, str(sample["metadata_path"]))
    metadata_path.relative_to(project_root)
    metadata = load_json(metadata_path)
    record_path = Path(output_root) / "frames" / str(sample["scene_id"]) / "render_record.json"
    if resume and record_path.is_file():
        record = load_json(record_path)
        if reusable_render_record(
            project_root,
            Path(output_root),
            sample,
            metadata_path,
            metadata,
            record,
            first_frame_only,
            gpu,
            Path(script),
        ):
            if not record.get("egl_device_verified"):
                record["egl_device_verified"] = True
                write_json(record_path, record)
            return {
                "scene_id": str(sample["scene_id"]),
                "ok": True,
                "returncode": 0,
                "gpu": None,
                "egl_device_verified": True,
                "wall_time_s": 0.0,
                "log_path": None,
                "render_record": record,
                "reused": True,
            }
    record_path.unlink(missing_ok=True)
    if not first_frame_only:
        video_path = project_path(
            project_root, metadata["visualization"]["render"]["video_path"]
        )
        if Path(output_root) not in video_path.parents:
            raise ValueError("video must remain below the render output root")
        video_path.unlink(missing_ok=True)
    command = [
        blender,
        "-b",
        "--python",
        script,
        "--",
        "--metadata",
        str(metadata_path),
        "--root",
        str(project_root),
    ]
    if first_frame_only:
        command.append("--first-frame-only")
    started = time.perf_counter()
    with isolated_blender_environment(gpu, Path(selector_path)) as (
        environment,
        selector_marker,
    ):
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    log_path = Path(output_root) / "logs" / f"{sample['scene_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    record = load_json(record_path) if completed.returncode == 0 and record_path.exists() else None
    selector_verified = selector_marker in completed.stdout
    if record is not None:
        record["egl_device_verified"] = selector_verified
        write_json(record_path, record)
    ok = (
        completed.returncode == 0
        and record is not None
        and selector_verified
        and reusable_render_record(
            project_root,
            Path(output_root),
            sample,
            metadata_path,
            metadata,
            record,
            first_frame_only,
            gpu,
            Path(script),
        )
    )
    return {
        "scene_id": str(sample["scene_id"]),
        "ok": ok,
        "returncode": completed.returncode,
        "gpu": gpu,
        "egl_device_verified": selector_verified,
        "wall_time_s": round(time.perf_counter() - started, 6),
        "log_path": str(log_path),
        "render_record": record,
        "reused": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--blender",
        type=Path,
        default=PROJECT_ROOT / "runtime/blender-3.4.0-linux-x64/blender",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6")
    parser.add_argument("--first-frame-only", action="store_true")
    parser.add_argument(
        "--sweep-kind",
        choices=("base", "sweep"),
        help="Render only canonical bases or only derived one-factor sweeps.",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        help="Hashed release metadata manifest whose scene ids define this run.",
    )
    parser.add_argument(
        "--selection-manifest-sha256",
        help="Required expected hash for --selection-manifest.",
    )
    parser.add_argument(
        "--selection-source-schema",
        help="Select only release records with this source_schema_version.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only hash-verified complete render outputs.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--profiles",
        nargs="+",
        help="Render only samples bound to these scene-visual profile ids.",
    )
    parser.add_argument(
        "--result-manifest",
        type=Path,
        help="Explicit summary path below the render output root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    root = args.root.resolve()
    manifest_path = project_path(root, args.manifest)
    manifest_path.relative_to(root)
    manifest = load_json(manifest_path)
    samples = list(manifest["samples"])
    selection_binding = None
    selection_manifest = None
    if (args.selection_manifest is None) != (
        args.selection_manifest_sha256 is None
    ):
        raise ValueError(
            "selection manifest and expected sha256 must be provided together"
        )
    if args.selection_source_schema and args.selection_manifest is None:
        raise ValueError("selection source schema requires a selection manifest")
    if args.selection_manifest is not None:
        selection_path = project_path(root, args.selection_manifest)
        expected_hash = str(args.selection_manifest_sha256)
        if not selection_path.is_file() or sha256(selection_path) != expected_hash:
            raise ValueError("render selection manifest hash mismatch")
        selection_manifest = load_json(selection_path)
        samples = select_release_records(
            samples,
            selection_manifest,
            args.selection_source_schema,
        )
        selection_binding = {
            "path": str(selection_path),
            "sha256": expected_hash,
            "source_schema_version": args.selection_source_schema,
            "selected_sample_count": len(samples),
        }
    samples = select_sweep_kind(root, samples, args.sweep_kind)
    if selection_manifest is not None:
        validate_release_source_bindings(
            root,
            samples,
            selection_manifest,
            args.selection_source_schema,
        )
        selection_binding["verified_render_sample_count"] = len(samples)
    if args.profiles:
        requested_profiles = {str(value) for value in args.profiles}
        selected_samples = []
        selected_profiles = set()
        for sample in samples:
            metadata_path = project_path(root, str(sample["metadata_path"]))
            metadata = load_json(metadata_path)
            profile_id = str(metadata["visualization"]["environment"]["profile_id"])
            if profile_id in requested_profiles:
                selected_samples.append(sample)
                selected_profiles.add(profile_id)
        missing_profiles = requested_profiles - selected_profiles
        if missing_profiles:
            raise ValueError(
                f"bound manifest lacks requested profiles: {sorted(missing_profiles)}"
            )
        samples = selected_samples
    if args.limit is not None:
        samples = samples[: args.limit]
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one id")
    output_root_value = Path(str(manifest["output_root"]))
    output_root = (
        output_root_value
        if output_root_value.is_absolute()
        else root / output_root_value
    ).resolve()
    if (root / "outputs").resolve() not in output_root.parents:
        raise ValueError(f"render output must be below root/outputs: {output_root}")
    result_manifest_path = project_path(
        root,
        args.result_manifest or output_root / result_manifest_name(args.sweep_kind),
    )
    if output_root not in result_manifest_path.parents:
        raise ValueError("render result manifest must remain below the output root")
    selector = build_egl_device_selector(root)
    selector_path = root / str(selector["binary_path"])
    script = PROJECT_ROOT / "tools/rendering/render_pybullet_rigid.py"
    if not script.is_file():
        raise FileNotFoundError(script)
    started = time.perf_counter()
    declared_blender = Path(args.blender)
    if not declared_blender.is_absolute():
        declared_blender = root / declared_blender
    declared_blender.absolute().relative_to(root)
    blender = declared_blender.resolve()
    if not blender.is_file():
        raise FileNotFoundError(blender)
    jobs = [
        (
            str(root),
            str(blender),
            str(script),
            str(output_root),
            sample,
            gpus[index % len(gpus)],
            bool(args.first_frame_only),
            str(selector_path),
            bool(args.resume),
        )
        for index, sample in enumerate(samples)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(lambda values: worker(*values), jobs))
    records.sort(key=lambda record: record["scene_id"])
    failures = [record for record in records if not record["ok"]]
    summary = {
        "schema_version": "physweep_pybullet_render_manifest_v1",
        "source_manifest": str(manifest_path.relative_to(root)),
        "source_manifest_sha256": sha256(manifest_path),
        "selection_manifest": selection_binding,
        "render_scope": "first_frame_only" if args.first_frame_only else "full_animation",
        "sample_count": len(records),
        "success_count": len(records) - len(failures),
        "failure_count": len(failures),
        "reused_count": sum(record.get("reused", False) for record in records),
        "wall_time_s": round(time.perf_counter() - started, 6),
        "egl_device_selector": selector,
        "records": records,
    }
    write_json(result_manifest_path, summary)
    print(f"render manifest: {result_manifest_path}")
    print(
        f"success={summary['success_count']} failures={summary['failure_count']} "
        f"wall_time_s={summary['wall_time_s']:.3f}"
    )
    if failures:
        print(json.dumps(failures[:5], indent=2, ensure_ascii=True))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
