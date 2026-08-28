#!/usr/bin/env python3
"""Render an asset-only proxy manifest with parallel Blender workers."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json_atomic as write_json
from tools.core.paths import resolve_project_path as project_path
from tools.core.paths import safe_scene_id
from tools.rendering.blender_worker_environment import (
    build_egl_device_selector,
    isolated_blender_environment,
)
from tools.physics.specialized_backend_registry import specialized_by_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_CONTRACT = "physweep_specialized_render_evidence_v2"
MASK_RESULT_SCHEMA = "physweep_specialized_mask_render_manifest_v1"
MASK_REQUIRED_SCHEMAS = frozenset(
    {
        "physweep_asset_proxy_scene_v3",
        "physweep_billiards_scene_v4",
        "physweep_passive_pinball_scene_v1",
        "physweep_marble_run_scene_v1",
    }
)
RENDER_RECORD_IMPLEMENTATION_SCHEMAS = frozenset(
    {
        "physweep_asset_proxy_scene_v3",
        "physweep_billiards_scene_v4",
    }
)


def renderer_table(root: Path) -> dict[str, tuple[str, str, str, str]]:
    return {
        record["renderer_id"]: (
            record["renderer_script"],
            record["render_manifest_schema"],
            record["render_manifest_name"],
            record["source_schema_version"],
        )
        for record in specialized_by_pipeline(root).values()
    }


def output_path(root: Path, value: str | Path) -> Path:
    path = project_path(root, value)
    if (root / "outputs").resolve() not in path.parents:
        raise ValueError(f"render output must be below root/outputs: {path}")
    return path


def result_manifest_path(
    root: Path,
    output: Path,
    value: str | Path | None,
    default_name: str,
) -> Path:
    path = output_path(root, value) if value is not None else output / default_name
    if output not in path.parents:
        raise ValueError("render result manifest must remain below its output root")
    return path


def render_samples_are_reusable(
    metadata: dict[str, Any], render_record: dict[str, Any]
) -> bool:
    schema = str(metadata.get("schema_version", ""))
    required = schema in {
        "physweep_passive_pinball_scene_v1",
        "physweep_marble_run_scene_v1",
    } or metadata.get("render", {}).get("evidence_contract") == EVIDENCE_CONTRACT
    if not required:
        return True
    declared = metadata.get("render", {}).get("samples")
    return declared is not None and int(
        render_record.get("render_samples", -1)
    ) == int(declared)


def expected_instance_mask_directory(
    root: Path,
    output: Path,
    source_record: Mapping[str, Any],
    mask_contract: Mapping[str, Any],
) -> Path:
    """Resolve the mask destination used by the current render invocation."""
    if isinstance(source_record.get("render_output"), dict):
        path = output / "masks" / str(source_record["scene_id"])
    else:
        declared = mask_contract.get("path")
        if not isinstance(declared, str) or not declared:
            raise ValueError("instance-mask contract has no output path")
        path = project_path(root, declared)
    path = path.resolve()
    if root.resolve() not in path.parents:
        raise ValueError("instance-mask output must remain below the project root")
    return path


def instance_masks_are_reusable(
    root: Path,
    render_record: dict[str, Any],
    frame_count: int,
    *,
    required: bool = False,
    expected_objects: Mapping[str, Any] | None = None,
    expected_directory: Path | None = None,
    expected_render_samples: int | None = None,
) -> bool:
    binding = render_record.get("instance_mask_output")
    if binding is None:
        return not required
    if not isinstance(binding, dict):
        return False
    manifest_path = project_path(root, binding["manifest_path"])
    if (
        root.resolve() not in manifest_path.parents
        or not manifest_path.is_file()
        or sha256(manifest_path) != str(binding["manifest_sha256"])
    ):
        return False
    manifest = load_json(manifest_path)
    if str(manifest.get("scene_id")) != str(render_record.get("scene_id")):
        return False
    if (
        expected_directory is not None
        and manifest_path.parent != expected_directory.resolve()
    ):
        return False
    schema = str(manifest.get("schema_version", ""))
    if schema == "physweep_instance_mask_manifest_v1":
        object_id = str(manifest.get("object_id", ""))
        binding_objects = binding.get("objects", {})
        manifest_objects = {
            object_id: {
                "frame_count": manifest.get("frame_count"),
                "records": manifest.get("records"),
            }
        }
        if not isinstance(binding_objects, dict) or set(binding_objects) != {object_id}:
            return False
    elif schema == "physweep_instance_mask_manifest_v2":
        manifest_objects = manifest.get("objects")
        if (
            expected_objects is None
            or expected_directory is None
            or expected_render_samples is None
            or int(binding.get("render_samples", -1)) != expected_render_samples
            or not isinstance(manifest_objects, dict)
            or set(manifest_objects) != set(expected_objects)
            or manifest_path.parent != expected_directory.resolve()
        ):
            return False
    else:
        return False
    if (
        not isinstance(manifest_objects, dict)
        or int(manifest.get("frame_count", -1)) != frame_count
    ):
        return False
    for object_id, value in manifest_objects.items():
        record = value if isinstance(value, dict) else {}
        records = record.get("records")
        if not isinstance(records, list) or len(records) != frame_count:
            return False
        directory = (manifest_path.parent / str(object_id)).resolve()
        if schema == "physweep_instance_mask_manifest_v1" and (
            project_path(root, binding_objects[object_id]["directory"]) != directory
            or int(record.get("frame_count", -1)) != frame_count
        ):
            return False
        if schema == "physweep_instance_mask_manifest_v2" and expected_objects is not None:
            expected = expected_objects[object_id]
            if not isinstance(expected, dict) or int(
                expected.get("instance_id", -3)
            ) != int(record.get("instance_id", -2)):
                return False
        if schema == "physweep_instance_mask_manifest_v2":
            report = record.get("validation", {})
            initial = float(report.get("initial_occupancy_fraction", -1.0))
            soft_edge = float(report.get("initial_soft_edge_fraction", -1.0))
            if (
                not 0.0 < initial < 1.0
                or not 0.0 < soft_edge <= 1.0
            ):
                return False
        if any(not isinstance(item, dict) for item in records):
            return False
        filenames = [str(item.get("filename", "")) for item in records]
        if len(filenames) != len(set(filenames)) or any(
            not filename or Path(filename).name != filename for filename in filenames
        ):
            return False
        if schema == "physweep_instance_mask_manifest_v2" and filenames != [
            f"frame_{frame:04d}.png" for frame in range(1, frame_count + 1)
        ]:
            return False
        if not all(
            isinstance(item, dict)
            and isinstance(item.get("sha256"), str)
            and (directory / filename).is_file()
            and sha256(directory / filename) == str(item["sha256"])
            for filename, item in zip(filenames, records)
        ):
            return False
    return True


def implementation_is_reusable(
    root: Path,
    metadata: dict[str, Any],
    script: Path,
) -> bool:
    if metadata.get("render", {}).get("evidence_contract") != EVIDENCE_CONTRACT:
        return True
    declared = metadata.get("implementation")
    expected_paths = {
        "renderer": script.resolve(),
        "render_evidence": (root / "tools/rendering/specialized_render_evidence.py").resolve(),
    }
    if not isinstance(declared, dict):
        return False
    for label, expected_path in expected_paths.items():
        declared_binding = declared.get(label)
        if not isinstance(declared_binding, dict):
            return False
        expected_hash = sha256(expected_path)
        if (
            project_path(root, str(declared_binding.get("path", "")))
            != expected_path
            or str(declared_binding.get("sha256")) != expected_hash
        ):
            return False
    return True


def render_record_implementation_is_reusable(
    root: Path,
    render_record: Mapping[str, Any],
    script: Path,
) -> bool:
    implementation = render_record.get("implementation")
    expected = {
        "renderer": script.resolve(),
        "render_evidence": (script.parent / "specialized_render_evidence.py").resolve(),
    }
    if not isinstance(implementation, dict):
        return False
    return all(
        isinstance(implementation.get(name), dict)
        and project_path(root, str(implementation[name].get("path", ""))) == path
        and str(implementation[name].get("sha256", "")) == sha256(path)
        for name, path in expected.items()
    )


def mask_record_is_reusable(
    root: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    render_record: dict[str, Any],
    mask_directory: Path,
    script: Path,
) -> bool:
    identity = metadata.get("object_identity")
    if not isinstance(identity, dict):
        return False
    identity_records = identity.get("objects")
    mask_contract = identity.get("instance_masks")
    if not isinstance(identity_records, list) or not isinstance(mask_contract, dict):
        return False
    object_ids = [
        str(record.get("object_id"))
        for record in identity_records
        if isinstance(record, dict) and record.get("object_id")
    ]
    expected_objects = mask_contract.get("objects")
    if (
        not object_ids
        or len(object_ids) != len(identity_records)
        or len(object_ids) != len(set(object_ids))
        or not isinstance(expected_objects, dict)
        or set(expected_objects) != set(object_ids)
    ):
        return False
    frame_count = int(
        metadata.get("simulation", {}).get("time", {}).get(
            "frame_count", metadata["physics"].get("frame_count")
        )
    )
    expected_samples = max(8, min(int(metadata["render"]["samples"]), 16))
    return (
        render_record.get("schema_version")
        == "physweep_specialized_mask_render_record_v1"
        and str(render_record.get("scene_id")) == str(metadata["scene_id"])
        and project_path(root, str(render_record.get("metadata_path", "")))
        == metadata_path
        and str(render_record.get("metadata_sha256", "")) == sha256(metadata_path)
        and render_record.get("render_scope") == "instance_masks_only"
        and render_record.get("mask_resolution")
        == [int(value) for value in metadata["render"]["resolution"]]
        and render_record.get("egl_device_verified") is True
        and render_record_implementation_is_reusable(root, render_record, script)
        and instance_masks_are_reusable(
            root,
            render_record,
            frame_count,
            required=True,
            expected_objects=expected_objects,
            expected_directory=mask_directory,
            expected_render_samples=expected_samples,
        )
    )


def reusable_render_record(
    root: Path,
    output: Path,
    source_record: dict[str, Any],
    metadata_path: Path,
    metadata: dict[str, Any],
    frame_dir: Path,
    video_path: Path,
    render_record: dict[str, Any],
    gpu: int,
    script: Path | None = None,
) -> bool:
    inspection_frames = [
        project_path(root, value)
        for value in render_record.get("inspection_frames", [])
    ]
    source_metadata = metadata.get("source_metadata")
    expected_metadata_sha256 = sha256(metadata_path)
    if isinstance(source_metadata, dict):
        source_path = project_path(root, str(source_metadata["path"]))
        source_path.relative_to(root)
        if sha256(source_path) != str(source_metadata["sha256"]):
            raise ValueError(f"source metadata hash mismatch: {source_path}")
    log_path = output / "logs" / f"{source_record['scene_id']}.log"
    egl_marker = f"PhysSweep EGL selector: CUDA device {gpu} "
    egl_verified = bool(render_record.get("egl_device_verified")) or (
        log_path.is_file()
        and egl_marker in log_path.read_text(encoding="utf-8", errors="replace")
    )
    frame_count = int(
        metadata.get("simulation", {}).get("time", {}).get(
            "frame_count", metadata["physics"].get("frame_count")
        )
    )
    expected_frames = [
        frame_dir / f"frame_{frame:04d}.png"
        for frame in (1, (frame_count + 1) // 2, frame_count)
    ]
    strict_evidence = (
        metadata.get("render", {}).get("evidence_contract") == EVIDENCE_CONTRACT
    )
    schema = str(metadata.get("schema_version", ""))
    renderer_requires_masks = schema in MASK_REQUIRED_SCHEMAS
    render_record_binds_implementation = (
        schema in RENDER_RECORD_IMPLEMENTATION_SCHEMAS
    )
    require_instance_masks = strict_evidence or renderer_requires_masks
    expected_mask_objects = None
    expected_mask_directory = None
    expected_mask_samples = None
    if require_instance_masks:
        identity = metadata.get("object_identity")
        if not isinstance(identity, dict):
            return False
        identity_records = identity.get("objects")
        mask_contract = identity.get("instance_masks")
        if not isinstance(identity_records, list) or not isinstance(mask_contract, dict):
            return False
        identity_ids = [
            str(record.get("object_id"))
            for record in identity_records
            if isinstance(record, dict) and record.get("object_id")
        ]
        expected_mask_objects = mask_contract.get("objects")
        if (
            not identity_ids
            or len(identity_ids) != len(identity_records)
            or len(identity_ids) != len(set(identity_ids))
            or not isinstance(expected_mask_objects, dict)
            or set(expected_mask_objects) != set(identity_ids)
        ):
            return False
        if (
            mask_contract.get("encoding")
            != "rgba_alpha_antialiased_silhouette_mask"
            or mask_contract.get("path_layout") != "object_id_subdirectories"
            or mask_contract.get("filename_pattern") != "frame_{frame:04d}.png"
        ):
            return False
        try:
            expected_mask_directory = expected_instance_mask_directory(
                root,
                output,
                source_record,
                mask_contract,
            )
        except ValueError:
            return False
        expected_mask_samples = max(
            8, min(int(metadata["render"]["samples"]), 16)
        )
    renderer_script = (script or Path(__file__)).resolve()
    return (
        str(render_record.get("scene_id")) == str(source_record["scene_id"])
        and project_path(root, str(render_record.get("metadata_path")))
        == metadata_path
        and str(render_record.get("metadata_sha256")) == expected_metadata_sha256
        and project_path(root, str(render_record.get("video_path"))) == video_path
        and str(render_record.get("video_sha256")) == sha256(video_path)
        and video_path.stat().st_size > 0
        and inspection_frames == expected_frames
        and all(
            frame.parent == frame_dir
            and frame.is_file()
            and frame.stat().st_size > 0
            for frame in inspection_frames
        )
        and render_samples_are_reusable(metadata, render_record)
        and instance_masks_are_reusable(
            root,
            render_record,
            frame_count,
            required=require_instance_masks,
            expected_objects=expected_mask_objects,
            expected_directory=expected_mask_directory,
            expected_render_samples=expected_mask_samples,
        )
        and implementation_is_reusable(
            root,
            metadata,
            renderer_script,
        )
        and (
            not render_record_binds_implementation
            or render_record_implementation_is_reusable(
                root,
                render_record,
                renderer_script,
            )
        )
        and egl_verified
    )


def render_source_records(
    root: Path,
    manifest: dict[str, Any],
    renderer: str,
    renderers: dict[str, tuple[str, str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    if renderers is None:
        renderers = renderer_table(root)
    if renderer not in renderers:
        raise ValueError(f"unknown specialized renderer: {renderer}")
    source_records = manifest.get("records")
    if source_records is None and renderer == "billiards":
        source_records = [
            {"metadata_path": str(value)}
            for value in manifest["billiards_metadata_paths"]
        ]
    if source_records is None:
        raise ValueError("render manifest has no records")
    expected_schema = renderers[renderer][3]
    records = [dict(record) for record in source_records]
    for record in records:
        metadata_path = project_path(root, record["metadata_path"])
        metadata_path.relative_to(root)
        metadata = load_json(metadata_path)
        if str(metadata["schema_version"]) != expected_schema:
            raise ValueError("render manifest contains the wrong scene schema")
        metadata_scene_id = safe_scene_id(metadata["scene_id"])
        if "scene_id" not in record:
            record["scene_id"] = metadata_scene_id
        elif safe_scene_id(record["scene_id"]) != metadata_scene_id:
            raise ValueError("render manifest scene id does not match metadata")
    return records


def worker(
    root: Path,
    blender: Path,
    script: Path,
    record: dict[str, Any],
    output: Path,
    gpu: int,
    selector_path: Path,
    resume: bool,
) -> dict[str, Any]:
    metadata_path = project_path(root, record["metadata_path"])
    metadata_path.relative_to(root)
    if record.get("metadata_sha256") and sha256(metadata_path) != str(
        record["metadata_sha256"]
    ):
        raise ValueError(f"render metadata hash mismatch: {metadata_path}")
    metadata = load_json(metadata_path)
    render_output = record.get("render_output", metadata["render"])
    frame_dir = output_path(root, render_output["inspection_frame_dir"])
    video_path = output_path(root, render_output["video_path"])
    mask_directory = output / "masks" / str(record["scene_id"])
    if (
        output not in frame_dir.parents
        or output not in video_path.parents
        or output not in mask_directory.parents
    ):
        raise ValueError("render paths must remain below the manifest output root")
    render_record_path = frame_dir / "render_record.json"
    if resume and render_record_path.is_file() and video_path.is_file():
        render_record = load_json(render_record_path)
        if reusable_render_record(
            root,
            output,
            record,
            metadata_path,
            metadata,
            frame_dir,
            video_path,
            render_record,
            gpu,
            script,
        ):
            if not render_record.get("egl_device_verified"):
                render_record["egl_device_verified"] = True
                write_json(render_record_path, render_record)
            return {
                "scene_id": record["scene_id"],
                "ok": True,
                "returncode": 0,
                "gpu": None,
                "egl_device_verified": True,
                "wall_time_s": 0.0,
                "log_path": None,
                "render_record": render_record,
                "reused": True,
            }
    render_record_path.unlink(missing_ok=True)
    video_path.unlink(missing_ok=True)
    command = [
        str(blender),
        "-b",
        "--python-exit-code",
        "1",
        "--python",
        str(script),
        "--",
        "--metadata",
        str(metadata_path),
        "--video-path",
        str(video_path),
        "--inspection-frame-dir",
        str(frame_dir),
    ]
    command.extend(("--instance-mask-dir", str(mask_directory)))
    started = time.perf_counter()
    with isolated_blender_environment(gpu, selector_path) as (
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
    log_path = output / "logs" / f"{record['scene_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    render_record = (
        load_json(render_record_path)
        if completed.returncode == 0 and render_record_path.is_file()
        else None
    )
    selector_verified = selector_marker in completed.stdout
    if render_record is not None:
        render_record["egl_device_verified"] = selector_verified
        write_json(render_record_path, render_record)
    ok = (
        completed.returncode == 0
        and render_record is not None
        and video_path.is_file()
        and selector_verified
        and reusable_render_record(
            root,
            output,
            record,
            metadata_path,
            metadata,
            frame_dir,
            video_path,
            render_record,
            gpu,
            script,
        )
    )
    return {
        "scene_id": record["scene_id"],
        "ok": ok,
        "returncode": completed.returncode,
        "gpu": gpu,
        "egl_device_verified": selector_verified,
        "wall_time_s": round(time.perf_counter() - started, 6),
        "log_path": str(log_path),
        "render_record": render_record,
        "reused": False,
    }


def mask_worker(
    root: Path,
    blender: Path,
    script: Path,
    record: dict[str, Any],
    output: Path,
    gpu: int,
    selector_path: Path,
    resume: bool,
) -> dict[str, Any]:
    metadata_path = project_path(root, record["metadata_path"])
    metadata_path.relative_to(root)
    if record.get("metadata_sha256") and sha256(metadata_path) != str(
        record["metadata_sha256"]
    ):
        raise ValueError(f"render metadata hash mismatch: {metadata_path}")
    metadata = load_json(metadata_path)
    scene_id = str(record["scene_id"])
    frame_dir = output / "records" / scene_id
    mask_directory = output / "masks" / scene_id
    render_record_path = frame_dir / "render_record.json"
    log_path = output / "logs" / f"{scene_id}.log"
    if resume and render_record_path.is_file():
        render_record = load_json(render_record_path)
        if mask_record_is_reusable(
            root,
            metadata_path,
            metadata,
            render_record,
            mask_directory,
            script,
        ):
            return {
                "scene_id": scene_id,
                "ok": True,
                "returncode": 0,
                "gpu": None,
                "egl_device_verified": bool(render_record.get("egl_device_verified")),
                "wall_time_s": 0.0,
                "log_path": None,
                "render_record": render_record,
                "reused": True,
            }
    render_record_path.unlink(missing_ok=True)
    command = [
        str(blender),
        "-b",
        "--python-exit-code",
        "1",
        "--python",
        str(script),
        "--",
        "--metadata",
        str(metadata_path),
        "--root",
        str(root),
        "--inspection-frame-dir",
        str(frame_dir),
        "--mask-only",
        "--instance-mask-dir",
        str(mask_directory),
    ]
    started = time.perf_counter()
    with isolated_blender_environment(gpu, selector_path) as (
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
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    render_record = (
        load_json(render_record_path)
        if completed.returncode == 0 and render_record_path.is_file()
        else None
    )
    selector_verified = selector_marker in completed.stdout
    if render_record is not None:
        render_record["egl_device_verified"] = selector_verified
        write_json(render_record_path, render_record)
    ok = (
        completed.returncode == 0
        and render_record is not None
        and selector_verified
        and mask_record_is_reusable(
            root,
            metadata_path,
            metadata,
            render_record,
            mask_directory,
            script,
        )
    )
    return {
        "scene_id": scene_id,
        "ok": ok,
        "returncode": completed.returncode,
        "gpu": gpu,
        "egl_device_verified": selector_verified,
        "wall_time_s": round(time.perf_counter() - started, 6),
        "log_path": str(log_path),
        "render_record": render_record,
        "reused": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--blender", type=Path, default=PROJECT_ROOT / "runtime/blender-3.4.0-linux-x64/blender")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--renderer", default="asset")
    parser.add_argument("--mask-only", action="store_true")
    parser.add_argument(
        "--mask-output-root",
        type=Path,
        help="Dedicated output root for mask-only records and masks.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a video only when its scene id and metadata/video hashes verify.",
    )
    parser.add_argument(
        "--result-manifest",
        type=Path,
        help="Explicit summary path; defaults to OUTPUT_ROOT/render_manifest.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    root = args.root.resolve()
    manifest_path = project_path(root, args.manifest)
    manifest_path.relative_to(root)
    manifest = load_json(manifest_path)
    if args.mask_only:
        if args.mask_output_root is None:
            raise ValueError("--mask-only requires --mask-output-root")
        output = output_path(root, args.mask_output_root)
    else:
        output = output_path(root, manifest.get("output_root", manifest_path.parent))
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one id")
    code_root = Path(__file__).resolve().parents[2]
    renderers = renderer_table(code_root if args.mask_only else root)
    if args.renderer not in renderers:
        raise ValueError(f"unknown specialized renderer: {args.renderer}")
    if args.mask_only and args.renderer not in {"asset", "billiards"}:
        raise ValueError("mask-only backfill supports asset and billiards renderers")
    script_name, schema_version, result_name = renderers[args.renderer][:3]
    script_root = code_root if args.mask_only else root
    script = (script_root / script_name).resolve()
    script.relative_to(script_root)
    if not script.is_file():
        raise FileNotFoundError(script)
    declared_blender = Path(args.blender)
    if not declared_blender.is_absolute():
        declared_blender = root / declared_blender
    declared_blender.absolute().relative_to(root)
    blender = declared_blender.resolve()
    if not blender.is_file():
        raise FileNotFoundError(blender)
    selector = build_egl_device_selector(root)
    selector_path = root / str(selector["binary_path"])
    started = time.perf_counter()
    source_records = render_source_records(
        root, manifest, args.renderer, renderers
    )
    jobs = [
        (
            root,
            blender,
            script,
            record,
            output,
            gpus[index % len(gpus)],
            selector_path,
            args.resume,
        )
        for index, record in enumerate(source_records)
    ]
    selected_worker = mask_worker if args.mask_only else worker
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(lambda values: selected_worker(*values), jobs))
    failures = [record for record in records if not record["ok"]]
    summary = {
        "schema_version": MASK_RESULT_SCHEMA if args.mask_only else schema_version,
        "source_manifest": str(manifest_path.relative_to(root)),
        "source_manifest_sha256": sha256(manifest_path),
        "sample_count": len(records),
        "success_count": len(records) - len(failures),
        "failure_count": len(failures),
        "reused_count": sum(record.get("reused", False) for record in records),
        "wall_time_s": round(time.perf_counter() - started, 6),
        "egl_device_selector": selector,
        "records": records,
    }
    result_manifest = result_manifest_path(
        root,
        output,
        args.result_manifest,
        "mask_render_manifest.json" if args.mask_only else result_name,
    )
    write_json(result_manifest, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))
    if failures:
        print(json.dumps(failures[:5], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
