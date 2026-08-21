#!/usr/bin/env python3
"""Collect rendered branches into one portable, provenance-complete batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def project_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def load_render_records(
    path: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest = load_json(path)
    records: dict[str, dict[str, Any]] = {}
    for record in manifest["records"]:
        if not record["ok"] or not record["render_record"]:
            raise ValueError(f"failed render remains in manifest: {record['scene_id']}")
        if not record.get("egl_device_verified"):
            raise ValueError(f"unverified EGL device: {record['scene_id']}")
        records[str(record["scene_id"])] = record
    if len(records) != int(manifest["sample_count"]):
        raise ValueError(f"duplicate or missing render records: {path}")
    return manifest, records


def verified_file(root: Path, path_value: str, declared_hash: str) -> Path:
    path = resolve_path(root, path_value)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(path)
    actual_hash = sha256(path)
    if actual_hash != declared_hash:
        raise ValueError(f"declared hash mismatch: {path}")
    return path


def normalized_render_metadata(
    metadata_path: Path,
    destination: Path,
    inspection_directory: Path,
) -> dict[str, Any]:
    metadata = load_json(metadata_path)
    visualization = metadata.get("visualization")
    if not isinstance(visualization, dict):
        raise ValueError(f"bound metadata lacks visualization: {metadata_path}")
    render = visualization.get("render")
    if not isinstance(render, dict):
        raise ValueError(f"bound metadata lacks render request: {metadata_path}")
    render["video_path"] = str(destination)
    render["inspection_frame_dir"] = str(inspection_directory)
    return metadata


def compact_render_provenance(rendered: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "schema_version",
        "implementation",
        "trajectory_path",
        "trajectory_sha256",
        "support_binding_sha256",
        "camera",
        "lighting_adaptation",
        "blender_version",
        "render_engine",
        "video_encoding",
        "render_scope",
        "render_output_overridden",
        "wall_time_s",
    )
    return {field: rendered[field] for field in fields if field in rendered}


def collect(
    *,
    root: Path,
    manifest_path: Path,
    generic_render_manifest_path: Path,
    asset_render_manifest_path: Path,
    billiards_render_manifest_path: Path,
    output: Path,
    overwrite: bool,
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "physweep_one_object_decoupled_manifest_v3":
        raise ValueError("collector requires a v3 decoupled staged manifest")
    source_manifest_value = manifest.get("source_manifest")
    if not source_manifest_value:
        raise ValueError(
            "collector requires staged_manifest.json from "
            "prepare_formal_render_manifests.py"
        )
    canonical_source_manifest = resolve_path(root, source_manifest_value)
    if not canonical_source_manifest.is_file():
        raise FileNotFoundError(canonical_source_manifest)

    branch_paths = {
        "generic_pybullet": generic_render_manifest_path.resolve(),
        "asset_proxy": asset_render_manifest_path.resolve(),
        "billiards": billiards_render_manifest_path.resolve(),
    }
    branch_manifests: dict[str, dict[str, Any]] = {}
    branches: dict[str, dict[str, dict[str, Any]]] = {}
    for pipeline, path in branch_paths.items():
        branch_manifests[pipeline], branches[pipeline] = load_render_records(path)

    selectors = [
        branch.get("egl_device_selector")
        for branch in branch_manifests.values()
        if branch.get("egl_device_selector") is not None
    ]
    if selectors and any(selector != selectors[0] for selector in selectors[1:]):
        raise ValueError("render branches used different EGL selector builds")

    output = output.resolve()
    if output.exists():
        if not overwrite:
            raise SystemExit(f"output exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    videos = output / "videos"
    render_metadata_root = output / "provenance" / "generic_render_metadata"
    videos.mkdir(parents=True, exist_ok=True)

    collected = []
    for outer in manifest["records"]:
        pipeline = str(outer["pipeline"])
        if pipeline == "generic_pybullet":
            source_metadata = load_json(root / str(outer["metadata_path"]))
            child_scene_id = str(source_metadata["scene_id"])
            suffix = child_scene_id
        elif pipeline == "asset_proxy":
            child_scene_id = str(outer["child_scene_id"])
            suffix = child_scene_id
        elif pipeline == "billiards":
            child_scene_id = str(outer["scene_id"])
            suffix = str(outer["profile"])
        else:
            raise ValueError(f"unknown render pipeline: {pipeline}")

        envelope = branches[pipeline][child_scene_id]
        rendered = envelope["render_record"]
        if rendered["render_engine"] != "BLENDER_EEVEE":
            raise ValueError(
                f"non-Eevee render in base batch: {outer['scene_id']} "
                f"{rendered['render_engine']}"
            )

        metadata_path = verified_file(
            root,
            str(outer["metadata_path"]),
            str(outer["metadata_sha256"]),
        )
        source_video = verified_file(
            root,
            str(rendered["video_path"]),
            str(rendered["video_sha256"]),
        )
        destination = videos / f"{outer['scene_id']}__{suffix}.mp4"
        shutil.copy2(source_video, destination)
        destination_hash = sha256(destination)
        if destination_hash != rendered["video_sha256"]:
            raise ValueError(f"copied video hash mismatch: {outer['scene_id']}")

        effective_metadata_path = verified_file(
            root,
            str(rendered["metadata_path"]),
            str(rendered["metadata_sha256"]),
        )
        if rendered.get("trajectory_path") or rendered.get("trajectory_sha256"):
            if not rendered.get("trajectory_path") or not rendered.get(
                "trajectory_sha256"
            ):
                raise ValueError(f"incomplete trajectory provenance: {child_scene_id}")
            verified_file(
                root,
                str(rendered["trajectory_path"]),
                str(rendered["trajectory_sha256"]),
            )
        effective_metadata_source_hash = str(rendered["metadata_sha256"])
        if pipeline == "generic_pybullet":
            retained_metadata_path = render_metadata_root / f"{child_scene_id}.json"
            retained_metadata = normalized_render_metadata(
                effective_metadata_path,
                destination,
                output / "provenance" / "inspection_frames" / child_scene_id,
            )
            write_json(retained_metadata_path, retained_metadata)
            effective_metadata_value = project_path(root, retained_metadata_path)
            effective_metadata_hash = sha256(retained_metadata_path)
        else:
            retained_metadata_path = metadata_path
            effective_metadata_value = project_path(root, retained_metadata_path)
            effective_metadata_hash = sha256(retained_metadata_path)

        collected.append(
            {
                "index": int(outer["index"]),
                "scene_id": str(outer["scene_id"]),
                "child_scene_id": child_scene_id,
                "motion_intent": str(outer["motion_intent"]),
                "environment_id": str(outer["environment_id"]),
                "profile": str(outer["profile"]),
                "pipeline": pipeline,
                "dynamic_asset_id": outer.get("dynamic_asset_id"),
                "support_asset_id": outer.get("support_asset_id"),
                "static_prop_asset_id": outer.get("static_prop_asset_id"),
                "metadata_path": project_path(root, metadata_path),
                "metadata_sha256": str(outer["metadata_sha256"]),
                "effective_render_metadata_path": effective_metadata_value,
                "effective_render_metadata_sha256": effective_metadata_hash,
                "effective_render_metadata_source_sha256": (
                    effective_metadata_source_hash
                ),
                "render_provenance": compact_render_provenance(rendered),
                "render_worker": {
                    "gpu": int(envelope["gpu"]),
                    "egl_device_verified": True,
                },
                "render_engine": rendered["render_engine"],
                "video_encoding": rendered.get("video_encoding"),
                "video_path": project_path(root, destination),
                "sha256": destination_hash,
            }
        )

    collected.sort(key=lambda record: record["index"])
    if len(collected) != int(manifest["sample_count"]):
        raise ValueError("collected render count does not match source manifest")

    result = {
        "schema_version": "physweep_decoupled_collected_renders_v4",
        "dataset_id": manifest["dataset_id"],
        "source_manifest": project_path(root, canonical_source_manifest),
        "source_manifest_sha256": sha256(canonical_source_manifest),
        "collection_input_sha256": sha256(manifest_path),
        "sample_count": len(collected),
        "render_engine": "BLENDER_EEVEE",
        "production_spec": manifest.get("production_spec"),
        "sampling_matrix": manifest.get("sampling_matrix"),
        "dependencies": manifest.get("dependencies"),
        "implementation": manifest.get("implementation"),
        "motion_counts": manifest.get("motion_counts"),
        "environment_counts": manifest.get("environment_counts"),
        "profile_counts": manifest.get("profile_counts"),
        "render_runtime": {
            "egl_device_selector": selectors[0] if selectors else None,
            "branch_manifest_sha256": {
                pipeline: sha256(path) for pipeline, path in branch_paths.items()
            },
        },
        "records": collected,
    }
    write_json(output / "collected_render_manifest.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="staged_manifest.json produced by prepare_formal_render_manifests.py",
    )
    parser.add_argument("--generic-render-manifest", type=Path, required=True)
    parser.add_argument("--asset-render-manifest", type=Path, required=True)
    parser.add_argument("--billiards-render-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = collect(
        root=args.root,
        manifest_path=args.manifest,
        generic_render_manifest_path=args.generic_render_manifest,
        asset_render_manifest_path=args.asset_render_manifest,
        billiards_render_manifest_path=args.billiards_render_manifest,
        output=args.output,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "dataset_id": result["dataset_id"],
                "sample_count": result["sample_count"],
                "render_engine": result["render_engine"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
