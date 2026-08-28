#!/usr/bin/env python3
"""Freeze each asset sweep group to the camera solved from its base sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.core.json_io import read_json as load_json
from tools.core.paths import resolve_project_path as project_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    contents = json_bytes(value)
    if path.is_file() and path.read_bytes() == contents:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(contents)
    temporary.replace(path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def indexed_records(manifest: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    records = {str(record["scene_id"]): record for record in manifest["records"]}
    if len(records) != len(manifest["records"]):
        raise ValueError(f"{label} contains duplicate scene ids")
    if int(manifest["sample_count"]) != len(records):
        raise ValueError(f"{label} sample count is inconsistent")
    return records


def source_metadata_hash(root: Path, metadata: dict[str, Any], scene_id: str) -> str:
    if str(metadata.get("schema_version")) != "physweep_asset_proxy_scene_v3":
        raise ValueError(f"invalid asset metadata schema: {scene_id}")
    source_metadata = metadata.get("source_metadata")
    if not isinstance(source_metadata, dict):
        raise ValueError(f"asset metadata lacks source provenance: {scene_id}")
    source_path = project_path(root, source_metadata["path"])
    source_path.relative_to(root / "datasets")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_hash != str(source_metadata["sha256"]):
        raise ValueError(f"asset source metadata hash mismatch: {scene_id}")
    return source_hash


def rendered_base_cameras(
    root: Path, base_manifest: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    cameras: dict[str, dict[str, Any]] = {}
    for scene_id, record in indexed_records(
        base_manifest, "asset base render manifest"
    ).items():
        metadata_path = project_path(root, record["metadata_path"])
        metadata_path.relative_to(root)
        metadata = load_json(metadata_path)
        if (
            str(metadata.get("schema_version")) != "physweep_asset_proxy_scene_v3"
            or str(metadata.get("scene_id")) != scene_id
        ):
            raise ValueError(f"invalid rendered asset base metadata: {scene_id}")
        frame_dir = project_path(
            root, record["render_output"]["inspection_frame_dir"]
        )
        render_record_path = frame_dir / "render_record.json"
        render_record = load_json(render_record_path)
        metadata_hash = sha256_bytes(metadata_path.read_bytes())
        if (
            str(render_record.get("schema_version"))
            != "physweep_asset_proxy_render_record_v1"
            or str(render_record.get("scene_id")) != scene_id
            or str(render_record.get("metadata_sha256")) != metadata_hash
        ):
            raise ValueError(f"base render provenance mismatch: {scene_id}")
        camera = render_record.get("camera")
        if (
            not isinstance(camera, dict)
            or len(camera.get("position_m", [])) != 3
            or len(camera.get("target_m", [])) != 3
            or float(camera.get("focal_length_mm", 0.0)) <= 0.0
        ):
            raise ValueError(f"invalid rendered base camera: {scene_id}")
        cameras[scene_id] = {
            "solver_version": "frozen_from_one_factor_base_v1",
            "source_base_scene_id": scene_id,
            "position_m": camera["position_m"],
            "target_m": camera["target_m"],
            "focal_length_mm": camera["focal_length_mm"],
            "focus_span_m": camera.get("focus_span_m"),
        }
    return cameras


def freeze_cameras(
    root: Path, manifest_path: Path, base_manifest_path: Path
) -> dict[str, int]:
    root = root.resolve()
    manifest_path = project_path(root, manifest_path)
    base_manifest_path = project_path(root, base_manifest_path)
    sweep_base_manifest_path = manifest_path.with_name(
        "base_render_input_manifest.json"
    )
    derived_manifest_path = manifest_path.with_name(
        "derived_render_input_manifest.json"
    )
    for path in (
        manifest_path,
        base_manifest_path,
        sweep_base_manifest_path,
        derived_manifest_path,
    ):
        path.relative_to(root / "outputs")

    manifest = load_json(manifest_path)
    base_manifest = load_json(base_manifest_path)
    sweep_base_manifest = load_json(sweep_base_manifest_path)
    derived_manifest = load_json(derived_manifest_path)
    records = indexed_records(manifest, "asset render manifest")
    sweep_base_records = indexed_records(
        sweep_base_manifest, "asset sweep-base manifest"
    )
    derived_records = indexed_records(derived_manifest, "asset derived manifest")
    if set(records) != set(sweep_base_records) | set(derived_records):
        raise ValueError("base and derived manifests do not partition the asset manifest")
    if set(sweep_base_records) & set(derived_records):
        raise ValueError("asset base and derived manifests overlap")

    cameras = rendered_base_cameras(root, base_manifest)

    updates: dict[Path, dict[str, Any]] = {}
    counts = {parent: {"base": 0, "sweep": 0} for parent in cameras}
    new_hashes: dict[str, str] = {}
    for scene_id, record in records.items():
        metadata_path = project_path(root, record["metadata_path"])
        metadata_path.relative_to(root / "outputs")
        metadata = load_json(metadata_path)
        if str(metadata["scene_id"]) != scene_id:
            raise ValueError(f"asset render scene id mismatch: {scene_id}")
        source_metadata_hash(root, metadata, scene_id)
        sweep = metadata["sweep"]
        parent = str(sweep["parent_scene_id"])
        kind = str(sweep["kind"])
        if parent not in cameras or kind not in {"base", "sweep"}:
            raise ValueError(f"invalid asset sweep membership: {scene_id}")
        counts[parent][kind] += 1

        declared_value = record.get("metadata_sha256")
        declared_hash = str(declared_value) if declared_value is not None else None
        actual_hash = sha256_bytes(json_bytes(metadata))
        without_binding = dict(metadata)
        without_binding.pop("camera_binding", None)
        baseline_hash = sha256_bytes(json_bytes(without_binding))
        if (
            declared_hash is not None
            and actual_hash != declared_hash
            and baseline_hash != declared_hash
        ):
            raise ValueError(f"asset render metadata hash mismatch: {scene_id}")
        if kind == "base":
            if scene_id not in sweep_base_records:
                raise ValueError(f"base partition mismatch: {scene_id}")
        else:
            if scene_id not in derived_records:
                raise ValueError(f"derived partition mismatch: {scene_id}")
        metadata["camera_binding"] = cameras[parent]
        updates[metadata_path] = metadata
        new_hashes[scene_id] = sha256_bytes(json_bytes(metadata))

    if set(counts) != set(cameras) or any(
        value != {"base": 1, "sweep": 12} for value in counts.values()
    ):
        raise ValueError("camera binding requires complete 13-sample asset groups")
    for scene_id, record in records.items():
        record["metadata_sha256"] = new_hashes[scene_id]
    for scene_id, record in derived_records.items():
        record["metadata_sha256"] = new_hashes[scene_id]
    for scene_id, record in sweep_base_records.items():
        record["metadata_sha256"] = new_hashes[scene_id]

    for path, metadata in updates.items():
        write_json(path, metadata)
    write_json(manifest_path, manifest)
    write_json(sweep_base_manifest_path, sweep_base_manifest)
    write_json(derived_manifest_path, derived_manifest)
    return {
        "group_count": len(cameras),
        "sample_count": sum(sum(value.values()) for value in counts.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(freeze_cameras(args.root, args.manifest, args.base_manifest), indent=2))


if __name__ == "__main__":
    main()
