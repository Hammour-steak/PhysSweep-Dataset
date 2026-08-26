#!/usr/bin/env python3
"""Build the canonical, compact, pipeline-classified PhysSweep base release."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from audit_release_provenance import (
        audit_release,
        load_json,
        manifest_binding,
        project_path,
    )
    from base_release_schema import (
        BASE_SAMPLE_SCHEMA,
        MASK_MANIFEST_SCHEMA,
        TRAJECTORY_FIELDS,
        TRAJECTORY_SCHEMA,
        materialize_base_sample,
        sha256,
        validate_base_metadata,
        verified_file,
        write_json,
    )
except ModuleNotFoundError:
    from tools.audit_release_provenance import (
        audit_release,
        load_json,
        manifest_binding,
        project_path,
    )
    from tools.base_release_schema import (
        BASE_SAMPLE_SCHEMA,
        MASK_MANIFEST_SCHEMA,
        TRAJECTORY_FIELDS,
        TRAJECTORY_SCHEMA,
        materialize_base_sample,
        sha256,
        validate_base_metadata,
        verified_file,
        write_json,
    )


VIEW_SCHEMA = "physweep_base_release_view_v3"
PIPELINE_SCHEMA = "physweep_base_pipeline_view_v3"
AUDIT_SCHEMA = "physweep_base_release_view_audit_v3"


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    source_schema_version: str
    project_root: Path
    render_root: Path


def safe_scene_id(value: Any) -> str:
    scene_id = str(value)
    if not scene_id or Path(scene_id).name != scene_id or scene_id in {".", ".."}:
        raise ValueError(f"invalid scene id: {scene_id!r}")
    return scene_id


def index_unique(
    records: Iterable[dict[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        scene_id = safe_scene_id(record["scene_id"])
        if scene_id in result:
            raise ValueError(f"duplicate {label} scene id: {scene_id}")
        result[scene_id] = record
    return result


def index_unique_string(
    records: Iterable[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        value = str(record.get(key, ""))
        if not value:
            raise ValueError(f"missing {label}: {key}")
        if value in result:
            raise ValueError(f"duplicate {label}: {value}")
        result[value] = record
    return result


def release_documents(
    release_project_root: Path,
    release_manifest: Path,
) -> tuple[
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
]:
    release_project_root = release_project_root.resolve()
    release_path = project_path(release_project_root, release_manifest)
    audit_release(release_path, release_project_root)
    release = load_json(release_path)
    base_path = manifest_binding(
        release_project_root,
        release,
        "base_manifest",
        "base_manifest_sha256",
        "release base_manifest",
    )
    metadata_path = manifest_binding(
        release_project_root,
        release,
        "metadata_manifest",
        "metadata_manifest_sha256",
        "release metadata_manifest",
    )
    physics_path = manifest_binding(
        release_project_root,
        release,
        "physics_manifest",
        "physics_manifest_sha256",
        "release physics_manifest",
    )
    return (
        release_path,
        release,
        base_path,
        load_json(base_path),
        metadata_path,
        load_json(metadata_path),
        physics_path,
        load_json(physics_path),
    )


def validate_pipeline_specs(
    specs: Iterable[PipelineSpec],
) -> dict[str, PipelineSpec]:
    by_schema: dict[str, PipelineSpec] = {}
    names: set[str] = set()
    for raw in specs:
        spec = PipelineSpec(
            name=safe_scene_id(raw.name),
            source_schema_version=str(raw.source_schema_version),
            project_root=raw.project_root.resolve(),
            render_root=(
                raw.render_root.resolve()
                if raw.render_root.is_absolute()
                else (raw.project_root / raw.render_root).resolve()
            ),
        )
        if spec.source_schema_version in by_schema:
            raise ValueError(f"duplicate pipeline schema: {spec.source_schema_version}")
        if spec.name in names:
            raise ValueError(f"duplicate pipeline name: {spec.name}")
        if not spec.render_root.is_dir():
            raise FileNotFoundError(f"render root: {spec.render_root}")
        by_schema[spec.source_schema_version] = spec
        names.add(spec.name)
    if not by_schema:
        raise ValueError("at least one pipeline is required")
    return by_schema


def resolved_record_path(
    project_root: Path,
    record: dict[str, Any],
    path_key: str,
    hash_key: str,
    label: str,
) -> Path:
    return verified_file(
        project_path(project_root, str(record[path_key])),
        str(record[hash_key]),
        label,
    )


def render_sources(
    spec: PipelineSpec,
    scene_id: str,
    physics_record: dict[str, Any],
) -> dict[str, Any]:
    if (
        not bool(physics_record.get("ok"))
        or not bool(physics_record.get("audit_passed"))
        or physics_record.get("failed_checks")
    ):
        raise ValueError(f"base physics audit did not pass: {scene_id}")
    metadata = resolved_record_path(
        spec.project_root,
        physics_record,
        "metadata_path",
        "metadata_sha256",
        f"{scene_id} metadata",
    )
    trajectory = resolved_record_path(
        spec.project_root,
        physics_record,
        "trajectory_path",
        "trajectory_sha256",
        f"{scene_id} trajectory",
    )
    resolved_record_path(
        spec.project_root,
        physics_record,
        "audit_path",
        "audit_sha256",
        f"{scene_id} trajectory audit",
    )
    resolved_scene = resolved_record_path(
        spec.project_root,
        physics_record,
        "resolved_scene_path",
        "resolved_scene_sha256",
        f"{scene_id} resolved scene",
    )
    render_record_path = spec.render_root / "frames" / scene_id / "render_record.json"
    render_record = load_json(render_record_path)
    if str(render_record.get("scene_id")) != scene_id:
        raise ValueError(f"render record scene id mismatch: {scene_id}")
    render_record_hash = sha256(render_record_path)
    render_metadata_hash = str(render_record["metadata_sha256"])
    if render_metadata_hash == str(physics_record["metadata_sha256"]):
        render_metadata = None
    else:
        render_metadata = verified_file(
            project_path(spec.project_root, str(render_record["metadata_path"])),
            render_metadata_hash,
            f"{scene_id} render metadata",
        )
        bound = load_json(render_metadata)
        source_binding = bound.get("source_metadata", {})
        trajectory_binding = bound.get("trajectory", {})
        if (
            str(source_binding.get("sha256"))
            != str(physics_record["metadata_sha256"])
            or str(trajectory_binding.get("sha256"))
            != str(physics_record["trajectory_sha256"])
            or project_path(spec.project_root, str(source_binding.get("path", ""))).resolve()
            != metadata
            or project_path(
                spec.project_root, str(trajectory_binding.get("path", ""))
            ).resolve()
            != trajectory
        ):
            raise ValueError(f"render metadata is not bound to release physics: {scene_id}")
    if render_record.get("trajectory_sha256") not in (
        None,
        physics_record["trajectory_sha256"],
    ):
        raise ValueError(f"render and release trajectories differ: {scene_id}")
    video = verified_file(
        project_path(spec.project_root, str(render_record["video_path"])),
        str(render_record["video_sha256"]),
        f"{scene_id} video",
    )
    source_metadata = load_json(metadata)
    render_config = source_metadata.get("render_request") or source_metadata.get(
        "render"
    )
    if not isinstance(render_config, dict):
        raise ValueError(f"render configuration is missing: {scene_id}")
    render_contract = {
        "engine": render_record.get("render_engine") or render_config.get("engine"),
        "samples": render_record.get("render_samples")
        or render_config.get("samples"),
        "video_encoding": render_record.get("video_encoding"),
    }
    if (
        not render_contract["engine"]
        or int(render_contract["samples"] or 0) <= 0
        or not isinstance(render_contract["video_encoding"], dict)
    ):
        raise ValueError(f"render contract is incomplete: {scene_id}")
    render_contract["samples"] = int(render_contract["samples"])
    masks = spec.render_root / "masks" / scene_id
    return {
        "metadata": metadata,
        "trajectory": trajectory,
        "resolved_scene": resolved_scene,
        "render_record": render_record_path.resolve(),
        "render_record_sha256": render_record_hash,
        "render_metadata": render_metadata,
        "render_metadata_sha256": (
            render_metadata_hash if render_metadata is not None else None
        ),
        "video": video,
        "masks": masks.resolve() if masks.is_dir() else None,
        "render_contract": render_contract,
        "hashes": {
            "metadata_sha256": str(physics_record["metadata_sha256"]),
            "trajectory_sha256": str(physics_record["trajectory_sha256"]),
            "resolved_scene_sha256": str(physics_record["resolved_scene_sha256"]),
            "video_sha256": str(render_record["video_sha256"]),
        },
    }


def build_view(
    *,
    release_project_root: Path,
    release_manifest: Path,
    output: Path,
    pipeline_specs: Iterable[PipelineSpec],
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"base release already exists: {output}")
    specs = validate_pipeline_specs(pipeline_specs)
    (
        release_path,
        release,
        base_path,
        base,
        metadata_path,
        metadata,
        physics_path,
        physics,
    ) = release_documents(release_project_root, release_manifest)
    base_records = [
        record for record in metadata["records"] if record.get("kind") == "base"
    ]
    expected_count = int(release["base_count"])
    if len(base_records) != expected_count or int(base["sample_count"]) != expected_count:
        raise ValueError("release base counts disagree")
    base_by_source = index_unique_string(
        base["records"], "metadata_path", "base source metadata path"
    )
    metadata_by_source = index_unique_string(
        base_records, "parent", "release base parent metadata path"
    )
    if set(base_by_source) != set(metadata_by_source):
        raise ValueError("base and metadata manifests select different source records")
    physics_by_id = index_unique(physics["records"], "physics")
    selected_schemas = {str(record["source_schema_version"]) for record in base_records}
    if selected_schemas != set(specs):
        raise ValueError(
            f"pipeline schemas differ: selected={sorted(selected_schemas)} "
            f"configured={sorted(specs)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        work = Path(temporary) / output.name
        work.mkdir()
        grouped: dict[str, list[dict[str, Any]]] = {
            spec.name: [] for spec in specs.values()
        }
        mask_counts = {spec.name: 0 for spec in specs.values()}
        render_contract: dict[str, Any] | None = None
        for metadata_record in sorted(base_records, key=lambda item: item["scene_id"]):
            scene_id = safe_scene_id(metadata_record["scene_id"])
            source_metadata_path = str(metadata_record["parent"])
            group_id = safe_scene_id(base_by_source[source_metadata_path]["scene_id"])
            spec = specs[str(metadata_record["source_schema_version"])]
            physics_record = physics_by_id.get(scene_id)
            if physics_record is None:
                raise ValueError(f"base physics record is missing: {scene_id}")
            if str(physics_record["metadata_sha256"]) != str(
                metadata_record["metadata_sha256"]
            ):
                raise ValueError(f"metadata manifest hash differs: {scene_id}")
            sources = render_sources(spec, scene_id, physics_record)
            if render_contract is None:
                render_contract = sources["render_contract"]
            elif render_contract != sources["render_contract"]:
                raise ValueError(f"release render contract differs: {scene_id}")
            compact = materialize_base_sample(
                target=work / spec.name / scene_id,
                family=spec.name,
                group_id=group_id,
                source_metadata_path=sources["metadata"],
                source_metadata_sha256=sources["hashes"]["metadata_sha256"],
                resolved_scene_path=sources["resolved_scene"],
                resolved_scene_sha256=sources["hashes"]["resolved_scene_sha256"],
                render_record_path=sources["render_record"],
                render_record_sha256=sources["render_record_sha256"],
                trajectory_source_path=sources["trajectory"],
                trajectory_source_sha256=sources["hashes"]["trajectory_sha256"],
                video_source_path=sources["video"],
                video_sha256=sources["hashes"]["video_sha256"],
                masks_source_path=sources["masks"],
                render_metadata_path=sources["render_metadata"],
                render_metadata_sha256=sources["render_metadata_sha256"],
            )
            mask_counts[spec.name] += int(compact.pop("has_masks"))
            compact["source_metadata_path"] = source_metadata_path
            grouped[spec.name].append(compact)

        pipeline_bindings: dict[str, Any] = {}
        total_masks = 0
        for spec in sorted(specs.values(), key=lambda value: value.name):
            records = grouped[spec.name]
            pipeline_path = work / spec.name / "manifest.json"
            pipeline_manifest = {
                "schema_version": PIPELINE_SCHEMA,
                "pipeline": spec.name,
                "source_schema_version": spec.source_schema_version,
                "sample_count": len(records),
                "mask_count": mask_counts[spec.name],
                "records": records,
            }
            write_json(pipeline_path, pipeline_manifest)
            pipeline_bindings[spec.name] = {
                "manifest": f"{spec.name}/manifest.json",
                "manifest_sha256": sha256(pipeline_path),
            }
            total_masks += mask_counts[spec.name]

        manifest = {
            "schema_version": VIEW_SCHEMA,
            "dataset_id": str(release["dataset_id"]),
            "kind": "base_only",
            "storage_mode": "compact_metadata_with_absolute_artifact_symlinks",
            "sample_count": expected_count,
            "mask_count": total_masks,
            "release_manifest": str(release_path),
            "release_manifest_sha256": sha256(release_path),
            "base_manifest_sha256": sha256(base_path),
            "metadata_manifest_sha256": sha256(metadata_path),
            "physics_manifest_sha256": sha256(physics_path),
            "render_contract": render_contract,
            "sample_schema_version": BASE_SAMPLE_SCHEMA,
            "trajectory_schema_version": TRAJECTORY_SCHEMA,
            "mask_manifest_schema_version": MASK_MANIFEST_SCHEMA,
            "pipelines": pipeline_bindings,
        }
        write_json(work / "manifest.json", manifest)
        (work / "README.txt").write_text(
            "Canonical PhysSweep base release.\n"
            "metadata.json is the sample authority; trajectory arrays use one object axis.\n"
            "Generation diagnostics and inspection frames are not release artifacts.\n",
            encoding="utf-8",
        )
        verify_view(work)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"base release appeared during build: {output}")
        work.replace(output)
    return verify_view(output)


def _release_root(release_path: Path, release: dict[str, Any]) -> Path:
    base_reference = Path(str(release["base_manifest"]))
    if base_reference.is_absolute():
        return release_path.parent
    candidates = [
        parent for parent in release_path.parents if (parent / base_reference).is_file()
    ]
    if len(candidates) != 1:
        raise ValueError("cannot identify the release project root")
    return candidates[0]


def _validate_trajectory(path: Path, metadata: dict[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != TRAJECTORY_FIELDS:
            raise ValueError(f"non-canonical trajectory fields: {path}")
        if str(np.asarray(archive["schema_version"]).item()) != TRAJECTORY_SCHEMA:
            raise ValueError(f"trajectory schema differs: {path}")
        object_ids = [str(value) for value in np.asarray(archive["object_ids"]).tolist()]
        metadata_ids = [
            str(record["object_id"]) for record in metadata["physics"]["objects"]
        ]
        if object_ids != metadata_ids:
            raise ValueError(f"trajectory object axis differs: {path}")


def _validate_masks(sample: Path, metadata: dict[str, Any]) -> None:
    binding = metadata["artifacts"]["masks"]
    manifest_path = verified_file(
        sample / "mask_manifest.json",
        str(binding["manifest_sha256"]),
        f"{metadata['scene_id']} mask manifest",
    )
    manifest = load_json(manifest_path)
    if (
        manifest.get("schema_version") != MASK_MANIFEST_SCHEMA
        or manifest.get("scene_id") != metadata.get("scene_id")
    ):
        raise ValueError("mask manifest identity differs")
    masks = sample / "masks"
    if not masks.is_symlink() or not masks.is_dir():
        raise ValueError("mask artifact is not a valid symlink")
    expected_objects = [
        (record["object_id"], int(record["mask_instance_id"]))
        for record in metadata["physics"]["objects"]
    ]
    records = manifest.get("objects", [])
    if [(record["object_id"], int(record["instance_id"])) for record in records] != expected_objects:
        raise ValueError("mask manifest object axis differs")
    frame_count = int(manifest["frame_count"])
    for record in records:
        object_id = safe_scene_id(record["object_id"])
        paths = sorted((masks / object_id).glob("frame_*.png"))
        hashes = record.get("frame_sha256", [])
        expected_names = [f"frame_{index:04d}.png" for index in range(1, frame_count + 1)]
        if [path.name for path in paths] != expected_names or len(hashes) != frame_count:
            raise ValueError("mask manifest frame count differs")
        for path, expected_hash in zip(paths, hashes):
            verified_file(
                path,
                str(expected_hash),
                f"{metadata['scene_id']} mask frame",
            )


def verify_view(output: Path) -> dict[str, Any]:
    output = output.resolve()
    manifest = load_json(output / "manifest.json")
    if (
        manifest.get("schema_version") != VIEW_SCHEMA
        or manifest.get("kind") != "base_only"
        or manifest.get("storage_mode")
        != "compact_metadata_with_absolute_artifact_symlinks"
        or manifest.get("sample_schema_version") != BASE_SAMPLE_SCHEMA
        or manifest.get("trajectory_schema_version") != TRAJECTORY_SCHEMA
        or manifest.get("mask_manifest_schema_version") != MASK_MANIFEST_SCHEMA
    ):
        raise ValueError("not a canonical PhysSweep base release")
    render_contract = manifest.get("render_contract")
    if (
        not isinstance(render_contract, dict)
        or not render_contract.get("engine")
        or int(render_contract.get("samples", 0)) <= 0
        or not isinstance(render_contract.get("video_encoding"), dict)
        or int(render_contract["video_encoding"].get("fps", 0)) <= 0
    ):
        raise ValueError("base render contract is incomplete")
    release_path = verified_file(
        Path(manifest["release_manifest"]),
        str(manifest["release_manifest_sha256"]),
        "base release source manifest",
    )
    release = load_json(release_path)
    release_root = _release_root(release_path, release)
    for key in ("base_manifest", "metadata_manifest", "physics_manifest"):
        verified_file(
            project_path(release_root, str(release[key])),
            str(manifest[f"{key}_sha256"]),
            f"base release {key}",
        )
    if (
        str(manifest["dataset_id"]) != str(release["dataset_id"])
        or int(manifest["sample_count"]) != int(release["base_count"])
    ):
        raise ValueError("base and source release identities differ")

    count = 0
    mask_count = 0
    scene_ids: set[str] = set()
    group_ids: set[str] = set()
    source_paths: set[str] = set()
    expected_top = {"manifest.json", "README.txt"}
    for family, binding in manifest["pipelines"].items():
        family = safe_scene_id(family)
        expected_top.add(family)
        relative_manifest = Path(family) / "manifest.json"
        if (
            set(binding) != {"manifest", "manifest_sha256"}
            or Path(str(binding.get("manifest", ""))) != relative_manifest
        ):
            raise ValueError(f"pipeline manifest path differs: {family}")
        pipeline_path = verified_file(
            output / relative_manifest,
            str(binding["manifest_sha256"]),
            f"{family} pipeline manifest",
        )
        document = load_json(pipeline_path)
        records = document.get("records", [])
        if (
            document.get("schema_version") != PIPELINE_SCHEMA
            or set(document)
            != {"schema_version", "pipeline", "source_schema_version", "sample_count", "mask_count", "records"}
            or document.get("pipeline") != family
            or not document.get("source_schema_version")
            or not isinstance(records, list)
            or int(document.get("sample_count", -1)) != len(records)
        ):
            raise ValueError(f"pipeline manifest differs: {family}")
        expected_samples = set()
        family_masks = 0
        for record in records:
            if set(record) != {
                "scene_id", "group_id", "metadata_sha256",
                "source_metadata_sha256", "source_metadata_path",
            }:
                raise ValueError(f"pipeline record fields differ: {family}")
            scene_id = safe_scene_id(record["scene_id"])
            group_id = safe_scene_id(record["group_id"])
            source_path = str(record.get("source_metadata_path", ""))
            if (
                scene_id in scene_ids
                or group_id in group_ids
                or not source_path
                or source_path in source_paths
            ):
                raise ValueError(f"duplicate base identity: {scene_id}")
            scene_ids.add(scene_id)
            group_ids.add(group_id)
            source_paths.add(source_path)
            expected_samples.add(scene_id)
            sample = output / family / scene_id
            if not sample.is_dir() or sample.is_symlink():
                raise ValueError(f"base sample is not a real directory: {scene_id}")
            metadata_path = verified_file(
                sample / "metadata.json",
                str(record["metadata_sha256"]),
                f"{scene_id} metadata",
            )
            if metadata_path.is_symlink():
                raise ValueError(f"metadata must be materialized: {scene_id}")
            metadata = load_json(metadata_path)
            summary = validate_base_metadata(metadata)
            if (
                summary["scene_id"] != scene_id
                or summary["group_id"] != group_id
                or summary["family"] != family
                or metadata["lineage"]["source_metadata_sha256"]
                != record["source_metadata_sha256"]
            ):
                raise ValueError(f"metadata identity differs: {scene_id}")
            if int(metadata["physics"]["time"]["output_fps"]) != int(
                render_contract["video_encoding"]["fps"]
            ):
                raise ValueError(f"video and physics frame rates differ: {scene_id}")
            trajectory_binding = metadata["artifacts"]["trajectory"]
            trajectory = verified_file(
                sample / "trajectory.npz",
                str(trajectory_binding["sha256"]),
                f"{scene_id} trajectory",
            )
            if trajectory.is_symlink():
                raise ValueError(f"trajectory must be materialized: {scene_id}")
            _validate_trajectory(trajectory, metadata)
            video_binding = metadata["artifacts"]["video"]
            video = verified_file(
                sample / "video.mp4",
                str(video_binding["sha256"]),
                f"{scene_id} video",
            )
            if not video.is_symlink():
                raise ValueError(f"video must remain a source symlink: {scene_id}")
            expected_entries = {"metadata.json", "trajectory.npz", "video.mp4"}
            if "masks" in metadata["artifacts"]:
                expected_entries.update({"masks", "mask_manifest.json"})
                _validate_masks(sample, metadata)
                family_masks += 1
            elif (sample / "masks").exists() or (sample / "masks").is_symlink():
                raise ValueError(f"unbound masks are present: {scene_id}")
            if {path.name for path in sample.iterdir()} != expected_entries:
                raise ValueError(f"unexpected base sample files: {scene_id}")
            count += 1
        actual_samples = {
            path.name for path in (output / family).iterdir() if path.is_dir()
        }
        if actual_samples != expected_samples:
            raise ValueError(f"pipeline sample directories differ: {family}")
        if family_masks != int(document.get("mask_count", -1)):
            raise ValueError(f"pipeline mask count differs: {family}")
        mask_count += family_masks
    if {path.name for path in output.iterdir()} != expected_top:
        raise ValueError("unexpected base root entries")
    if count != int(manifest["sample_count"]) or mask_count != int(manifest["mask_count"]):
        raise ValueError("base release totals differ")
    return {
        "schema_version": AUDIT_SCHEMA,
        "view": str(output),
        "sample_count": count,
        "mask_count": mask_count,
        "pipeline_count": len(manifest["pipelines"]),
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-project-root", type=Path)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pipeline",
        nargs=4,
        action="append",
        metavar=("NAME", "SOURCE_SCHEMA", "PROJECT_ROOT", "RENDER_ROOT"),
        help="Repeat once per release pipeline.",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        result = verify_view(args.output)
    else:
        if args.release_project_root is None or args.release_manifest is None:
            raise SystemExit(
                "--release-project-root and --release-manifest are required when building"
            )
        specs = [
            PipelineSpec(name, schema, Path(root), Path(render_root))
            for name, schema, root, render_root in (args.pipeline or [])
        ]
        result = build_view(
            release_project_root=args.release_project_root,
            release_manifest=args.release_manifest,
            output=args.output,
            pipeline_specs=specs,
        )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
