#!/usr/bin/env python3
"""Build a non-destructive, pipeline-classified view of release base samples."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from audit_release_provenance import (
        audit_release,
        load_json,
        manifest_binding,
        project_path,
        sha256,
    )
except ModuleNotFoundError:
    from tools.audit_release_provenance import (
        audit_release,
        load_json,
        manifest_binding,
        project_path,
        sha256,
    )


VIEW_SCHEMA = "physweep_base_release_view_v1"
PIPELINE_SCHEMA = "physweep_base_pipeline_view_v1"


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    source_schema_version: str
    project_root: Path
    render_root: Path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


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


def verified_file(path: Path, expected_hash: str, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")
    actual = sha256(path)
    if actual != expected_hash:
        raise ValueError(
            f"{label} hash mismatch: {path}; expected={expected_hash} actual={actual}"
        )
    return path


def linked_file(path: Path, source: Path) -> None:
    path.symlink_to(source.resolve())


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


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


def release_documents(
    release_project_root: Path,
    release_manifest: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
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
        load_json(base_path),
        load_json(metadata_path),
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
            raise ValueError(
                f"duplicate pipeline schema: {spec.source_schema_version}"
            )
        if spec.name in names:
            raise ValueError(f"duplicate pipeline name: {spec.name}")
        if not spec.render_root.is_dir():
            raise FileNotFoundError(f"render root: {spec.render_root}")
        by_schema[spec.source_schema_version] = spec
        names.add(spec.name)
    if not by_schema:
        raise ValueError("at least one pipeline is required")
    return by_schema


def render_sources(
    spec: PipelineSpec,
    scene_id: str,
    physics_record: dict[str, Any],
) -> dict[str, Any]:
    frame_root = spec.render_root / "frames" / scene_id
    render_record_path = frame_root / "render_record.json"
    render_record = load_json(render_record_path)
    if str(render_record.get("scene_id")) != scene_id:
        raise ValueError(f"render record scene id mismatch: {scene_id}")
    if not bool(physics_record.get("ok")) or not bool(
        physics_record.get("audit_passed")
    ):
        raise ValueError(f"base physics audit did not pass: {scene_id}")
    if physics_record.get("failed_checks"):
        raise ValueError(f"base physics has failed checks: {scene_id}")

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
    audit = resolved_record_path(
        spec.project_root,
        physics_record,
        "audit_path",
        "audit_sha256",
        f"{scene_id} audit",
    )
    resolved_scene = resolved_record_path(
        spec.project_root,
        physics_record,
        "resolved_scene_path",
        "resolved_scene_sha256",
        f"{scene_id} resolved scene",
    )
    render_metadata_hash = str(render_record["metadata_sha256"])
    if render_metadata_hash == str(physics_record["metadata_sha256"]):
        render_metadata = metadata
    else:
        render_metadata = verified_file(
            project_path(spec.project_root, str(render_record["metadata_path"])),
            render_metadata_hash,
            f"{scene_id} render metadata",
        )
        bound_metadata = load_json(render_metadata)
        source_binding = bound_metadata.get("source_metadata", {})
        trajectory_binding = bound_metadata.get("trajectory", {})
        if (
            str(source_binding.get("sha256"))
            != str(physics_record["metadata_sha256"])
            or str(trajectory_binding.get("sha256"))
            != str(physics_record["trajectory_sha256"])
            or project_path(
                spec.project_root, str(source_binding.get("path", ""))
            ).resolve()
            != metadata
            or project_path(
                spec.project_root, str(trajectory_binding.get("path", ""))
            ).resolve()
            != trajectory
        ):
            raise ValueError(
                f"render metadata is not bound to release physics: {scene_id}"
            )
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
    inspection_frames = [
        project_path(spec.project_root, str(value))
        for value in render_record.get("inspection_frames", [])
    ]
    if not inspection_frames or any(not path.is_file() for path in inspection_frames):
        raise FileNotFoundError(f"inspection frames are incomplete: {scene_id}")
    if len({path.resolve() for path in inspection_frames}) != len(inspection_frames):
        raise ValueError(f"inspection frames contain duplicates: {scene_id}")
    if any(
        path.parent.name != scene_id
        or not is_within(path, spec.project_root)
        for path in inspection_frames
    ):
        raise ValueError(f"inspection frame has an invalid source path: {scene_id}")

    masks = spec.render_root / "masks" / scene_id
    hashes = {
        "metadata_sha256": str(physics_record["metadata_sha256"]),
        "trajectory_sha256": str(physics_record["trajectory_sha256"]),
        "audit_sha256": str(physics_record["audit_sha256"]),
        "resolved_scene_sha256": str(physics_record["resolved_scene_sha256"]),
        "video_sha256": str(render_record["video_sha256"]),
        "render_record_sha256": sha256(render_record_path),
    }
    linked_render_metadata = (
        render_metadata if render_metadata != metadata else None
    )
    if linked_render_metadata is not None:
        hashes["render_metadata_sha256"] = render_metadata_hash
    return {
        "metadata": metadata,
        "render_metadata": linked_render_metadata,
        "trajectory": trajectory,
        "audit": audit,
        "resolved_scene": resolved_scene,
        "video": video,
        "render_record": render_record_path.resolve(),
        "inspection_frames": [path.resolve() for path in inspection_frames],
        "masks": masks.resolve() if masks.is_dir() else None,
        "hashes": hashes,
    }


def materialize_sample(
    sample_dir: Path,
    sources: dict[str, Any],
) -> None:
    sample_dir.mkdir(parents=True)
    for name, filename in (
        ("metadata", "metadata.json"),
        ("trajectory", "trajectory.npz"),
        ("audit", "trajectory_audit.json"),
        ("resolved_scene", "resolved_scene.json"),
        ("video", "video.mp4"),
        ("render_record", "render_record.json"),
    ):
        linked_file(sample_dir / filename, sources[name])
    if sources["render_metadata"] is not None:
        linked_file(sample_dir / "render_metadata.json", sources["render_metadata"])
    frame_dir = sample_dir / "frames"
    frame_dir.mkdir()
    for source in sources["inspection_frames"]:
        linked_file(frame_dir / source.name, source)
    if sources["masks"] is not None:
        (sample_dir / "masks").symlink_to(
            sources["masks"], target_is_directory=True
        )


def record_hashes(
    scene_id: str,
    logical_base_id: str,
    source_metadata_path: str,
    sources: dict[str, Any],
) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "logical_base_id": logical_base_id,
        "source_metadata_path": source_metadata_path,
        "sample_path": scene_id,
        **sources["hashes"],
    }


def build_view(
    *,
    release_project_root: Path,
    release_manifest: Path,
    output: Path,
    pipeline_specs: Iterable[PipelineSpec],
) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"base view already exists: {output}")
    specs = validate_pipeline_specs(pipeline_specs)
    release_path, release, base, metadata, physics = release_documents(
        release_project_root, release_manifest
    )
    base_records = [
        record for record in metadata["records"] if record.get("kind") == "base"
    ]
    expected_count = int(release["base_count"])
    if (
        len(base_records) != expected_count
        or int(base["sample_count"]) != expected_count
    ):
        raise ValueError("release base counts disagree")
    base_by_source = index_unique_string(
        base["records"], "metadata_path", "base source metadata path"
    )
    metadata_by_source = index_unique_string(
        base_records, "parent", "release base parent metadata path"
    )
    if set(base_by_source) != set(metadata_by_source):
        raise ValueError("base and metadata manifests select different source records")
    logical_base_ids = {
        safe_scene_id(record["scene_id"]) for record in base["records"]
    }
    generated_base_ids = {
        safe_scene_id(record["scene_id"]) for record in base_records
    }
    if (
        len(logical_base_ids) != expected_count
        or len(generated_base_ids) != expected_count
    ):
        raise ValueError("base manifests contain duplicate scene ids")
    physics_by_id = index_unique(physics["records"], "physics")
    selected_schemas = {str(record["source_schema_version"]) for record in base_records}
    if selected_schemas != set(specs):
        raise ValueError(
            f"pipeline schemas differ: selected={sorted(selected_schemas)} "
            f"configured={sorted(specs)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as temporary:
        work = Path(temporary) / output.name
        work.mkdir()
        grouped: dict[str, list[dict[str, Any]]] = {
            spec.name: [] for spec in specs.values()
        }
        mask_counts = {spec.name: 0 for spec in specs.values()}
        for metadata_record in sorted(base_records, key=lambda item: item["scene_id"]):
            scene_id = safe_scene_id(metadata_record["scene_id"])
            source_metadata_path = str(metadata_record["parent"])
            logical_base_id = safe_scene_id(
                base_by_source[source_metadata_path]["scene_id"]
            )
            spec = specs[str(metadata_record["source_schema_version"])]
            physics_record = physics_by_id.get(scene_id)
            if physics_record is None:
                raise ValueError(f"base physics record is missing: {scene_id}")
            if str(physics_record["metadata_sha256"]) != str(
                metadata_record["metadata_sha256"]
            ):
                raise ValueError(f"metadata manifest hash differs: {scene_id}")
            sources = render_sources(spec, scene_id, physics_record)
            materialize_sample(work / spec.name / scene_id, sources)
            grouped[spec.name].append(
                record_hashes(
                    scene_id,
                    logical_base_id,
                    source_metadata_path,
                    sources,
                )
            )
            mask_counts[spec.name] += int(sources["masks"] is not None)

        pipelines: dict[str, Any] = {}
        for spec in sorted(specs.values(), key=lambda item: item.name):
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
            pipelines[spec.name] = {
                "source_schema_version": spec.source_schema_version,
                "sample_count": len(records),
                "mask_count": mask_counts[spec.name],
                "manifest": f"{spec.name}/manifest.json",
                "manifest_sha256": sha256(pipeline_path),
            }

        base_manifest_path = project_path(
            release_project_root, str(release["base_manifest"])
        )
        metadata_manifest_path = project_path(
            release_project_root, str(release["metadata_manifest"])
        )
        physics_manifest_path = project_path(
            release_project_root, str(release["physics_manifest"])
        )
        view_manifest = {
            "schema_version": VIEW_SCHEMA,
            "dataset_id": str(release["dataset_id"]),
            "kind": "base_only",
            "storage_mode": "absolute_symlink_view",
            "sample_count": expected_count,
            "release_manifest": str(release_path),
            "release_manifest_sha256": sha256(release_path),
            "base_manifest_sha256": sha256(base_manifest_path),
            "metadata_manifest_sha256": sha256(metadata_manifest_path),
            "physics_manifest_sha256": sha256(physics_manifest_path),
            "pipelines": pipelines,
        }
        write_json(work / "manifest.json", view_manifest)
        (work / "README.txt").write_text(
            "This is a non-authoritative symlink view of release base samples.\n"
            "Source metadata and release manifests remain the sole authority.\n"
            "Derived sweep samples are deliberately excluded.\n",
            encoding="utf-8",
        )
        verify_view(work)
        if output.exists():
            raise FileExistsError(f"base view appeared during build: {output}")
        work.replace(output)
    return verify_view(output)


def verify_view(output: Path) -> dict[str, Any]:
    output = output.resolve()
    manifest = load_json(output / "manifest.json")
    if (
        manifest.get("schema_version") != VIEW_SCHEMA
        or manifest.get("kind") != "base_only"
        or manifest.get("storage_mode") != "absolute_symlink_view"
    ):
        raise ValueError("not a PhysSweep base release view")
    release_path = verified_file(
        Path(manifest["release_manifest"]),
        str(manifest["release_manifest_sha256"]),
        "view release manifest",
    )
    release = load_json(release_path)
    base_reference = Path(str(release["base_manifest"]))
    if base_reference.is_absolute():
        release_root = release_path.parent
    else:
        candidates = [
            parent
            for parent in release_path.parents
            if (parent / base_reference).is_file()
        ]
        if len(candidates) != 1:
            raise ValueError("cannot identify the release project root")
        release_root = candidates[0]
    for key in ("base_manifest", "metadata_manifest", "physics_manifest"):
        verified_file(
            project_path(release_root, str(release[key])),
            str(manifest[f"{key}_sha256"]),
            f"view {key}",
        )
    if (
        str(manifest["dataset_id"]) != str(release["dataset_id"])
        or int(manifest["sample_count"]) != int(release["base_count"])
    ):
        raise ValueError("view and release identities differ")

    verified_count = 0
    mask_count = 0
    scene_ids: set[str] = set()
    logical_base_ids: set[str] = set()
    source_metadata_paths: set[str] = set()
    pipeline_names: set[str] = set()
    for raw_name, pipeline in manifest["pipelines"].items():
        name = safe_scene_id(raw_name)
        pipeline_names.add(name)
        expected_manifest = Path(name) / "manifest.json"
        if Path(str(pipeline["manifest"])) != expected_manifest:
            raise ValueError(f"pipeline manifest path is invalid: {name}")
        pipeline_path = verified_file(
            output / expected_manifest,
            str(pipeline["manifest_sha256"]),
            f"{name} pipeline manifest",
        )
        document = load_json(pipeline_path)
        if (
            document.get("schema_version") != PIPELINE_SCHEMA
            or document.get("pipeline") != name
            or document.get("source_schema_version")
            != pipeline.get("source_schema_version")
            or len(document.get("records", [])) != int(pipeline["sample_count"])
            or int(document.get("mask_count", -1))
            != int(pipeline.get("mask_count", -2))
        ):
            raise ValueError(f"pipeline manifest is inconsistent: {name}")
        expected_directories: set[str] = set()
        for record in document["records"]:
            scene_id = safe_scene_id(record["scene_id"])
            logical_base_id = safe_scene_id(record["logical_base_id"])
            sample_path = safe_scene_id(record["sample_path"])
            source_metadata_path = str(record.get("source_metadata_path", ""))
            if sample_path != scene_id or not source_metadata_path:
                raise ValueError(f"base identity record is invalid: {scene_id}")
            if (
                scene_id in scene_ids
                or logical_base_id in logical_base_ids
                or source_metadata_path in source_metadata_paths
            ):
                raise ValueError(f"duplicate base identity in view: {scene_id}")
            scene_ids.add(scene_id)
            logical_base_ids.add(logical_base_id)
            source_metadata_paths.add(source_metadata_path)
            expected_directories.add(sample_path)
            sample = pipeline_path.parent / sample_path
            if not sample.is_dir() or sample.is_symlink():
                raise ValueError(f"base sample is not a real directory: {scene_id}")
            for filename, hash_key in (
                ("metadata.json", "metadata_sha256"),
                ("trajectory.npz", "trajectory_sha256"),
                ("trajectory_audit.json", "audit_sha256"),
                ("resolved_scene.json", "resolved_scene_sha256"),
                ("video.mp4", "video_sha256"),
                ("render_record.json", "render_record_sha256"),
            ):
                path = sample / filename
                if not path.is_symlink():
                    raise ValueError(f"view file is not a symlink: {path}")
                verified_file(path, str(record[hash_key]), f"{scene_id} {filename}")
            render_metadata = sample / "render_metadata.json"
            if "render_metadata_sha256" in record:
                if not render_metadata.is_symlink():
                    raise ValueError(
                        f"view render metadata is not a symlink: {render_metadata}"
                    )
                verified_file(
                    render_metadata,
                    str(record["render_metadata_sha256"]),
                    f"{scene_id} render_metadata.json",
                )
            elif render_metadata.exists() or render_metadata.is_symlink():
                raise ValueError(f"unexpected render metadata link: {scene_id}")
            render_record = load_json(sample / "render_record.json")
            expected_frames = {
                Path(value).name for value in render_record["inspection_frames"]
            }
            actual_frames = {
                path.name for path in (sample / "frames").iterdir()
            }
            if actual_frames != expected_frames or any(
                not (sample / "frames" / name).is_symlink()
                or not (sample / "frames" / name).is_file()
                for name in actual_frames
            ):
                raise ValueError(f"inspection frame view differs: {scene_id}")
            mask_path = sample / "masks"
            if mask_path.exists() or mask_path.is_symlink():
                if not mask_path.is_symlink() or not mask_path.is_dir():
                    raise ValueError(f"mask link is broken: {scene_id}")
                mask_count += 1
            if not (sample / "frames").is_dir() or (sample / "frames").is_symlink():
                raise ValueError(f"frame view is not a real directory: {scene_id}")
            expected_entries = {
                "metadata.json",
                "trajectory.npz",
                "trajectory_audit.json",
                "resolved_scene.json",
                "video.mp4",
                "render_record.json",
                "frames",
            }
            for optional in ("masks", "render_metadata.json"):
                if (sample / optional).exists() or (sample / optional).is_symlink():
                    expected_entries.add(optional)
            if {path.name for path in sample.iterdir()} != expected_entries:
                raise ValueError(f"unexpected files in base view: {scene_id}")
            verified_count += 1
        if int(document["mask_count"]) != sum(
            (pipeline_path.parent / record["sample_path"] / "masks").is_symlink()
            for record in document["records"]
        ):
            raise ValueError(f"pipeline mask count differs: {name}")
        actual_directories = {
            path.name for path in pipeline_path.parent.iterdir() if path.is_dir()
        }
        if actual_directories != expected_directories:
            raise ValueError(f"pipeline sample directories differ: {name}")
    if verified_count != int(manifest["sample_count"]):
        raise ValueError("verified base count differs from the view manifest")
    expected_top_entries = pipeline_names | {"manifest.json", "README.txt"}
    if {path.name for path in output.iterdir()} != expected_top_entries:
        raise ValueError("unexpected files in the base view root")
    return {
        "schema_version": "physweep_base_release_view_audit_v1",
        "view": str(output),
        "sample_count": verified_count,
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
                "--release-project-root and --release-manifest are required "
                "when building a view"
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
