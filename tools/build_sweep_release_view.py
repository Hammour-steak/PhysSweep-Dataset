#!/usr/bin/env python3
"""Build the canonical derived-only PhysSweep sweep release beside the base view."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not __package__ and str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.base_release_attribution import load_billiards_templates
from tools.base_release_schema import (
    FIXTURE_SCHEMA,
    SAMPLE_ENTRIES,
    SWEEP_DERIVED_LEVELS,
    SWEEP_PARAMETER_FIELDS,
    SWEEP_SAMPLE_SCHEMA,
    materialize_sweep_sample,
    sha256,
    validate_base_metadata,
    validate_sweep_metadata,
    verified_file,
    write_json,
)
from tools.build_base_release_view import (
    DEFAULT_RELEASE_ROOT,
    FIXTURE_CATALOG_SCHEMA,
    PipelineSpec,
    collect_sample_attribution,
    index_unique,
    index_unique_string,
    load_json,
    load_pipeline_records,
    one_object_release_roots,
    release_contract_fields,
    release_documents,
    render_sources,
    safe_scene_id,
    validate_mask_artifacts,
    validate_pipeline_specs,
    validate_release_manifest_contract,
    validate_trajectory_artifact,
    write_pipeline_manifests,
    write_release_catalogs,
)


VIEW_SCHEMA = "physweep_sweep_release_view_v1"
PIPELINE_SCHEMA = "physweep_sweep_pipeline_view_v1"
GROUP_SCHEMA = "physweep_sweep_group_manifest_v1"
AUDIT_SCHEMA = "physweep_sweep_release_view_audit_v1"
SWEEP_AXES = SWEEP_PARAMETER_FIELDS
DERIVED_LEVELS = SWEEP_DERIVED_LEVELS
SWEEP_INDEX_FIELDS = {
    "scene_id",
    "path",
    "metadata_sha256",
    "parameter",
    "level_index",
}


def sweep_descriptor(record: dict[str, Any]) -> dict[str, Any]:
    descriptor = {
        "target_object_id": safe_scene_id(record.get("target_object_id")),
        "parameter": str(record.get("axis", "")),
        "level_index": record.get("level_index"),
        "value": record.get("value"),
    }
    if (
        descriptor["parameter"] not in SWEEP_AXES
        or isinstance(descriptor["level_index"], bool)
        or not isinstance(descriptor["level_index"], int)
        or descriptor["level_index"] not in DERIVED_LEVELS
        or isinstance(descriptor["value"], bool)
        or not isinstance(descriptor["value"], (int, float))
    ):
        raise ValueError(f"invalid sweep descriptor: {record.get('scene_id')}")
    return descriptor


def relative_path(path: Path, start: Path) -> str:
    return Path(os.path.relpath(path.resolve(), start.resolve())).as_posix()


def sibling_release_roots(
    base_root: Path,
    sweep_root: Path,
    *,
    allow_staging_markers: bool = False,
) -> tuple[Path, Path]:
    base_root = base_root.resolve()
    sweep_root = sweep_root.resolve()
    sweep_name = sweep_root.name
    if (
        allow_staging_markers
        and sweep_name.startswith(".")
        and sweep_name.endswith(".building")
    ):
        sweep_name = sweep_name[1 : -len(".building")]
    if (
        base_root.name != "base"
        or sweep_name != "sweep"
        or base_root.parent.name != "one_object"
        or base_root.parent != sweep_root.parent
    ):
        raise ValueError("release roots must be one_object/base and one_object/sweep")
    return base_root, sweep_root


def release_directory_name(output: Path, allow_staging_markers: bool) -> str:
    name = output.name
    if allow_staging_markers and name.startswith(".") and name.endswith(".building"):
        name = name[1 : -len(".building")]
    return safe_scene_id(name)


def fixture_asset_bindings(value: Any) -> list[tuple[str, str]]:
    bindings: list[tuple[str, str]] = []
    if isinstance(value, list):
        for item in value:
            bindings.extend(fixture_asset_bindings(item))
    elif isinstance(value, dict):
        path = value.get("path")
        digest = value.get("sha256")
        if isinstance(path, str) and path.startswith("fixture_assets/"):
            bindings.append((path, str(digest)))
        for item in value.values():
            bindings.extend(fixture_asset_bindings(item))
    return bindings


def sweep_sort_key(record: dict[str, Any]) -> tuple[int, int]:
    return SWEEP_AXES.index(str(record["parameter"])), int(record["level_index"])


def load_base_groups(base_root: Path, release_parent: Path) -> dict[str, dict[str, Any]]:
    base_root = base_root.resolve()
    root_manifest_path = base_root / "manifest.json"
    root_manifest = load_json(root_manifest_path)
    groups: dict[str, dict[str, Any]] = {}
    base_prefix = relative_path(base_root, release_parent)
    for family, binding in root_manifest["pipelines"].items():
        pipeline_path = verified_file(
            base_root / str(binding["manifest"]),
            str(binding["manifest_sha256"]),
            f"{family} base pipeline manifest",
        )
        for record in load_json(pipeline_path)["records"]:
            scene_id = safe_scene_id(record["scene_id"])
            metadata_path = verified_file(
                base_root / family / scene_id / "metadata.json",
                str(record["metadata_sha256"]),
                f"{scene_id} base metadata",
            )
            summary = validate_base_metadata(load_json(metadata_path))
            group_id = safe_scene_id(summary["group_id"])
            if group_id in groups:
                raise ValueError(f"duplicate base group: {group_id}")
            groups[group_id] = {
                "family": family,
                "scene_id": scene_id,
                "metadata_sha256": str(record["metadata_sha256"]),
                "path": f"{base_prefix}/{family}/{scene_id}",
            }
    if len(groups) != int(root_manifest["sample_count"]):
        raise ValueError("base group count differs")
    return groups


def validate_groups(
    sweep_records: list[dict[str, Any]],
    base_by_source: dict[str, dict[str, Any]],
    base_groups: dict[str, dict[str, Any]],
) -> dict[str, str]:
    group_by_scene: dict[str, str] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in sweep_records:
        scene_id = safe_scene_id(record["scene_id"])
        if scene_id in group_by_scene:
            raise ValueError(f"duplicate sweep scene id: {scene_id}")
        parent = str(record["parent"])
        if parent not in base_by_source:
            raise ValueError(f"sweep parent is absent from base manifest: {scene_id}")
        group_id = safe_scene_id(base_by_source[parent]["scene_id"])
        if group_id not in base_groups:
            raise ValueError(f"canonical base view lacks group: {group_id}")
        group_by_scene[scene_id] = group_id
        grouped[group_id].append(record)
    if set(grouped) != set(base_groups):
        raise ValueError("base and sweep group sets differ")
    expected_axis_levels = {
        (axis, level) for axis in SWEEP_AXES for level in DERIVED_LEVELS
    }
    for group_id, records in grouped.items():
        observed = {
            (str(record.get("axis")), record.get("level_index"))
            for record in records
        }
        targets = {str(record.get("target_object_id")) for record in records}
        if len(records) != 12 or observed != expected_axis_levels or len(targets) != 1:
            raise ValueError(f"derived one-factor group differs: {group_id}")
    return group_by_scene


def completed_record(
    marker_path: Path,
    sample_path: Path,
    scene_id: str,
) -> dict[str, Any] | None:
    if not marker_path.is_file() or marker_path.is_symlink():
        return None
    try:
        marker = load_json(marker_path)
        compact = marker.get("record", {})
        metadata_path = sample_path / "metadata.json"
        if (
            set(marker)
            != {"family", "fixture_sha256", "record", "render_contract"}
            or marker.get("family") != sample_path.parent.name
            or not isinstance(compact, dict)
            or set(compact) != {"scene_id", "metadata_sha256"}
            or compact.get("scene_id") != scene_id
            or not sample_path.is_dir()
            or sample_path.is_symlink()
            or {path.name for path in sample_path.iterdir()} != SAMPLE_ENTRIES
            or any(path.is_symlink() for path in sample_path.rglob("*"))
        ):
            return None
        metadata_path = verified_file(
            metadata_path,
            str(compact.get("metadata_sha256", "")),
            f"{scene_id} resumed metadata",
        )
        metadata = load_json(metadata_path)
        validate_sweep_metadata(metadata)
        trajectory = verified_file(
            sample_path / "trajectory.npz",
            str(metadata["artifacts"]["trajectory"]["sha256"]),
            f"{scene_id} resumed trajectory",
        )
        validate_trajectory_artifact(trajectory, metadata)
        verified_file(
            sample_path / "video.mp4",
            str(metadata["artifacts"]["video"]["sha256"]),
            f"{scene_id} resumed video",
        )
        validate_mask_artifacts(sample_path, metadata)
        fixture_hash = str(marker["fixture_sha256"])
        if fixture_hash != str(metadata["physics"]["fixture"]["sha256"]):
            return None
        verified_file(
            sample_path.parents[1] / "fixtures" / f"{fixture_hash}.json",
            fixture_hash,
            f"{scene_id} resumed fixture",
        )
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return marker


def materialize_record(
    *,
    work: Path,
    metadata_record: dict[str, Any],
    group_id: str,
    physics_by_id: dict[str, dict[str, Any]],
    specs: dict[str, PipelineSpec],
    billiards_templates: dict[str, dict[str, str]],
) -> dict[str, Any]:
    scene_id = safe_scene_id(metadata_record["scene_id"])
    spec = specs[str(metadata_record["source_schema_version"])]
    sample_path = work / spec.name / scene_id
    marker_path = work / ".completed" / spec.name / f"{scene_id}.json"
    marker = completed_record(marker_path, sample_path, scene_id)
    if marker is not None:
        return marker
    if sample_path.exists() or sample_path.is_symlink():
        if sample_path.parent != work / spec.name:
            raise ValueError(f"unsafe incomplete sample path: {sample_path}")
        if sample_path.is_symlink() or not sample_path.is_dir():
            sample_path.unlink()
        else:
            shutil.rmtree(sample_path)
    physics_record = physics_by_id.get(scene_id)
    if physics_record is None:
        raise ValueError(f"sweep physics record is missing: {scene_id}")
    if str(physics_record["metadata_sha256"]) != str(
        metadata_record["metadata_sha256"]
    ):
        raise ValueError(f"sweep metadata manifest hash differs: {scene_id}")
    sources = render_sources(spec, scene_id, physics_record)
    compact, fixture_hash = materialize_sweep_sample(
        sweep=sweep_descriptor(metadata_record),
        target=sample_path,
        family=spec.name,
        group_id=group_id,
        source=sources["source_metadata"],
        source_metadata_sha256=sources["hashes"]["metadata_sha256"],
        resolved_scene=sources["resolved_scene"],
        render_record=sources["render_record"],
        trajectory_source_path=sources["trajectory"],
        video_source_path=sources["video"],
        video_sha256=sources["hashes"]["video_sha256"],
        masks_source_path=sources["masks"],
        release_root=work,
        source_project_root=spec.project_root,
        billiards_templates=billiards_templates,
        render_metadata=sources["render_metadata"],
    )
    marker = {
        "family": spec.name,
        "fixture_sha256": fixture_hash,
        "record": compact,
        "render_contract": sources["render_contract"],
    }
    write_json(marker_path, marker)
    return marker


def group_manifest(
    *,
    output_name: str,
    results: list[dict[str, Any]],
    base_groups: dict[str, dict[str, Any]],
    work: Path,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        family = str(result["family"])
        compact = result["record"]
        scene_id = str(compact["scene_id"])
        metadata = load_json(work / family / scene_id / "metadata.json")
        descriptor = metadata["sweep"]
        grouped[str(metadata["group_id"])].append(
            {
                "scene_id": scene_id,
                "path": f"{output_name}/{family}/{scene_id}",
                "metadata_sha256": str(compact["metadata_sha256"]),
                "target_object_id": descriptor["target_object_id"],
                "parameter": descriptor["parameter"],
                "level_index": descriptor["level_index"],
            }
        )
    records = []
    for group_id in sorted(base_groups):
        base = base_groups[group_id]
        sweeps = sorted(
            grouped[group_id],
            key=sweep_sort_key,
        )
        targets = {record.pop("target_object_id") for record in sweeps}
        if len(targets) != 1:
            raise ValueError(f"sweep group target differs: {group_id}")
        records.append(
            {
                "group_id": group_id,
                "family": base["family"],
                "target_object_id": targets.pop(),
                "base": {
                    key: base[key]
                    for key in ("scene_id", "path", "metadata_sha256")
                },
                "sweeps": sweeps,
            }
        )
    return {
        "schema_version": GROUP_SCHEMA,
        "path_base": "release_parent",
        "group_count": len(records),
        "sweep_count": sum(len(record["sweeps"]) for record in records),
        "records": records,
    }


def build_view(
    *,
    release_project_root: Path,
    release_manifest: Path,
    base_root: Path,
    output: Path,
    pipeline_specs: Iterable[PipelineSpec],
    workers: int,
    resume: bool,
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be positive")
    base_root, output = sibling_release_roots(base_root, output)
    work = output.with_name(f".{output.name}.building")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"sweep release already exists: {output}")
    if work.exists() and not resume:
        raise FileExistsError(f"sweep staging already exists: {work}")
    specs = validate_pipeline_specs(pipeline_specs)
    release_path, release, base, metadata_path, metadata, physics = release_documents(
        release_project_root, release_manifest
    )
    sweep_records = [
        record for record in metadata["records"] if record.get("kind") == "sweep"
    ]
    expected_count = int(release["derived_count"])
    if len(sweep_records) != expected_count:
        raise ValueError("release derived counts disagree")
    selected_schemas = {
        str(record["source_schema_version"]) for record in sweep_records
    }
    if selected_schemas != set(specs):
        raise ValueError("pipeline schemas differ from selected sweep records")
    base_by_source = index_unique_string(
        base["records"], "metadata_path", "base source metadata path"
    )
    base_groups = load_base_groups(base_root, output.parent)
    group_by_scene = validate_groups(sweep_records, base_by_source, base_groups)
    for record in sweep_records:
        scene_id = safe_scene_id(record["scene_id"])
        family = specs[str(record["source_schema_version"])].name
        if base_groups[group_by_scene[scene_id]]["family"] != family:
            raise ValueError(f"base and sweep families differ: {scene_id}")
    physics_by_id = index_unique(physics["records"], "physics")

    output.parent.mkdir(parents=True, exist_ok=True)
    if not work.exists():
        work.mkdir()
        (work / "assets").mkdir()
        (work / "fixtures").mkdir()
        (work / "fixture_assets").mkdir()
        (work / ".completed").mkdir()
    elif work.is_symlink() or not work.is_dir():
        raise ValueError(f"invalid sweep staging directory: {work}")
    for directory in ("assets", "fixtures", "fixture_assets", ".completed"):
        (work / directory).mkdir(exist_ok=True)
    for spec in specs.values():
        (work / spec.name).mkdir(exist_ok=True)
        (work / ".completed" / spec.name).mkdir(exist_ok=True)
    billiards_templates = (
        load_billiards_templates(PROJECT_ROOT)
        if any(spec.name == "billiards" for spec in specs.values())
        else {}
    )
    ordered = sorted(sweep_records, key=lambda record: str(record["scene_id"]))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                materialize_record,
                work=work,
                metadata_record=record,
                group_id=group_by_scene[str(record["scene_id"])],
                physics_by_id=physics_by_id,
                specs=specs,
                billiards_templates=billiards_templates,
            )
            for record in ordered
        ]
        results = [future.result() for future in futures]

    render_contracts = {json.dumps(result["render_contract"], sort_keys=True) for result in results}
    if len(render_contracts) != 1:
        raise ValueError("release render contracts differ")
    render_contract = json.loads(next(iter(render_contracts)))
    fixture_usage = Counter(str(result["fixture_sha256"]) for result in results)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    used_asset_ids: set[str] = set()
    used_hdri: dict[str, str] = {}
    for result in results:
        family = str(result["family"])
        compact = result["record"]
        grouped[family].append(compact)
        collect_sample_attribution(
            load_json(work / family / str(compact["scene_id"]) / "metadata.json"),
            used_asset_ids,
            used_hdri,
        )

    pipeline_bindings = write_pipeline_manifests(
        work=work,
        specs=specs,
        grouped=grouped,
        pipeline_schema=PIPELINE_SCHEMA,
        sample_schema=SWEEP_SAMPLE_SCHEMA,
    )
    catalogs = write_release_catalogs(
        work=work,
        fixture_usage=fixture_usage,
        used_asset_ids=used_asset_ids,
        used_hdri=used_hdri,
    )
    groups = group_manifest(
        output_name=output.name,
        results=results,
        base_groups=base_groups,
        work=work,
    )
    group_path = work / "group_manifest.json"
    write_json(group_path, groups)
    base_manifest_path = base_root.resolve() / "manifest.json"
    manifest = {
        "schema_version": VIEW_SCHEMA,
        "dataset_id": str(release["dataset_id"]),
        "sample_kind": "sweep",
        "sample_count": expected_count,
        "group_count": len(base_groups),
        **release_contract_fields(
            render_contract=render_contract,
            sample_schema=SWEEP_SAMPLE_SCHEMA,
        ),
        "base_release": {
            "manifest": f"{relative_path(base_root, output.parent)}/manifest.json",
            "manifest_sha256": sha256(base_manifest_path),
        },
        "group_index": {
            "schema_version": GROUP_SCHEMA,
            "manifest": "group_manifest.json",
            "manifest_sha256": sha256(group_path),
            "group_count": len(base_groups),
        },
        **catalogs,
        "provenance": {
            "source_generation_release_metadata_sha256": sha256(metadata_path),
            "source_sweep_release_manifest_sha256": sha256(release_path),
        },
        "pipelines": pipeline_bindings,
    }
    write_json(work / "manifest.json", manifest)
    (work / "README.txt").write_text(
        "Canonical PhysSweep derived sweep release v1.\n"
        "metadata.json is the sample authority; group_manifest.json indexes each base and its 12 one-factor variants.\n"
        "The release excludes base samples and generation-only frames, logs, and source metadata copies.\n",
        encoding="utf-8",
    )
    audit = verify_view(work, base_root=base_root, allow_staging_markers=True)
    shutil.rmtree(work / ".completed")
    expected_entries = {
        "README.txt",
        "assets",
        "fixture_assets",
        "fixtures",
        "group_manifest.json",
        "manifest.json",
        *(spec.name for spec in specs.values()),
    }
    if {path.name for path in work.iterdir()} != expected_entries:
        raise ValueError("unexpected entries after removing staging markers")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"sweep release appeared during build: {output}")
    work.replace(output)
    return {**audit, "view": str(output)}


def verify_view(
    output: Path,
    *,
    base_root: Path,
    allow_staging_markers: bool = False,
) -> dict[str, Any]:
    base_root, output = sibling_release_roots(
        base_root,
        output,
        allow_staging_markers=allow_staging_markers,
    )
    manifest = load_json(output / "manifest.json")
    render_contract = validate_release_manifest_contract(
        manifest,
        view_schema=VIEW_SCHEMA,
        sample_schema=SWEEP_SAMPLE_SCHEMA,
        label="sweep",
        extra_fields={
            "sample_kind",
            "group_count",
            "base_release",
            "group_index",
        },
    )
    if manifest.get("sample_kind") != "sweep":
        raise ValueError("not a canonical PhysSweep sweep release")
    provenance = manifest.get("provenance", {})
    if (
        set(provenance)
        != {
            "source_generation_release_metadata_sha256",
            "source_sweep_release_manifest_sha256",
        }
        or any(len(str(value)) != 64 for value in provenance.values())
    ):
        raise ValueError("sweep provenance contract is incomplete")
    base_binding = manifest.get("base_release", {})
    expected_base_manifest = (
        f"{relative_path(base_root, output.parent)}/manifest.json"
    )
    if (
        set(base_binding) != {"manifest", "manifest_sha256"}
        or base_binding.get("manifest") != expected_base_manifest
    ):
        raise ValueError("base release binding differs")
    verified_file(
        base_root.resolve() / "manifest.json",
        str(base_binding.get("manifest_sha256", "")),
        "base release manifest",
    )
    base_groups = load_base_groups(base_root, output.parent)
    base_families = {str(record["family"]) for record in base_groups.values()}
    if (
        len(base_groups) != int(manifest["group_count"])
        or set(manifest["pipelines"]) != base_families
    ):
        raise ValueError("base and sweep release families differ")
    fixture_binding = manifest["fixture_catalog"]
    if (
        set(fixture_binding)
        != {"schema_version", "manifest", "manifest_sha256", "entry_count"}
        or fixture_binding.get("schema_version") != FIXTURE_CATALOG_SCHEMA
        or fixture_binding.get("manifest") != "fixtures/manifest.json"
    ):
        raise ValueError("sweep fixture catalog binding differs")
    fixture_path = verified_file(
        output / str(fixture_binding["manifest"]),
        str(fixture_binding["manifest_sha256"]),
        "sweep fixture catalog",
    )
    fixture_manifest = load_json(fixture_path)
    if (
        set(fixture_manifest) != {"schema_version", "entry_count", "records"}
        or fixture_manifest.get("schema_version") != FIXTURE_CATALOG_SCHEMA
        or int(fixture_binding["entry_count"])
        != int(fixture_manifest.get("entry_count", -1))
        or int(fixture_manifest.get("entry_count", -1))
        != len(fixture_manifest.get("records", []))
    ):
        raise ValueError("sweep fixture catalog differs")
    fixture_hashes = set()
    expected_fixture_usage = {}
    expected_fixture_assets: set[str] = set()
    for record in fixture_manifest["records"]:
        if set(record) != {"sha256", "usage_count"} or int(record["usage_count"]) <= 0:
            raise ValueError("sweep fixture catalog record differs")
        digest = str(record["sha256"])
        if digest in fixture_hashes:
            raise ValueError("duplicate sweep fixture hash")
        fixture_document = load_json(
            verified_file(
                output / "fixtures" / f"{digest}.json", digest, "sweep fixture"
            )
        )
        if fixture_document.get("schema_version") != FIXTURE_SCHEMA:
            raise ValueError("sweep fixture schema differs")
        for relative, asset_digest in fixture_asset_bindings(fixture_document):
            if Path(relative).parent.as_posix() != "fixture_assets":
                raise ValueError("nested fixture asset path differs")
            verified_file(output / relative, asset_digest, "sweep fixture asset")
            expected_fixture_assets.add(Path(relative).name)
        fixture_hashes.add(digest)
        expected_fixture_usage[digest] = int(record["usage_count"])
    group_binding = manifest["group_index"]
    if (
        set(group_binding)
        != {"schema_version", "manifest", "manifest_sha256", "group_count"}
        or group_binding.get("schema_version") != GROUP_SCHEMA
        or group_binding.get("manifest") != "group_manifest.json"
        or int(group_binding.get("group_count", -1)) != len(base_groups)
    ):
        raise ValueError("sweep group binding differs")
    group_path = verified_file(
        output / str(group_binding["manifest"]),
        str(group_binding["manifest_sha256"]),
        "sweep group manifest",
    )
    groups = load_json(group_path)
    if (
        set(groups)
        != {"schema_version", "path_base", "group_count", "sweep_count", "records"}
        or groups.get("schema_version") != GROUP_SCHEMA
        or groups.get("path_base") != "release_parent"
        or int(groups.get("group_count", -1)) != int(manifest["group_count"])
        or int(groups.get("group_count", -1)) != len(groups.get("records", []))
        or int(groups.get("sweep_count", -1)) != int(manifest["sample_count"])
    ):
        raise ValueError("sweep group manifest counts differ")
    expected_axis_levels = {
        (axis, level) for axis in SWEEP_AXES for level in DERIVED_LEVELS
    }
    expected_axis_level_order = [
        (axis, level) for axis in SWEEP_AXES for level in DERIVED_LEVELS
    ]
    canonical_output_name = release_directory_name(output, allow_staging_markers)
    observed_groups = set()
    for group in groups["records"]:
        group_id = safe_scene_id(group.get("group_id"))
        if group_id in observed_groups or group_id not in base_groups:
            raise ValueError(f"sweep group identity differs: {group_id}")
        observed_groups.add(group_id)
        expected_base = base_groups[group_id]
        sweeps = group.get("sweeps", [])
        if (
            set(group)
            != {"group_id", "family", "target_object_id", "base", "sweeps"}
            or group.get("family") != expected_base["family"]
            or group.get("base")
            != {
                key: expected_base[key]
                for key in ("scene_id", "path", "metadata_sha256")
            }
            or len(sweeps) != 12
            or any(set(record) != SWEEP_INDEX_FIELDS for record in sweeps)
            or {
                (record.get("parameter"), record.get("level_index"))
                for record in sweeps
            }
            != expected_axis_levels
            or [
                (record.get("parameter"), record.get("level_index"))
                for record in sweeps
            ]
            != expected_axis_level_order
        ):
            raise ValueError(f"sweep group record differs: {group_id}")
        target_object_id = safe_scene_id(group["target_object_id"])
        for record in sweeps:
            scene_id = safe_scene_id(record["scene_id"])
            if record["path"] != (
                f"{canonical_output_name}/{expected_base['family']}/{scene_id}"
            ):
                raise ValueError(f"sweep group path differs: {scene_id}")
    if observed_groups != set(base_groups):
        raise ValueError("sweep group coverage differs")
    indexed = {
        str(sweep["scene_id"]): (
            str(group["group_id"]),
            str(group["target_object_id"]),
            sweep,
        )
        for group in groups["records"]
        for sweep in group["sweeps"]
    }
    if len(indexed) != sum(len(group["sweeps"]) for group in groups["records"]):
        raise ValueError("duplicate sweep identity in group index")
    attribution_binding = manifest["asset_attribution"]
    if (
        set(attribution_binding) != {"manifest", "manifest_sha256", "record_count"}
        or attribution_binding.get("manifest")
        != "assets/attribution_manifest.json"
    ):
        raise ValueError("sweep attribution binding differs")
    attribution_path = verified_file(
        output / str(attribution_binding["manifest"]),
        str(attribution_binding["manifest_sha256"]),
        "sweep asset attribution",
    )
    attribution = load_json(attribution_path)
    if int(attribution_binding["record_count"]) != int(attribution["record_count"]):
        raise ValueError("sweep attribution count differs")
    count = 0
    scene_ids = set()
    observed_fixture_usage: Counter[str] = Counter()
    expected_top = {
        "README.txt",
        "assets",
        "fixture_assets",
        "fixtures",
        "group_manifest.json",
        "manifest.json",
    }
    if allow_staging_markers:
        expected_top.add(".completed")
    for family, binding in manifest["pipelines"].items():
        family = safe_scene_id(family)
        expected_top.add(family)
        records = load_pipeline_records(
            output=output,
            family=family,
            binding=binding,
            pipeline_schema=PIPELINE_SCHEMA,
            sample_schema=SWEEP_SAMPLE_SCHEMA,
            label="sweep",
        )
        expected_samples = set()
        for record in records:
            if set(record) != {"scene_id", "metadata_sha256"}:
                raise ValueError(f"sweep pipeline record differs: {family}")
            scene_id = safe_scene_id(record["scene_id"])
            if scene_id in scene_ids:
                raise ValueError(f"duplicate sweep scene id: {scene_id}")
            scene_ids.add(scene_id)
            expected_samples.add(scene_id)
            sample = output / family / scene_id
            if not sample.is_dir() or sample.is_symlink():
                raise ValueError(f"sweep sample is not a real directory: {scene_id}")
            metadata_path = verified_file(
                sample / "metadata.json",
                str(record["metadata_sha256"]),
                f"{scene_id} sweep metadata",
            )
            metadata = load_json(metadata_path)
            summary = validate_sweep_metadata(metadata)
            if summary["scene_id"] != scene_id or summary["family"] != family:
                raise ValueError(f"sweep metadata identity differs: {scene_id}")
            if metadata_path.is_symlink():
                raise ValueError(f"sweep metadata must be materialized: {scene_id}")
            indexed_group, indexed_target, indexed_record = indexed[scene_id]
            if (
                indexed_group != summary["group_id"]
                or indexed_target != metadata["sweep"]["target_object_id"]
                or indexed_record["metadata_sha256"] != record["metadata_sha256"]
                or indexed_record["parameter"] != metadata["sweep"]["parameter"]
                or indexed_record["level_index"] != metadata["sweep"]["level_index"]
            ):
                raise ValueError(f"sweep group index differs: {scene_id}")
            fixture_hash = str(metadata["physics"]["fixture"]["sha256"])
            if fixture_hash not in fixture_hashes:
                raise ValueError(f"sweep fixture is absent: {scene_id}")
            observed_fixture_usage[fixture_hash] += 1
            if int(metadata["physics"]["time"]["output_fps"]) != int(
                render_contract["video_encoding"]["fps"]
            ):
                raise ValueError(f"sweep video and physics frame rates differ: {scene_id}")
            trajectory = verified_file(
                sample / "trajectory.npz",
                str(metadata["artifacts"]["trajectory"]["sha256"]),
                f"{scene_id} trajectory",
            )
            if trajectory.is_symlink():
                raise ValueError(f"sweep trajectory must be materialized: {scene_id}")
            validate_trajectory_artifact(trajectory, metadata)
            video = verified_file(
                sample / "video.mp4",
                str(metadata["artifacts"]["video"]["sha256"]),
                f"{scene_id} video",
            )
            if video.is_symlink():
                raise ValueError(f"sweep video must be materialized: {scene_id}")
            validate_mask_artifacts(sample, metadata)
            if {path.name for path in sample.iterdir()} != SAMPLE_ENTRIES:
                raise ValueError(f"unexpected sweep sample files: {scene_id}")
            count += 1
        actual_entries = {path.name for path in (output / family).iterdir()}
        if actual_entries != expected_samples | {"manifest.json"}:
            raise ValueError(f"sweep pipeline directories differ: {family}")
    if count != int(manifest["sample_count"]) or len(indexed) != count:
        raise ValueError("sweep release totals differ")
    if dict(observed_fixture_usage) != expected_fixture_usage:
        raise ValueError("sweep fixture usage differs")
    if {path.name for path in (output / "fixtures").iterdir()} != {
        "manifest.json",
        *(f"{digest}.json" for digest in fixture_hashes),
    }:
        raise ValueError("unexpected sweep fixture files")
    if {path.name for path in (output / "fixture_assets").iterdir()} != (
        expected_fixture_assets
    ):
        raise ValueError("unexpected sweep fixture assets")
    if {path.name for path in (output / "assets").iterdir()} != {
        "attribution_manifest.json"
    }:
        raise ValueError("unexpected sweep asset files")
    if {path.name for path in output.iterdir()} != expected_top:
        raise ValueError("unexpected sweep release root entries")
    symlinks = [path for path in output.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"sweep release contains symlinks: {symlinks[:3]}")
    return {
        "schema_version": AUDIT_SCHEMA,
        "view": str(output),
        "sample_count": count,
        "group_count": int(manifest["group_count"]),
        "pipeline_count": len(manifest["pipelines"]),
        "passed": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-project-root", type=Path)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=DEFAULT_RELEASE_ROOT,
        help="Canonical output root; must be named one_object.",
    )
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--pipeline",
        nargs=5,
        action="append",
        metavar=("NAME", "SOURCE_SCHEMA", "PROJECT_ROOT", "RENDER_ROOT", "MASK_ROOT"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_root, output = one_object_release_roots(args.release_root)
    if args.verify_only:
        result = verify_view(output, base_root=base_root)
    else:
        if args.release_project_root is None or args.release_manifest is None:
            raise SystemExit(
                "--release-project-root and --release-manifest are required when building"
            )
        specs = [
            PipelineSpec(name, schema, Path(root), Path(render), Path(masks))
            for name, schema, root, render, masks in (args.pipeline or [])
        ]
        result = build_view(
            release_project_root=args.release_project_root,
            release_manifest=args.release_manifest,
            base_root=base_root,
            output=output,
            pipeline_specs=specs,
            workers=args.workers,
            resume=args.resume,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
