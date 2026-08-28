#!/usr/bin/env python3
"""Canonical base-release library shared by base and sweep publishers."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from tools.core.paths import safe_scene_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from tools.release.audit_release_provenance import (
    audit_release,
    load_json,
    manifest_binding,
    project_path,
)
from tools.release.base_release_attribution import (
    build_attribution,
    collect_asset_ids,
    load_billiards_templates,
)
from tools.release.base_release_schema import (
    BASE_SAMPLE_SCHEMA,
    FIXTURE_SCHEMA,
    MASK_MANIFEST_SCHEMA,
    SAMPLE_ENTRIES,
    SAMPLE_LAYOUT_CONTRACT,
    TRAJECTORY_FIELDS,
    TRAJECTORY_SCHEMA,
    materialize_base_sample,
    sha256,
    validate_base_metadata,
    verified_file,
    write_json,
)


VIEW_SCHEMA = "physweep_base_release_view_v14"
PIPELINE_SCHEMA = "physweep_base_pipeline_view_v12"
AUDIT_SCHEMA = "physweep_base_release_view_audit_v14"
FIXTURE_CATALOG_SCHEMA = "physweep_static_fixture_catalog_v2"
DEFAULT_RELEASE_ROOT = PROJECT_ROOT / "outputs" / "one_object"
VIDEO_ENCODING_FIELDS = (
    "codec",
    "constant_rate_factor",
    "container",
    "fps",
    "gop_size_frames",
    "preset",
    "profile_version",
)
COMMON_RELEASE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "storage_mode",
        "sample_count",
        "render_contract",
        "coordinate_contract",
        "camera_projection_contract",
        "temporal_contract",
        "trajectory_contract",
        "mask_contract",
        "sample_layout_contract",
        "text_contract",
        "collision_proxy_contract",
        "object_visual_contract",
        "object_dynamics_contract",
        "fixture_binding_contract",
        "object_identity_contract",
        "object_semantics_contract",
        "sample_schema_version",
        "trajectory_schema_version",
        "mask_manifest_schema_version",
        "fixture_schema_version",
        "fixture_catalog",
        "asset_attribution",
        "provenance",
        "pipelines",
    }
)
PIPELINE_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "pipeline",
        "source_schema_version",
        "sample_schema_version",
        "sample_count",
        "records",
    }
)


def one_object_release_roots(release_root: Path) -> tuple[Path, Path]:
    release_root = Path(release_root)
    if not release_root.is_absolute():
        release_root = PROJECT_ROOT / release_root
    release_root = release_root.resolve(strict=False)
    if release_root.name != "one_object":
        raise ValueError("release root must be named one_object")
    return release_root / "base", release_root / "sweep"

COORDINATE_CONTRACT = {
    "units": {
        "length": "meter",
        "time": "second",
        "angle_default": "radian",
        "explicit_field_suffix_overrides_default": True,
    },
    "world_frame": {"handedness": "right_handed", "up_axis": "+Z"},
    "camera_pose": {
        "method": "look_at",
        "world_up": [0.0, 0.0, 1.0],
    },
    "camera_frame": {
        "right_axis": "+X",
        "up_axis": "+Y",
        "view_axis": "-Z",
    },
    "image": {
        "origin": "top_left",
        "horizontal_axis": "right",
        "vertical_axis": "down",
        "pixel_center": "integer_plus_half",
    },
}

CAMERA_PROJECTION_CONTRACT = {
    "model": "perspective_pinhole",
    "focal_length_source": "metadata.visual.camera.focal_length_mm",
    "sensor_width_source": "metadata.visual.camera.sensor_width_mm",
    "blender_sensor_fit": "AUTO",
    "effective_sensor_fit": "HORIZONTAL",
    "principal_point": "image_center",
    "camera_shift_xy": [0.0, 0.0],
    "pixel_aspect_xy": [1.0, 1.0],
}

TEMPORAL_CONTRACT = {
    "interval": "closed_[0,duration_s]",
    "frame_count_rule": "round(duration_s*output_fps)+1",
    "trajectory_time_rule": "time_s[k]=k/output_fps",
    "trajectory_video_alignment": "trajectory_index_k_equals_zero_based_video_frame_k",
    "mask_alignment": "trajectory_index_k_equals_mask_frame_{k+1:04d}.png",
}

TRAJECTORY_CONTRACT = {
    "frame_axis": 0,
    "object_axis": 1,
    "object_axis_ids": "trajectory.npz:object_ids",
    "object_metadata_binding": "metadata.physics.objects[].object_id",
    "reference_frame": "world",
    "float_dtype": "float64",
    "contact_count_dtype": "int32",
    "contact_count": "adapter_normalized_total_contact_points_involving_the_object",
    "quaternion": {
        "component_order": "wxyz",
        "equivalence": "q_and_neg_q_represent_the_same_rotation",
        "stored_sign": "frame_zero_matches_canonical_initial_sign_then_adjacent_dot_nonnegative",
    },
}

MASK_CONTRACT = {
    "encoding": "grayscale_png_uint8",
    "value_range": [0, 255],
    "signal": "antialiased_coverage_alpha",
    "occlusion_policy": "unoccluded_dynamic_silhouette",
    "static_scene_visibility": "hidden",
    "nonselected_dynamic_object_visibility": "hidden",
    "resolution_source": "render_contract.resolution",
    "path_layout": "masks/{object_id}/frame_{one_based_frame:04d}.png",
    "cross_modal_key": "object_id",
}

TEXT_CONTRACT = {
    "language": "en",
    "caption": "metadata.text.caption",
    "object_mentions": "metadata.text.object_mentions",
    "mention_identity": "object_id",
    "span_field": "char_span",
    "span_indexing": "zero_based_unicode_code_point_offsets",
    "span_interval": "half_open_[start,end)",
    "surface_text_rule": "caption[char_span[0]:char_span[1]]",
    "internal_asset_ids_in_caption": "forbidden",
    "valid_object_mention_coverage": "at_least_one_mention_per_object_valid_true_object",
    "mention_surface_label_binding": (
        "caption[char_span[0]:char_span[1]]_equals_the_plus_"
        "metadata.semantics.objects[object_id].semantic_label"
    ),
}

COLLISION_PROXY_CONTRACT = {
    "tag_field": "type",
    "sphere": {"radius_m": "radius"},
    "cuboid": {"size_m": "full_edge_lengths_xyz"},
    "cylinder": {
        "size_m": "diameter_x_diameter_y_height",
        "radius_rule": "max(size_m[0],size_m[1])/2",
    },
    "compound": {
        "colliders": "ordered_local_colliders",
        "collider_tag_field": "shape",
        "size_m": "full_local_dimensions_xyz",
        "box_size_m": "full_edge_lengths_xyz",
        "sphere_size_m": "repeated_diameter_xyz",
        "cylinder_size_m": "diameter_x_diameter_y_height",
        "position_m": "object_local_translation",
        "rotation_euler_degrees": {
            "input_order": "xyz",
            "active_rotation_matrix": "Rz@Ry@Rx",
        },
    },
    "inertia_diagonal_kg_m2": {
        "frame": "object_body_frame",
        "center_of_mass": "object_body_origin",
    },
}

OBJECT_VISUAL_CONTRACT = {
    "path": "metadata.physics.objects[].visual",
    "optional": True,
    "appearance_source_union": {
        "embedded_asset": "metadata.physics.objects[].asset_id",
        "explicit_base_color": "metadata.physics.objects[].visual.base_color_srgb_rgba",
        "fixture_material_template": "metadata.physics.objects[].visual.material_template",
    },
    "base_color_field": "base_color_srgb_rgba",
    "material_template_fields": ["source_fixture_asset_id", "source_object_name"],
    "one_effective_source_required": True,
    "top_level_dynamic_object_material": "forbidden",
}

OBJECT_DYNAMICS_CONTRACT = {
    "contact_processing_threshold": "metadata.physics.objects[].material.contact_processing_threshold_m",
    "ccd_swept_sphere_radius": "metadata.physics.objects[].ccd_swept_sphere_radius_m",
    "ccd_optional": True,
    "generic_rigid_ccd_rule": "0.22*min(source_generation_metadata.simulation.objects[].geometry.size_m)",
}

FIXTURE_BINDING_CONTRACT = {
    "content_hash": "metadata.physics.fixture.sha256",
    "path_template": "fixtures/{sha256}.json",
    "catalog_path_in_sample_metadata": "forbidden",
}

OBJECT_SEMANTICS_CONTRACT = {
    "path": "metadata.semantics.objects[]",
    "canonical_key": "object_id",
    "required_fields": ["object_id", "semantic_label"],
    "coverage": "exactly_one_semantic_record_per_metadata.physics.objects[]_object_id",
    "order": "same_as_metadata.physics.objects[]",
    "family_specific_annotations": "optional_fields_on_the_corresponding_semantic_object",
    "physics_object_semantic_label": "forbidden",
}

OBJECT_IDENTITY_CONTRACT = {
    "canonical_key": "object_id",
    "array_position": "not_a_cross_modal_identity",
    "numeric_instance_ids": "not_used",
    "trajectory_binding": "trajectory.npz:object_ids",
    "mask_binding": "masks/{object_id}",
    "semantics_binding": "metadata.semantics.objects[].object_id",
    "text_binding": "metadata.text.object_mentions[].object_id_and_char_span",
    "validity_binding": "metadata.physics.objects[].object_valid",
    "validity_semantics": "true_means_the_object_slot_is_present_and_all_cross_modal_bindings_are_defined",
    "dataset_object_axis": "variable_length_O_of_present_objects",
    "dataset_object_count": "1_to_3",
    "release_object_valid_policy": "every_listed_object_is_true",
    "training_object_axis": "fixed_length_3_created_by_downstream_compiler",
    "training_padding_policy": "missing_slots_have_object_valid_false",
}


@dataclass(frozen=True)
class PipelineSpec:
    name: str
    source_schema_version: str
    project_root: Path
    render_root: Path
    mask_root: Path


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


def write_pipeline_manifests(
    *,
    work: Path,
    specs: Mapping[str, PipelineSpec],
    grouped: Mapping[str, list[dict[str, Any]]],
    pipeline_schema: str,
    sample_schema: str,
) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for spec in sorted(specs.values(), key=lambda value: value.name):
        records = sorted(grouped[spec.name], key=lambda record: record["scene_id"])
        path = work / spec.name / "manifest.json"
        write_json(
            path,
            {
                "schema_version": pipeline_schema,
                "pipeline": spec.name,
                "source_schema_version": spec.source_schema_version,
                "sample_schema_version": sample_schema,
                "sample_count": len(records),
                "records": records,
            },
        )
        bindings[spec.name] = {
            "manifest": f"{spec.name}/manifest.json",
            "manifest_sha256": sha256(path),
        }
    return bindings


def collect_sample_attribution(
    metadata: Mapping[str, Any],
    used_asset_ids: set[str],
    used_hdri: dict[str, str],
) -> None:
    collect_asset_ids(metadata, used_asset_ids)
    environment = metadata.get("visual", {}).get("environment")
    if not isinstance(environment, dict):
        return
    name = str(environment["name"])
    digest = str(environment["sha256"])
    previous = used_hdri.setdefault(name, digest)
    if previous != digest:
        raise ValueError(f"two hashes for HDRI {name}")


def write_release_catalogs(
    *,
    work: Path,
    fixture_usage: Mapping[str, int],
    used_asset_ids: set[str],
    used_hdri: dict[str, str],
) -> dict[str, dict[str, Any]]:
    fixture_records = [
        {"sha256": digest, "usage_count": count}
        for digest, count in sorted(fixture_usage.items())
    ]
    fixture_path = work / "fixtures" / "manifest.json"
    write_json(
        fixture_path,
        {
            "schema_version": FIXTURE_CATALOG_SCHEMA,
            "entry_count": len(fixture_records),
            "records": fixture_records,
        },
    )
    for path in (work / "fixtures").glob("*.json"):
        if path != fixture_path:
            collect_asset_ids(load_json(path), used_asset_ids)
    attribution = build_attribution(PROJECT_ROOT, used_asset_ids, used_hdri)
    attribution_path = work / "assets" / "attribution_manifest.json"
    write_json(attribution_path, attribution)
    return {
        "fixture_catalog": {
            "schema_version": FIXTURE_CATALOG_SCHEMA,
            "manifest": "fixtures/manifest.json",
            "manifest_sha256": sha256(fixture_path),
            "entry_count": len(fixture_records),
        },
        "asset_attribution": {
            "manifest": "assets/attribution_manifest.json",
            "manifest_sha256": sha256(attribution_path),
            "record_count": int(attribution["record_count"]),
        },
    }


def release_contract_fields(
    *, render_contract: Mapping[str, Any], sample_schema: str
) -> dict[str, Any]:
    return {
        "storage_mode": (
            "materialized_compact_portable_release_with_content_addressed_fixtures"
        ),
        "render_contract": dict(render_contract),
        "coordinate_contract": COORDINATE_CONTRACT,
        "camera_projection_contract": CAMERA_PROJECTION_CONTRACT,
        "temporal_contract": TEMPORAL_CONTRACT,
        "trajectory_contract": TRAJECTORY_CONTRACT,
        "mask_contract": MASK_CONTRACT,
        "sample_layout_contract": SAMPLE_LAYOUT_CONTRACT,
        "text_contract": TEXT_CONTRACT,
        "collision_proxy_contract": COLLISION_PROXY_CONTRACT,
        "object_visual_contract": OBJECT_VISUAL_CONTRACT,
        "object_dynamics_contract": OBJECT_DYNAMICS_CONTRACT,
        "fixture_binding_contract": FIXTURE_BINDING_CONTRACT,
        "object_identity_contract": OBJECT_IDENTITY_CONTRACT,
        "object_semantics_contract": OBJECT_SEMANTICS_CONTRACT,
        "sample_schema_version": sample_schema,
        "trajectory_schema_version": TRAJECTORY_SCHEMA,
        "mask_manifest_schema_version": MASK_MANIFEST_SCHEMA,
        "fixture_schema_version": FIXTURE_SCHEMA,
    }


def validate_release_manifest_contract(
    manifest: Mapping[str, Any],
    *,
    view_schema: str,
    sample_schema: str,
    label: str,
    extra_fields: Iterable[str] = (),
) -> dict[str, Any]:
    render_contract = manifest.get("render_contract")
    if (
        not isinstance(render_contract, dict)
        or set(render_contract) != {"engine", "resolution", "video_encoding"}
        or not render_contract.get("engine")
        or not isinstance(render_contract.get("resolution"), list)
        or len(render_contract["resolution"]) != 2
        or any(int(value) <= 0 for value in render_contract["resolution"])
        or not isinstance(render_contract.get("video_encoding"), dict)
        or set(render_contract["video_encoding"]) != set(VIDEO_ENCODING_FIELDS)
        or int(render_contract["video_encoding"].get("fps", 0)) <= 0
    ):
        raise ValueError(f"{label} render contract is incomplete")
    expected_contract = release_contract_fields(
        render_contract=render_contract,
        sample_schema=sample_schema,
    )
    if (
        set(manifest) != COMMON_RELEASE_MANIFEST_FIELDS | set(extra_fields)
        or manifest.get("schema_version") != view_schema
        or any(manifest.get(key) != value for key, value in expected_contract.items())
    ):
        raise ValueError(f"not a canonical PhysSweep {label} release")
    return render_contract


def load_pipeline_records(
    *,
    output: Path,
    family: str,
    binding: Mapping[str, Any],
    pipeline_schema: str,
    sample_schema: str,
    label: str,
) -> list[dict[str, Any]]:
    relative_manifest = Path(family) / "manifest.json"
    if (
        set(binding) != {"manifest", "manifest_sha256"}
        or Path(str(binding.get("manifest", ""))) != relative_manifest
    ):
        raise ValueError(f"{label} pipeline binding differs: {family}")
    path = verified_file(
        output / relative_manifest,
        str(binding["manifest_sha256"]),
        f"{family} {label} pipeline manifest",
    )
    manifest = load_json(path)
    records = manifest.get("records", [])
    if (
        set(manifest) != PIPELINE_MANIFEST_FIELDS
        or manifest.get("schema_version") != pipeline_schema
        or manifest.get("pipeline") != family
        or not manifest.get("source_schema_version")
        or manifest.get("sample_schema_version") != sample_schema
        or not isinstance(records, list)
        or int(manifest.get("sample_count", -1)) != len(records)
    ):
        raise ValueError(f"{label} pipeline manifest differs: {family}")
    return records


def release_documents(
    release_project_root: Path,
    release_manifest: Path,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, Any],
    Path,
    dict[str, Any],
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
        load_json(base_path),
        metadata_path,
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
            mask_root=(
                raw.mask_root.resolve()
                if raw.mask_root.is_absolute()
                else (raw.project_root / raw.mask_root).resolve()
            ),
        )
        if spec.source_schema_version in by_schema:
            raise ValueError(f"duplicate pipeline schema: {spec.source_schema_version}")
        if spec.name in names:
            raise ValueError(f"duplicate pipeline name: {spec.name}")
        if not spec.render_root.is_dir():
            raise FileNotFoundError(f"render root: {spec.render_root}")
        if not spec.mask_root.is_dir():
            raise FileNotFoundError(f"mask root: {spec.mask_root}")
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
    resolved_scene_path = resolved_record_path(
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
    render_metadata_hash = str(render_record["metadata_sha256"])
    if render_metadata_hash == str(physics_record["metadata_sha256"]):
        render_metadata = None
    else:
        render_metadata_path = verified_file(
            project_path(spec.project_root, str(render_record["metadata_path"])),
            render_metadata_hash,
            f"{scene_id} render metadata",
        )
        bound = load_json(render_metadata_path)
        source_binding = bound.get("source_metadata", {})
        trajectory_binding = bound.get("trajectory", {})
        source_matches = (
            str(source_binding.get("sha256"))
            == str(physics_record["metadata_sha256"])
            and project_path(
                spec.project_root, str(source_binding.get("path", ""))
            ).resolve()
            == metadata
        )
        trajectory_matches = (
            isinstance(trajectory_binding, dict)
            and str(trajectory_binding.get("sha256"))
            == str(physics_record["trajectory_sha256"])
            and project_path(
                spec.project_root, str(trajectory_binding.get("path", ""))
            ).resolve()
            == trajectory
        )
        if not trajectory_matches:
            bound_physics = bound.get("physics", {})
            if not isinstance(bound_physics, dict):
                bound_physics = {}
            simulation_record_path = project_path(
                spec.project_root,
                str(bound_physics.get("simulation_record_path", "")),
            )
            simulation_record = (
                load_json(simulation_record_path)
                if simulation_record_path.is_file()
                else {}
            )
            path_fields = (
                "metadata_path",
                "trajectory_path",
                "resolved_scene_path",
                "audit_path",
            )
            value_fields = (
                "scene_id",
                "source_schema_version",
                "adapter_id",
                "metadata_sha256",
                "trajectory_sha256",
                "resolved_scene_sha256",
                "audit_sha256",
                "audit_passed",
                "adapter_audit_passed",
                "failed_checks",
            )
            trajectory_matches = (
                project_path(
                    spec.project_root,
                    str(bound_physics.get("trajectory_path", "")),
                ).resolve()
                == trajectory
                and all(
                    project_path(
                        spec.project_root,
                        str(simulation_record.get(key, "")),
                    ).resolve()
                    == project_path(
                        spec.project_root,
                        str(physics_record.get(key, "")),
                    ).resolve()
                    for key in path_fields
                )
                and all(
                    simulation_record.get(key) == physics_record.get(key)
                    for key in value_fields
                )
            )
        if not source_matches or not trajectory_matches:
            raise ValueError(f"render metadata is not bound to release physics: {scene_id}")
        render_metadata = bound
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
    resolved_scene = load_json(resolved_scene_path)
    render_config = source_metadata.get("render_request") or source_metadata.get(
        "render"
    )
    if not isinstance(render_config, dict):
        raise ValueError(f"render configuration is missing: {scene_id}")
    video_encoding = render_record.get("video_encoding")
    if not isinstance(video_encoding, dict) or set(video_encoding) != set(
        VIDEO_ENCODING_FIELDS
    ):
        raise ValueError(f"video encoding contract differs: {scene_id}")
    render_contract = {
        "engine": render_record.get("render_engine") or render_config.get("engine"),
        "resolution": render_config.get("resolution"),
        "video_encoding": {
            key: video_encoding[key] for key in VIDEO_ENCODING_FIELDS
        },
    }
    if (
        not render_contract["engine"]
        or not isinstance(render_contract["resolution"], list)
        or len(render_contract["resolution"]) != 2
        or any(int(value) <= 0 for value in render_contract["resolution"])
        or not isinstance(render_contract["video_encoding"], dict)
    ):
        raise ValueError(f"render contract is incomplete: {scene_id}")
    render_contract["resolution"] = [int(value) for value in render_contract["resolution"]]
    masks = spec.mask_root / scene_id
    if not masks.is_dir():
        raise FileNotFoundError(f"{scene_id} mask directory: {masks}")
    return {
        "source_metadata": source_metadata,
        "trajectory": trajectory,
        "resolved_scene": resolved_scene,
        "render_record": render_record,
        "render_metadata": render_metadata,
        "video": video,
        "masks": masks.resolve(),
        "render_contract": render_contract,
        "hashes": {
            "metadata_sha256": str(physics_record["metadata_sha256"]),
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
    if output.name != "base" or output.parent.name != "one_object":
        raise ValueError("base output must be one_object/base")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"base release already exists: {output}")
    specs = validate_pipeline_specs(pipeline_specs)
    (
        release_path,
        release,
        base,
        metadata_path,
        metadata,
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
        (work / "assets").mkdir()
        (work / "fixtures").mkdir()
        (work / "fixture_assets").mkdir()
        grouped: dict[str, list[dict[str, Any]]] = {
            spec.name: [] for spec in specs.values()
        }
        fixture_usage: dict[str, int] = {}
        used_asset_ids: set[str] = set()
        used_hdri: dict[str, str] = {}
        billiards_templates = (
            load_billiards_templates(PROJECT_ROOT)
            if any(spec.name == "billiards" for spec in specs.values())
            else {}
        )
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
            compact, fixture_hash = materialize_base_sample(
                target=work / spec.name / scene_id,
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
            grouped[spec.name].append(compact)
            fixture_usage[fixture_hash] = fixture_usage.get(fixture_hash, 0) + 1
            collect_sample_attribution(
                load_json(work / spec.name / scene_id / "metadata.json"),
                used_asset_ids,
                used_hdri,
            )

        pipeline_bindings = write_pipeline_manifests(
            work=work,
            specs=specs,
            grouped=grouped,
            pipeline_schema=PIPELINE_SCHEMA,
            sample_schema=BASE_SAMPLE_SCHEMA,
        )
        catalogs = write_release_catalogs(
            work=work,
            fixture_usage=fixture_usage,
            used_asset_ids=used_asset_ids,
            used_hdri=used_hdri,
        )

        manifest = {
            "schema_version": VIEW_SCHEMA,
            "dataset_id": str(release["dataset_id"]),
            "sample_count": expected_count,
            **release_contract_fields(
                render_contract=render_contract,
                sample_schema=BASE_SAMPLE_SCHEMA,
            ),
            **catalogs,
            "provenance": {
                "sample_lineage_fields": {
                    "source_fixture_binding_sha256": "original_static_fixture_binding",
                    "source_generation_metadata_sha256": "original_generation_metadata",
                },
                "source_generation_release_metadata": {
                    "schema_version": str(metadata["schema_version"]),
                    "manifest_sha256": sha256(metadata_path),
                },
                "source_sweep_release": {
                    "manifest_sha256": sha256(release_path),
                },
            },
            "pipelines": pipeline_bindings,
        }
        write_json(work / "manifest.json", manifest)
        (work / "README.txt").write_text(
            "Canonical PhysSweep base release v14.\n"
            "metadata.json is the sample authority; object_id is the only cross-modal identity.\n"
            "physics.objects[] stores physical and appearance state; semantics.objects[] stores object labels and annotations.\n"
            "Both object arrays bind by object_id and use the same present-object order.\n"
            "The release stores variable-length O=1..3 present objects; fixed three-slot padding belongs to downstream training compilation.\n"
            "Every listed release object has object_valid=true; downstream padding slots use false.\n"
            "All families share one sample layout and one sample schema.\n"
            "Sample lineage retains generation metadata and original fixture-binding hashes.\n"
            "trajectory.npz uses [frame, object, ...] arrays with sign-continuous wxyz quaternions.\n"
            "masks/ contains single-channel antialiased unoccluded silhouettes.\n"
            "fixtures/ contains content-addressed static collision context.\n"
            "assets/attribution_manifest.json records source and content provenance.\n",
            encoding="utf-8",
        )
        verify_view(work)
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"base release appeared during build: {output}")
        work.replace(output)
    return verify_view(output)


def validate_trajectory_artifact(path: Path, metadata: dict[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != TRAJECTORY_FIELDS:
            raise ValueError(f"non-canonical trajectory fields: {path}")
        if str(np.asarray(archive["schema_version"]).item()) != TRAJECTORY_SCHEMA:
            raise ValueError(f"trajectory schema differs: {path}")
        object_ids = [str(value) for value in np.asarray(archive["object_ids"]).tolist()]
        frame_count = int(np.asarray(archive["time_s"]).shape[0])
        metadata_ids = [
            str(record["object_id"]) for record in metadata["physics"]["objects"]
        ]
        if object_ids != metadata_ids:
            raise ValueError(f"trajectory object axis differs: {path}")
        time = metadata["physics"]["time"]
        expected_frames = round(float(time["duration_s"]) * int(time["output_fps"])) + 1
        if frame_count != expected_frames:
            raise ValueError(f"trajectory frame count differs: {path}")


def validate_mask_artifacts(sample: Path, metadata: dict[str, Any]) -> None:
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
    if not masks.is_dir() or masks.is_symlink():
        raise ValueError("mask projection is not a materialized directory")
    expected_objects = [
        str(record["object_id"]) for record in metadata["physics"]["objects"]
    ]
    records = manifest.get("objects", [])
    if (
        [str(record.get("object_id")) for record in records] != expected_objects
        or any(set(record) != {"object_id", "frame_sha256"} for record in records)
    ):
        raise ValueError("mask manifest object axis differs")
    expected_entries = {safe_scene_id(record["object_id"]) for record in records}
    if {path.name for path in masks.iterdir()} != expected_entries:
        raise ValueError("mask projection object directories differ")
    frame_count = int(manifest["frame_count"])
    time = metadata["physics"]["time"]
    expected_frame_count = (
        round(float(time["duration_s"]) * int(time["output_fps"])) + 1
    )
    if frame_count != expected_frame_count:
        raise ValueError("mask and trajectory frame counts differ")
    for record in records:
        object_id = safe_scene_id(record["object_id"])
        if (masks / object_id).is_symlink() or not (masks / object_id).is_dir():
            raise ValueError("mask object artifact is not materialized")
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
            if path.is_symlink():
                raise ValueError("mask frame must be materialized")
            with Image.open(path) as image:
                if image.mode != "L":
                    raise ValueError("mask frame must be grayscale")


def verify_view(output: Path) -> dict[str, Any]:
    output = output.resolve()
    manifest = load_json(output / "manifest.json")
    render_contract = validate_release_manifest_contract(
        manifest,
        view_schema=VIEW_SCHEMA,
        sample_schema=BASE_SAMPLE_SCHEMA,
        label="base",
    )
    provenance = manifest.get("provenance", {})
    if (
        provenance.get("sample_lineage_fields")
        != {
            "source_fixture_binding_sha256": "original_static_fixture_binding",
            "source_generation_metadata_sha256": "original_generation_metadata",
        }
        or len(str(provenance.get("source_generation_release_metadata", {}).get("manifest_sha256", ""))) != 64
        or len(str(provenance.get("source_sweep_release", {}).get("manifest_sha256", ""))) != 64
    ):
        raise ValueError("base provenance contract is incomplete")

    fixture_binding = manifest.get("fixture_catalog", {})
    fixture_manifest_path = verified_file(
        output / "fixtures" / "manifest.json",
        str(fixture_binding.get("manifest_sha256", "")),
        "fixture catalog",
    )
    fixture_manifest = load_json(fixture_manifest_path)
    if (
        fixture_binding.get("schema_version") != FIXTURE_CATALOG_SCHEMA
        or fixture_binding.get("manifest") != "fixtures/manifest.json"
        or fixture_manifest.get("schema_version") != FIXTURE_CATALOG_SCHEMA
        or int(fixture_binding.get("entry_count", -1))
        != int(fixture_manifest.get("entry_count", -2))
        or int(fixture_manifest.get("entry_count", -1))
        != len(fixture_manifest.get("records", []))
    ):
        raise ValueError("fixture catalog contract differs")
    fixture_hashes = set()
    for record in fixture_manifest["records"]:
        if set(record) != {"sha256", "usage_count"} or int(record["usage_count"]) <= 0:
            raise ValueError("fixture catalog record differs")
        digest = str(record["sha256"])
        verified_file(output / "fixtures" / f"{digest}.json", digest, "fixture")
        fixture_hashes.add(digest)
    attribution_binding = manifest.get("asset_attribution", {})
    attribution_path = verified_file(
        output / "assets" / "attribution_manifest.json",
        str(attribution_binding.get("manifest_sha256", "")),
        "asset attribution",
    )
    attribution = load_json(attribution_path)
    if (
        attribution_binding.get("manifest") != "assets/attribution_manifest.json"
        or int(attribution_binding.get("record_count", -1))
        != int(attribution.get("record_count", -2))
    ):
        raise ValueError("asset attribution contract differs")

    count = 0
    scene_ids: set[str] = set()
    group_ids: set[str] = set()
    observed_fixture_usage: dict[str, int] = {}
    expected_top = {
        "manifest.json", "README.txt", "assets", "fixtures", "fixture_assets"
    }
    for family, binding in manifest["pipelines"].items():
        family = safe_scene_id(family)
        expected_top.add(family)
        records = load_pipeline_records(
            output=output,
            family=family,
            binding=binding,
            pipeline_schema=PIPELINE_SCHEMA,
            sample_schema=BASE_SAMPLE_SCHEMA,
            label="base",
        )
        expected_samples = set()
        for record in records:
            if set(record) != {
                "scene_id", "metadata_sha256",
            }:
                raise ValueError(f"pipeline record fields differ: {family}")
            scene_id = safe_scene_id(record["scene_id"])
            if scene_id in scene_ids:
                raise ValueError(f"duplicate base identity: {scene_id}")
            scene_ids.add(scene_id)
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
            group_id = safe_scene_id(summary["group_id"])
            if (
                summary["scene_id"] != scene_id
                or summary["family"] != family
            ):
                raise ValueError(f"metadata identity differs: {scene_id}")
            if group_id in group_ids:
                raise ValueError(f"duplicate base group identity: {group_id}")
            group_ids.add(group_id)
            fixture_hash = str(metadata["physics"]["fixture"]["sha256"])
            if fixture_hash not in fixture_hashes:
                raise ValueError(f"sample fixture is absent from catalog: {scene_id}")
            observed_fixture_usage[fixture_hash] = (
                observed_fixture_usage.get(fixture_hash, 0) + 1
            )
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
            validate_trajectory_artifact(trajectory, metadata)
            video_binding = metadata["artifacts"]["video"]
            video = verified_file(
                sample / "video.mp4",
                str(video_binding["sha256"]),
                f"{scene_id} video",
            )
            if video.is_symlink():
                raise ValueError(f"video must be materialized: {scene_id}")
            validate_mask_artifacts(sample, metadata)
            if {path.name for path in sample.iterdir()} != SAMPLE_ENTRIES:
                raise ValueError(f"unexpected base sample files: {scene_id}")
            count += 1
        actual_samples = {
            path.name for path in (output / family).iterdir() if path.is_dir()
        }
        if actual_samples != expected_samples:
            raise ValueError(f"pipeline sample directories differ: {family}")
    if {path.name for path in output.iterdir()} != expected_top:
        raise ValueError("unexpected base root entries")
    if count != int(manifest["sample_count"]):
        raise ValueError("base release totals differ")
    expected_fixture_usage = {
        str(record["sha256"]): int(record["usage_count"])
        for record in fixture_manifest["records"]
    }
    if observed_fixture_usage != expected_fixture_usage:
        raise ValueError("fixture usage counts differ")
    symlinks = [path for path in output.rglob("*") if path.is_symlink()]
    if symlinks:
        raise ValueError(f"base release contains symlinks: {symlinks[:3]}")
    return {
        "schema_version": AUDIT_SCHEMA,
        "view": str(output),
        "sample_count": count,
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
    parser.add_argument(
        "--pipeline",
        nargs=5,
        action="append",
        metavar=("NAME", "SOURCE_SCHEMA", "PROJECT_ROOT", "RENDER_ROOT", "MASK_ROOT"),
        help="Repeat once per release pipeline.",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, _ = one_object_release_roots(args.release_root)
    if args.verify_only:
        result = verify_view(output)
    else:
        if args.release_project_root is None or args.release_manifest is None:
            raise SystemExit(
                "--release-project-root and --release-manifest are required when building"
            )
        specs = [
            PipelineSpec(
                name,
                schema,
                Path(root),
                Path(render_root),
                Path(mask_root),
            )
            for name, schema, root, render_root, mask_root in (args.pipeline or [])
        ]
        result = build_view(
            release_project_root=args.release_project_root,
            release_manifest=args.release_manifest,
            output=output,
            pipeline_specs=specs,
        )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
