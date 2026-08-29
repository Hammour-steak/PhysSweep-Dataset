#!/usr/bin/env python3
"""Bind deterministic camera, lighting, and render paths to rigid trajectories."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json_atomic as write_json
from tools.dataset_contract.object_identity_contract import (
    attach_object_identity,
    require_simulation_objects,
)

from tools.assets.environment_collision import validate_environment_binding
from tools.dataset_contract.trajectory_contract import object_trajectory_view


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_RULES_PATH = PROJECT_ROOT / "configs/one_object_sampling_rules.json"
SUPPORTED_DYNAMIC_OBJECT_COUNTS = (1, 2)

from tools.rendering.camera_solver import solve_camera

def shadow_readable_lighting(metadata: dict[str, Any]) -> dict[str, Any]:
    objects = require_simulation_objects(
        metadata, SUPPORTED_DYNAMIC_OBJECT_COUNTS, __name__
    )
    object_sizes = [
        np.asarray(obj["geometry"]["size_m"], dtype=np.float64)
        for obj in objects
    ]
    footprint_m = max(float(max(size[0], size[1])) for size in object_sizes)
    thickness_m = min(float(min(size)) for size in object_sizes)
    key_size_m = float(np.clip(4.5 * footprint_m, 0.95, 1.60))
    key_energy_w = min(460.0, 160.0 * key_size_m**2)
    fill_energy_w = min(55.0, 0.18 * key_energy_w)
    contact_shadow_bias_m = float(np.clip(0.05 * thickness_m, 0.0005, 0.004))
    contact_shadow_distance_m = float(np.clip(2.0 * footprint_m, 0.20, 0.60))
    return {
        "rule_version": "object_scale_shadow_readability_v1",
        "object_footprint_m": round(footprint_m, 6),
        "object_thickness_m": round(thickness_m, 6),
        "key_size_m": round(key_size_m, 6),
        "key_energy_w": round(key_energy_w, 6),
        "key_energy_per_square_meter": 160.0,
        "fill_energy_w": round(fill_energy_w, 6),
        "contact_shadow_bias_m": round(contact_shadow_bias_m, 6),
        "contact_shadow_distance_m": round(contact_shadow_distance_m, 6),
        "contact_shadow_thickness": 0.05,
    }


def frozen_environment_binding(
    metadata: dict[str, Any], camera: dict[str, Any]
) -> dict[str, Any]:
    """Bind lighting around geometry whose visual/collision pose is already frozen."""

    binding = validate_environment_binding(metadata)
    scene_anchor = np.asarray(
        camera.get("diagnostics", {}).get("motion_target_m", camera["target_m"]),
        dtype=np.float64,
    )
    camera_position = np.asarray(camera["position_m"], dtype=np.float64)
    outward = camera_position[:2] - scene_anchor[:2]
    outward /= np.linalg.norm(outward)
    lateral = np.asarray([-outward[1], outward[0]])
    key_xy = scene_anchor[:2] + outward * 1.10 + lateral * 1.35
    fill_xy = scene_anchor[:2] + outward * 0.65 - lateral * 1.25
    lighting = shadow_readable_lighting(metadata)
    return {
        "profile_id": str(binding["profile_id"]),
        "environment_binding_sha256": str(binding["binding_sha256"]),
        "collision_authority": (
            "frozen_static_environment_proxy_plus_analytic_action_surface"
        ),
        "static_background_objects": copy.deepcopy(binding["visual_objects"]),
        "shadow_readability_rule": lighting,
        "key_light": {
            "type": "AREA",
            "position_m": [
                float(key_xy[0]),
                float(key_xy[1]),
                float(scene_anchor[2] + 2.5),
            ],
            "target_m": [float(value) for value in scene_anchor],
            "energy_w": lighting["key_energy_w"],
            "size_m": lighting["key_size_m"],
            "cast_shadow": True,
            "contact_shadow": True,
            "contact_shadow_bias_m": lighting["contact_shadow_bias_m"],
            "contact_shadow_distance_m": lighting["contact_shadow_distance_m"],
            "contact_shadow_thickness": lighting["contact_shadow_thickness"],
        },
        "fill_light": {
            "type": "AREA",
            "position_m": [
                float(fill_xy[0]),
                float(fill_xy[1]),
                float(scene_anchor[2] + 1.65),
            ],
            "target_m": [float(value) for value in scene_anchor],
            "energy_w": lighting["fill_energy_w"],
            "size_m": 3.0,
            "cast_shadow": False,
            "contact_shadow": False,
        },
    }


def resolve_render_request(
    metadata: dict[str, Any],
    resolution: tuple[int, int] | None,
    samples: int | None,
) -> tuple[tuple[int, int], int]:
    render_request = metadata["render_request"]
    resolved_resolution = (
        resolution
        if resolution is not None
        else tuple(int(value) for value in render_request["resolution"])
    )
    if len(resolved_resolution) != 2 or min(resolved_resolution) <= 0:
        raise ValueError("render resolution must contain two positive values")
    resolved_samples = int(
        samples if samples is not None else render_request["samples"]
    )
    if resolved_samples <= 0:
        raise ValueError("render samples must be positive")
    return (int(resolved_resolution[0]), int(resolved_resolution[1])), resolved_samples


def bind_scene(
    root: Path,
    metadata_path: Path,
    simulation_record_path: Path,
    trajectory_path: Path,
    output_root: Path,
    rules: dict[str, Any],
    resolution: tuple[int, int] | None,
    samples: int | None,
) -> dict[str, Any]:
    metadata = load_json(metadata_path)
    resolution, samples = resolve_render_request(metadata, resolution, samples)
    scene_id = str(metadata["scene_id"])
    simulation_record = load_json(simulation_record_path)
    if simulation_record.get("schema_version") not in {
        "physweep_pybullet_simulation_record_v1",
        "physweep_dispatched_simulation_record_v1",
    }:
        raise ValueError("unsupported PyBullet simulation record")
    record_scene = simulation_record.get("scene_id")
    expected_scene = scene_id
    if record_scene != expected_scene:
        raise ValueError("simulation record scene id mismatch")
    if sha256(metadata_path) != simulation_record.get("metadata_sha256"):
        raise ValueError("source metadata changed after simulation")
    if Path(simulation_record["metadata_path"]).resolve() != metadata_path.resolve():
        raise ValueError("simulation record metadata path mismatch")
    if Path(simulation_record["trajectory_path"]).resolve() != trajectory_path.resolve():
        raise ValueError("simulation record trajectory path mismatch")
    if sha256(trajectory_path) != simulation_record.get("trajectory_sha256"):
        raise ValueError("trajectory changed after simulation")
    audit_path = Path(simulation_record["audit_path"])
    if sha256(audit_path) != simulation_record.get("audit_sha256"):
        raise ValueError("trajectory audit changed after simulation")
    with np.load(trajectory_path) as source:
        trajectory = {key: source[key] for key in source.files}
    trajectory = object_trajectory_view(metadata, trajectory)
    camera = solve_camera(metadata, trajectory, rules)
    environment = frozen_environment_binding(metadata, camera)
    support_static_objects = copy.deepcopy(
        metadata["simulation"]["support"]["colliders"]
    )
    support_visual = metadata["appearance"].get(
        "support_visual",
        {
            "id": "procedural_support_proxy",
            "visual_type": "procedural_proxy",
        },
    )
    support_visual_binding = {
        "requested_profile": str(support_visual["id"]),
        "requested_type": str(support_visual["visual_type"]),
        "selected_type": "procedural_proxy",
        "fallback_reason": None,
        "selection_phase": "source_metadata_before_simulation",
    }
    support = metadata["simulation"]["support"]
    scene_composition = metadata["appearance"]["scene_visual"].get(
        "composition"
    )
    integrated_ground = (
        isinstance(scene_composition, dict)
        and str(scene_composition.get("review_status")) == "approved"
        and str(scene_composition.get("composition_mode")) == "integrated_ground"
    )
    if integrated_ground:
        scene_class = str(support["scene_class"])
        hidden_roles = (
            {"primary_support", "environment_floor"}
            if scene_class == "ground_flat"
            else {"environment_floor"}
        )
        for record in support_static_objects:
            if str(record["role"]) in hidden_roles:
                record["visible"] = False
        if str(support_visual["visual_type"]) == "mesh_support":
            raise ValueError(
                "integrated ground metadata cannot bind a second support mesh"
            )
    visual_geometry = support.get("visual_geometry")
    if (
        str(support_visual["visual_type"]) == "procedural_proxy"
        and isinstance(visual_geometry, dict)
        and str(visual_geometry.get("primitive")) == "solid_wedge"
    ):
        for record in support_static_objects:
            if bool(record.get("render_replaced_by_solid_wedge", False)):
                record["visible"] = False
        support_static_objects.append(
            {
                "id": "solid_ramp_wedge",
                "primitive": "solid_wedge",
                "role": "render_only_support",
                "material_role": "support_surface",
                "structure_material_role": "support_structure",
                "size_xy_m": [
                    float(value) for value in visual_geometry["size_xy_m"]
                ],
                "base_z_m": float(visual_geometry["base_z_m"]),
                "high_top_z_m": float(visual_geometry["high_top_z_m"]),
                "slope_axis": str(visual_geometry["slope_axis"]),
                "visible": True,
                "collision_enabled": False,
            }
        )
        support_visual_binding["selected_type"] = "procedural_solid_wedge"
    if str(support_visual["visual_type"]) == "mesh_support":
        if str(support["topology"]) != "flat_surface":
            raise ValueError(
                f"support mesh requires a flat surface: {support['semantic_type']}"
            )
        binding = support.get("exact_static_binding")
        if binding is None:
            raise ValueError(
                f"mesh support was not frozen before simulation: {expected_scene}"
            )
        if str(binding["asset_id"]) != str(support_visual["asset_id"]):
            raise ValueError(
                f"support visual and collision assets differ: {expected_scene}"
            )
        if str(support["collision_authority"]) != "exact_static_proxy":
            raise ValueError(
                f"mesh support lacks exact collision authority: {expected_scene}"
            )
        support_visual_binding["selected_type"] = "exact_static_proxy"
        support_visual_binding["binding_sha256"] = str(
            binding["binding_sha256"]
        )
        for record in support_static_objects:
            if str(record["role"]) in {
                "primary_support",
                "support_structure",
            }:
                record["visible"] = False
        support_static_objects.append(
            {
                "id": "support_visual_mesh",
                "primitive": "exact_support_visual",
                "role": "authoritative_support_visual",
                "material_role": "support_surface",
                "binding": copy.deepcopy(binding),
                "material_policy": str(support_visual["material_policy"]),
                "requires_image_texture": bool(
                    support_visual["requires_image_texture"]
                ),
                "license": str(support_visual["license"]),
                "visible": True,
                "collision_enabled": False,
                "occludes_camera": False,
            }
        )
    frame_count = int(metadata["simulation"]["time"]["frame_count"])
    inspection_frames = sorted({1, max(1, (frame_count + 1) // 2), frame_count})
    bound = copy.deepcopy(metadata)
    attach_object_identity(
        bound,
        trajectory_path=str(trajectory_path.relative_to(root)),
        mask_path=str((output_root / "masks" / scene_id).relative_to(root)),
    )
    bound["schema_version"] = "physweep_pybullet_rigid_bound_metadata_v1"
    bound["source_metadata"] = {
        "path": str(metadata_path.relative_to(root)),
        "sha256": sha256(metadata_path),
    }
    bound["trajectory"] = {
        "path": str(trajectory_path.relative_to(root)),
        "sha256": sha256(trajectory_path),
    }
    bound["simulation_record"] = {
        "path": str(simulation_record_path.relative_to(root)),
        "sha256": sha256(simulation_record_path),
    }
    bound["visualization"] = {
        "binding_version": "physweep_pybullet_visual_binding_v3",
        "support_visual_binding": support_visual_binding,
        "camera": camera,
        "materials": metadata["appearance"]["materials"],
        "hdri": metadata["appearance"]["hdri"],
        "static_objects": support_static_objects,
        "environment": environment,
        "render": {
            "engine": "BLENDER_EEVEE",
            "resolution_x": int(resolution[0]),
            "resolution_y": int(resolution[1]),
            "resolution_percentage": 100,
            "samples": int(samples),
            "fps": int(metadata["simulation"]["time"]["output_fps"]),
            "frame_start": 1,
            "frame_end": frame_count,
            "video_path": str((output_root / "videos" / f"{scene_id}.mp4").relative_to(root)),
            "inspection_frame_dir": str((output_root / "frames" / scene_id).relative_to(root)),
            "instance_mask_dir": str((output_root / "masks" / scene_id).relative_to(root)),
            "inspection_frames": inspection_frames,
            "color_management": {
                "view_transform": "Filmic",
                "look": "Medium High Contrast",
                "exposure": 0.0,
                "gamma": 1.0,
            },
        },
    }
    output_path = output_root / "metadata" / f"{scene_id}.json"
    write_json(output_path, bound)
    return {
        "scene_id": scene_id,
        "metadata_path": str(output_path.relative_to(root)),
        "metadata_sha256": sha256(output_path),
        "trajectory_path": str(trajectory_path.relative_to(root)),
        "camera_diagnostics": camera["diagnostics"],
    }


def parse_resolution(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("resolution must look like 640x360")
    width, height = [int(item) for item in parts]
    if min(width, height) <= 0:
        raise argparse.ArgumentTypeError("resolution must be positive")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--resolution",
        type=parse_resolution,
        help="Explicit override; otherwise inherit render_request.resolution.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Explicit override; otherwise inherit render_request.samples.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def manifest_rules_path(root: Path, manifest: dict[str, Any]) -> Path:
    declared = manifest.get("rules_path")
    path = root / str(declared) if declared else ACTIVE_RULES_PATH
    path = path.resolve()
    expected = manifest.get("rules_sha256")
    if expected is not None and sha256(path) != str(expected):
        raise ValueError(f"rules hash mismatch: {path}")
    return path


def binding_samples(
    root: Path, manifest: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resolve a sampled or audited-physics manifest into binding jobs."""

    samples = manifest.get("samples")
    if isinstance(samples, list):
        return manifest, list(samples)
    if manifest.get("schema_version") != "physweep_pybullet_batch_record_v1":
        raise ValueError("binder requires a sampling or physics batch manifest")
    source_value = Path(str(manifest.get("source_manifest", "")))
    source_path = source_value if source_value.is_absolute() else root / source_value
    source_path = source_path.resolve()
    source_path.relative_to(root)
    source = load_json(source_path)
    source_samples = source.get("samples")
    records = manifest.get("records")
    if not isinstance(source_samples, list) or not isinstance(records, list):
        raise ValueError("physics batch manifest has no valid source samples")
    if int(manifest.get("sample_count", -1)) != len(records):
        raise ValueError("physics batch manifest count differs from its records")
    records_by_id = {str(record.get("scene_id", "")): record for record in records}
    source_ids = [str(sample.get("scene_id", "")) for sample in source_samples]
    if (
        "" in records_by_id
        or len(records_by_id) != len(records)
        or "" in source_ids
        or len(source_ids) != len(set(source_ids))
        or set(source_ids) != set(records_by_id)
    ):
        raise ValueError("sampling and physics manifests select different scenes")
    result = []
    for sample in source_samples:
        scene_id = str(sample["scene_id"])
        record = records_by_id[scene_id]
        if not record.get("ok") or not record.get("audit_passed"):
            raise ValueError(f"physics record is not audited: {scene_id}")
        source_metadata = (root / str(sample["metadata_path"])).resolve()
        record_metadata = Path(str(record["metadata_path"])).resolve()
        if (
            source_metadata != record_metadata
            or str(sample["metadata_sha256"]) != str(record["metadata_sha256"])
        ):
            raise ValueError(f"physics metadata binding mismatch for scene {scene_id}")
        trajectory_path = Path(str(record["trajectory_path"])).resolve()
        result.append(
            {
                **sample,
                "metadata_path": str(record_metadata),
                "simulation_record_path": str(
                    trajectory_path.with_name("simulation_record.json")
                ),
                "trajectory_path": str(trajectory_path),
            }
        )
    return source, result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    manifest = load_json(args.manifest.resolve())
    source_manifest, samples = binding_samples(root, manifest)
    rules_path = manifest_rules_path(root, source_manifest)
    rules = load_json(rules_path)
    output_root = args.output_root.resolve()
    if args.limit is not None:
        samples = samples[: args.limit]
    def sample_path(sample: dict[str, Any], key: str, fallback: Path) -> Path:
        value = sample.get(key)
        if value is None:
            return fallback
        path = Path(str(value))
        return path if path.is_absolute() else root / path

    jobs = [
        (
            root,
            (metadata_path := root / str(sample["metadata_path"])),
            sample_path(
                sample,
                "simulation_record_path",
                metadata_path.parent / "physics" / "simulation_record.json",
            ),
            sample_path(
                sample,
                "trajectory_path",
                metadata_path.parent / "physics" / "trajectory.npz",
            ),
            output_root,
            rules,
            args.resolution,
            args.samples,
        )
        for sample in samples
    ]
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if args.workers == 1:
        records = [bind_scene(*job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            records = list(executor.map(bind_scene, *zip(*jobs)))
    bound_manifest = {
        "schema_version": "physweep_pybullet_bound_manifest_v2",
        "source_manifest": str(args.manifest.resolve()),
        "output_root": str(output_root),
        "implementation": {
            "path": str(Path(__file__).resolve().relative_to(root)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "camera_rules": {
            "path": str(rules_path.relative_to(root)),
            "sha256": sha256(rules_path),
        },
        "sample_count": len(records),
        "samples": records,
    }
    write_json(output_root / "bound_manifest.json", bound_manifest)
    print(f"bound manifest: {output_root / 'bound_manifest.json'}")
    print(f"samples: {len(records)}")


if __name__ == "__main__":
    main()
