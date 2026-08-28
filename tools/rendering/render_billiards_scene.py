#!/usr/bin/env python3
"""Render a specialized one-ball or three-ball billiards scene."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import bpy
import mathutils
import numpy as np

if "bool" not in np.__dict__:
    np.bool = np.bool_  # type: ignore[attr-defined]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.rendering.render_asset_proxy_scene import (  # pylint: disable=wrong-import-position
    PROJECT_ROOT,
    add_environment,
    add_lighting,
    add_support,
    apply_hdri,
    configure_project_root,
    load_json,
    resolve,
    setup_scene,
    sha256,
    write_json,
)
from tools.dataset_contract.immutable_scene_contract import validate_simulation_record
from tools.assets.blender_asset_import import clear_scene
from tools.rendering.blender_scene import look_at
from tools.rendering.video_encoding import configure_h264_output, normalize_h264_container
from tools.dataset_contract.trajectory_contract import adapter_trajectory_view
from tools.rendering.specialized_render_evidence import (
    render_instance_mask_record,
    render_instance_masks,
    render_implementation,
)


def blender_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--inspection-frame-dir", type=Path)
    parser.add_argument("--mask-only", action="store_true")
    parser.add_argument("--instance-mask-dir", type=Path)
    return parser.parse_args(values)


def add_camera(binding: dict[str, Any]) -> dict[str, Any]:
    bpy.ops.object.camera_add(location=tuple(float(value) for value in binding["position_m"]))
    camera = bpy.context.object
    camera.name = "billiards_camera"
    camera.data.lens = float(binding["focal_length_mm"])
    camera.data.sensor_width = float(binding["sensor_width_mm"])
    camera.data.clip_start = 0.03
    camera.data.clip_end = 100.0
    look_at(camera, mathutils.Vector(binding["target_m"]))
    bpy.context.scene.camera = camera
    return dict(binding)


def hidden_ball_materials(meshes: list[Any], policy: dict[str, Any]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    names = {
        role: str(object_name)
        for role, object_name in policy["material_template_objects"].items()
    }
    hidden_prefixes = tuple(str(value).lower() for value in policy["hidden_prefixes"])
    support_prefixes = tuple(str(value).lower() for value in policy["support_prefixes"])
    for obj in meshes:
        lowered = obj.name.lower()
        if lowered.startswith(hidden_prefixes):
            obj.hide_render = True
            obj.hide_viewport = True
        elif not lowered.startswith(support_prefixes):
            raise ValueError(f"unclassified pool-table mesh component: {obj.name}")
        for role, object_name in names.items():
            if obj.name == object_name and obj.data.materials:
                selected.setdefault(role, obj.data.materials[0])
    if set(selected) != set(names):
        raise ValueError(f"pool asset does not expose expected ball materials: {sorted(selected)}")
    return selected


def add_balls(
    materials: dict[str, Any],
    trajectory: dict[str, np.ndarray],
    radius: float,
    roles: list[str],
) -> list[Any]:
    result = []
    if trajectory["position_m"].shape[1] != len(roles):
        raise ValueError("trajectory object count does not match metadata roles")
    for object_index, role in enumerate(roles):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=radius)
        ball = bpy.context.object
        ball.name = role
        ball.data.materials.append(materials[role])
        ball.rotation_mode = "QUATERNION"
        bevel = ball.modifiers.new("ball_surface_smooth", "BEVEL")
        bevel.width = radius * 0.015
        bevel.segments = 2
        bpy.ops.object.shade_smooth()
        for frame, (position, quaternion) in enumerate(
            zip(trajectory["position_m"][:, object_index], trajectory["quaternion_xyzw"][:, object_index]),
            start=1,
        ):
            ball.location = tuple(float(value) for value in position)
            x, y, z, w = [float(value) for value in quaternion]
            ball.rotation_quaternion = (w, x, y, z)
            ball.keyframe_insert(data_path="location", frame=frame)
            ball.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        if ball.animation_data and ball.animation_data.action:
            for curve in ball.animation_data.action.fcurves:
                for point in curve.keyframe_points:
                    point.interpolation = "LINEAR"
        result.append(ball)
    return result


def render(
    metadata_path: Path,
    video_path_override: Path | None = None,
    frame_dir_override: Path | None = None,
    *,
    mask_only: bool = False,
    instance_mask_dir: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    metadata = load_json(metadata_path)
    if metadata["schema_version"] != "physweep_billiards_scene_v4":
        raise ValueError("unsupported metadata schema")
    registry_path = resolve(metadata["registry"]["path"])
    if sha256(registry_path) != metadata["registry"]["sha256"]:
        raise ValueError("asset proxy registry hash mismatch")
    semantic_rules_path = resolve(metadata["semantic_rules"]["path"])
    if sha256(semantic_rules_path) != metadata["semantic_rules"]["sha256"]:
        raise ValueError("asset semantic rules hash mismatch")
    composition_rules_path = resolve(metadata["composition_rules"]["path"])
    if sha256(composition_rules_path) != metadata["composition_rules"]["sha256"]:
        raise ValueError("asset scene composition rules hash mismatch")
    visual_rules_path = resolve(metadata["visual_rules"]["path"])
    if sha256(visual_rules_path) != metadata["visual_rules"]["sha256"]:
        raise ValueError("visual sampling rules hash mismatch")
    catalog_path = resolve(metadata["physical_proxy_catalog"]["path"])
    if sha256(catalog_path) != metadata["physical_proxy_catalog"]["sha256"]:
        raise ValueError("physical proxy catalog hash mismatch")
    composition = next(
        record
        for record in load_json(composition_rules_path)["records"]
        if record["asset_id"] == metadata["assets"]["support_asset_id"]
    )
    if composition["sampling_status"] != "ready_specialized":
        raise ValueError("billiards support is not approved for specialized sampling")
    simulation_record, trajectory_path, _ = validate_simulation_record(
        root=PROJECT_ROOT,
        metadata_path=metadata_path,
        metadata=metadata,
    )
    with np.load(trajectory_path) as source:
        trajectory = {key: source[key] for key in source.files}
    trajectory = adapter_trajectory_view(trajectory)
    clear_scene()
    setup_scene(metadata)
    if not mask_only:
        apply_hdri(metadata["render"]["environment"])
    support_meshes = add_support(
        metadata["physics"]["static_support_binding"],
        include_all_source_meshes=True,
    )
    ball_materials = hidden_ball_materials(
        support_meshes, composition["component_policy"]
    )
    roles = [
        str(record["object_id"])
        for record in metadata["physics"]["initial_states"]
    ]
    balls = add_balls(
        ball_materials,
        trajectory,
        float(metadata["physics"]["ball_radius_m"]),
        roles,
    )
    if len(balls) != len(roles):
        raise ValueError("rendered ball count does not match the identity contract")
    camera = add_camera(metadata["camera"])
    if mask_only:
        if frame_dir_override is None or instance_mask_dir is None:
            raise ValueError(
                "mask-only rendering requires an inspection record directory and mask directory"
            )
        frame_dir = frame_dir_override.resolve()
        frame_dir.mkdir(parents=True, exist_ok=True)
        record = render_instance_mask_record(
            root=PROJECT_ROOT,
            metadata_path=metadata_path,
            metadata=metadata,
            camera=camera,
            dynamic_objects={role: [ball] for role, ball in zip(roles, balls)},
            mask_root=instance_mask_dir,
            renderer_path=Path(__file__),
        )
        write_json(frame_dir / "render_record.json", record)
        print(json.dumps(record, indent=2))
        return record
    add_environment(metadata["render"]["environment"], camera)
    add_lighting(
        mathutils.Vector(camera["target_m"]),
        2.2,
        metadata["render"]["environment"]["lighting"],
    )

    scene = bpy.context.scene
    frame_dir = (
        frame_dir_override.resolve()
        if frame_dir_override is not None
        else resolve(metadata["render"]["inspection_frame_dir"])
    )
    frame_dir.mkdir(parents=True, exist_ok=True)
    inspection_frames = [1, (scene.frame_end + 1) // 2, scene.frame_end]
    inspection_paths = []
    scene.render.image_settings.file_format = "PNG"
    for frame in inspection_frames:
        scene.frame_set(frame)
        path = frame_dir / f"frame_{frame:04d}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        inspection_paths.append(path)
    video_path = (
        video_path_override.resolve()
        if video_path_override is not None
        else resolve(metadata["render"]["video_path"])
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(video_path)
    video_encoding = configure_h264_output(
        scene,
        fps=int(metadata["physics"]["output_fps"]),
        frame_count=int(metadata["physics"]["frame_count"]),
    )
    render_samples = int(scene.eevee.taa_render_samples)
    bpy.ops.render.render(animation=True)
    normalize_h264_container(video_path)
    mask_path = metadata["object_identity"]["instance_masks"].get("path")
    instance_mask_output = None
    if instance_mask_dir is not None or (
        isinstance(mask_path, str) and bool(mask_path)
    ):
        instance_mask_output = render_instance_masks(
            root=PROJECT_ROOT,
            metadata=metadata,
            dynamic_objects={role: [ball] for role, ball in zip(roles, balls)},
            mask_root_override=instance_mask_dir,
        )
    record = {
        "schema_version": "physweep_billiards_render_record_v1",
        "scene_id": metadata["scene_id"],
        "metadata_sha256": sha256(metadata_path),
        "metadata_path": str(metadata_path),
        "render_output_overridden": bool(
            video_path_override is not None or frame_dir_override is not None
        ),
        "support_binding_sha256": metadata["physics"]["static_support_binding"][
            "binding_sha256"
        ],
        "video_path": str(video_path),
        "video_sha256": sha256(video_path),
        "inspection_frames": [str(path) for path in inspection_paths],
        "camera": camera,
        "render_engine": scene.render.engine,
        "render_samples": render_samples,
        "video_encoding": video_encoding,
        "instance_mask_output": instance_mask_output,
        "implementation": render_implementation(Path(__file__)),
        "wall_time_s": round(time.perf_counter() - started, 6),
    }
    write_json(frame_dir / "render_record.json", record)
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    args = blender_args()
    PROJECT_ROOT = args.root.resolve()
    configure_project_root(PROJECT_ROOT)
    render(
        args.metadata.resolve(),
        args.video_path,
        args.inspection_frame_dir,
        mask_only=args.mask_only,
        instance_mask_dir=args.instance_mask_dir,
    )
