#!/usr/bin/env python3
"""Render a metadata-bound formal passive-pinball scene."""

from __future__ import annotations

import argparse
import hashlib
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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_passive_pinball_scene import (  # pylint: disable=wrong-import-position
    SCHEMA_VERSION,
    _validate_metadata_files,
    load_json,
    sha256,
    write_json,
)
from immutable_scene_contract import validate_simulation_record
from render_asset_proxy_reviews import clear_scene, look_at
from trajectory_contract import adapter_trajectory_view
from video_encoding import configure_h264_output


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def blender_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--inspection-frame-dir", type=Path)
    return parser.parse_args(values)


def material(name: str, rgba: list[float], metallic: float = 0.0) -> Any:
    result = bpy.data.materials.new(name)
    result.diffuse_color = tuple(float(value) for value in rgba)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = tuple(float(value) for value in rgba)
    shader.inputs["Roughness"].default_value = 0.24 if metallic else 0.46
    shader.inputs["Metallic"].default_value = float(metallic)
    return result


def add_fixture(metadata: dict[str, Any]) -> list[Any]:
    result = []
    for collider in metadata["physics"]["fixture"]["colliders"]:
        if collider["shape"] == "box":
            bpy.ops.mesh.primitive_cube_add(
                size=2.0,
                location=tuple(float(value) for value in collider["position_m"]),
            )
            obj = bpy.context.object
            obj.scale = tuple(float(value) for value in collider["half_extents_m"])
        elif collider["shape"] == "cylinder":
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=32,
                radius=float(collider["radius_m"]),
                depth=float(collider["length_m"]),
                location=tuple(float(value) for value in collider["position_m"]),
            )
            obj = bpy.context.object
            bevel = obj.modifiers.new("peg_edge_bevel", "BEVEL")
            bevel.width = min(float(collider["radius_m"]) * 0.12, 0.003)
            bevel.segments = 2
            bpy.ops.object.shade_smooth()
        else:
            raise ValueError(f"unsupported passive-pinball visual collider: {collider['shape']}")
        obj.name = str(collider["id"])
        obj.rotation_mode = "QUATERNION"
        x, y, z, w = [float(value) for value in collider["orientation_quaternion_xyzw"]]
        obj.rotation_quaternion = mathutils.Quaternion((w, x, y, z))
        obj.data.materials.append(
            material(f"{collider['id']}_material", collider["color_rgba"])
        )
        result.append(obj)
    return result


def add_ball(metadata: dict[str, Any], trajectory: dict[str, np.ndarray]) -> Any:
    dynamic = metadata["simulation"]["objects"][0]
    positions = np.asarray(trajectory["position_m"], dtype=np.float64)
    orientations = np.asarray(trajectory["quaternion_xyzw"], dtype=np.float64)
    if positions.shape[1:] != (1, 3) or orientations.shape[1:] != (1, 4):
        raise ValueError("passive-pinball trajectory must contain exactly one ball")
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=64,
        ring_count=32,
        radius=float(dynamic["collision_proxy"]["radius_m"]),
    )
    ball = bpy.context.object
    ball.name = str(dynamic["object_id"])
    ball.data.materials.append(
        material("pinball_material", dynamic["visual"]["color_rgba"], metallic=0.72)
    )
    ball.rotation_mode = "QUATERNION"
    bpy.ops.object.shade_smooth()
    for frame, (position, quaternion) in enumerate(
        zip(positions[:, 0], orientations[:, 0]),
        start=1,
    ):
        ball.location = tuple(float(value) for value in position)
        x, y, z, w = [float(value) for value in quaternion]
        ball.rotation_quaternion = mathutils.Quaternion((w, x, y, z))
        ball.keyframe_insert(data_path="location", frame=frame)
        ball.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    if ball.animation_data and ball.animation_data.action:
        for curve in ball.animation_data.action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"
    return ball


def add_camera(binding: dict[str, Any]) -> dict[str, Any]:
    bpy.ops.object.camera_add(location=tuple(float(value) for value in binding["position_m"]))
    camera = bpy.context.object
    camera.name = "passive_pinball_camera"
    camera.data.lens = float(binding["focal_length_mm"])
    camera.data.sensor_width = float(binding["sensor_width_mm"])
    camera.data.clip_start = 0.03
    camera.data.clip_end = 100.0
    look_at(camera, mathutils.Vector(binding["target_m"]))
    bpy.context.scene.camera = camera
    return dict(binding)


def add_lights(metadata: dict[str, Any]) -> None:
    target = mathutils.Vector(metadata["camera"]["target_m"])
    for binding in metadata["render"]["lights"]:
        light_data = bpy.data.lights.new(str(binding["id"]), str(binding["type"]))
        light_data.energy = float(binding["energy_w"])
        light_data.size = float(binding["size_m"])
        light_data.color = tuple(float(value) for value in binding["color_rgb"])
        light = bpy.data.objects.new(str(binding["id"]), light_data)
        bpy.context.collection.objects.link(light)
        light.location = tuple(float(value) for value in binding["position_m"])
        look_at(light, target)


def configure_scene(metadata: dict[str, Any]) -> None:
    scene = bpy.context.scene
    render = metadata["render"]
    time_binding = metadata["simulation"]["time"]
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3.0
    scene.eevee.gtao_factor = 1.2
    scene.render.resolution_x = int(render["resolution"][0])
    scene.render.resolution_y = int(render["resolution"][1])
    scene.render.resolution_percentage = 100
    scene.render.fps = int(time_binding["output_fps"])
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.frame_start = 1
    scene.frame_end = int(time_binding["frame_count"])
    scene.view_settings.look = "Medium High Contrast"
    world = bpy.data.worlds.new("passive_pinball_world") if scene.world is None else scene.world
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (
        *[float(value) for value in render["world_color_rgb"]],
        1.0,
    )
    background.inputs["Strength"].default_value = 0.32


def render_instance_masks(
    metadata: dict[str, Any],
    ball: Any,
    fixture_objects: list[Any],
    frame_dir: Path,
) -> dict[str, Any]:
    scene = bpy.context.scene
    object_id = str(metadata["simulation"]["objects"][0]["object_id"])
    declared = metadata["object_identity"]["instance_masks"].get("path")
    if declared:
        declared_root = Path(str(declared))
        if declared_root.is_absolute():
            raise ValueError("passive-pinball mask path must be project-relative")
        mask_root = (PROJECT_ROOT / declared_root).resolve()
    else:
        mask_root = frame_dir.parent.parent / "masks" / str(metadata["scene_id"])
    if PROJECT_ROOT.resolve() not in mask_root.parents:
        raise ValueError("passive-pinball mask path is outside the project")
    mask_dir = mask_root / object_id
    mask_dir.mkdir(parents=True, exist_ok=True)
    for obj in fixture_objects:
        obj.hide_render = True
    mask_material = bpy.data.materials.new("pinball_mask_material")
    mask_material.use_nodes = True
    nodes = mask_material.node_tree.nodes
    links = mask_material.node_tree.links
    for node in list(nodes):
        nodes.remove(node)
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    ball.data.materials.clear()
    ball.data.materials.append(mask_material)
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    paths = []
    for frame in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(frame)
        path = mask_dir / f"frame_{frame:04d}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        paths.append(path)
    probes = [paths[0], paths[len(paths) // 2], paths[-1]]
    nonempty = 0
    for path in probes:
        image = bpy.data.images.load(str(path), check_existing=False)
        alpha = np.asarray(image.pixels[:], dtype=np.float32)[3::4]
        nonempty += int(bool(alpha.size and float(alpha.max()) > 0.01))
        bpy.data.images.remove(image)
    if nonempty != len(probes):
        raise RuntimeError("passive-pinball instance mask is empty")
    mask_manifest_path = mask_root / "mask_manifest.json"
    mask_manifest = {
        "schema_version": "physweep_instance_mask_manifest_v1",
        "scene_id": str(metadata["scene_id"]),
        "object_id": object_id,
        "frame_count": len(paths),
        "records": [
            {"filename": path.name, "sha256": sha256(path)} for path in paths
        ],
    }
    write_json(mask_manifest_path, mask_manifest)
    return {
        "encoding": "rgba_alpha_antialiased_silhouette_mask",
        "occlusion_policy": "unoccluded_dynamic_silhouette",
        "directory": str(mask_dir),
        "manifest_path": str(mask_manifest_path),
        "manifest_sha256": sha256(mask_manifest_path),
        "objects": {
            object_id: {
                "instance_id": 1,
                "directory": str(mask_dir),
                "frame_count": len(paths),
            }
        },
        "validation": {
            "policy_version": "physweep_antialiased_silhouette_validation_v1",
            "pixel_probe_frames": [1, (len(paths) + 1) // 2, len(paths)],
            "nonempty_probe_count": nonempty,
        },
    }


def render(
    metadata_path: Path,
    video_path_override: Path | None = None,
    frame_dir_override: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    metadata_path = metadata_path.resolve()
    metadata = load_json(metadata_path)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported passive-pinball metadata schema")
    _validate_metadata_files(PROJECT_ROOT, metadata)
    renderer_binding = metadata["implementation"]["renderer"]
    if Path(__file__).resolve() != (PROJECT_ROOT / renderer_binding["path"]).resolve():
        raise ValueError("metadata is bound to a different passive-pinball renderer")
    simulation_record, trajectory_path, _ = validate_simulation_record(
        root=PROJECT_ROOT,
        metadata_path=metadata_path,
        metadata=metadata,
    )
    with np.load(trajectory_path, allow_pickle=False) as source:
        trajectory = adapter_trajectory_view({key: source[key] for key in source.files})
    clear_scene()
    configure_scene(metadata)
    fixture_objects = add_fixture(metadata)
    ball = add_ball(metadata, trajectory)
    camera = add_camera(metadata["camera"])
    add_lights(metadata)
    scene = bpy.context.scene
    frame_dir = (
        frame_dir_override.resolve()
        if frame_dir_override is not None
        else (PROJECT_ROOT / metadata["render"]["inspection_frame_dir"]).resolve()
    )
    frame_dir.mkdir(parents=True, exist_ok=True)
    inspection_frames = [1, (scene.frame_end + 1) // 2, scene.frame_end]
    inspection_paths = []
    for frame in inspection_frames:
        scene.frame_set(frame)
        path = frame_dir / f"frame_{frame:04d}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        inspection_paths.append(path)
    video_path = (
        video_path_override.resolve()
        if video_path_override is not None
        else (PROJECT_ROOT / metadata["render"]["video_path"]).resolve()
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(video_path)
    video_encoding = configure_h264_output(
        scene,
        fps=int(metadata["simulation"]["time"]["output_fps"]),
        frame_count=int(metadata["simulation"]["time"]["frame_count"]),
    )
    bpy.ops.render.render(animation=True)
    instance_mask_output = render_instance_masks(
        metadata, ball, fixture_objects, frame_dir
    )
    fixture_payload = json.dumps(
        metadata["physics"]["fixture"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    record = {
        "schema_version": "physweep_passive_pinball_render_record_v1",
        "scene_id": metadata["scene_id"],
        "metadata_path": str(metadata_path),
        "metadata_sha256": simulation_record["metadata"]["sha256"],
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": simulation_record["trajectory"]["sha256"],
        "render_output_overridden": bool(
            video_path_override is not None or frame_dir_override is not None
        ),
        "fixture_sha256": hashlib.sha256(fixture_payload).hexdigest(),
        "video_path": str(video_path),
        "video_sha256": sha256(video_path),
        "inspection_frames": [str(path) for path in inspection_paths],
        "instance_mask_output": instance_mask_output,
        "camera": camera,
        "render_engine": scene.render.engine,
        "video_encoding": video_encoding,
        "wall_time_s": round(time.perf_counter() - started, 6),
    }
    write_json(frame_dir / "render_record.json", record)
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    args = blender_args()
    render(args.metadata, args.video_path, args.inspection_frame_dir)
