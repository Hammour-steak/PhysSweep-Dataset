#!/usr/bin/env python3
"""Render a metadata-bound formal one-marble passive track scene."""

from __future__ import annotations

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.physics.generate_marble_run_scene import (  # pylint: disable=wrong-import-position
    SCHEMA_VERSION,
    _validate_metadata_files,
    load_json,
    sha256,
    write_json,
)
from tools.dataset_contract.immutable_scene_contract import validate_simulation_record
from tools.assets.blender_asset_import import clear_scene
from tools.rendering.blender_scene import (
    add_bound_lights,
    look_at,
    parse_scene_render_args,
)
from tools.dataset_contract.trajectory_contract import adapter_trajectory_view
from tools.rendering.video_encoding import (
    configure_h264_output,
    normalize_h264_container,
    require_render_finished,
)

def material(name: str, rgba: list[float], metallic: float = 0.0) -> Any:
    result = bpy.data.materials.new(name)
    values = tuple(float(value) for value in rgba)
    result.diffuse_color = values
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = values
    shader.inputs["Roughness"].default_value = 0.26 if metallic else 0.46
    shader.inputs["Metallic"].default_value = float(metallic)
    shader.inputs["Alpha"].default_value = values[3]
    if values[3] < 1.0:
        result.blend_method = "BLEND"
        result.use_screen_refraction = True
        result.show_transparent_back = True
        result.alpha_threshold = 0.0
    return result


def add_box(binding: dict[str, Any]) -> Any:
    bpy.ops.mesh.primitive_cube_add(
        size=2.0,
        location=tuple(float(value) for value in binding["position_m"]),
    )
    obj = bpy.context.object
    obj.name = str(binding["id"])
    obj.scale = tuple(float(value) for value in binding["half_extents_m"])
    obj.data.materials.append(
        material(f"{binding['id']}_material", binding["color_rgba"])
    )
    return obj


def import_track(component: dict[str, Any], *, root: Path = PROJECT_ROOT) -> Any:
    declared = (root / str(component["source_path"])).absolute()
    declared.relative_to(root)
    source = declared.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if sha256(source) != str(component["source_sha256"]):
        raise ValueError(f"marble-run source hash changed: {component['id']}")
    before = set(bpy.context.scene.objects)
    bpy.ops.import_mesh.stl(filepath=str(source))
    meshes = [
        obj
        for obj in bpy.context.scene.objects
        if obj not in before and obj.type == "MESH"
    ]
    if len(meshes) != 1:
        raise ValueError(f"marble-run track must import as one mesh: {component['id']}")
    obj = meshes[0]
    obj.name = str(component["id"])
    obj.location = tuple(float(value) for value in component["base_position_m"])
    obj.rotation_mode = "QUATERNION"
    x, y, z, w = [
        float(value)
        for value in component["base_orientation_quaternion_xyzw"]
    ]
    obj.rotation_quaternion = mathutils.Quaternion((w, x, y, z))
    obj.scale = tuple(float(value) for value in component["mesh_scale"])
    obj.data.materials.append(
        material(f"{component['id']}_material", component["color_rgba"])
    )
    return obj


def add_physical_fixture(metadata: dict[str, Any], *, root: Path = PROJECT_ROOT) -> list[Any]:
    fixture = metadata["physics"]["fixture"]
    result = [import_track(component, root=root) for component in fixture["mesh_components"]]
    result.extend(add_box(collider) for collider in fixture["analytic_colliders"])
    return result


def add_fixture(metadata: dict[str, Any], *, root: Path = PROJECT_ROOT) -> list[Any]:
    result = add_physical_fixture(metadata, root=root)
    backboard = metadata["render"]["context"]["backboard"]
    if backboard.get("physics_role") != "render_only_context":
        raise ValueError("marble-run backboard must remain render-only")
    result.append(
        add_box(
            {
                "id": "backboard",
                "half_extents_m": backboard["half_extents_m"],
                "position_m": backboard["position_m"],
                "color_rgba": backboard["color_rgba"],
            }
        )
    )
    return result


def add_marble(metadata: dict[str, Any], trajectory: dict[str, np.ndarray]) -> Any:
    dynamic = metadata["simulation"]["objects"][0]
    positions = np.asarray(trajectory["position_m"], dtype=np.float64)
    orientations = np.asarray(trajectory["quaternion_xyzw"], dtype=np.float64)
    if positions.shape[1:] != (1, 3) or orientations.shape[1:] != (1, 4):
        raise ValueError("marble-run trajectory must contain exactly one object")
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=64,
        ring_count=32,
        radius=float(dynamic["collision_proxy"]["radius_m"]),
    )
    marble = bpy.context.object
    marble.name = str(dynamic["object_id"])
    marble.data.materials.append(
        material("marble_material", dynamic["visual"]["color_rgba"], metallic=0.68)
    )
    marble.rotation_mode = "QUATERNION"
    bpy.ops.object.shade_smooth()
    for frame, (position, quaternion) in enumerate(
        zip(positions[:, 0], orientations[:, 0]), start=1
    ):
        marble.location = tuple(float(value) for value in position)
        x, y, z, w = [float(value) for value in quaternion]
        marble.rotation_quaternion = mathutils.Quaternion((w, x, y, z))
        marble.keyframe_insert(data_path="location", frame=frame)
        marble.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    if marble.animation_data and marble.animation_data.action:
        for curve in marble.animation_data.action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"
    return marble


def add_camera(binding: dict[str, Any]) -> dict[str, Any]:
    bpy.ops.object.camera_add(
        location=tuple(float(value) for value in binding["position_m"])
    )
    camera = bpy.context.object
    camera.name = "marble_run_camera"
    camera.data.lens = float(binding["focal_length_mm"])
    camera.data.sensor_width = float(binding["sensor_width_mm"])
    camera.data.clip_start = 0.03
    camera.data.clip_end = 100.0
    look_at(camera, mathutils.Vector(binding["target_m"]))
    bpy.context.scene.camera = camera
    return dict(binding)


def configure_scene(metadata: dict[str, Any]) -> None:
    scene = bpy.context.scene
    render = metadata["render"]
    time_binding = metadata["simulation"]["time"]
    if render["engine"] != "BLENDER_EEVEE":
        raise ValueError("marble-run renderer requires BLENDER_EEVEE")
    scene.render.engine = str(render["engine"])
    scene.eevee.taa_render_samples = int(render["samples"])
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3.0
    scene.eevee.gtao_factor = 1.25
    scene.render.resolution_x = int(render["resolution"][0])
    scene.render.resolution_y = int(render["resolution"][1])
    scene.render.resolution_percentage = 100
    scene.render.fps = int(time_binding["output_fps"])
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.frame_start = 1
    scene.frame_end = int(time_binding["frame_count"])
    scene.view_settings.look = "Medium High Contrast"
    world = bpy.data.worlds.new("marble_run_world") if scene.world is None else scene.world
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (
        *[float(value) for value in render["world_color_rgb"]],
        1.0,
    )
    background.inputs["Strength"].default_value = 0.32


def render(
    metadata_path: Path,
    video_path_override: Path | None = None,
    frame_dir_override: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    metadata_path = metadata_path.resolve()
    metadata = load_json(metadata_path)
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported marble-run metadata schema")
    _validate_metadata_files(PROJECT_ROOT, metadata)
    renderer_binding = metadata["implementation"]["renderer"]
    if Path(__file__).resolve() != (PROJECT_ROOT / renderer_binding["path"]).resolve():
        raise ValueError("metadata is bound to a different marble-run renderer")
    simulation_record, trajectory_path, _ = validate_simulation_record(
        root=PROJECT_ROOT,
        metadata_path=metadata_path,
        metadata=metadata,
    )
    with np.load(trajectory_path, allow_pickle=False) as source:
        trajectory = adapter_trajectory_view(
            {key: source[key] for key in source.files}
        )
    clear_scene()
    configure_scene(metadata)
    add_fixture(metadata)
    add_marble(metadata, trajectory)
    camera = add_camera(metadata["camera"])
    add_bound_lights(metadata)
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
    require_render_finished(
        bpy.ops.render.render(animation=True),
        label=f"video animation render for {metadata['scene_id']}",
    )
    normalize_h264_container(
        video_path,
        expected_frame_count=scene.frame_end - scene.frame_start + 1,
    )
    fixture_payload = json.dumps(
        metadata["physics"]["fixture"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    record = {
        "schema_version": "physweep_marble_run_render_record_v1",
        "scene_id": metadata["scene_id"],
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": simulation_record["trajectory"]["sha256"],
        "render_output_overridden": bool(
            video_path_override is not None or frame_dir_override is not None
        ),
        "fixture_sha256": hashlib.sha256(fixture_payload).hexdigest(),
        "video_path": str(video_path),
        "video_sha256": sha256(video_path),
        "inspection_frames": [str(path) for path in inspection_paths],
        "instance_mask_output": None,
        "camera": camera,
        "render_engine": scene.render.engine,
        "render_samples": int(scene.eevee.taa_render_samples),
        "video_encoding": video_encoding,
        "wall_time_s": round(time.perf_counter() - started, 6),
    }
    write_json(frame_dir / "render_record.json", record)
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    args = parse_scene_render_args(__doc__)
    render(
        args.metadata,
        args.video_path,
        args.inspection_frame_dir,
    )
