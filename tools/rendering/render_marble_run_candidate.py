#!/usr/bin/env python3
"""Render metadata-bound marble-run candidate inspection frames or video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import bpy
import mathutils
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.core.json_io import read_json as load_json
from tools.core.paths import resolve_project_path as project_path
from tools.physics.generate_marble_run_candidate import _validate_metadata_files


def blender_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frames", default="0,12,24,36,48,60,72")
    parser.add_argument("--video", type=Path)
    return parser.parse_args(values)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def material(name: str, rgba: list[float], metallic: float = 0.0) -> Any:
    result = bpy.data.materials.new(name)
    result.diffuse_color = tuple(float(value) for value in rgba)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = tuple(float(value) for value in rgba)
    shader.inputs["Roughness"].default_value = 0.32 if metallic else 0.48
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Alpha"].default_value = float(rgba[3])
    if float(rgba[3]) < 1.0:
        result.blend_method = "BLEND"
        result.use_screen_refraction = True
        result.show_transparent_back = True
        result.alpha_threshold = 0.0
    return result


def add_box(name: str, half_extents: list[float], position: list[float], rgba: list[float]) -> Any:
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=position)
    obj = bpy.context.object
    obj.name = name
    obj.scale = tuple(float(value) for value in half_extents)
    obj.data.materials.append(material(f"{name}_material", rgba))
    return obj


def import_track(root: Path, component: dict[str, Any]) -> Any:
    path = project_path(root, component["source_path"])
    before = set(bpy.context.scene.objects)
    bpy.ops.import_mesh.stl(filepath=str(path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if len(meshes) != 1:
        raise ValueError(f"track source must import as one mesh: {component['id']}")
    obj = meshes[0]
    obj.name = str(component["id"])
    obj.location = tuple(float(value) for value in component["base_position_m"])
    obj.rotation_mode = "QUATERNION"
    x, y, z, w = [
        float(value) for value in component["base_orientation_quaternion_xyzw"]
    ]
    obj.rotation_quaternion = mathutils.Quaternion((w, x, y, z))
    obj.scale = tuple(float(value) for value in component["mesh_scale"])
    obj.data.materials.append(
        material(f"{component['id']}_material", component["color_rgba"])
    )
    return obj


def add_marble(metadata: dict[str, Any], trajectory: dict[str, np.ndarray]) -> Any:
    dynamic = metadata["physics"]["objects"][0]
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=64,
        ring_count=32,
        radius=float(dynamic["radius_m"]),
        location=trajectory["positions"][0],
    )
    obj = bpy.context.object
    obj.name = str(dynamic["object_id"])
    obj.data.materials.append(
        material("marble_material", dynamic["color_rgba"], metallic=0.65)
    )
    for frame, (position, orientation) in enumerate(
        zip(trajectory["positions"], trajectory["orientations_xyzw"])
    ):
        obj.location = tuple(float(value) for value in position)
        x, y, z, w = [float(value) for value in orientation]
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = mathutils.Quaternion((w, x, y, z))
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    for curve in obj.animation_data.action.fcurves:
        for point in curve.keyframe_points:
            point.interpolation = "LINEAR"
    return obj


def point_at(obj: Any, target: list[float]) -> None:
    direction = mathutils.Vector(target) - obj.location
    if direction.length <= 1.0e-9:
        raise ValueError("camera or light position equals its target")
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera(render: dict[str, Any]) -> Any:
    binding = render["camera"]
    camera_data = bpy.data.cameras.new("candidate_camera")
    camera = bpy.data.objects.new("candidate_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = tuple(float(value) for value in binding["position_m"])
    camera_data.lens = float(binding["focal_length_mm"])
    camera_data.sensor_width = 36.0
    point_at(camera, binding["target_m"])
    bpy.context.scene.camera = camera
    return camera


def add_lights(render: dict[str, Any]) -> None:
    target = render["camera"]["target_m"]
    for binding in render["lights"]:
        light_data = bpy.data.lights.new(str(binding["id"]), str(binding["type"]))
        light_data.energy = float(binding["energy_w"])
        light_data.color = tuple(float(value) for value in binding["color_rgb"])
        light_data.size = float(binding["size_m"])
        light = bpy.data.objects.new(str(binding["id"]), light_data)
        bpy.context.collection.objects.link(light)
        light.location = tuple(float(value) for value in binding["position_m"])
        point_at(light, target)


def configure_scene(metadata: dict[str, Any]) -> None:
    render = metadata["render"]
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3.0
    scene.eevee.gtao_factor = 1.25
    scene.render.resolution_x = int(render["resolution"][0])
    scene.render.resolution_y = int(render["resolution"][1])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.fps = int(render["fps"])
    scene.frame_start = 0
    scene.frame_end = int(metadata["physics"]["frame_count"]) - 1
    scene.view_settings.look = "Medium High Contrast"
    world = bpy.data.worlds.new("candidate_world") if scene.world is None else scene.world
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        *[float(value) for value in render["world_color_rgb"]],
        1.0,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.32


def build_scene(root: Path, metadata: dict[str, Any], trajectory: dict[str, np.ndarray]) -> None:
    clear_scene()
    for component in metadata["fixture"]["mesh_components"]:
        import_track(root, component)
    for collider in metadata["fixture"]["analytic_colliders"]:
        add_box(
            str(collider["id"]),
            collider["half_extents_m"],
            collider["position_m"],
            collider["color_rgba"],
        )
    backboard = metadata["render"]["context"]["backboard"]
    if backboard.get("physics_role") != "render_only_context":
        raise ValueError("backboard must remain render-only context")
    add_box(
        "backboard",
        backboard["half_extents_m"],
        backboard["position_m"],
        backboard["color_rgba"],
    )
    add_marble(metadata, trajectory)
    add_camera(metadata["render"])
    add_lights(metadata["render"])
    configure_scene(metadata)


def main() -> None:
    args = blender_args()
    root = args.root.resolve()
    metadata_path = project_path(root, args.metadata)
    trajectory_path = project_path(root, args.trajectory)
    metadata = load_json(metadata_path)
    _validate_metadata_files(root, metadata)
    renderer_path = project_path(root, metadata["render"]["implementation"]["path"])
    if renderer_path != Path(__file__).resolve():
        raise ValueError("metadata is bound to a different candidate renderer")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        trajectory = {key: archive[key] for key in archive.files}
    expected_frames = int(metadata["physics"]["frame_count"])
    expected_shapes = {
        "positions": (expected_frames, 3),
        "orientations_xyzw": (expected_frames, 4),
        "linear_velocities": (expected_frames, 3),
        "angular_velocities": (expected_frames, 3),
        "times": (expected_frames,),
    }
    expected_keys = {*expected_shapes, "object_ids"}
    if set(trajectory) != expected_keys:
        raise ValueError("trajectory fields differ from the candidate contract")
    for key, shape in expected_shapes.items():
        if trajectory[key].shape != shape or not np.isfinite(trajectory[key]).all():
            raise ValueError(f"trajectory {key} layout differs from metadata")
    expected_times = np.arange(expected_frames, dtype=np.float64) / int(
        metadata["physics"]["output_fps"]
    )
    if not np.array_equal(trajectory["times"], expected_times):
        raise ValueError("trajectory time axis differs from metadata")
    if not np.allclose(
        np.linalg.norm(trajectory["orientations_xyzw"], axis=1),
        1.0,
        atol=1.0e-6,
    ):
        raise ValueError("trajectory contains non-unit orientations")
    physics_objects = metadata["physics"]["objects"]
    identity_objects = metadata["object_identity"]["objects"]
    if len(physics_objects) != 1 or len(identity_objects) != 1:
        raise ValueError("candidate rendering requires exactly one object")
    object_id = str(physics_objects[0]["object_id"])
    if identity_objects[0]["object_id"] != object_id:
        raise ValueError("metadata object identity is inconsistent")
    if trajectory["object_ids"].tolist() != [object_id]:
        raise ValueError("trajectory object identity differs from metadata")
    build_scene(root, metadata, trajectory)
    scene = bpy.context.scene

    if args.output_dir is not None:
        output_dir = project_path(root, args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = [int(value) for value in args.frames.split(",") if value.strip()]
        if not frames or min(frames) < 0 or max(frames) >= expected_frames:
            raise ValueError("inspection frame is outside the trajectory")
        for frame in frames:
            scene.frame_set(frame)
            scene.render.filepath = str(output_dir / f"frame_{frame:04d}.png")
            bpy.ops.render.render(write_still=True)

    if args.video is not None:
        video = project_path(root, args.video)
        video.parent.mkdir(parents=True, exist_ok=True)
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"
        scene.render.filepath = str(video)
        bpy.ops.render.render(animation=True)

    if args.output_dir is None and args.video is None:
        raise ValueError("provide --output-dir, --video, or both")


if __name__ == "__main__":
    main()
