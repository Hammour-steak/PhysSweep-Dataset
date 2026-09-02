#!/usr/bin/env python3
"""Render two spheres in one reviewed billiards, pinball, or marble-run fixture."""

from __future__ import annotations

import copy
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

from tools.assets.blender_asset_import import clear_scene
from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json
from tools.dataset_contract.immutable_scene_contract import validate_simulation_record
from tools.dataset_contract.trajectory_contract import adapter_trajectory_view
from tools.rendering.blender_scene import look_at, parse_scene_render_args
from tools.rendering.render_asset_proxy_scene import (
    add_environment,
    add_lighting,
    add_support,
    apply_hdri,
    configure_project_root,
    setup_scene,
)
from tools.rendering.render_billiards_scene import hidden_ball_materials
from tools.rendering.render_marble_run_scene import add_fixture as add_marble_fixture
from tools.rendering.render_passive_pinball_scene import add_fixture as add_pinball_fixture
from tools.rendering.specialized_render_evidence import (
    render_implementation,
    render_instance_masks,
)
from tools.rendering.video_encoding import (
    configure_h264_output,
    normalize_h264_container,
    require_render_finished,
)


SUPPORTED_SCHEMAS = {
    "physweep_billiards_scene_v4": "billiards",
    "physweep_passive_pinball_scene_v1": "passive_pinball",
    "physweep_marble_run_scene_v1": "marble_run",
}


def _verify_implementation(metadata: dict[str, Any]) -> None:
    expected = {
        "sampler": PROJECT_ROOT / "tools/sampling/sample_two_object_specialized.py",
        "renderer": Path(__file__).resolve(),
    }
    for name, path in expected.items():
        binding = metadata["implementation"][name]
        if (PROJECT_ROOT / str(binding["path"])).resolve() != path.resolve():
            raise ValueError(f"specialized {name} path changed")
        if sha256(path) != str(binding["sha256"]):
            raise ValueError(f"specialized {name} hash changed")


def _setup(metadata: dict[str, Any]) -> None:
    normalized = copy.deepcopy(metadata)
    if "output_fps" not in normalized["physics"]:
        normalized["physics"]["output_fps"] = int(
            normalized["simulation"]["time"]["output_fps"]
        )
        normalized["physics"]["frame_count"] = int(
            normalized["simulation"]["time"]["frame_count"]
        )
    setup_scene(normalized)


def _camera(binding: dict[str, Any]) -> dict[str, Any]:
    bpy.ops.object.camera_add(
        location=tuple(float(value) for value in binding["position_m"])
    )
    camera = bpy.context.object
    camera.name = "two_object_specialized_camera"
    camera.data.lens = float(binding["focal_length_mm"])
    camera.data.sensor_width = float(binding["sensor_width_mm"])
    camera.data.clip_start = 0.03
    camera.data.clip_end = 100.0
    look_at(camera, mathutils.Vector(binding["target_m"]))
    bpy.context.scene.camera = camera
    return dict(binding)


def _sphere_material(name: str, rgba: list[float]) -> Any:
    values = tuple(float(value) for value in rgba)
    result = bpy.data.materials.new(name)
    result.diffuse_color = values
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = values
    shader.inputs["Metallic"].default_value = 0.68
    shader.inputs["Roughness"].default_value = 0.24
    return result


def _add_spheres(
    metadata: dict[str, Any],
    trajectory: dict[str, np.ndarray],
    materials: list[Any] | None = None,
) -> list[Any]:
    objects = metadata.get("simulation", {}).get("objects")
    if not objects:
        radius = float(metadata["physics"]["ball_radius_m"])
        objects = [
            {
                "object_id": state["object_id"],
                "collision_proxy": {"radius_m": radius},
                "visual": {"color_rgba": [0.92, 0.92, 0.92, 1.0]},
            }
            for state in metadata["physics"]["initial_states"]
        ]
    positions = np.asarray(trajectory["position_m"], dtype=np.float64)
    orientations = np.asarray(trajectory["quaternion_xyzw"], dtype=np.float64)
    if positions.shape[1:] != (2, 3) or orientations.shape[1:] != (2, 4):
        raise ValueError("specialized trajectory must contain exactly two objects")
    if materials is not None and len(materials) < 2:
        raise ValueError("specialized fixture exposes fewer than two materials")
    result = []
    for index, dynamic in enumerate(objects):
        radius = float(dynamic["collision_proxy"].get("radius_m", metadata["physics"].get("ball_radius_m")))
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=48, ring_count=24, radius=radius
        )
        sphere = bpy.context.object
        sphere.name = str(dynamic["object_id"])
        sphere.data.materials.append(
            materials[index]
            if materials is not None
            else _sphere_material(
                f"{sphere.name}_material", dynamic["visual"]["color_rgba"]
            )
        )
        sphere.rotation_mode = "QUATERNION"
        bpy.ops.object.shade_smooth()
        for frame, (position, quaternion) in enumerate(
            zip(positions[:, index], orientations[:, index]), start=1
        ):
            sphere.location = tuple(float(value) for value in position)
            x, y, z, w = [float(value) for value in quaternion]
            sphere.rotation_quaternion = (w, x, y, z)
            sphere.keyframe_insert(data_path="location", frame=frame)
            sphere.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        if sphere.animation_data and sphere.animation_data.action:
            for curve in sphere.animation_data.action.fcurves:
                for point in curve.keyframe_points:
                    point.interpolation = "LINEAR"
        result.append(sphere)
    return result


def _fixture(metadata: dict[str, Any], family: str) -> tuple[list[Any], list[Any] | None]:
    if family == "passive_pinball":
        return add_pinball_fixture(metadata), None
    if family == "marble_run":
        return add_marble_fixture(metadata), None
    composition_path = PROJECT_ROOT / str(metadata["composition_rules"]["path"])
    if sha256(composition_path) != str(metadata["composition_rules"]["sha256"]):
        raise ValueError("billiards composition rules changed")
    composition = next(
        record
        for record in read_json(composition_path)["records"]
        if record["asset_id"] == metadata["assets"]["support_asset_id"]
    )
    meshes = add_support(
        metadata["physics"]["static_support_binding"],
        include_all_source_meshes=True,
    )
    material_map = hidden_ball_materials(meshes, composition["component_policy"])
    return meshes, list(material_map.values())


def render(
    metadata_path: Path,
    video_path_override: Path | None = None,
    frame_dir_override: Path | None = None,
    mask_root_override: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    metadata_path = metadata_path.resolve()
    metadata = read_json(metadata_path)
    family = SUPPORTED_SCHEMAS.get(str(metadata.get("schema_version")))
    if family is None or str(metadata.get("semantics", {}).get("scene_family")) != family:
        raise ValueError("unsupported specialized two-object metadata")
    if int(metadata["semantics"]["dynamic_object_count"]) != 2:
        raise ValueError("specialized two-object renderer requires two objects")
    _verify_implementation(metadata)
    simulation_record, trajectory_path, _ = validate_simulation_record(
        root=PROJECT_ROOT, metadata_path=metadata_path, metadata=metadata
    )
    with np.load(trajectory_path, allow_pickle=False) as source:
        trajectory = adapter_trajectory_view(
            {key: source[key] for key in source.files}
        )
    clear_scene()
    _setup(metadata)
    apply_hdri(metadata["render"]["environment"])
    fixture_objects, materials = _fixture(metadata, family)
    spheres = _add_spheres(metadata, trajectory, materials)
    camera = _camera(metadata["camera"])
    environment_objects = add_environment(
        metadata["render"]["environment"], camera
    )
    add_lighting(
        mathutils.Vector(camera["target_m"]),
        2.2,
        metadata["render"]["environment"]["lighting"],
    )
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
    frame_count = scene.frame_end - scene.frame_start + 1
    video_encoding = configure_h264_output(
        scene, fps=int(scene.render.fps), frame_count=frame_count
    )
    require_render_finished(
        bpy.ops.render.render(animation=True),
        label=f"video animation render for {metadata['scene_id']}",
    )
    normalize_h264_container(video_path, expected_frame_count=frame_count)
    object_ids = [
        str(record["object_id"])
        for record in metadata["object_identity"]["objects"]
    ]
    masks = render_instance_masks(
        root=PROJECT_ROOT,
        metadata=metadata,
        dynamic_objects={
            object_id: [sphere] for object_id, sphere in zip(object_ids, spheres)
        },
        mask_root_override=mask_root_override,
    )
    record = {
        "schema_version": "physweep_two_object_specialized_render_record_v1",
        "scene_id": metadata["scene_id"],
        "family": family,
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": simulation_record["trajectory"]["sha256"],
        "video_path": str(video_path),
        "video_sha256": sha256(video_path),
        "inspection_frames": [str(path) for path in inspection_paths],
        "instance_mask_output": masks,
        "camera": camera,
        "render_engine": scene.render.engine,
        "render_samples": int(scene.eevee.taa_render_samples),
        "video_encoding": video_encoding,
        "render_only_context_object_count": len(environment_objects),
        "fixture_object_count": len(fixture_objects),
        "implementation": render_implementation(Path(__file__)),
        "wall_time_s": round(time.perf_counter() - started, 6),
    }
    record_path = frame_dir / "render_record.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    args = parse_scene_render_args(
        __doc__, project_root=PROJECT_ROOT, include_mask_output=True
    )
    PROJECT_ROOT = args.root.resolve()
    configure_project_root(PROJECT_ROOT)
    render(
        args.metadata,
        args.video_path,
        args.inspection_frame_dir,
        args.instance_mask_dir,
    )
