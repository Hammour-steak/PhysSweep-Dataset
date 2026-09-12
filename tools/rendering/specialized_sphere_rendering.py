#!/usr/bin/env python3
"""Shared sphere animation and rendering in explicitly reviewed fixtures."""

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

CODE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = CODE_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.assets.blender_asset_import import clear_scene
from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json, write_json_atomic
from tools.dataset_contract.camera_contract import camera_clipping_range
from tools.dataset_contract.immutable_scene_contract import validate_simulation_record
from tools.dataset_contract.trajectory_contract import adapter_trajectory_view
from tools.rendering.blender_scene import look_at, parse_scene_render_args as parse_scene_render_args
from tools.rendering.render_asset_proxy_scene import (
    add_environment,
    add_lighting,
    add_support,
    apply_hdri,
    configure_project_root as configure_project_root,
    setup_scene,
)
from tools.rendering.render_billiards_scene import hidden_ball_materials
from tools.rendering.render_marble_run_scene import add_fixture as add_marble_fixture, add_physical_fixture as add_marble_physical_fixture
from tools.rendering.render_passive_pinball_scene import add_fixture as add_pinball_fixture
from tools.rendering.specialized_render_evidence import (
    render_implementation,
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


def _verify_implementation(metadata: dict[str, Any], sampler_path: Path, renderer_path: Path) -> None:
    expected = {
        "renderer": renderer_path,
    }
    if metadata["semantics"]["dynamic_object_count"] == 3:
        expected["sphere_render_core"] = Path(__file__).resolve()
    else:
        expected["sampler"] = sampler_path
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
    clip_start, clip_end = camera_clipping_range(binding)
    bpy.ops.object.camera_add(
        location=tuple(float(value) for value in binding["position_m"])
    )
    camera = bpy.context.object
    camera.name = "two_object_specialized_camera"
    camera.data.lens = float(binding["focal_length_mm"])
    camera.data.sensor_width = float(binding["sensor_width_mm"])
    camera.data.clip_start = clip_start
    camera.data.clip_end = clip_end
    look_at(camera, mathutils.Vector(binding["target_m"]))
    bpy.context.scene.camera = camera
    return {
        **binding,
        "clip_start_m": float(camera.data.clip_start),
        "clip_end_m": float(camera.data.clip_end),
    }


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
    *, object_count: int = 2,
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
    if len(objects) != object_count or positions.shape[1:] != (object_count, 3) or orientations.shape[1:] != (object_count, 4):
        raise ValueError("specialized trajectory count differs from explicit renderer scope")
    if materials is not None and len(materials) < object_count:
        raise ValueError("specialized fixture exposes too few materials")
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
        builder = add_marble_physical_fixture if metadata.get("schema_version") == "physweep_marble_run_three_object_scene_v1" else add_marble_fixture
        return builder(metadata, root=PROJECT_ROOT), None
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
    if metadata.get('schema_version') == 'physweep_billiards_three_object_scene_v1':
        slots=[o['visual_profile']['material_slot'] for o in metadata['simulation']['objects']]
        if sorted(slots) != ['cue_ball','object_ball_1','object_ball_2']:
            raise ValueError('three-object billiards material slots are incomplete')
        return meshes, [material_map[slot] for slot in slots]
    # Bind A/B explicitly; imported mesh order is not an object identity contract.
    return meshes, [material_map[f"object_ball_{index + 1}"] for index in range(2)]


def render_specialized(
    metadata_path: Path,
    video_path_override: Path | None = None,
    frame_dir_override: Path | None = None,
    *, object_count: int, schema_map: dict, renderer_path: Path, sampler_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    metadata_path = metadata_path.resolve()
    metadata = read_json(metadata_path)
    family = schema_map.get(str(metadata.get("schema_version")))
    if family is None or str(metadata.get("semantics", {}).get("scene_family")) != family:
        raise ValueError("unsupported explicitly registered specialized metadata")
    if int(metadata["semantics"]["dynamic_object_count"]) != object_count:
        raise ValueError("specialized renderer count differs from its explicit entry")
    _verify_implementation(metadata, sampler_path, renderer_path)
    if object_count == 3:
        from tools.assets.three_object_resources import validate_resources
        from tools.motion_rules.three_object.billiards import validate_billiards_contract
        from tools.motion_rules.three_object.pinball import validate_pinball_contract
        from tools.motion_rules.three_object.marble import validate_marble_contract
        validators={'billiards':validate_billiards_contract,'passive_pinball':validate_pinball_contract,'marble_run':validate_marble_contract}
        if family not in validators:raise ValueError('unsupported three-object specialized render family')
        validators[family](metadata)
        validate_resources(PROJECT_ROOT, metadata, metadata['visual_resource_binding'])
    simulation_record, trajectory_path, _ = validate_simulation_record(
        root=PROJECT_ROOT, metadata_path=metadata_path, metadata=metadata
    )
    with np.load(trajectory_path, allow_pickle=False) as source:
        trajectory = adapter_trajectory_view(
            {key: source[key] for key in source.files}
        )
    clear_scene()
    _setup(metadata)
    if object_count == 3:
        if metadata['render'].get('use_motion_blur') is not False:
            raise ValueError('three-object group must explicitly disable motion blur')
        bpy.context.scene.eevee.use_motion_blur = False
    apply_hdri(metadata["render"]["environment"])
    fixture_objects, materials = _fixture(metadata, family)
    spheres = _add_spheres(metadata, trajectory, materials, object_count=object_count)
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
    video_render_samples = int(scene.eevee.taa_render_samples)
    require_render_finished(
        bpy.ops.render.render(animation=True),
        label=f"video animation render for {metadata['scene_id']}",
    )
    normalize_h264_container(video_path, expected_frame_count=frame_count)
    record = {
        "schema_version": ("physweep_two_object_specialized_render_record_v2" if object_count == 2 else "physweep_three_object_specialized_render_record_v1"),
        "scene_id": metadata["scene_id"],
        "family": family,
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": simulation_record["trajectory"]["sha256"],
        "video_path": str(video_path),
        "video_sha256": sha256(video_path),
        "inspection_frames": [str(path) for path in inspection_paths],
        "camera": camera,
        "render_engine": scene.render.engine,
        "render_samples": video_render_samples,
        "video_encoding": video_encoding,
        "render_only_context_object_count": len(environment_objects),
        "fixture_object_count": len(fixture_objects),
        "implementation": {**render_implementation(renderer_path), "sphere_render_core": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))}},
        "wall_time_s": round(time.perf_counter() - started, 6),
    }
    record_path = frame_dir / "render_record.json"
    if object_count == 3:
        record['object_materials'] = [{'object_id': obj['object_id'], **({'material_slot': obj['visual_profile']['material_slot']} if family=='billiards' else {'recipe_id':obj['visual_profile']['recipe_id']}),
            'actual_material_name': sphere.data.materials[0].name,
            'images': [{'name': node.image.name, 'size': list(node.image.size), 'packed': node.image.packed_file is not None}
                       for node in sphere.data.materials[0].node_tree.nodes if node.type == 'TEX_IMAGE' and node.image is not None]}
            for obj, sphere in zip(metadata['simulation']['objects'], spheres)]
        if family in {'passive_pinball','marble_run'}:
            for obj,sphere,evidence in zip(metadata['simulation']['objects'],spheres,record['object_materials']):
                shader=sphere.data.materials[0].node_tree.nodes.get('Principled BSDF')
                rgba=list(shader.inputs['Base Color'].default_value)
                metallic=float(shader.inputs['Metallic'].default_value);roughness=float(shader.inputs['Roughness'].default_value)
                if rgba!=np.asarray(obj['visual']['color_rgba'],dtype=np.float32).astype(float).tolist():
                    raise ValueError('actual specialized sphere color differs from frozen recipe')
                if metallic!=float(np.float32(obj['visual_profile']['metallic'])) or roughness!=float(np.float32(obj['visual_profile']['roughness'])):
                    raise ValueError('actual specialized sphere shader differs from frozen recipe')
                evidence.update(base_color_rgba=rgba,metallic=metallic,roughness=roughness,recipe_binding=obj['visual_profile']['recipe_binding'])
        record['fixed_appearance'] = {'motion_blur': bool(scene.eevee.use_motion_blur),
            'exposure': float(scene.view_settings.exposure), 'resolution': [scene.render.resolution_x, scene.render.resolution_y]}
    write_json_atomic(record_path, record)
    print(json.dumps(record, indent=2))
    return record
