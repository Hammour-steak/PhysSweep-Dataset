#!/usr/bin/env python3
"""Render reviewed action-surface anchors inside their source environments."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from tools.core.blender_runtime import blender_argv
from tools.core.blender_runtime import patch_numpy_for_blender_gltf as patch_numpy
from tools.rendering.blender_scene import look_at

from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def bbox(meshes: list[Any]) -> tuple[Any, Any]:
    import mathutils  # pylint: disable=import-outside-toplevel

    low = mathutils.Vector((float("inf"), float("inf"), float("inf")))
    high = mathutils.Vector((float("-inf"), float("-inf"), float("-inf")))
    for obj in meshes:
        for corner in obj.bound_box:
            point = obj.matrix_world @ mathutils.Vector(corner)
            low.x = min(low.x, point.x)
            low.y = min(low.y, point.y)
            low.z = min(low.z, point.z)
            high.x = max(high.x, point.x)
            high.y = max(high.y, point.y)
            high.z = max(high.z, point.z)
    return low, high


def setup_scene(width: int, height: int, samples: int) -> None:
    import bpy  # pylint: disable=import-outside-toplevel

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = samples
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 3.0
    scene.eevee.gtao_factor = 1.2
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.15
    scene.world = bpy.data.worlds.new("probe_world") if scene.world is None else scene.world
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.08, 0.09, 0.11, 1.0)
    background.inputs["Strength"].default_value = 0.6

    bpy.ops.object.light_add(type="AREA", location=(-2.8, -3.2, 4.5))
    key = bpy.context.object
    key.data.energy = 700.0
    key.data.size = 4.0
    key.data.use_shadow = True
    bpy.ops.object.light_add(type="AREA", location=(3.2, 0.8, 3.0))
    fill = bpy.context.object
    fill.data.energy = 220.0
    fill.data.size = 5.0


def import_asset(asset: dict[str, Any]) -> tuple[list[Any], Any, Any]:
    patch_numpy()
    import bpy  # pylint: disable=import-outside-toplevel

    path = PROJECT_ROOT / str(asset["path"])
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    for obj in list(imported):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
            imported.remove(obj)
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if not meshes:
        raise ValueError(f"environment contains no mesh: {asset['asset_id']}")
    for obj in meshes:
        matrix = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_world = matrix
    low, high = bbox(meshes)
    excluded_names = {str(value) for value in asset.get("exclude_object_names", [])}
    excluded_prefixes = tuple(
        str(value) for value in asset.get("exclude_object_name_prefixes", [])
    )
    for obj in list(meshes):
        if obj.name in excluded_names or any(
            obj.name.startswith(prefix) for prefix in excluded_prefixes
        ):
            bpy.data.objects.remove(obj, do_unlink=True)
            meshes.remove(obj)
    return meshes, low, high


def place_environment(
    meshes: list[Any],
    low: Any,
    high: Any,
    asset: dict[str, Any],
    anchor_local_m: list[float],
) -> None:
    import mathutils  # pylint: disable=import-outside-toplevel

    size = high - low
    axis = {"x": 0, "y": 1, "z": 2}[str(asset["normalization_axis"])]
    scale = float(asset["target_extent_m"]) / max(float(size[axis]), 1.0e-8)
    bottom_center = mathutils.Vector(
        ((low.x + high.x) / 2.0, (low.y + high.y) / 2.0, low.z)
    )
    review_yaw = float(asset.get("review_yaw_degrees", 0.0))
    anchor = mathutils.Vector(tuple(float(value) for value in anchor_local_m))
    transform = (
        mathutils.Matrix.Rotation(math.radians(review_yaw), 4, "Z")
        @ mathutils.Matrix.Translation(-anchor)
        @ mathutils.Matrix.Scale(scale, 4)
        @ mathutils.Matrix.Translation(-bottom_center)
    )
    for obj in meshes:
        obj.matrix_world = transform @ obj.matrix_world


def marker_material(name: str, color: tuple[float, float, float, float]) -> Any:
    import bpy  # pylint: disable=import-outside-toplevel

    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = color
    principled.inputs["Roughness"].default_value = 0.35
    principled.inputs["Emission"].default_value = color
    principled.inputs["Emission Strength"].default_value = 0.2
    return material


def add_markers(clear_radius_m: float) -> list[Any]:
    import bpy  # pylint: disable=import-outside-toplevel

    cyan = marker_material("approved_anchor_cyan", (0.02, 0.8, 1.0, 1.0))
    yellow = marker_material("clearance_ring_yellow", (1.0, 0.62, 0.02, 1.0))
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=32,
        ring_count=16,
        radius=0.11,
        location=(0.0, 0.0, 0.11),
    )
    anchor = bpy.context.object
    anchor.name = "action_anchor"
    anchor.data.materials.append(cyan)
    bpy.ops.mesh.primitive_torus_add(
        major_segments=96,
        minor_segments=12,
        location=(0.0, 0.0, 0.025),
        major_radius=max(0.16, clear_radius_m),
        minor_radius=0.018,
    )
    ring = bpy.context.object
    ring.name = "clearance_ring"
    ring.data.materials.append(yellow)
    return [anchor, ring]


def add_camera() -> Any:
    import bpy  # pylint: disable=import-outside-toplevel

    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.lens = 44.0
    camera.data.clip_start = 0.03
    camera.data.clip_end = 100.0
    bpy.context.scene.camera = camera
    return camera


def render_views(
    output_root: Path,
    profile_id: str,
    camera: Any,
    clear_radius_m: float,
) -> dict[str, str]:
    import bpy  # pylint: disable=import-outside-toplevel

    context_radius = max(1.35, clear_radius_m * 1.45)
    distance = max(3.0, context_radius * 2.6)
    oblique_height = max(2.0, distance * 0.58)
    views = {
        "oblique_east": (
            (distance, 0.0, oblique_height),
            (0.0, 0.0, 0.28),
        ),
        "oblique_north": (
            (0.0, distance, oblique_height),
            (0.0, 0.0, 0.28),
        ),
        "oblique_west": (
            (-distance, 0.0, oblique_height),
            (0.0, 0.0, 0.28),
        ),
        "oblique_south": (
            (0.0, -distance, oblique_height),
            (0.0, 0.0, 0.28),
        ),
        "top": (
            (0.0, 0.0, max(4.2, context_radius * 3.1)),
            (0.0, 0.0, 0.0),
        ),
    }
    paths = {}
    for label, (position, target) in views.items():
        camera.location = position
        look_at(camera, target)
        path = output_root / profile_id / f"{label}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        bpy.context.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        paths[label] = str(path.relative_to(PROJECT_ROOT))
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-id", action="append", default=[])
    parser.add_argument(
        "--all-candidates",
        action="store_true",
        help=(
            "Render every audited candidate instead of only the "
            "highest-clearance one."
        ),
    )
    parser.add_argument("--resolution", default="960x720")
    parser.add_argument("--samples", type=int, default=24)
    return parser.parse_args(blender_argv())


def main() -> None:
    args = parse_args()
    width, height = [int(value) for value in args.resolution.lower().split("x", 1)]
    profile_payload = load_json(args.profiles)
    audit_payload = load_json(args.audit)
    profiles = {str(record["id"]): record for record in profile_payload["profiles"]}
    requested = {str(value) for value in args.profile_id}
    records = []
    for audit in audit_payload["records"]:
        profile_id = str(audit["profile_id"])
        if requested and profile_id not in requested:
            continue
        if audit["status"] != "candidate" or not audit.get("candidates"):
            continue
        profile = profiles[profile_id]
        floor_z = float(audit["floor_z_local_m"])
        candidates = (
            audit["candidates"]
            if args.all_candidates
            else audit["candidates"][:1]
        )
        for candidate_index, candidate in enumerate(candidates):
            anchor_local = [
                float(candidate["anchor_xy_local_m"][0]),
                float(candidate["anchor_xy_local_m"][1]),
                floor_z,
            ]
            print(f"probe {profile_id} candidate {candidate_index + 1}", flush=True)
            setup_scene(width, height, int(args.samples))
            meshes, low, high = import_asset(profile["asset"])
            place_environment(meshes, low, high, profile["asset"], anchor_local)
            add_markers(float(candidate["clear_radius_m"]))
            camera = add_camera()
            output_profile_id = profile_id
            if args.all_candidates:
                output_profile_id = f"{profile_id}/candidate_{candidate_index + 1:02d}"
            paths = render_views(
                args.output.resolve(),
                output_profile_id,
                camera,
                float(candidate["clear_radius_m"]),
            )
            records.append(
                {
                    "profile_id": profile_id,
                    "asset_id": str(profile["asset"]["asset_id"]),
                    "candidate_index": candidate_index,
                    "anchor_local_m": [round(value, 6) for value in anchor_local],
                    "clear_radius_m": float(candidate["clear_radius_m"]),
                    "images": paths,
                }
            )
    manifest = {
        "schema_version": "physweep_environment_action_surface_probe_v1",
        "source_profiles": str(args.profiles),
        "source_audit": str(args.audit),
        "record_count": len(records),
        "records": records,
    }
    write_json(args.output.resolve() / "render_manifest.json", manifest)
    print(args.output.resolve() / "render_manifest.json")


if __name__ == "__main__":
    main()
