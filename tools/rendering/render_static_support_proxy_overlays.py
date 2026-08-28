#!/usr/bin/env python3
"""Render visual/exact-collision overlays for every static support usage."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

import bpy
import mathutils
import numpy as np
from mathutils.bvhtree import BVHTree

if "bool" not in np.__dict__:
    np.bool = np.bool_  # type: ignore[attr-defined]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json as load_json
from tools.assets.physical_proxy_catalog import load_catalog, write_json  # noqa: E402
from tools.assets.static_support_proxy import compile_static_support_binding  # noqa: E402


def blender_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--catalog", type=Path, default=Path("assets/proxies/catalog.json")
    )
    parser.add_argument(
        "--registry", type=Path, default=Path("configs/asset_proxy_registry.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--catalog-report",
        type=Path,
        default=Path("assets/proxies/v1/static_support_visual_validation.json"),
    )
    parser.add_argument("--resolution", nargs=2, type=int, default=[640, 360])
    return parser.parse_args(values)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (bpy.data.meshes, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def import_glb(
    path: Path, include_names: list[str], reference_frame: int
) -> tuple[list[Any], Any]:
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if include_names:
        requested = set(include_names)
        selected = [
            obj
            for obj in meshes
            if obj.name in requested or obj.data.name in requested
        ]
        if not selected:
            raise ValueError(f"no requested visual components found: {include_names}")
        for obj in meshes:
            if obj not in selected:
                bpy.data.objects.remove(obj, do_unlink=True)
                imported.remove(obj)
        meshes = selected
    if not meshes:
        raise ValueError(f"no visual mesh in {path}")
    bpy.context.scene.frame_set(int(reference_frame))
    bpy.context.view_layer.update()
    root = bpy.data.objects.new(f"{path.stem}_static_root", None)
    bpy.context.collection.objects.link(root)
    imported_set = set(imported)
    for obj in imported:
        if obj.parent not in imported_set:
            world = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = world
    return meshes, root


def import_obj(path: Path) -> list[Any]:
    vertices = []
    faces = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("v "):
            values = line.split()
            vertices.append(tuple(float(value) for value in values[1:4]))
        elif line.startswith("f "):
            values = line.split()[1:]
            indices = [int(value.split("/", 1)[0]) for value in values]
            if any(index <= 0 for index in indices):
                raise ValueError(f"proxy OBJ requires positive indices: {path}")
            faces.append(tuple(index - 1 for index in indices))
    if not vertices or not faces:
        raise ValueError(f"no collision mesh in {path}")
    mesh = bpy.data.meshes.new(f"{path.stem}_zup")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(f"{path.stem}_zup", mesh)
    bpy.context.collection.objects.link(obj)
    return [obj]


def bounds(objects: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points: list[np.ndarray] = []
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        transform = evaluated.matrix_world
        points.extend(
            np.asarray(transform @ vertex.co, dtype=np.float64)
            for vertex in mesh.vertices
        )
        evaluated.to_mesh_clear()
    if not points:
        raise ValueError("cannot measure bounds from empty evaluated geometry")
    values = np.asarray(points, dtype=np.float64)
    return values.min(axis=0), values.max(axis=0)


def world_bvh(objects: list[Any]) -> BVHTree:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices: list[mathutils.Vector] = []
    polygons: list[tuple[int, ...]] = []
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        offset = len(vertices)
        transform = evaluated.matrix_world
        vertices.extend(transform @ vertex.co for vertex in mesh.vertices)
        polygons.extend(
            tuple(offset + index for index in polygon.vertices)
            for polygon in mesh.polygons
        )
        evaluated.to_mesh_clear()
    if not vertices or not polygons:
        raise ValueError("cannot build BVH from empty objects")
    return BVHTree.FromPolygons(vertices, polygons, all_triangles=False)


def compare_contact_surfaces(
    visual: list[Any],
    proxy: list[Any],
    usage: dict[str, Any],
    low: np.ndarray,
    high: np.ndarray,
) -> dict[str, Any]:
    visual_bvh = world_bvh(visual)
    proxy_bvh = world_bvh(proxy)
    safe = usage["safe_surface"]
    center = np.asarray(safe["center_xy_m"], dtype=np.float64)
    size = np.asarray(safe["size_xy_m"], dtype=np.float64)
    xs = np.linspace(center[0] - 0.42 * size[0], center[0] + 0.42 * size[0], 13)
    ys = np.linspace(center[1] - 0.42 * size[1], center[1] + 0.42 * size[1], 9)
    ray_top = float(high[2] + max(0.5, high[2] - low[2]))
    ray_length = float(ray_top - low[2] + 0.5)
    down = mathutils.Vector((0.0, 0.0, -1.0))
    plane = float(usage["target_support_plane_z_m"])
    union_hits = 0
    paired_hits = 0
    height_errors: list[float] = []
    support_height_errors: list[float] = []
    paired_samples: list[dict[str, float]] = []
    unmatched_samples: list[dict[str, float | str]] = []
    visual_only = 0
    proxy_only = 0
    for y in ys:
        for x in xs:
            origin = mathutils.Vector((float(x), float(y), ray_top))
            visual_hit = visual_bvh.ray_cast(origin, down, ray_length)[0]
            proxy_hit = proxy_bvh.ray_cast(origin, down, ray_length)[0]
            if visual_hit is None and proxy_hit is None:
                continue
            union_hits += 1
            if visual_hit is None:
                proxy_only += 1
                unmatched_samples.append(
                    {"x_m": float(x), "y_m": float(y), "kind": "proxy_only"}
                )
                continue
            if proxy_hit is None:
                visual_only += 1
                unmatched_samples.append(
                    {"x_m": float(x), "y_m": float(y), "kind": "visual_only"}
                )
                continue
            paired_hits += 1
            error = abs(float(visual_hit.z) - float(proxy_hit.z))
            height_errors.append(error)
            paired_samples.append(
                {
                    "x_m": float(x),
                    "y_m": float(y),
                    "visual_z_m": float(visual_hit.z),
                    "proxy_z_m": float(proxy_hit.z),
                    "absolute_error_m": error,
                }
            )
            if abs(float(proxy_hit.z) - plane) <= 0.02:
                support_height_errors.append(error)
    hit_match_fraction = 1.0 if union_hits == 0 else paired_hits / union_hits
    max_support_error = (
        None if not support_height_errors else max(support_height_errors)
    )
    passed = bool(
        paired_hits >= 12
        and hit_match_fraction >= 0.95
        and max_support_error is not None
        and max_support_error <= 0.004
    )
    return {
        "tested_rays": int(len(xs) * len(ys)),
        "union_hit_count": union_hits,
        "paired_hit_count": paired_hits,
        "visual_only_hit_count": visual_only,
        "proxy_only_hit_count": proxy_only,
        "hit_match_fraction": hit_match_fraction,
        "maximum_paired_height_error_m": (
            None if not height_errors else max(height_errors)
        ),
        "support_plane_pair_count": len(support_height_errors),
        "maximum_support_plane_height_error_m": max_support_error,
        "support_plane_height_tolerance_m": 0.004,
        "largest_height_error_samples": sorted(
            paired_samples,
            key=lambda sample: sample["absolute_error_m"],
            reverse=True,
        )[:8],
        "unmatched_hit_samples": unmatched_samples[:8],
        "passed": passed,
    }


def matrix(value: list[list[float]]) -> mathutils.Matrix:
    return mathutils.Matrix(tuple(tuple(float(item) for item in row) for row in value))


def runtime_transform(binding: dict[str, Any]) -> mathutils.Matrix:
    mesh = binding["mesh"]
    scale = [float(value) for value in mesh["scale"]]
    return mathutils.Matrix.Translation(
        tuple(float(value) for value in mesh["base_position_m"])
    ) @ mathutils.Matrix.Diagonal(mathutils.Vector((*scale, 1.0)))


def proxy_material() -> Any:
    material = bpy.data.materials.new("exact_collision_wire")
    material.diffuse_color = (0.02, 0.95, 0.22, 0.0)
    material.use_nodes = True
    material.blend_method = "BLEND"
    material.show_transparent_back = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    emission = nodes.new("ShaderNodeEmission")
    wire = nodes.new("ShaderNodeWireframe")
    wire.use_pixel_size = True
    wire.inputs["Size"].default_value = 0.8
    mix = nodes.new("ShaderNodeMixShader")
    emission.inputs["Color"].default_value = (0.01, 1.0, 0.12, 1.0)
    emission.inputs["Strength"].default_value = 1.3
    material.node_tree.links.new(wire.outputs["Fac"], mix.inputs["Fac"])
    material.node_tree.links.new(transparent.outputs["BSDF"], mix.inputs[1])
    material.node_tree.links.new(emission.outputs["Emission"], mix.inputs[2])
    material.node_tree.links.new(mix.outputs["Shader"], output.inputs["Surface"])
    return material


def look_at(obj: Any, target: np.ndarray) -> None:
    obj.rotation_euler = (
        mathutils.Vector(tuple(float(value) for value in target)) - obj.location
    ).to_track_quat("-Z", "Y").to_euler()


def setup_scene(
    output: Path,
    resolution: list[int],
    low: np.ndarray,
    high: np.ndarray,
) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = 32
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output)
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    world = scene.world or bpy.data.worlds.new("ProxyOverlayWorld")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.055,
        0.065,
        0.075,
        1.0,
    )
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.55

    center = (low + high) * 0.5
    extent = np.maximum(high - low, 0.1)
    diagonal = float(np.linalg.norm(extent))
    bpy.ops.object.camera_add(
        location=tuple(center + np.asarray([1.25, -1.55, 1.05]) * diagonal)
    )
    camera = bpy.context.object
    camera.data.lens = 54.0
    look_at(camera, center + np.asarray([0.0, 0.0, 0.08 * extent[2]]))
    scene.camera = camera

    light_data = bpy.data.lights.new("key", type="AREA")
    light_data.energy = 650.0 * max(0.8, min(1.5, diagonal / 2.5))
    light_data.size = max(1.2, 0.7 * diagonal)
    light = bpy.data.objects.new("key", light_data)
    bpy.context.collection.objects.link(light)
    light.location = tuple(center + np.asarray([-0.8, -1.0, 1.8]) * diagonal)
    look_at(light, center)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def render_record(
    root: Path,
    output: Path,
    resolution: list[int],
    record: dict[str, Any],
    usage: dict[str, Any],
    registry_record: dict[str, Any] | None,
) -> dict[str, Any]:
    binding = compile_static_support_binding(
        record,
        target_size_xy_m=usage["target_size_xy_m"],
        target_center_xy_m=usage["target_center_xy_m"],
        target_support_plane_z_m=usage["target_support_plane_z_m"],
        usage_id=usage["id"],
        maximum_axis_scale_ratio=usage["maximum_axis_scale_ratio"],
    )
    clear_scene()
    visual_path = root / str(record["source"]["visual_path"])
    include_names = []
    if registry_record is not None:
        include_names = [
            str(value)
            for value in registry_record["visual"].get("include_object_names", [])
        ]
    extraction = record["qa"]["geometry"].get("extraction", {})
    reference_frame = int(extraction.get("reference_frame", 1))
    visual, visual_root = import_glb(visual_path, include_names, reference_frame)
    canonical = matrix(record["qa"]["geometry"]["canonical_transform_after_z_up"])
    target_transform = runtime_transform(binding)
    visual_root.matrix_world = target_transform @ canonical

    proxy = import_obj(root / str(binding["mesh"]["path"]))
    wire_material = proxy_material()
    for obj in proxy:
        obj.matrix_world = target_transform @ obj.matrix_world
    bpy.context.view_layer.update()
    visual_low, visual_high = bounds(visual)
    proxy_low, proxy_high = bounds(proxy)
    combined_low = np.minimum(visual_low, proxy_low)
    combined_high = np.maximum(visual_high, proxy_high)
    contact_surfaces = compare_contact_surfaces(
        visual,
        proxy,
        usage,
        combined_low,
        combined_high,
    )
    diagonal = float(np.linalg.norm(visual_high - visual_low))
    for obj in proxy:
        obj.data.materials.clear()
        obj.data.materials.append(wire_material)
    bound_error = float(
        max(
            np.max(np.abs(visual_low - proxy_low)),
            np.max(np.abs(visual_high - proxy_high)),
        )
    )
    bound_tolerance = max(0.002, diagonal * 0.005)
    bounds_passed = bool(bound_error <= bound_tolerance)
    setup_scene(output, resolution, combined_low, combined_high)
    bpy.ops.render.render(write_still=True)
    return {
        "asset_id": record["asset_id"],
        "usage_id": usage["id"],
        "image_path": str(output.relative_to(root)).replace("\\", "/"),
        "visual_bounds_m": [visual_low.tolist(), visual_high.tolist()],
        "proxy_bounds_m": [proxy_low.tolist(), proxy_high.tolist()],
        "maximum_bound_error_m": bound_error,
        "bound_tolerance_m": bound_tolerance,
        "bounds_passed": bounds_passed,
        "contact_surfaces": contact_surfaces,
        "passed": bool(bounds_passed and contact_surfaces["passed"]),
    }


def main() -> None:
    args = blender_args()
    root = args.root.resolve()
    catalog_path = args.catalog if args.catalog.is_absolute() else root / args.catalog
    manifest, records = load_catalog(root, catalog_path)
    registry_path = args.registry if args.registry.is_absolute() else root / args.registry
    registry = {
        str(item["asset_id"]): item
        for item in load_json(registry_path)["records"]
    }
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    results = []
    static_records = [
        record
        for record in records
        if record["proxy"]["representation"] == "static_concave_mesh"
    ]
    for record in static_records:
        for usage in record["proxy"]["usages"]:
            image_path = output / (
                f"{record['asset_id']}__{slug(str(usage['id']))}.png"
            )
            results.append(
                render_record(
                    root,
                    image_path,
                    [int(value) for value in args.resolution],
                    record,
                    usage,
                    registry.get(str(record["asset_id"])),
                )
            )
    report = {
        "version": "physweep_static_support_visual_proxy_validation_v2",
        "catalog_records_sha256": str(manifest["records_sha256"]),
        "implementation": {
            "path": str(Path(__file__).resolve().relative_to(root)).replace(
                "\\", "/"
            ),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "coordinate_contract": {
            "visual_import": "glTF converted to Blender Z-up",
            "proxy_import": "direct OBJ vertices in PhysSweep/PyBullet Z-up",
        },
        "counts": {
            "tested": len(results),
            "passed": sum(bool(result["passed"]) for result in results),
            "failed": sum(not bool(result["passed"]) for result in results),
        },
        "records": results,
    }
    write_json(output / "report.json", report)
    catalog_report = (
        args.catalog_report
        if args.catalog_report.is_absolute()
        else root / args.catalog_report
    )
    write_json(catalog_report, report)
    updated_manifest = copy.deepcopy(manifest)
    updated_manifest["visual_validation"] = {
        "path": str(catalog_report.relative_to(root)).replace("\\", "/"),
        "sha256": sha256(catalog_report),
        "counts": copy.deepcopy(report["counts"]),
        "scope": "all_static_support_usages",
        "version": report["version"],
    }
    write_json(catalog_path, updated_manifest)
    print(json.dumps(report["counts"], ensure_ascii=True))
    if report["counts"]["failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
