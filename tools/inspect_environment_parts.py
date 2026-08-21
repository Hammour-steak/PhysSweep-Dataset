#!/usr/bin/env python3
"""Inspect imported GLB mesh parts for removable room shells."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    args = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("assets", nargs="+", type=Path)
    return parser.parse_args(args)


def reset_scene() -> None:
    import bpy

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def import_glb(path: Path) -> list[object]:
    import bpy
    import numpy as np

    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def world_bounds(obj: object) -> tuple[list[float], list[float]]:
    import mathutils

    points = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    low = [min(float(point[axis]) for point in points) for axis in range(3)]
    high = [max(float(point[axis]) for point in points) for axis in range(3)]
    return low, high


def upward_surface_clusters(obj: object) -> list[dict[str, float | int]]:
    """Return large, nearly horizontal, upward-facing surfaces in world space."""
    clusters: dict[float, dict[str, float | int]] = defaultdict(
        lambda: {"area": 0.0, "faces": 0}
    )
    matrix = obj.matrix_world
    mesh = obj.data
    for polygon in mesh.polygons:
        points = [matrix @ mesh.vertices[index].co for index in polygon.vertices]
        if len(points) < 3:
            continue
        cross = (points[1] - points[0]).cross(points[2] - points[0])
        if cross.length == 0.0 or cross.normalized().z < 0.98:
            continue
        z_values = [float(point.z) for point in points]
        if max(z_values) - min(z_values) > 1e-4:
            continue
        area = 0.0
        for index in range(1, len(points) - 1):
            area += 0.5 * (points[index] - points[0]).cross(
                points[index + 1] - points[0]
            ).length
        z_key = round(sum(z_values) / len(z_values), 4)
        clusters[z_key]["area"] = float(clusters[z_key]["area"]) + area
        clusters[z_key]["faces"] = int(clusters[z_key]["faces"]) + 1
    return [
        {"z": z, "area": round(float(values["area"]), 6), "faces": int(values["faces"])}
        for z, values in sorted(clusters.items(), key=lambda item: item[1]["area"], reverse=True)
    ]


def horizontal_surface_clusters(obj: object) -> list[dict[str, float | int | str]]:
    """Return nearly horizontal surfaces, including ceiling undersides."""
    clusters: dict[tuple[float, str], dict[str, float | int]] = defaultdict(
        lambda: {"area": 0.0, "faces": 0}
    )
    matrix = obj.matrix_world
    mesh = obj.data
    for polygon in mesh.polygons:
        points = [matrix @ mesh.vertices[index].co for index in polygon.vertices]
        if len(points) < 3:
            continue
        cross = (points[1] - points[0]).cross(points[2] - points[0])
        if cross.length == 0.0 or abs(cross.normalized().z) < 0.98:
            continue
        z_values = [float(point.z) for point in points]
        if max(z_values) - min(z_values) > 1e-4:
            continue
        area = 0.0
        for index in range(1, len(points) - 1):
            area += 0.5 * (points[index] - points[0]).cross(
                points[index + 1] - points[0]
            ).length
        orientation = "up" if cross.normalized().z > 0.0 else "down"
        key = (round(sum(z_values) / len(z_values), 4), orientation)
        clusters[key]["area"] = float(clusters[key]["area"]) + area
        clusters[key]["faces"] = int(clusters[key]["faces"]) + 1
    return [
        {
            "z": z,
            "orientation": orientation,
            "area": round(float(values["area"]), 6),
            "faces": int(values["faces"]),
        }
        for (z, orientation), values in sorted(
            clusters.items(), key=lambda item: item[1]["area"], reverse=True
        )
    ]


def axis_aligned_surface_clusters(obj: object) -> list[dict[str, object]]:
    """Return planar X/Y/Z-facing source surfaces with their in-plane bounds."""
    clusters: dict[tuple[str, float, str], dict[str, object]] = {}
    matrix = obj.matrix_world
    mesh = obj.data
    axis_names = ("x", "y", "z")
    for polygon in mesh.polygons:
        points = [matrix @ mesh.vertices[index].co for index in polygon.vertices]
        if len(points) < 3:
            continue
        cross = (points[1] - points[0]).cross(points[2] - points[0])
        if cross.length == 0.0:
            continue
        normal = cross.normalized()
        axis = max(range(3), key=lambda index: abs(float(normal[index])))
        if abs(float(normal[axis])) < 0.98:
            continue
        coordinates = [float(point[axis]) for point in points]
        if max(coordinates) - min(coordinates) > 1e-4:
            continue
        area = 0.0
        for index in range(1, len(points) - 1):
            area += 0.5 * (points[index] - points[0]).cross(
                points[index + 1] - points[0]
            ).length
        orientation = "positive" if float(normal[axis]) > 0.0 else "negative"
        key = (
            axis_names[axis],
            round(sum(coordinates) / len(coordinates), 4),
            orientation,
        )
        if key not in clusters:
            clusters[key] = {
                "area": 0.0,
                "faces": 0,
                "bbox_min": [float("inf")] * 3,
                "bbox_max": [float("-inf")] * 3,
            }
        cluster = clusters[key]
        cluster["area"] = float(cluster["area"]) + area
        cluster["faces"] = int(cluster["faces"]) + 1
        for point in points:
            for point_axis in range(3):
                cluster["bbox_min"][point_axis] = min(
                    float(cluster["bbox_min"][point_axis]), float(point[point_axis])
                )
                cluster["bbox_max"][point_axis] = max(
                    float(cluster["bbox_max"][point_axis]), float(point[point_axis])
                )
    return [
        {
            "axis": axis,
            "coordinate": coordinate,
            "orientation": orientation,
            "area": round(float(values["area"]), 6),
            "faces": int(values["faces"]),
            "bbox_min": [round(float(value), 6) for value in values["bbox_min"]],
            "bbox_max": [round(float(value), 6) for value in values["bbox_max"]],
        }
        for (axis, coordinate, orientation), values in sorted(
            clusters.items(), key=lambda item: item[1]["area"], reverse=True
        )
    ]


def axis_aligned_faces(obj: object) -> list[dict[str, object]]:
    """Return individual source-space planar faces for surgical shell review."""
    matrix = obj.matrix_world
    mesh = obj.data
    axis_names = ("x", "y", "z")
    records = []
    for polygon in mesh.polygons:
        points = [matrix @ mesh.vertices[index].co for index in polygon.vertices]
        if len(points) < 3:
            continue
        cross = (points[1] - points[0]).cross(points[2] - points[0])
        if cross.length == 0.0:
            continue
        normal = cross.normalized()
        axis = max(range(3), key=lambda index: abs(float(normal[index])))
        if abs(float(normal[axis])) < 0.98:
            continue
        coordinates = [float(point[axis]) for point in points]
        if max(coordinates) - min(coordinates) > 1e-4:
            continue
        area = 0.0
        for index in range(1, len(points) - 1):
            area += 0.5 * (points[index] - points[0]).cross(
                points[index + 1] - points[0]
            ).length
        records.append(
            {
                "polygon_index": int(polygon.index),
                "axis": axis_names[axis],
                "coordinate": round(sum(coordinates) / len(coordinates), 6),
                "orientation": "positive" if float(normal[axis]) > 0.0 else "negative",
                "area": round(float(area), 6),
                "bbox_min": [
                    round(min(float(point[index]) for point in points), 6)
                    for index in range(3)
                ],
                "bbox_max": [
                    round(max(float(point[index]) for point in points), 6)
                    for index in range(3)
                ],
            }
        )
    return sorted(records, key=lambda record: float(record["area"]), reverse=True)


def material_settings(obj: object) -> list[dict[str, object]]:
    """Report imported transparency settings that can affect Eevee shadows."""
    records = []
    for material in obj.data.materials:
        if material is None:
            continue
        settings: dict[str, object] = {
            "name": material.name,
            "blend_method": str(getattr(material, "blend_method", "OPAQUE")),
            "shadow_method": str(getattr(material, "shadow_method", "OPAQUE")),
            "diffuse_alpha": round(float(material.diffuse_color[3]), 6),
        }
        if material.use_nodes and material.node_tree:
            principled = next(
                (
                    node
                    for node in material.node_tree.nodes
                    if node.type == "BSDF_PRINCIPLED"
                ),
                None,
            )
            if principled is not None:
                for input_name in ("Alpha", "Transmission", "Transmission Weight"):
                    socket = principled.inputs.get(input_name)
                    if socket is not None and not socket.is_linked:
                        settings[input_name.lower().replace(" ", "_")] = round(
                            float(socket.default_value), 6
                        )
        records.append(settings)
    return records


def main() -> None:
    args = parse_args()
    records = []
    for path in args.assets:
        reset_scene()
        meshes = import_glb(path.resolve())
        parts = []
        for obj in meshes:
            low, high = world_bounds(obj)
            horizontal_surfaces = upward_surface_clusters(obj)
            all_horizontal_surfaces = horizontal_surface_clusters(obj)
            axis_aligned_surfaces = axis_aligned_surface_clusters(obj)
            planar_faces = axis_aligned_faces(obj)
            parts.append(
                {
                    "name": obj.name,
                    "mesh_name": obj.data.name,
                    "vertices": len(obj.data.vertices),
                    "polygons": len(obj.data.polygons),
                    "bbox_min": [round(value, 6) for value in low],
                    "bbox_max": [round(value, 6) for value in high],
                    "bbox_size": [round(high[i] - low[i], 6) for i in range(3)],
                    "materials": [material.name for material in obj.data.materials if material],
                    "material_settings": material_settings(obj),
                    "upward_surfaces": horizontal_surfaces[:20],
                    "horizontal_surfaces": all_horizontal_surfaces[:30],
                    "axis_aligned_surfaces": axis_aligned_surfaces[:60],
                    "axis_aligned_faces": planar_faces[:120],
                }
            )
        parts.sort(key=lambda part: part["polygons"], reverse=True)
        records.append({"path": str(path), "mesh_count": len(parts), "parts": parts})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"records": records}, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
