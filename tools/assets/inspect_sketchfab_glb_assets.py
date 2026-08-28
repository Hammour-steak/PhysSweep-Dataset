#!/usr/bin/env python3
"""Inspect downloaded Sketchfab GLB assets with Blender."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json


def inspect_glb(path: Path) -> dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    if not hasattr(np, "bool"):
        np.bool = bool  # type: ignore[attr-defined]

    import bpy  # pylint: disable=import-outside-toplevel
    from mathutils import Vector  # pylint: disable=import-outside-toplevel

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    before_images = set(bpy.data.images)
    before_materials = set(bpy.data.materials)

    bpy.ops.import_scene.gltf(filepath=str(path))
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        return {"path": str(path), "status": "no_mesh_objects"}

    mins = Vector((float("inf"), float("inf"), float("inf")))
    maxs = Vector((float("-inf"), float("-inf"), float("-inf")))
    vertices = 0
    polygons = 0
    material_names: set[str] = set()
    for obj in mesh_objects:
        vertices += len(obj.data.vertices)
        polygons += len(obj.data.polygons)
        for slot in obj.material_slots:
            if slot.material:
                material_names.add(slot.material.name)
        for corner in obj.bound_box:
            point = obj.matrix_world @ Vector(corner)
            mins.x = min(mins.x, point.x)
            mins.y = min(mins.y, point.y)
            mins.z = min(mins.z, point.z)
            maxs.x = max(maxs.x, point.x)
            maxs.y = max(maxs.y, point.y)
            maxs.z = max(maxs.z, point.z)

    new_images = [img for img in bpy.data.images if img not in before_images]
    new_materials = [mat for mat in bpy.data.materials if mat not in before_materials]
    texture_images = [img.name for img in new_images if getattr(img, "source", "") != "GENERATED"]
    return {
        "path": str(path),
        "status": "ok",
        "mesh_object_count": len(mesh_objects),
        "mesh_object_names": sorted(obj.name for obj in mesh_objects),
        "vertex_count": vertices,
        "polygon_count": polygons,
        "material_count": len(material_names) or len(new_materials),
        "texture_image_count": len(texture_images),
        "texture_images": sorted(texture_images)[:30],
        "bbox_min": [round(v, 6) for v in mins],
        "bbox_max": [round(v, 6) for v in maxs],
        "bbox_size": [round(v, 6) for v in (maxs - mins)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect downloaded Sketchfab GLBs.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]
    args = parser.parse_args(argv)

    manifest = load_json(args.manifest)
    records = []
    for record in manifest.get("records", []):
        path = Path(record.get("archive_path", ""))
        if not path.exists():
            records.append({"candidate_id": record.get("candidate_id"), "status": "missing_archive", "path": str(path)})
            continue
        info = inspect_glb(path)
        info["candidate_id"] = record.get("candidate_id")
        info["name"] = record.get("name")
        info["archive_kind"] = record.get("archive_kind")
        info["sha256"] = record.get("sha256")
        records.append(info)
        print(record.get("candidate_id"), info.get("status"), "meshes", info.get("mesh_object_count"), "textures", info.get("texture_image_count"), "bbox", info.get("bbox_size"))

    summary = {
        "count": len(records),
        "ok": sum(1 for r in records if r.get("status") == "ok"),
        "with_textures": sum(1 for r in records if int(r.get("texture_image_count", 0) or 0) > 0),
    }
    write_json(args.output, {"manifest": str(args.manifest), "summary": summary, "records": records})
    print("output:", args.output)
    print("summary:", summary)


if __name__ == "__main__":
    main()
