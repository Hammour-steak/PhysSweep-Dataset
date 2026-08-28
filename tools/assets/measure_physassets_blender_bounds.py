#!/usr/bin/env python3
"""Measure source GLB bounds exactly as Blender imports them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def blender_args() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(values)


def main() -> None:
    import bpy
    import mathutils
    import numpy as np

    if "bool" not in np.__dict__:
        np.bool = np.bool_

    args = blender_args()
    rows = [json.loads(line) for line in args.input_index.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = []
    for row in rows:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=False)
        before = set(bpy.context.scene.objects)
        bpy.ops.import_scene.gltf(filepath=str(row["source_glb"]))
        meshes = [obj for obj in bpy.context.scene.objects if obj not in before and obj.type == "MESH"]
        if not meshes:
            raise RuntimeError(f"no renderable mesh for {row['sample_id']}")
        for obj in meshes:
            world_matrix = obj.matrix_world.copy()
            obj.parent = None
            obj.matrix_world = world_matrix
        bpy.context.view_layer.update()
        bpy.ops.object.select_all(action="DESELECT")
        for obj in meshes:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = meshes[0]
        bpy.ops.object.join()
        obj = bpy.context.object
        bpy.context.view_layer.update()
        low = mathutils.Vector((float("inf"),) * 3)
        high = mathutils.Vector((float("-inf"),) * 3)
        for corner in obj.bound_box:
            point = obj.matrix_world @ mathutils.Vector(corner)
            for axis in range(3):
                low[axis] = min(low[axis], point[axis])
                high[axis] = max(high[axis], point[axis])
        output.append({
            "sample_id": str(row["sample_id"]),
            "source_glb": str(row["source_glb"]),
            "blender_import_bounds": [list(low), list(high)],
            "blender_import_extent": [float(high[i] - low[i]) for i in range(3)],
            "mesh_object_count": len(meshes)
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in output), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(output)}, indent=2))


if __name__ == "__main__":
    main()
