#!/usr/bin/env python3
"""Extract static support collision meshes from Blender-evaluated GLB geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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

from tools.assets.physical_proxy_catalog import load_json, read_jsonl, write_json  # noqa: E402


EXTRACTION_VERSION = "physweep_blender_static_support_extraction_v1"


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
    parser.add_argument("--reference-frame", type=int, default=1)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("assets/proxies/v1/static_blender_extraction_report.json"),
    )
    return parser.parse_args(values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
        bpy.data.armatures,
        bpy.data.actions,
    ):
        for item in list(collection):
            if item.users == 0:
                collection.remove(item)


def import_visual(path: Path, include_names: list[str], frame: int) -> list[Any]:
    clear_scene()
    bpy.context.scene.frame_set(frame)
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    imported = [obj for obj in bpy.context.scene.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if include_names:
        requested = set(include_names)
        meshes = [
            obj
            for obj in meshes
            if obj.name in requested or obj.data.name in requested
        ]
        if not meshes:
            raise ValueError(f"no requested visual components found: {include_names}")
    if not meshes:
        raise ValueError(f"no visual mesh in {path}")
    bpy.context.scene.frame_set(frame)
    bpy.context.view_layer.update()
    return meshes


def evaluated_triangles(
    objects: list[Any], canonical: mathutils.Matrix
) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]], list[str]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    selected_names: list[str] = []
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        transform = canonical @ evaluated.matrix_world
        offset = len(vertices)
        world_vertices = [transform @ vertex.co for vertex in mesh.vertices]
        vertices.extend(tuple(float(value) for value in vertex) for vertex in world_vertices)
        mesh.calc_loop_triangles()
        for triangle in mesh.loop_triangles:
            indices = tuple(int(index) for index in triangle.vertices)
            a, b, c = (world_vertices[index] for index in indices)
            if (b - a).cross(c - a).length_squared <= 1.0e-20:
                continue
            faces.append(tuple(offset + index for index in indices))
        selected_names.append(str(obj.name))
        evaluated.to_mesh_clear()
    if not vertices or not faces:
        raise ValueError("evaluated support contains no finite triangle geometry")
    values = np.asarray(vertices, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("evaluated support contains non-finite vertices")
    return vertices, faces, sorted(selected_names)


def write_obj(
    path: Path,
    vertices: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as output:
        output.write("# PhysSweep Blender-evaluated static support, Z-up\n")
        output.writelines(
            f"v {x:.12g} {y:.12g} {z:.12g}\n" for x, y, z in vertices
        )
        output.writelines(
            f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces
        )


def extraction_record(
    root: Path,
    record: dict[str, Any],
    registry_record: dict[str, Any] | None,
    reference_frame: int,
) -> dict[str, Any]:
    source = root / str(record["source"]["visual_path"])
    if sha256(source) != str(record["source"]["sha256"]):
        raise ValueError(f"visual source hash changed: {record['asset_id']}")
    include_names = []
    if registry_record is not None:
        include_names = sorted(
            str(value)
            for value in registry_record["visual"].get("include_object_names", [])
        )
    meshes = import_visual(source, include_names, reference_frame)
    canonical_values = record["qa"]["geometry"][
        "canonical_transform_after_z_up"
    ]
    canonical = mathutils.Matrix(
        tuple(tuple(float(value) for value in row) for row in canonical_values)
    )
    vertices, faces, selected_names = evaluated_triangles(meshes, canonical)
    collision_path = root / str(record["proxy"]["mesh"]["path"])
    write_obj(collision_path, vertices, faces)
    values = np.asarray(vertices, dtype=np.float64)
    low = values.min(axis=0)
    high = values.max(axis=0)
    sidecar = {
        "version": EXTRACTION_VERSION,
        "asset_id": str(record["asset_id"]),
        "source_visual_path": str(record["source"]["visual_path"]),
        "source_visual_sha256": str(record["source"]["sha256"]),
        "include_object_names": include_names,
        "selected_object_names": selected_names,
        "reference_frame": int(reference_frame),
        "static_animation_policy": "evaluate_reference_frame_then_freeze",
        "coordinate_contract": {
            "source": "Blender glTF importer Z-up evaluated world geometry",
            "canonical_transform_after_z_up": canonical_values,
            "output": "PhysSweep/PyBullet Z-up OBJ",
        },
        "blender_version": bpy.app.version_string,
        "extractor": {
            "path": str(Path(__file__).resolve().relative_to(root)).replace(
                "\\", "/"
            ),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "mesh": {
            "path": str(collision_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256(collision_path),
            "vertex_count": len(vertices),
            "face_count": len(faces),
            "source_bounds": [low.round(9).tolist(), high.round(9).tolist()],
            "source_extents": (high - low).round(9).tolist(),
        },
    }
    sidecar_path = collision_path.with_name("blender_extraction.json")
    write_json(sidecar_path, sidecar)
    return {
        "asset_id": str(record["asset_id"]),
        "sidecar_path": str(sidecar_path.relative_to(root)).replace("\\", "/"),
        "mesh_sha256": sidecar["mesh"]["sha256"],
        "vertex_count": len(vertices),
        "face_count": len(faces),
    }


def main() -> None:
    args = blender_args()
    root = args.root.resolve()
    catalog_path = args.catalog if args.catalog.is_absolute() else root / args.catalog
    manifest = load_json(catalog_path)
    records = read_jsonl(root / str(manifest["records_path"]))
    registry_path = args.registry if args.registry.is_absolute() else root / args.registry
    registry = {
        str(item["asset_id"]): item for item in load_json(registry_path)["records"]
    }
    static_records = [
        record
        for record in records
        if record["proxy"]["representation"] == "static_concave_mesh"
    ]
    results = [
        extraction_record(
            root,
            record,
            registry.get(str(record["asset_id"])),
            int(args.reference_frame),
        )
        for record in static_records
    ]
    report = {
        "version": EXTRACTION_VERSION,
        "catalog_path": str(catalog_path.relative_to(root)).replace("\\", "/"),
        "reference_frame": int(args.reference_frame),
        "implementation": {
            "path": str(Path(__file__).resolve().relative_to(root)).replace(
                "\\", "/"
            ),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "counts": {"tested": len(results), "extracted": len(results), "failed": 0},
        "records": results,
    }
    report_path = args.report if args.report.is_absolute() else root / args.report
    write_json(report_path, report)
    print(json.dumps(report["counts"], ensure_ascii=True))


if __name__ == "__main__":
    main()
