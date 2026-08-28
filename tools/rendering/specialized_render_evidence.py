#!/usr/bin/env python3
"""Shared render-evidence writer for specialized Blender backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import bpy
import numpy as np

from tools.dataset_contract.immutable_scene_contract import sha256, write_json


MASK_MANIFEST_SCHEMA = "physweep_instance_mask_manifest_v2"
MASK_RENDER_RECORD_SCHEMA = "physweep_specialized_mask_render_record_v1"


def render_implementation(renderer_path: Path) -> dict[str, Any]:
    renderer_path = renderer_path.resolve()
    evidence_path = Path(__file__).resolve()
    return {
        "renderer": {
            "path": str(renderer_path),
            "sha256": sha256(renderer_path),
        },
        "render_evidence": {
            "path": str(evidence_path),
            "sha256": sha256(evidence_path),
        },
    }


def _mask_root(
    root: Path,
    metadata: Mapping[str, Any],
    override: Path | None,
) -> Path:
    if override is not None:
        result = override.resolve()
    else:
        declared = metadata["object_identity"]["instance_masks"].get("path")
        if not isinstance(declared, str) or not declared:
            raise ValueError("specialized mask path is absent and no override was given")
        relative = Path(declared)
        if relative.is_absolute():
            raise ValueError("specialized mask path must be project-relative")
        result = (root / relative).resolve()
    result.relative_to(root.resolve())
    if (root / "outputs").resolve() not in result.parents:
        raise ValueError("specialized masks must remain below root/outputs")
    return result


def _white_emission_material(name: str) -> Any:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def _probe_mask(path: Path) -> dict[str, Any]:
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        width, height = [int(value) for value in image.size]
        rgba = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)
        alpha = rgba[:, :, 3]
        if (
            not np.isfinite(alpha).all()
            or float(alpha.min()) < 0.0
            or float(alpha.max()) > 1.0
        ):
            raise ValueError(f"instance mask alpha is invalid: {path}")
        occupancy = float(np.mean(alpha > 1.0e-6))
        soft_edge = float(np.mean((alpha > 1.0e-6) & (alpha < 1.0 - 1.0e-6)))
    finally:
        bpy.data.images.remove(image)
    if not 0.0 < occupancy < 1.0:
        raise ValueError("initial instance mask must be nonempty and non-full")
    if soft_edge <= 0.0:
        raise ValueError("instance mask does not contain antialiased edge pixels")
    return {
        "initial_occupancy_fraction": round(occupancy, 9),
        "initial_soft_edge_fraction": round(soft_edge, 9),
    }


def render_instance_masks(
    *,
    root: Path,
    metadata: Mapping[str, Any],
    dynamic_objects: Mapping[str, Sequence[Any]],
    mask_root_override: Path | None = None,
) -> dict[str, Any]:
    """Render one unoccluded silhouette sequence per declared identity object."""
    identity_objects = {
        str(record["object_id"]): record
        for record in metadata["object_identity"]["objects"]
    }
    mask_objects = metadata["object_identity"]["instance_masks"]["objects"]
    expected_ids = set(identity_objects)
    if set(dynamic_objects) != expected_ids or set(mask_objects) != expected_ids:
        raise ValueError("rendered objects do not match the object identity contract")
    if any(not objects for objects in dynamic_objects.values()):
        raise ValueError("every identity object must have at least one render object")

    scene = bpy.context.scene
    frames = list(range(int(scene.frame_start), int(scene.frame_end) + 1))
    mask_root = _mask_root(root, metadata, mask_root_override)
    renderables = [
        obj
        for obj in scene.objects
        if obj.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"}
    ]
    scene.render.use_compositing = False
    scene.use_nodes = False
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    mask_samples = max(8, min(int(metadata["render"]["samples"]), 16))
    if scene.render.engine == "BLENDER_EEVEE":
        scene.eevee.taa_render_samples = mask_samples

    manifest_objects: dict[str, Any] = {}
    for object_id in sorted(expected_ids):
        selected = set(dynamic_objects[object_id])
        for obj in renderables:
            obj.hide_render = obj not in selected
        material = _white_emission_material(f"{object_id}_mask_material")
        for obj in selected:
            if obj.type != "MESH":
                raise ValueError(f"mask object must be a mesh: {object_id}")
            obj.data.materials.clear()
            obj.data.materials.append(material)
        object_dir = mask_root / object_id
        object_dir.mkdir(parents=True, exist_ok=True)
        for stale in object_dir.glob("frame_*.png"):
            stale.unlink()
        paths = []
        for frame in frames:
            scene.frame_set(frame)
            path = object_dir / f"frame_{frame:04d}.png"
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            paths.append(path)
        validation = _probe_mask(paths[0])
        records = [
            {"filename": path.name, "sha256": sha256(path)} for path in paths
        ]
        instance_id = int(mask_objects[object_id]["instance_id"])
        manifest_objects[object_id] = {
            "instance_id": instance_id,
            "validation": validation,
            "records": records,
        }

    manifest_path = mask_root / "mask_manifest.json"
    manifest = {
        "schema_version": MASK_MANIFEST_SCHEMA,
        "scene_id": str(metadata["scene_id"]),
        "frame_count": len(frames),
        "objects": manifest_objects,
    }
    write_json(manifest_path, manifest)
    return {
        "render_samples": mask_samples,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
    }


def render_instance_mask_record(
    *,
    root: Path,
    metadata_path: Path,
    metadata: Mapping[str, Any],
    camera: Mapping[str, Any],
    dynamic_objects: Mapping[str, Sequence[Any]],
    mask_root: Path,
    renderer_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": MASK_RENDER_RECORD_SCHEMA,
        "scene_id": str(metadata["scene_id"]),
        "metadata_sha256": sha256(metadata_path),
        "metadata_path": str(metadata_path),
        "render_scope": "instance_masks_only",
        "camera": dict(camera),
        "render_engine": bpy.context.scene.render.engine,
        "mask_resolution": [int(value) for value in metadata["render"]["resolution"]],
        "instance_mask_output": render_instance_masks(
            root=root,
            metadata=metadata,
            dynamic_objects=dynamic_objects,
            mask_root_override=mask_root,
        ),
        "implementation": render_implementation(renderer_path),
    }
