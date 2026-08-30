"""Assemble reviewed 1obj records without changing their object appearance."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence


_ANNOTATION_KEYS = ("semantic_category", "scale_bin", "uniform_scale")


def _is_canonical_base(metadata: Mapping[str, Any]) -> bool:
    sweep = metadata.get("sweep")
    if sweep is None:
        return False
    if not isinstance(sweep, Mapping) or sweep.get("kind") != "base":
        raise ValueError("object candidates must be unswept or canonical bases")
    return True


def _object_id(role: Mapping[str, Any]) -> str:
    if set(role) != {"object_id"}:
        raise ValueError("object roles may only declare object_id")
    object_id = str(role["object_id"]).strip()
    if not object_id:
        raise ValueError("object role identity is empty")
    return object_id


def _source_parts(
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    if metadata.get("schema_version") != "physweep_pybullet_rigid_metadata_v1":
        raise ValueError("object candidates must use rigid metadata v1")
    _is_canonical_base(metadata)
    simulation = metadata.get("simulation")
    if not isinstance(simulation, Mapping):
        raise ValueError("object candidate lacks a simulation record")
    objects = simulation.get("objects")
    if not isinstance(objects, list) or len(objects) != 1:
        raise ValueError("each object candidate must contain one dynamic object")
    obj = objects[0]
    if not isinstance(obj, dict) or obj.get("body_model") != "rigid_body":
        raise ValueError("object candidates must contain one rigid body")
    if not isinstance(obj.get("visual_profile"), dict):
        raise ValueError("object candidate lacks a visual profile")

    semantic_sampling = metadata.get("semantic_sampling")
    if not isinstance(semantic_sampling, Mapping):
        raise ValueError("object candidate lacks semantic sampling")
    dimensions = semantic_sampling.get("five_dimensions")
    if not isinstance(dimensions, Mapping):
        raise ValueError("object candidate lacks semantic dimensions")
    foreground = dimensions.get("foreground_object")
    if not isinstance(foreground, dict) or any(
        key not in foreground for key in _ANNOTATION_KEYS
    ):
        raise ValueError("object candidate lacks foreground annotations")

    appearance = metadata.get("appearance")
    if not isinstance(appearance, Mapping):
        raise ValueError("object candidate lacks appearance")
    materials = appearance.get("materials")
    if not isinstance(materials, Mapping):
        raise ValueError("object candidate lacks appearance materials")
    material = materials.get("dynamic_object")
    if material is not None and not isinstance(material, dict):
        raise ValueError("object candidate has an invalid dynamic appearance material")
    if material is None:
        visual = obj.get("visual_profile")
        if not isinstance(visual, dict) or (
            visual.get("type") != "mesh"
            or visual.get("material_policy") != "source_or_bound_fallback"
        ):
            raise ValueError("object candidate lacks a dynamic appearance material")
    return obj, foreground, material


def compile_object_collection_scene(
    host_template: Mapping[str, Any],
    object_templates: Sequence[Mapping[str, Any]],
    roles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Inject any positive number of reviewed objects into one frozen host."""

    if not object_templates or len(object_templates) != len(roles):
        raise ValueError("object collection requires one role per candidate")
    object_ids = [_object_id(role) for role in roles]
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("object role ids must be unique")

    scene = copy.deepcopy(host_template)
    if scene.get("schema_version") != "physweep_pybullet_rigid_metadata_v1":
        raise ValueError("object collection host must use rigid metadata v1")
    canonical_host = _is_canonical_base(scene)
    simulation = scene.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError("object collection host lacks a simulation record")
    semantic_sampling = scene.get("semantic_sampling")
    if not isinstance(semantic_sampling, dict):
        raise ValueError("object collection host lacks semantic sampling")
    dimensions = semantic_sampling.get("five_dimensions")
    if not isinstance(dimensions, dict):
        raise ValueError("object collection host lacks semantic dimensions")
    dimensions.pop("foreground_object", None)

    appearance = scene.get("appearance")
    if not isinstance(appearance, dict):
        raise ValueError("object collection host lacks appearance")
    materials = appearance.get("materials")
    if not isinstance(materials, dict):
        raise ValueError("object collection host lacks appearance materials")
    host_dynamic_material = materials.get("dynamic_object")
    if not isinstance(host_dynamic_material, dict):
        raise ValueError("object collection host lacks a fallback dynamic material")
    materials.pop("dynamic_object", None)

    objects = []
    foreground_objects = []
    dynamic_materials = {}
    for template, object_id in zip(object_templates, object_ids):
        source, foreground, material = _source_parts(template)
        obj = copy.deepcopy(source)
        obj["object_id"] = object_id
        objects.append(obj)
        foreground_objects.append(
            {
                "object_id": object_id,
                **{
                    key: copy.deepcopy(foreground[key])
                    for key in _ANNOTATION_KEYS
                },
            }
        )
        dynamic_materials[object_id] = copy.deepcopy(
            material if material is not None else host_dynamic_material
        )

    simulation["objects"] = objects
    dimensions["foreground_objects"] = foreground_objects
    materials["dynamic_objects"] = dynamic_materials
    if canonical_host and str(scene["scene_id"]).endswith("__base"):
        scene["scene_id"] = str(scene["scene_id"])[: -len("__base")]
    scene.pop("sweep", None)
    scene.pop("object_identity", None)
    return scene
