"""Compile two reviewed 1obj object records into one ordered object pair."""

from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

from tools.core.rigid_geometry import finite_vector


PAIR_SIZE = 2
_ANNOTATION_KEYS = ("semantic_category", "scale_bin", "uniform_scale")


def _is_canonical_base(metadata: Mapping[str, Any]) -> bool:
    sweep = metadata.get("sweep")
    if sweep is None:
        return False
    if not isinstance(sweep, Mapping) or sweep.get("kind") != "base":
        raise ValueError("object candidates must be unswept or canonical bases")
    return True


def _role_record(role: Mapping[str, Any]) -> dict[str, Any]:
    if set(role) != {
        "object_id",
        "semantic_label",
        "semantic_color_srgb",
    }:
        raise ValueError("object-pair roles have unsupported fields")
    object_id = str(role["object_id"]).strip()
    semantic_label = str(role["semantic_label"]).strip()
    color = finite_vector(
        role["semantic_color_srgb"], 4, f"{object_id} semantic color"
    )
    if (
        not object_id
        or not semantic_label
        or any(value < 0.0 or value > 1.0 for value in color)
    ):
        raise ValueError("object-pair role identity is invalid")
    return {
        "object_id": object_id,
        "semantic_label": semantic_label,
        "semantic_color_srgb": color,
    }


def _source_parts(
    metadata: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if metadata.get("schema_version") != "physweep_pybullet_rigid_metadata_v1":
        raise ValueError("object candidates must use rigid metadata v1")
    _is_canonical_base(metadata)
    simulation = metadata.get("simulation")
    if not isinstance(simulation, Mapping):
        raise ValueError("object candidate lacks a simulation record")
    objects = simulation.get("objects")
    if not isinstance(objects, list):
        raise ValueError("object candidate lacks simulation objects")
    if len(objects) != 1 or not isinstance(objects[0], dict):
        raise ValueError("each object candidate must contain one dynamic object")
    obj = objects[0]
    if obj.get("body_model") != "rigid_body":
        raise ValueError("object candidates must contain rigid bodies")
    visual_profile = obj.get("visual_profile")
    if not isinstance(visual_profile, dict):
        raise ValueError("object candidate lacks a visual profile")
    semantic_sampling = metadata.get("semantic_sampling")
    if not isinstance(semantic_sampling, Mapping):
        raise ValueError("object candidate lacks semantic sampling")
    five_dimensions = semantic_sampling.get("five_dimensions")
    if not isinstance(five_dimensions, Mapping):
        raise ValueError("object candidate lacks semantic dimensions")
    foreground = five_dimensions.get("foreground_object")
    if not isinstance(foreground, dict) or any(
        key not in foreground for key in _ANNOTATION_KEYS
    ):
        raise ValueError("object candidate lacks foreground annotations")
    appearance = metadata.get("appearance")
    if not isinstance(appearance, Mapping):
        raise ValueError("object candidate lacks appearance metadata")
    materials = appearance.get("materials")
    if not isinstance(materials, Mapping):
        raise ValueError("object candidate lacks appearance materials")
    material = materials.get("dynamic_object")
    if not isinstance(material, dict):
        raise ValueError("object candidate lacks a dynamic appearance material")
    return obj, foreground, material


def compile_object_pair_scene(
    host_template: Mapping[str, Any],
    object_templates: Sequence[Mapping[str, Any]],
    roles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Inject two independently reviewed objects into one frozen host scene."""

    if len(object_templates) != PAIR_SIZE or len(roles) != PAIR_SIZE:
        raise ValueError(
            "object-pair compilation requires exactly two candidates and roles"
        )
    resolved_roles = [_role_record(role) for role in roles]
    object_ids = [record["object_id"] for record in resolved_roles]
    if len(set(object_ids)) != PAIR_SIZE:
        raise ValueError("object-pair role ids must be unique")

    scene = copy.deepcopy(host_template)
    if scene.get("schema_version") != "physweep_pybullet_rigid_metadata_v1":
        raise ValueError("object-pair host must use rigid metadata v1")
    canonical_host = _is_canonical_base(scene)
    simulation = scene.get("simulation")
    if not isinstance(simulation, dict):
        raise ValueError("object-pair host lacks a simulation record")
    semantic_sampling = scene.get("semantic_sampling")
    if not isinstance(semantic_sampling, dict):
        raise ValueError("object-pair host lacks semantic sampling")
    five_dimensions = semantic_sampling.get("five_dimensions")
    if not isinstance(five_dimensions, dict):
        raise ValueError("object-pair host lacks semantic dimensions")
    five_dimensions.pop("foreground_object", None)

    appearance = scene.get("appearance")
    if not isinstance(appearance, dict):
        raise ValueError("object-pair host lacks appearance metadata")
    materials = appearance.get("materials")
    if not isinstance(materials, dict):
        raise ValueError("object-pair host lacks appearance materials")
    materials.pop("dynamic_object", None)

    objects = []
    foreground_objects = []
    dynamic_materials = {}
    for template, role in zip(object_templates, resolved_roles):
        source, foreground, source_material = _source_parts(template)
        obj = copy.deepcopy(source)
        obj["object_id"] = role["object_id"]
        obj["semantic_type"] = role["semantic_label"]
        obj["visual_profile"]["material_policy"] = "bound_role_override"
        objects.append(obj)
        foreground_objects.append(
            {
                "object_id": role["object_id"],
                **{
                    key: copy.deepcopy(foreground[key])
                    for key in _ANNOTATION_KEYS
                },
            }
        )
        appearance = copy.deepcopy(source_material)
        appearance["semantic_color_srgb"] = list(role["semantic_color_srgb"])
        appearance["semantic_color_mix"] = 0.60
        dynamic_materials[role["object_id"]] = appearance

    simulation["objects"] = objects
    five_dimensions["foreground_objects"] = foreground_objects
    materials["dynamic_objects"] = dynamic_materials
    if canonical_host and str(scene["scene_id"]).endswith("__base"):
        scene["scene_id"] = str(scene["scene_id"])[: -len("__base")]
    scene.pop("sweep", None)
    scene.pop("object_identity", None)
    return scene
