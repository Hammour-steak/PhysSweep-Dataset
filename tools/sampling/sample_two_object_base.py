#!/usr/bin/env python3
"""Build one deterministic two-sphere collision base from a reviewed 1obj template."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.json_io import read_json, write_json
from tools.core.rigid_geometry import finite_vector, positive_vector
from tools.dataset_contract.object_identity_contract import attach_object_identity


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "two_object_sampling.json"


def _recolored_material(
    material: dict[str, Any], color: list[float]
) -> dict[str, Any]:
    result = copy.deepcopy(material)
    result["semantic_color_srgb"] = [float(value) for value in color]
    result["semantic_color_mix"] = 0.60
    return result


def build_two_sphere_collision(
    template: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Reuse a frozen sphere/support scene while replacing only 2obj semantics."""

    if config.get("schema_version") != "physweep_two_object_sampling_v1":
        raise ValueError("unsupported two-object sampling configuration")
    if template.get("schema_version") != "physweep_pybullet_rigid_metadata_v1":
        raise ValueError("two-object template must use rigid metadata v1")
    source_objects = template.get("simulation", {}).get("objects", [])
    if len(source_objects) != 1:
        raise ValueError("two-object template must contain one dynamic object")
    source = source_objects[0]
    if source.get("body_model") != "rigid_body":
        raise ValueError("two-object template must contain a rigid body")
    geometry = source.get("geometry", {})
    if geometry.get("type") != "sphere":
        raise ValueError("two-object reference scene requires a sphere template")
    size_m = positive_vector(geometry["size_m"], 3, "sphere dimensions")
    if max(size_m) - min(size_m) > 1.0e-8:
        raise ValueError("sphere template dimensions must be isotropic")
    support = template["simulation"]["support"]
    slope = float(support["surface_frame"]["slope_angle_degrees"])
    if abs(slope) > 1.0e-8:
        raise ValueError("two-sphere reference collision requires a flat support")
    bounds = support["safe_surface_bounds"]
    radius = 0.5 * size_m[0]
    source_position = [
        float(value) for value in source["initial_state"]["position_m"]
    ]
    center_x, center_y = source_position[:2]
    placement = template.get("environment_binding", {}).get("placement", {})
    if placement.get("action_anchor_rule") == "initial_object_xy":
        scene_anchor = [float(value) for value in placement["scene_anchor_m"][:2]]
        if any(
            abs(anchor - center) > 1.0e-6
            for anchor, center in zip(scene_anchor, (center_x, center_y))
        ):
            raise ValueError(
                "template environment anchor differs from the source initial position"
            )
    surface_gap = finite_vector(
        [config["initial_surface_gap_m"]], 1, "initial sphere surface gap"
    )[0]
    if surface_gap <= 0.0:
        raise ValueError("initial sphere surface gap must be positive")
    center_distance = 2.0 * radius + surface_gap
    positions_x = [center_x - 0.5 * center_distance, center_x + 0.5 * center_distance]
    margin = radius + 0.02
    if (
        positions_x[0] < float(bounds["x"][0]) + margin
        or positions_x[1] > float(bounds["x"][1]) - margin
        or center_y < float(bounds["y"][0]) + margin
        or center_y > float(bounds["y"][1]) - margin
    ):
        raise ValueError("template support is too small for the reference collision")
    position_z = float(support["surface_center_z_m"]) + radius + 0.0005
    velocities_x = finite_vector(
        config["initial_velocity_x_m_s"], 2, "two-object velocities"
    )
    if velocities_x[0] <= velocities_x[1]:
        raise ValueError("two-object velocities must form a closing pair")

    scene = copy.deepcopy(template)
    scene["scene_id"] = f"{template['scene_id']}__two_sphere_collision"
    scene["dataset_stage"] = "two_object_base_candidate"
    scene["sample_index"] = 1
    motion = str(config["reference_motion"])
    source_semantics = scene.get("semantic_sampling", {}).get(
        "five_dimensions", {}
    )
    source_semantics["motion"] = {
        "family": motion,
        "subtype": "direct_pair_collision",
        "direction": "positive_x",
        "direction_angle_degrees": 0.0,
        "trajectory_extent": "medium",
        "initial_position_zone": "opposed_pair",
    }
    source_foreground = source_semantics.pop("foreground_object", None)
    if not isinstance(source_foreground, dict):
        raise ValueError("two-object template lacks foreground object annotations")
    annotation_keys = ("semantic_category", "scale_bin", "uniform_scale")
    if any(key not in source_foreground for key in annotation_keys):
        raise ValueError("two-object template has incomplete foreground annotations")
    shared_annotations = {
        key: copy.deepcopy(source_foreground[key]) for key in annotation_keys
    }
    source_semantics["foreground_objects"] = [
        {
            "object_id": "object_a",
            **copy.deepcopy(shared_annotations),
        },
        {
            "object_id": "object_b",
            **copy.deepcopy(shared_annotations),
        },
    ]
    contact_friction, contact_restitution, minimum_displacement = finite_vector(
        [
            config["contact_friction"],
            config["contact_restitution"],
            config["minimum_displacement_m"],
        ],
        3,
        "two-object physical rule values",
    )
    minimum_support_fraction = finite_vector(
        [config["minimum_support_contact_fraction"]],
        1,
        "minimum support contact fraction",
    )[0]
    if contact_friction < 0.0:
        raise ValueError("contact friction must be nonnegative")
    if not 0.0 <= contact_restitution <= 1.0:
        raise ValueError("contact restitution must lie in [0, 1]")
    if minimum_displacement <= 0.0:
        raise ValueError("minimum displacement must be positive")
    if not 0.0 < minimum_support_fraction <= 1.0:
        raise ValueError("minimum support contact fraction must lie in (0, 1]")
    material = copy.deepcopy(source["material"])
    material["contact_friction"] = contact_friction
    material["contact_restitution"] = contact_restitution
    expected_common = {
        "motion_family": motion,
        "contact_mode": "supported_pair_collision",
        "must_contact_primary_support": True,
        "minimum_displacement_m": minimum_displacement,
        "minimum_support_contact_fraction": minimum_support_fraction,
    }
    objects = []
    semantic_labels = ("red sphere", "blue sphere")
    for index, object_id in enumerate(("object_a", "object_b")):
        obj = copy.deepcopy(source)
        obj["object_id"] = object_id
        obj["semantic_type"] = semantic_labels[index]
        obj["visual_profile"]["material_policy"] = "bound_role_override"
        obj["material"] = copy.deepcopy(material)
        obj["initial_state"] = {
            "pose_profile": "support_normal",
            "position_m": [positions_x[index], center_y, position_z],
            "orientation_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "linear_velocity_m_s": [velocities_x[index], 0.0, 0.0],
            "angular_velocity_rad_s": [
                0.0,
                velocities_x[index] / radius,
                0.0,
            ],
        }
        obj["expected_motion"] = {
            **expected_common,
            "required_object_contact_id": (
                "object_b" if object_id == "object_a" else "object_a"
            ),
        }
        objects.append(obj)
    scene["simulation"]["objects"] = objects
    scene["simulation"]["interaction"] = {
        "type": "pairwise_collision",
        "object_ids": ["object_a", "object_b"],
        "approach_axis_xy": [1.0, 0.0],
        **copy.deepcopy(config["interaction_audit"]),
    }
    dynamic_material = scene["appearance"]["materials"].pop("dynamic_object")
    scene["appearance"]["materials"]["dynamic_objects"] = {
        "object_a": _recolored_material(dynamic_material, [0.82, 0.18, 0.12, 1.0]),
        "object_b": _recolored_material(dynamic_material, [0.10, 0.32, 0.86, 1.0]),
    }
    scene.pop("object_identity", None)
    attach_object_identity(scene)
    return scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    template_path = args.template.resolve()
    config_path = args.config.resolve()
    output_path = args.output.resolve()
    scene = build_two_sphere_collision(
        read_json(template_path), read_json(config_path)
    )
    write_json(output_path, scene)
    manifest = {
        "schema_version": "physweep_pybullet_base_manifest_v1",
        "dataset_id": "physweep_two_object",
        "sample_count": 1,
        "config_path": str(config_path.relative_to(root)),
        "config_sha256": sha256(config_path),
        "template_path": str(template_path.relative_to(root)),
        "template_sha256": sha256(template_path),
        "samples": [
            {
                "scene_id": scene["scene_id"],
                "metadata_path": str(output_path.relative_to(root)),
                "metadata_sha256": sha256(output_path),
            }
        ],
        "status": "sampled_pending_simulation",
    }
    manifest_path = output_path.parent / "manifest.json"
    write_json(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
