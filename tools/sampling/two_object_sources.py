"""Resolve released 1obj metadata into strict 2obj object and host sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tools.sampling.released_asset_sources import (
    asset_object_template,
)
from tools.sampling.released_object_sources import verified_generation_records
from tools.scene_rules.two_object import (
    resolve_scene_rule,
    resolved_two_object_scene_rules,
)


def released_source_pool(
    *,
    root: Path,
    released_base_manifest_path: Path,
    source_root: Path,
    source_manifest_path: Path,
    matrix: dict[str, Any],
    scene_rules: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load generic hosts and exact generic/asset objects from the 1obj release."""

    resolved_scene_rules = resolved_two_object_scene_rules(scene_rules)
    source_contract = matrix["candidate_pool"]["source_release"]
    eligibility = matrix["candidate_pool"]["object_eligibility"]
    host_eligibility = resolved_scene_rules["host_eligibility"]
    family_schemas = {
        str(record["generation_metadata_schema_version"]): str(record["id"])
        for record in matrix["candidate_pool"]["object_source_families"]
    }
    sources = verified_generation_records(
        root=root, released_base_manifest_path=released_base_manifest_path,
        source_root=source_root, source_manifest_path=source_manifest_path,
        source_contract=source_contract, family_schemas=family_schemas,
    )

    objects: list[dict[str, Any]] = []
    hosts: list[dict[str, Any]] = []
    registry_cache: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    visual_hash_cache: dict[str, str] = {}
    allowed_visual_types = set(host_eligibility["allowed_visual_types"])
    allowed_scale_bins = set(eligibility["scale_bins"])
    geometry_types = {
        str(record["geometry_type"])
        for record in matrix["shape_families"]["families"]
    }
    environment_categories = {
        str(record["id"]): set(record["allowed_scene_rules"])
        for record in resolved_scene_rules["visual_environment_coverage"][
            "categories"
        ]
    }

    for verified in sources:
        metadata = verified["metadata"]
        release_metadata = verified["release_metadata"]
        source = verified["source"]
        source_family = source["source_family"]

        if source_family == "generic":
            simulation_objects = metadata.get("simulation", {}).get("objects")
            if not isinstance(simulation_objects, list) or len(simulation_objects) != 1:
                raise ValueError("eligible generic metadata must contain one object")
            obj = simulation_objects[0]
            support = metadata["simulation"]["support"]
            scene_class = str(support.get("scene_class", ""))
            scene_rule = resolve_scene_rule(resolved_scene_rules, support)
            if scene_rule is not None:
                roles = {
                    str(collider.get("role", ""))
                    for collider in support.get("colliders", [])
                    if isinstance(collider, dict)
                }
                required_roles = set(host_eligibility["required_collider_roles"])
                allowed_roles = set(host_eligibility["allowed_collider_roles"])
                motion_neutral = required_roles.issubset(roles) and roles.issubset(
                    allowed_roles
                )
                camera_unbounded = support.get("camera_envelope") is None
                if motion_neutral and camera_unbounded:
                    scene_visual = metadata["appearance"]["scene_visual"]
                    visual_id = str(scene_visual.get("id", ""))
                    visual_type = str(scene_visual.get("visual_type", ""))
                    environment_category = str(
                        scene_visual.get("environment_category", "")
                    )
                    if not visual_id or not visual_type or not environment_category:
                        raise ValueError(
                            "eligible two-object host lacks a visual profile"
                        )
                    if (
                        environment_category not in environment_categories
                        or str(scene_rule["id"])
                        not in environment_categories[environment_category]
                    ):
                        raise ValueError(
                            "eligible two-object host contradicts the declared "
                            "visual-environment coverage"
                        )
                    if visual_type in allowed_visual_types:
                        hosts.append(
                            {
                                "metadata": metadata,
                                "source": source,
                                "scene_rule_id": str(scene_rule["id"]),
                                "scene_class": scene_class,
                                "visual_profile_id": visual_id,
                                "visual_type": visual_type,
                                "environment_category": environment_category,
                            }
                        )
            template = metadata
        else:
            template, _ = asset_object_template(
                source_root=source_root,
                runtime_root=root,
                generation_metadata=metadata,
                release_metadata=release_metadata,
                eligibility=eligibility,
                registry_cache=registry_cache,
                visual_hash_cache=visual_hash_cache,
            )
            if template is None:
                continue
            obj = template["simulation"]["objects"][0]

        geometry = obj.get("geometry", {})
        shape = str(geometry.get("type", ""))
        collision = obj.get("collision_profile", {})
        pose_profile = str(obj.get("initial_state", {}).get("pose_profile", ""))
        if (
            obj.get("body_model") != eligibility["body_model"]
            or shape not in geometry_types
            or pose_profile != eligibility["required_pose_profile"]
        ):
            continue
        size = np.asarray(geometry.get("size_m"), dtype=np.float64)
        if (
            size.shape != (3,)
            or not np.isfinite(size).all()
            or bool(np.any(size <= 0.0))
            or (
                shape == "sphere"
                and not np.allclose(size, size[0], atol=1.0e-8, rtol=0.0)
            )
            or (
                shape == "cylinder"
                and not np.isclose(size[0], size[1], atol=1.0e-8, rtol=0.0)
            )
            or not (
                collision.get("type") == shape
                or (
                    source_family == "asset"
                    and shape == "cylinder"
                    and collision.get("type") == "compound"
                    and isinstance(collision.get("colliders"), list)
                    and len(collision["colliders"]) >= 2
                )
            )
        ):
            raise ValueError("eligible object lacks a matching collision proxy")
        foreground = template["semantic_sampling"]["five_dimensions"][
            "foreground_object"
        ]
        scale_bin = str(foreground.get("scale_bin", ""))
        visual_profile_id = str(obj.get("visual_profile", {}).get("id", ""))
        if scale_bin not in allowed_scale_bins or not visual_profile_id:
            raise ValueError("eligible object lacks scale or a visual profile")
        objects.append(
            {
                "metadata": template,
                "source": source,
                "source_family": source_family,
                "shape_family_id": shape,
                "scale_bin": scale_bin,
                "visual_profile_id": visual_profile_id,
            }
        )

    if not objects or not hosts:
        raise ValueError("released 1obj metadata yields no eligible 2obj sources")
    declared_scene_rules = {
        str(rule["id"]) for rule in resolved_scene_rules["physical_rules"]
    }
    eligible_scene_rules = {str(record["scene_rule_id"]) for record in hosts}
    if eligible_scene_rules != declared_scene_rules:
        missing = sorted(declared_scene_rules - eligible_scene_rules)
        raise ValueError(
            "released 1obj hosts do not cover declared physical scene rules: "
            f"{missing}"
        )
    declared_scene_environments = {
        (scene_rule_id, category)
        for category, scene_rule_ids in environment_categories.items()
        for scene_rule_id in scene_rule_ids
    }
    eligible_scene_environments = {
        (str(record["scene_rule_id"]), str(record["environment_category"]))
        for record in hosts
    }
    if eligible_scene_environments != declared_scene_environments:
        missing = sorted(declared_scene_environments - eligible_scene_environments)
        raise ValueError(
            "released 1obj hosts do not cover declared visual environments: "
            f"{missing}"
        )
    return objects, hosts
