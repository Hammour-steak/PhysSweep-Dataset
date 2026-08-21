"""Compile and instantiate immutable visual-environment collision bindings."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any


BINDING_VERSION = "physweep_environment_binding_v3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def binding_sha256(binding: dict[str, Any]) -> str:
    value = copy.deepcopy(binding)
    value.pop("binding_sha256", None)
    return record_sha256(value)


def camera_azimuth_degrees(
    camera_axis: list[dict[str, Any]], profile: str
) -> float:
    for record in camera_axis:
        if str(record["label"]) == profile:
            return float(record["overrides"]["view_rule"]["azimuth_degrees"])
    raise ValueError(f"unknown camera profile for environment binding: {profile}")


def _box(
    *,
    collider_id: str,
    role: str,
    material_role: str,
    size_m: list[float],
    position_m: list[float],
    yaw_degrees: float,
) -> dict[str, Any]:
    return {
        "id": collider_id,
        "primitive": "box",
        "role": role,
        "material_role": material_role,
        "size_m": [float(value) for value in size_m],
        "position_m": [float(value) for value in position_m],
        "rotation_euler_degrees": [0.0, 0.0, float(yaw_degrees)],
        "visible": True,
        "collision_enabled": True,
        "occludes_camera": role == "room_wall",
    }


def compile_environment_binding(
    metadata: dict[str, Any], camera_axis: list[dict[str, Any]]
) -> dict[str, Any]:
    """Freeze one environment's visual and collision geometry before simulation."""

    scene_visual = metadata["appearance"]["scene_visual"]
    camera_profile = str(metadata["camera_request"]["profile"])
    azimuth_degrees = camera_azimuth_degrees(camera_axis, camera_profile)
    azimuth = math.radians(azimuth_degrees)
    outward = [math.cos(azimuth), math.sin(azimuth)]
    lateral = [-outward[1], outward[0]]
    initial_position = metadata["simulation"]["objects"][0]["initial_state"][
        "position_m"
    ]
    scene_class = str(metadata["simulation"]["support"]["scene_class"])
    composition = scene_visual.get("composition")
    integrated_ground = (
        isinstance(composition, dict)
        and str(composition.get("review_status")) == "approved"
        and str(composition.get("composition_mode")) == "integrated_ground"
    )
    if integrated_ground:
        if scene_class != "ground_flat":
            raise ValueError(
                f"integrated ground environment cannot host {scene_class}"
            )
        anchor_xy = [float(initial_position[0]), float(initial_position[1])]
        action_anchor_rule = "initial_object_xy"
        scene_anchor = [anchor_xy[0], anchor_xy[1], 0.0]
        if (
            bool(scene_visual.get("wall_enabled", False))
            or scene_visual.get("side_wall")
            or scene_visual.get("decor")
            or scene_visual.get("set_pieces")
        ):
            raise ValueError(
                "integrated environment cannot include procedural room geometry"
            )
    else:
        scene_anchor = [
            float(initial_position[0]),
            float(initial_position[1]),
            0.0,
        ]
        action_anchor_rule = "initial_object_xy"
    wall_distance = 0.0 if integrated_ground else float(
        scene_visual["back_wall_distance_m"]
    )
    if not integrated_ground and scene_class.startswith("ground_"):
        wall_distance += 0.25
    wall_center = [
        scene_anchor[0] - outward[0] * wall_distance,
        scene_anchor[1] - outward[1] * wall_distance,
    ]
    wall_yaw = azimuth_degrees - 90.0
    visual_objects: list[dict[str, Any]] = []
    colliders: list[dict[str, Any]] = []

    if not integrated_ground and bool(scene_visual.get("wall_enabled", True)):
        wall = _box(
            collider_id="environment_back_wall",
            role="room_wall",
            material_role="back_wall",
            size_m=[6.5, 0.06, 2.8],
            position_m=[wall_center[0], wall_center[1], 1.4],
            yaw_degrees=wall_yaw,
        )
        baseboard = _box(
            collider_id="environment_wall_baseboard",
            role="room_detail",
            material_role="support_structure",
            size_m=[6.5, 0.08, 0.10],
            position_m=[
                wall_center[0] + outward[0] * 0.04,
                wall_center[1] + outward[1] * 0.04,
                0.05,
            ],
            yaw_degrees=wall_yaw,
        )
        visual_objects.extend([wall, baseboard])
        colliders.extend(copy.deepcopy([wall, baseboard]))

    side_wall = scene_visual.get("side_wall")
    if side_wall and not integrated_ground:
        side = float(side_wall["side"])
        side_distance = float(side_wall["distance_m"])
        side_depth = float(side_wall["depth_m"])
        side_center = [
            wall_center[0]
            + lateral[0] * side * side_distance
            + outward[0] * side_depth / 2.0,
            wall_center[1]
            + lateral[1] * side * side_distance
            + outward[1] * side_depth / 2.0,
        ]
        record = _box(
            collider_id="environment_side_wall",
            role="room_wall",
            material_role="back_wall",
            size_m=[side_depth, 0.06, 2.8],
            position_m=[side_center[0], side_center[1], 1.4],
            yaw_degrees=wall_yaw + 90.0,
        )
        visual_objects.append(record)
        colliders.append(copy.deepcopy(record))

    for decor in ([] if integrated_ground else scene_visual.get("decor", [])):
        lateral_offset, depth_offset, z = [
            float(value) for value in decor["offset_lateral_depth_z"]
        ]
        center = [
            wall_center[0] + lateral[0] * lateral_offset + outward[0] * depth_offset,
            wall_center[1] + lateral[1] * lateral_offset + outward[1] * depth_offset,
        ]
        record = _box(
            collider_id=f"environment_decor_{decor['id']}",
            role="room_detail",
            material_role=str(decor["material_role"]),
            size_m=[float(value) for value in decor["size_m"]],
            position_m=[center[0], center[1], z],
            yaw_degrees=wall_yaw,
        )
        visual_objects.append(record)
        colliders.append(copy.deepcopy(record))

    for piece in ([] if integrated_ground else scene_visual.get("set_pieces", [])):
        lateral_offset, outward_offset, z = [
            float(value) for value in piece["offset_lateral_outward_z"]
        ]
        center = [
            scene_anchor[0]
            + lateral[0] * lateral_offset
            + outward[0] * outward_offset,
            scene_anchor[1]
            + lateral[1] * lateral_offset
            + outward[1] * outward_offset,
        ]
        record = _box(
            collider_id=f"environment_piece_{piece['id']}",
            role="room_detail",
            material_role=str(piece["material_role"]),
            size_m=[float(value) for value in piece["size_m"]],
            position_m=[center[0], center[1], z],
            yaw_degrees=wall_yaw,
        )
        visual_objects.append(record)
        colliders.append(copy.deepcopy(record))

    if str(scene_visual.get("visual_type", "procedural_room")) == "mesh_backdrop":
        if not integrated_ground:
            raise ValueError(
                f"mesh environment is not composition-ready: {scene_visual['id']}"
            )
        asset = scene_visual["asset"]
        proxy = asset.get("collision_proxy")
        if not isinstance(proxy, dict):
            raise ValueError(
                f"mesh environment lacks collision proxy: {scene_visual['id']}"
            )
        local_anchor = [
            float(value) for value in composition["action_surface"]["anchor_local_m"]
        ]
        local_camera_azimuth = float(
            composition["camera"]["preferred_local_azimuth_degrees"]
        )
        reviewed_asset_yaw_degrees = float(asset.get("review_yaw_degrees", 0.0))
        reviewed_frame_world_yaw_degrees = (
            azimuth_degrees - local_camera_azimuth
        )
        asset_world_yaw_degrees = (
            reviewed_frame_world_yaw_degrees + reviewed_asset_yaw_degrees
        )
        asset_world_yaw = math.radians(asset_world_yaw_degrees)
        rotated_anchor = [
            math.cos(asset_world_yaw) * local_anchor[0]
            - math.sin(asset_world_yaw) * local_anchor[1],
            math.sin(asset_world_yaw) * local_anchor[0]
            + math.cos(asset_world_yaw) * local_anchor[1],
        ]
        mesh_center = [
            scene_anchor[0] - rotated_anchor[0],
            scene_anchor[1] - rotated_anchor[1],
        ]
        visual_record = {
            "id": f"scene_mesh_{asset['asset_id']}",
            "primitive": "mesh",
            "role": "room_context_mesh",
            "collision_enabled": True,
            "visible": True,
            "asset_id": str(asset["asset_id"]),
            "path": str(asset["path"]),
            "sha256": str(asset["sha256"]),
            "license": str(asset["license"]),
            "source_bbox_size": [
                float(value) for value in asset["source_bbox_size"]
            ],
            "normalization_axis": str(asset["normalization_axis"]),
            "target_extent_m": float(asset["target_extent_m"]),
            "position_m": [
                float(mesh_center[0]),
                float(mesh_center[1]),
                -local_anchor[2],
            ],
            "rotation_euler_degrees": [0.0, 0.0, asset_world_yaw_degrees],
            "requires_image_texture": True,
            "exclude_object_names": [
                str(value) for value in asset.get("exclude_object_names", [])
            ],
            "exclude_object_name_prefixes": [
                str(value)
                for value in asset.get("exclude_object_name_prefixes", [])
            ],
            "source_space_face_exclusions": copy.deepcopy(
                asset.get("source_space_face_exclusions", [])
            ),
            "composition_mode": "integrated_ground",
            "action_surface_owner": "source_environment",
            "transform_contract": {
                "asset_local_frame": (
                    "normalized_visual_asset_local_bottom_center_z_up"
                ),
                "reviewed_asset_yaw_degrees": reviewed_asset_yaw_degrees,
                "reviewed_frame_world_yaw_degrees": (
                    reviewed_frame_world_yaw_degrees
                ),
                "asset_world_yaw_degrees": asset_world_yaw_degrees,
                "action_anchor_local_m": copy.deepcopy(local_anchor),
            },
        }
        collider_record = {
            "id": f"environment_mesh_{asset['asset_id']}",
            "primitive": "static_concave_mesh",
            "role": "environment_structure",
            "asset_id": str(asset["asset_id"]),
            "mesh_path": str(proxy["path"]),
            "mesh_sha256": str(proxy["sha256"]),
            "mesh_flags": [str(value) for value in proxy["flags"]],
            "position_m": copy.deepcopy(visual_record["position_m"]),
            "rotation_euler_degrees": copy.deepcopy(
                visual_record["rotation_euler_degrees"]
            ),
            "transform_contract": copy.deepcopy(
                visual_record["transform_contract"]
            ),
            "visible": False,
            "collision_enabled": True,
            "occludes_camera": False,
        }
        visual_objects.append(visual_record)
        colliders.append(collider_record)

    collider_ids = [str(record["id"]) for record in colliders]
    if len(collider_ids) != len(set(collider_ids)):
        raise ValueError(f"duplicate environment collider: {scene_visual['id']}")
    binding = {
        "schema_version": BINDING_VERSION,
        "profile_id": str(scene_visual["id"]),
        "visual_type": str(scene_visual.get("visual_type", "procedural_room")),
        "placement": {
            "rule_version": (
                "reviewed_action_surface_camera_axis_v3"
                if integrated_ground
                else "initial_action_anchor_camera_axis_v1"
            ),
            "camera_profile": camera_profile,
            "camera_azimuth_degrees": azimuth_degrees,
            "scene_anchor_m": [round(value, 12) for value in scene_anchor],
            "action_anchor_rule": action_anchor_rule,
            "outward_direction_xy": [round(value, 12) for value in outward],
            "lateral_direction_xy": [round(value, 12) for value in lateral],
            "back_wall_distance_m": round(wall_distance, 12),
            "composition_mode": (
                "integrated_ground" if integrated_ground else "procedural_room"
            ),
            "reviewed_asset_yaw_degrees": (
                reviewed_asset_yaw_degrees if integrated_ground else None
            ),
            "reviewed_frame_world_yaw_degrees": (
                reviewed_frame_world_yaw_degrees if integrated_ground else None
            ),
            "asset_world_yaw_degrees": (
                asset_world_yaw_degrees if integrated_ground else None
            ),
        },
        "dynamics": {
            "policy": "inherit_primary_support",
            **copy.deepcopy(metadata["simulation"]["support"]["dynamics"]),
        },
        "visual_objects": visual_objects,
        "colliders": colliders,
        "policy": {
            "always_loaded_during_simulation": True,
            "visual_and_collision_world_pose_is_identical": True,
            "unexpected_environment_contacts_are_physically_resolved": True,
            "initial_dynamic_environment_penetration_is_forbidden": True,
            "visual_action_surface_owner": (
                "source_environment" if integrated_ground else "procedural_scene_kit"
            ),
            "collision_action_surface_owner": "analytic_scene_kit",
        },
    }
    binding["binding_sha256"] = binding_sha256(binding)
    return binding


def validate_environment_binding(metadata: dict[str, Any]) -> dict[str, Any]:
    binding = metadata.get("environment_binding")
    if not isinstance(binding, dict):
        raise ValueError("metadata lacks an immutable environment binding")
    if binding.get("schema_version") != BINDING_VERSION:
        raise ValueError("unsupported environment binding schema")
    if binding_sha256(binding) != str(binding.get("binding_sha256")):
        raise ValueError("environment binding hash mismatch")
    scene_visual = metadata["appearance"]["scene_visual"]
    if str(binding["profile_id"]) != str(scene_visual["id"]):
        raise ValueError("environment binding profile mismatch")
    if not bool(binding["policy"]["always_loaded_during_simulation"]):
        raise ValueError("environment collision cannot be disabled")
    if any(not bool(record.get("collision_enabled", False)) for record in binding["colliders"]):
        raise ValueError("environment binding contains a disabled collider")
    return binding


def create_pybullet_environment_bodies(
    pb: Any,
    root: Path,
    binding: dict[str, Any],
) -> dict[str, int]:
    bodies: dict[str, int] = {}
    for collider in binding["colliders"]:
        collider_id = str(collider["id"])
        primitive = str(collider["primitive"])
        orientation = pb.getQuaternionFromEuler(
            [
                math.radians(float(value))
                for value in collider["rotation_euler_degrees"]
            ]
        )
        if primitive == "box":
            size = [float(value) for value in collider["size_m"]]
            shape = pb.createCollisionShape(
                pb.GEOM_BOX, halfExtents=[value / 2.0 for value in size]
            )
        elif primitive == "static_concave_mesh":
            path = root / str(collider["mesh_path"])
            if not path.is_file() or sha256(path) != str(collider["mesh_sha256"]):
                raise ValueError(f"environment collision mesh changed: {collider_id}")
            if collider.get("mesh_flags") != ["GEOM_FORCE_CONCAVE_TRIMESH"]:
                raise ValueError(f"unsupported environment mesh flags: {collider_id}")
            shape = pb.createCollisionShape(
                pb.GEOM_MESH,
                fileName=str(path.resolve()),
                flags=pb.GEOM_FORCE_CONCAVE_TRIMESH,
            )
        else:
            raise ValueError(f"unsupported environment collider: {primitive}")
        body = int(
            pb.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=shape,
                basePosition=[float(value) for value in collider["position_m"]],
                baseOrientation=orientation,
            )
        )
        bodies[collider_id] = body
    return bodies
