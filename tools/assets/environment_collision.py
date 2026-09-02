"""Compile and instantiate immutable visual-environment collision bindings."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file as sha256
from tools.core.hashing import sha256_json_binding as binding_sha256
from tools.dataset_contract.object_identity_contract import (
    require_simulation_objects,
    require_single_simulation_object,
)


BINDING_VERSION = "physweep_environment_binding_v3"
SUPPORTED_DYNAMIC_OBJECT_COUNTS = (1, 2)


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
        "occludes_camera": True,
    }


def _object_back_wall_clearance_m(
    simulation: dict[str, Any],
    obj: dict[str, Any],
    outward: list[float],
) -> float:
    duration = simulation.get("time", {}).get("duration_s")
    if duration is None:
        return 0.0
    velocity = obj.get("initial_state", {}).get("linear_velocity_m_s")
    size = obj.get("geometry", {}).get("size_m")
    if not isinstance(velocity, list) or not isinstance(size, list):
        return 0.0
    toward_wall_speed = max(
        0.0,
        -float(velocity[0]) * float(outward[0])
        - float(velocity[1]) * float(outward[1]),
    )
    if toward_wall_speed <= 1.0e-8:
        return 0.0
    planar_radius = math.hypot(float(size[0]), float(size[1])) / 2.0
    expected = obj.get("expected_motion", {})
    downhill_allowance = float(
        expected.get("minimum_downhill_displacement_m", 0.0)
    )
    slope_angle = math.radians(
        float(
            simulation.get("support", {})
            .get("surface_frame", {})
            .get("slope_angle_degrees", 0.0)
        )
    )
    vertical_drop = max(0.0, downhill_allowance * math.tan(slope_angle))
    post_slope_speed = math.sqrt(
        toward_wall_speed * toward_wall_speed + 2.0 * 9.81 * vertical_drop
    )
    return post_slope_speed * float(duration) + planar_radius + 0.25


def dynamic_back_wall_clearance_m(
    metadata: dict[str, Any], outward: list[float]
) -> float:
    """Conservative no-contact distance for motion directed toward a room wall."""

    return _object_back_wall_clearance_m(
        metadata["simulation"],
        require_single_simulation_object(metadata, __name__),
        outward,
    )


def _object_motion_lane(
    simulation: dict[str, Any], obj: dict[str, Any]
) -> dict[str, Any] | None:
    duration = simulation.get("time", {}).get("duration_s")
    if duration is None:
        return None
    state = obj.get("initial_state", {})
    velocity = state.get("linear_velocity_m_s")
    position = state.get("position_m")
    size = obj.get("geometry", {}).get("size_m")
    if not all(isinstance(value, list) for value in (velocity, position, size)):
        return None
    speed = math.hypot(float(velocity[0]), float(velocity[1]))
    if speed <= 1.0e-8:
        return None
    expected = obj.get("expected_motion", {})
    downhill = float(expected.get("minimum_downhill_displacement_m", 0.0))
    slope_angle = math.radians(
        float(
            simulation.get("support", {})
            .get("surface_frame", {})
            .get("slope_angle_degrees", 0.0)
        )
    )
    vertical_drop = max(0.0, downhill * math.tan(slope_angle))
    post_slope_speed = math.sqrt(speed * speed + 2.0 * 9.81 * vertical_drop)
    return {
        "start_xy": [float(position[0]), float(position[1])],
        "direction_xy": [
            float(velocity[0]) / speed,
            float(velocity[1]) / speed,
        ],
        "length_m": post_slope_speed * float(duration),
        "radius_m": math.hypot(float(size[0]), float(size[1])) / 2.0 + 0.12,
    }


def dynamic_motion_lane(
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Return a conservative planar capsule for set-piece clearance."""

    return _object_motion_lane(
        metadata["simulation"],
        require_single_simulation_object(metadata, __name__),
    )


def _two_object_back_wall_clearance_m(
    metadata: dict[str, Any], outward: list[float]
) -> float:
    simulation = metadata["simulation"]
    return max(
        (
            _object_back_wall_clearance_m(simulation, obj, outward)
            for obj in require_simulation_objects(metadata, (2,), __name__)
        ),
        default=0.0,
    )


def _two_object_motion_lanes(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    simulation = metadata["simulation"]
    return [
        lane
        for obj in require_simulation_objects(metadata, (2,), __name__)
        if (lane := _object_motion_lane(simulation, obj)) is not None
    ]


def clear_set_piece_from_motion_lane(
    center_xy: list[float],
    size_xy: list[float],
    lane: dict[str, Any] | None,
    preferred_side: float,
) -> tuple[list[float], float]:
    if lane is None:
        return center_xy, 0.0
    start = [float(value) for value in lane["start_xy"]]
    direction = [float(value) for value in lane["direction_xy"]]
    relative = [center_xy[0] - start[0], center_xy[1] - start[1]]
    projection = max(
        0.0,
        min(
            float(lane["length_m"]),
            relative[0] * direction[0] + relative[1] * direction[1],
        ),
    )
    nearest = [
        start[0] + direction[0] * projection,
        start[1] + direction[1] * projection,
    ]
    offset = [center_xy[0] - nearest[0], center_xy[1] - nearest[1]]
    distance = math.hypot(offset[0], offset[1])
    required = float(lane["radius_m"]) + math.hypot(*size_xy) / 2.0
    if distance >= required:
        return center_xy, 0.0
    if distance > 1.0e-8:
        normal = [offset[0] / distance, offset[1] / distance]
    else:
        side = -1.0 if preferred_side < 0.0 else 1.0
        normal = [-direction[1] * side, direction[0] * side]
    shift = required - distance + 0.05
    return [
        center_xy[0] + normal[0] * shift,
        center_xy[1] + normal[1] * shift,
    ], shift


def procedural_room_objects(
    scene_visual: dict[str, Any],
    *,
    scene_anchor: list[float],
    outward: list[float],
    lateral: list[float],
    wall_distance: float,
    wall_yaw_degrees: float,
    motion_lanes: list[dict[str, Any]] | None = None,
    dynamic_object_count: int = 0,
    collision_enabled: bool = True,
) -> list[dict[str, Any]]:
    """Compile one procedural room layout for physical or render-only use."""

    wall_center = [
        scene_anchor[0] - outward[0] * wall_distance,
        scene_anchor[1] - outward[1] * wall_distance,
    ]
    result: list[dict[str, Any]] = []

    def append(record: dict[str, Any]) -> None:
        record["collision_enabled"] = bool(collision_enabled)
        result.append(record)

    if bool(scene_visual.get("wall_enabled", True)):
        append(
            _box(
                collider_id="environment_back_wall",
                role="room_wall",
                material_role="back_wall",
                size_m=[6.5, 0.06, 2.8],
                position_m=[wall_center[0], wall_center[1], 1.4],
                yaw_degrees=wall_yaw_degrees,
            )
        )
        append(
            _box(
                collider_id="environment_wall_baseboard",
                role="room_detail",
                material_role="support_structure",
                size_m=[6.5, 0.08, 0.10],
                position_m=[
                    wall_center[0] + outward[0] * 0.04,
                    wall_center[1] + outward[1] * 0.04,
                    0.05,
                ],
                yaw_degrees=wall_yaw_degrees,
            )
        )

    side_wall = scene_visual.get("side_wall")
    if side_wall:
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
        append(
            _box(
                collider_id="environment_side_wall",
                role="room_wall",
                material_role="back_wall",
                size_m=[side_depth, 0.06, 2.8],
                position_m=[side_center[0], side_center[1], 1.4],
                yaw_degrees=wall_yaw_degrees + 90.0,
            )
        )

    for decor in scene_visual.get("decor", []):
        lateral_offset, depth_offset, z = [
            float(value) for value in decor["offset_lateral_depth_z"]
        ]
        append(
            _box(
                collider_id=f"environment_decor_{decor['id']}",
                role="room_detail",
                material_role=str(decor["material_role"]),
                size_m=[float(value) for value in decor["size_m"]],
                position_m=[
                    wall_center[0]
                    + lateral[0] * lateral_offset
                    + outward[0] * depth_offset,
                    wall_center[1]
                    + lateral[1] * lateral_offset
                    + outward[1] * depth_offset,
                    z,
                ],
                yaw_degrees=wall_yaw_degrees,
            )
        )

    for piece in scene_visual.get("set_pieces", []):
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
        original_center = list(center)
        lane_shift = 0.0
        if motion_lanes is not None:
            for motion_lane in motion_lanes:
                center, shift = clear_set_piece_from_motion_lane(
                    center,
                    [float(piece["size_m"][0]), float(piece["size_m"][1])],
                    motion_lane,
                    lateral_offset,
                )
                if dynamic_object_count == 1:
                    lane_shift = shift
            if dynamic_object_count == 2:
                lane_shift = math.hypot(
                    center[0] - original_center[0], center[1] - original_center[1]
                )
        record = _box(
            collider_id=f"environment_piece_{piece['id']}",
            role="room_detail",
            material_role=str(piece["material_role"]),
            size_m=[float(value) for value in piece["size_m"]],
            position_m=[center[0], center[1], z],
            yaw_degrees=wall_yaw_degrees,
        )
        if motion_lanes is not None:
            record["dynamic_lane_shift_m"] = round(lane_shift, 12)
        append(record)
    return result


def compile_environment_binding(
    metadata: dict[str, Any],
    camera_axis: list[dict[str, Any]],
    *,
    azimuth_override_degrees: float | None = None,
) -> dict[str, Any]:
    """Freeze an environment around one or two dynamic objects."""

    scene_visual = metadata["appearance"]["scene_visual"]
    camera_profile = str(metadata["camera_request"]["profile"])
    azimuth_degrees = (
        camera_azimuth_degrees(camera_axis, camera_profile)
        if azimuth_override_degrees is None
        else float(azimuth_override_degrees)
    )
    if not math.isfinite(azimuth_degrees):
        raise ValueError("environment camera azimuth must be finite")
    azimuth = math.radians(azimuth_degrees)
    outward = [math.cos(azimuth), math.sin(azimuth)]
    lateral = [-outward[1], outward[0]]
    objects = require_simulation_objects(
        metadata, SUPPORTED_DYNAMIC_OBJECT_COUNTS, __name__
    )
    initial_positions = [obj["initial_state"]["position_m"] for obj in objects]
    initial_position = [
        sum(float(position[axis]) for position in initial_positions) / len(objects)
        for axis in range(3)
    ]
    action_anchor_rule = (
        "initial_object_xy"
        if len(objects) == 1
        else "initial_dynamic_object_centroid_xy"
    )
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
    wall_distance = 0.0 if integrated_ground else float(
        scene_visual["back_wall_distance_m"]
    )
    if not integrated_ground and scene_class.startswith("ground_"):
        wall_distance += 0.25
    dynamic_clearance = (
        0.0
        if integrated_ground
        else (
            dynamic_back_wall_clearance_m(metadata, outward)
            if len(objects) == 1
            else _two_object_back_wall_clearance_m(metadata, outward)
        )
    )
    wall_distance = max(wall_distance, dynamic_clearance)
    wall_yaw = azimuth_degrees - 90.0
    motion_lanes = (
        [dynamic_motion_lane(metadata)]
        if len(objects) == 1
        else _two_object_motion_lanes(metadata)
    )
    motion_lanes = [lane for lane in motion_lanes if lane is not None]
    visual_objects = (
        []
        if integrated_ground
        else procedural_room_objects(
            scene_visual,
            scene_anchor=scene_anchor,
            outward=outward,
            lateral=lateral,
            wall_distance=wall_distance,
            wall_yaw_degrees=wall_yaw,
            motion_lanes=motion_lanes,
            dynamic_object_count=len(objects),
        )
    )
    colliders = copy.deepcopy(visual_objects)

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
            "dynamic_back_wall_clearance_m": round(dynamic_clearance, 12),
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
