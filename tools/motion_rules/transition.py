"""Wall-impact and support-exit transition rules."""

from __future__ import annotations

import math

import numpy as np

from rigid_geometry import clamp, pose_on_support

from .common import entry_speed_after_coulomb_travel
from .contracts import MotionAuditContext, MotionDerivationContext, MotionPlan


MOTIONS = frozenset({"wall_impact_1obj", "edge_fall_1obj"})


def derive(context: MotionDerivationContext, plan: MotionPlan) -> MotionPlan:
    lateral = [-context.direction[1], context.direction[0]]
    if context.motion == "wall_impact_1obj":
        wall = next(
            item
            for item in context.support["colliders"]
            if item["id"] == "impact_wall"
        )
        approach = min(
            float(context.subtype["approach_distance_m"]),
            context.limit * 0.82,
        )
        object_radius = max(
            float(context.size_m[0]), float(context.size_m[1])
        ) / 2.0
        wall_half_thickness = float(wall["size_m"][0]) / 2.0
        lateral_shift = clamp(
            context.zone_x * lateral[0] + context.zone_y * lateral[1],
            -0.18,
            0.18,
        )
        start_x = (
            float(wall["position_m"][0])
            - context.direction[0]
            * (wall_half_thickness + object_radius + approach)
            + lateral[0] * lateral_shift
        )
        start_y = (
            float(wall["position_m"][1])
            - context.direction[1]
            * (wall_half_thickness + object_radius + approach)
            + lateral[1] * lateral_shift
        )
        plan.pose = pose_on_support(
            context.support,
            context.shape,
            context.size_m,
            context.center_x
            + clamp(start_x - context.center_x, -context.half_x, context.half_x),
            context.center_y
            + clamp(start_y - context.center_y, -context.half_y, context.half_y),
            context.yaw,
            context.clearance,
            context.pose_profile,
            context.direction,
        )
        plan.effective_contact_friction = min(
            context.sampled_friction, 0.24
        )
        target_impact_speed = float(context.subtype["impact_speed_m_s"])
        speed = entry_speed_after_coulomb_travel(
            target_impact_speed,
            plan.effective_contact_friction,
            9.81,
            approach,
        )
        plan.linear_velocity_m_s = [
            context.direction[0] * speed,
            context.direction[1] * speed,
            0.0,
        ]
        plan.expected_motion.update(
            {
                "contact_mode": "support_motion_then_wall_impact",
                "required_collider_contact_id": "impact_wall",
                "impact_normal_xy": [
                    round(context.direction[0], 6),
                    round(context.direction[1], 6),
                ],
                "minimum_normal_speed_change_m_s": round(
                    max(0.28, target_impact_speed * 0.45), 6
                ),
                "minimum_post_impact_rebound_speed_m_s": round(
                    target_impact_speed * context.restitution * 0.25
                    if context.restitution >= 0.40
                    else 0.0,
                    6,
                ),
                "minimum_displacement_m": round(approach * 0.72, 6),
                "minimum_support_contact_fraction": 0.30,
            }
        )
        return plan

    if context.motion == "edge_fall_1obj":
        transition = context.support["transition_contract"]
        approach = min(
            float(context.subtype["approach_distance_m"]),
            context.limit * 0.72,
        )
        lateral_shift = clamp(
            context.zone_x * lateral[0] + context.zone_y * lateral[1],
            -0.16,
            0.16,
        )
        start_x = (
            context.center_x
            + context.direction[0] * (context.limit - approach)
            + lateral[0] * lateral_shift
        )
        start_y = (
            context.center_y
            + context.direction[1] * (context.limit - approach)
            + lateral[1] * lateral_shift
        )
        plan.pose = pose_on_support(
            context.support,
            context.shape,
            context.size_m,
            context.center_x
            + clamp(start_x - context.center_x, -context.half_x, context.half_x),
            context.center_y
            + clamp(start_y - context.center_y, -context.half_y, context.half_y),
            context.yaw,
            context.clearance,
            context.pose_profile,
            context.direction,
        )
        bounds = context.support["safe_surface_bounds"]
        start_xy = [float(value) for value in plan.pose["contact_point_m"][:2]]
        planar_radius = math.hypot(
            float(context.size_m[0]), float(context.size_m[1])
        ) / 2.0
        exit_distances = []
        for axis, key in enumerate(("x", "y")):
            component = float(context.direction[axis])
            if abs(component) < 1.0e-8:
                continue
            boundary = (
                float(bounds[key][1]) + planar_radius
                if component > 0.0
                else float(bounds[key][0]) - planar_radius
            )
            distance = (boundary - start_xy[axis]) / component
            if distance > 0.0:
                exit_distances.append(distance)
        if not exit_distances:
            raise ValueError(
                "edge-fall direction does not intersect a support edge"
            )
        clearance_distance = min(exit_distances) + 0.04
        plan.effective_contact_friction = min(
            context.sampled_friction, 0.18
        )
        target_exit_speed = float(context.subtype["exit_speed_m_s"])
        speed = entry_speed_after_coulomb_travel(
            target_exit_speed,
            plan.effective_contact_friction,
            9.81,
            clearance_distance,
        )
        plan.linear_velocity_m_s = [
            context.direction[0] * speed,
            context.direction[1] * speed,
            0.0,
        ]
        plan.expected_motion.update(
            {
                "contact_mode": "support_motion_then_edge_fall",
                "minimum_initial_support_contact_frames": 3,
                "must_exit_primary_support": True,
                "required_collider_contact_id": str(
                    transition["destination_collider_id"]
                ),
                "transition_contract_version": str(transition["version"]),
                "minimum_vertical_drop_m": round(
                    max(
                        0.18,
                        float(context.support["surface_center_z_m"]) * 0.45,
                    ),
                    6,
                ),
                "minimum_displacement_m": round(approach * 0.72, 6),
                "calculated_clearance_distance_m": round(
                    clearance_distance, 6
                ),
            }
        )
        return plan

    raise ValueError(f"unsupported transition motion: {context.motion}")


def audit(context: MotionAuditContext) -> None:
    if context.motion == "wall_impact_1obj":
        minimum_fraction = float(
            context.expected.get("minimum_support_contact_fraction", 0.0)
        )
        context.check(
            "sustained_support_contact",
            context.support_fraction >= minimum_fraction,
            round(context.support_fraction, 6),
            minimum_fraction,
        )
        if context.required_contact_index is None:
            return
        normal = np.asarray(
            context.expected["impact_normal_xy"], dtype=np.float64
        )
        projected_speed = context.velocities[:, :2] @ normal
        before = float(
            np.max(projected_speed[: context.required_contact_index + 1])
        )
        after = float(
            np.min(projected_speed[context.required_contact_index :])
        )
        speed_change = before - after
        threshold = float(
            context.expected["minimum_normal_speed_change_m_s"]
        )
        context.check(
            "wall_normal_speed_change",
            speed_change >= threshold,
            round(speed_change, 6),
            threshold,
        )
        rebound_threshold = float(
            context.expected.get("minimum_post_impact_rebound_speed_m_s", 0.0)
        )
        if rebound_threshold > 0.0:
            context.check(
                "wall_post_impact_rebound_speed",
                -after + 1.0e-9 >= rebound_threshold,
                round(-after, 6),
                rebound_threshold,
            )
        return

    if context.motion == "edge_fall_1obj":
        primary = context.primary_contacts > 0
        contact_indices = np.flatnonzero(primary)
        first_support = (
            int(contact_indices[0]) if contact_indices.size else None
        )
        exit_index = None
        if first_support is not None:
            for index in range(first_support + 1, len(primary) - 2):
                if not primary[index : index + 3].any():
                    exit_index = index
                    break
        minimum_contact_frames = int(
            context.expected["minimum_initial_support_contact_frames"]
        )
        context.check(
            "initial_support_contact",
            int(primary.sum()) >= minimum_contact_frames,
            int(primary.sum()),
            minimum_contact_frames,
        )
        context.check(
            "primary_support_exit",
            exit_index is not None,
            exit_index,
            "three_consecutive_frames_without_primary_contact",
        )
        vertical_drop = float(
            context.positions[0, 2] - context.positions[:, 2].min()
        )
        threshold = float(context.expected["minimum_vertical_drop_m"])
        context.check(
            "edge_vertical_drop",
            vertical_drop >= threshold,
            round(vertical_drop, 6),
            threshold,
        )
        return

    raise ValueError(f"unsupported transition motion: {context.motion}")
