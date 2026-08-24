"""Inclined-surface and ramp-transition rules."""

from __future__ import annotations

import math

import numpy as np

from rigid_geometry import clamp, cross, pose_on_support, slope_tangent_velocity

from .common import (
    climb_speed_for_distance,
    distance_lower_bound,
    projected_displacement,
)
from .contracts import MotionAuditContext, MotionDerivationContext, MotionPlan


MOTIONS = frozenset(
    {
        "slope_slide_down_1obj",
        "slope_slide_up_1obj",
        "ramp_to_flat_1obj",
    }
)


def derive(context: MotionDerivationContext, plan: MotionPlan) -> MotionPlan:
    angle = math.radians(
        float(context.support["surface_frame"]["slope_angle_degrees"])
    )
    if context.motion == "ramp_to_flat_1obj":
        transition = context.support["transition_contract"]
        plan.effective_contact_friction = min(
            context.sampled_friction, max(0.03, 0.62 * math.tan(angle))
        )
        x = context.center_x + context.rng.uniform(
            -context.half_x * 0.10, context.half_x * 0.10
        )
        y = context.center_y + context.half_y * context.rng.uniform(0.58, 0.76)
        plan.pose = pose_on_support(
            context.support,
            context.shape,
            context.size_m,
            x,
            y,
            0.0,
            context.clearance,
            context.pose_profile,
            context.direction,
        )
        sampled_initial_speed = context.rng.uniform(0.82, 1.08) * float(
            context.subtype["speed"]
        )
        minimum_post_transition_travel = clamp(
            0.40 * float(context.support["landing_length_m"]),
            0.35,
            0.60,
        )
        required_exit_speed = math.sqrt(
            2.0
            * plan.effective_contact_friction
            * 9.81
            * minimum_post_transition_travel
        )
        initial_speed = sampled_initial_speed
        if context.shape != "sphere":
            initial_speed = max(
                initial_speed,
                1.38 * required_exit_speed / math.cos(angle),
            )
        plan.linear_velocity_m_s = slope_tangent_velocity(
            context.support, initial_speed, uphill=False
        )
        plan.expected_motion.update(
            {
                "contact_mode": "inclined_motion_then_flat_contact",
                "required_collider_contact_id": str(
                    transition["destination_collider_id"]
                ),
                "transition_contract_version": str(transition["version"]),
                "minimum_downhill_displacement_m": round(
                    context.half_y * 0.85, 6
                ),
                "minimum_post_transition_travel_m": round(
                    minimum_post_transition_travel, 6
                ),
                "must_contact_primary_support": True,
            }
        )
        return plan

    plan.effective_contact_friction = min(
        context.sampled_friction, max(0.035, 0.72 * math.tan(angle))
    )
    x_fraction = 0.22 if "offset" in str(context.subtype["label"]) else 0.08
    x = context.center_x + context.rng.uniform(
        -context.half_x * x_fraction, context.half_x * x_fraction
    )
    if context.motion == "slope_slide_down_1obj":
        y = context.center_y + context.half_y * context.rng.uniform(0.52, 0.76)
        plan.pose = pose_on_support(
            context.support,
            context.shape,
            context.size_m,
            x,
            y,
            0.0,
            context.clearance,
            context.pose_profile,
            context.direction,
        )
        initial_speed = context.rng.uniform(0.02, 0.14)
        plan.linear_velocity_m_s = slope_tangent_velocity(
            context.support, initial_speed, uphill=False
        )
        plan.expected_motion.update(
            {
                "contact_mode": "inclined_slide_down",
                "minimum_downhill_displacement_m": 0.28,
                "must_contact_primary_support": True,
            }
        )
    elif context.motion == "slope_slide_up_1obj":
        y = context.center_y - context.half_y * context.rng.uniform(0.50, 0.72)
        plan.pose = pose_on_support(
            context.support,
            context.shape,
            context.size_m,
            x,
            y,
            0.0,
            context.clearance,
            context.pose_profile,
            context.direction,
        )
        target_climb = clamp(
            context.half_y * context.trajectory_extent_fraction, 0.26, 0.58
        )
        uphill_deceleration = 9.81 * (
            math.sin(angle)
            + plan.effective_contact_friction * math.cos(angle)
        )
        speed = climb_speed_for_distance(uphill_deceleration, target_climb)
        plan.linear_velocity_m_s = slope_tangent_velocity(
            context.support, speed, uphill=True
        )
        plan.expected_motion.update(
            {
                "contact_mode": "inclined_slide_up_then_reverse",
                "minimum_uphill_displacement_m": round(target_climb * 0.55, 6),
                "must_reverse_downhill": True,
                "must_contact_primary_support": True,
            }
        )
    else:
        raise ValueError(f"unsupported incline motion: {context.motion}")

    if context.shape == "sphere" or context.pose_profile == "side_on_motion":
        normal = [
            float(value)
            for value in context.support["surface_frame"]["normal"]
        ]
        radius = float(context.size_m[0]) / 2.0
        plan.angular_velocity_rad_s = [
            value / radius for value in cross(normal, plan.linear_velocity_m_s)
        ]
        if context.shape == "sphere":
            # A spherical object on an incline is sampled as rolling motion,
            # so keep a scale-aware minimum in the immutable motion contract.
            angular_speed = float(np.linalg.norm(plan.angular_velocity_rad_s))
            plan.expected_motion["minimum_peak_angular_speed_rad_s"] = round(
                max(0.1, 0.5 * angular_speed), 6
            )
    return plan


def audit(context: MotionAuditContext) -> None:
    uphill = np.asarray(
        context.metadata["simulation"]["support"]["surface_frame"][
            "tangent_uphill"
        ],
        dtype=np.float64,
    )
    projected = projected_displacement(context.positions, uphill)
    if context.motion == "ramp_to_flat_1obj":
        downhill_extent = max(0.0, -float(projected.min()))
        threshold = float(
            context.expected["minimum_downhill_displacement_m"]
        )
        context.check(
            "ramp_transition_downhill_extent",
            downhill_extent >= threshold,
            round(downhill_extent, 6),
            threshold,
        )
        required_contacts = context.required_contacts
        if required_contacts is None:
            raise ValueError("ramp transition requires landing contact samples")
        landing_only = np.flatnonzero(
            (required_contacts > 0) & (context.primary_contacts == 0)
        )
        transition_index = int(landing_only[0]) if len(landing_only) else None
        environment_hits = {}
        for collider in context.metadata.get("environment_binding", {}).get(
            "colliders", []
        ):
            collider_id = str(collider["id"])
            contact_key = (
                f"{context.object_id}__collider_contact_count__{collider_id}"
            )
            if contact_key not in context.trajectory:
                raise ValueError(
                    f"trajectory is missing environment contact samples: {contact_key}"
                )
            contact_count = int(np.max(context.trajectory[contact_key]))
            if contact_count > 0:
                environment_hits[collider_id] = contact_count
        context.check(
            "no_unplanned_environment_contact",
            not environment_hits,
            environment_hits,
            {},
        )
        post_transition_travel = 0.0
        if transition_index is not None:
            downhill_xy = -uphill[:2]
            downhill_xy /= np.linalg.norm(downhill_xy)
            relative_xy = (
                context.positions[transition_index:, :2]
                - context.positions[transition_index, :2]
            )
            post_transition_travel = max(
                0.0, float(np.max(relative_xy @ downhill_xy))
            )
        post_transition_threshold = float(
            context.expected["minimum_post_transition_travel_m"]
        )
        accepted_post_transition_distance = distance_lower_bound(
            post_transition_threshold,
            context.absolute_distance_tolerance,
            context.relative_distance_tolerance,
        )
        context.check(
            "minimum_post_transition_travel",
            post_transition_travel >= accepted_post_transition_distance,
            round(post_transition_travel, 6),
            round(accepted_post_transition_distance, 6),
        )
        return

    if context.motion == "slope_slide_down_1obj":
        downhill_extent = max(0.0, -float(projected.min()))
        threshold = float(
            context.expected["minimum_downhill_displacement_m"]
        )
        context.check(
            "downhill_extent",
            downhill_extent >= threshold,
            round(downhill_extent, 6),
            threshold,
        )
    elif context.motion == "slope_slide_up_1obj":
        apex_index = int(np.argmax(projected))
        uphill_extent = float(projected[apex_index])
        return_extent = float(projected[apex_index] - projected[-1])
        threshold = float(context.expected["minimum_uphill_displacement_m"])
        context.check(
            "uphill_extent",
            uphill_extent >= threshold,
            round(uphill_extent, 6),
            threshold,
        )
        context.check(
            "downhill_reversal",
            apex_index < len(projected) - 3 and return_extent >= 0.05,
            {
                "apex_frame": apex_index,
                "return_extent_m": round(return_extent, 6),
            },
            {
                "minimum_frames_after_apex": 3,
                "minimum_return_extent_m": 0.05,
            },
        )
    else:
        raise ValueError(f"unsupported incline motion: {context.motion}")
