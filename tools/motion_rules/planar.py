"""Planar sliding and rolling rules."""

from __future__ import annotations

import math

import numpy as np

from rigid_geometry import clamp, cross

from .contracts import MotionAuditContext, MotionDerivationContext, MotionPlan


MOTIONS = frozenset({"slide_push_1obj", "roll_or_slide_1obj"})


def derive(context: MotionDerivationContext, plan: MotionPlan) -> MotionPlan:
    target_time = {"short": 1.10, "medium": 1.42, "long": 1.72}[
        str(context.trajectory_extent["label"])
    ]
    if context.motion == "roll_or_slide_1obj" and (
        context.shape == "sphere" or context.pose_profile == "side_on_motion"
    ):
        minimum_rolling_distance = context.desired_distance * 0.48
        calibrated_speed_floor = float(
            context.backend["contact"]["rolling_speed_safety_factor"]
        ) * math.sqrt(
            2.0
            * float(
                context.backend["contact"][
                    "rolling_deceleration_calibration_m_s2"
                ]
            )
            * minimum_rolling_distance
        )
        speed = clamp(
            max(
                1.18 * context.desired_distance / max(target_time, 1.0),
                calibrated_speed_floor,
            ),
            0.52,
            float(context.backend["contact"]["rolling_speed_max_m_s"]),
        )
        plan.linear_velocity_m_s = [
            context.direction[0] * speed,
            context.direction[1] * speed,
            0.0,
        ]
        radius = float(context.size_m[0]) / 2.0
        if context.pose_profile == "side_on_motion":
            normal = [
                float(value)
                for value in context.support["surface_frame"]["normal"]
            ]
            tangent_velocity = [
                context.direction[0] * speed,
                context.direction[1] * speed,
                0.0,
            ]
            tangent_velocity = [
                tangent_velocity[index]
                - normal[index]
                * sum(
                    tangent_velocity[axis] * normal[axis]
                    for axis in range(3)
                )
                for index in range(3)
            ]
            plan.angular_velocity_rad_s = [
                value / radius for value in cross(normal, tangent_velocity)
            ]
        else:
            plan.angular_velocity_rad_s = [
                -context.direction[1] * speed / radius,
                context.direction[0] * speed / radius,
                0.0,
            ]
        plan.effective_contact_friction = clamp(
            context.sampled_friction, 0.25, 0.85
        )
        plan.expected_motion["contact_mode"] = "rolling"
    else:
        speed = clamp(
            2.0 * context.desired_distance / target_time, 0.55, 2.45
        )
        plan.effective_contact_friction = min(
            context.sampled_friction,
            max(
                0.035,
                speed**2 / (2.0 * 9.81 * context.desired_distance),
            ),
        )
        plan.linear_velocity_m_s = [
            context.direction[0] * speed,
            context.direction[1] * speed,
            0.0,
        ]
        if context.shape == "cylinder":
            plan.angular_velocity_rad_s[2] = context.rng.uniform(-1.2, 1.2)
        plan.expected_motion["contact_mode"] = "sliding_or_tumbling"
    plan.expected_motion.update(
        {
            "target_displacement_m": round(context.desired_distance, 6),
            "minimum_displacement_m": round(
                context.desired_distance * 0.48, 6
            ),
            "minimum_support_contact_fraction": 0.45,
        }
    )
    return plan


def audit(context: MotionAuditContext) -> None:
    minimum_fraction = float(
        context.expected.get("minimum_support_contact_fraction", 0.0)
    )
    primary_contacts = context.primary_contacts > 0
    observed_fraction = context.support_fraction
    observed_duration = float(
        context.metadata["simulation"]["time"]["duration_s"]
    )
    exit_frame = None
    if context.expected.get("allow_support_exit_after_primary_motion"):
        contact_indices = np.flatnonzero(primary_contacts)
        if contact_indices.size:
            first_contact = int(contact_indices[0])
            exit_frame = None
            for index in range(first_contact + 1, len(primary_contacts) - 2):
                if not primary_contacts[index : index + 3].any():
                    exit_frame = index
                    break
            end_frame = exit_frame if exit_frame is not None else len(primary_contacts)
            primary_window = primary_contacts[first_contact:end_frame]
            observed_fraction = (
                float(np.mean(primary_window)) if primary_window.size else 0.0
            )
            output_fps = float(context.metadata["simulation"]["time"]["output_fps"])
            observed_duration = max(0.0, (end_frame - first_contact) / output_fps)
        else:
            observed_fraction = 0.0
            observed_duration = 0.0
    minimum_duration = float(
        context.expected.get("minimum_active_duration_s", 0.0)
    )
    context.check(
        "sustained_support_contact",
        observed_fraction >= minimum_fraction and observed_duration >= minimum_duration,
        {
            "primary_contact_fraction": round(observed_fraction, 6),
            "primary_contact_duration_s": round(observed_duration, 6),
            "support_exit_frame": exit_frame,
        },
        {
            "minimum_contact_fraction": minimum_fraction,
            "minimum_primary_duration_s": minimum_duration,
        },
    )
    if context.expected.get("contact_mode") != "rolling":
        return
    radius = float(context.obj["geometry"]["size_m"][0]) / 2.0
    active_speed_threshold = max(0.05, 0.10 * float(np.max(context.speed)))
    active = (
        (context.speed >= active_speed_threshold)
        & (context.primary_contacts > 0)
    )
    if np.any(active):
        rolling_ratio = float(
            np.median(
                context.angular_speed[active]
                * radius
                / np.maximum(context.speed[active], 1.0e-6)
            )
        )
    else:
        rolling_ratio = 0.0
    rolling_range = [
        float(value)
        for value in context.limits.get(
            "rolling_coupling_ratio_range", [0.75, 1.35]
        )
    ]
    context.check(
        "rolling_coupling",
        rolling_range[0] <= rolling_ratio <= rolling_range[1],
        round(rolling_ratio, 6),
        rolling_range,
    )
