"""Free-fall, projectile, and bounce rules."""

from __future__ import annotations

import math

import numpy as np

from tools.core.rigid_geometry import clamp

from .common import (
    distance_upper_bound,
    precontact_lateral_drift,
    sampled_extremum_tolerance,
)
from .contracts import MotionAuditContext, MotionDerivationContext, MotionPlan


MOTIONS = frozenset(
    {
        "drop_fall_1obj",
        "projectile_1obj",
        "arc_projectile_1obj",
        "bounce_1obj",
    }
)


def bounce_observation_contract(
    backend: dict, shape: str, size_m: list[float]
) -> dict:
    rules = backend["base_parameter_rules"]["bounce_observation"]
    height_rules = rules["minimum_rebound_height"]
    minimum_height = clamp(
        float(height_rules["fraction_of_vertical_extent"]) * float(size_m[2]),
        float(height_rules["minimum_m"]),
        float(height_rules["maximum_m"]),
    )
    try:
        ratio_range = [
            float(value)
            for value in rules[
                "restitution_observation_ratio_range_by_shape"
            ][shape]
        ]
    except KeyError as error:
        raise ValueError(
            f"bounce observation has no rule for shape: {shape}"
        ) from error
    return {
        "minimum_rebound_height_m": round(minimum_height, 6),
        "restitution_observation_ratio_range": ratio_range,
    }


def derive(context: MotionDerivationContext, plan: MotionPlan) -> MotionPlan:
    if context.motion == "drop_fall_1obj":
        height_range = context.subtype.get("height", [0.4, 0.7])
        height = context.rng.uniform(
            float(height_range[0]), float(height_range[1])
        ) * 1.35
        plan.pose["position_m"][2] += height
        plan.expected_motion.update(
            {
                "contact_mode": "airborne_then_contact",
                "minimum_drop_m": round(height * 0.72, 6),
                "must_contact_primary_support": True,
            }
        )
        return plan

    if context.motion in {"projectile_1obj", "arc_projectile_1obj"}:
        height_range = context.subtype.get("height", [0.28, 0.48])
        launch_height = context.rng.uniform(
            float(height_range[0]), float(height_range[1])
        ) * 1.25
        plan.pose["position_m"][2] += launch_height
        if context.motion == "projectile_1obj":
            vertical_speed = 0.0
            flight_time = math.sqrt(2.0 * launch_height / 9.81)
        else:
            vertical_speed = context.rng.uniform(1.65, 2.35) * float(
                context.subtype.get("speed", 1.0)
            )
            flight_time = (
                vertical_speed
                + math.sqrt(vertical_speed**2 + 2.0 * 9.81 * launch_height)
            ) / 9.81
        horizontal_speed = clamp(
            context.desired_distance * 0.72 / max(flight_time, 0.2),
            0.65,
            2.65,
        )
        plan.linear_velocity_m_s = [
            context.direction[0] * horizontal_speed,
            context.direction[1] * horizontal_speed,
            vertical_speed,
        ]
        plan.expected_motion.update(
            {
                "contact_mode": "ballistic_then_contact",
                "minimum_airborne_frames": 4,
                "minimum_horizontal_displacement_m": round(
                    context.desired_distance * 0.35, 6
                ),
                "must_contact_primary_support": True,
            }
        )
        if context.motion == "arc_projectile_1obj":
            plan.expected_motion["must_have_vertical_apex"] = True
        return plan

    if context.motion == "bounce_1obj":
        launch_height = context.rng.uniform(0.30, 0.52)
        plan.pose["position_m"][2] += launch_height
        horizontal_speed = context.rng.uniform(0.62, 1.15) * float(
            context.subtype.get("speed", 1.0)
        )
        downward_speed = context.rng.uniform(0.35, 0.80) * float(
            context.subtype.get("vertical", 1.0)
        )
        plan.linear_velocity_m_s = [
            context.direction[0] * horizontal_speed,
            context.direction[1] * horizontal_speed,
            -downward_speed,
        ]
        plan.expected_motion.update(
            {
                "contact_mode": "impact_and_rebound",
                "must_contact_primary_support": True,
                **bounce_observation_contract(
                    context.backend, context.shape, context.size_m
                ),
            }
        )
        return plan

    raise ValueError(f"unsupported ballistic motion: {context.motion}")


def audit(context: MotionAuditContext) -> None:
    if context.motion == "drop_fall_1obj":
        lateral_drift = precontact_lateral_drift(
            context.positions, context.first_primary_contact
        )
        maximum_lateral_drift = float(
            context.limits.get("maximum_unforced_lateral_drift_m", 0.035)
        )
        effective_lateral_drift_limit = distance_upper_bound(
            maximum_lateral_drift,
            context.absolute_distance_tolerance,
            context.relative_distance_tolerance,
        )
        context.check(
            "unforced_lateral_drift",
            lateral_drift <= effective_lateral_drift_limit,
            round(lateral_drift, 6),
            effective_lateral_drift_limit,
        )
        drop = float(
            context.positions[0, 2] - context.positions[:, 2].min()
        )
        threshold = float(context.expected["minimum_drop_m"])
        context.check(
            "drop_extent", drop >= threshold, round(drop, 6), threshold
        )
        return

    if context.motion in {"projectile_1obj", "arc_projectile_1obj"}:
        contact_index = (
            context.first_primary_contact
            if context.first_primary_contact is not None
            else len(context.positions) - 1
        )
        airborne_frames = int(
            np.sum(context.all_contacts[: max(1, contact_index)] == 0)
        )
        horizontal_before_contact = float(
            np.linalg.norm(
                context.positions[contact_index, :2]
                - context.positions[0, :2]
            )
        )
        context.check(
            "airborne_frames",
            airborne_frames
            >= int(context.expected["minimum_airborne_frames"]),
            airborne_frames,
            int(context.expected["minimum_airborne_frames"]),
        )
        threshold = float(
            context.expected["minimum_horizontal_displacement_m"]
        )
        context.check(
            "ballistic_horizontal_extent",
            horizontal_before_contact >= threshold,
            round(horizontal_before_contact, 6),
            threshold,
        )
        if context.expected.get("must_have_vertical_apex"):
            apex_gain = float(
                context.positions[:, 2].max() - context.positions[0, 2]
            )
            context.check(
                "vertical_apex", apex_gain >= 0.05, round(apex_gain, 6), 0.05
            )
        return

    if context.motion == "bounce_1obj":
        if context.first_primary_contact is None:
            return
        contact_z = float(
            context.positions[context.first_primary_contact, 2]
        )
        rebound_height = float(
            context.positions[context.first_primary_contact :, 2].max()
            - contact_z
        )
        threshold = float(context.expected["minimum_rebound_height_m"])
        output_fps = float(context.metadata["simulation"]["time"]["output_fps"])
        extremum_tolerance = sampled_extremum_tolerance(
            context.gravity,
            output_fps,
            context.extremum_interval_error_multiplier,
        )
        effective_rebound_threshold = max(
            0.0,
            threshold
            - max(
                context.absolute_distance_tolerance,
                threshold * context.relative_distance_tolerance,
                extremum_tolerance,
            ),
        )
        context.check(
            "visible_rebound",
            rebound_height >= effective_rebound_threshold,
            round(rebound_height, 6),
            round(effective_rebound_threshold, 6),
        )
        impact_drop = max(
            0.0, float(context.positions[0, 2] - contact_z)
        )
        effective_restitution = float(
            context.obj["material"]["contact_restitution"]
        ) * float(
            context.metadata["simulation"]["support"]["dynamics"][
                "restitution"
            ]
        )
        if impact_drop > 1.0e-6 and effective_restitution > 1.0e-6:
            observed_restitution = math.sqrt(
                max(rebound_height, 0.0) / impact_drop
            )
            restitution_ratio = observed_restitution / effective_restitution
            ratio_range = [
                float(value)
                for value in context.expected[
                    "restitution_observation_ratio_range"
                ]
            ]
            context.check(
                "bounce_response_matches_restitution",
                ratio_range[0] - context.dimensionless_ratio_tolerance
                <= restitution_ratio
                <= ratio_range[1] + context.dimensionless_ratio_tolerance,
                round(restitution_ratio, 6),
                [
                    ratio_range[0] - context.dimensionless_ratio_tolerance,
                    ratio_range[1] + context.dimensionless_ratio_tolerance,
                ],
            )
        return

    raise ValueError(f"unsupported ballistic motion: {context.motion}")
