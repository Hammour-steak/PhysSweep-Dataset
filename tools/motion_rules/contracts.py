"""Shared contracts for motion planners and trajectory auditors."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


Check = Callable[[str, bool, Any, Any], None]


@dataclass(frozen=True)
class MotionDerivationContext:
    rng: random.Random
    backend: dict[str, Any]
    motion: str
    subtype: dict[str, Any]
    trajectory_extent: dict[str, Any]
    shape: str
    size_m: list[float]
    pose_profile: str
    support: dict[str, Any]
    sampled_friction: float
    restitution: float
    clearance: float
    yaw: float
    direction: list[float]
    center_x: float
    center_y: float
    half_x: float
    half_y: float
    zone_x: float
    zone_y: float
    limit: float
    desired_distance: float
    trajectory_extent_fraction: float


@dataclass
class MotionPlan:
    pose: dict[str, Any]
    linear_velocity_m_s: list[float]
    angular_velocity_rad_s: list[float]
    effective_contact_friction: float
    expected_motion: dict[str, Any]

    def as_metadata(self) -> dict[str, Any]:
        return {
            "pose": self.pose,
            "linear_velocity_m_s": [
                round(float(value), 6) for value in self.linear_velocity_m_s
            ],
            "angular_velocity_rad_s": [
                round(float(value), 6) for value in self.angular_velocity_rad_s
            ],
            "effective_contact_friction": round(
                float(self.effective_contact_friction), 6
            ),
            "expected_motion": self.expected_motion,
        }


@dataclass(frozen=True)
class MotionAuditContext:
    metadata: dict[str, Any]
    trajectory: dict[str, np.ndarray]
    obj: dict[str, Any]
    object_id: str
    expected: dict[str, Any]
    motion: str
    positions: np.ndarray
    velocities: np.ndarray
    angular: np.ndarray
    primary_contacts: np.ndarray
    all_contacts: np.ndarray
    speed: np.ndarray
    angular_speed: np.ndarray
    support_fraction: float
    first_primary_contact: int | None
    required_contact_index: int | None
    limits: dict[str, Any]
    absolute_distance_tolerance: float
    relative_distance_tolerance: float
    dimensionless_ratio_tolerance: float
    extremum_interval_error_multiplier: float
    gravity: float
    check: Check
