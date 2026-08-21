"""Single registry for grouped motion planners and auditors."""

from __future__ import annotations

from collections.abc import Callable

from . import ballistic, incline, planar, transition
from .contracts import MotionAuditContext, MotionDerivationContext, MotionPlan


Deriver = Callable[[MotionDerivationContext, MotionPlan], MotionPlan]
Auditor = Callable[[MotionAuditContext], None]


_GROUPS = {
    "planar": planar.MOTIONS,
    "ballistic": ballistic.MOTIONS,
    "incline": incline.MOTIONS,
    "transition": transition.MOTIONS,
}

_ASSET_PROFILE_GROUPS = {
    "vertical_drop": "ballistic",
    "workbench_clear_zone_drop": "ballistic",
    "resting_push": "planar",
    "diagonal_push": "planar",
    "workbench_long_axis_push": "planar",
    "edge_exit": "transition",
}

_DERIVERS: dict[str, Deriver] = {
    **{motion: planar.derive for motion in planar.MOTIONS},
    **{motion: ballistic.derive for motion in ballistic.MOTIONS},
    **{motion: incline.derive for motion in incline.MOTIONS},
    **{motion: transition.derive for motion in transition.MOTIONS},
}

_AUDITORS: dict[str, Auditor] = {
    **{motion: planar.audit for motion in planar.MOTIONS},
    **{motion: ballistic.audit for motion in ballistic.MOTIONS},
    **{motion: incline.audit for motion in incline.MOTIONS},
    **{motion: transition.audit for motion in transition.MOTIONS},
}

if _DERIVERS.keys() != _AUDITORS.keys():
    raise RuntimeError("motion planner and auditor registries must match")


def registered_motion_families() -> frozenset[str]:
    return frozenset(_DERIVERS)


def motion_group(motion: str) -> str:
    for group, motions in _GROUPS.items():
        if motion in motions:
            return group
    raise ValueError(f"unsupported motion family: {motion}")


def registered_asset_motion_profiles() -> frozenset[str]:
    return frozenset(_ASSET_PROFILE_GROUPS)


def asset_motion_group(profile: str) -> str:
    try:
        return _ASSET_PROFILE_GROUPS[profile]
    except KeyError as error:
        raise ValueError(
            f"unsupported curated-asset motion profile: {profile}"
        ) from error


def derive_motion(
    context: MotionDerivationContext, plan: MotionPlan
) -> MotionPlan:
    try:
        deriver = _DERIVERS[context.motion]
    except KeyError as error:
        raise ValueError(
            f"unsupported motion family: {context.motion}"
        ) from error
    return deriver(context, plan)


def audit_motion(context: MotionAuditContext) -> None:
    try:
        auditor = _AUDITORS[context.motion]
    except KeyError as error:
        raise ValueError(
            f"unsupported motion family: {context.motion}"
        ) from error
    auditor(context)
