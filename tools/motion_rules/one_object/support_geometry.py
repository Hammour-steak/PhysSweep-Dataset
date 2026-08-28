"""Map one-object motion semantics to backend-neutral support features."""

from __future__ import annotations

from tools.core.rigid_geometry import SupportGeometryPolicy


_WIDE_FLAT_MOTIONS = frozenset(
    {
        "drop_fall_1obj",
        "projectile_1obj",
        "arc_projectile_1obj",
        "edge_fall_1obj",
    }
)


def support_geometry_policy(motion: str) -> SupportGeometryPolicy:
    return SupportGeometryPolicy(
        layout_hint="wide_flat" if motion in _WIDE_FLAT_MOTIONS else None,
        transition_type={
            "edge_fall_1obj": "raised_edge_to_floor",
            "ramp_to_flat_1obj": "incline_to_horizontal",
        }.get(motion),
        add_landing_surface=motion == "ramp_to_flat_1obj",
        add_impact_wall=motion == "wall_impact_1obj",
    )
