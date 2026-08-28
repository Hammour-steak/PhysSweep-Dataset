"""Object-count-neutral one-factor sweep value construction."""

from __future__ import annotations

import math
from typing import Any


SWEEP_AXES = ("mass_kg", "contact_friction", "contact_restitution")
SWEEP_LEVEL_COUNT = 5
SWEEP_BASE_LEVEL_INDEX = SWEEP_LEVEL_COUNT // 2
SWEEP_DERIVED_LEVELS = tuple(
    index for index in range(SWEEP_LEVEL_COUNT) if index != SWEEP_BASE_LEVEL_INDEX
)
SWEEP_DERIVED_COUNT = len(SWEEP_AXES) * len(SWEEP_DERIVED_LEVELS)
SWEEP_GROUP_SIZE = 1 + SWEEP_DERIVED_COUNT


def round_sweep_value(value: float) -> float:
    """Round a sweep value using the published metadata precision."""

    return round(float(value), 6)


def allowed_sweep_domain(
    base_value: float,
    axis_rules: dict[str, Any],
    mass_bounds: list[float] | None,
    axis: str,
    domain_override: list[float] | None,
) -> list[float]:
    if axis == "mass_kg":
        if mass_bounds is None:
            return [base_value * 0.5, base_value * 2.0]
        return [float(mass_bounds[0]), float(mass_bounds[1])]
    if domain_override is not None:
        return [float(domain_override[0]), float(domain_override[1])]
    return [float(value) for value in axis_rules["domain"]]


def resolve_sweep_domain(
    base_value: float,
    axis_rules: dict[str, Any],
    allowed_domain: list[float],
) -> list[float]:
    allowed_low, allowed_high = allowed_domain
    policy = axis_rules.get("range_policy", {})
    mode = str(policy.get("mode", "global"))
    if mode == "relative_multipliers":
        low = max(allowed_low, base_value * float(policy["lower_multiplier"]))
        high = min(allowed_high, base_value * float(policy["upper_multiplier"]))
    elif mode == "linear_symmetric_span":
        span = max(
            float(policy["minimum_absolute_span"]),
            abs(base_value) * float(policy["relative_span"]),
        )
        low = max(allowed_low, base_value - span)
        high = min(allowed_high, base_value + span)
    elif mode == "global":
        low, high = allowed_low, allowed_high
    else:
        raise ValueError(f"unsupported range policy: {mode}")

    if low < high and low <= base_value <= high:
        return [low, high]
    return [allowed_low, allowed_high]


def sweep_values(
    base_value: float,
    axis_rules: dict[str, Any],
    mass_bounds: list[float] | None,
    axis: str,
    domain_override: list[float] | None = None,
    endpoint_policy: dict[str, Any] | None = None,
) -> list[float]:
    allowed_domain = allowed_sweep_domain(
        base_value,
        axis_rules,
        mass_bounds,
        axis,
        domain_override,
    )
    low, high = resolve_sweep_domain(base_value, axis_rules, allowed_domain)
    if not (math.isfinite(base_value) and low < high):
        raise ValueError(f"invalid sweep domain for {axis}: {low}, {high}")
    if not (low <= base_value <= high):
        raise ValueError(
            f"base value {base_value} is outside the declared {axis} domain "
            f"[{low}, {high}]"
        )

    endpoint_policy = endpoint_policy or {}
    fixed_positions = [
        float(value)
        for value in endpoint_policy.get(
            "normalized_positions", axis_rules["level_positions"]
        )
    ]
    expected_count = int(endpoint_policy.get("level_count", axis_rules["level_count"]))
    if len(fixed_positions) != expected_count:
        raise ValueError(f"{axis} has inconsistent level count")
    scale = axis_rules["scale"]
    if scale == "log" and (low <= 0.0 or high <= 0.0 or base_value <= 0.0):
        raise ValueError(f"log sweep requires positive {axis} bounds")
    if scale not in {"linear", "log"}:
        raise ValueError(f"unsupported sweep scale: {scale}")

    def interpolate(start: float, end: float, fraction: float) -> float:
        if scale == "log":
            return start * math.exp(fraction * math.log(end / start))
        return start + fraction * (end - start)

    base_policy = endpoint_policy.get(
        "base_value_policy", "preserve_exactly_at_nearest_position"
    )
    middle_policy = base_policy == "preserve_exactly_at_middle_position"
    middle_index = expected_count // 2
    is_interior_base = low < base_value < high

    if middle_policy and expected_count % 2 == 1 and is_interior_base:
        if middle_index == 0:
            raise ValueError(f"{axis} middle policy needs at least three levels")
        if (
            fixed_positions[0] != 0.0
            or fixed_positions[middle_index] != 0.5
            or fixed_positions[-1] != 1.0
            or any(left >= right for left, right in zip(fixed_positions, fixed_positions[1:]))
        ):
            raise ValueError(
                f"{axis} middle policy requires ordered positions with a 0.5 center"
            )
        values = []
        for index, position in enumerate(fixed_positions):
            if index <= middle_index:
                fraction = position / fixed_positions[middle_index]
                value = interpolate(low, base_value, fraction)
            else:
                fraction = (
                    position - fixed_positions[middle_index]
                ) / (fixed_positions[-1] - fixed_positions[middle_index])
                value = interpolate(base_value, high, fraction)
            values.append(round_sweep_value(value))
        values[middle_index] = round_sweep_value(base_value)
    else:
        if middle_policy and is_interior_base:
            raise ValueError(
                f"{axis} middle policy requires an odd number of sweep levels"
            )
        if middle_policy and not is_interior_base:
            edge_policy = endpoint_policy.get("edge_policy")
            raise ValueError(
                f"{axis} base cannot occupy the middle level under edge policy "
                f"{edge_policy!r}; widen the declared domain or reject this base"
            )

        if scale == "log":
            base_position = math.log(base_value / low) / math.log(high / low)
        else:
            base_position = (base_value - low) / (high - low)
        base_position = min(1.0, max(0.0, base_position))
        nearest = min(
            range(len(fixed_positions)),
            key=lambda index: abs(fixed_positions[index] - base_position),
        )
        fixed_positions[nearest] = base_position
        positions = sorted(fixed_positions)
        values = [
            round_sweep_value(interpolate(low, high, position))
            for position in positions
        ]
        closest = min(
            range(len(values)), key=lambda index: abs(values[index] - base_value)
        )
        values[closest] = round_sweep_value(base_value)
    if len(set(values)) != len(values):
        raise ValueError(f"{axis} domain is too narrow for five distinct levels")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError(f"{axis} values are not strictly ordered after rounding")
    return values
