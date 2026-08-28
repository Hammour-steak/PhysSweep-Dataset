#!/usr/bin/env python3
"""Pure helpers for explicit, physics-invariant PhysSweep lighting profiles."""

from __future__ import annotations

import copy
import math
from typing import Any


DEFAULT_LIGHTING_QUALITY_RULE = {
    "rule_id": "pbr_surface_glare_guard_v1",
    "physics_invariant": True,
    "area_key": {
        "minimum_size_m": 1.6,
        "maximum_energy_per_square_meter": 200.0,
    },
    "environment_floor": {
        "minimum_roughness": 0.62,
        "maximum_specular": 0.14,
    },
}


def default_lighting_quality_rule() -> dict[str, Any]:
    """Return the versioned visual-only default without sharing mutable state."""

    return copy.deepcopy(DEFAULT_LIGHTING_QUALITY_RULE)


def finite_number(value: Any, label: str, *, minimum: float = 0.0, maximum: float | None = None) -> float:
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{label} must be finite and at least {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label} must be at most {maximum}")
    return number


def validated_lighting_quality_rule(value: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate an explicit glare guard and return a normalized copy."""

    rule = default_lighting_quality_rule()
    if value:
        for key, item in value.items():
            if isinstance(item, dict) and isinstance(rule.get(key), dict):
                rule[key].update(item)
            else:
                rule[key] = item
    if str(rule.get("rule_id")) != "pbr_surface_glare_guard_v1":
        raise ValueError("unsupported lighting quality rule")
    area_key = rule["area_key"]
    floor = rule["environment_floor"]
    area_key["minimum_size_m"] = finite_number(
        area_key["minimum_size_m"], "area_key.minimum_size_m", minimum=0.05
    )
    area_key["maximum_energy_per_square_meter"] = finite_number(
        area_key["maximum_energy_per_square_meter"],
        "area_key.maximum_energy_per_square_meter",
        minimum=1.0,
    )
    floor["minimum_roughness"] = finite_number(
        floor["minimum_roughness"], "environment_floor.minimum_roughness", minimum=0.0, maximum=1.0
    )
    floor["maximum_specular"] = finite_number(
        floor["maximum_specular"], "environment_floor.maximum_specular", minimum=0.0, maximum=1.0
    )
    rule["physics_invariant"] = bool(rule.get("physics_invariant", True))
    return rule


def derive_glare_guarded_lighting_rule(
    lighting_rule: dict[str, Any], quality_rule: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create an auditable visual-only lighting candidate from a source rule.

    Blender AREA lights are square by default, so energy divided by ``size**2``
    is a useful source-independent proxy for their radiance. The derived rule
    writes the effective values into metadata rather than relying on a hidden
    renderer override.
    """

    result = copy.deepcopy(lighting_rule)
    quality = validated_lighting_quality_rule(quality_rule)
    key_light = result.setdefault("key_light", {})
    source_size = finite_number(key_light.get("size", 1.6), "key_light.size", minimum=0.05)
    source_energy = finite_number(key_light.get("energy", 0.0), "key_light.energy", minimum=0.0)
    area_key = quality["area_key"]
    effective_size = max(source_size, float(area_key["minimum_size_m"]))
    energy_cap = float(area_key["maximum_energy_per_square_meter"]) * effective_size**2
    effective_energy = min(source_energy, energy_cap)
    key_light["size"] = round(effective_size, 6)
    key_light["energy"] = round(effective_energy, 6)
    result["lighting_quality_rule"] = quality
    result["lighting_quality_derivation"] = {
        "source_key_light": {
            "energy": round(source_energy, 6),
            "size": round(source_size, 6),
            "energy_per_square_meter": round(source_energy / source_size**2, 6),
        },
        "effective_key_light": {
            "energy": round(effective_energy, 6),
            "size": round(effective_size, 6),
            "energy_per_square_meter": round(effective_energy / effective_size**2, 6),
        },
        "derivation_policy": "minimum_area_and_maximum_radiance_v1",
    }
    return result


def floor_glare_guard(lighting_rule: dict[str, Any] | None) -> dict[str, float] | None:
    """Return explicit backdrop-floor material limits, if the metadata requests them."""

    if not isinstance(lighting_rule, dict):
        return None
    raw_rule = lighting_rule.get("lighting_quality_rule")
    if not isinstance(raw_rule, dict):
        return None
    quality = validated_lighting_quality_rule(raw_rule)
    floor = quality["environment_floor"]
    return {
        "minimum_roughness": float(floor["minimum_roughness"]),
        "maximum_specular": float(floor["maximum_specular"]),
    }
