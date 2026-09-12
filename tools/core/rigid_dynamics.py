"""Analytic primitive rigid-body inertia in the body frame."""
from __future__ import annotations
import numpy as np

from tools.core.hashing import sha256_json


def principal_inertia(shape: str, size_m: np.ndarray, mass_kg: float) -> np.ndarray:
    x, y, z = np.asarray(size_m, dtype=np.float64)
    if shape == "sphere":
        value = 0.4 * mass_kg * (x / 2.0) ** 2
        return np.repeat(value, 3)
    if shape == "cylinder":
        radius = max(x, y) / 2.0
        transverse = mass_kg * (3.0 * radius * radius + z * z) / 12.0
        return np.asarray([transverse, transverse, 0.5 * mass_kg * radius * radius])
    return mass_kg * np.asarray([y * y + z * z, x * x + z * z, x * x + y * y]) / 12.0


def expected_object_inertia(record: dict) -> np.ndarray:
    """Return an independently bound inertia for a primitive or compound body."""
    material = record.get("material", {})
    mass = float(material.get("mass_kg", 0.0))
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("rigid object mass must be finite and positive")
    profile = record.get("collision_profile", {})
    if profile.get("type") != "compound":
        geometry = record.get("geometry", {})
        return principal_inertia(
            str(geometry.get("type", "")),
            np.asarray(geometry.get("size_m"), dtype=np.float64),
            mass,
        )
    reference = profile.get("inertia_reference")
    if not isinstance(reference, dict) or set(reference) != {
        "schema_version",
        "method",
        "reference_mass_kg",
        "diagonal_kg_m2",
        "collision_profile_sha256",
    }:
        raise ValueError("compound collision profile lacks a bound inertia reference")
    if (
        reference["schema_version"] != "physweep_compound_inertia_reference_v1"
        or reference["method"] != "released_base_runtime_v1"
        or reference["collision_profile_sha256"]
        != sha256_json({"type": "compound", "colliders": profile.get("colliders")})
    ):
        raise ValueError("compound inertia reference does not match its colliders")
    reference_mass = float(reference["reference_mass_kg"])
    diagonal = np.asarray(reference["diagonal_kg_m2"], dtype=np.float64)
    if (
        not np.isfinite(reference_mass)
        or reference_mass <= 0.0
        or diagonal.shape != (3,)
        or not np.isfinite(diagonal).all()
        or bool(np.any(diagonal <= 0.0))
    ):
        raise ValueError("compound inertia reference is invalid")
    return diagonal * (mass / reference_mass)
