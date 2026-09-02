"""Deterministic visual-environment bindings for admitted scene assets."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from tools.assets.environment_collision import procedural_room_objects
from tools.core.hashing import sha256_file as sha256


def pbr_binding(root: Path, material_id: str) -> dict[str, Any]:
    texture_dir = root / "assets/library/polyhaven/materials" / material_id / "textures"
    patterns = {
        "base_color": "*_diff_4k.*",
        "roughness": "*_rough_4k.*",
        "normal": "*_nor_gl_4k.*",
    }
    channels: dict[str, Any] = {}
    for channel, pattern in patterns.items():
        matches = sorted(texture_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"missing {channel} texture for {material_id}")
        path = matches[0]
        channels[channel] = {
            "path": str(path.relative_to(root)),
            "sha256": sha256(path),
        }
    return {"material_id": material_id, "channels": channels}


def choose_environment(
    root: Path,
    support: dict[str, Any],
    hdri_records: list[dict[str, Any]],
    visual_rules: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    support_category = str(support["semantic_category"])
    if "game_table" in support_category:
        preferred_names = {"abandoned_games_room_01", "abandoned_games_room_02"}
        candidates = [record for record in hdri_records if record["name"] in preferred_names]
    else:
        candidates = [
            record
            for record in hdri_records
            if record["role"] in {"indoor_neutral", "studio_soft"}
            and record["tier"] == "primary"
        ]
    if not candidates:
        raise ValueError(f"no HDRI candidates for {support_category}")
    weights = [float(record.get("sample_weight", 1.0)) for record in candidates]
    selected = rng.choices(candidates, weights=weights, k=1)[0]
    floor_ids = (
        ["wood_floor_4k", "rectangular_parquet_4k"]
        if "game_table" in support_category
        else ["wood_floor_4k", "concrete_floor_worn_001_4k", "garage_floor_4k"]
    )
    wall_ids = ["white_plaster_02_4k"]
    render_rules = visual_rules["asset_proxy_render"]
    return {
        "name": str(selected["name"]),
        "path": str(selected["source_path"]),
        "sha256": str(selected["sha256"]),
        "role": str(selected["role"]),
        "tier": str(selected["tier"]),
        "strength": round(
            rng.uniform(*[float(value) for value in render_rules["world_strength"]]),
            4,
        ),
        "rotation_degrees": round(rng.uniform(0.0, 360.0), 4),
        "lighting": render_rules,
        "room": {
            "floor_material": pbr_binding(root, rng.choice(floor_ids)),
            "wall_material": pbr_binding(root, rng.choice(wall_ids)),
            "half_extent_m": 3.4,
            "height_m": 3.2,
            "texture_repeat_per_m": 0.72,
        },
    }


def render_only_backdrop_objects(
    scene_profile: dict[str, Any],
    camera: dict[str, Any],
    scene_anchor_m: list[float],
    minimum_back_wall_distance_m: float,
) -> list[dict[str, Any]]:
    """Resolve a procedural room profile without creating physics colliders."""

    anchor = [float(value) for value in scene_anchor_m]
    if len(anchor) != 3 or not all(math.isfinite(value) for value in anchor):
        raise ValueError("specialized background anchor must be a finite xyz vector")
    position = [float(value) for value in camera["position_m"]]
    if len(position) != 3 or not all(math.isfinite(value) for value in position):
        raise ValueError("specialized background camera must contain a finite position")
    dx = position[0] - anchor[0]
    dy = position[1] - anchor[1]
    distance = math.hypot(dx, dy)
    if distance <= 1.0e-8:
        raise ValueError("specialized background camera has no horizontal direction")
    outward = [dx / distance, dy / distance]
    lateral = [-outward[1], outward[0]]
    wall_distance = max(
        float(scene_profile["back_wall_distance_m"]),
        float(minimum_back_wall_distance_m),
    )
    if not math.isfinite(wall_distance) or wall_distance <= 0.0:
        raise ValueError("specialized background wall distance must be positive")
    wall_yaw = math.degrees(math.atan2(outward[1], outward[0])) - 90.0
    return procedural_room_objects(
        scene_profile,
        scene_anchor=anchor,
        outward=outward,
        lateral=lateral,
        wall_distance=wall_distance,
        wall_yaw_degrees=wall_yaw,
        collision_enabled=False,
    )


def choose_specialized_environment(
    root: Path,
    *,
    family_id: str,
    background_contract: dict[str, Any],
    scene_profile: dict[str, Any],
    camera: dict[str, Any],
    scene_anchor_m: list[float],
    hdri_records: list[dict[str, Any]],
    visual_rules: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any]:
    """Freeze a family-approved, non-physical room around a specialized fixture."""

    profile_id = str(scene_profile["id"])
    if profile_id not in set(background_contract["profile_ids"]):
        raise ValueError(f"{profile_id} is not admitted for {family_id}")
    if (
        background_contract.get("physics_role") != "render_only_context"
        or background_contract.get("collision_enabled") is not False
    ):
        raise ValueError("specialized backgrounds must remain render-only")
    category = str(scene_profile["environment_category"])
    roles = list(visual_rules["hdri_roles_by_environment"][category])
    candidates = [
        record
        for record in hdri_records
        if str(record["role"]) in roles and str(record["tier"]) == "primary"
    ]
    if not candidates:
        candidates = [
            record for record in hdri_records if str(record["role"]) in roles
        ]
    if not candidates:
        raise ValueError(f"no admitted HDRI for specialized environment {category}")
    selected = rng.choices(
        candidates,
        weights=[float(record.get("sample_weight", 1.0)) for record in candidates],
        k=1,
    )[0]
    floor_id = rng.choice(visual_rules["room_floor_pools_by_environment"][category])
    wall_id = rng.choice(visual_rules["wall_pools_by_environment"][category])
    strength_range = visual_rules["hdri_strength_ranges_by_environment"][category]
    render_rules = visual_rules["asset_proxy_render"]
    backdrop = render_only_backdrop_objects(
        scene_profile,
        camera,
        scene_anchor_m,
        float(background_contract["minimum_back_wall_distance_m"]),
    )
    return {
        "binding_version": "physweep_specialized_render_context_v1",
        "family_id": family_id,
        "profile_id": profile_id,
        "environment_category": category,
        "physics_role": "render_only_context",
        "collision_enabled": False,
        "name": str(selected["name"]),
        "path": str(selected["source_path"]),
        "sha256": str(selected["sha256"]),
        "role": str(selected["role"]),
        "tier": str(selected["tier"]),
        "strength": round(
            rng.uniform(float(strength_range[0]), float(strength_range[1])), 4
        ),
        "rotation_degrees": round(rng.uniform(0.0, 360.0), 4),
        "lighting": render_rules,
        "room": {
            "floor_material": pbr_binding(root, str(floor_id)),
            "wall_material": pbr_binding(root, str(wall_id)),
            "center_xy_m": [float(scene_anchor_m[0]), float(scene_anchor_m[1])],
            "half_extent_m": 4.2,
            "height_m": 3.6,
            "texture_repeat_per_m": 0.72,
            "wall_mode": "profile_backdrop_only",
        },
        "backdrop_objects": backdrop,
    }


def resolve_specialized_environment_binding(
    root: Path,
    contract: dict[str, Any],
    family: dict[str, Any],
    profile: dict[str, Any],
    camera: dict[str, Any],
    *,
    scene_anchor_m: list[float],
    seed: int,
) -> dict[str, Any]:
    """Freeze paths and deterministic choices for one specialized background."""

    root = root.resolve()
    loaded: dict[str, tuple[Path, dict[str, Any]]] = {}
    for name, source in contract["visual_sources"].items():
        path = (root / str(source["path"])).resolve()
        path.relative_to(root)
        document = json.loads(path.read_text(encoding="utf-8"))
        version = document.get("schema_version", document.get("version"))
        if version != source["schema_version"]:
            raise ValueError(f"{name} source schema changed")
        loaded[name] = (path, document)
    profile_id = str(profile.get("id", ""))
    if profile_id not in {str(record["id"]) for record in family["profiles"]}:
        raise ValueError(f"{profile_id} does not belong to {family['id']}")
    scene_profiles = {
        str(record["id"]): record
        for record in loaded["scene_profiles"][1]["profiles"]
    }
    background_id = str(profile["background_profile_id"])
    binding = choose_specialized_environment(
        root,
        family_id=str(family["id"]),
        background_contract=family["background_contract"],
        scene_profile=scene_profiles[background_id],
        camera=camera,
        scene_anchor_m=scene_anchor_m,
        hdri_records=list(loaded["hdri_manifest"][1]["records"]),
        visual_rules=loaded["visual_sampling"][1],
        rng=random.Random(int(seed)),
    )
    binding["sources"] = {
        name: {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
        }
        for name, (path, _) in loaded.items()
    }
    return binding
