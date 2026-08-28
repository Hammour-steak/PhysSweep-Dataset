"""Deterministic visual-environment bindings for admitted scene assets."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

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
