"""Semantic and provenance guards for the unsimulated two-object checkpoint."""

from pathlib import Path
from typing import Any

from tools.core.hashing import sha256_file
from tools.core.json_io import read_json


def validate_two_object_candidate_metadata(root: Path, scene: dict[str, Any]) -> None:
    if "admission" in scene:
        raise ValueError("two-object candidate inherits a source admission declaration")
    dimensions = scene.get("semantic_sampling", {}).get("five_dimensions")
    if dimensions:
        request = scene["camera_request"]
        expected = {
            "camera_profile": request["profile"],
            "observation_intent": request["observation"]["intent"],
            "structure_context": request["observation"]["structure_context"],
        }
        if dimensions.get("camera_observation") != expected:
            raise ValueError("two-object camera semantics differ from the camera request")
    else:
        semantics = scene["semantics"]
        motion = semantics.get("motion_profile")
        if (semantics.get("dynamic_object_count") != 2 or not motion
                or motion != scene["simulation"]["interaction"]["motion_pattern"]):
            raise ValueError("two-object specialized motion semantics disagree")
        if semantics["scene_family"] == "billiards":
            for key in ("semantic_rules", "physical_proxy_catalog"):
                binding = scene[key]
                path = root / binding["path"]
                if sha256_file(path) != binding["sha256"]:
                    raise ValueError(f"stale two-object billiards {key} binding")
            catalog = read_json(root / scene["physical_proxy_catalog"]["path"])
            records_hash = scene["physical_proxy_catalog"]["records_sha256"]
            if (catalog["records_sha256"] != records_hash
                    or sha256_file(root / catalog["records_path"]) != records_hash):
                raise ValueError("stale two-object billiards catalog records binding")
