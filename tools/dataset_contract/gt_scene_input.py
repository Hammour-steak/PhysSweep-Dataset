"""Compile an exact initial-scene surface into the fixed model point contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

GT_SURFACE_SCHEMA = "physweep.gt_initial_surface.v13"
MODEL_SCENE_SCHEMA = "physweep.model_scene_condition.v17"
ENVIRONMENT_SURFACE_POLICY = (
    "complete_ground_plus_camera_first_hit_visible_non_ground_at_t0"
)
DEFAULT_OBJECT_POINTS = 2048
DEFAULT_ENVIRONMENT_POINTS = 8192
MIN_VISIBLE_NON_GROUND_PART_POINTS = 8
REQUIRED_SURFACE_ARRAYS = {
    "xyz",
    "xyz_world",
    "normal",
    "rgb",
    "rgb_valid",
    "body_id",
    "scene_part_id",
    "visible_mask",
    "ground_completion_mask",
    "camera_from_world",
    "camera_intrinsics",
    "image_size_px",
    "controlled_object_id",
    "metadata_json",
}


def interaction_collider_ids(metadata: dict) -> tuple[str, ...]:
    """Return colliders used by the planned primary interaction."""
    simulation = metadata["simulation"]
    colliders = list(simulation["support"]["colliders"])
    by_id = {str(record["id"]): record for record in colliders}
    interaction = [
        str(record["id"])
        for record in colliders
        if str(record["role"]) == "primary_support"
    ]
    if not interaction:
        raise ValueError("scene metadata has no primary support collider")
    expected = simulation["objects"][0].get("expected_motion", {})
    required_id = expected.get("required_collider_contact_id")
    if required_id is not None:
        required_id = str(required_id)
        if required_id not in by_id:
            raise ValueError(f"unknown required contact collider: {required_id}")
        interaction.append(required_id)
    return tuple(dict.fromkeys(interaction))


def ground_collider_id(metadata: dict) -> str:
    """Return the scene's single semantic ground."""
    support = metadata["simulation"]["support"]
    ground_flat = str(support.get("scene_class")) == "ground_flat"
    candidates = [
        str(record["id"])
        for record in support["colliders"]
        if str(record["role"]) == "environment_floor"
        or (ground_flat and str(record["role"]) == "primary_support")
    ]
    if len(candidates) != 1:
        raise ValueError(f"scene requires exactly one semantic ground; got {candidates}")
    return candidates[0]


def _stable_seed(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{value}:{seed}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_indices(
    mask: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    candidates = np.flatnonzero(mask)
    if len(candidates) < count:
        raise ValueError(
            f"scene category has only {len(candidates)} points; {count} required"
        )
    return np.sort(rng.choice(candidates, size=count, replace=False))


def sample_metric_surface_indices(
    xyz: np.ndarray,
    scene_part_id: np.ndarray,
    count: int,
    rng: np.random.Generator,
    retained_part_ids: tuple[str, ...] = (),
) -> tuple[np.ndarray, dict]:
    """Select a fixed metric-density surface subset with part retention."""
    if len(xyz) < count:
        raise ValueError(
            f"surface has only {len(xyz)} points; {count} required"
        )
    if scene_part_id.shape != (len(xyz),):
        raise ValueError("scene_part_id must have shape N")
    points = xyz.astype(np.float64)
    origin = points.min(axis=0)
    extent = max(float(np.ptp(points, axis=0).max()), 1e-6)
    tie_order = rng.permutation(len(points))

    def representatives(voxel_size: float) -> np.ndarray:
        cells = np.floor((points - origin) / voxel_size).astype(np.int64)
        _, first = np.unique(cells[tie_order], axis=0, return_index=True)
        return tie_order[first]

    low = extent * 1e-7
    high = extent
    best = np.arange(len(points), dtype=np.int64)
    best_voxel_size = low
    for _ in range(20):
        voxel_size = (low + high) * 0.5
        current = representatives(voxel_size)
        if len(current) >= count:
            best = current
            best_voxel_size = voxel_size
            low = voxel_size
        else:
            high = voxel_size
    candidate_part_ids = scene_part_id.astype(str)
    retained_parts = sorted(set(str(value) for value in retained_part_ids))
    unknown_parts = sorted(set(retained_parts) - set(candidate_part_ids.tolist()))
    if unknown_parts:
        raise ValueError(f"retained scene parts have no candidates: {unknown_parts}")
    if len(retained_parts) > count:
        raise ValueError(
            "scene has more retained parts than point slots"
        )
    minimum_per_part = min(
        MIN_VISIBLE_NON_GROUND_PART_POINTS,
        max(1, count // max(len(retained_parts), 1)),
    )
    reserved: list[np.ndarray] = []
    part_candidate_counts: dict[str, int] = {}
    for part_id in retained_parts:
        part_candidates = np.flatnonzero(candidate_part_ids == part_id)
        part_candidate_counts[part_id] = int(len(part_candidates))
        take = min(minimum_per_part, len(part_candidates))
        voxel_part_candidates = best[np.isin(best, part_candidates)]
        pool = (
            voxel_part_candidates
            if len(voxel_part_candidates) >= take
            else part_candidates
        )
        reserved.append(rng.choice(pool, size=take, replace=False))
    reserved_indices = (
        np.unique(np.concatenate(reserved))
        if reserved
        else np.empty(0, dtype=np.int64)
    )
    voxel_pool = np.setdiff1d(best, reserved_indices, assume_unique=False)
    remaining = count - len(reserved_indices)
    if len(voxel_pool) < remaining:
        raise ValueError("adaptive voxel pool cannot fill the environment point budget")
    fill = rng.choice(voxel_pool, size=remaining, replace=False)
    selected_relative = np.concatenate([reserved_indices, fill])
    selected = np.sort(selected_relative)
    selected_part_ids = scene_part_id[selected].astype(str)
    part_output_counts = {
        part_id: int(np.count_nonzero(selected_part_ids == part_id))
        for part_id in retained_parts
    }
    return selected, {
        "method": "adaptive_metric_voxel_surface_uniform_with_part_retention",
        "candidate_points": int(len(points)),
        "output_points": int(count),
        "voxel_size_m": float(best_voxel_size),
        "object_proximity_reweighting": False,
        "minimum_retained_part_points": int(minimum_per_part),
        "retained_part_candidate_counts": part_candidate_counts,
        "retained_part_output_counts": part_output_counts,
    }


def _environment_indices(
    xyz: np.ndarray,
    environment_mask: np.ndarray,
    scene_part_id: np.ndarray,
    ground_part_id: str,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict]:
    """Sample one metric-density environment pool and retain visible structures."""
    candidates = np.flatnonzero(environment_mask)
    candidate_part_ids = scene_part_id[candidates].astype(str)
    visible_non_ground_parts = tuple(
        sorted(set(candidate_part_ids.tolist()) - {str(ground_part_id)})
    )
    selected_relative, report = sample_metric_surface_indices(
        xyz[candidates],
        candidate_part_ids,
        count,
        rng,
        retained_part_ids=visible_non_ground_parts,
    )
    return np.sort(candidates[selected_relative]), {
        **report,
        "ground_part_id": str(ground_part_id),
    }


def _load_bound_metadata(
    source_metadata: dict,
    project_root: Path | None,
    bound_metadata: dict | None,
) -> dict:
    if bound_metadata is not None:
        return bound_metadata
    if project_root is None:
        raise ValueError(
            "project_root or bound_metadata is required to bind environment physics"
        )
    record = source_metadata.get("source_metadata", {})
    relative = record.get("path")
    if not relative:
        raise ValueError("GT surface does not identify its bound source metadata")
    path = (project_root.resolve() / str(relative)).resolve()
    if not path.is_relative_to(project_root.resolve()) or not path.is_file():
        raise FileNotFoundError(f"invalid bound source metadata: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _environment_material_arrays(
    bound_metadata: dict,
    scene_part_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Bind every static surface point to the material used by PyBullet."""
    support = bound_metadata["simulation"]["support"]
    default_dynamics = support.get("dynamics", {})
    default_friction = float(default_dynamics.get("lateral_friction", 1.0))
    default_restitution = float(default_dynamics.get("restitution", 1.0))
    if not np.isfinite(default_friction) or default_friction < 0.0:
        raise ValueError("environment friction must be finite and non-negative")
    if not np.isfinite(default_restitution) or not 0.0 <= default_restitution <= 1.0:
        raise ValueError("environment restitution must be finite and in [0, 1]")

    material_by_part: dict[str, tuple[float, float]] = {}

    def bind_colliders(
        colliders: list[dict], fallback: dict, environment: bool = False
    ) -> None:
        for collider in colliders:
            dynamics = collider.get("dynamics", fallback)
            friction = float(dynamics.get("lateral_friction", default_friction))
            restitution_value = float(
                dynamics.get("restitution", default_restitution)
            )
            if not np.isfinite(friction) or friction < 0.0:
                raise ValueError(
                    f"invalid environment friction for {collider['id']}"
                )
            if (
                not np.isfinite(restitution_value)
                or not 0.0 <= restitution_value <= 1.0
            ):
                raise ValueError(
                    f"invalid environment restitution for {collider['id']}"
                )
            collider_id = str(collider["id"])
            values = (friction, restitution_value)
            material_by_part[collider_id] = values
            if collider.get("render_replaced_by_solid_wedge"):
                material_by_part["solid_ramp_wedge"] = values
            if collider_id == "support":
                material_by_part["support_visual_mesh"] = values
            if environment and collider_id.startswith("environment_mesh_"):
                material_by_part[
                    "scene_mesh_" + collider_id[len("environment_mesh_") :]
                ] = values

    bind_colliders(support.get("colliders", []), default_dynamics)

    environment_binding = bound_metadata.get("environment_binding", {})
    binding_dynamics = environment_binding.get("dynamics")
    if binding_dynamics is not None:
        policy = str(binding_dynamics.get("policy", ""))
        if policy != "inherit_primary_support":
            raise ValueError(f"unsupported environment material policy: {policy}")
        if not np.isclose(
            float(binding_dynamics["lateral_friction"]), default_friction
        ) or not np.isclose(
            float(binding_dynamics["restitution"]), default_restitution
        ):
            raise ValueError("environment binding diverges from support dynamics")
        bind_colliders(
            environment_binding.get("colliders", []),
            binding_dynamics,
            environment=True,
        )

    friction = np.empty((len(scene_part_ids), 1), dtype=np.float32)
    restitution = np.empty((len(scene_part_ids), 1), dtype=np.float32)
    resolved_counts: dict[str, int] = {}
    string_part_ids = scene_part_ids.astype(str)
    for part_id in np.unique(string_part_ids):
        values = material_by_part.get(
            str(part_id), (default_friction, default_restitution)
        )
        mask = string_part_ids == str(part_id)
        friction[mask, 0] = values[0]
        restitution[mask, 0] = values[1]
        resolved_counts[str(part_id)] = int(mask.sum())
    return friction, restitution, {
        "policy": "per_surface_point_from_static_collider_material",
        "current_default": {
            "friction": default_friction,
            "restitution": default_restitution,
        },
        "resolved_scene_part_point_counts": resolved_counts,
    }


def inspect_gt_surface(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(REQUIRED_SURFACE_ARRAYS - set(archive.files))
        if missing:
            raise ValueError(f"GT scene surface is missing: {', '.join(missing)}")
        metadata = json.loads(str(archive["metadata_json"]))
        if metadata.get("schema") != GT_SURFACE_SCHEMA:
            raise ValueError("unexpected GT scene surface schema")
        if metadata.get("complete_object_surface") is not True:
            raise ValueError(
                "GT scene input requires the complete controlled-object surface"
            )
        if metadata.get("camera_first_hit_visible_non_ground_surface") is not True:
            raise ValueError(
                "GT scene input requires every camera-first-hit non-ground surface"
            )
        if metadata.get("ground_surface_completion") is not True:
            raise ValueError("GT scene input requires ground completion")
        if metadata.get("hidden_non_ground_surface_completion") is not False:
            raise ValueError("GT scene input forbids hidden non-ground completion")
        if metadata.get("metric_environment_density") is not True:
            raise ValueError("GT scene input requires metric environment density")
        if metadata.get("environment_surface_policy") != ENVIRONMENT_SURFACE_POLICY:
            raise ValueError(
                "GT scene input has an unexpected environment surface policy"
            )
        count = len(archive["xyz"])
        for name in ("xyz", "xyz_world", "normal", "rgb"):
            if archive[name].shape != (count, 3):
                raise ValueError(f"{name} must have shape N x 3")
            if not np.isfinite(archive[name]).all():
                raise ValueError(f"{name} contains non-finite values")
        for name in (
            "rgb_valid",
            "body_id",
            "scene_part_id",
            "visible_mask",
            "ground_completion_mask",
        ):
            if archive[name].shape != (count,):
                raise ValueError(f"{name} must have shape N")
        if archive["camera_from_world"].shape != (4, 4):
            raise ValueError("camera_from_world must have shape 4 x 4")
        if archive["camera_intrinsics"].shape != (3, 3):
            raise ValueError("camera_intrinsics must have shape 3 x 3")
        if archive["image_size_px"].shape != (2,):
            raise ValueError("image_size_px must contain width and height")
        object_mask = archive["body_id"] == 1
        environment_mask = ~object_mask
        if not object_mask.any() or not environment_mask.any():
            raise ValueError("GT scene surface must contain object and environment")
        environment_xyz = archive["xyz"][environment_mask].astype(np.float64)
        intrinsics = archive["camera_intrinsics"].astype(np.float64)
        width, height = [int(value) for value in archive["image_size_px"]]
        safe_z = np.maximum(environment_xyz[:, 2], 1e-6)
        u = intrinsics[0, 0] * environment_xyz[:, 0] / safe_z + intrinsics[0, 2]
        v = intrinsics[1, 2] - intrinsics[1, 1] * environment_xyz[:, 1] / safe_z
        clip_start = max(
            0.03,
            float(
                metadata.get("local_context", {}).get(
                    "camera_clip_start_m", 0.03
                )
            ),
        )
        inside_initial_frame = (
            (environment_xyz[:, 2] > clip_start)
            & (u >= 0.0)
            & (u < width)
            & (v >= 0.0)
            & (v < height)
        )
        if not inside_initial_frame.all():
            raise ValueError("environment points must project inside the initial frame")
        ground_completion_mask = archive["ground_completion_mask"].astype(bool)
        scene_part_id = archive["scene_part_id"].astype(str)
        if np.any(np.char.str_len(scene_part_id) == 0):
            raise ValueError("scene_part_id values must be non-empty")
        controlled_object_id = str(archive["controlled_object_id"])
        if set(scene_part_id[object_mask].tolist()) != {controlled_object_id}:
            raise ValueError("object points must use the controlled object scene_part_id")
        ground_part_id = str(metadata.get("ground_collider_id", ""))
        if not ground_part_id:
            raise ValueError("GT scene metadata requires ground_collider_id")
        if (ground_completion_mask & object_mask).any():
            raise ValueError("ground completion may contain only environment points")
        if set(scene_part_id[ground_completion_mask].tolist()) - {ground_part_id}:
            raise ValueError("ground completion points must belong to semantic ground")
        first_hit_mask = environment_mask & ~ground_completion_mask
        first_hit_xyz = archive["xyz"][first_hit_mask].astype(np.float64)
        if not len(first_hit_xyz):
            raise ValueError(
                "GT scene surface requires camera-first-hit environment points"
            )
        first_hit_z = first_hit_xyz[:, 2]
        first_hit_u = (
            intrinsics[0, 0] * first_hit_xyz[:, 0] / first_hit_z + intrinsics[0, 2]
        )
        first_hit_v = (
            intrinsics[1, 2] - intrinsics[1, 1] * first_hit_xyz[:, 1] / first_hit_z
        )
        pixel_x = np.floor(first_hit_u).astype(np.int64)
        pixel_y = np.floor(first_hit_v).astype(np.int64)
        pixel_ids = pixel_y * width + pixel_x
        if len(np.unique(pixel_ids)) != len(pixel_ids):
            raise ValueError(
                "camera-first-hit environment points must use unique pixels"
            )
        center_error = np.maximum(
            np.abs(first_hit_u - (pixel_x + 0.5)),
            np.abs(first_hit_v - (pixel_y + 0.5)),
        )
        # Float32 camera-space coordinates can amplify rounding near clip_start.
        if float(center_error.max()) > 0.05:
            raise ValueError(
                "camera-first-hit environment points must lie on pixel-center rays"
            )
        normal_lengths = np.linalg.norm(archive["normal"], axis=1)
        if float(np.min(normal_lengths)) < 0.98 or float(np.max(normal_lengths)) > 1.02:
            raise ValueError("GT scene surface normals are not unit length")
        return {
            "schema": metadata["schema"],
            "scene_id": metadata["scene_id"],
            "controlled_object_id": str(archive["controlled_object_id"]),
            "point_count": count,
            "object_point_count": int(object_mask.sum()),
            "environment_point_count": int(environment_mask.sum()),
            "ground_completion_point_count": int(ground_completion_mask.sum()),
            "visible_non_ground_scene_part_count": len(
                set(scene_part_id[first_hit_mask].tolist()) - {ground_part_id}
            ),
            "visible_point_count": int(archive["visible_mask"].astype(bool).sum()),
            "rgb_valid_point_count": int(archive["rgb_valid"].astype(bool).sum()),
            "units": metadata.get("units"),
            "local_context": metadata.get("local_context"),
        }


def compile_model_scene_condition(
    source_path: Path,
    output_path: Path,
    seed: int = 20260810,
    object_points: int = DEFAULT_OBJECT_POINTS,
    environment_points: int = DEFAULT_ENVIRONMENT_POINTS,
    project_root: Path | None = None,
    bound_metadata: dict | None = None,
) -> dict:
    """Compile one exact GT surface into the fixed model tensor contract."""
    inspect_gt_surface(source_path)
    with np.load(source_path, allow_pickle=False) as archive:
        source = {name: archive[name] for name in archive.files}
        source_metadata = json.loads(str(archive["metadata_json"]))
    scene_id = str(source_metadata["scene_id"])
    rng = np.random.default_rng(_stable_seed(scene_id, seed))
    body_id = source["body_id"].astype(np.int16)
    object_mask = body_id == 1
    environment_mask = ~object_mask
    environment_indices, environment_sampling = _environment_indices(
        source["xyz"],
        environment_mask,
        source["scene_part_id"],
        str(source_metadata["ground_collider_id"]),
        environment_points,
        rng,
    )
    indices = np.concatenate(
        [
            _sample_indices(object_mask, object_points, rng),
            environment_indices,
        ]
    )
    object_indices = indices[:object_points]
    environment_indices = indices[object_points:]
    resolved_bound_metadata = _load_bound_metadata(
        source_metadata, project_root, bound_metadata
    )
    environment_friction, environment_restitution, material_binding = (
        _environment_material_arrays(
            resolved_bound_metadata,
            source["scene_part_id"][environment_indices],
        )
    )
    width, height = [float(value) for value in source["image_size_px"]]
    intrinsics = source["camera_intrinsics"].astype(np.float64)
    camera_intrinsics_normalized = np.asarray(
        [
            intrinsics[0, 0] / width,
            intrinsics[1, 1] / height,
            intrinsics[0, 2] / width,
            intrinsics[1, 2] / height,
        ],
        dtype=np.float32,
    )
    scene = {
        "object_xyz_camera_m": source["xyz"][object_indices].astype(np.float32),
        "object_normal_camera": source["normal"][object_indices].astype(np.float32),
        "environment_xyz_camera_m": source["xyz"][environment_indices].astype(
            np.float32
        ),
        "environment_normal_camera": source["normal"][environment_indices].astype(
            np.float32
        ),
        "environment_friction": environment_friction,
        "environment_restitution": environment_restitution,
        "camera_intrinsics_normalized": camera_intrinsics_normalized,
    }
    scene_metadata = {
        "schema": MODEL_SCENE_SCHEMA,
        "source": "simulation_gt_complete_object_complete_ground_plus_camera_first_hit_visible_non_ground_at_t0",
        "source_surface": str(source_path),
        "source_surface_sha256": _sha256(source_path),
        "scene_id": scene_id,
        "coordinate_frame": "camera_right_up_forward",
        "units": "meters",
        "controlled_object_id": str(source["controlled_object_id"]),
        "complete_object_surface": bool(source_metadata["complete_object_surface"]),
        "camera_first_hit_visible_non_ground_surface": bool(
            source_metadata["camera_first_hit_visible_non_ground_surface"]
        ),
        "ground_surface_completion": bool(
            source_metadata["ground_surface_completion"]
        ),
        "hidden_non_ground_surface_completion": bool(
            source_metadata["hidden_non_ground_surface_completion"]
        ),
        "metric_environment_density": bool(
            source_metadata["metric_environment_density"]
        ),
        "environment_surface_policy": source_metadata["environment_surface_policy"],
        "point_layout": {
            "object": int(object_points),
            "environment": int(environment_points),
            "separate_arrays": True,
        },
        "environment_sampling": environment_sampling,
        "environment_material_binding": material_binding,
        "camera_condition": "normalized_pinhole_intrinsics",
        "approximations": source_metadata.get("approximations", []),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **scene,
        metadata_json=np.asarray(json.dumps(scene_metadata, sort_keys=True)),
    )
    inspection = inspect_model_scene_condition(output_path)
    return {
        "path": str(output_path),
        **inspection,
        "counts": {
            "object": int(object_points),
            "environment": int(environment_points),
        },
        "sampling_counts": {"uniform_environment": int(environment_points)},
        "source_surface_sha256": scene_metadata["source_surface_sha256"],
    }


def inspect_model_scene_condition(path: Path) -> dict:
    required = {
        "object_xyz_camera_m",
        "object_normal_camera",
        "environment_xyz_camera_m",
        "environment_normal_camera",
        "environment_friction",
        "environment_restitution",
        "camera_intrinsics_normalized",
        "metadata_json",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"model scene condition is missing: {', '.join(missing)}")
        unexpected = sorted(set(archive.files) - required)
        if unexpected:
            raise ValueError(
                f"model scene condition contains obsolete fields: {', '.join(unexpected)}"
            )
        metadata = json.loads(str(archive["metadata_json"]))
        if metadata.get("schema") != MODEL_SCENE_SCHEMA:
            raise ValueError("unexpected model scene condition schema")
        if metadata.get("complete_object_surface") is not True:
            raise ValueError(
                "model scene requires the complete controlled-object surface"
            )
        if metadata.get("camera_first_hit_visible_non_ground_surface") is not True:
            raise ValueError(
                "model scene requires every camera-first-hit non-ground surface"
            )
        if metadata.get("ground_surface_completion") is not True:
            raise ValueError("model scene requires ground completion")
        if metadata.get("hidden_non_ground_surface_completion") is not False:
            raise ValueError("model scene forbids hidden non-ground completion")
        if metadata.get("metric_environment_density") is not True:
            raise ValueError("model scene requires metric environment density")
        if metadata.get("environment_surface_policy") != ENVIRONMENT_SURFACE_POLICY:
            raise ValueError("model scene has an unexpected environment surface policy")
        object_count = len(archive["object_xyz_camera_m"])
        environment_count = len(archive["environment_xyz_camera_m"])
        for name, count in (
            ("object_xyz_camera_m", object_count),
            ("object_normal_camera", object_count),
            ("environment_xyz_camera_m", environment_count),
            ("environment_normal_camera", environment_count),
        ):
            if archive[name].shape != (count, 3):
                raise ValueError(f"{name} must have shape N x 3")
            if not np.isfinite(archive[name]).all():
                raise ValueError(f"{name} contains non-finite values")
        for name in ("environment_friction", "environment_restitution"):
            if archive[name].shape != (environment_count, 1):
                raise ValueError(f"{name} must have shape N x 1")
            if not np.isfinite(archive[name]).all():
                raise ValueError(f"{name} contains non-finite values")
        if np.any(archive["environment_friction"] < 0.0):
            raise ValueError("environment friction must be non-negative")
        if np.any(
            (archive["environment_restitution"] < 0.0)
            | (archive["environment_restitution"] > 1.0)
        ):
            raise ValueError("environment restitution must be in [0, 1]")
        normal_lengths = np.concatenate(
            [
                np.linalg.norm(archive["object_normal_camera"], axis=1),
                np.linalg.norm(archive["environment_normal_camera"], axis=1),
            ]
        )
        if float(normal_lengths.min()) < 0.98 or float(normal_lengths.max()) > 1.02:
            raise ValueError("model scene normals are not unit length")
        camera = archive["camera_intrinsics_normalized"]
        if camera.shape != (4,) or not np.isfinite(camera).all():
            raise ValueError("camera intrinsics must contain four finite values")
        if camera[0] <= 0.0 or camera[1] <= 0.0:
            raise ValueError("normalized focal lengths must be positive")
        return {
            "schema": metadata["schema"],
            "scene_id": metadata["scene_id"],
            "controlled_object_id": metadata["controlled_object_id"],
            "point_count": object_count + environment_count,
            "object_point_count": object_count,
            "environment_point_count": environment_count,
            "environment_friction_range": [
                float(archive["environment_friction"].min()),
                float(archive["environment_friction"].max()),
            ],
            "environment_restitution_range": [
                float(archive["environment_restitution"].min()),
                float(archive["environment_restitution"].max()),
            ],
        }


def write_gt_surface(
    output_path: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict,
) -> dict:
    if metadata.get("schema") != GT_SURFACE_SCHEMA:
        raise ValueError("GT scene metadata has an unexpected schema")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return inspect_gt_surface(output_path)
