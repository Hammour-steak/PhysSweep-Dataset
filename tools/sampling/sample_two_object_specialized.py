#!/usr/bin/env python3
"""Bind reviewed two-object motions to frozen specialized fixture templates."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

from tools.assets.billiards_template import refresh_billiards_fixture

from tools.assets.visual_environment_binding import (
    resolve_specialized_environment_binding,
)
from tools.core.hashing import implementation_file_binding, relative_file_binding, sha256_file, sha256_json
from tools.core.json_io import read_json, write_json_atomic
from tools.dataset_contract.object_identity_contract import attach_object_identity
from tools.motion_rules.two_object.specialized import (
    family_index,
    load_two_object_specialized_rules,
    resolve_billiards_initial_states,
    resolve_marble_run_initial_states,
    resolve_pinball_initial_states,
    resolve_specialized_camera_binding,
)
from tools.sampling.two_object_sampling_request import validate_sampling_request, varied_profiles


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES = Path("configs/two_object_specialized_scene_rules.json")
RENDERER = Path("tools/rendering/render_two_object_specialized_scene.py")
SCHEMA_VERSION = "physweep_two_object_specialized_base_manifest_v1"
SOURCE_SCHEMAS = {
    "billiards": "physweep_billiards_scene_v4",
    "passive_pinball": "physweep_passive_pinball_scene_v1",
    "marble_run": "physweep_marble_run_scene_v1",
}


def _binding(root: Path, path: Path) -> dict[str, str]:
    resolved = root / path if not path.is_absolute() else path
    return relative_file_binding(root, resolved)


def _output_paths(
    root: Path,
    physics_output: Path,
    render_output: Path,
    scene_id: str,
) -> dict[str, str]:
    def relative(path: Path) -> str:
        return path.resolve().relative_to(root.resolve()).as_posix()

    return {
        "trajectory": relative(physics_output / "trajectory.npz"),
        "audit": relative(physics_output / "trajectory_audit.json"),
        "simulation_record": relative(physics_output / "simulation_record.json"),
        "inspection_frames": relative(render_output / "inspection_frames"),
        "video": relative(render_output / f"{scene_id}.mp4"),
        "masks": relative(render_output / "masks"),
    }


def _template_object(template: dict[str, Any]) -> dict[str, Any]:
    objects = template.get("simulation", {}).get("objects", [])
    if len(objects) != 1:
        raise ValueError("specialized fixture template must contain exactly one object")
    return objects[0]


def _sphere_objects(
    template: dict[str, Any], profile: dict[str, Any], states: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    source = _template_object(template)
    result = []
    for declared, state in zip(profile["objects"], states):
        record = copy.deepcopy(source)
        record["object_id"] = str(declared["object_id"])
        record["semantic_type"] = str(declared["semantic_label"])
        record["initial_state"] = copy.deepcopy(state)
        record["initial_state"].pop("object_id", None)
        result.append(record)
    return result


def prepare_specialized_templates(
    root: Path, templates: dict[str, dict[str, Any]], rules_path: Path,
) -> dict[str, dict[str, Any]]:
    """Rebind current provenance only after verifying unchanged fixture content."""
    prepared = copy.deepcopy(templates)
    billiards = prepared["billiards"]
    billiards = refresh_billiards_fixture(root, billiards)
    prepared["billiards"] = billiards
    billiards["semantic_rules"] = _binding(root, rules_path)
    return prepared


def build_specialized_scene(
    root: Path,
    dataset_root: Path,
    template: dict[str, Any],
    contract: dict[str, Any],
    family: dict[str, Any],
    profile: dict[str, Any],
    *,
    seed: int,
    variation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one immutable 2obj source scene without changing fixture physics."""

    family_id = str(family["id"])
    if template.get("schema_version") != SOURCE_SCHEMAS[family_id]:
        raise ValueError(f"wrong {family_id} template schema")
    if template.get("sweep", {}).get("kind") == "sweep":
        raise ValueError("specialized templates must be canonical base records")
    scene = copy.deepcopy(template)
    scene.pop("sweep", None)
    # The hash-bound source template preserves its own 1obj admission evidence.
    scene.pop("admission", None)
    scene_id = f"physweep2specialized_{profile['id']}"
    if variation is not None:
        scene_id += f"_v{variation['variant_index']:03d}_{variation['initial_profile_sha256'][:12]}"
    paths = _output_paths(
        root,
        dataset_root / "physics" / scene_id,
        dataset_root / "render" / family_id / scene_id,
        scene_id,
    )
    scene["scene_id"] = scene_id
    scene["seed"] = int(seed)
    scene["dataset_id"] = "physweep_two_object"
    scene["dataset_stage"] = "two_object_specialized_base_candidate"
    if variation is not None:
        scene['two_object_sampling'] = copy.deepcopy(variation)
    scene["semantics"]["scene_family"] = family_id
    scene["semantics"]["dynamic_object_count"] = 2
    scene["semantics"]["motion_profile"] = str(profile["id"])
    scene["semantics"]["description"] = str(profile["description"])
    scene.setdefault("simulation", {})["interaction"] = {
        "motion_pattern": str(profile["id"]),
        "interaction_class": "interacting",
        "contact_requirement": "must_contact",
    }

    if family_id == "billiards":
        physics = scene["physics"]
        radius = float(physics["ball_radius_m"])
        bed_z = float(
            physics["static_support_binding"]["target_support_frame"]
            ["safe_surface"]["z_m"]
        )
        states = resolve_billiards_initial_states(
            profile, bed_z_m=bed_z, ball_radius_m=radius
        )
        for declared, state in zip(profile["objects"], states):
            state["semantic_type"] = str(declared["semantic_label"])
            state["velocity_m_s"] = state.pop("linear_velocity_m_s")
            state.pop("orientation_quaternion_xyzw")
            state.pop("angular_velocity_rad_s")
        physics["initial_states"] = states
        anchor = [0.0, 0.0, bed_z]
    elif family_id == "passive_pinball":
        source = _template_object(scene)
        radius = float(source["collision_proxy"]["radius_m"])
        states = resolve_pinball_initial_states(
            profile,
            fixture_source=scene["physics"]["fixture_source"],
            fixture_frame=scene["physics"]["fixture"]["frame"],
            ball_radius_m=radius,
        )
        scene["simulation"]["objects"] = _sphere_objects(scene, profile, states)
        anchor = [float(value) for value in scene["camera"]["target_m"]]
    else:
        source = _template_object(scene)
        radius = float(source["collision_proxy"]["radius_m"])
        states = resolve_marble_run_initial_states(
            profile,
            base_initial_state=source["initial_state"],
            ball_radius_m=radius,
        )
        scene["simulation"]["objects"] = _sphere_objects(scene, profile, states)
        anchor = [float(value) for value in scene["camera"]["target_m"]]

    scene["physics"]["two_object_quality"] = copy.deepcopy(profile["quality"])
    if family.get("sweep_domains"):
        scene["physics"]["sweep_domains"] = copy.deepcopy(
            family["sweep_domains"]
        )
    scene["physics"]["trajectory_path"] = paths["trajectory"]
    scene["physics"]["audit_path"] = paths["audit"]
    scene["physics"]["simulation_record_path"] = paths["simulation_record"]
    scene["camera"] = resolve_specialized_camera_binding(
        family, profile, scene["camera"]
    )
    scene["render"]["inspection_frame_dir"] = paths["inspection_frames"]
    scene["render"]["video_path"] = paths["video"]
    scene["render"]["environment"] = resolve_specialized_environment_binding(
        root,
        contract,
        family,
        profile,
        scene["camera"],
        scene_anchor_m=anchor,
        seed=int(seed),
    )
    scene["implementation"] = {
        "sampler": implementation_file_binding(root, Path(__file__)),
        "renderer": implementation_file_binding(root, PROJECT_ROOT / RENDERER),
        "render_evidence": implementation_file_binding(
            root, PROJECT_ROOT / "tools/rendering/specialized_render_evidence.py"
        ),
    }
    attach_object_identity(
        scene,
        trajectory_path=paths["trajectory"],
        mask_path=paths["masks"],
    )
    return scene


def build_specialized_scenes(
    root: Path,
    output_root: Path,
    templates: dict[str, dict[str, Any]],
    contract: dict[str, Any],
    *,
    seed: int,
    sampling_request: dict[str, Any] | None = None,
    rules_path: Path = DEFAULT_RULES,
) -> list[tuple[Path, dict[str, Any]]]:
    templates = prepare_specialized_templates(root, templates, rules_path)
    families = family_index(contract)
    if sampling_request is not None:
        validate_sampling_request(sampling_request, contract)
    result = []
    index = 0
    for family_id in ("billiards", "passive_pinball", "marble_run"):
        family = families[family_id]
        profiles = (
            [(profile, None) for profile in family['profiles']]
            if sampling_request is None else varied_profiles(
                family, sampling_request['family_base_counts'][family_id],
                sampling_request['specialized_variation'][family_id], seed,
            )
        )
        for profile, variation in profiles:
            index += 1
            scene = build_specialized_scene(
                root, output_root, templates[family_id], contract, family, profile,
                seed=int(seed) + index, variation=variation,
            )
            metadata_path = (
                output_root
                / "scenes"
                / family_id
                / (str(profile['id']) if variation is None else scene['scene_id'])
                / "metadata.json"
            )
            result.append(
                (
                    metadata_path,
                    scene,
                )
            )
    fingerprints = set()
    for _, scene in result:
        family_id = scene['semantics']['scene_family']
        states = (scene['physics']['initial_states'] if family_id == 'billiards'
                  else [obj['initial_state'] for obj in scene['simulation']['objects']])
        fingerprint = sha256_json([family_id, [
            [state['position_m'], state.get('linear_velocity_m_s', state.get('velocity_m_s'))]
            for state in states
        ]])
        if fingerprint in fingerprints:
            raise ValueError('specialized sampling produced duplicate physical initial states')
        fingerprints.add(fingerprint)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--billiards-template", type=Path, required=True)
    parser.add_argument("--passive-pinball-template", type=Path, required=True)
    parser.add_argument("--marble-run-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument('--sampling-config', type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rules_path = (root / args.rules).resolve() if not args.rules.is_absolute() else args.rules.resolve()
    output = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    output.relative_to(root)
    if output.exists() and any(output.iterdir()):
        raise ValueError('specialized sampling output directory must be empty')
    template_paths = {
        "billiards": args.billiards_template.resolve(),
        "passive_pinball": args.passive_pinball_template.resolve(),
        "marble_run": args.marble_run_template.resolve(),
    }
    for path in template_paths.values():
        path.relative_to(root)
    contract = load_two_object_specialized_rules(root, rules_path)
    request_path = (root / args.sampling_config).resolve() if args.sampling_config else None
    if request_path is not None:
        request_path.relative_to(root)
    scenes = build_specialized_scenes(
        root,
        output,
        {family: read_json(path) for family, path in template_paths.items()},
        contract,
        seed=int(args.seed),
        sampling_request=read_json(request_path) if request_path else None,
        rules_path=rules_path,
    )
    samples = []
    for metadata_path, scene in scenes:
        write_json_atomic(metadata_path, scene)
        samples.append(
            {
                "scene_id": scene["scene_id"],
                "family": scene["semantics"]["scene_family"],
                "motion_profile": scene["semantics"]["motion_profile"],
                "metadata_path": metadata_path.relative_to(root).as_posix(),
                "metadata_sha256": sha256_file(metadata_path),
                "render_output": {
                    "inspection_frame_dir": scene["render"][
                        "inspection_frame_dir"
                    ],
                    "video_path": scene["render"]["video_path"],
                },
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "physweep_two_object",
        "sample_count": len(samples),
        "rules": _binding(root, rules_path),
        "templates": {
            family: _binding(root, path) for family, path in template_paths.items()
        },
        "samples": samples,
        "status": "sampled_pending_simulation",
    }
    manifest_path = output / "manifest.json"
    if request_path is not None:
        manifest['sampling_request'] = _binding(root, request_path)
        manifest['family_counts'] = {
            family: sum(sample['family'] == family for sample in samples)
            for family in SOURCE_SCHEMAS
        }
    write_json_atomic(manifest_path, manifest)
    print(manifest_path)


if __name__ == "__main__":
    main()
