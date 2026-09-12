#!/usr/bin/env python3
"""Generate, render, audit, and publish the canonical two-object dataset."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tools.cli.build_two_object_dataset import publish_dataset
from tools.cli.dataset_generation import (
    Layout,
    bind_generation_plan,
    generation_code_sha256,
    generation_layout as object_generation_layout,
    run,
    run_once,
    verify_render_manifest,
)
from tools.cli.two_object_admission import (
    PHYSICS_SWEEP_CONFIG,
    SWEEP_TARGET_OBJECT_INDICES,
    admit_two_object_groups,
)
from tools.core.hashing import sha256_file, sha256_json
from tools.core.json_io import read_json, write_json_atomic, write_json_atomic_sorted
from tools.core.paths import safe_scene_id
from tools.dataset_contract.two_object_metadata import validate_two_object_candidate_metadata
from tools.core.sweep_values import SWEEP_VARIANTS_PER_TARGET, sweep_group_size, sweep_target_indices
from tools.release.base_release_view import PipelineSpec
from tools.release.source_release import publish_source_release
from tools.sampling.assemble_two_object_base import SPECIALIZED_FAMILIES, assemble_base_manifest
from tools.sampling.two_object_sampling_request import effective_matrix, validate_sampling_request
from tools.sampling.sample_two_object_coverage import coverage_cells


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "physweep_two_object_generation_plan_v1"
GENERIC_SCHEMA = "physweep_pybullet_rigid_metadata_v1"
OBJECT_COUNT = 2
BLENDER_RUNTIME = Path("runtime/blender-3.4.0-linux-x64/blender")
TWO_OBJECT_MATRIX = Path("configs/two_object_sampling_matrix.json")
TWO_OBJECT_SCENE_RULES = Path("configs/two_object_scene_rules.json")
TWO_OBJECT_SPECIALIZED_RULES = Path(
    "configs/two_object_specialized_scene_rules.json"
)


def generation_layout(root: Path, work_id: str, release_root: Path) -> Layout:
    return object_generation_layout(
        root, work_id, release_root, object_count=OBJECT_COUNT
    )


def _source_binding(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def sampling_settings(root: Path, sampling_config: Path | None, generic_limit: int | None):
    rules = read_json(root / TWO_OBJECT_SPECIALIZED_RULES)
    request = None
    counts = {family['id']: len(family['profiles']) for family in rules['scene_families']}
    if sampling_config is not None:
        path = (root / sampling_config).resolve()
        path.relative_to(root.resolve())
        request = read_json(path)
        validate_sampling_request(request, rules)
        counts = request['family_base_counts']
        if generic_limit is not None and generic_limit != counts['generic']:
            raise ValueError('--generic-limit conflicts with the sampling config quota')
        generic_limit = counts['generic']
    matrix = effective_matrix(read_json(root / TWO_OBJECT_MATRIX), request)
    coverage_cells(matrix, generic_limit, scene_rules=read_json(root / TWO_OBJECT_SCENE_RULES))
    return request, matrix, counts, generic_limit


def validate_requested_base(base: dict, counts: dict, generic_limit: int | None) -> None:
    actual = base.get('family_counts', {})
    if (base.get('object_count') != OBJECT_COUNT
        or any(actual.get(name) != counts[name] for name in SPECIALIZED_FAMILIES)
        or (generic_limit is not None and actual.get('generic') != generic_limit)):
        raise ValueError('assembled two-object base differs from the request')


def freeze_effective_matrix(path: Path, matrix: dict) -> None:
    # Scale-bin maps follow the declared bin order in the existing matrix contract.
    if path.exists():
        if json.dumps(read_json(path)) != json.dumps(matrix):
            raise ValueError('frozen effective sampling matrix changed')
    else:
        write_json_atomic(path, matrix)


def freeze_sampled_metadata(root: Path, layout: Layout, counts: dict,
                            generic_limit: int | None, resume: bool) -> Path:
    """Recheck every source hash, including resume, without marking it admitted."""
    sampled = assemble_base_manifest(
        root, layout.base_dataset / 'generic/manifest.json',
        layout.base_dataset / 'specialized/manifest.json',
    )
    validate_requested_base(sampled, counts, generic_limit)
    for record in sampled["records"]:
        validate_two_object_candidate_metadata(root, read_json(root / record["metadata_path"]))
    path = layout.base_dataset / 'sampled_manifest.json'
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        if read_json(path) != sampled:
            raise ValueError('sampled metadata differs from its frozen checkpoint')
    else:
        write_json_atomic_sorted(path, sampled)
    return path


def generation_plan(
    root: Path,
    work_id: str,
    release_root: Path,
    *,
    released_base_manifest: Path,
    source_root: Path,
    source_manifest: Path,
    templates: dict[str, Path],
    generic_limit: int | None,
    specialized_seed: int,
    max_admission_attempts: int,
    sampling_config: Path | None = None,
) -> dict[str, Any]:
    layout = generation_layout(root, work_id, release_root)
    _, matrix, counts, generic_limit = sampling_settings(root, sampling_config, generic_limit)
    source_manifest_path = (
        source_manifest.resolve()
        if source_manifest.is_absolute()
        else (source_root / source_manifest).resolve()
    )
    return {
        "schema_version": PLAN_SCHEMA,
        "generator_code_sha256": generation_code_sha256(),
        "work_id": safe_scene_id(work_id),
        "request": {
            "generic_limit": generic_limit,
            "specialized_base_count": sum(counts[name] for name in SPECIALIZED_FAMILIES),
            'specialized_family_counts': {name: counts[name] for name in SPECIALIZED_FAMILIES},
            'effective_matrix_sha256': sha256_json(matrix),
            "sweep_target_object_indices": list(SWEEP_TARGET_OBJECT_INDICES),
            "specialized_seed": int(specialized_seed),
            "max_admission_attempts": int(max_admission_attempts),
        },
        "layout": {
            key: value.resolve().relative_to(root.resolve()).as_posix()
            for key, value in asdict(layout).items()
        },
        "sources": {
            **({'sampling_request': _source_binding(root / sampling_config)} if sampling_config else {}),
            "released_one_object_base": _source_binding(released_base_manifest),
            "one_object_generation_manifest": _source_binding(source_manifest_path),
            "specialized_templates": {
                family: _source_binding(path) for family, path in templates.items()
            },
            "rule_contracts": {
                name: _source_binding(root / path)
                for name, path in {
                    "sampling_matrix": TWO_OBJECT_MATRIX,
                    "generic_scene_rules": TWO_OBJECT_SCENE_RULES,
                    "specialized_scene_rules": TWO_OBJECT_SPECIALIZED_RULES,
                    "physics_sweep": PHYSICS_SWEEP_CONFIG,
                }.items()
            },
        },
        "stages": [
            "sample_generic_and_specialized_base",
            "admit_physics_groups_and_base_cameras",
            "publish_source_release",
            "bind_sweep_to_base_camera",
            "render_all_base_then_derived",
            "materialize_canonical_base_and_sweep",
            "verify_canonical_release",
        ],
        "pipelines": [
            {
                "family": "generic",
                "source_schema_version": GENERIC_SCHEMA,
                "renderer_id": "generic",
            },
            *[
                {
                    "family": family,
                    "source_schema_version": schema,
                    "renderer_id": "two_object_specialized",
                }
                for family, schema in SPECIALIZED_FAMILIES.items()
            ],
        ],
    }


def render_sweep(
    *, root: Path, layout: Layout, workers: int, gpus: str, resume: bool
) -> None:
    """Render each admitted sample once, completing all bases before derivatives."""
    plan = read_json(layout.sweep_render / "manifest.json")
    if int(plan.get("object_count", -1)) != OBJECT_COUNT:
        raise ValueError("sweep render plan has the wrong object count")
    counts = {str(key): int(value) for key, value in plan["branch_counts"].items()}
    targets = sweep_target_indices(OBJECT_COUNT, plan.get("sweep_target_object_indices"))
    if targets != SWEEP_TARGET_OBJECT_INDICES:
        raise ValueError("two-object generation requires object_a-only sweep targets")
    group_size = sweep_group_size(len(targets))
    if set(counts) - {"generic", *SPECIALIZED_FAMILIES}:
        raise ValueError("sweep render plan contains unsupported families")
    if any(count < 0 or count % group_size for count in counts.values()):
        raise ValueError("sweep render plan must contain complete groups")

    bound_root = layout.sweep_render / "generic" / "bound"
    bound = bound_root / "bound_manifest.json"
    if counts.get("generic", 0):
        run_once(
            [
                sys.executable, "-m", "tools.rendering.bind_physics_sweep_visuals",
                "--root", str(root),
                "--sweep-manifest", str(layout.sweep_render / "generic" / "physics_manifest.json"),
                "--base-bound-manifest", str(layout.base_render / "generic" / "bound_manifest.json"),
                "--output-root", str(bound_root),
            ],
            completion=bound,
            resume=resume,
        )
    for kind, per_group, result_name in (
        ("base", 1, "base_render_manifest.json"),
        ("sweep", len(targets) * SWEEP_VARIANTS_PER_TARGET, "derived_render_manifest.json"),
    ):
        for branch in ("generic", *SPECIALIZED_FAMILIES):
            count = counts.get(branch, 0)
            if not count:
                continue
            render_root = bound_root if branch == "generic" else layout.sweep_render / branch
            result = render_root / result_name
            if branch == "generic":
                command = [
                    sys.executable, "-m", "tools.rendering.render_pybullet_manifest",
                    "--manifest", str(bound), "--sweep-kind", kind,
                ]
            else:
                partition = "base" if kind == "base" else "derived"
                command = [
                    sys.executable, "-m", "tools.rendering.render_asset_proxy_manifest",
                    "--renderer", "two_object_specialized",
                    "--manifest", str(render_root / f"{partition}_render_input_manifest.json"),
                ]
            command.extend([
                "--root", str(root), "--result-manifest", str(result),
                "--blender", str(BLENDER_RUNTIME), "--workers", str(workers), "--gpus", gpus,
            ])
            if resume:
                command.append("--resume")
            run(command)
            verify_render_manifest(result, count // group_size * per_group)


def release_specs(root: Path, layout: Layout) -> list[PipelineSpec]:
    release = read_json(layout.source_release / "manifest.json")
    metadata = read_json(root / str(release["metadata_manifest"]))
    selected = {str(record["source_schema_version"]) for record in metadata["records"]}
    branches = {GENERIC_SCHEMA: "generic", **{schema: family for family, schema in SPECIALIZED_FAMILIES.items()}}
    if selected - set(branches):
        raise ValueError(f"release contains unsupported schemas: {sorted(selected - set(branches))}")
    return [
        PipelineSpec(
            branches[schema],
            schema,
            root,
            (
                layout.sweep_render / "generic" / "bound"
                if schema == GENERIC_SCHEMA
                else layout.sweep_render / branches[schema]
            ),
        )
        for schema in sorted(selected)
    ]


def bind_render_stage(
    root: Path, layout: Layout, generation_plan_path: Path,
    camera_path: Path | None, *, resume: bool,
) -> dict[str, str] | None:
    """Check pending inputs; freeze them only after preparation commits."""
    def binding(path: Path) -> dict[str, str]:
        path = path.resolve()
        return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}

    camera_binding = binding(camera_path) if camera_path is not None else None
    release_binding = binding(layout.source_release / "manifest.json")
    base_binding = binding(layout.base_manifest)
    stage = {
        "schema_version": "physweep_two_object_render_execution_plan_v2",
        "generation_plan": binding(generation_plan_path),
        "source_release": release_binding,
        "admitted_base": base_binding,
        "accepted_base_cameras": camera_binding,
    }
    prepared = layout.sweep_render / "manifest.json"
    stage_path = generation_plan_path.with_name("render_execution_plan.json")
    existing = read_json(stage_path) if stage_path.exists() else None
    if not prepared.exists():
        if existing and existing.get("schema_version") == stage["schema_version"]:
            raise ValueError("committed render preparation is missing")
        return camera_binding
    previous = read_json(prepared)
    for path_key, hash_key, expected in (
        ("source_release", "source_release_sha256", release_binding),
        ("source_staged_base_manifest", "source_staged_base_manifest_sha256", base_binding),
    ):
        if (root / previous[path_key]).resolve() != (root / expected["path"]).resolve() or previous[hash_key] != expected["sha256"]:
            raise ValueError("prepared render inputs differ from the admitted release")
    previous_camera = previous.get("accepted_base_cameras")
    camera_matches = previous_camera is None and camera_binding is None
    if isinstance(previous_camera, dict) and camera_binding is not None:
        camera_matches = (
            (root / previous_camera["path"]).resolve() == (root / camera_binding["path"]).resolve()
            and previous_camera["sha256"] == camera_binding["sha256"]
        )
    if not camera_matches:
        raise ValueError("accepted cameras differ from the prepared render stage")
    stage["prepared_manifest"] = binding(prepared)
    if existing and existing.get("schema_version") == "physweep_two_object_render_execution_plan_v1":
        # The completed preparation above independently proves the actual inputs.
        # Keep immutable sampling and admitted data checks when migrating v1.
        for key in ("generation_plan", "source_release", "admitted_base"):
            if existing.get(key) != stage[key]:
                raise ValueError("legacy render plan differs from admitted data")
        write_json_atomic_sorted(stage_path, stage)
    else:
        bind_generation_plan(stage_path, stage, resume)
    return camera_binding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--work-id", required=True)
    parser.add_argument("--released-base-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--billiards-template", type=Path, required=True)
    parser.add_argument("--passive-pinball-template", type=Path, required=True)
    parser.add_argument("--marble-run-template", type=Path, required=True)
    parser.add_argument("--generic-limit", type=int)
    parser.add_argument('--sampling-config', type=Path,
                        help='Explicit family quotas and physical variation; production preset: configs/two_object_production_sampling.json')
    parser.add_argument('--accepted-base-cameras', type=Path,
                        help='Hash-bound reviewed specialized base cameras to inherit in all group members')
    parser.add_argument(
        "--specialized-seed", type=int, default=20260902,
        help="Seed for fixture scenes; generic sampling uses coverage_plan.seed in the matrix",
    )
    parser.add_argument("--physics-workers", type=int, default=24)
    parser.add_argument("--render-workers", type=int, default=64)
    parser.add_argument("--max-admission-attempts", type=int, default=8)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--release-root", type=Path, default=Path("outputs/two_object"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument('--metadata-only', action='store_true',
                        help='Stop after validating sampled base metadata; continue with the same command and --resume, omitting this flag')
    parser.add_argument('--admission-only', action='store_true',
                        help='Stop after physics and base admission, before camera acceptance and rendering; resume with --accepted-base-cameras after review')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metadata_only and getattr(args, "admission_only", False):
        raise ValueError("choose either --metadata-only or --admission-only")
    if min(args.physics_workers, args.render_workers) < 1:
        raise ValueError("worker values must be positive")
    if args.max_admission_attempts < 1:
        raise ValueError("--max-admission-attempts must be positive")
    if args.generic_limit is not None and args.generic_limit < 1:
        raise ValueError("--generic-limit must be positive")
    if not [value for value in args.gpus.split(",") if value.strip()]:
        raise ValueError("--gpus must contain at least one id")
    root = args.root.resolve()
    source_root = args.source_root.resolve()
    released_base_manifest = args.released_base_manifest.resolve()
    templates = {
        "billiards": args.billiards_template.resolve(),
        "passive_pinball": args.passive_pinball_template.resolve(),
        "marble_run": args.marble_run_template.resolve(),
    }
    plan = generation_plan(
        root,
        args.work_id,
        args.release_root,
        released_base_manifest=released_base_manifest,
        source_root=source_root,
        source_manifest=args.source_manifest,
        templates=templates,
        generic_limit=args.generic_limit,
        specialized_seed=args.specialized_seed,
        max_admission_attempts=args.max_admission_attempts,
        sampling_config=args.sampling_config,
    )
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    layout = generation_layout(root, args.work_id, args.release_root)
    plan_path = root / "outputs" / safe_scene_id(args.work_id) / "generation_plan.json"
    bind_generation_plan(plan_path, plan, args.resume)
    _, matrix, counts, args.generic_limit = sampling_settings(root, args.sampling_config, args.generic_limit)
    matrix_path = root / TWO_OBJECT_MATRIX
    if args.sampling_config is not None:
        matrix_path = layout.base_dataset.parent / 'inputs/sampling_matrix.json'
        freeze_effective_matrix(matrix_path, matrix)

    generic_root = layout.base_dataset / "generic"
    generic_command = [
        sys.executable,
        "-m",
        "tools.sampling.sample_two_object_coverage",
        "--root",
        str(root),
        "--released-base-manifest",
        str(released_base_manifest),
        "--source-root",
        str(source_root),
        "--source-manifest",
        str(args.source_manifest),
        "--matrix",
        str(matrix_path),
        "--scene-rules",
        str(root / TWO_OBJECT_SCENE_RULES),
        "--output-dir",
        str(generic_root),
    ]
    if args.generic_limit is not None:
        generic_command.extend(("--limit", str(args.generic_limit)))
    run_once(
        generic_command,
        completion=generic_root / "manifest.json",
        resume=args.resume,
    )
    specialized_root = layout.base_dataset / "specialized"
    run_once(
        [
            sys.executable,
            "-m",
            "tools.sampling.sample_two_object_specialized",
            "--root",
            str(root),
            "--rules",
            str(root / TWO_OBJECT_SPECIALIZED_RULES),
            "--billiards-template",
            str(templates["billiards"]),
            "--passive-pinball-template",
            str(templates["passive_pinball"]),
            "--marble-run-template",
            str(templates["marble_run"]),
            "--output-dir",
            str(specialized_root),
            "--seed",
            str(args.specialized_seed),
            *(['--sampling-config', str(root / args.sampling_config)] if args.sampling_config else []),
        ],
        completion=specialized_root / "manifest.json",
        resume=args.resume,
    )
    sampled_path = freeze_sampled_metadata(root, layout, counts, args.generic_limit, args.resume)
    current_plan = generation_plan(
        root, args.work_id, args.release_root,
        released_base_manifest=released_base_manifest, source_root=source_root,
        source_manifest=args.source_manifest, templates=templates,
        generic_limit=args.generic_limit, specialized_seed=args.specialized_seed,
        max_admission_attempts=args.max_admission_attempts, sampling_config=args.sampling_config,
    )
    if current_plan != plan:
        raise ValueError('generation inputs or code changed during metadata sampling')
    if args.metadata_only:
        print(json.dumps({'status': 'sampled_pending_simulation',
                          'manifest': str(sampled_path),
                          'family_counts': read_json(sampled_path)['family_counts']}, sort_keys=True))
        return
    admitted = admit_two_object_groups(
        root=root,
        layout=layout,
        initial_generic_manifest=generic_root / "manifest.json",
        specialized_manifest=specialized_root / "manifest.json",
        physics_workers=args.physics_workers,
        camera_workers=args.render_workers,
        max_attempts=args.max_admission_attempts,
        resume=args.resume,
    )
    base = read_json(admitted.base)
    validate_requested_base(base, counts, args.generic_limit)
    if getattr(args, "admission_only", False):
        print(json.dumps({"status": "admitted_pending_camera_acceptance_and_render",
                          "base_manifest": str(admitted.base),
                          "base_physics_manifest": str(admitted.base_physics),
                          "sweep_physics_manifest": str(admitted.sweep_physics)}, sort_keys=True))
        return
    if not (layout.source_release / "manifest.json").is_file():
        publish_source_release(
            root=root,
            base_manifest_path=admitted.base,
            sweep_metadata_manifest_path=admitted.sweep_metadata,
            sweep_physics_manifest_path=admitted.sweep_physics,
            output=layout.source_release,
            object_count=OBJECT_COUNT,
            target_object_indices=SWEEP_TARGET_OBJECT_INDICES,
            dataset_id="physweep_two_object",
            release_schema="physweep_two_object_source_release_v1",
        )
    elif not args.resume:
        raise FileExistsError(layout.source_release)
    camera_path = getattr(args, "accepted_base_cameras", None)
    if camera_path is not None:
        camera_path = (root / camera_path).resolve()
        camera_path.relative_to(root)
    camera_binding = bind_render_stage(root, layout, plan_path, camera_path, resume=args.resume)
    run_once(
        [
            sys.executable,
            "-m",
            "tools.rendering.prepare_sweep_render_manifests",
            "--root",
            str(root),
            "--release-manifest",
            str(layout.source_release / "manifest.json"),
            "--staged-base-manifest",
            str(layout.base_manifest),
            "--output-root",
            str(layout.sweep_render),
            *(['--accepted-base-cameras', str(camera_path),
               '--accepted-base-cameras-sha256', camera_binding['sha256']]
              if camera_binding is not None else []),
        ],
        completion=layout.sweep_render / "manifest.json",
        resume=args.resume,
    )
    # Recheck after preparation as well: run_once on resume must not hide a stale
    # camera binding, and a camera file changed during preparation cannot render.
    bind_render_stage(root, layout, plan_path, camera_path, resume=True)
    render_sweep(
        root=root,
        layout=layout,
        workers=args.render_workers,
        gpus=args.gpus,
        resume=args.resume,
    )
    result = publish_dataset(
        release_project_root=root,
        release_manifest=layout.source_release / "manifest.json",
        release_root=layout.canonical_release,
        pipeline_specs=release_specs(root, layout),
        workers=args.render_workers,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
