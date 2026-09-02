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
    generation_layout as object_generation_layout,
    run,
    run_once,
    verify_render_manifest,
)
from tools.core.hashing import sha256_file
from tools.core.json_io import read_json
from tools.core.paths import safe_scene_id
from tools.core.sweep_values import SWEEP_VARIANTS_PER_TARGET, sweep_group_size
from tools.release.base_release_view import PipelineSpec
from tools.release.source_release import publish_source_release
from tools.sampling.assemble_two_object_base import SPECIALIZED_FAMILIES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAN_SCHEMA = "physweep_two_object_generation_plan_v1"
GENERIC_SCHEMA = "physweep_pybullet_rigid_metadata_v1"
OBJECT_COUNT = 2
SPECIALIZED_BASE_COUNT = 9


def generation_layout(root: Path, work_id: str, release_root: Path) -> Layout:
    return object_generation_layout(
        root, work_id, release_root, object_count=OBJECT_COUNT
    )


def _source_binding(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


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
    seed: int,
) -> dict[str, Any]:
    layout = generation_layout(root, work_id, release_root)
    source_manifest_path = (
        source_manifest.resolve()
        if source_manifest.is_absolute()
        else (source_root / source_manifest).resolve()
    )
    return {
        "schema_version": PLAN_SCHEMA,
        "work_id": safe_scene_id(work_id),
        "request": {
            "generic_limit": generic_limit,
            "specialized_base_count": SPECIALIZED_BASE_COUNT,
            "seed": int(seed),
        },
        "layout": {
            key: value.resolve().relative_to(root.resolve()).as_posix()
            for key, value in asdict(layout).items()
        },
        "sources": {
            "released_one_object_base": _source_binding(released_base_manifest),
            "one_object_generation_manifest": _source_binding(source_manifest_path),
            "specialized_templates": {
                family: _source_binding(path) for family, path in templates.items()
            },
        },
        "stages": [
            "sample_generic_and_specialized_base",
            "assemble_and_audit_base",
            "derive_and_audit_per_object_sweep",
            "stage_and_render_base_with_group_camera",
            "publish_source_release",
            "stage_and_render_sweep",
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


def render_base(
    *,
    root: Path,
    layout: Layout,
    sweep_physics_manifest: Path,
    workers: int,
    gpus: str,
    resume: bool,
) -> None:
    plan = read_json(layout.base_render / "render_plan.json")
    counts = {str(key): int(value) for key, value in plan["branch_counts"].items()}
    if counts.get("generic", 0):
        bound = layout.base_render / "generic" / "bound_manifest.json"
        run_once(
            [
                sys.executable,
                "-m",
                "tools.rendering.bind_pybullet_visuals",
                "--root",
                str(root),
                "--manifest",
                str(layout.base_render / "generic" / "physics_manifest.json"),
                "--output-root",
                str(layout.base_render / "generic"),
                "--workers",
                str(min(workers, counts["generic"])),
                "--camera-group-manifest",
                str(sweep_physics_manifest),
            ],
            root=root,
            completion=bound,
            resume=resume,
        )
        command = [
            sys.executable,
            "-m",
            "tools.rendering.render_pybullet_manifest",
            "--root",
            str(root),
            "--manifest",
            str(bound),
            "--workers",
            str(workers),
            "--gpus",
            gpus,
        ]
        if resume:
            command.append("--resume")
        run(command, root)
        verify_render_manifest(
            layout.base_render / "generic" / "render_manifest.json",
            counts["generic"],
        )
    for branch in SPECIALIZED_FAMILIES:
        count = counts.get(branch, 0)
        if not count:
            continue
        command = [
            sys.executable,
            "-m",
            "tools.rendering.render_asset_proxy_manifest",
            "--root",
            str(root),
            "--renderer",
            "two_object_specialized",
            "--manifest",
            str(layout.base_render / branch / "render_input_manifest.json"),
            "--workers",
            str(workers),
            "--gpus",
            gpus,
        ]
        if resume:
            command.append("--resume")
        run(command, root)
        verify_render_manifest(
            layout.base_render / branch / "render_manifest.json", count
        )


def render_sweep(
    *, root: Path, layout: Layout, workers: int, gpus: str, resume: bool
) -> None:
    plan = read_json(layout.sweep_render / "manifest.json")
    if int(plan.get("object_count", -1)) != OBJECT_COUNT:
        raise ValueError("sweep render plan has the wrong object count")
    counts = {str(key): int(value) for key, value in plan["branch_counts"].items()}
    group_size = sweep_group_size(OBJECT_COUNT)
    if counts.get("generic", 0):
        count = counts["generic"]
        if count % group_size:
            raise ValueError("generic two-object sweep groups are incomplete")
        groups = count // group_size
        bound_root = layout.sweep_render / "generic" / "bound"
        bound = bound_root / "bound_manifest.json"
        run_once(
            [
                sys.executable,
                "-m",
                "tools.rendering.bind_physics_sweep_visuals",
                "--root",
                str(root),
                "--sweep-manifest",
                str(layout.sweep_render / "generic" / "physics_manifest.json"),
                "--base-bound-manifest",
                str(layout.base_render / "generic" / "bound_manifest.json"),
                "--output-root",
                str(bound_root),
            ],
            root=root,
            completion=bound,
            resume=resume,
        )
        for kind, expected, result_name in (
            ("base", groups, "base_render_manifest.json"),
            (
                "sweep",
                groups * OBJECT_COUNT * SWEEP_VARIANTS_PER_TARGET,
                "derived_render_manifest.json",
            ),
        ):
            result = bound_root / result_name
            command = [
                sys.executable,
                "-m",
                "tools.rendering.render_pybullet_manifest",
                "--root",
                str(root),
                "--manifest",
                str(bound),
                "--sweep-kind",
                kind,
                "--result-manifest",
                str(result),
                "--workers",
                str(workers),
                "--gpus",
                gpus,
            ]
            if resume:
                command.append("--resume")
            run(command, root)
            verify_render_manifest(result, expected)
    for branch in SPECIALIZED_FAMILIES:
        count = counts.get(branch, 0)
        if not count:
            continue
        if count % group_size:
            raise ValueError(f"{branch} two-object sweep groups are incomplete")
        groups = count // group_size
        for partition, expected, result_name in (
            ("base", groups, "base_render_manifest.json"),
            (
                "derived",
                groups * OBJECT_COUNT * SWEEP_VARIANTS_PER_TARGET,
                "derived_render_manifest.json",
            ),
        ):
            result = layout.sweep_render / branch / result_name
            command = [
                sys.executable,
                "-m",
                "tools.rendering.render_asset_proxy_manifest",
                "--root",
                str(root),
                "--renderer",
                "two_object_specialized",
                "--manifest",
                str(
                    layout.sweep_render
                    / branch
                    / f"{partition}_render_input_manifest.json"
                ),
                "--result-manifest",
                str(result),
                "--workers",
                str(workers),
                "--gpus",
                gpus,
            ]
            if resume:
                command.append("--resume")
            run(command, root)
            verify_render_manifest(result, expected)


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
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--physics-workers", type=int, default=24)
    parser.add_argument("--render-workers", type=int, default=64)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--release-root", type=Path, default=Path("outputs/two_object"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.physics_workers, args.render_workers) < 1:
        raise ValueError("worker values must be positive")
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
        seed=args.seed,
    )
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    layout = generation_layout(root, args.work_id, args.release_root)
    plan_path = root / "outputs" / safe_scene_id(args.work_id) / "generation_plan.json"
    bind_generation_plan(plan_path, plan, args.resume)

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
        "--output-dir",
        str(generic_root),
    ]
    if args.generic_limit is not None:
        generic_command.extend(("--limit", str(args.generic_limit)))
    run_once(
        generic_command,
        root=root,
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
            "--billiards-template",
            str(templates["billiards"]),
            "--passive-pinball-template",
            str(templates["passive_pinball"]),
            "--marble-run-template",
            str(templates["marble_run"]),
            "--output-dir",
            str(specialized_root),
            "--seed",
            str(args.seed),
        ],
        root=root,
        completion=specialized_root / "manifest.json",
        resume=args.resume,
    )
    run_once(
        [
            sys.executable,
            "-m",
            "tools.sampling.assemble_two_object_base",
            "--root",
            str(root),
            "--generic-manifest",
            str(generic_root / "manifest.json"),
            "--specialized-manifest",
            str(specialized_root / "manifest.json"),
            "--output",
            str(layout.base_manifest),
        ],
        root=root,
        completion=layout.base_manifest,
        resume=args.resume,
    )
    base = read_json(layout.base_manifest)
    if (
        int(base.get("object_count", -1)) != OBJECT_COUNT
        or sum(
            int(base.get("family_counts", {}).get(family, 0))
            for family in SPECIALIZED_FAMILIES
        )
        != SPECIALIZED_BASE_COUNT
        or (
            args.generic_limit is not None
            and int(base.get("family_counts", {}).get("generic", -1))
            != args.generic_limit
        )
    ):
        raise ValueError("assembled two-object base differs from the request")
    base_physics = layout.base_dataset / "physics"
    run_once(
        [
            sys.executable,
            "-m",
            "tools.physics.run_pybullet_batch",
            "--root",
            str(root),
            "--manifest",
            str(layout.base_manifest),
            "--output-root",
            str(base_physics),
            "--workers",
            str(args.physics_workers),
        ],
        root=root,
        completion=base_physics / "manifest.json",
        resume=args.resume,
    )
    run_once(
        [
            sys.executable,
            "-m",
            "tools.sampling.derive_physics_sweep",
            "--root",
            str(root),
            "--base-manifest",
            str(layout.base_manifest),
            "--output-dir",
            str(layout.sweep_metadata),
        ],
        root=root,
        completion=layout.sweep_metadata / "manifest.json",
        resume=args.resume,
    )
    run_once(
        [
            sys.executable,
            "-m",
            "tools.physics.run_pybullet_batch",
            "--root",
            str(root),
            "--manifest",
            str(layout.sweep_metadata / "manifest.json"),
            "--output-root",
            str(layout.sweep_physics),
            "--workers",
            str(args.physics_workers),
        ],
        root=root,
        completion=layout.sweep_physics / "manifest.json",
        resume=args.resume,
    )
    run_once(
        [
            sys.executable,
            "-m",
            "tools.rendering.prepare_two_object_base_render_manifests",
            "--root",
            str(root),
            "--base-manifest",
            str(layout.base_manifest),
            "--physics-manifest",
            str(base_physics / "manifest.json"),
            "--output-root",
            str(layout.base_render),
        ],
        root=root,
        completion=layout.base_render / "render_plan.json",
        resume=args.resume,
    )
    render_base(
        root=root,
        layout=layout,
        sweep_physics_manifest=layout.sweep_physics / "manifest.json",
        workers=args.render_workers,
        gpus=args.gpus,
        resume=args.resume,
    )
    if not (layout.source_release / "manifest.json").is_file():
        publish_source_release(
            root=root,
            base_manifest_path=layout.base_manifest,
            sweep_metadata_manifest_path=layout.sweep_metadata / "manifest.json",
            sweep_physics_manifest_path=layout.sweep_physics / "manifest.json",
            output=layout.source_release,
            object_count=OBJECT_COUNT,
            dataset_id="physweep_two_object",
            release_schema="physweep_two_object_source_release_v1",
        )
    elif not args.resume:
        raise FileExistsError(layout.source_release)
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
        ],
        root=root,
        completion=layout.sweep_render / "manifest.json",
        resume=args.resume,
    )
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
