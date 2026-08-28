#!/usr/bin/env python3
"""Generate, render, audit, and publish a fresh one-object PhysSweep dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tools.cli.build_one_object_dataset import publish_dataset
from tools.core.json_io import read_json as load_json
from tools.core.json_io import write_json_atomic_sorted as write_json
from tools.core.paths import safe_scene_id
from tools.core.sweep_values import SWEEP_VARIANTS_PER_TARGET, sweep_group_size
from tools.physics.specialized_backend_registry import load_specialized_backends
from tools.release.base_release_view import PipelineSpec
from tools.release.one_object_source_release import publish_source_release


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERIC_SCHEMA = "physweep_pybullet_rigid_metadata_v1"
PLAN_SCHEMA = "physweep_one_object_generation_plan_v1"


@dataclass(frozen=True)
class Layout:
    base_dataset: Path
    base_manifest: Path
    sweep_metadata: Path
    sweep_physics: Path
    source_release: Path
    base_render: Path
    sweep_render: Path
    canonical_release: Path


def generation_layout(root: Path, work_id: str, release_root: Path) -> Layout:
    work_id = safe_scene_id(work_id)
    root = root.resolve()
    canonical = release_root if release_root.is_absolute() else root / release_root
    canonical = canonical.resolve()
    if canonical.name != "one_object" or (root / "outputs").resolve() not in canonical.parents:
        raise ValueError("canonical release must be an outputs/.../one_object directory")
    base_dataset = root / "datasets" / work_id / "base"
    return Layout(
        base_dataset=base_dataset,
        base_manifest=base_dataset / "manifest.json",
        sweep_metadata=root / "datasets" / work_id / "sweep" / "metadata",
        sweep_physics=root / "datasets" / work_id / "sweep" / "physics",
        source_release=root / "datasets" / work_id / "release",
        base_render=root / "outputs" / work_id / "base",
        sweep_render=root / "outputs" / work_id / "sweep",
        canonical_release=canonical,
    )


def generation_plan(
    root: Path,
    work_id: str,
    release_root: Path,
    *,
    count: int,
    seed: int,
) -> dict[str, Any]:
    layout = generation_layout(root, work_id, release_root)
    backends = load_specialized_backends(root)
    return {
        "schema_version": PLAN_SCHEMA,
        "work_id": work_id,
        "request": {"base_count": count, "seed": seed},
        "layout": {
            key: value.resolve().relative_to(root.resolve()).as_posix()
            for key, value in asdict(layout).items()
        },
        "stages": [
            "sample_and_audit_base",
            "stage_and_render_base",
            "derive_and_audit_sweep",
            "publish_source_release",
            "stage_and_render_sweep",
            "materialize_canonical_base_and_sweep",
            "verify_canonical_release",
        ],
        "pipelines": [
            {
                "pipeline": "generic_pybullet",
                "family": "generic",
                "source_schema_version": GENERIC_SCHEMA,
                "renderer_id": "generic",
            },
            *[
                {
                    "pipeline": record["pipeline"],
                    "family": record["sweep_branch"],
                    "source_schema_version": record["source_schema_version"],
                    "renderer_id": record["renderer_id"],
                }
                for record in backends
            ],
        ],
    }


def bind_generation_plan(path: Path, plan: dict[str, Any], resume: bool) -> None:
    if path.is_file():
        if not resume:
            raise FileExistsError(f"generation plan already exists: {path}")
        if load_json(path) != plan:
            raise ValueError("resume request differs from the frozen generation plan")
        return
    write_json(path, plan)


def run(command: list[str], root: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def run_once(
    command: list[str],
    *,
    root: Path,
    completion: Path,
    resume: bool,
) -> None:
    if completion.is_file():
        if not resume:
            raise FileExistsError(f"stage already exists; pass --resume: {completion}")
        return
    run(command, root)
    if not completion.is_file():
        raise RuntimeError(f"stage did not create its completion artifact: {completion}")


def verify_render_manifest(path: Path, expected_count: int) -> None:
    manifest = load_json(path)
    if (
        int(manifest.get("sample_count", -1)) != expected_count
        or int(manifest.get("failure_count", -1)) != 0
        or int(manifest.get("success_count", -1)) != expected_count
    ):
        raise ValueError(f"render manifest is incomplete: {path}")


def render_base(
    *,
    root: Path,
    layout: Layout,
    workers: int,
    gpus: str,
    resume: bool,
) -> None:
    plan = load_json(layout.base_render / "render_plan.json")
    counts = {str(key): int(value) for key, value in plan["pipeline_counts"].items()}
    if counts.get("generic_pybullet", 0):
        bound = Path(plan["generic_bound_manifest"])
        run_once(
            [
                sys.executable,
                "-m",
                "tools.rendering.bind_pybullet_visuals",
                "--root",
                str(root),
                "--manifest",
                str(plan["generic_source_manifest"]),
                "--output-root",
                str(layout.base_render / "generic"),
                "--workers",
                str(min(workers, counts["generic_pybullet"])),
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
            Path(plan["generic_render_manifest"]), counts["generic_pybullet"]
        )

    for backend in load_specialized_backends(root):
        pipeline = str(backend["pipeline"])
        count = counts.get(pipeline, 0)
        if not count:
            continue
        branch = str(backend["sweep_branch"])
        command = [
            sys.executable,
            "-m",
            "tools.rendering.render_asset_proxy_manifest",
            "--root",
            str(root),
            "--renderer",
            str(backend["renderer_id"]),
            "--manifest",
            str(plan[f"{branch}_render_input_manifest"]),
            "--workers",
            str(workers),
            "--gpus",
            gpus,
        ]
        if resume:
            command.append("--resume")
        run(command, root)
        verify_render_manifest(Path(plan[f"{branch}_render_manifest"]), count)


def render_sweep(
    *,
    root: Path,
    layout: Layout,
    workers: int,
    gpus: str,
    resume: bool,
) -> None:
    plan = load_json(layout.sweep_render / "manifest.json")
    counts = {str(key): int(value) for key, value in plan["branch_counts"].items()}
    if counts.get("asset", 0):
        base_plan = load_json(layout.base_render / "render_plan.json")
        run(
            [
                sys.executable,
                "-m",
                "tools.rendering.freeze_asset_sweep_cameras",
                "--root",
                str(root),
                "--manifest",
                str(layout.sweep_render / "asset" / "render_input_manifest.json"),
                "--base-manifest",
                str(base_plan["asset_render_input_manifest"]),
            ],
            root,
        )
    if counts.get("generic", 0):
        if counts["generic"] % sweep_group_size(1):
            raise ValueError("generic sweep branch is not composed of complete groups")
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
        expected = (
            counts["generic"] // sweep_group_size(1) * SWEEP_VARIANTS_PER_TARGET
        )
        base_expected = counts["generic"] // sweep_group_size(1)
        base_result = bound_root / "base_render_manifest.json"
        base_command = [
            sys.executable,
            "-m",
            "tools.rendering.render_pybullet_manifest",
            "--root",
            str(root),
            "--manifest",
            str(bound),
            "--sweep-kind",
            "base",
            "--result-manifest",
            str(base_result),
            "--workers",
            str(workers),
            "--gpus",
            gpus,
        ]
        if resume:
            base_command.append("--resume")
        run(base_command, root)
        verify_render_manifest(base_result, base_expected)
        result = bound_root / "derived_render_manifest.json"
        command = [
            sys.executable,
            "-m",
            "tools.rendering.render_pybullet_manifest",
            "--root",
            str(root),
            "--manifest",
            str(bound),
            "--sweep-kind",
            "sweep",
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

    for backend in load_specialized_backends(root):
        branch = str(backend["sweep_branch"])
        count = counts.get(branch, 0)
        if not count:
            continue
        if count % sweep_group_size(1):
            raise ValueError(f"{branch} sweep branch is not composed of complete groups")
        expected = count // sweep_group_size(1) * SWEEP_VARIANTS_PER_TARGET
        base_expected = count // sweep_group_size(1)
        base_result = layout.sweep_render / branch / "base_render_manifest.json"
        base_command = [
            sys.executable,
            "-m",
            "tools.rendering.render_asset_proxy_manifest",
            "--root",
            str(root),
            "--renderer",
            str(backend["renderer_id"]),
            "--manifest",
            str(layout.sweep_render / branch / "base_render_input_manifest.json"),
            "--result-manifest",
            str(base_result),
            "--workers",
            str(workers),
            "--gpus",
            gpus,
        ]
        if resume:
            base_command.append("--resume")
        run(base_command, root)
        verify_render_manifest(base_result, base_expected)
        result = layout.sweep_render / branch / "derived_render_manifest.json"
        command = [
            sys.executable,
            "-m",
            "tools.rendering.render_asset_proxy_manifest",
            "--root",
            str(root),
            "--renderer",
            str(backend["renderer_id"]),
            "--manifest",
            str(layout.sweep_render / branch / "derived_render_input_manifest.json"),
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


def release_specs(
    root: Path, layout: Layout
) -> list[PipelineSpec]:
    release = load_json(layout.source_release / "manifest.json")
    metadata = load_json(root / str(release["metadata_manifest"]))
    selected = {str(record["source_schema_version"]) for record in metadata["records"]}
    branches = {
        GENERIC_SCHEMA: "generic",
        **{
            str(record["source_schema_version"]): str(record["sweep_branch"])
            for record in load_specialized_backends(root)
        },
    }
    missing = selected - set(branches)
    if missing:
        raise ValueError(f"release contains unregistered source schemas: {sorted(missing)}")
    specs = []
    for schema in sorted(selected):
        branch = branches[schema]
        sweep_root = (
            layout.sweep_render / branch / "bound"
            if schema == GENERIC_SCHEMA
            else layout.sweep_render / branch
        )
        specs.append(PipelineSpec(branch, schema, root, sweep_root))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--work-id", required=True)
    parser.add_argument("--count", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--physics-workers", type=int, default=24)
    parser.add_argument("--render-workers", type=int, default=64)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--release-root", type=Path, default=Path("outputs/one_object"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.count, args.physics_workers, args.render_workers) < 1:
        raise ValueError("counts and worker values must be positive")
    if not [value for value in args.gpus.split(",") if value.strip()]:
        raise ValueError("--gpus must contain at least one id")
    root = args.root.resolve()
    plan = generation_plan(
        root,
        args.work_id,
        args.release_root,
        count=args.count,
        seed=args.seed,
    )
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    layout = generation_layout(root, args.work_id, args.release_root)
    plan_path = root / "outputs" / safe_scene_id(args.work_id) / "generation_plan.json"
    bind_generation_plan(plan_path, plan, args.resume)

    run_once(
        [
            sys.executable,
            "-m",
            "tools.sampling.sample_one_object_scene_matrix",
            "--root",
            str(root),
            "--output-dataset",
            f"{args.work_id}/base",
            "--count",
            str(args.count),
            "--seed",
            str(args.seed),
            "--physics-workers",
            str(args.physics_workers),
        ],
        root=root,
        completion=layout.base_manifest,
        resume=args.resume,
    )
    base = load_json(layout.base_manifest)
    if (
        base.get("schema_version") != "physweep_one_object_decoupled_manifest_v5"
        or int(base.get("sample_count", -1)) != args.count
        or int(base.get("seed", -1)) != args.seed
    ):
        raise ValueError("generated base identity differs from the request")
    run_once(
        [
            sys.executable,
            "-m",
            "tools.rendering.prepare_formal_render_manifests",
            "--root",
            str(root),
            "--manifest",
            str(layout.base_manifest),
            "--output-root",
            str(layout.base_render),
            "--selection",
            "all",
        ],
        root=root,
        completion=layout.base_render / "render_plan.json",
        resume=args.resume,
    )
    render_base(
        root=root,
        layout=layout,
        workers=args.render_workers,
        gpus=args.gpus,
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
    if not (layout.source_release / "manifest.json").is_file():
        publish_source_release(
            root=root,
            base_manifest_path=layout.base_manifest,
            sweep_metadata_manifest_path=layout.sweep_metadata / "manifest.json",
            sweep_physics_manifest_path=layout.sweep_physics / "manifest.json",
            output=layout.source_release,
        )
    elif not args.resume:
        raise FileExistsError(
            f"stage already exists; pass --resume: {layout.source_release}"
        )
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
            str(layout.base_render / "staged_manifest.json"),
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
    specs = release_specs(root, layout)
    result = publish_dataset(
        release_project_root=root,
        release_manifest=layout.source_release / "manifest.json",
        release_root=layout.canonical_release,
        pipeline_specs=specs,
        workers=args.render_workers,
        resume=args.resume,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
