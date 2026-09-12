"""Admit two-object bases and preserve complete counterfactual sweep groups."""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from tools.cli.dataset_generation import Layout, run, run_once
from tools.core.hashing import sha256_file
from tools.core.json_io import read_json, write_json_atomic_sorted
from tools.core.paths import resolve_project_path_within_root
from tools.release.sweep_validation import validate_groups
from tools.sampling.derive_physics_sweep import load_sweep_config


PHYSICS_SWEEP_CONFIG = Path("configs/two_object_physics_sweep.json")
SWEEP_TARGET_OBJECT_INDICES = (0,)
ADMISSION_FAILURE_SCHEMA = "physweep_two_object_admission_failure_manifest_v1"


@dataclass(frozen=True)
class AdmittedManifests:
    base: Path
    base_physics: Path
    sweep_metadata: Path
    sweep_physics: Path
    base_bound: Path


def _validate_sweep_metadata(root: Path, path: Path) -> None:
    config_path = root / PHYSICS_SWEEP_CONFIG
    load_sweep_config(config_path)  # Also verify the pinned shared configuration.
    manifest = read_json(path)
    if manifest.get("config") != {
        "path": str(PHYSICS_SWEEP_CONFIG), "sha256": sha256_file(config_path),
    }:
        raise ValueError("admitted sweep configuration differs; use a new work id")
    validate_groups(manifest["records"], expected_target_indices=SWEEP_TARGET_OBJECT_INDICES)


def _physics_rejections(
    root: Path,
    manifest_path: Path,
    generic_scene_ids: set[str],
    *,
    sweep: bool,
) -> list[dict[str, str]]:
    """Replace rejected bases; stop on sweep integrity errors without resampling."""

    manifest = read_json(manifest_path)
    records = manifest.get("records")
    if (
        manifest.get("schema_version") != "physweep_pybullet_batch_record_v1"
        or not isinstance(records, list)
        or int(manifest.get("sample_count", -1)) != len(records)
    ):
        raise ValueError(f"invalid two-object physics manifest: {manifest_path}")
    scene_ids = [str(record.get("scene_id", "")) for record in records]
    if any(not scene_id for scene_id in scene_ids) or len(set(scene_ids)) != len(
        scene_ids
    ):
        raise ValueError("two-object physics manifest has invalid scene ids")
    passed = [
        record
        for record in records
        if record.get("ok") and record.get("audit_passed")
    ]
    errors = [record for record in records if not record.get("ok")]
    rejected = [
        record
        for record in records
        if record.get("ok") and not record.get("audit_passed")
    ]
    if (
        int(manifest.get("passed_count", -1)) != len(passed)
        or int(manifest.get("rejected_count", -1)) != len(rejected)
        or int(manifest.get("error_count", -1)) != len(errors)
        or len(passed) + len(rejected) + len(errors) != len(records)
    ):
        raise ValueError("two-object physics manifest counts differ")
    if errors:
        summary = [
            f"{record.get('scene_id')}: {record.get('error', 'unknown error')}"
            for record in errors[:3]
        ]
        raise RuntimeError(f"two-object simulation errors are not replaceable: {summary}")

    if sweep and rejected:
        summary = [str(record["scene_id"]) for record in rejected[:3]]
        raise RuntimeError(
            "two-object sweep integrity failures require investigation; "
            f"base replacement is forbidden: {summary}"
        )

    failures: dict[str, list[str]] = defaultdict(list)
    for record in rejected:
        scene_id = str(record.get("scene_id", ""))
        if scene_id not in generic_scene_ids:
            raise RuntimeError(
                f"fixed specialized scene failed admission and cannot be replaced: {scene_id}"
            )
        failed_checks = [str(value) for value in record.get("failed_checks", [])]
        failures[scene_id].extend(failed_checks or ["trajectory_audit"])
    return [
        {
            "scene_id": scene_id,
            "error": ", ".join(sorted(set(checks))),
        }
        for scene_id, checks in sorted(failures.items())
    ]


def _write_admission_failures(
    path: Path, *, stage: str, records: list[dict[str, str]]
) -> None:
    if not records:
        raise ValueError("an admission failure manifest cannot be empty")
    write_json_atomic_sorted(
        path,
        {
            "schema_version": ADMISSION_FAILURE_SCHEMA,
            "stage": stage,
            "failed_count": len(records),
            "failures": records,
        },
    )


def _freeze_manifest(source: Path, destination: Path, *, resume: bool) -> None:
    document = read_json(source)
    if destination.is_file():
        if not resume:
            raise FileExistsError(destination)
        if read_json(destination) != document:
            raise ValueError(f"admitted manifest changed: {destination}")
        return
    write_json_atomic_sorted(destination, document)


def _generic_scene_ids(base_manifest: Path) -> set[str]:
    base = read_json(base_manifest)
    records = base.get("records")
    if (
        base.get("schema_version") != "physweep_two_object_base_manifest_v1"
        or not isinstance(records, list)
        or int(base.get("sample_count", -1)) != len(records)
    ):
        raise ValueError("invalid assembled two-object base manifest")
    return {
        str(record["scene_id"])
        for record in records
        if str(record.get("family")) == "generic"
    }


def _replace_failed_generic(
    *,
    root: Path,
    generic_manifest: Path,
    failure_manifest: Path,
    output_root: Path,
    resume: bool,
) -> Path:
    output_manifest = output_root / "manifest.json"
    run_once(
        [
            sys.executable,
            "-m",
            "tools.sampling.resample_two_object_failures",
            "--root",
            str(root),
            "--base-manifest",
            str(generic_manifest),
            "--failure-manifest",
            str(failure_manifest),
            "--output-dir",
            str(output_root),
        ],
        completion=output_manifest,
        resume=resume,
    )
    return output_manifest


def _run_physics_batch(
    *,
    root: Path,
    source_manifest: Path,
    output_root: Path,
    workers: int,
    resume: bool,
) -> Path:
    command = [
        sys.executable,
        "-m",
        "tools.physics.run_pybullet_batch",
        "--root",
        str(root),
        "--manifest",
        str(source_manifest),
        "--output-root",
        str(output_root),
        "--workers",
        str(workers),
        "--allow-audit-rejections",
    ]
    if resume:
        command.append("--allow-existing-output")
    result = output_root / "manifest.json"
    run_once(command, completion=result, resume=resume)
    return result


def _validate_base_binding(
    root: Path, bound_path: Path, physics_path: Path, scene_ids: set[str]
) -> None:
    """Reuse only cameras bound to the admitted base metadata and trajectories."""
    bound = read_json(bound_path)
    samples = bound.get("samples", [])
    selected = [str(sample.get("scene_id", "")) for sample in samples]
    if (
        bound.get("schema_version") != "physweep_pybullet_bound_manifest_v2"
        or int(bound.get("sample_count", -1)) != len(samples)
        or len(selected) != len(set(selected))
        or set(selected) != scene_ids
    ):
        raise ValueError("base camera binding selects different scenes")
    rules = bound["camera_rules"]
    if sha256_file(resolve_project_path_within_root(root, rules["path"])) != rules["sha256"]:
        raise ValueError("base camera rules changed; use a new work id")
    physics = {
        str(record["scene_id"]): record for record in read_json(physics_path)["records"]
    }
    for sample in samples:
        scene_id = str(sample["scene_id"])
        path = resolve_project_path_within_root(root, sample["metadata_path"])
        if sha256_file(path) != str(sample["metadata_sha256"]):
            raise ValueError(f"base camera metadata changed: {scene_id}")
        metadata = read_json(path)
        if (
            metadata.get("scene_id") != scene_id
            or metadata.get("schema_version") != "physweep_pybullet_rigid_bound_metadata_v1"
        ):
            raise ValueError(f"base camera metadata identity differs: {scene_id}")
        for field, prefix in (("source_metadata", "metadata"), ("trajectory", "trajectory")):
            binding = metadata[field]
            source = resolve_project_path_within_root(root, binding["path"])
            record = physics[scene_id]
            if (
                source != resolve_project_path_within_root(root, record[f"{prefix}_path"])
                or binding["sha256"] != record[f"{prefix}_sha256"]
                or sha256_file(source) != binding["sha256"]
            ):
                raise ValueError(f"base camera {field} binding differs: {scene_id}")


def _camera_failure_manifest(
    *,
    root: Path,
    render_root: Path,
    generic_manifest: Path,
    base_physics: Path,
    workers: int,
) -> Path | None:
    bound = render_root / "generic" / "bound_manifest.json"
    failure = render_root / "generic" / "failure_manifest.json"
    if bound.is_file():
        return None
    if failure.is_file():
        return failure
    source = read_json(generic_manifest)
    generic_ids = {str(sample["scene_id"]) for sample in source["samples"]}
    physics = read_json(base_physics)
    records = [record for record in physics["records"] if str(record["scene_id"]) in generic_ids]
    physics_manifest = render_root / "generic" / "physics_manifest.json"
    write_json_atomic_sorted(physics_manifest, {
        **physics,
        "source_manifest": str(generic_manifest.resolve()),
        "sample_count": len(records),
        "passed_count": sum(bool(record.get("ok") and record.get("audit_passed")) for record in records),
        "rejected_count": sum(bool(record.get("ok") and not record.get("audit_passed")) for record in records),
        "error_count": sum(not record.get("ok") for record in records),
        "source_schema_counts": {"physweep_pybullet_rigid_metadata_v1": len(records)},
        "records": records,
    })
    command = [
        sys.executable,
        "-m",
        "tools.rendering.bind_pybullet_visuals",
        "--root",
        str(root),
        "--manifest",
        str(physics_manifest),
        "--output-root",
        str(render_root / "generic"),
        "--workers",
        str(workers),
    ]
    completed = run(command, check=False)
    if completed.returncode == 0 and bound.is_file():
        return None
    if failure.is_file():
        return failure
    if completed.returncode == 0:
        raise RuntimeError("camera binding produced neither a bound nor a failure manifest")
    raise subprocess.CalledProcessError(completed.returncode, command)


def admit_two_object_groups(
    *,
    root: Path,
    layout: Layout,
    initial_generic_manifest: Path,
    specialized_manifest: Path,
    physics_workers: int,
    camera_workers: int,
    max_attempts: int,
    resume: bool,
) -> AdmittedManifests:
    """Admit 13-member groups targeting object_a, with both objects simulated."""

    stable = AdmittedManifests(
        base=layout.base_manifest,
        base_physics=layout.base_dataset / "physics" / "manifest.json",
        sweep_metadata=layout.sweep_metadata / "manifest.json",
        sweep_physics=layout.sweep_physics / "manifest.json",
        base_bound=layout.base_render / "generic" / "bound_manifest.json",
    )
    if all(path.is_file() for path in asdict(stable).values()):
        if not resume:
            raise FileExistsError(stable.base)
        _validate_sweep_metadata(root, stable.sweep_metadata)
        generic_ids = _generic_scene_ids(stable.base)
        if _physics_rejections(
            root, stable.base_physics, generic_ids, sweep=False
        ) or _physics_rejections(
            root, stable.sweep_physics, generic_ids, sweep=True
        ):
            raise ValueError("frozen admitted manifests contain rejections")
        _validate_base_binding(root, stable.base_bound, stable.base_physics, generic_ids)
        return stable

    admission_root = layout.base_dataset.parent / "admission"
    camera_root = layout.base_render.parent / "admission"
    current_generic = initial_generic_manifest
    for attempt in range(max_attempts):
        attempt_name = f"attempt_{attempt:02d}"
        attempt_root = admission_root / attempt_name

        def next_generic(failure_manifest: Path) -> Path:
            if attempt + 1 >= max_attempts:
                raise RuntimeError(
                    "two-object admission exhausted its replacement attempts"
                )
            return _replace_failed_generic(
                root=root,
                generic_manifest=current_generic,
                failure_manifest=failure_manifest,
                output_root=(
                    admission_root
                    / f"attempt_{attempt + 1:02d}"
                    / "generic"
                ),
                resume=resume,
            )

        base_manifest = attempt_root / "base" / "manifest.json"
        run_once(
            [
                sys.executable,
                "-m",
                "tools.sampling.assemble_two_object_base",
                "--root",
                str(root),
                "--generic-manifest",
                str(current_generic),
                "--specialized-manifest",
                str(specialized_manifest),
                "--output",
                str(base_manifest),
            ],
            completion=base_manifest,
            resume=resume,
        )
        generic_ids = _generic_scene_ids(base_manifest)
        base_physics = _run_physics_batch(
            root=root,
            source_manifest=base_manifest,
            output_root=attempt_root / "base" / "physics",
            workers=physics_workers,
            resume=resume,
        )
        failures = _physics_rejections(
            root, base_physics, generic_ids, sweep=False
        )
        if failures:
            failure_path = attempt_root / "base_failures.json"
            _write_admission_failures(
                failure_path, stage="base_physics", records=failures
            )
            current_generic = next_generic(failure_path)
            continue

        attempt_render = camera_root / attempt_name
        camera_failure = _camera_failure_manifest(
            root=root,
            render_root=attempt_render,
            generic_manifest=current_generic,
            base_physics=base_physics,
            workers=min(camera_workers, len(generic_ids)),
        )
        if camera_failure is not None:
            current_generic = next_generic(camera_failure)
            continue
        base_bound = attempt_render / "generic" / "bound_manifest.json"
        _validate_base_binding(root, base_bound, base_physics, generic_ids)

        sweep_metadata_root = attempt_root / "sweep" / "metadata"
        sweep_metadata = sweep_metadata_root / "manifest.json"
        run_once(
            [
                sys.executable,
                "-m",
                "tools.sampling.derive_physics_sweep",
                "--root",
                str(root),
                "--base-manifest",
                str(base_manifest),
                "--config",
                str(root / PHYSICS_SWEEP_CONFIG),
                "--target-object-index",
                str(SWEEP_TARGET_OBJECT_INDICES[0]),
                "--output-dir",
                str(sweep_metadata_root),
            ],
            completion=sweep_metadata,
            resume=resume,
        )
        _validate_sweep_metadata(root, sweep_metadata)
        sweep_physics = _run_physics_batch(
            root=root,
            source_manifest=sweep_metadata,
            output_root=attempt_root / "sweep" / "physics",
            workers=physics_workers,
            resume=resume,
        )
        _physics_rejections(
            root, sweep_physics, generic_ids, sweep=True
        )
        for source, destination in (
            (base_manifest, stable.base),
            (base_physics, stable.base_physics),
            (sweep_metadata, stable.sweep_metadata),
            (sweep_physics, stable.sweep_physics),
            (base_bound, stable.base_bound),
        ):
            _freeze_manifest(source, destination, resume=resume)
        return stable
    raise RuntimeError(
        f"two-object admission did not converge after {max_attempts} attempts"
    )
