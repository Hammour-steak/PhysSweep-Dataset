#!/usr/bin/env python3
"""Build exact fixed-size GT scene conditions for the formal training set."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
from tools.core.hashing import sha256_file as _sha256
from tools.core.paths import project_relative_path
from tools.training_export.gt_scene_input import (
    DEFAULT_ENVIRONMENT_POINTS,
    DEFAULT_OBJECT_POINTS,
    inspect_gt_surface,
    inspect_model_scene_condition,
)
from tools.dataset_contract.schema import iter_jsonl


DEFAULT_BLENDER = Path("runtime/blender-3.4.0-linux-x64/blender")
DEFAULT_EXPORTER = Path("tools/training_export/export_gt_initial_surface.py")


def _base_records(path: Path) -> list[dict]:
    records = [
        record
        for record in iter_jsonl(path)
        if record["sweep"]["mode"] == "base"
    ]
    by_scene = {record["base_scene_id"]: record for record in records}
    if len(by_scene) != len(records):
        raise ValueError("training manifest contains duplicate base records")
    return [by_scene[key] for key in sorted(by_scene)]


def _base_records_from_bound_manifest(path: Path) -> list[dict]:
    """Adapt a rendered base manifest without requiring a compiled dataset."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for sample in manifest["samples"]:
        scene_id = str(sample["scene_id"])
        metadata = Path(str(sample["metadata_path"]))
        if metadata.name != f"{scene_id}.json" or metadata.parent.name != "metadata":
            raise ValueError(f"unexpected bound metadata path: {metadata}")
        frame = metadata.parent.parent / "frames" / scene_id / "frame_0001.png"
        records.append(
            {
                "base_scene_id": scene_id,
                "provenance": {"base_bound_metadata": metadata.as_posix()},
                "conditioning": {"first_frame": frame.as_posix()},
            }
        )
    if len({record["base_scene_id"] for record in records}) != len(records):
        raise ValueError("bound manifest contains duplicate base scenes")
    return sorted(records, key=lambda record: record["base_scene_id"])


def _outputs(output: Path, scene_id: str) -> dict[str, Path]:
    return {
        "source": output / "source" / f"{scene_id}.npz",
        "model": output / "model" / f"{scene_id}.npz",
        "report": output / "reports" / f"{scene_id}.json",
        "log": output / "logs" / f"{scene_id}.log",
    }


def _inspect_complete(
    paths: dict[str, Path], scene_id: str, strict: bool = False
) -> dict | None:
    if not all(paths[name].is_file() for name in ("source", "model", "report")):
        if strict:
            missing = [
                name
                for name in ("source", "model", "report")
                if not paths[name].is_file()
            ]
            raise RuntimeError(f"missing GT scene outputs: {missing}")
        return None
    try:
        source_inspection = inspect_gt_surface(paths["source"])
        inspection = inspect_model_scene_condition(paths["model"])
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        if strict:
            raise RuntimeError(f"GT scene inspection failed: {error}") from error
        return None
    if (
        source_inspection["scene_id"] != scene_id
        or inspection["scene_id"] != scene_id
        or report.get("scene_id") != scene_id
    ):
        if strict:
            raise RuntimeError(
                "GT scene output ids do not match the requested base scene"
            )
        return None
    if inspection["object_point_count"] != DEFAULT_OBJECT_POINTS:
        if strict:
            raise RuntimeError("GT scene object point quota mismatch")
        return None
    if inspection["environment_point_count"] != DEFAULT_ENVIRONMENT_POINTS:
        if strict:
            raise RuntimeError("GT scene environment point quota mismatch")
        return None
    return {
        **inspection,
        "source_point_count": int(source_inspection["point_count"]),
    }


def _build_one(
    record: dict,
    root: Path,
    output: Path,
    blender: Path,
    exporter: Path,
    seed: int,
    overwrite: bool,
) -> dict:
    scene_id = str(record["base_scene_id"])
    paths = _outputs(output, scene_id)
    if not overwrite:
        inspection = _inspect_complete(paths, scene_id)
        if inspection is not None:
            return {"scene_id": scene_id, "status": "reused", **inspection}
    for name in ("source", "model", "report"):
        paths[name].unlink(missing_ok=True)
    metadata = (root / record["provenance"]["base_bound_metadata"]).resolve()
    first_frame = (root / record["conditioning"]["first_frame"]).resolve()
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    if not first_frame.is_file():
        raise FileNotFoundError(first_frame)
    command = [
        str(blender),
        "--background",
        "--python-exit-code",
        "1",
        "--python",
        str(exporter),
        "--",
        "--project-root",
        str(root),
        "--metadata",
        str(metadata),
        "--scene-id",
        scene_id,
        "--first-frame",
        str(first_frame),
        "--source-output",
        str(paths["source"]),
        "--model-output",
        str(paths["model"]),
        "--report-output",
        str(paths["report"]),
        "--model-object-points",
        str(DEFAULT_OBJECT_POINTS),
        "--model-environment-points",
        str(DEFAULT_ENVIRONMENT_POINTS),
        "--seed",
        str(seed),
    ]
    with paths["log"].open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(
            f"GT scene export failed for {scene_id}; see {paths['log']}"
        )
    inspection = _inspect_complete(paths, scene_id, strict=True)
    assert inspection is not None
    return {"scene_id": scene_id, "status": "built", **inspection}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-dataset", type=Path)
    source.add_argument(
        "--source-bound-manifest",
        type=Path,
        help="build directly from a rendered base manifest before dataset publication",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--exporter", type=Path, default=DEFAULT_EXPORTER)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit-base-scenes", type=int)
    parser.add_argument("--scene-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    root = args.project_root.resolve()
    output = (root / args.output_root).resolve()
    blender = (root / args.blender).resolve()
    exporter = (root / args.exporter).resolve()
    if not blender.is_file():
        raise FileNotFoundError(blender)
    if not exporter.is_file():
        raise FileNotFoundError(exporter)
    if args.source_bound_manifest is not None:
        bound_manifest = (root / args.source_bound_manifest).resolve()
        records = _base_records_from_bound_manifest(bound_manifest)
        source = {
            "kind": "bound_manifest",
            "path": project_relative_path(root, bound_manifest),
        }
    else:
        source_dataset = (root / args.source_dataset).resolve()
        records = _base_records(source_dataset / "manifest.jsonl")
        source = {
            "kind": "published_dataset",
            "path": project_relative_path(root, source_dataset / "manifest.jsonl"),
        }
    if args.scene_id:
        requested = set(args.scene_id)
        records = [record for record in records if record["base_scene_id"] in requested]
        found = {record["base_scene_id"] for record in records}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(f"unknown base scene ids: {missing}")
    if args.limit_base_scenes is not None:
        records = records[: args.limit_base_scenes]
    if not records:
        raise ValueError("no base scenes selected")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("source", "model", "reports", "logs"):
        (output / name).mkdir(parents=True, exist_ok=True)

    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _build_one,
                record,
                root,
                output,
                blender,
                exporter,
                args.seed,
                args.overwrite,
            ): record["base_scene_id"]
            for record in records
        }
        for completed, future in enumerate(as_completed(futures), 1):
            scene_id = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[{completed}/{len(futures)}] {result['status']} {scene_id}",
                    flush=True,
                )
            except Exception as error:
                failures.append({"scene_id": scene_id, "error": str(error)})
                print(
                    f"[{completed}/{len(futures)}] failed {scene_id}: {error}",
                    flush=True,
                )

    records_out = []
    for result in sorted(results, key=lambda item: item["scene_id"]):
        paths = _outputs(output, result["scene_id"])
        records_out.append(
            {
                **result,
                "source": project_relative_path(root, paths["source"]),
                "source_sha256": _sha256(paths["source"]),
                "model": project_relative_path(root, paths["model"]),
                "model_sha256": _sha256(paths["model"]),
                "report": project_relative_path(root, paths["report"]),
            }
        )
    manifest = {
        "schema": "physweep.gt_training_scene_build.v1",
        "source": source,
        "seed": args.seed,
        "requested_scene_count": len(records),
        "completed_scene_count": len(records_out),
        "failed_scene_count": len(failures),
        "point_contract": {
            "object": DEFAULT_OBJECT_POINTS,
            "environment": DEFAULT_ENVIRONMENT_POINTS,
            "total": DEFAULT_OBJECT_POINTS + DEFAULT_ENVIRONMENT_POINTS,
        },
        "records": records_out,
        "failures": sorted(failures, key=lambda item: item["scene_id"]),
    }
    manifest_path = output / "manifest.json"
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
