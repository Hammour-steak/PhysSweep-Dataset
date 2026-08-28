#!/usr/bin/env python3
"""Download Objaverse meshes and audit their rigid-body proxy suitability."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import objaverse
import trimesh
from scipy import sparse
from scipy.sparse import csgraph


def load_rows(path: Path, limit: int | None) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def scene_to_mesh(path: Path) -> tuple[trimesh.Trimesh, int]:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return loaded, 1
    meshes = [
        geometry
        for geometry in loaded.dump()
        if isinstance(geometry, trimesh.Trimesh) and len(geometry.faces) > 0
    ]
    if not meshes:
        raise ValueError("GLB contains no triangle meshes")
    return trimesh.util.concatenate(meshes), len(meshes)


def audit_mesh(path: Path) -> dict:
    mesh, geometry_count = scene_to_mesh(path)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    extents = np.asarray(mesh.extents)
    finite = bool(
        np.isfinite(vertices).all()
        and np.isfinite(faces).all()
        and np.isfinite(extents).all()
    )
    positive_extents = extents[extents > 1e-9]
    aspect_ratio = (
        float(positive_extents.max() / positive_extents.min())
        if len(positive_extents) == 3
        else math.inf
    )
    face_areas = np.asarray(mesh.area_faces)
    degenerate_ratio = (
        float(np.count_nonzero(face_areas <= 1e-14) / len(face_areas))
        if len(face_areas)
        else 1.0
    )
    face_count = len(faces)
    adjacency = np.asarray(mesh.face_adjacency, dtype=np.int64)
    if face_count and len(adjacency):
        rows = np.concatenate((adjacency[:, 0], adjacency[:, 1]))
        columns = np.concatenate((adjacency[:, 1], adjacency[:, 0]))
        graph = sparse.coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
            shape=(face_count, face_count),
        ).tocsr()
        component_count, labels = csgraph.connected_components(
            graph,
            directed=False,
            return_labels=True,
        )
        component_areas = np.bincount(labels, weights=face_areas)
    elif face_count:
        component_count = face_count
        component_areas = face_areas
    else:
        component_count = 0
        component_areas = np.asarray([], dtype=np.float64)
    total_component_area = float(component_areas.sum())
    largest_component_fraction = (
        float(component_areas.max() / total_component_area)
        if total_component_area > 0 and len(component_areas)
        else 0.0
    )

    fatal_reasons: list[str] = []
    warnings: list[str] = []
    if not finite:
        fatal_reasons.append("non_finite_geometry")
    if len(vertices) < 4 or len(faces) < 4:
        fatal_reasons.append("insufficient_geometry")
    if len(positive_extents) != 3:
        fatal_reasons.append("zero_thickness_axis")
    if degenerate_ratio > 0.02:
        fatal_reasons.append("excessive_degenerate_faces")
    elif degenerate_ratio > 0.001:
        warnings.append("some_degenerate_faces")
    complex_proxy = False
    if aspect_ratio > 100:
        warnings.append("extreme_aspect_ratio")
        complex_proxy = True
    if component_count > 256 and largest_component_fraction < 0.5:
        warnings.append("many_significant_components")
        complex_proxy = True
    if not mesh.is_watertight:
        warnings.append("not_watertight")
    if not mesh.is_winding_consistent:
        warnings.append("inconsistent_winding")

    if fatal_reasons:
        proxy_status = "mesh_reject"
    elif complex_proxy:
        proxy_status = "complex_proxy_review"
    else:
        proxy_status = "proxy_candidate"
    return {
        "mesh_path": str(path),
        "file_size_bytes": path.stat().st_size,
        "geometry_count": geometry_count,
        "vertex_count": int(len(vertices)),
        "face_count": int(len(faces)),
        "component_count": int(component_count),
        "largest_component_fraction": largest_component_fraction,
        "extents_source_units": extents.tolist(),
        "aspect_ratio": aspect_ratio,
        "degenerate_face_ratio": degenerate_ratio,
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "proxy_status": proxy_status,
        "requires_mesh_repair": bool(
            not mesh.is_watertight
            or not mesh.is_winding_consistent
            or degenerate_ratio > 0
        ),
        "fatal_reasons": fatal_reasons,
        "warnings": warnings,
    }


def download_one(
    uid: str,
    object_path: str | None,
    versioned_root: Path,
    timeout: int,
) -> tuple[str, str | None, str | None]:
    if not object_path:
        return uid, None, "uid_not_found"
    target = versioned_root / object_path
    if target.exists() and target.stat().st_size > 0:
        return uid, str(target), None
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    url = (
        "https://huggingface.co/datasets/allenai/objaverse/resolve/main/"
        + object_path
    )
    last_error = ""
    for attempt in range(2):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "PhysSweep-asset-audit/1.0"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                with temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            os.replace(temporary, target)
            return uid, str(target), None
        except (OSError, urllib.error.URLError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            temporary.unlink(missing_ok=True)
            time.sleep(2**attempt)
    return uid, None, last_error


def download_batch(
    uids: list[str],
    object_paths: dict[str, str],
    versioned_root: Path,
    workers: int,
    timeout: int,
) -> tuple[dict[str, str], dict[str, str]]:
    paths: dict[str, str] = {}
    errors: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_one,
                uid,
                object_paths.get(uid),
                versioned_root,
                timeout,
            ): uid
            for uid in uids
        }
        for future in concurrent.futures.as_completed(futures):
            uid, path, error = future.result()
            if path:
                paths[uid] = path
            else:
                errors[uid] = error or "download_failed"
    return paths, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--download-processes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--download-timeout", type=int, default=45)
    args = parser.parse_args()

    rows = load_rows(args.input, args.limit)
    completed: set[str] = set()
    if args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            try:
                completed.add(str(json.loads(line)["sample_id"]))
            except (KeyError, json.JSONDecodeError):
                continue
    rows = [row for row in rows if str(row["sample_id"]) not in completed]
    args.download_dir.mkdir(parents=True, exist_ok=True)
    objaverse.BASE_PATH = str(args.download_dir)
    objaverse._VERSIONED_PATH = str(args.download_dir / "hf-objaverse-v1")
    object_paths = objaverse._load_object_paths()
    versioned_root = Path(objaverse._VERSIONED_PATH)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with args.output.open("a", encoding="utf-8") as handle:
        for batch_start in range(0, len(rows), args.batch_size):
            batch = rows[batch_start : batch_start + args.batch_size]
            uids = [str(row["objaverse_uid"]) for row in batch]
            paths, download_errors = download_batch(
                uids,
                object_paths,
                versioned_root,
                args.download_processes,
                args.download_timeout,
            )
            for row in batch:
                uid = str(row["objaverse_uid"])
                mesh_path = paths.get(uid)
                result = {
                    "sample_id": row["sample_id"],
                    "objaverse_uid": uid,
                    "object_name": row.get("object_name", ""),
                    "material": row.get("material", ""),
                }
                if mesh_path is None:
                    result.update(
                        {
                            "proxy_status": "download_missing",
                            "fatal_reasons": [
                                download_errors.get(uid, "download_failed")
                            ],
                            "warnings": [],
                        }
                    )
                else:
                    try:
                        result.update(audit_mesh(Path(mesh_path)))
                    except Exception as exc:
                        result.update(
                            {
                                "mesh_path": str(mesh_path),
                                "proxy_status": "audit_error",
                                "fatal_reasons": [
                                    f"{type(exc).__name__}: {str(exc)[:300]}"
                                ],
                                "warnings": [],
                            }
                        )
                handle.write(json.dumps(result, ensure_ascii=True) + "\n")
                handle.flush()
                processed += 1
                print(
                    f"completed={processed}/{len(rows)} "
                    f"sample={result['sample_id']} "
                    f"status={result['proxy_status']}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
