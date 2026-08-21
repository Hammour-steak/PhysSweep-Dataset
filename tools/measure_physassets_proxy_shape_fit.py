#!/usr/bin/env python3
"""Measure local shape fit that global proxy/hull volume ratios cannot detect."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    meshes = [loaded] if isinstance(loaded, trimesh.Trimesh) else [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    mesh = trimesh.util.concatenate(meshes)
    mesh.remove_unreferenced_vertices()
    rotation = trimesh.transformations.rotation_matrix(math.pi / 2.0, [1.0, 0.0, 0.0])
    mesh.apply_transform(rotation)
    bounds = np.asarray(mesh.bounds)
    mesh.vertices = (mesh.vertices - bounds.mean(axis=0)) * (0.2 / float(mesh.extents.max()))
    return mesh


def radial_profile(vertices: np.ndarray, axis: int, bands: int = 12) -> dict:
    axial = vertices[:, axis]
    radial_axes = [index for index in range(3) if index != axis]
    radial_center = (vertices[:, radial_axes].min(axis=0) + vertices[:, radial_axes].max(axis=0)) * 0.5
    radial_extents = np.ptp(vertices[:, radial_axes], axis=0)
    radial_extent_ratio = float(radial_extents.min() / max(1e-9, radial_extents.max()))
    radii = np.linalg.norm(vertices[:, radial_axes] - radial_center, axis=1)
    edges = np.linspace(float(axial.min()), float(axial.max()), bands + 1)
    profile = []
    for index in range(bands):
        upper = axial <= edges[index + 1] if index == bands - 1 else axial < edges[index + 1]
        selected = radii[(axial >= edges[index]) & upper]
        if len(selected) >= 8:
            profile.append(float(np.quantile(selected, 0.95)))
    values = np.asarray(profile, dtype=np.float64)
    if len(values) < max(4, bands // 2) or values.max() <= 1e-9:
        return {"radial_profile_valid": False, "radial_profile": profile, "radial_extent_min_over_max": radial_extent_ratio}
    return {
        "radial_profile_valid": True,
        "radial_profile": profile,
        "radial_profile_min_over_max": float(values.min() / values.max()),
        "radial_profile_cv": float(values.std() / values.mean()),
        "radial_extent_min_over_max": radial_extent_ratio,
    }


def measure(row: dict) -> dict:
    mesh = load_mesh(Path(row["source_glb"]))
    method = str(row["method"])
    result = {"sample_id": str(row["sample_id"]), "method": method}
    if method == "semantic_upright_cylinder":
        result.update(radial_profile(np.asarray(mesh.vertices), 2))
    elif method == "semantic_axis_cylinder":
        result.update(radial_profile(np.asarray(mesh.vertices), int(np.argmax(mesh.extents))))
    elif method == "semantic_sphere":
        center = np.asarray(mesh.bounds).mean(axis=0)
        radii = np.linalg.norm(np.asarray(mesh.vertices) - center, axis=1)
        result.update({
            "sphere_radius_cv": float(radii.std() / radii.mean()),
            "sphere_radius_p10_over_p90": float(np.quantile(radii, 0.1) / np.quantile(radii, 0.9)),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.selection.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(measure, row) for row in rows]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: int(row["sample_id"]))
    args.output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in results), encoding="utf-8")
    print(f"measured {len(results)}")


if __name__ == "__main__":
    main()
