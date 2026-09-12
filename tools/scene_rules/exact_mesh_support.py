"""Runtime-backed initial placement checks for exact static support meshes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tools.assets.static_support_proxy import create_pybullet_static_support


def validate_exact_mesh_sphere_placement(
    *, root: Path, support: dict[str, Any], objects: list[dict[str, Any]]
) -> dict[str, Any]:
    binding = support['exact_static_binding']
    plane = float(support['surface_center_z_m'])
    if any(obj['geometry']['type'] != 'sphere' for obj in objects):
        raise ValueError('exact-mesh host placement currently requires three spheres')
    import pybullet as pb

    client = pb.connect(pb.DIRECT)
    if client < 0:
        raise RuntimeError('exact-mesh host placement world could not connect')

    class Client:
        def __getattr__(self, name):
            value = getattr(pb, name)
            if callable(value):
                return lambda *args, **kwargs: value(
                    *args, **kwargs, physicsClientId=client
                )
            return value

    try:
        body = create_pybullet_static_support(Client(), root, binding)
        points = []
        owners = []
        for obj in objects:
            center = np.asarray(obj['initial_state']['contact_point_m'][:2], dtype=float)
            radius = float(obj['geometry']['size_m'][0]) / 2.0
            for dx, dy in (
                (0.0, 0.0),
                (-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5),
                (-0.35, -0.35), (-0.35, 0.35), (0.35, -0.35), (0.35, 0.35),
            ):
                xy = center + radius * np.asarray([dx, dy])
                points.append(xy)
                owners.append(str(obj['object_id']))
        hits = pb.rayTestBatch(
            [[float(x), float(y), plane + 0.15] for x, y in points],
            [[float(x), float(y), plane - 0.03] for x, y in points],
            physicsClientId=client,
        )
        invalid = []
        for owner, point, hit in zip(owners, points, hits):
            passed = bool(
                int(hit[0]) == body
                and abs(float(hit[3][2]) - plane) <= 0.004
                and float(hit[4][2]) >= 0.95
            )
            if not passed:
                invalid.append({
                    'object_id': owner,
                    'xy_m': point.tolist(),
                    'hit_body': int(hit[0]),
                    'hit_z_m': float(hit[3][2]),
                    'normal_z': float(hit[4][2]),
                })
        if invalid:
            raise ValueError(f'exact-mesh initial footprint lacks flat support: {invalid}')
        return {
            'method': 'exact_static_mesh_9_rays_per_sphere_v1',
            'binding_sha256': str(binding['binding_sha256']),
            'object_count': len(objects),
            'ray_count': len(points),
            'valid_ray_count': len(points),
            'support_plane_tolerance_m': 0.004,
            'minimum_up_normal_z': 0.95,
        }
    finally:
        pb.disconnect(client)
