"""Normalize released exact-mesh supports for object-count consumers.

The support collision, visual, semantic identity, and license remain bound to
the released asset lineage. A separate released generic host supplies only the
room/environment context required by the common rigid renderer.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from tools.assets.environment_collision import binding_sha256 as environment_binding_sha256
from tools.assets.static_support_proxy import validate_static_support_binding_files
from tools.core.hashing import sha256_file
from tools.core.json_io import read_json
from tools.sampling.released_object_sources import declared_within


def _registry_record(
    *,
    source_root: Path,
    metadata: dict[str, Any],
    asset_id: str,
    registry_cache: dict[tuple[str, str], dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    reference = metadata.get('registry')
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
        raise ValueError('asset host source lacks a pinned registry')
    path = declared_within(source_root, Path(str(reference['path'])))
    digest = str(reference['sha256'])
    if sha256_file(path) != digest:
        raise ValueError('asset host registry hash mismatch')
    key = (str(path), digest)
    if key not in registry_cache:
        records = read_json(path).get('records')
        if not isinstance(records, list):
            raise ValueError('asset host registry has no records')
        by_id = {str(record['asset_id']): record for record in records}
        if len(by_id) != len(records):
            raise ValueError('asset host registry has duplicate ids')
        registry_cache[key] = by_id
    record = registry_cache[key].get(asset_id)
    if record is None:
        raise ValueError(f'asset host is absent from its registry: {asset_id}')
    return record


def _effective_surface(
    declared: dict[str, Any], override: dict[str, Any] | None
) -> dict[str, Any]:
    result = copy.deepcopy(declared)
    if override is None:
        return result
    if set(override) != {'center_xy_m', 'size_xy_m', 'z_m'}:
        raise ValueError('asset host surface override has an invalid shape')
    center = [float(value) for value in override['center_xy_m']]
    size = [float(value) for value in override['size_xy_m']]
    declared_center = [float(value) for value in declared['center_xy_m']]
    declared_size = [float(value) for value in declared['size_xy_m']]
    if (
        len(center) != 2
        or len(size) != 2
        or min(size) <= 0.0
        or abs(float(override['z_m']) - float(declared['z_m'])) > 1.0e-12
        or any(
            abs(center[index] - declared_center[index]) + size[index] / 2.0
            > declared_size[index] / 2.0 + 1.0e-12
            for index in range(2)
        )
    ):
        raise ValueError('asset host surface override exceeds the released safe surface')
    return {
        'id': str(declared['id']),
        'center_xy_m': center,
        'size_xy_m': size,
        'z_m': float(override['z_m']),
    }


def asset_host_template(
    *,
    source_root: Path,
    runtime_root: Path,
    generation_metadata: dict[str, Any],
    release_metadata: dict[str, Any],
    environment_host_template: dict[str, Any],
    effective_surface_override: dict[str, Any] | None,
    registry_cache: dict[tuple[str, str], dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Return one common rigid host backed by a released exact support mesh."""
    if generation_metadata.get('schema_version') != 'physweep_asset_proxy_scene_v3':
        raise ValueError('asset host requires released asset-proxy metadata v3')
    physics = generation_metadata.get('physics')
    assets = generation_metadata.get('assets')
    if not isinstance(physics, dict) or not isinstance(assets, dict):
        raise ValueError('asset host source lacks physics or asset identity')
    asset_id = str(assets.get('support_asset_id', ''))
    binding = copy.deepcopy(physics.get('static_support_binding'))
    if not asset_id or not isinstance(binding, dict) or binding.get('asset_id') != asset_id:
        raise ValueError('asset host identity differs from its exact binding')
    validate_static_support_binding_files(runtime_root, binding, include_visual=True)
    declared_surface = physics.get('support_surface')
    if declared_surface != binding['target_support_frame']['safe_surface']:
        raise ValueError('asset host support surface differs from its exact binding')
    release_physics = release_metadata.get('physics', {})
    fixture = release_physics.get('fixture')
    lineage = release_metadata.get('lineage', {})
    if (
        not isinstance(fixture, dict)
        or fixture.get('id') != asset_id
        or fixture.get('representation') != 'static_concave_mesh'
        or lineage.get('source_fixture_binding_sha256') != binding['binding_sha256']
    ):
        raise ValueError('released asset host fixture lineage is inconsistent')
    registry = _registry_record(
        source_root=source_root,
        metadata=generation_metadata,
        asset_id=asset_id,
        registry_cache=registry_cache,
    )
    if (
        not bool(registry.get('admission', {}).get('sampling_enabled', False))
        or registry.get('proxy', {}).get('kind') != 'support_compound'
        or registry.get('visual', {}).get('sha256') != binding['visual']['sha256']
    ):
        raise ValueError('asset host registry record is not sampling-enabled or bound')

    backend_reference = physics.get('backend_config')
    if not isinstance(backend_reference, dict) or set(backend_reference) != {'path', 'sha256'}:
        raise ValueError('asset host source lacks a pinned backend')
    backend_path = declared_within(runtime_root, Path(str(backend_reference['path'])))
    if sha256_file(backend_path) != str(backend_reference['sha256']):
        raise ValueError('asset host backend hash mismatch')
    backend = read_json(backend_path)
    contact = backend['asset_proxy_rules']['contact']['support']
    support_dynamics = {
        'lateral_friction': float(contact['lateral_friction']),
        'restitution': float(contact['restitution']),
    }
    surface = _effective_surface(declared_surface, effective_surface_override)
    center = [float(value) for value in surface['center_xy_m']]
    size = [float(value) for value in surface['size_xy_m']]
    plane = float(surface['z_m'])
    target_size = [float(value) for value in binding['target_support_frame']['size_xy_m']]

    scene = copy.deepcopy(environment_host_template)
    if scene.get('schema_version') != 'physweep_pybullet_rigid_metadata_v1':
        raise ValueError('asset host environment donor must use rigid metadata v1')
    support = {
        'support_shape': 'rectangular_slab',
        'topology': 'flat_surface',
        'layout': 'exact_mesh_tabletop',
        'semantic_type': str(registry['semantic_category']),
        'scene_class': 'raised_flat',
        'size_m': [target_size[0], target_size[1], 0.05],
        'surface_center_z_m': plane,
        'safe_surface_bounds': {
            'x': [round(center[0] - size[0] / 2.0, 6), round(center[0] + size[0] / 2.0, 6)],
            'y': [round(center[1] - size[1] / 2.0, 6), round(center[1] + size[1] / 2.0, 6)],
        },
        'surface_frame': {
            'normal': [0.0, 0.0, 1.0],
            'slope_angle_degrees': 0.0,
            'tangent_cross': [1.0, 0.0, 0.0],
            'tangent_uphill': [0.0, 1.0, 0.0],
        },
        'colliders': [{
            'id': 'support',
            'role': 'primary_support',
            'primitive': 'box',
            'size_m': [target_size[0], target_size[1], 0.05],
            'position_m': [
                float(binding['target_support_frame']['center_xy_m'][0]),
                float(binding['target_support_frame']['center_xy_m'][1]),
                plane - 0.025,
            ],
            'rotation_euler_degrees': [0.0, 0.0, 0.0],
            'collision_enabled': False,
            'visible': False,
            'occludes_camera': False,
            'replaced_by_static_support_binding': str(binding['binding_sha256']),
        }],
        'dynamics': support_dynamics,
        'exact_static_binding': binding,
        'collision_authority': 'exact_static_proxy',
        'placement_surface': surface,
        'asset_id': asset_id,
    }
    scene['simulation']['support'] = support
    environment = scene['environment_binding']
    environment['dynamics'] = {'policy': 'inherit_primary_support', **support_dynamics}
    environment['binding_sha256'] = environment_binding_sha256(environment)

    visual = registry['visual']
    support_visual = {
        'id': f'exact_asset_host_{asset_id}',
        'visual_type': 'mesh_support',
        'support_ids': [str(registry['semantic_category'])],
        'asset_id': asset_id,
        'path': str(visual['path']),
        'sha256': str(visual['sha256']),
        'material_policy': 'source_or_bound_fallback',
        'requires_image_texture': bool(visual.get('requires_image_texture', False)),
        'license': str(visual['license']),
    }
    scene['appearance']['support_visual'] = support_visual
    interaction = scene['semantic_sampling']['five_dimensions']['support_interaction']
    interaction.update({
        'scene_class': 'raised_flat',
        'support_type': str(registry['semantic_category']),
        'support_layout': 'exact_mesh_tabletop',
        'support_shape': 'rectangular_slab',
        'support_visual_profile': support_visual['id'],
        'support_visual_type': 'mesh_support',
        'collision_authority': 'exact_static_proxy',
    })
    interaction['scene_visual_profile'] = scene['appearance']['scene_visual']['id']
    interaction['scene_visual_type'] = scene['appearance']['scene_visual']['visual_type']
    return scene
