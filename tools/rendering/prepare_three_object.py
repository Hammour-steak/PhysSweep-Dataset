"""Freeze admitted base cameras and bind all three-object group members."""

import copy
from pathlib import Path

from tools.core.hashing import sha256_file
from tools.core.json_io import frozen_json, read_json
from tools.rendering.bind_pybullet_visuals import bind_scene
from tools.rendering.bind_physics_sweep_visuals import bind_one
from tools.rendering.three_object_specialized_binding import bind_render_implementation

RENDERERS = {'billiards': 'billiards_three_object', 'passive_pinball': 'passive_pinball_three_object',
             'marble_run': 'marble_run_three_object'}


def prepare_group(root: Path, output: Path, group: dict, members: list[dict]) -> tuple[str, list[dict]]:
    family = group['family']
    branch = output / family
    camera = group['camera']
    base = group['physics']
    records = []
    if family == 'generic':
        base_result = bind_scene(root, Path(base['metadata_path']), Path(base['trajectory_path']).with_name('simulation_record.json'),
                                 Path(base['trajectory_path']), branch/'base_bindings', {}, None, None)
        bound = read_json(root / base_result['metadata_path'])
        if bound['visualization']['camera'] != camera:
            raise ValueError('base camera changed between admission and visual binding')
        for physical in members:
            sample = {**physical, 'simulation_record_path': str(Path(physical['trajectory_path']).with_name('simulation_record.json'))}
            records.append(bind_one(root, sample, {base_result['scene_id']: base_result}, branch))
        return family, records
    for physical in members:
        metadata_path = Path(physical['metadata_path'])
        bound = copy.deepcopy(read_json(metadata_path))
        bound['source_metadata'] = {'path': metadata_path.relative_to(root).as_posix(), 'sha256': sha256_file(metadata_path)}
        simulation_path = Path(physical['trajectory_path']).with_name('simulation_record.json')
        bound.setdefault('physics', {}).update({
            **{key: physical[key] for key in ('trajectory_sha256', 'audit_sha256')},
            **{key: Path(physical[key]).relative_to(root).as_posix() for key in ('trajectory_path', 'audit_path')},
            'simulation_record_path': simulation_path.relative_to(root).as_posix(), 'simulation_record_sha256': sha256_file(simulation_path),
        })
        bound['camera'] = copy.deepcopy(camera)
        scene_id = bound['scene_id']
        video = branch/'videos'/f'{scene_id}.mp4'
        frames = branch/'frames'/scene_id
        bound['render'].update(video_path=video.relative_to(root).as_posix(), inspection_frame_dir=frames.relative_to(root).as_posix(),
                               use_motion_blur=False)
        bound['object_identity']['trajectory'].update(path=Path(physical['trajectory_path']).relative_to(root).as_posix(), path_policy='metadata_relative')
        bound['object_identity']['instance_masks'].update(path=None, path_policy='bound_render_manifest')
        bind_render_implementation(root, bound)
        path = branch/'metadata'/f'{scene_id}.json'
        frozen_json(path, bound)
        records.append({'scene_id': scene_id, 'metadata_path': path.relative_to(root).as_posix(),
                        'metadata_sha256': sha256_file(path), 'render_output': {
                            'video_path': video.relative_to(root).as_posix(), 'inspection_frame_dir': frames.relative_to(root).as_posix()}})
    return family, records


def write_manifest(root: Path, output: Path, family: str, records: list[dict]) -> Path:
    branch = output/family
    generic = family == 'generic'
    path = branch / ('bound_manifest.json' if generic else 'render_input_manifest.json')
    frozen_json(path, {
        'schema_version': 'physweep_pybullet_sweep_bound_manifest_v1' if generic else f'physweep_sweep_{family}_render_manifest_v1',
        'dataset_id': 'physweep_three_object', 'output_root': branch.relative_to(root).as_posix(),
        'sample_count': len(records), 'samples' if generic else 'records': records,
    })
    return path
