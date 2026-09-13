"""Generate admitted three-object bases, their sweeps, videos and canonical data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

from tools.cli.dataset_generation import bind_generation_plan, generation_code_sha256, generation_layout, run, verify_render_manifest
from tools.cli.build_three_object_dataset import publish_dataset, verify_dataset
from tools.core.hashing import sha256_file
from tools.core.json_io import frozen_json, read_json
from tools.dataset_contract.trajectory_contract import object_trajectory_view
from tools.physics.pybullet_backend_dispatcher import dispatch_simulation
from tools.physics.run_pybullet_batch import worker_context
from tools.release.base_release_view import PipelineSpec
from tools.release.source_release import publish_source_release
from tools.rendering.prepare_three_object import prepare_group, write_manifest, RENDERERS
from tools.rendering.three_object_camera import solve_three_object_camera
from tools.rendering.three_object_billiards_camera import solve_billiards_camera
from tools.rendering.three_object_pinball_camera import solve_pinball_camera
from tools.rendering.three_object_marble_camera import solve_marble_camera
from tools.sampling.derive_physics_sweep import load_sweep_config, _load_prior_indexes
from tools.sampling.three_object_fingerprints import physics_fingerprint
from tools.core.sampling_counts import allocate_counts
from tools.sampling.three_object_generation import CandidateFactory, GENERIC_RULES
from tools.sampling.three_object_sweeps import derive_group
from tools.sampling.three_object_rule_matrix import load_and_validate_rule_matrix

CODE_ROOT = Path(__file__).resolve().parents[2]
MODE_WEIGHTS = {'sphere': 30, 'mixed': 18, 'motion': 18, 'multi_mesh': 20, 'inclined': 6,
                'exact_support': 5, 'billiards': 1, 'passive_pinball': 1, 'marble_run': 1}
SCHEMAS = {'generic': 'physweep_pybullet_rigid_metadata_v1', 'billiards': 'physweep_billiards_three_object_scene_v1',
           'passive_pinball': 'physweep_passive_pinball_three_object_scene_v1', 'marble_run': 'physweep_marble_run_three_object_scene_v1'}


def simulation(metadata_path: Path, output: Path, root: Path, resume: bool, *, require_admission: bool = True) -> dict:
    path = output/'simulation_record.json'
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        result = read_json(path)
        if Path(result['metadata_path']).resolve() != metadata_path.resolve() or result['metadata_sha256'] != sha256_file(metadata_path):
            raise ValueError('resumed metadata content or path changed')
        if result['scene_id'] != read_json(metadata_path)['scene_id']:
            raise ValueError('resumed scene identity changed')
        for field in ('metadata', 'resolved_scene', 'trajectory', 'audit'):
            if sha256_file(Path(result[field+'_path'])) != result[field+'_sha256']:
                raise ValueError(f'resumed {field} changed')
    else:
        result = dispatch_simulation(metadata_path, output, root)
    if require_admission and not result['audit_passed']:
        raise ValueError(f'physics admission failed: {result["scene_id"]} {result["failed_checks"]}')
    return result


def solve_camera(root, metadata, arrays):
    schema = metadata['schema_version']
    solvers = {'physweep_billiards_three_object_scene_v1': solve_billiards_camera,
               'physweep_passive_pinball_three_object_scene_v1': solve_pinball_camera,
               'physweep_marble_run_three_object_scene_v1': solve_marble_camera}
    if schema in solvers:
        return solvers[schema](root, metadata, arrays)
    return solve_three_object_camera(metadata, object_trajectory_view(metadata, arrays), root=root)


def initialize_candidates(factory_args):
    global FACTORY
    FACTORY = CandidateFactory(*factory_args)


def admit_slot(job):
    root, base_dir, mode, ordinal, attempts, resume = job
    result_path = base_dir/'groups'/f'{mode}_{ordinal:06d}.json'
    if result_path.exists():
        if not resume:
            raise FileExistsError(result_path)
        record = read_json(result_path)
        physical = record['physics']
        simulation(Path(physical['metadata_path']), Path(physical['trajectory_path']).parent, root, True)
        with np.load(physical['trajectory_path']) as arrays:
            camera = solve_camera(root, read_json(Path(physical['metadata_path'])), {key: arrays[key] for key in arrays.files})
        if camera != record['camera']:
            raise ValueError('resumed base camera changed')
        return record
    if mode in GENERIC_RULES:
        FACTORY.generic_context(mode)
    else:
        FACTORY.special_context(mode)
    failures = []
    for attempt in range(attempts):
        try:
            metadata = FACTORY.build(mode, ordinal, attempt)
            path = base_dir/'scenes'/metadata['scene_id']/'metadata.json'
            frozen_json(path, metadata)
            physical = simulation(path, base_dir/'physics'/metadata['scene_id'], root, resume)
            with np.load(physical['trajectory_path']) as arrays:
                camera = solve_camera(root, metadata, {key: arrays[key] for key in arrays.files})
            record = {'scene_id': metadata['scene_id'], 'metadata_path': path.relative_to(root).as_posix(),
                      'metadata_sha256': sha256_file(path), 'source_schema_version': metadata['schema_version'],
                      'family': 'generic' if mode in GENERIC_RULES else mode, 'mode': mode, 'audit_passed': True,
                      'physics': physical, 'camera': camera, 'physics_fingerprint': physics_fingerprint(metadata, root=root)}
            frozen_json(result_path, record)
            return record
        except ValueError as error:
            failures.append({'attempt': attempt, 'reason': str(error)})
    frozen_json(base_dir/'failures'/f'{mode}_{ordinal:06d}.json', failures)
    raise ValueError(f'{mode} slot {ordinal} exhausted {attempts} attempts: {failures[-3:]}')


def simulate_member(job):
    root, path, output, resume, kind = job
    return {**simulation(path, output, root, resume), 'ok': True, 'kind': kind}


def execute(args):
    root = args.root.resolve()
    load_and_validate_rule_matrix(CODE_ROOT)
    if not args.gpus or any(not value.strip().isdigit() for value in args.gpus.split(',')):
        raise ValueError('GPU ids must be comma-separated nonnegative integers')
    modes = [value.strip() for value in args.families.split(',') if value.strip()]
    if len(set(modes)) != len(modes) or set(modes) - set(MODE_WEIGHTS):
        raise ValueError('unknown or duplicate family/mode')
    counts = allocate_counts(args.count, {key: MODE_WEIGHTS[key] for key in modes}, minimum=1)
    if min(args.physics_workers, args.render_workers, args.max_attempts) < 1:
        raise ValueError('worker and attempt counts must be positive')
    layout = generation_layout(root, args.work_id, args.release_root, object_count=3)
    sources = {name: (root/getattr(args, name)).resolve() for name in ('source_root', 'source_manifest', 'released_base_manifest')}
    plan = {'schema_version': 'physweep_three_object_generation_plan_v1', 'work_id': args.work_id,
            'code_sha256': generation_code_sha256(), 'seed': args.seed, 'base_count': args.count,
            'sample_count': args.count*13, 'mode_counts': counts, 'max_attempts': args.max_attempts,
            'release_root': str(layout.canonical_release),
            'sources': {name: {'path': str(path), **({'sha256': sha256_file(path)} if name != 'source_root' else {})}
                        for name, path in sources.items()}}
    if args.plan_only:
        return plan
    plan_path = root/'outputs'/args.work_id/'generation_plan.json'
    bind_generation_plan(plan_path, plan, args.resume)
    factory_args = (root, sources['source_root'], sources['source_manifest'], sources['released_base_manifest'], args.work_id, args.seed)
    jobs = [(root, layout.base_dataset, mode, index, args.max_attempts, args.resume)
            for mode, count in counts.items() for index in range(count)]
    groups = []
    adapter_jobs = defaultdict(list)
    for job in jobs:
        adapter_jobs['generic' if job[2] in GENERIC_RULES else job[2]].append(job)
    for batch in adapter_jobs.values():
        with ProcessPoolExecutor(max_workers=min(args.physics_workers, len(batch)), mp_context=worker_context(),
                                 initializer=initialize_candidates, initargs=(factory_args,)) as executor:
            groups.extend(executor.map(admit_slot, batch))
    fingerprints = [group['physics_fingerprint'] for group in groups]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError('duplicate admitted physical groups; use another seed')
    frozen_json(layout.base_manifest, {'schema_version': 'physweep_three_object_base_manifest_v1',
                                      'sample_count': len(groups), 'records': groups})
    if args.stage == 'base':
        return {'status': 'base_admitted', 'base_count': len(groups), 'manifest': str(layout.base_manifest)}
    metadata_records = []
    jobs = []
    parent_by_scene = {}
    configs = {}
    for group in groups:
        family = group['family']
        config_name = {'generic': 'three_object_physics_sweep', 'billiards': 'three_object_billiards_physics_sweep',
                       'passive_pinball': 'three_object_pinball_physics_sweep', 'marble_run': 'three_object_marble_physics_sweep'}[family]
        config_path = CODE_ROOT/'configs'/f'{config_name}.json'
        if family not in configs:
            config = load_sweep_config(config_path)
            configs[family] = (config, *_load_prior_indexes(root, config))
        config, profiles, registry = configs[family]
        path = root/group['metadata_path']
        for member in derive_group(read_json(path), path, root, config, config_path, profiles, registry):
            member_path = layout.sweep_metadata/member['scene_id']/'metadata.json'
            frozen_json(member_path, member)
            sweep = member['sweep']
            metadata_records.append({'path': member_path.relative_to(root).as_posix(), 'metadata_sha256': sha256_file(member_path),
                                     'scene_id': member['scene_id'], 'parent': sweep['parent_metadata_path'],
                                     'source_schema_version': member['schema_version'],
                                     **{key: sweep.get(key) for key in ('kind', 'axis', 'level_index', 'value', 'target_object_id', 'target_object_index')}})
            parent_by_scene[member['scene_id']] = group['scene_id']
            jobs.append((root, member_path, layout.sweep_physics/member['scene_id'], args.resume, sweep['kind']))
    frozen_json(layout.sweep_metadata/'manifest.json', {'schema_version': 'physweep_physics_sweep_metadata_manifest_v1',
                                                       'sample_count': len(metadata_records), 'records': metadata_records})
    adapter_jobs = defaultdict(list)
    for record, job in zip(metadata_records, jobs):
        adapter_jobs[record['source_schema_version']].append(job)
    physical = []
    for batch in adapter_jobs.values():
        with ProcessPoolExecutor(max_workers=min(args.physics_workers, len(batch)), mp_context=worker_context()) as executor:
            physical.extend(executor.map(simulate_member, batch))
    frozen_json(layout.sweep_physics/'manifest.json', {'schema_version': 'physweep_dispatched_physics_manifest_v1',
                                                     'sample_count': len(physical), 'records': physical})
    if not (layout.source_release/'manifest.json').exists():
        publish_source_release(root=root, base_manifest_path=layout.base_manifest,
                               sweep_metadata_manifest_path=layout.sweep_metadata/'manifest.json',
                               sweep_physics_manifest_path=layout.sweep_physics/'manifest.json', output=layout.source_release,
                               object_count=3, target_object_indices=(0,), dataset_id='physweep_three_object',
                               release_schema='physweep_three_object_source_release_v1')
    if args.stage == 'physics':
        return {'status': 'physics_complete', 'sample_count': len(physical), 'source_release': str(layout.source_release)}
    by_parent = defaultdict(list)
    for record in physical:
        by_parent[parent_by_scene[record['scene_id']]].append(record)
    bound = defaultdict(list)
    for group in groups:
        family, records = prepare_group(root, layout.sweep_render, group, by_parent[group['scene_id']])
        bound[family].extend(records)
    specs = []
    for family, records in bound.items():
        manifest = write_manifest(root, layout.sweep_render, family, records)
        generic = family == 'generic'
        result_path = layout.sweep_render/family/'render_manifest.json'
        command = [sys.executable, '-m', 'tools.rendering.render_pybullet_manifest' if generic else 'tools.rendering.render_asset_proxy_manifest',
                   '--root', str(root), '--manifest', str(manifest), '--result-manifest', str(result_path),
                   '--workers', str(args.render_workers), '--gpus', args.gpus]
        if not generic:
            command.extend(('--renderer', RENDERERS[family]))
        if args.resume:
            command.append('--resume')
        run(command)
        verify_render_manifest(result_path, len(records))
        specs.append(PipelineSpec(family, SCHEMAS[family], root, layout.sweep_render/family))
    result = publish_dataset(release_project_root=root, release_manifest=layout.source_release/'manifest.json',
                             release_root=layout.canonical_release, pipeline_specs=specs,
                             workers=args.render_workers, resume=args.resume)
    verify_dataset(layout.canonical_release)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=CODE_ROOT)
    parser.add_argument('--work-id', required=True)
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260912)
    parser.add_argument('--source-root', type=Path, default=Path('assets/source_pool/generation'))
    parser.add_argument('--source-manifest', type=Path, default=Path('assets/source_pool/generation/datasets/one_object_v5/release/metadata_manifest.json'))
    parser.add_argument('--released-base-manifest', type=Path, default=Path('assets/source_pool/released/base/manifest.json'))
    parser.add_argument('--families', default=','.join(MODE_WEIGHTS))
    parser.add_argument('--release-root', type=Path, default=Path('outputs/three_object'))
    parser.add_argument('--physics-workers', type=int, default=4)
    parser.add_argument('--render-workers', type=int, default=4)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--max-attempts', type=int, default=64)
    parser.add_argument('--stage', choices=('base', 'physics', 'all'), default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    print(json.dumps(execute(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
