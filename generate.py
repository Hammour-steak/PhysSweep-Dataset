#!/usr/bin/env python3
"""Install resources and generate PhysSweep datasets with one command."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
NAMES = {1: 'one_object', 2: 'two_object', 3: 'three_object'}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--objects', choices=('1', '2', '3', 'all'), default='1')
    result.add_argument('--count', type=int, default=100, help='Number of bases per selected object count; each produces 13 samples')
    result.add_argument('--seed', type=int, default=20260912)
    result.add_argument('--run-id', default='generation', help='Stable checkpoint name; use the same name with --resume')
    result.add_argument('--output', type=Path, default=Path('outputs'), help='Parent of one_object, two_object and three_object')
    result.add_argument('--gpus', default='0')
    result.add_argument('--physics-workers', type=int, default=4)
    result.add_argument('--render-workers', type=int, default=4)
    result.add_argument('--resume', action='store_true')
    result.add_argument('--setup', action='store_true', help='Create the Python environment and download Blender and all required assets')
    result.add_argument('--setup-only', action='store_true', help='Install and verify resources, without sampling')
    result.add_argument('--plan-only', action='store_true', help='Print the request without installing, downloading or generating')
    return result


def two_object_request(count, seed):
    from tools.core.sampling_counts import allocate_counts
    from tools.sampling.two_object_sampling_request import validate_sampling_request
    rules = json.loads((ROOT/'configs/two_object_specialized_scene_rules.json').read_text())
    request = json.loads((ROOT/'configs/two_object_production_sampling.json').read_text())
    minimum = {f['id']: len(f['profiles']) for f in rules['scene_families']}
    if count < 1 + sum(minimum.values()):
        raise ValueError(f'2obj requires at least {1+sum(minimum.values())} bases to cover the fixture profiles')
    counts = allocate_counts(count - sum(minimum.values()), {'generic': 92.8, 'billiards': 2.4, 'passive_pinball': 2.4, 'marble_run': 2.4})
    for family, value in minimum.items():
        counts[family] += value
        variation = request['specialized_variation'][family]
        capacity = value * len(variation['separation_scales']) * len(variation['speed_scales'])
        counts['generic'] += max(0, counts[family]-capacity)
        counts[family] = min(counts[family], capacity)
    request['family_base_counts'] = counts
    request['generic_coverage']['seed'] = seed
    validate_sampling_request(request, rules)
    return request


def source_inputs(work_id):
    from tools.core.json_io import frozen_json
    from tools.sampling.released_object_sources import verified_generation_records
    from tools.sampling.released_object_sources import localize_marble_source_rows
    source = ROOT/'assets/source_pool/generation'
    manifest = source/'datasets/one_object_v5/release/metadata_manifest.json'
    released = ROOT/'assets/source_pool/released/base/manifest.json'
    schemas = {'physweep_billiards_scene_v4': 'billiards', 'physweep_passive_pinball_scene_v1': 'passive_pinball',
               'physweep_marble_run_scene_v1': 'marble_run'}
    rows = verified_generation_records(root=ROOT, released_base_manifest_path=released, source_root=source,
        source_manifest_path=manifest, source_contract={'released_base_manifest_schema_version': 'physweep_base_release_view_v15',
            'generation_manifest_schema_version': 'physweep_release_metadata_manifest_v2', 'sample_kind': 'base'}, family_schemas=schemas)
    directory = ROOT/'datasets'/work_id/'inputs'
    templates = {}
    for schema, family in schemas.items():
        row = next(row for row in rows if row['metadata']['schema_version'] == schema)
        if family == 'marble_run':
            row = localize_marble_source_rows([row], source, ROOT, directory/'collision')[0]
        path = directory/(family+'_template.json')
        frozen_json(path, row['metadata'])
        templates[family] = path
    return source, manifest, released, templates


def main(argv=None):
    args = parser().parse_args(argv)
    if min(args.count, args.physics_workers, args.render_workers) < 1:
        raise ValueError('Counts and workers must be positive')
    from tools.core.paths import safe_scene_id
    safe_scene_id(args.run_id)
    if not args.gpus or any(not value.strip().isdigit() for value in args.gpus.split(',')):
        raise ValueError('--gpus must contain comma-separated GPU numbers')
    objects = list(NAMES) if args.objects == 'all' else [int(args.objects)]
    if not args.setup_only and ((1 in objects and args.count < 24) or (2 in objects and args.count < 10) or (3 in objects and args.count < 9)):
        raise ValueError('Use at least 24 bases for 1obj, 10 for 2obj and 9 for 3obj')
    if not args.setup_only:
        from tools.cli.dataset_generation import generation_layout
        for count in objects:
            generation_layout(ROOT, args.run_id+'_'+NAMES[count], args.output/NAMES[count], object_count=count)
    if args.plan_only:
        print(json.dumps({'objects': objects, 'bases_per_object_count': args.count,
            'samples_per_object_count': args.count*13, 'seed': args.seed,
            'outputs': [str(args.output/NAMES[n]) for n in objects],
            'setup': args.setup or args.setup_only, 'resume': args.resume}, indent=2))
        return
    os.environ['PATH'] = str(ROOT/'.venv/bin')+os.pathsep+os.environ.get('PATH', '')
    if args.setup or args.setup_only:
        from tools.assets.prepare_runtime import setup_environment, prepare_assets
        executable = setup_environment(ROOT)
        if Path(sys.executable).absolute() != executable.absolute():
            os.execv(str(executable), [str(executable), str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)])
        prepare_assets(ROOT)
        if args.setup_only:
            return
    for count in objects:
        name = NAMES[count]
        work_id = args.run_id+'_'+name
        command = [sys.executable, '-m', 'tools.cli.generate_'+name+'_dataset', '--work-id', work_id,
            '--release-root', str(args.output/name), '--physics-workers', str(args.physics_workers),
            '--render-workers', str(args.render_workers), '--gpus', args.gpus]
        if count == 2:
            from tools.core.json_io import frozen_json
            request = two_object_request(args.count, args.seed)
            config = ROOT/'datasets'/work_id/'inputs/sampling_request.json'
            frozen_json(config, request)
            source, manifest, released, templates = source_inputs(work_id)
            command += ['--sampling-config', str(config), '--specialized-seed', str(args.seed),
                '--source-root', str(source), '--source-manifest', str(manifest), '--released-base-manifest', str(released)]
            for family, template in templates.items():
                command += ['--'+family.replace('_','-')+'-template', str(template)]
        else:
            command += ['--count', str(args.count), '--seed', str(args.seed)]
        if args.resume:
            command += ['--resume']
        print('+ '+ ' '.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
