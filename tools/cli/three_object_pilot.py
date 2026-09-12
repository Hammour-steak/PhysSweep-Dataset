"""Explicit pilot checkpoints; candidates remain separate from admitted releases."""
from collections import Counter
from pathlib import Path

from tools.cli.dataset_generation import generation_layout,generation_code_sha256,bind_generation_plan
from tools.core.json_io import frozen_json
from tools.core.hashing import sha256_file
from tools.core.json_io import read_json
from tools.sampling.three_object_sampling_request import load_pilot_rules,ROOT
from tools.sampling.three_object_sources import load_sources
from tools.sampling.three_object_coverage import build_plan,candidates
from tools.sampling.derive_physics_sweep import load_sweep_config


def counter_records(counter):
    return [{'value':list(key) if isinstance(key,tuple) else key,'count':value} for key,value in sorted(counter.items())]


def metadata_reports(records):
    counters={key:Counter() for key in ('template_host','template_target_role','template_requested_camera','target_role_object_visual',
                                       'environment','support_material','object_source_uses','host_source_uses','rejections')}
    for record in records:
        for attempt in record['attempts']:
            if attempt['status']=='rejected_metadata':counters['rejections'][attempt['reason']]+=1
        if record['status']!='metadata_ready':continue
        scene=read_json(Path(record['metadata_path']));cell=scene['coverage']['cell'];roles=scene['three_object']['roles']
        counters['template_host'][(cell['template'],cell['host_class'])]+=1
        counters['template_target_role'][(cell['template'],cell['target_role'])]+=1
        counters['template_requested_camera'][(cell['template'],cell['requested_camera_family'])]+=1
        for obj in scene['simulation']['objects']:
            role=next(role for role,value in roles.items() if value==obj['object_id'])
            counters['target_role_object_visual'][(cell['target_role'],role,obj['visual_profile']['id'])]+=1
        counters['environment'][scene['appearance']['scene_visual']['environment_category']]+=1
        counters['support_material'][scene['appearance']['materials']['support_surface']['record']['asset_id']]+=1
        counters['object_source_uses'].update(r['scene_id'] for r in scene['source_binding']['objects'])
        counters['host_source_uses'][scene['source_binding']['host']['scene_id']]+=1
    return {'schema_version':'physweep_three_object_metadata_coverage_v1',
            'counters':{key:counter_records(value) for key,value in counters.items()},
            'observation_limit':'requested cameras only; physical events and final camera coverage are not yet evaluated',
            'near_duplicate_policy':'exact physical signatures reject duplicates; continuous near-duplicate analysis remains a later pilot diagnostic'}


def run_pilot(args):
    if args.checkpoint not in ('metadata','physics'):raise ValueError('pilot currently implements metadata and physics checkpoints')
    root=args.root.resolve();rules=load_pilot_rules(matrix_path=getattr(args,'rules_matrix',None))
    scope_gate=getattr(args,'scope_gate',None)
    if rules['matrix'].get('geometry_scope')=='sphere_cross_slope_pair_v1':
        if scope_gate is None:raise ValueError('inclined extension requires a passed D5a completion gate')
        previous=read_json(scope_gate)
        if previous.get('stage')!='D5a' or previous.get('status')!='passed':raise ValueError('D5a has not passed')
        for name,digest in previous['evidence'].items():
            if sha256_file(scope_gate.parent/name)!=digest:raise ValueError('D5a evidence changed')
        archive=previous['source_archive']
        if sha256_file(Path(archive['path']))!=archive['sha256']:raise ValueError('D5a source archive changed')
    if rules['matrix'].get('geometry_scope') in {'mixed_flat_R_v1','multi_mesh_pair_roles_v1'}:
        if scope_gate is None:raise ValueError('mixed extension requires a passed D4 scope gate')
        previous=read_json(scope_gate)
        if previous.get('stage')!='D4' or previous.get('status')!='passed':raise ValueError('D4 has not passed')
        for name,digest in previous['evidence'].items():
            if sha256_file(scope_gate.parent/name)!=digest:raise ValueError('D4 evidence changed')
        for key in ('source_archive','main_media_source_archive'):
            if sha256_file(Path(previous[key]['path']))!=previous[key]['sha256']:raise ValueError('D4 source archive changed')
    if args.d1_gate is None:raise ValueError('pilot requires a passed --d1-gate')
    gate=read_json(args.d1_gate)
    if gate.get('stage')!='D1' or gate.get('status')!='passed':raise ValueError('D1 has not passed')
    for name,digest in gate['evidence'].items():
        if sha256_file(args.d1_gate.parent/name)!=digest:raise ValueError('D1 evidence changed')
    if sha256_file(Path(gate['source_archive']['path']))!=gate['source_archive']['sha256']:raise ValueError('D1 source archive changed')
    d0=read_json(args.d0_freeze)
    if d0.get('status')!='passed' or d0['rules_sha256']!=rules['rules_sha256']:raise ValueError('D0 frozen rules differ')
    layout=generation_layout(root,args.work_id,Path(f'outputs/work/{args.work_id}/three_object'),object_count=3)
    plan={'schema_version':'physweep_three_object_pilot_generation_v1','work_id':args.work_id,'object_count':3,'target_object_indices':[0],
          'code_sha256':generation_code_sha256(),'source_code_root':str(ROOT),'rules_sha256':rules['rules_sha256'],
          'source_root':str(args.source_root.resolve()),'candidate_budget':rules['matrix']['candidate_budget'],
          'bindings':{key:{'path':str(path.resolve()),'sha256':sha256_file(path)} for key,path in (
              ('source_manifest',args.source_manifest),('released_base_manifest',args.released_base_manifest),
              ('d0_freeze',args.d0_freeze),('d1_gate',args.d1_gate))}}
    if scope_gate is not None:plan['bindings']['scope_gate']={'path':str(scope_gate.resolve()),'sha256':sha256_file(scope_gate)}
    plan_path=root/'outputs'/args.work_id/'generation_plan.json';bind_generation_plan(plan_path,plan,args.resume)
    pool=load_sources(root=root,source_root=args.source_root,source_manifest=args.source_manifest,
                      released_manifest=args.released_base_manifest,rules=rules)
    inputs=layout.base_dataset.parent/'inputs';frozen_json(inputs/'rules.json',rules)
    frozen_json(inputs/'source_pool.json',{'summary':pool['summary'],
                'objects':[r['source'] for r in pool['objects']],'hosts':[r['source'] for r in pool['hosts']]})
    allocation=build_plan(pool,rules);frozen_json(inputs/'coverage_plan.json',allocation)
    sweep_config=load_sweep_config(ROOT/'configs/three_object_physics_sweep.json')
    frozen_json(inputs/'resolved_sweep.json',sweep_config)
    records=[]
    for outcome in candidates(pool,rules,allocation,root,args.work_id,sweep_config):
        scene=outcome.pop('scene')
        if scene is not None:
            path=layout.base_dataset/'scenes'/scene['scene_id']/'metadata.json';frozen_json(path,scene)
            outcome.update(scene_id=scene['scene_id'],metadata_path=str(path),metadata_sha256=sha256_file(path))
        records.append(outcome)
        if len(records)%10==0:print(f'metadata candidates completed: {len(records)}/{allocation["candidate_count"]}',flush=True)
    frozen_json(layout.base_dataset/'candidates_manifest.json',{'schema_version':'physweep_three_object_candidates_manifest_v1',
                'candidate_count':len(records),'metadata_ready_count':sum(r['status']=='metadata_ready' for r in records),'records':records})
    report=metadata_reports(records);frozen_json(root/'outputs'/args.work_id/'metadata_coverage.json',report)
    result={'status':'metadata_pilot_complete_pending_physics','candidate_count':len(records),
            'metadata_ready_count':sum(r['status']=='metadata_ready' for r in records),
            'unavailable_count':sum(r['status']!='metadata_ready' for r in records),
            'code_sha256':generation_code_sha256(),'rules_sha256':rules['rules_sha256'],
            'candidate_manifest':str(layout.base_dataset/'candidates_manifest.json')}
    frozen_json(root/'outputs'/args.work_id/'metadata_result.json',result)
    if args.checkpoint=='physics':
        from tools.cli.three_object_pilot_physics import run_physics
        return run_physics(root,layout,records,args)
    return result
