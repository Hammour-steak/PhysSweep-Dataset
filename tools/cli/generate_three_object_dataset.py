#!/usr/bin/env python3
"""Run a frozen three-object pilot through implemented checkpoints."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from tools.cli.dataset_generation import generation_layout,generation_code_sha256,bind_generation_plan
from tools.core.hashing import sha256_file
from tools.core.json_io import read_json, frozen_json as immutable_json
from tools.dataset_contract.three_object_group import validate_group_inputs
from tools.physics.pybullet_backend_dispatcher import dispatch_simulation
from tools.release.source_release import publish_source_release
from tools.release.audit_release_provenance import audit_release
from tools.sampling.derive_physics_sweep import load_sweep_config,_load_prior_indexes
from tools.sampling.released_object_sources import verified_generation_records
from tools.sampling.sample_three_object_base import build_three_object_scene
from tools.sampling.three_object_sampling_request import load_pilot_rules,ROOT


def simulation(metadata_path: Path, output: Path, root: Path, resume: bool, *, require_admission: bool=True) -> dict:
    path=output/'simulation_record.json'
    if path.exists():
        if not resume:raise FileExistsError(path)
        result=read_json(path)
        if Path(result['metadata_path']).resolve()!=metadata_path.resolve():raise ValueError('resumed metadata path changed')
        if result['metadata_sha256']!=sha256_file(metadata_path):raise ValueError('resumed metadata changed')
        if result['scene_id']!=read_json(metadata_path)['scene_id']:raise ValueError('resumed scene identity changed')
        for field in ('metadata','resolved_scene','trajectory','audit'):
            if sha256_file(Path(result[field+'_path']))!=result[field+'_sha256']:raise ValueError(f'resumed {field} changed')
    else:result=dispatch_simulation(metadata_path,output,root)
    if require_admission and not result['audit_passed']:raise ValueError(f"physics admission failed: {result['scene_id']} {result['failed_checks']}")
    return result


def verify_source_release(root: Path, release_path: Path, base_manifest: Path,
                          metadata_manifest: Path, physics_manifest: Path, base_count: int = 1) -> None:
    audit_release(release_path,root)
    release=read_json(release_path)
    expected={'schema_version':'physweep_three_object_source_release_v1',
              'dataset_id':'physweep_three_object_pilot_v1','object_count':3,
              'sweep_target_object_indices':[0],'sample_count':13*base_count,'base_count':base_count,'derived_count':12*base_count}
    if any(release.get(key)!=value for key,value in expected.items()):
        raise ValueError('source release identity or group totals changed')
    for key,source in (('base_manifest',base_manifest),('metadata_manifest',metadata_manifest),('physics_manifest',physics_manifest)):
        published=read_json(root/release[key])
        if published['records']!=read_json(source)['records']:
            raise ValueError(f'source release records differ: {key}')
        if key!='base_manifest':
            binding=published['sources']
            if len(binding)!=1:raise ValueError('source release has unexpected source bindings')
            for name,path in (('metadata_manifest',metadata_manifest),('physics_manifest',physics_manifest)):
                if (root/binding[0][name]).resolve()!=path.resolve() or binding[0][name+'_sha256']!=sha256_file(path):
                    raise ValueError(f'source release input differs: {name}')


def run(args: argparse.Namespace) -> dict:
    if getattr(args,'pilot',False):
        from tools.cli.three_object_pilot import run_pilot
        return run_pilot(args)
    if getattr(args,'rules_matrix',None) is not None or getattr(args,'scope_gate',None) is not None:
        raise ValueError('coverage-extension rules require the pilot checkpoint route')
    if args.source_selection is None:raise ValueError('the single-group checkpoint requires --source-selection')
    root=args.root.resolve();rules=load_pilot_rules();selection_path=args.source_selection.resolve()
    freeze=read_json(args.d0_freeze)
    if freeze.get('status')!='passed' or freeze.get('rules_sha256')!=rules['rules_sha256']:
        raise ValueError('D0 freeze is missing or differs from the current rules')
    layout=generation_layout(root,args.work_id,Path(f'outputs/work/{args.work_id}/three_object'),object_count=3)
    input_dir=layout.base_dataset.parent/'inputs'
    plan={'schema_version':'physweep_three_object_pilot_plan_v1','work_id':args.work_id,'object_count':3,'target_object_indices':[0],
          'code_sha256':generation_code_sha256(),'source_code_root':str(ROOT),'rules':rules['bindings'],'rules_sha256':rules['rules_sha256'],
          'source_selection':{'path':str(selection_path),'sha256':sha256_file(selection_path)},
          'd0_freeze':{'path':str(args.d0_freeze.resolve()),'sha256':sha256_file(args.d0_freeze)},
          'source_root':str(args.source_root.resolve()),'source_manifest':{'path':str(args.source_manifest.resolve()),'sha256':sha256_file(args.source_manifest)},
          'released_base_manifest':{'path':str(args.released_base_manifest.resolve()),'sha256':sha256_file(args.released_base_manifest)},
          'role_order':args.role_order,'scope':'D1 single chain_transfer group','release_root':str(layout.canonical_release)}
    plan_path=root/'outputs'/args.work_id/'generation_plan.json'
    bind_generation_plan(plan_path,plan,args.resume)
    immutable_json(input_dir/'resolved_rules.json',rules)
    selection=read_json(selection_path)
    source_contract={'released_base_manifest_schema_version':rules['matrix']['source']['release_schema'],
                     'generation_manifest_schema_version':rules['matrix']['source']['generation_manifest_schema'],'sample_kind':'base'}
    rows=verified_generation_records(root=root,released_base_manifest_path=args.released_base_manifest,
        source_root=args.source_root,source_manifest_path=args.source_manifest,source_contract=source_contract,
        family_schemas={rules['matrix']['source']['generation_metadata_schema']:'generic'})
    by_id={r['source']['scene_id']:r for r in rows}
    host=by_id[selection['host']['source']['scene_id']]
    objects=[by_id[r['source']['scene_id']] for r in selection['objects']]
    for requested,verified in zip([selection['host'],*selection['objects']],[host,*objects]):
        if requested['source']!=verified['source']:raise ValueError('selected source binding differs from verified release lineage')
    candidate=build_three_object_scene(host_source=host,object_sources=objects,rules=rules,
        scene_id=f'{args.work_id}_chain_000',role_order=args.role_order,root=root)
    candidate_path=layout.base_dataset/'scenes'/candidate['scene_id']/'metadata.json'
    immutable_json(candidate_path,candidate)
    if args.checkpoint=='metadata':return {'status':'metadata_only','candidate':str(candidate_path)}
    physical=simulation(candidate_path,layout.base_dataset/'physics'/candidate['scene_id'],root,args.resume)
    immutable_json(layout.base_manifest,{'schema_version':'physweep_three_object_base_manifest_v1','sample_count':1,
        'records':[{'scene_id':candidate['scene_id'],'metadata_path':candidate_path.relative_to(root).as_posix(),
                    'metadata_sha256':sha256_file(candidate_path),'source_schema_version':candidate['schema_version'],
                    'family':'generic','audit_passed':True,'physics':physical}]})
    # The legacy derivation API binds configs relative to its data root.
    # Place an exact frozen copy there, retaining the executable source separately.
    config=load_sweep_config(ROOT/'configs/three_object_physics_sweep.json')
    config_path=input_dir/'resolved_sweep.json';immutable_json(config_path,config)
    profiles,registry=_load_prior_indexes(root,config)
    from tools.sampling.three_object_sweeps import derive_group
    members=derive_group(candidate,candidate_path,root,config,config_path,profiles,registry)
    invariant=validate_group_inputs(candidate,members,[0])
    metadata_records=[];physics_records=[]
    for member in members:
        path=layout.sweep_metadata/member['scene_id']/'metadata.json';immutable_json(path,member)
        sweep=member['sweep']
        meta={'path':path.relative_to(root).as_posix(),'metadata_sha256':sha256_file(path),'scene_id':member['scene_id'],
            'parent':sweep['parent_metadata_path'],'source_schema_version':member['schema_version'],
            **{k:sweep.get(k) for k in ('kind','axis','level_index','value','target_object_id','target_object_index')}}
        metadata_records.append(meta)
        result=simulation(path,layout.sweep_physics/member['scene_id'],root,args.resume)
        physics_records.append({**result,'ok':True,'kind':sweep['kind']})
    immutable_json(layout.sweep_metadata/'manifest.json',{'schema_version':'physweep_physics_sweep_metadata_manifest_v1','sample_count':13,'records':metadata_records})
    immutable_json(layout.sweep_physics/'manifest.json',{'schema_version':'physweep_dispatched_physics_manifest_v1','sample_count':13,'records':physics_records})
    if not (layout.source_release/'manifest.json').exists():
        publish_source_release(root=root,base_manifest_path=layout.base_manifest,
            sweep_metadata_manifest_path=layout.sweep_metadata/'manifest.json',sweep_physics_manifest_path=layout.sweep_physics/'manifest.json',
            output=layout.source_release,object_count=3,dataset_id='physweep_three_object_pilot_v1',
            release_schema='physweep_three_object_source_release_v1',target_object_indices=(0,))
    verify_source_release(root,layout.source_release/'manifest.json',layout.base_manifest,
                          layout.sweep_metadata/'manifest.json',layout.sweep_physics/'manifest.json')
    result={'status':'physics_passed_pending_camera_and_render','source_release':str(layout.source_release/'manifest.json'),
            'code_sha256':generation_code_sha256(),'group_invariants':invariant,'members':13}
    immutable_json(root/'outputs'/args.work_id/'physics_result.json',result)
    if args.checkpoint not in ('metadata','physics'):
        from tools.cli.three_object_visual_stage import execute
        return execute(root,layout,physical,physics_records,args,plan_path)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('root','source-root','source-manifest','released-base-manifest','d0-freeze'):
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--source-selection',type=Path)
    parser.add_argument('--pilot',action='store_true',help='Run the explicit 120-candidate coverage pilot')
    parser.add_argument('--d1-gate',type=Path)
    parser.add_argument('--rules-matrix',type=Path,help='Explicit versioned coverage matrix under the code configs directory')
    parser.add_argument('--scope-gate',type=Path,help='Passed preceding stage for a coverage extension')
    parser.add_argument('--physics-workers',type=int,default=4,help='Independent groups per bounded physics wave')
    parser.add_argument('--work-id',required=True)
    parser.add_argument('--checkpoint',choices=['metadata','physics','visual','render','publish'],default='metadata')
    parser.add_argument('--gpus',help='Explicit CUDA device indices for the render execution plan')
    parser.add_argument('--blender',type=Path)
    parser.add_argument('--role-order',nargs=3,default=['P','Q','R'])
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();print(json.dumps(run(args),sort_keys=True))


if __name__=='__main__':main()
