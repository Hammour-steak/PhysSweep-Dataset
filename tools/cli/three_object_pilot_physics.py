"""Admit pilot bases, retaining failures, then execute complete counterfactual groups."""
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from tools.cli.dataset_generation import generation_code_sha256
from tools.core.json_io import frozen_json
from tools.core.hashing import sha256_file
from tools.core.json_io import read_json
from tools.motion_rules.three_object.interaction import audit_hard_results
from tools.sampling.derive_physics_sweep import _load_prior_indexes,resolve_prior_provenance
from tools.sampling.three_object_sweeps import derive_group
from tools.release.source_release import publish_source_release
from tools.physics.run_pybullet_batch import worker_context


def candidate_group(job):
    """Own one disjoint candidate directory; never allocate sources in workers."""
    from tools.cli.generate_three_object_dataset import simulation
    root,layout,record,resume=job
    path=Path(record['metadata_path'])
    if sha256_file(path)!=record['metadata_sha256']:raise ValueError('candidate changed after metadata validation')
    candidate=read_json(path)
    physical=simulation(path,layout.base_dataset/'physics'/record['scene_id'],root,resume,require_admission=False)
    audit=read_json(Path(physical['audit_path']))
    integrity=all(audit_hard_results(candidate['simulation']['objects'],audit['adapter_audit'],True)) and all(
        c['passed'] for c in audit['checks'] if c['id']!='adapter_hard_invariants')
    decision={'scene_id':record['scene_id'],'cell':record['cell'],'audit_passed':physical['audit_passed'],
              'integrity_passed':integrity,'physics':physical,
              'semantic_failures':[c['id'] for c in audit['adapter_audit']['checks'] if c['category']=='base_semantics' and not c['passed']]}
    frozen_json(root/'outputs'/layout.base_dataset.parent.name/'admission'/f'{record["scene_id"]}.json',decision)
    if not integrity:raise ValueError(f'pilot base integrity failure requires investigation: {record["scene_id"]}')
    if not physical['audit_passed']:return decision,None,[],[]
    base={'scene_id':record['scene_id'],'metadata_path':path.relative_to(root).as_posix(),'metadata_sha256':sha256_file(path),
          'source_schema_version':candidate['schema_version'],
          'family':{'physweep_billiards_three_object_scene_v1':'billiards',
                    'physweep_passive_pinball_three_object_scene_v1':'passive_pinball',
                    'physweep_marble_run_three_object_scene_v1':'marble_run'}.get(candidate['schema_version'],'generic'),
          'audit_passed':True,'physics':physical}
    config_path=layout.base_dataset.parent/'inputs/resolved_sweep.json';config=read_json(config_path)
    profiles,registry=_load_prior_indexes(root,config)
    members=derive_group(candidate,path,root,config,config_path,profiles,registry);metadata_records=[];physics_records=[]
    for member in members:
        member_path=layout.sweep_metadata/member['scene_id']/'metadata.json';frozen_json(member_path,member)
        sweep=member['sweep']
        metadata_records.append({'path':member_path.relative_to(root).as_posix(),'metadata_sha256':sha256_file(member_path),
            'scene_id':member['scene_id'],'parent':sweep['parent_metadata_path'],'source_schema_version':member['schema_version'],
            **{key:sweep.get(key) for key in ('kind','axis','level_index','value','target_object_id','target_object_index')}})
        result=simulation(member_path,layout.sweep_physics/member['scene_id'],root,resume)
        physics_records.append({**result,'ok':True,'kind':sweep['kind']})
    return decision,base,metadata_records,physics_records


def completed_groups(jobs,workers):
    if type(workers) is not int or workers<1:raise ValueError('physics workers must be a positive integer')
    if workers==1:
        for job in jobs:yield candidate_group(job)
        return
    # Bounded waves keep failure handling finite. A failure prevents the next
    # wave; already running independent groups may finish and remain resumable.
    with ProcessPoolExecutor(max_workers=workers,mp_context=worker_context()) as executor:
        for start in range(0,len(jobs),workers):
            yield from executor.map(candidate_group,jobs[start:start+workers])


def ready_candidate_schema_versions(records):
    """Inspect only constructible candidates; unavailable cells remain audit records."""
    return {
        read_json(Path(record['metadata_path']))['schema_version']
        for record in records
        if record.get('status')=='metadata_ready'
    }


def run_physics(root,layout,records,args):
    from tools.cli.generate_three_object_dataset import verify_source_release
    config_path=layout.base_dataset.parent/'inputs/resolved_sweep.json';config=read_json(config_path)
    plan={'schema_version':'physweep_three_object_pilot_physics_plan_v1','code_sha256':generation_code_sha256(),
          'candidate_manifest_sha256':sha256_file(layout.base_dataset/'candidates_manifest.json'),
          'sweep_config_sha256':sha256_file(config_path),'prior_sources':resolve_prior_provenance(root,config,None)}
    schema_versions=ready_candidate_schema_versions(records)
    if 'physweep_billiards_three_object_scene_v1' in schema_versions:
        from tools.physics.three_object_specialized_simulation import integrity_configuration
        plan['specialized_integrity_configuration']=integrity_configuration()[1]
    if 'physweep_passive_pinball_three_object_scene_v1' in schema_versions:
        from tools.physics.three_object_specialized_simulation import integrity_configuration
        plan['pinball_integrity_configuration']=integrity_configuration('passive_pinball_three_object_v1')[1]
    if 'physweep_marble_run_three_object_scene_v1' in schema_versions:
        from tools.physics.three_object_specialized_simulation import integrity_configuration
        plan['marble_integrity_configuration']=integrity_configuration('marble_run_three_object_v1')[1]
    output=root/'outputs'/args.work_id;frozen_json(output/'physics_plan.json',plan)
    bases=[];decisions=[];metadata_records=[];physics_records=[]
    jobs=[(root,layout,r,args.resume) for r in records if r['status']=='metadata_ready']
    for index,(decision,base,group_metadata,group_physics) in enumerate(completed_groups(jobs,args.physics_workers)):
        decisions.append(decision)
        if base is not None:bases.append(base)
        metadata_records.extend(group_metadata);physics_records.extend(group_physics)
        print(f'base admission {index+1}/{len(jobs)}; passing groups={len(bases)}; complete sweep members={len(physics_records)}',flush=True)
    if not bases:raise ValueError('pilot has no passing bases; retain all failures and investigate')
    frozen_json(layout.base_manifest,{'schema_version':'physweep_three_object_base_manifest_v1','sample_count':len(bases),'records':bases})
    metadata_manifest=layout.sweep_metadata/'manifest.json';physics_manifest=layout.sweep_physics/'manifest.json'
    frozen_json(metadata_manifest,{'schema_version':'physweep_physics_sweep_metadata_manifest_v1','sample_count':len(metadata_records),'records':metadata_records})
    frozen_json(physics_manifest,{'schema_version':'physweep_dispatched_physics_manifest_v1','sample_count':len(physics_records),'records':physics_records})
    release_path=layout.source_release/'manifest.json'
    if not release_path.exists():
        publish_source_release(root=root,base_manifest_path=layout.base_manifest,sweep_metadata_manifest_path=metadata_manifest,
            sweep_physics_manifest_path=physics_manifest,output=layout.source_release,object_count=3,
            dataset_id='physweep_three_object_pilot_v1',release_schema='physweep_three_object_source_release_v1',target_object_indices=(0,))
    verify_source_release(root,release_path,layout.base_manifest,metadata_manifest,physics_manifest,base_count=len(bases))
    counters=Counter((r['cell']['template'],r['cell']['target_role'],r['audit_passed']) for r in decisions)
    result={'status':'pilot_physics_complete_pending_camera_trial','code_sha256':generation_code_sha256(),'base_attempts':len(decisions),
            'accepted_bases':len(bases),'rejected_bases':len(decisions)-len(bases),'complete_members':len(physics_records),
            'source_release':str(release_path),'template_target_outcomes':[{'template':t,'target_role':r,'passed':p,'count':n}
                for (t,r,p),n in sorted(counters.items())]}
    frozen_json(output/'physics_result.json',result)
    return result
