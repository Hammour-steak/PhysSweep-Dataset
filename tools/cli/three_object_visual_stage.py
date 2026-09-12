"""Pilot visual checkpoints composed from existing binders, renderer and publisher."""
from pathlib import Path
import copy
import subprocess
import shutil
import sys
import numpy as np

from tools.assets.three_object_resources import validate_resources
from tools.cli.dataset_generation import generation_code_sha256,verify_render_manifest
from tools.cli.build_three_object_dataset import publish_dataset,verify_dataset
from tools.core.hashing import sha256_file
from tools.core.json_io import read_json, frozen_json
from tools.dataset_contract.trajectory_contract import object_trajectory_view
from tools.release.base_release_view import PipelineSpec
from tools.rendering.bind_pybullet_visuals import bind_scene
from tools.rendering.bind_physics_sweep_visuals import bind_one
from tools.rendering.render_pybullet_manifest import reusable_render_record
from tools.rendering.three_object_camera import audit_camera

CODE_ROOT=Path(__file__).resolve().parents[2]


def checked_bound(root,record,physical):
    path=root/record['metadata_path']
    if sha256_file(path)!=record['metadata_sha256']:raise ValueError('bound metadata changed')
    bound=read_json(path)
    for field,key in (('source_metadata','metadata'),('trajectory','trajectory')):
        if (root/bound[field]['path']).resolve()!=Path(physical[key+'_path']).resolve() or bound[field]['sha256']!=physical[key+'_sha256']:
            raise ValueError('visual input differs from admitted physics')
    simulation_path=Path(physical['trajectory_path']).with_name('simulation_record.json')
    if (root/bound['simulation_record']['path']).resolve()!=simulation_path.resolve() or bound['simulation_record']['sha256']!=sha256_file(simulation_path):
        raise ValueError('bound simulation record changed')
    validate_resources(root,bound,bound['visualization']['resource_binding'])
    return bound


def visual_payload(bound):
    visual=copy.deepcopy(bound['visualization'])
    for key in ('binding_version','camera_inheritance'):visual.pop(key,None)
    for key in ('video_path','inspection_frame_dir','instance_mask_dir'):visual['render'].pop(key,None)
    return visual


def _audit_bound_camera(root, bound, trajectory, camera):
    """Keep exact-support camera rays bound to the same resource root as the scene."""
    return audit_camera(bound, trajectory, camera, root=root)


def prepare(root,layout,physical,physics_records,resume,*,group_key=None):
    if group_key is not None:
        from tools.core.paths import safe_scene_id
        group_key=safe_scene_id(group_key)
    base_result=layout.base_render/(f'bindings/{group_key}.json' if group_key else 'binding_result.json')
    if base_result.exists():
        if not resume:raise FileExistsError(base_result)
        base=read_json(base_result)
    else:
        base=bind_scene(root,Path(physical['metadata_path']),Path(physical['trajectory_path']).with_name('simulation_record.json'),
                        Path(physical['trajectory_path']),layout.base_render,{},None,None)
        frozen_json(base_result,base)
    base_bound=checked_bound(root,base,physical)
    members=[];diagnostics=[]
    for physical_member in physics_records:
        scene_id=physical_member['scene_id'];result_path=layout.sweep_render/'bindings'/f'{scene_id}.json'
        if result_path.exists():
            if not resume:raise FileExistsError(result_path)
            record=read_json(result_path)
        else:
            sample={**physical_member,'simulation_record_path':str(Path(physical_member['trajectory_path']).with_name('simulation_record.json'))}
            record=bind_one(root,sample,{base['scene_id']:base},layout.sweep_render)
            frozen_json(result_path,record)
        bound=checked_bound(root,record,physical_member)
        if visual_payload(bound)!=visual_payload(base_bound):raise ValueError('group visual inputs changed')
        with np.load(physical_member['trajectory_path']) as arrays:
            trajectory=object_trajectory_view(bound,{key:arrays[key] for key in arrays.files})
        # Diagnostic only: a physically valid counterfactual can leave the frame.
        try:
            visibility=_audit_bound_camera(
                root, bound, trajectory, base_bound['visualization']['camera']
            )
        except ValueError as error:
            if physical_member['kind']=='base':raise
            visibility={'status':'not_measurable','reason':str(error),'policy':'diagnostic_only'}
        diagnostics.append({'scene_id':scene_id,'kind':physical_member['kind'],'camera_audit':visibility})
        members.append(record)
    manifest={'schema_version':'physweep_pybullet_sweep_bound_manifest_v1','dataset_id':'physweep_three_object_pilot_v1',
              'source_manifest':str(layout.sweep_physics/'manifest.json'),
              'output_root':str(layout.sweep_render.relative_to(root)),'sample_count':len(members),'samples':members}
    evidence_root=layout.sweep_render/'groups'/group_key if group_key else layout.sweep_render
    path=evidence_root/'bound_manifest.json';frozen_json(path,manifest)
    frozen_json(evidence_root/'visibility_diagnostics.json',{'policy':'diagnostic_only','records':diagnostics})
    return path,manifest


def execute(root,layout,physical,physics_records,args,plan_path):
    bound_path,manifest=prepare(root,layout,physical,physics_records,args.resume)
    return finish_prepared(root,layout,args,plan_path,bound_path,manifest)


def unfinished_render(existing,samples,bound_path,root):
    """Accept a recorded renderer failure only for the same complete input set."""
    records=existing.get('records');expected={s['scene_id'] for s in samples}
    if (existing.get('schema_version')!='physweep_pybullet_render_manifest_v1'
        or existing.get('render_scope')!='full_animation'
        or existing.get('source_manifest_sha256')!=sha256_file(bound_path)
        or (root/str(existing.get('source_manifest',''))).resolve()!=bound_path.resolve()
        or not isinstance(records,list) or len(records)!=len(samples)
        or existing.get('sample_count')!=len(samples)
        or {r.get('scene_id') for r in records}!=expected or len(expected)!=len(samples)
        or any(type(r.get('ok')) is not bool for r in records)):
        raise ValueError('render manifest differs from frozen inputs')
    success=sum(r['ok'] for r in records);failed=len(records)-success
    if existing.get('success_count')!=success or existing.get('failure_count')!=failed:
        raise ValueError('render outcome counters are inconsistent')
    return failed>0


def finish_prepared(root,layout,args,plan_path,bound_path,manifest):
    """Render and publish any explicitly prepared collection of complete groups."""
    count=manifest['sample_count']
    if count<13 or count%13 or len(manifest['samples'])!=count:raise ValueError('render input lacks complete three-object groups')
    if args.checkpoint=='visual':return {'status':'visual_inputs_ready','bound_manifest':str(bound_path)}
    if not args.gpus:raise ValueError('render checkpoint requires explicit --gpus')
    gpus=[int(value) for value in args.gpus.split(',')]
    if len(gpus)!=len(set(gpus)) or min(gpus)<0:raise ValueError('invalid GPU list')
    blender=(args.blender or root/'runtime/blender-3.4.0-linux-x64/blender').resolve()
    execution_plan={'schema_version':'physweep_three_object_render_plan_v1','code_sha256':generation_code_sha256(),
        'generation_plan':{'path':str(plan_path),'sha256':sha256_file(plan_path)},
        'bound_manifest':{'path':str(bound_path),'sha256':sha256_file(bound_path)},
        'blender':{'path':str(blender),'sha256':sha256_file(blender)},'gpus':gpus,'workers':len(gpus),
        'source_release_sha256':sha256_file(layout.source_release/'manifest.json')}
    frozen_json(layout.sweep_render/'render_plan.json',execution_plan)
    render_manifest=layout.sweep_render/'render_manifest.json'
    pending=not render_manifest.exists()
    if render_manifest.exists():
        if not args.resume:raise FileExistsError(render_manifest)
        existing=read_json(render_manifest)
        pending=unfinished_render(existing,manifest['samples'],bound_path,root)
        if pending:
            digest=sha256_file(render_manifest);history=layout.sweep_render/'failed_render_manifests'/f'{digest}.json'
            history.parent.mkdir(parents=True,exist_ok=True)
            if not history.exists():shutil.copyfile(render_manifest,history)
            if sha256_file(history)!=digest:raise ValueError('failed render evidence changed')
        else:
            verify_render_manifest(render_manifest,count)
            records={r['scene_id']:r for r in existing['records']}
            for index,sample in enumerate(manifest['samples']):
                path=root/sample['metadata_path']
                if not reusable_render_record(root,layout.sweep_render,sample,path,read_json(path),records[sample['scene_id']]['render_record'],
                                              False,gpus[index%len(gpus)],CODE_ROOT/'tools/rendering/render_pybullet_rigid.py'):
                    raise ValueError('render checkpoint artifact changed')
    if pending:
        command=[sys.executable,'-B','-m','tools.rendering.render_pybullet_manifest','--root',str(root),
                 '--manifest',str(bound_path),'--blender',str(blender),'--workers',str(len(gpus)),'--gpus',args.gpus]
        if args.resume:command.append('--resume')
        subprocess.run(command,cwd=CODE_ROOT,check=True)
        verify_render_manifest(render_manifest,count)
    if args.checkpoint=='render':return {'status':'rendered_pending_publication','render_manifest':str(render_manifest)}
    result=publish_dataset(release_project_root=root,release_manifest=layout.source_release/'manifest.json',
        release_root=layout.canonical_release,pipeline_specs=[PipelineSpec('generic','physweep_pybullet_rigid_metadata_v1',root,layout.sweep_render)],
        workers=min(len(gpus),4),resume=args.resume)
    verified=verify_dataset(layout.canonical_release)
    return {'status':'pilot_published_and_verified','release_root':str(layout.canonical_release),'publication':result,'verification':verified}
