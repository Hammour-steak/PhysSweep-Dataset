#!/usr/bin/env python3
"""Prepare a bounded base-selected video trial from a verified D3 source release."""
import argparse
import json
from pathlib import Path
import zipfile

from tools.cli.dataset_generation import generation_layout,generation_code_sha256,bind_generation_plan
from tools.cli.generate_three_object_dataset import verify_source_release
from tools.cli.three_object_visual_stage import prepare,finish_prepared
from tools.core.json_io import frozen_json
from tools.core.hashing import sha256_file
from tools.core.json_io import read_json
from tools.dataset_contract.three_object_group import validate_group_inputs
from tools.release.audit_release_provenance import audit_release
from tools.release.source_release import publish_source_release
from tools.rendering.three_object_camera import ThreeObjectCameraAdmissionError
from tools.sampling.three_object_video_selection import build_video_plan,final_video_coverage
from tools.sampling.three_object_sampling_request import load_pilot_rules

CODE_ROOT=Path(__file__).resolve().parents[2]


def verified_physics_gate(path):
    gate=read_json(path)
    if gate.get('stage')!='D3' or gate.get('status')!='passed':raise ValueError('video trial requires a passed D3 gate')
    for name,digest in gate['evidence'].items():
        if sha256_file(path.parent/name)!=digest:raise ValueError('D3 evidence changed')
    metadata_gates=[read_json(path.parent/name) for name in gate['evidence'] if name.endswith('gate.json')]
    metadata_gate=next(g for g in metadata_gates if g.get('stage')=='D2')
    archive=Path(metadata_gate['source_archive']['path'])
    if sha256_file(archive)!=metadata_gate['source_archive']['sha256']:raise ValueError('frozen physics source archive changed')
    # A new rendering version consumes old verified physics. It must retain the
    # numerical, identity and initial-state definitions that produced that data.
    import hashlib
    with zipfile.ZipFile(archive) as frozen:
        for name in frozen.namelist():
            if name.endswith('.py') and name.startswith(tuple('tools/'+p+'/' for p in ('physics','core','motion_rules','scene_rules','sampling','dataset_contract'))):
                if hashlib.sha256(frozen.read(name)).hexdigest()!=sha256_file(CODE_ROOT/name):raise ValueError(f'physics compatibility changed: {name}')
    release=Path(gate['source_release']['path'])
    if sha256_file(release)!=gate['source_release']['sha256']:raise ValueError('D3 source release changed')
    return gate,release


def run(args):
    root=args.root.resolve();gate,source_path=verified_physics_gate(args.physics_gate);rules=load_pilot_rules(matrix_path=getattr(args,'rules_matrix',None))
    audit_release(source_path,root);source=read_json(source_path)
    if source['object_count']!=3 or source['sweep_target_object_indices']!=[0]:raise ValueError('unexpected source object count or target')
    layout=generation_layout(root,args.work_id,Path(f'outputs/work/{args.work_id}/three_object'),object_count=3)
    output=root/'outputs'/args.work_id;plan_path=output/'generation_plan.json'
    bind_generation_plan(plan_path,{'schema_version':'physweep_three_object_video_trial_plan_v1','work_id':args.work_id,'code_sha256':generation_code_sha256(),
        'physics_gate':{'path':str(args.physics_gate.resolve()),'sha256':sha256_file(args.physics_gate)},'physics_code_sha256':gate['code_sha256'],
        'source_release':{'path':str(source_path),'sha256':sha256_file(source_path)},'object_count':3,'target_object_indices':[0],
        'rules':rules['bindings'],'rules_sha256':rules['rules_sha256']},args.resume)
    source_bases=read_json(root/source['base_manifest'])['records'];base_by_id={r['scene_id']:r for r in source_bases}
    scenes={}
    for row in source_bases:
        path=root/row['metadata_path']
        if sha256_file(path)!=row['metadata_sha256']:raise ValueError('admitted base metadata changed')
        scenes[row['scene_id']]=read_json(path)
        if scenes[row['scene_id']]['three_object']['rules_sha256']!=rules['rules_sha256']:raise ValueError('video rules differ from admitted physics')
    selection=build_video_plan(list(scenes.values()),geometry_scope=rules['matrix'].get('geometry_scope','sphere_flat_v1'));frozen_json(output/'video_selection_plan.json',selection)
    # Sweep inputs are opened only after the base-only candidate order freezes.
    metadata=read_json(root/source['metadata_manifest'])['records'];physics=read_json(root/source['physics_manifest'])['records'];physics_by_id={r['scene_id']:r for r in physics}
    bases=[];selected_metadata=[];selected_physics=[];samples=[];decisions=[];used=set()
    for slot in selection['slots']:
        accepted=None
        for scene_id in slot['candidate_order']:
            if scene_id in used:continue
            base=base_by_id[scene_id];parent=(root/base['metadata_path']).resolve()
            group=[r for r in metadata if (root/r['parent']).resolve()==parent]
            validate_group_inputs(scenes[scene_id],[read_json(root/r['path']) for r in group],[0])
            physical_group=[physics_by_id[r['scene_id']] for r in group]
            failure_path=output/'camera_rejections'/f'{scene_id}.json'
            if failure_path.exists():
                failure=read_json(failure_path)
                if failure['metadata_sha256']!=base['metadata_sha256']:raise ValueError('rejected camera input changed')
                decisions.append({'slot':slot['slot_id'],'scene_id':scene_id,'status':'rejected_camera','reason':failure['reason']});continue
            try:
                _,bound=prepare(root,layout,base['physics'],physical_group,args.resume,group_key=scene_id)
            except ThreeObjectCameraAdmissionError as error:
                failure={'metadata_sha256':base['metadata_sha256'],'reason':str(error)};frozen_json(failure_path,failure)
                decisions.append({'slot':slot['slot_id'],'scene_id':scene_id,'status':'rejected_camera','reason':str(error)});continue
            accepted=scene_id;used.add(scene_id);bases.append(base);selected_metadata.extend(group);selected_physics.extend(physical_group);samples.extend(bound['samples'])
            decisions.append({'slot':slot['slot_id'],'scene_id':scene_id,'status':'accepted_camera'});break
        if accepted is None:decisions.append({'slot':slot['slot_id'],'status':'unfilled_no_cross_template_reallocation'})
    frozen_json(output/'camera_admission.json',{'requested_groups':selection['requested_groups'],'accepted_groups':len(bases),'records':decisions,
        'input_visual_coverage':final_video_coverage([scenes[r['scene_id']] for r in bases]),'final_camera_coverage_location':'base bound metadata visualization.camera.diagnostics'})
    if len(bases)!=selection['requested_groups']:raise ValueError('video trial has unfilled camera slots; inspect before rendering')
    frozen_json(layout.base_manifest,{'schema_version':'physweep_three_object_base_manifest_v1','sample_count':len(bases),'records':bases})
    mpath=layout.sweep_metadata/'manifest.json';ppath=layout.sweep_physics/'manifest.json'
    frozen_json(mpath,{'schema_version':'physweep_physics_sweep_metadata_manifest_v1','sample_count':len(selected_metadata),'records':selected_metadata})
    frozen_json(ppath,{'schema_version':'physweep_dispatched_physics_manifest_v1','sample_count':len(selected_physics),'records':selected_physics})
    release=layout.source_release/'manifest.json'
    if not release.exists():publish_source_release(root=root,base_manifest_path=layout.base_manifest,sweep_metadata_manifest_path=mpath,sweep_physics_manifest_path=ppath,
        output=layout.source_release,object_count=3,dataset_id='physweep_three_object_pilot_v1',release_schema='physweep_three_object_source_release_v1',target_object_indices=(0,))
    verify_source_release(root,release,layout.base_manifest,mpath,ppath,base_count=len(bases))
    manifest={'schema_version':'physweep_pybullet_sweep_bound_manifest_v1','dataset_id':'physweep_three_object_pilot_v1','source_manifest':str(ppath),
        'output_root':str(layout.sweep_render.relative_to(root)),'sample_count':len(samples),'samples':samples}
    path=layout.sweep_render/'bound_manifest.json';frozen_json(path,manifest)
    result=finish_prepared(root,layout,args,plan_path,path,manifest)
    # Cache hits and elapsed times can differ on a successful resume. Freeze
    # the verified artifact identities, not those per-invocation statistics.
    frozen_json(output/f'{args.checkpoint}_result.json',{'status':result['status'],'checkpoint':args.checkpoint,
        'code_sha256':generation_code_sha256(),'group_count':len(bases),'sample_count':len(samples),
        'bound_manifest':{'path':str(path),'sha256':sha256_file(path)},
        'source_release':{'path':str(release),'sha256':sha256_file(release)}})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--physics-gate',type=Path,required=True)
    p.add_argument('--work-id',required=True);p.add_argument('--checkpoint',choices=('visual','render','publish'),default='visual')
    p.add_argument('--gpus');p.add_argument('--blender',type=Path);p.add_argument('--resume',action='store_true')
    p.add_argument('--rules-matrix',type=Path)
    print(json.dumps(run(p.parse_args())))


if __name__=='__main__':main()
