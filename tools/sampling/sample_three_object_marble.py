"""Compose source-preserving marble metadata; physics and camera admission are later."""
import copy,random
from pathlib import Path
from tools.core.hashing import sha256_file,sha256_json
from tools.core.json_io import read_json
from tools.core.paths import safe_scene_id
from tools.core.marble_fixture_layout import construct_marble_fixture
from tools.dataset_contract.object_identity_contract import attach_object_identity
from tools.assets.visual_environment_binding import choose_specialized_environment
from tools.motion_rules.three_object.marble import IDS,ROLES,SCHEMA,initial_states,validate_marble_contract


def localize_marble_source_rows(rows,source_root,data_root,collision_directory):
    """Copy verified collision contents into the new work; retain original bindings."""
    collision_directory=Path(collision_directory)
    collision_directory.resolve().relative_to(data_root.resolve())
    result=copy.deepcopy(rows)
    for row in result:
        bindings=[]
        for c in row['metadata']['physics']['fixture']['mesh_components']:
            original=copy.deepcopy(c['collision']);source=source_root/original['path']
            if sha256_file(source)!=original['sha256']:raise ValueError('original marble collision hash mismatch')
            target=collision_directory/(original['sha256']+'.obj')
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                if sha256_file(target)!=original['sha256']:raise ValueError('localized marble collision changed')
            else:
                with target.open('xb') as output:output.write(source.read_bytes())
                if sha256_file(target)!=original['sha256']:raise ValueError('marble collision copy failed')
            c['collision']['path']=target.relative_to(data_root).as_posix()
            bindings.append({'component_id':c['id'],'original':{'path':str(source),'sha256':original['sha256']},'localized':copy.deepcopy(c['collision'])})
        row['fixture_resource_localization']=bindings
    return result


def load_marble_rules(code_root,data_root):
    path=code_root/'configs/three_object_marble_rules.json';config=read_json(path)
    if config['schema_version'] not in {'physweep_three_object_marble_rules_v1','physweep_three_object_marble_rules_v2'}:raise ValueError('unsupported marble rules')
    if (config['schema_version']=='physweep_three_object_marble_rules_v2')!=('fixture_layout' in config):raise ValueError('marble fixture layout requires explicit v2 rules')
    bindings={'marble':{'path':str(path),'sha256':sha256_file(path)}};documents={}
    for key in ('motion_rules','scene_rules'):
        binding=config[key];p=code_root/binding['path']
        if sha256_file(p)!=binding['sha256']:raise ValueError('shared marble rule reference changed')
        documents[key]=read_json(p);bindings[key]={'path':str(p),'sha256':binding['sha256']}
    for key,binding in config['visual_sources'].items():
        p=data_root/binding['path']
        if sha256_file(p)!=binding['sha256']:raise ValueError('marble shared visual source changed')
        documents[key]=read_json(p);bindings[key]={'path':str(p),'sha256':binding['sha256']}
    return {'config':config,**documents,'bindings':bindings,'rules_sha256':sha256_json(bindings)}


def build_three_object_marble_scene(*,data_root,host_source,object_sources,rules,scene_id,role_order,palette_order,parameters,seeds,requested_view,background_profile):
    config=rules['config']
    if len(object_sources)!=3 or len({r['source']['scene_id'] for r in object_sources})!=3:raise ValueError('three distinct marble source lineages required')
    for row in [host_source,*object_sources]:
        m=row['metadata']
        if (m['schema_version']!=config['source_schema'] or m['semantics']['dynamic_object_count']!=1
            or ('sweep' in m and m['sweep'].get('kind')!='base')):
            raise ValueError('marble requires single-object base generation sources')
        binding=m['physics']['backend_config']
        if sha256_file(data_root/binding['path'])!=binding['sha256']:raise ValueError('marble backend source changed')
    if sorted(role_order)!=sorted(ROLES) or sorted(palette_order)!=sorted(config['appearance_palette']):raise ValueError('complete marble role and palette assignment required')
    scene=copy.deepcopy(host_source['metadata'])
    for key in ('sweep','admission','object_identity','simulation_record','outputs','implementation','camera'):
        scene.pop(key,None)
    radius=object_sources[0]['metadata']['simulation']['objects'][0]['collision_proxy']['radius_m']
    physics=scene['physics'];states=initial_states(radius,config['initial_limits'],parameters)
    for component in physics['fixture']['mesh_components']:
        for binding in (component['collision'],{'path':component['source_path'],'sha256':component['source_sha256']}):
            if sha256_file(data_root/binding['path'])!=binding['sha256']:raise ValueError('marble geometry source changed')
    construction=None
    if 'fixture_layout' in config:
        original=copy.deepcopy(physics['fixture'])
        physics['fixture']=construct_marble_fixture(original,config['fixture_layout'])
        construction={'source_fixture':original,'source_fixture_sha256':sha256_json(original),
            'layout':copy.deepcopy(config['fixture_layout']),'result_fixture_sha256':sha256_json(physics['fixture'])}
    objects=[]
    for oid,role,palette,source in zip(IDS,role_order,palette_order,object_sources):
        obj=copy.deepcopy(source['metadata']['simulation']['objects'][0])
        if obj['collision_proxy']!={'type':'sphere','radius_m':radius}:raise ValueError('marble pilot requires equal source-sized spheres')
        obj.update(object_id=oid,semantic_type='marble',geometry={'type':'sphere','size_m':[2*radius]*3},initial_state=copy.deepcopy(states[role]))
        obj['visual']['color_rgba']=copy.deepcopy(config['appearance_palette'][palette])
        obj['visual_profile']={'kind':'declared_procedural_sphere_color','recipe_id':palette,'recipe_binding':copy.deepcopy(rules['bindings']['marble']),
            'source_visual':copy.deepcopy(source['metadata']['simulation']['objects'][0]['visual']),
            'metallic':.68,'roughness':.24}
        objects.append(obj)
    scene.update(schema_version=SCHEMA,scene_id=safe_scene_id(scene_id),dataset_id='physweep_three_object_marble_pilot_v1',dataset_stage='three_object_base_candidate',seed=seeds['physics'])
    scene['simulation']['objects']=objects
    for key in ('quality','two_object_quality','trajectory_path','audit_path','simulation_record_path','initial_track_offset_m'):physics.pop(key,None)
    physics['backend']='pybullet_marble_run_three_object_v1'
    scene['semantics']={'scene_family':'marble_run','dynamic_object_count':3,'profile':physics['profile'],'motion_profile':'ordered_contacts',
        'description':'Three separated marbles start along the inclined initial channel. P initially moves toward Q and R, which start at rest. Gravity acts on all three; contact order does not imply causal transfer.'}
    media=config['media'];scene['simulation']['time'].update({k:media[k] for k in ('duration_s','output_fps','frame_count','simulation_hz')})
    scene['render']={'engine':media['render_engine'],'resolution':copy.deepcopy(media['resolution']),'samples':media['samples'],'use_motion_blur':False}
    anchor=[.7,0.,1.1];reference={'position_m':[.7,3.8,2.0],'target_m':anchor}
    profile=next(p for p in rules['scene_profiles']['profiles'] if p['id']==background_profile)
    environment=choose_specialized_environment(data_root,family_id='marble_run',background_contract=config['background_contract'],
        scene_profile=profile,camera=reference,scene_anchor_m=anchor,hdri_records=rules['hdri_manifest']['records'],visual_rules=rules['visual_sampling'],rng=random.Random(seeds['appearance']))
    environment['sources']={k:copy.deepcopy(rules['bindings'][k]) for k in config['visual_sources']}
    scene['render']['environment']=environment
    if requested_view not in {v['id'] for v in config['camera_rules']['view_families']}:raise ValueError('unknown marble camera request')
    scene['camera_request']={'policy':'three_object_base_events','requested_view_family':requested_view}
    template=copy.deepcopy(config['motion_template'])
    thresholds=copy.deepcopy(rules['motion_rules']['thresholds']);thresholds['first_event_min_s']=config['first_event_min_s']
    c={'schema_version':'physweep_three_object_marble_motion_contract_v1','template':template,'roles':dict(zip(role_order,IDS)),
       'thresholds':thresholds,'rules_sha256':rules['rules_sha256'],'initial_limits':copy.deepcopy(config['initial_limits']),'initial_parameters':copy.deepcopy(parameters),
       'camera_rules':copy.deepcopy(config['camera_rules']),'appearance_palette':copy.deepcopy(config['appearance_palette']),
       'fixture_sha256':sha256_json(physics['fixture']),'minimum_path_length_per_object_m':config['initial_limits']['minimum_path_length_per_object_m']}
    if construction is not None:
        c.update(schema_version='physweep_three_object_marble_motion_contract_v2',fixture_construction=copy.deepcopy(construction))
    scene['three_object']={**c,'contract_sha256':sha256_json(c)}
    scene['semantic_rules']=copy.deepcopy(rules['bindings']['marble'])
    scene['source_binding']={'objects':[copy.deepcopy(s['source']) for s in object_sources],'host':copy.deepcopy(host_source['source']),
        'rules':copy.deepcopy(rules['bindings']),'seeds':copy.deepcopy(seeds),'appearance_assignment':{'policy':'explicit_procedural_color_only_v1','object_recipes':dict(zip(IDS,palette_order)),
        'scope':'Shared specialized sphere shader, source radius and physical materials unchanged. Three color recipes, zero new texture images. Original white source visuals retained as provenance.'}}
    scene['source_binding']['fixture_resource_localization']=copy.deepcopy(host_source['fixture_resource_localization'])
    if construction is not None:scene['source_binding']['fixture_construction']=copy.deepcopy(construction)
    validate_marble_contract(scene);attach_object_identity(scene)
    return scene
