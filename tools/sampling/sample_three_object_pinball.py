"""Compose source-preserving pinball metadata; physics and camera admission are later."""
import copy,random
from tools.core.hashing import sha256_file,sha256_json
from tools.core.json_io import read_json
from tools.core.paths import safe_scene_id
from tools.dataset_contract.object_identity_contract import attach_object_identity
from tools.assets.visual_environment_binding import choose_specialized_environment
from tools.motion_rules.three_object.pinball import IDS,ROLES,SCHEMA,initial_states,validate_pinball_contract


def load_pinball_rules(code_root,data_root):
    path=code_root/'configs/three_object_pinball_rules.json';config=read_json(path)
    if config['schema_version']!='physweep_three_object_pinball_rules_v1':raise ValueError('unsupported pinball rules')
    bindings={'pinball':{'path':str(path),'sha256':sha256_file(path)}};documents={}
    for key in ('motion_rules','scene_rules'):
        binding=config[key];p=code_root/binding['path']
        if sha256_file(p)!=binding['sha256']:raise ValueError('shared pinball rule reference changed')
        documents[key]=read_json(p);bindings[key]={'path':str(p),'sha256':binding['sha256']}
    for key,binding in config['visual_sources'].items():
        p=data_root/binding['path']
        if sha256_file(p)!=binding['sha256']:raise ValueError('pinball shared visual source changed')
        documents[key]=read_json(p);bindings[key]={'path':str(p),'sha256':binding['sha256']}
    return {'config':config,**documents,'bindings':bindings,'rules_sha256':sha256_json(bindings)}


def build_three_object_pinball_scene(*,data_root,host_source,object_sources,rules,scene_id,role_order,palette_order,parameters,seeds,requested_view,background_profile):
    config=rules['config']
    if len(object_sources)!=3 or len({r['source']['scene_id'] for r in object_sources})!=3:raise ValueError('three distinct pinball source lineages required')
    for row in [host_source,*object_sources]:
        m=row['metadata']
        if (m['schema_version']!=config['source_schema'] or m['semantics']['dynamic_object_count']!=1
            or ('sweep' in m and m['sweep'].get('kind')!='base')):
            raise ValueError('pinball requires single-object base generation sources')
        binding=m['physics']['backend_config']
        if sha256_file(data_root/binding['path'])!=binding['sha256']:raise ValueError('pinball backend source changed')
    if sorted(role_order)!=sorted(ROLES) or sorted(palette_order)!=sorted(config['appearance_palette']):raise ValueError('complete pinball role and palette assignment required')
    scene=copy.deepcopy(host_source['metadata'])
    for key in ('sweep','admission','object_identity','simulation_record','outputs','implementation','camera'):
        scene.pop(key,None)
    radius=object_sources[0]['metadata']['simulation']['objects'][0]['collision_proxy']['radius_m']
    physics=scene['physics'];states=initial_states(radius,physics['fixture_source'],physics['fixture']['frame'],config['initial_limits'],parameters)
    objects=[]
    for oid,role,palette,source in zip(IDS,role_order,palette_order,object_sources):
        obj=copy.deepcopy(source['metadata']['simulation']['objects'][0])
        if obj['collision_proxy']!={'type':'sphere','radius_m':radius}:raise ValueError('pinball pilot requires equal source-sized spheres')
        obj.update(object_id=oid,semantic_type='pinball',geometry={'type':'sphere','size_m':[2*radius]*3},initial_state=copy.deepcopy(states[role]))
        obj['visual']['color_rgba']=copy.deepcopy(config['appearance_palette'][palette])
        obj['visual_profile']={'kind':'declared_procedural_sphere_color','recipe_id':palette,'recipe_binding':copy.deepcopy(rules['bindings']['pinball']),
            'source_visual':copy.deepcopy(source['metadata']['simulation']['objects'][0]['visual']),
            'metallic':.68,'roughness':.24}
        objects.append(obj)
    scene.update(schema_version=SCHEMA,scene_id=safe_scene_id(scene_id),dataset_id='physweep_three_object_pinball_pilot_v1',dataset_stage='three_object_base_candidate',seed=seeds['physics'])
    scene['simulation']['objects']=objects
    for key in ('quality','two_object_quality','trajectory_path','audit_path','simulation_record_path'):physics.pop(key,None)
    scene['semantics']={'scene_family':'passive_pinball','dynamic_object_count':3,'profile':physics['profile'],'motion_profile':'pair_control',
        'description':'One ball initially approaches a stationary ball across the upper pinfield; a third ball starts down a separate path. Gravity acts on all three.'}
    media=config['media'];scene['simulation']['time'].update({k:media[k] for k in ('duration_s','output_fps','frame_count')})
    scene['render']={'engine':media['render_engine'],'resolution':copy.deepcopy(media['resolution']),'samples':media['samples'],'use_motion_blur':False}
    anchor=[0.,0.,1.35];reference={'position_m':[0.,3.,1.8],'target_m':anchor}
    profile=next(p for p in rules['scene_profiles']['profiles'] if p['id']==background_profile)
    environment=choose_specialized_environment(data_root,family_id='passive_pinball',background_contract=config['background_contract'],
        scene_profile=profile,camera=reference,scene_anchor_m=anchor,hdri_records=rules['hdri_manifest']['records'],visual_rules=rules['visual_sampling'],rng=random.Random(seeds['appearance']))
    environment['sources']={k:copy.deepcopy(rules['bindings'][k]) for k in config['visual_sources']}
    scene['render']['environment']=environment
    if requested_view not in {v['id'] for v in config['camera_rules']['view_families']}:raise ValueError('unknown pinball camera request')
    scene['camera_request']={'policy':'three_object_base_events','requested_view_family':requested_view}
    template=copy.deepcopy(next(t for t in rules['motion_rules']['templates'] if t['id']=='pair_control'))
    template.update(constructor='pinfield_early_pair_with_separate_down_motion',caption_template='three_object_pinball_initial_v1')
    thresholds=copy.deepcopy(rules['motion_rules']['thresholds']);thresholds['first_event_min_s']=config['first_event_min_s']
    c={'schema_version':'physweep_three_object_pinball_motion_contract_v1','template':template,'roles':dict(zip(role_order,IDS)),
       'thresholds':thresholds,'rules_sha256':rules['rules_sha256'],'initial_limits':copy.deepcopy(config['initial_limits']),'initial_parameters':copy.deepcopy(parameters),
       'camera_rules':copy.deepcopy(config['camera_rules']),'appearance_palette':copy.deepcopy(config['appearance_palette']),
       'fixture_sha256':sha256_json({'fixture':physics['fixture'],'fixture_source':physics['fixture_source']}),'minimum_path_length_per_object_m':.8}
    scene['three_object']={**c,'contract_sha256':sha256_json(c)}
    scene['semantic_rules']=copy.deepcopy(rules['bindings']['pinball'])
    scene['source_binding']={'objects':[copy.deepcopy(s['source']) for s in object_sources],'host':copy.deepcopy(host_source['source']),
        'rules':copy.deepcopy(rules['bindings']),'seeds':copy.deepcopy(seeds),'appearance_assignment':{'policy':'explicit_procedural_color_only_v1','object_recipes':dict(zip(IDS,palette_order)),
        'scope':'Shared specialized sphere shader, source radius and physical materials unchanged. Three color recipes, zero new texture images. Original white source visuals retained as provenance.'}}
    validate_pinball_contract(scene);attach_object_identity(scene)
    return scene
