"""A bounded conditional coverage plan and deterministic candidate construction."""
from collections import Counter,defaultdict
import itertools
import numpy as np

from tools.core.hashing import sha256_json
from tools.core.sweep_values import sweep_values
from tools.sampling.three_object_sampling_request import candidate_seed
from tools.sampling.sample_three_object_base import build_three_object_scene
from tools.sampling.three_object_fingerprints import physics_fingerprint,visual_fingerprint
from tools.assets.three_object_resources import resource_snapshot
from tools.core.primitive_support import initial_extents

ROLES=('P','Q','R')
SCALE_PATTERNS=(('small',)*3,('medium',)*3,('large',)*3,('small','medium','large'),('large','medium','small'))


def source_scale(row):
    return row['metadata']['semantic_sampling']['five_dimensions']['foreground_object']['scale_bin']


def environment_category(row):
    return row['metadata']['appearance']['scene_visual']['environment_category']


def build_plan(pool,rules):
    if rules['matrix'].get('coverage_mode')=='multi_mesh_pair_roles_v1':return multi_mesh_role_plan(pool,rules)
    if rules['matrix'].get('coverage_mode')=='motion_diversity_v1':return motion_diversity_plan(pool,rules)
    if rules['matrix'].get('coverage_mode')=='exact_mesh_host_v1':return mesh_host_plan(pool,rules)
    if rules['matrix'].get('coverage_mode')=='primitive_asset_proxy_v1':return primitive_asset_plan(pool,rules)
    if rules['matrix'].get('coverage_mode')=='compound_asset_proxy_v1':return compound_asset_plan(pool,rules)
    if rules['matrix'].get('coverage_mode')=='all_medium_visual_assets_v1':return visual_asset_plan(pool,rules)
    if rules['matrix'].get('geometry_scope')=='sphere_cross_slope_pair_v1':return inclined_plan(pool,rules)
    if rules['matrix'].get('geometry_scope')=='mixed_flat_R_v1':return mixed_plan(pool,rules)
    hosts=defaultdict(list)
    for row in pool['hosts']:hosts[row['metadata']['simulation']['support']['scene_class']].append(row)
    scale_counts=Counter(source_scale(row) for row in pool['objects'])
    cells=[]
    families=[f['id'] for f in rules['scene']['camera']['view_families']]
    for template_index,template in enumerate(rules['matrix']['template_ids']):
        for host_index,host in enumerate(sorted(hosts)):
            categories=sorted({environment_category(row) for row in hosts[host]})
            for role_index,target_role in enumerate(ROLES):
                for scale_index,scales in enumerate(SCALE_PATTERNS):
                    cell={'template':template,'host_class':host,'target_role':target_role,
                          'shape_by_role':dict.fromkeys(ROLES,'sphere'),'scale_by_role':dict(zip(ROLES,scales)),
                          'requested_environment_category':categories[(template_index+role_index+scale_index)%len(categories)],
                          'requested_camera_family':families[(template_index+host_index+role_index+scale_index)%len(families)]}
                    cell['cell_id']=template+'_'+sha256_json(cell)[:12]
                    cell['capacity_condition']=all(scale_counts[scale]>=scales.count(scale) for scale in scales)
                    cells.append(cell)
    cells.sort(key=lambda c:c['cell_id'])
    if len(cells)!=rules['matrix']['candidate_budget']:
        raise ValueError('available pilot cells differ from the frozen 120-candidate budget; revise the explicit allocation plan')
    return {'schema_version':'physweep_three_object_pilot_coverage_v1','candidate_count':len(cells),'cells':cells,
            'construction':{'spacing_ratio':[1.6,2.8],'ballistic_first_contact_time_s':[.28,.6],
                            'first_contact_offset_ratio':[.45,.7],
                            'multi_contact_templates':['chain_transfer','converging_hits','successive_hits'],
                            'multi_contact_speed_m_s':[1.1,1.3],
                            'multi_contact_nominal_first_contact_time_s':[.26,.34],
                            'multi_contact_spacing_ratio':list(rules['motion']['initial_state']['separation_in_pair_diameters']),
                            'parameter_policy':'multi-contact speed and nominal time jointly determine bounded spacing; pair control retains its shorter layout; full physics admission required'},
            'scale_bin_meaning':'inherited source relative-scale category; actual SI diameters are reported separately',
            'unallocated_scale_patterns':[list(p) for p in itertools.product(('small','medium','large'),repeat=3) if p not in SCALE_PATTERNS],
            'scope_limitations':['sphere proxies and two flat host classes only','five selected scale patterns, not all 27',
                                 'initial +X reference only; mirrored directions require a later explicit scope extension',
                                 'source-bound physics and appearance are jointly selected; no claim of statistical independence'],
            'source_capacity':pool['summary'],'source_scale_counts':dict(scale_counts)}


def candidates(pool,rules,plan,root,work_id,sweep_config):
    object_pools={(shape,scale):[r for r in pool['objects'] if source_scale(r)==scale and r['metadata']['simulation']['objects'][0]['geometry']['type']==shape] for shape in ('sphere','cuboid','cylinder') for scale in ('small','medium','large')}
    host_pools=defaultdict(list)
    for row in pool['hosts']:host_pools[(row['metadata']['simulation']['support']['scene_class'],environment_category(row))].append(row)
    object_uses=Counter();host_uses=Counter();seen=set()
    for index,cell in enumerate(plan['cells']):
        attempts=[];scene=None
        physics_rng=np.random.default_rng(candidate_seed(rules['matrix']['master_seed'],cell['cell_id'],index,'physics'))
        appearance_rng=np.random.default_rng(candidate_seed(rules['matrix']['master_seed'],cell['cell_id'],index,'appearance'))
        choices=host_pools[(cell['host_class'],cell['requested_environment_category'])]
        if 'required_support_asset_id' in cell:
            choices=[row for row in choices if row['metadata']['simulation']['support'].get('asset_id')==cell['required_support_asset_id']]
        if 'required_host_source_scene_id' in cell:
            choices=[row for row in choices if row['source']['scene_id']==cell['required_host_source_scene_id']]
        for attempt in range(rules['matrix']['maximum_source_attempts_per_candidate']):
            if not cell['capacity_condition'] or not choices:break
            target=cell['target_role'];remaining=[r for r in ROLES if r!=target]
            if attempt%2:remaining.reverse()
            role_order=[target,*remaining]
            role_pools=[bound_object_pool(object_pools,cell,role) for role in role_order]
            if any(not p for p in role_pools):break
            objects=[p[int(physics_rng.integers(len(p)))] for p in role_pools]
            host=choices[int(appearance_rng.integers(len(choices)))]
            ids=[r['source']['scene_id'] for r in objects];host_id=host['source']['scene_id']
            attempt_record={'attempt':attempt,'object_source_ids':ids,'host_source_id':host_id}
            try:
                if len(set(ids))!=3:raise ValueError('duplicate source scene within candidate')
                if any(object_uses[i]>=rules['matrix']['source_budgets']['maximum_object_uses'] for i in ids):raise ValueError('object source budget exhausted')
                if host_uses[host_id]>=rules['matrix']['source_budgets']['maximum_host_uses']:raise ValueError('host source budget exhausted')
                source_objects={role:objects[j]['metadata']['simulation']['objects'][0] for j,role in enumerate(role_order)}
                radius={role:source_objects[role]['geometry']['size_m'][0]/2 for role in ROLES}
                parameters=plan['construction']
                if rules['matrix'].get('coverage_mode')=='multi_mesh_pair_roles_v1':
                    speed,spacing,offset,desired_time=_multi_mesh_initial_parameters(source_objects,cell,rules,parameters)
                else:
                    spacing=float(physics_rng.uniform(*parameters['spacing_ratio']))
                    offset=float(physics_rng.uniform(*parameters['first_contact_offset_ratio']))
                    pair_radius=radius['P']+radius['Q']
                    contact_x=np.sqrt(max(0.,pair_radius**2-(radius['P']-radius['Q'])**2-
                                      (offset*pair_radius if cell['template']=='successive_hits' else 0.)**2))
                    first_gap=spacing*pair_radius-contact_x
                    desired_time=float(physics_rng.uniform(*parameters['ballistic_first_contact_time_s']))
                    speed=float(np.clip(first_gap/desired_time,*rules['motion']['initial_state']['initial_speed_m_s']))
                if rules['matrix'].get('coverage_mode')=='exact_mesh_host_v1':
                    if 'initial_speed_m_s' in cell:
                        spacing=float(cell['spacing_ratio']);speed=float(cell['initial_speed_m_s'])
                        first_gap=spacing*pair_radius-contact_x
                        desired_time=float(cell['nominal_first_contact_time_s'])
                        if abs(first_gap/speed-desired_time)>1e-12:
                            raise ValueError('exact mesh timing cell contradicts its source geometry')
                    else:
                        speed=float(physics_rng.uniform(*parameters['initial_speed_m_s']))
                        desired_time=first_gap/speed
                elif cell['template'] in parameters['multi_contact_templates']:
                    speed=float(physics_rng.uniform(*parameters['multi_contact_speed_m_s']))
                    desired_time=float(physics_rng.uniform(*parameters['multi_contact_nominal_first_contact_time_s']))
                    spacing=float(np.clip((speed*desired_time+contact_x)/pair_radius,*parameters['multi_contact_spacing_ratio']))
                if 'initial_speed_m_s' in cell:speed=float(cell['initial_speed_m_s'])
                if 'spacing_ratio' in cell:spacing=float(cell['spacing_ratio'])
                if 'offset_ratio' in cell:offset=float(cell['offset_ratio'])
                candidate=build_three_object_scene(host_source=host,object_sources=objects,rules=rules,
                    scene_id=f'{work_id}_{cell["cell_id"]}',candidate_index=index,role_order=role_order,
                    template_id=cell['template'],speed_m_s=speed,spacing_ratio=spacing,
                    offset_ratio=offset,requested_view_family=cell['requested_camera_family'],sampling_cell_id=cell['cell_id'],
                    motion_axis=cell.get('motion_axis','world_x'),layout_variant=cell.get('layout_variant','axis_aligned'),root=root)
                # Generic three-object rules use the common scalar domains, with
                # no motion-result bound and a relative mass intervention.
                if candidate.get('physics',{}).get('sweep_domains'):raise ValueError('unexpected specialized sweep domain on a generic candidate')
                for axis,axis_rules in sweep_config['axes'].items():
                    sweep_values(candidate['simulation']['objects'][0]['material'][axis],axis_rules,None,axis,
                                 endpoint_policy=sweep_config['endpoint_policy'])
                fingerprint=physics_fingerprint(candidate)
                if fingerprint in seen:raise ValueError('duplicate physical initial state after removing identity labels')
                candidate['visual_resource_binding']=resource_snapshot(root,candidate)
                candidate['coverage']={'cell':cell,'candidate_index':index,'source_attempt':attempt,
                    'physics_sha256':fingerprint,'visual_sha256':visual_fingerprint(candidate),
                    'actual_diameter_m_by_role':{role:2*radius[role] for role in ROLES},
                    'requested_ballistic_first_contact_time_s':desired_time}
                if rules['matrix'].get('geometry_scope') in {'mixed_flat_R_v1','motion_diversity_v1','multi_mesh_pair_roles_v1'}:
                    candidate['coverage'].pop('actual_diameter_m_by_role')
                    candidate['coverage']['actual_size_m_by_role']={role:objects[j]['metadata']['simulation']['objects'][0]['geometry']['size_m'] for j,role in enumerate(role_order)}
                if rules['matrix'].get('coverage_mode')=='multi_mesh_pair_roles_v1':
                    candidate['coverage'].pop('requested_ballistic_first_contact_time_s')
                    candidate['coverage']['requested_first_contact_time_s']=desired_time
                    candidate['coverage']['initial_gap_model']='sphere_ballistic_or_supported_sliding_constant_coulomb_v1'
                scene=candidate;seen.add(fingerprint);object_uses.update(ids);host_uses[host_id]+=1
                attempt_record['status']='accepted_metadata'
            except (ValueError,FileNotFoundError) as error:
                attempt_record.update(status='rejected_metadata',reason=str(error))
            attempts.append(attempt_record)
            if scene is not None:break
        yield {'cell':cell,'candidate_index':index,'scene':scene,'attempts':attempts,
               'status':'metadata_ready' if scene is not None else 'metadata_unavailable_no_reallocation'}


def _mesh_layout_size(trio,template,motion_axis,spacing=1.5,edge_factor=1.5):
    radii=[float(row['metadata']['simulation']['objects'][0]['geometry']['size_m'][0])/2 for row in trio]
    p,q,r=radii;x=[-spacing*(p+q),0.,spacing*(q+r)];y=[0.,0.,0.]
    if template=='pair_control':x[2]=x[0];y[2]=spacing*(r+max(p,q))
    half=[edge_factor*value for value in radii]
    width=max(value+radius for value,radius in zip(x,half))-min(value-radius for value,radius in zip(x,half))
    height=max(value+radius for value,radius in zip(y,half))-min(value-radius for value,radius in zip(y,half))
    return (width,height) if motion_axis=='world_x' else (height,width)


def _mesh_maximum_spacing(trio,template,motion_axis,available):
    low,high=1.5,4.0
    if any(_mesh_layout_size(trio,template,motion_axis,low)[axis]>available[axis]+1e-12 for axis in range(2)):
        return None
    if all(_mesh_layout_size(trio,template,motion_axis,high)[axis]<=available[axis]+1e-12 for axis in range(2)):
        return high
    for _ in range(80):
        middle=(low+high)/2
        if all(_mesh_layout_size(trio,template,motion_axis,middle)[axis]<=available[axis]+1e-12 for axis in range(2)):
            low=middle
        else:high=middle
    return low


def _mesh_timing_parameters(trio,template,motion_axis,available,rules):
    policy=rules['matrix'].get('exact_mesh_timing_policy')
    if policy is None:return None
    maximum_spacing=_mesh_maximum_spacing(trio,template,motion_axis,available)
    if maximum_spacing is None:return None
    reserve=float(policy['extra_capacity_reserve_fraction'])
    usable_spacing=1.5+(1.-reserve)*(maximum_spacing-1.5)
    radii=[float(row['metadata']['simulation']['objects'][0]['geometry']['size_m'][0])/2 for row in trio]
    p,q,_=radii;pair_radius=p+q;contact_x=np.sqrt(max(0.,pair_radius**2-(p-q)**2))
    target_time=float(policy['nominal_first_contact_time_s'])
    maximum_speed=float(policy['maximum_initial_speed_m_s_by_template'][template])
    speed=min(maximum_speed,(usable_spacing*pair_radius-contact_x)/target_time)
    minimum_speed=float(rules['motion']['initial_state']['initial_speed_m_s'][0])
    if speed<minimum_speed-1e-12:return None
    speed=max(speed,minimum_speed)
    spacing=max(1.5,(speed*target_time+contact_x)/pair_radius)
    nominal_time=(spacing*pair_radius-contact_x)/speed
    required_size=_mesh_layout_size(trio,template,motion_axis,spacing)
    if (spacing>usable_spacing+1e-12 or nominal_time<target_time-1e-12
            or any(required_size[axis]>available[axis]+1e-12 for axis in range(2))):return None
    return {'initial_speed_m_s':float(speed),'spacing_ratio':float(spacing),
            'nominal_first_contact_time_s':float(nominal_time),
            'maximum_spacing_ratio':float(maximum_spacing),'usable_spacing_ratio':float(usable_spacing),
            'required_layout_size_xy_m':list(required_size)}


def mesh_host_plan(pool,rules):
    """Freeze one source-compatible sphere scene for every admitted mesh host."""
    assignments=rules['matrix']['mesh_host_assignments'];required=rules['matrix']['required_support_asset_ids']
    hosts={row['metadata']['simulation']['support'].get('asset_id'):row for row in pool['hosts']
           if row['metadata']['simulation']['support'].get('asset_id') in set(required)}
    if set(hosts)!=set(required):raise ValueError('exact mesh host pool differs from the frozen assignment')
    sphere_pools={scale:sorted([
        row for row in pool['objects'] if row['source']['source_family']=='generic'
        and source_scale(row)==scale
        and row['metadata']['simulation']['objects'][0]['geometry']['type']=='sphere'],
        key=lambda row:(row['metadata']['simulation']['objects'][0]['geometry']['size_m'][0],row['source']['scene_id']))
        for scale in ('small','medium')}
    used_sources=set();cells=[];views=[row['id'] for row in rules['scene']['camera']['view_families']]
    for index,assignment in enumerate(assignments):
        host=hosts[assignment['asset_id']];support=host['metadata']['simulation']['support']
        bounds=support['safe_surface_bounds'];available=(bounds['x'][1]-bounds['x'][0],bounds['y'][1]-bounds['y'][0])
        available_rows=[row for row in sphere_pools[assignment['scale']] if row['source']['scene_id'] not in used_sources]
        selected=None;selected_masses=None;timing_parameters=None
        policy=rules['matrix'].get('exact_mesh_timing_policy')
        for trio in itertools.permutations(available_rows[:64],3):
            masses=[float(row['metadata']['simulation']['objects'][0]['material']['mass_kg']) for row in trio]
            if max(masses)/min(masses)>rules['motion']['initial_state']['maximum_role_mass_ratio']:continue
            if (assignment['template']=='chain_transfer' and policy is not None
                    and masses[0]/masses[1]<policy['chain_transfer_minimum_driver_to_receiver_mass_ratio']):continue
            timing=_mesh_timing_parameters(trio,assignment['template'],assignment['motion_axis'],available,rules)
            if rules['matrix'].get('exact_mesh_timing_policy') is not None and timing is None:continue
            required_size=(_mesh_layout_size(trio,assignment['template'],assignment['motion_axis'])
                           if timing is None else timing['required_layout_size_xy_m'])
            if all(required_size[axis]<=available[axis]+1e-12 for axis in range(2)):
                selected=trio;selected_masses=masses;timing_parameters=timing;break
        if selected is None or selected_masses is None:raise ValueError(f"no frozen source trio fits exact mesh host: {assignment['asset_id']}")
        source_ids={role:selected[role_index]['source']['scene_id'] for role_index,role in enumerate(ROLES)}
        used_sources.update(source_ids.values())
        environment=environment_category(host);target=ROLES[index%len(ROLES)]
        cell={'template':assignment['template'],'host_class':'raised_flat','target_role':target,
              'shape_by_role':dict.fromkeys(ROLES,'sphere'),'scale_by_role':dict.fromkeys(ROLES,assignment['scale']),
              'required_source_family_by_role':dict.fromkeys(ROLES,'generic'),
              'required_source_scene_id_by_role':source_ids,
              'required_support_asset_id':assignment['asset_id'],'motion_axis':assignment['motion_axis'],
              'requested_environment_category':environment,'requested_camera_family':views[index%len(views)],
              'capacity_condition':True,'required_layout_size_xy_m':list(required_size)}
        if timing_parameters is not None:cell.update(timing_parameters)
        if assignment['template']=='chain_transfer' and timing_parameters is not None:
            cell['driver_to_receiver_mass_ratio']=float(selected_masses[0]/selected_masses[1])
        cell['cell_id']=assignment['template']+'_'+sha256_json(cell)[:12];cells.append(cell)
    cells.sort(key=lambda cell:cell['cell_id'])
    if len(cells)!=rules['matrix']['candidate_budget'] or len(used_sources)!=3*len(cells):
        raise ValueError('exact mesh host plan count or source reuse differs from the frozen policy')
    result={'schema_version':'physweep_three_object_pilot_coverage_v1','candidate_count':len(cells),'cells':cells,
        'construction':{'spacing_ratio':[1.5,1.5],'ballistic_first_contact_time_s':[.28,.6],
                        'first_contact_offset_ratio':[.45,.7],'initial_speed_m_s':[1.1,1.3],
                        'multi_contact_templates':[]},
        'assignment_counts':{'chain_transfer':sum(row['template']=='chain_transfer' for row in assignments),
                             'pair_control':sum(row['template']=='pair_control' for row in assignments),
                             'world_x':sum(row['motion_axis']=='world_x' for row in assignments),
                             'world_y':sum(row['motion_axis']=='world_y' for row in assignments),
                             'small':sum(row['scale']=='small' for row in assignments),
                             'medium':sum(row['scale']=='medium' for row in assignments)},
        'scope_limitations':['One bounded metadata cell per 19 released exact-mesh hosts.',
            'Only released generic spheres; 14 medium trios and five small trios selected without resizing.',
            'Each host uses one predeclared compatible chain-transfer or pair-control template at spacing 1.5.',
            'World-X and world-Y are explicit rigid rotations; exact mesh placement, physics and camera evidence remain required.',
            'Metadata coverage does not establish base physics, camera or media admission.'],
        'source_capacity':pool['summary']}
    if rules['matrix'].get('exact_mesh_timing_policy') is not None:
        result['construction']={
            'spacing_ratio':[1.5,4.0],
            'ballistic_first_contact_time_s':[.21,.21],
            'first_contact_offset_ratio':[.45,.7],
            'initial_speed_m_s':[.3,1.1],
            'multi_contact_templates':[],
            'exact_mesh_timing_policy':rules['matrix']['exact_mesh_timing_policy'],
            'parameter_policy':'Per-cell speed and spacing are fixed from source geometry and reserved safe-surface capacity before physics.',
        }
        counts=result['assignment_counts'];excluded=rules['matrix'].get('excluded_support_asset_ids',[])
        result['scope_limitations']=[
            f'One bounded metadata cell per {len(cells)} admitted released exact-mesh hosts.',
            f"Only released generic spheres; {counts['medium']} medium trios and {counts['small']} small trios selected without resizing.",
            'Each cell fixes speed and spacing from geometry/capacity so nominal first contact is at least 0.21 seconds.',
            'World-X and world-Y are explicit rigid rotations; exact mesh placement, physics and camera evidence remain required.',
            f'Explicitly excluded after preserved base diagnostics: {excluded}.',
            'Metadata coverage does not establish base physics, camera or media admission.',
        ]
        result['excluded_support_asset_ids']=list(excluded)
    return result


def mixed_plan(pool,rules):
    """24 cells: two templates, two terminal/control shapes, two hosts, three targets."""
    hosts=defaultdict(list)
    for row in pool['hosts']:hosts[row['metadata']['simulation']['support']['scene_class']].append(row)
    cells=[];families=[f['id'] for f in rules['scene']['camera']['view_families']]
    for ti,template in enumerate(('chain_transfer','pair_control')):
        for si,shape in enumerate(('cuboid','cylinder')):
            for hi,host in enumerate(sorted(hosts)):
                categories=sorted({environment_category(row) for row in hosts[host]})
                for ri,role in enumerate(ROLES):
                    cell={'template':template,'host_class':host,'target_role':role,'shape_by_role':{'P':'sphere','Q':'sphere','R':shape},
                          'scale_by_role':dict.fromkeys(ROLES,'medium'),'requested_environment_category':categories[(ti+si+ri)%len(categories)],
                          'requested_camera_family':families[(ti+si+hi+ri)%len(families)]}
                    cell['cell_id']=template+'_'+sha256_json(cell)[:12]
                    cell['capacity_condition']=all(sum(source_scale(r)=='medium' and r['metadata']['simulation']['objects'][0]['geometry']['type']==s for r in pool['objects'])>=n for s,n in Counter(cell['shape_by_role'].values()).items())
                    cells.append(cell)
    cells.sort(key=lambda c:c['cell_id'])
    if len(cells)!=rules['matrix']['candidate_budget']:raise ValueError('mixed extension capacity differs from frozen 24-cell budget')
    return {'schema_version':'physweep_three_object_pilot_coverage_v1','candidate_count':len(cells),'cells':cells,
      'construction':{'spacing_ratio':[1.6,2.8],'ballistic_first_contact_time_s':[.28,.6],'first_contact_offset_ratio':[.45,.7],
        'multi_contact_templates':['chain_transfer'],'multi_contact_speed_m_s':[1.1,1.3],
        'multi_contact_nominal_first_contact_time_s':[.26,.34],'multi_contact_spacing_ratio':list(rules['motion']['initial_state']['separation_in_pair_diameters'])},
      'scope_limitations':['R-only mixed shapes; other shapes/roles are not admitted','upright primitive source pose retained','one medium source-scale pattern','flat support and inherited appearance only'],
      'source_capacity':pool['summary']}


def bound_object_pool(object_pools,cell,role):
    """A predeclared asset target is a constraint, never a fallback preference."""
    pool=object_pools[(cell['shape_by_role'][role],cell['scale_by_role'][role])]
    bindings=cell.get('required_visual_asset_by_role',{})
    families=cell.get('required_source_family_by_role',{})
    source_ids=cell.get('required_source_scene_id_by_role',{})
    if not isinstance(bindings,dict) or not set(bindings)<=set(ROLES) or any(not isinstance(v,str) or not v for v in bindings.values()):
        raise ValueError('invalid role asset binding')
    if not isinstance(families,dict) or not set(families)<=set(ROLES) or any(v not in {'generic','asset'} for v in families.values()):
        raise ValueError('invalid role source-family binding')
    if not isinstance(source_ids,dict) or not set(source_ids)<=set(ROLES) or any(not isinstance(v,str) or not v for v in source_ids.values()):
        raise ValueError('invalid role source-scene binding')
    if role in families:pool=[row for row in pool if row['source']['source_family']==families[role]]
    if role in bindings:pool=[r for r in pool if r['metadata']['simulation']['objects'][0]['visual_profile']['id']==bindings[role]]
    if role in source_ids:pool=[r for r in pool if r['source']['scene_id']==source_ids[role]]
    return pool


def _source_mass(row):
    return float(row['metadata']['simulation']['objects'][0]['material']['mass_kg'])


def _median_log_mass(rows):
    return float(np.median(np.log([_source_mass(row) for row in rows])))


def _mass_compatible_rows(rows_by_role,maximum_ratio):
    best=None
    for trio in itertools.product(*(rows_by_role[role] for role in ROLES)):
        ids=tuple(row['source']['scene_id'] for row in trio)
        if len(set(ids))!=3:continue
        masses=[_source_mass(row) for row in trio];ratio=max(masses)/min(masses)
        if ratio>maximum_ratio:continue
        key=(ratio,max(masses)-min(masses),ids)
        if best is None or key<best[0]:best=(key,trio)
    return None if best is None else dict(zip(ROLES,best[1]))


def _pair_is_mass_compatible(left,right,maximum_ratio):
    return any(max(_source_mass(a),_source_mass(b))/min(_source_mass(a),_source_mass(b))<=maximum_ratio
        for a in left for b in right)


def _multi_mesh_initial_parameters(source_objects,cell,rules,construction):
    axis_index=0 if cell.get('motion_axis','world_x')=='world_x' else 1
    extents={role:float(initial_extents(source_objects[role])[axis_index]) for role in ROLES}
    contact_span=extents['P']+extents['Q']
    if source_objects['P']['geometry']['type']!='sphere':
        desired_time=float(construction['sliding_nominal_first_contact_time_s'])
        friction=float(source_objects['P']['material']['contact_friction'])
        arrival_speed=float(construction['minimum_sliding_arrival_speed_m_s'])
        minimum_gap=.5*contact_span
        speed=max(float(construction['minimum_sliding_initial_speed_m_s']),
            friction*9.81*desired_time+arrival_speed,
            (minimum_gap+.5*friction*9.81*desired_time**2)/desired_time)
        speed=min(speed,float(construction['maximum_sliding_initial_speed_m_s']))
        gap=speed*desired_time-.5*friction*9.81*desired_time**2
        spacing=float(1.+gap/contact_span)
    else:
        desired_time=float(construction['nominal_first_contact_time_s'])
        maximum_speed=float(construction['maximum_initial_speed_m_s_by_template'][cell['template']])
        low_speed,high_speed=rules['motion']['initial_state']['initial_speed_m_s']
        speed=float(np.clip(3.*contact_span/desired_time,low_speed,min(high_speed,maximum_speed)))
        spacing=float(1.+speed*desired_time/contact_span)
    if not rules['motion']['initial_state']['separation_in_pair_diameters'][0]<=spacing<=rules['motion']['initial_state']['separation_in_pair_diameters'][1]:
        raise ValueError('multi-mesh source geometry cannot satisfy the frozen first-contact timing')
    return speed,spacing,.25,desired_time


def multi_mesh_role_plan(pool,rules):
    """Use every released generic visual and rotate every eligible nonball asset through P/Q/R."""
    matrix=rules['matrix']
    if matrix.get('geometry_scope')!='multi_mesh_pair_roles_v1' or matrix.get('coverage_mode')!='multi_mesh_pair_roles_v1':
        raise ValueError('multi-mesh coverage is limited to the explicit pair-control scope')
    required_assets=matrix['required_asset_ids'];required_generic=matrix['required_generic_visual_asset_ids']
    variants=defaultdict(set);generic_variants=defaultdict(set);asset_rows=defaultdict(list);generic_rows=defaultdict(list);hosts=defaultdict(list)
    for row in pool['objects']:
        obj=row['metadata']['simulation']['objects'][0];visual=obj['visual_profile']['id']
        value=(obj['geometry']['type'],source_scale(row))
        if row['source']['source_family']=='asset' and visual in set(required_assets):
            variants[visual].add(value);asset_rows[visual].append(row)
        if row['source']['source_family']=='generic' and visual in set(required_generic):
            generic_variants[visual].add(value)
            if source_scale(row)=='medium':generic_rows[visual].append(row)
    if set(variants)!=set(required_assets) or any(len(values)!=1 for values in variants.values()):
        raise ValueError('eligible nonball asset variants differ from the frozen multi-mesh scope')
    if any(next(iter(values))[0]=='sphere' for values in variants.values()):
        raise ValueError('multi-mesh asset targets must be nonball assets')
    if set(generic_variants)!=set(required_generic):raise ValueError('generic visual inventory differs from the frozen 84-asset scope')
    generic_shape={}
    for visual,values in generic_variants.items():
        shapes={shape for shape,_ in values}
        if len(shapes)!=1 or (next(iter(shapes)),'medium') not in values:
            raise ValueError('generic visual lacks one unambiguous medium proxy shape')
        generic_shape[visual]=next(iter(shapes))
    counts=Counter(generic_shape.values())
    if counts!={'sphere':22,'cuboid':35,'cylinder':27}:
        raise ValueError('generic visual proxy counts differ from the released 1obj inventory')
    for row in pool['hosts']:hosts[row['metadata']['simulation']['support']['scene_class']].append(row)
    if set(hosts)!={'ground_flat','raised_flat'}:raise ValueError('multi-mesh trial requires both flat host classes')
    views=[view['id'] for view in rules['scene']['camera']['view_families']]
    if len(views)!=3:raise ValueError('multi-mesh trial requires the three admitted camera families')
    construction={'spacing_ratio':[1.5,4.0],'ballistic_first_contact_time_s':[.3,.3],
        'first_contact_offset_ratio':[.25,.25],'multi_contact_templates':[],
        'nominal_first_contact_time_s':.3,'maximum_initial_speed_m_s_by_template':{'pair_control':.8},
        'sliding_nominal_first_contact_time_s':.22,'minimum_sliding_arrival_speed_m_s':.25,
        'minimum_sliding_initial_speed_m_s':.8,'maximum_sliding_initial_speed_m_s':2.,
        'parameter_policy':'Spheres retain a 0.3-second ballistic gap. Non-sphere P uses its released friction in a constant-Coulomb estimate for a 0.22-second contact with at least 0.25 m/s arrival speed; complete physics admission remains required.'}
    specs=[]
    for asset_index,asset in enumerate(required_assets):
        shape,scale=next(iter(variants[asset]))
        for role_index,asset_role in enumerate(ROLES):
            index=len(specs)
            specs.append({'template':'pair_control',
                'asset_id':asset,'asset_role':asset_role,'asset_shape':shape,'asset_scale':scale,'index':index})
    for index in range(12):
        specs.append({'template':'pair_control','index':len(specs)})
    if len(specs)!=matrix['candidate_budget']:raise ValueError('multi-mesh specification count differs from the frozen budget')
    maximum_ratio=float(rules['motion']['initial_state']['maximum_role_mass_ratio'])
    log_mass={visual:_median_log_mass(rows) for visual,rows in generic_rows.items()}
    unassigned=set(required_generic);assignments={};selected_rows={};used_once=set();repeated=set()
    asset_specs=[spec for spec in specs if 'asset_id' in spec]
    compatible_count={asset:sum(_pair_is_mass_compatible(asset_rows[asset],generic_rows[visual],maximum_ratio)
        for visual in required_generic) for asset in required_assets}
    asset_specs.sort(key=lambda spec:(compatible_count[spec['asset_id']],spec['asset_id'],spec['asset_role']))
    for spec in asset_specs:
        index=spec['index'];asset=spec['asset_id'];asset_role=spec['asset_role'];open_roles=[role for role in ROLES if role!=asset_role]
        target_log=_median_log_mass(asset_rows[asset])
        ranked=sorted(unassigned,key=lambda visual:(abs(log_mass[visual]-target_log),visual))
        pairs=sorted(itertools.combinations(ranked,2),key=lambda pair:(
            abs(log_mass[pair[0]]-target_log)+abs(log_mass[pair[1]]-target_log),pair))
        chosen=None
        for pair in pairs:
            if sum(generic_shape[visual]=='sphere' for visual in pair)>1:continue
            visuals={asset_role:asset,open_roles[0]:pair[0],open_roles[1]:pair[1]}
            rows_by_role={role:(asset_rows[visuals[role]] if role==asset_role else generic_rows[visuals[role]]) for role in ROLES}
            rows=_mass_compatible_rows(rows_by_role,maximum_ratio)
            if rows is not None:chosen=(pair,visuals,rows);break
        if chosen is None:raise ValueError(f'no mass-compatible generic visual pair for eligible asset: {asset}')
        pair,visuals,rows=chosen
        for role in ROLES:assignments[(index,role)]=visuals[role];selected_rows[(index,role)]=rows[role]
        unassigned.difference_update(pair);used_once.update(pair)
    generic_specs=sorted((spec for spec in specs if 'asset_id' not in spec),key=lambda spec:spec['index'])
    for spec in generic_specs:
        index=spec['index'];ranked=sorted(unassigned,key=lambda visual:(log_mass[visual],visual));chosen=None
        pairs=sorted(itertools.combinations(ranked,2),key=lambda pair:(abs(log_mass[pair[0]]-log_mass[pair[1]]),pair))
        for pair in pairs:
            sphere_count=sum(generic_shape[visual]=='sphere' for visual in pair)
            if sphere_count>1:continue
            target_log=.5*(log_mass[pair[0]]+log_mass[pair[1]])
            companions=sorted(used_once-repeated-set(pair),key=lambda visual:(abs(log_mass[visual]-target_log),visual))
            for companion in companions:
                if sphere_count+(generic_shape[companion]=='sphere')>1:continue
                visuals=dict(zip(ROLES,(*pair,companion)))
                rows=_mass_compatible_rows({role:generic_rows[visual] for role,visual in visuals.items()},maximum_ratio)
                if rows is not None:chosen=(pair,companion,visuals,rows);break
            if chosen is not None:break
        if chosen is None:raise ValueError('remaining generic visuals cannot form a mass-compatible multi-mesh cell')
        pair,companion,visuals,rows=chosen
        for role in ROLES:assignments[(index,role)]=visuals[role];selected_rows[(index,role)]=rows[role]
        unassigned.difference_update(pair);used_once.update(pair);repeated.add(companion)
    if unassigned or len(used_once)!=84 or len(repeated)!=12:
        raise ValueError('mass-compatible allocation did not consume the frozen generic visual inventory')
    cells=[];selected_host_uses=Counter()
    for spec in specs:
        index=spec['index'];bindings={};families={};shapes={};scales={}
        for role in ROLES:
            if role==spec.get('asset_role'):
                bindings[role]=spec['asset_id'];families[role]='asset';shapes[role]=spec['asset_shape'];scales[role]=spec['asset_scale']
            else:
                visual=assignments[(index,role)];bindings[role]=visual;families[role]='generic'
                shapes[role]=generic_shape[visual];scales[role]='medium'
        if len(set(bindings.values()))!=3 or sum(shape=='sphere' for shape in shapes.values())>1:
            raise ValueError('multi-mesh cell must have distinct visuals and at most one sphere')
        desired_host=('ground_flat','raised_flat')[index%2]
        target_role=ROLES[index%3];remaining=[role for role in ROLES if role!=target_role];role_order=[target_role,*remaining]
        source_objects={role:selected_rows[(index,role)]['metadata']['simulation']['objects'][0] for role in ROLES}
        probe_cell={'template':spec['template'],'motion_axis':('world_x','world_y')[index%2]}
        speed,spacing,offset,_=_multi_mesh_initial_parameters(source_objects,probe_cell,rules,construction)
        host_candidates=[]
        for host_class in (desired_host,('raised_flat' if desired_host=='ground_flat' else 'ground_flat')):
            host_candidates.extend(sorted(hosts[host_class],key=lambda row:(selected_host_uses[row['source']['scene_id']],row['source']['scene_id'])))
        selected_host=None
        for host_row in host_candidates:
            try:
                build_three_object_scene(host_source=host_row,
                    object_sources=[selected_rows[(index,role)] for role in role_order],rules=rules,
                    scene_id=f'multi_mesh_host_probe_{index:03d}',candidate_index=index,role_order=role_order,
                    template_id=spec['template'],speed_m_s=speed,spacing_ratio=spacing,offset_ratio=offset,
                    requested_view_family=views[index%len(views)],motion_axis=probe_cell['motion_axis'])
            except ValueError:
                continue
            selected_host=host_row;break
        if selected_host is None:raise ValueError('no released flat host fits a mass-compatible multi-mesh cell')
        host=selected_host['metadata']['simulation']['support']['scene_class'];selected_host_uses[selected_host['source']['scene_id']]+=1
        source_ids={role:selected_rows[(index,role)]['source']['scene_id'] for role in ROLES}
        cell={'template':spec['template'],'layout_variant':'axis_aligned','motion_axis':('world_x','world_y')[index%2],
            'host_class':host,'target_role':target_role,'shape_by_role':shapes,'scale_by_role':scales,
            'required_visual_asset_by_role':bindings,'required_source_family_by_role':families,
            'required_source_scene_id_by_role':source_ids,
            'required_host_source_scene_id':selected_host['source']['scene_id'],
            'requested_environment_category':environment_category(selected_host),
            'requested_camera_family':views[index%len(views)],'capacity_condition':True}
        cell['cell_id']='multi_mesh_'+sha256_json(cell)[:12];cells.append(cell)
    cells.sort(key=lambda cell:cell['cell_id'])
    visual_uses=Counter(value for cell in cells for role,value in cell['required_visual_asset_by_role'].items()
        if cell['required_source_family_by_role'][role]=='generic')
    asset_role_uses=Counter((cell['required_visual_asset_by_role'][role],role) for cell in cells for role in ROLES
        if cell['required_source_family_by_role'][role]=='asset')
    if set(visual_uses)!=set(required_generic) or set(visual_uses.values())!={1,2} or sum(value==2 for value in visual_uses.values())!=12:
        raise ValueError('generic visual allocation does not cover all 84 assets exactly as frozen')
    if set(asset_role_uses)!={(asset,role) for asset in required_assets for role in ROLES} or set(asset_role_uses.values())!={1}:
        raise ValueError('eligible asset allocation does not cover every role exactly once')
    return {'schema_version':'physweep_three_object_pilot_coverage_v1','candidate_count':len(cells),'cells':cells,
        'construction':construction,
        'generic_visual_usage_counts':dict(sorted(visual_uses.items())),
        'eligible_asset_role_usage_counts':[{'asset_id':asset,'role':role,'count':asset_role_uses[(asset,role)]}
            for asset in required_assets for role in ROLES],
        'scope_limitations':['Diagnostic 42-base trial only; no production quota.',
            'Axis-aligned pair-control on flat supports only.',
            'Successive, converging, angled, crossing and inclined layouts retain their existing sphere-only compatibility.',
            'Every released generic visual appears at least once; all ten eligible true nonball assets appear once in each role.'],
        'source_capacity':pool['summary']}


def motion_diversity_plan(pool,rules):
    """Sixteen diagnostic cells for visibly non-collinear motion, without a quota claim."""
    hosts=defaultdict(list)
    for row in pool['hosts']:hosts[row['metadata']['simulation']['support']['scene_class']].append(row)
    if set(hosts)!={'ground_flat','raised_flat'}:raise ValueError('motion diversity trial requires both flat host classes')
    views=[view['id'] for view in rules['scene']['camera']['view_families']]
    specs=[]
    for axis in ('world_x','world_y'):
        for side in ('left','right'):
            specs.append(('chain_transfer',f'angled_chain_{side}',axis,'sphere'))
            specs.append(('successive_hits',f'offset_successive_{side}',axis,'sphere'))
            specs.append(('pair_control',f'crossing_control_{side}',axis,'sphere'))
    specs.extend([
        ('pair_control','crossing_control_left','world_x','cuboid'),
        ('pair_control','crossing_control_right','world_y','cuboid'),
        ('pair_control','crossing_control_right','world_x','cylinder'),
        ('pair_control','crossing_control_left','world_y','cylinder'),
    ])
    if len(specs)!=rules['matrix']['candidate_budget']:raise ValueError('motion diversity trial requires exactly sixteen cells')
    available=Counter((row['metadata']['simulation']['objects'][0]['geometry']['type'],source_scale(row)) for row in pool['objects'])
    primitive_scale={shape:next((scale for scale in ('medium','small','large') if available[(shape,scale)]),None)
                     for shape in ('cuboid','cylinder')}
    cells=[]
    for index,(template,variant,axis,primitive_shape) in enumerate(specs):
        host='ground_flat' if template=='pair_control' else sorted(hosts)[index%2]
        categories=sorted({environment_category(row) for row in hosts[host]})
        shapes={'P':'sphere','Q':'sphere','R':primitive_shape}
        scales={'P':'medium','Q':'medium','R':primitive_scale[primitive_shape] if primitive_shape!='sphere' else 'medium'}
        cell={'template':template,'layout_variant':variant,'motion_axis':axis,'host_class':host,
              'target_role':ROLES[index%3],'shape_by_role':shapes,'scale_by_role':scales,
              'requested_environment_category':categories[(index//2)%len(categories)],
              'requested_camera_family':views[index%len(views)],
              'initial_speed_m_s':.8 if template=='pair_control' else (1.25 if template=='chain_transfer' else 1.15),
              'spacing_ratio':2.6 if template=='pair_control' else (2.2 if template=='chain_transfer' else 2.8),
              'offset_ratio':.25 if template=='chain_transfer' else .56}
        cell['cell_id']='motion_'+sha256_json(cell)[:12]
        needed=Counter((shapes[role],scales[role]) for role in ROLES)
        cell['capacity_condition']=all(available[key]>=count for key,count in needed.items())
        cells.append(cell)
    cells.sort(key=lambda cell:cell['cell_id'])
    return {'schema_version':'physweep_three_object_pilot_coverage_v1','candidate_count':len(cells),'cells':cells,
        'construction':{'spacing_ratio':[1.5,4.0],'ballistic_first_contact_time_s':[.28,.6],
            'first_contact_offset_ratio':[.45,.7],'multi_contact_templates':[],
            'multi_contact_speed_m_s':[1.1,1.3],'multi_contact_nominal_first_contact_time_s':[.26,.34],
            'multi_contact_spacing_ratio':list(rules['motion']['initial_state']['separation_in_pair_diameters'])},
        'scope_limitations':['diagnostic sixteen-base trial only; no production quota','flat supports and medium sources only',
            'angled chain, mirrored successive-hit and crossing-control layouts only','cuboid/cylinder motion is limited to role R in pair control'],
        'source_capacity':pool['summary']}


def visual_asset_plan(pool,rules):
    """Exercise each medium-scale visual in both already supported R roles."""
    if rules['matrix'].get('geometry_scope')!='mixed_flat_R_v1':raise ValueError('asset coverage requires mixed flat role compatibility')
    assets=defaultdict(set);all_assets=set();hosts=defaultdict(list)
    for row in pool['objects']:
        obj=row['metadata']['simulation']['objects'][0];asset=obj['visual_profile']['id'];all_assets.add(asset)
        if source_scale(row)=='medium':assets[asset].add(obj['geometry']['type'])
    if not assets or any(len(shapes)!=1 for shapes in assets.values()):raise ValueError('asset coverage requires unambiguous source shapes')
    for row in pool['hosts']:hosts[row['metadata']['simulation']['support']['scene_class']].append(row)
    if set(hosts)!={'ground_flat','raised_flat'}:raise ValueError('asset coverage requires both flat host classes')
    spheres=sum(source_scale(r)=='medium' and r['metadata']['simulation']['objects'][0]['geometry']['type']=='sphere' for r in pool['objects'])
    if spheres<3:raise ValueError('asset coverage requires three distinct medium sphere source records')
    views=[v['id'] for v in rules['scene']['camera']['view_families']];cells=[]
    for ti,template in enumerate(('chain_transfer','pair_control')):
        for ai,(asset,shapes) in enumerate(sorted(assets.items())):
            shape=next(iter(shapes));host=sorted(hosts)[(ai+ti)%2]
            categories=sorted({environment_category(row) for row in hosts[host]})
            cell={'template':template,'host_class':host,'target_role':ROLES[(ai+ti)%3],
                  'shape_by_role':{'P':'sphere','Q':'sphere','R':shape},'scale_by_role':dict.fromkeys(ROLES,'medium'),
                  'required_visual_asset_by_role':{'R':asset},
                  'requested_environment_category':categories[(ai//2+ti)%len(categories)],
                  'requested_camera_family':views[(ai//3+ti)%len(views)]}
            cell['cell_id']=template+'_'+sha256_json(cell)[:12];cell['capacity_condition']=True;cells.append(cell)
    cells.sort(key=lambda c:c['cell_id'])
    if len(cells)!=rules['matrix']['candidate_budget']:raise ValueError('asset coverage capacity differs from frozen candidate budget')
    # Reuse the established mixed constructor ranges without changing old plans.
    reference_rules={**rules,'matrix':{**rules['matrix'],'candidate_budget':24}}
    reference=mixed_plan(pool,reference_rules)
    return {**reference,'candidate_count':len(cells),'cells':cells,'required_visual_asset_ids':sorted(assets),
            'unavailable_medium_visual_asset_ids':sorted(all_assets-set(assets)),
            'scope_limitations':['Every available medium visual is requested as R in chain_transfer and pair_control; failed targets stay unfilled.',
                'Metadata coverage does not establish physics, actual camera or rendered asset coverage.',
                'Primitive source geometry, original materials/appearance and existing motion rules retained; no asset-proxy branch or compound support admitted.']}


def _released_asset_proxy_plan(pool,rules,*,profile_kind):
    """Build the common two-template plan for one frozen released-asset class."""
    if rules['matrix'].get('geometry_scope')!='mixed_flat_R_v1':raise ValueError('released assets require mixed flat role compatibility')
    required=rules['matrix']['required_asset_ids'];variants=defaultdict(set);hosts=defaultdict(list)
    for row in pool['objects']:
        if row['source']['source_family']!='asset':continue
        obj=row['metadata']['simulation']['objects'][0];asset=obj['visual_profile']['id']
        primitive=obj['collision_profile']['type']==obj['geometry']['type']
        compound=obj['collision_profile']['type']=='compound' and obj['geometry']['type']=='cylinder'
        if (profile_kind=='primitive' and primitive) or (profile_kind=='compound' and compound):
            variants[asset].add((obj['geometry']['type'],source_scale(row)))
    if set(variants)!=set(required):raise ValueError(f'released {profile_kind} asset ids differ from the frozen scope')
    if any(len(values)!=1 for values in variants.values()):raise ValueError(f'{profile_kind} asset geometry or scale is ambiguous')
    for row in pool['hosts']:hosts[row['metadata']['simulation']['support']['scene_class']].append(row)
    if set(hosts)!={'ground_flat','raised_flat'}:raise ValueError('released asset coverage requires both flat host classes')
    generic_medium_spheres=sum(row['source']['source_family']=='generic' and source_scale(row)=='medium'
        and row['metadata']['simulation']['objects'][0]['geometry']['type']=='sphere' for row in pool['objects'])
    if generic_medium_spheres<2:raise ValueError('released asset coverage requires two generic medium sphere sources')
    views=[view['id'] for view in rules['scene']['camera']['view_families']];cells=[]
    for ti,template in enumerate(('chain_transfer','pair_control')):
        for ai,asset in enumerate(required):
            shape,scale=next(iter(variants[asset]));host=sorted(hosts)[(ai+ti)%2]
            categories=sorted({environment_category(row) for row in hosts[host]})
            cell={'template':template,'host_class':host,'target_role':ROLES[(ai+ti)%3],
                  'shape_by_role':{'P':'sphere','Q':'sphere','R':shape},
                  'scale_by_role':{'P':'medium','Q':'medium','R':scale},
                  'required_visual_asset_by_role':{'R':asset},
                  'required_source_family_by_role':{'P':'generic','Q':'generic','R':'asset'},
                  'requested_environment_category':categories[(ai//2+ti)%len(categories)],
                  'requested_camera_family':views[(ai//3+ti)%len(views)]}
            cell['cell_id']=template+'_'+sha256_json(cell)[:12];cell['capacity_condition']=True;cells.append(cell)
    cells.sort(key=lambda cell:cell['cell_id'])
    if len(cells)!=rules['matrix']['candidate_budget']:raise ValueError('released asset coverage differs from frozen two-cell-per-asset budget')
    reference_rules={**rules,'matrix':{**rules['matrix'],'candidate_budget':24,'coverage_mode':None}}
    reference=mixed_plan(pool,reference_rules)
    return {**reference,'candidate_count':len(cells),'cells':cells,'required_asset_ids':list(required)}


def primitive_asset_plan(pool,rules):
    """Exercise each explicitly admitted primitive asset as R in two templates."""
    result=_released_asset_proxy_plan(pool,rules,profile_kind='primitive')
    result['scope_limitations']=['Eight released centered primitive asset proxies only; compound and mesh-host assets remain excluded.',
        'Asset appears only as R; P/Q remain released generic medium spheres.',
        'Chain-transfer and pair-control on two flat host classes; complete physics and camera admission required.',
        'Exact released proxy, material, mesh and semantic label retained; no geometry or mass rewriting.']
    return result


def compound_asset_plan(pool,rules):
    """Exercise each admitted centered cylinder compound as R in two templates."""
    result=_released_asset_proxy_plan(pool,rules,profile_kind='compound')
    result['scope_limitations']=['Three released centered axisymmetric compound assets only; off-axis and non-axisymmetric assets remain excluded.',
        'Asset appears only as R; P/Q remain released generic medium spheres.',
        'Chain-transfer and pair-control on two flat host classes; complete physics, exact compound camera and media admission required.',
        'Child-cylinder collision union, released inertia reference, material, mesh and semantic label retained; the cylinder geometry is layout envelope only.']
    return result


def inclined_plan(pool,rules):
    """24 explicit cells; repeated draws are distinct initial inputs, not relabels."""
    hosts=defaultdict(list)
    for row in pool['hosts']:hosts[row['metadata']['simulation']['support']['scene_class']].append(row)
    if set(hosts)!={'ground_feature','raised_feature'}:raise ValueError('inclined pilot lacks both declared host classes')
    families=[f['id'] for f in rules['scene']['camera']['view_families']];cells=[]
    for hi,host in enumerate(sorted(hosts)):
        categories=sorted({environment_category(row) for row in hosts[host]})
        for ri,role in enumerate(ROLES):
            for si,scale in enumerate(('small','medium')):
                for repeat in range(2):
                    cell={'template':'pair_control','host_class':host,'target_role':role,'replicate':repeat,
                      'shape_by_role':dict.fromkeys(ROLES,'sphere'),'scale_by_role':dict.fromkeys(ROLES,scale),
                      'requested_environment_category':categories[(ri+si+repeat)%len(categories)],
                      'requested_camera_family':families[(hi+ri+si+repeat)%len(families)],
                      'initial_speed_m_s':.8,'spacing_ratio':1.7 if scale=='small' else 1.5}
                    cell['cell_id']='pair_control_'+sha256_json(cell)[:12]
                    cell['capacity_condition']=sum(source_scale(row)==scale for row in pool['objects'])>=3
                    cells.append(cell)
    cells.sort(key=lambda c:c['cell_id'])
    if len(cells)!=rules['matrix']['candidate_budget']:raise ValueError('inclined allocation differs from frozen 24-cell budget')
    return {'schema_version':'physweep_three_object_pilot_coverage_v1','candidate_count':len(cells),'cells':cells,
      'construction':{'spacing_ratio':[1.6,2.8],'ballistic_first_contact_time_s':[.28,.6],
        'first_contact_offset_ratio':[.45,.7],'multi_contact_templates':[],
        'camera_bounded_initial_speed_m_s':.8,
        'camera_bounded_spacing_ratio_by_scale':{'small':1.7,'medium':1.5},
        **({'independent_lane_lead_in_pair_radii':rules['motion']['initial_state']['independent_lane_lead_in_pair_radii']} if 'independent_lane_lead_in_pair_radii' in rules['motion']['initial_state'] else {})},
      'scope_limitations':['8-12 degree centered Y ramps without side rails; unchanged source fixtures',
        'initial +X cross-slope pair control only; Q is stationary only at t=0',
        'small/small/small and medium/medium/medium source-scale categories; actual radii differ',
        'source appearance is inherited; the declared incline-only friction policy is applied before physics',
        'one shared static lighting preset'],
      'source_capacity':pool['summary']}
