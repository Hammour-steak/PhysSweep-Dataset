"""Construct supported initial states; never drive objects during simulation."""
from __future__ import annotations
import copy
import numpy as np

from tools.core.hashing import sha256_json
from tools.core.primitive_support import initial_bounds, initial_extents
from tools.motion_rules.three_object.geometry import validate_role_geometry

OBJECT_IDS=('object_a','object_b','object_c')
ROLES=('P','Q','R')


def center_layout_envelope(xy: dict[str,np.ndarray],extents: dict[str,np.ndarray],edge_factor: float):
    """Rigidly center the declared XY clearance envelope around the host origin."""
    if not np.isfinite(edge_factor) or edge_factor<=0:
        raise ValueError('layout edge clearance must be finite and positive')
    low=np.min(np.asarray([xy[role]-edge_factor*extents[role][:2] for role in ROLES]),axis=0)
    high=np.max(np.asarray([xy[role]+edge_factor*extents[role][:2] for role in ROLES]),axis=0)
    offset=(low+high)/2
    return {role:value-offset for role,value in xy.items()},offset


def apply_motion(scene: dict, rules: dict, *, role_order: list[str], speed_m_s: float,
                 spacing_ratio: float, template_id: str, offset_ratio: float=0.6,
                 motion_axis: str='world_x',layout_variant: str='axis_aligned') -> dict:
    if len(role_order)!=3 or set(role_order)!=set(ROLES):raise ValueError('roles must be a permutation of P/Q/R')
    templates={t['id']:t for t in rules['motion']['templates']}
    if template_id not in templates:raise ValueError('unknown three-object template')
    template=templates[template_id]
    admitted_variants=template.get('layout_variants',['axis_aligned'])
    if layout_variant not in admitted_variants:raise ValueError('layout variant is not admitted by the motion rules')
    initial=rules['motion']['initial_state']
    if not initial['initial_speed_m_s'][0]<=speed_m_s<=initial['initial_speed_m_s'][1]:raise ValueError('speed outside frozen pilot range')
    if not initial['separation_in_pair_diameters'][0]<=spacing_ratio<=initial['separation_in_pair_diameters'][1]:raise ValueError('spacing outside frozen pilot range')
    if motion_axis not in {'world_x','world_y'}:raise ValueError('motion axis must be world_x or world_y')
    objects=scene['simulation']['objects']
    if [o['object_id'] for o in objects]!=list(OBJECT_IDS):raise ValueError('object order must be A/B/C')
    by_role=dict(zip(role_order,objects))
    validate_role_geometry(objects,{r:o['object_id'] for r,o in by_role.items()},template)
    mass=[o['material']['mass_kg'] for o in objects]
    if min(mass)<=0 or max(mass)/min(mass)>initial['maximum_role_mass_ratio']:raise ValueError('role mass ratio outside pilot prior')
    relative_bounds={r:initial_bounds(by_role[r]) for r in ROLES}
    extents={r:initial_extents(by_role[r]) for r in ROLES}
    longitudinal_axis=0 if motion_axis=='world_x' else 1
    lateral_axis=1-longitudinal_axis
    longitudinal_extent={r:float(extents[r][longitudinal_axis]) for r in ROLES}
    lateral_extent={r:float(extents[r][lateral_axis]) for r in ROLES}
    radius={r:float(extents[r][0]) for r in ROLES}
    x={'P':-spacing_ratio*(longitudinal_extent['P']+longitudinal_extent['Q']),'Q':0.0,
       'R':spacing_ratio*(longitudinal_extent['Q']+longitudinal_extent['R'])}
    xy={role:np.array([value,0.]) for role,value in x.items()}
    velocities={r:np.array([speed_m_s if r=='P' else 0.,0.,0.]) for r in ROLES}
    parameters={'speed_m_s':speed_m_s,'spacing_ratio':spacing_ratio}
    if layout_variant!='axis_aligned':parameters['layout_variant']=layout_variant
    if template_id=='chain_transfer' and layout_variant.startswith('angled_chain_'):
        if any(by_role[role]['geometry']['type']!='sphere' for role in ROLES):
            raise ValueError('angled chain is implemented only for spheres')
        sign=1. if layout_variant.endswith('_left') else -1.
        low,high=initial['first_contact_offset_in_pair_radii']
        if not low<offset_ratio<=high:raise ValueError('angled chain requires a nonzero admitted offset')
        pair_radius=radius['P']+radius['Q']
        vertical_normal=(radius['Q']-radius['P'])/pair_radius
        horizontal_square=1-offset_ratio**2-vertical_normal**2
        if horizontal_square<=0:raise ValueError('offset and unequal radii leave no first contact')
        normal=np.array([np.sqrt(horizontal_square),sign*offset_ratio])
        xy['Q'][1]=sign*offset_ratio*pair_radius
        xy['R']=xy['Q']+spacing_ratio*(radius['Q']+radius['R'])*normal
        parameters.update(offset_ratio=offset_ratio,turn_side='left' if sign>0 else 'right',
                          predicted_Q_direction_after_first_contact=normal.tolist())
    if template_id=='successive_hits':
        low,high=initial['first_contact_offset_in_pair_radii']
        if not low<offset_ratio<=high:raise ValueError('successive hits require a nonzero admitted first-contact offset')
        pair_radius=radius['P']+radius['Q']
        sign=-1. if layout_variant=='offset_successive_right' else 1.
        xy['Q'][1]=sign*offset_ratio*pair_radius
        vertical_normal=(radius['Q']-radius['P'])/pair_radius
        horizontal_square=1-offset_ratio**2-vertical_normal**2
        if horizontal_square<=0:raise ValueError('offset and unequal radii leave no first contact')
        normal=np.array([np.sqrt(horizontal_square),sign*offset_ratio])
        first_position=np.array([-pair_radius*normal[0],0.])
        restitution=by_role['P']['material']['contact_restitution']*by_role['Q']['material']['contact_restitution']
        coefficient=(1+restitution)*by_role['Q']['material']['mass_kg']/(by_role['P']['material']['mass_kg']+by_role['Q']['material']['mass_kg'])
        outgoing=np.array([1.,0.])-coefficient*normal[0]*normal
        if outgoing[0]<=0 or np.linalg.norm(outgoing)<.05:raise ValueError('first collision estimate leaves no forward continuation')
        outgoing/=np.linalg.norm(outgoing)
        xy['R']=first_position+spacing_ratio*(radius['P']+radius['R'])*outgoing
        parameters.update(offset_ratio=offset_ratio,placement_estimator='isolated_smooth_sphere_normal_impulse_v1',
                          turn_side='left' if sign>0 else 'right',
                          predicted_P_direction_after_first_contact=outgoing.tolist())
    elif template_id=='converging_hits':
        right_spacing=min(initial['separation_in_pair_diameters'][1],1.5*spacing_ratio)
        xy['R'][0]=right_spacing*(radius['Q']+radius['R'])
        right_speed=max(initial['initial_speed_m_s'][0],.65*speed_m_s)
        velocities['R']=np.array([-right_speed,0.,0.])
        parameters.update(R_speed_m_s=right_speed,R_spacing_ratio=right_spacing)
    elif template_id=='pair_control':
        lane=spacing_ratio*(lateral_extent['R']+max(lateral_extent['P'],lateral_extent['Q']))
        sign=-1. if layout_variant=='crossing_control_right' else 1.
        crossing_x=(xy['P'][0]-1.1*(longitudinal_extent['P']+longitudinal_extent['R'])
                    if layout_variant.startswith('crossing_control_') else xy['P'][0])
        xy['R']=np.array([crossing_x,sign*lane])
        if layout_variant.startswith('crossing_control_'):
            if by_role['R']['geometry']['type']=='sphere':
                cross_speed=max(initial['initial_speed_m_s'][0],.75*speed_m_s)
            else:
                cross_speed=float(initial.get('crossing_primitive_speed_m_s',0.))
                if not initial['initial_speed_m_s'][0]<=cross_speed<=initial['initial_speed_m_s'][1]:
                    raise ValueError('primitive crossing speed is absent or outside the admitted range')
            velocities['R']=np.array([0.,-sign*cross_speed,0.])
            parameters.update(R_speed_m_s=cross_speed,crossing_lane_separation_m=float(sign*lane),
                              crossing_longitudinal_position_m=float(crossing_x),
                              crossing_trailing_clearance_ratio=1.1,
                              crossing_speed_ratio=float(cross_speed/speed_m_s),
                              crossing_side='left' if sign>0 else 'right')
        else:
            velocities['R']=np.array([speed_m_s,0.,0.])
            parameters.update(R_speed_m_s=speed_m_s,parallel_lane_separation_m=float(xy['R'][1]))
    support=scene['simulation']['support'];bounds=support['safe_surface_bounds']
    inclined=support.get('support_shape')=='inclined_ramp'
    if inclined and motion_axis!='world_x':raise ValueError('inclined support retains its declared world_x construction')
    if inclined and layout_variant!='axis_aligned':raise ValueError('inclined support does not admit trial layout variants')
    if inclined:
        from tools.core.inclined_support import inclined_frame,supported_sphere_state
        if template_id!='pair_control' or template.get('support_scope')!='inclined_cross_slope_pair_v1' or any(o['geometry']['type']!='sphere' for o in objects):
            raise ValueError('inclined support is only implemented for cross-slope sphere pair control')
        material_policy=initial.get('inclined_material_policy')
        if material_policy is not None:
            expected_keys={'schema_version','contact_friction_minimum','rolling_friction','application_stage'}
            if (set(material_policy)!=expected_keys
                    or material_policy['schema_version']!='physweep_three_object_inclined_material_policy_v1'
                    or material_policy['application_stage']!='initial_state_construction_before_physics_and_sweeps'
                    or not 0<float(material_policy['contact_friction_minimum'])<=1
                    or not 0<float(material_policy['rolling_friction'])<1):
                raise ValueError('invalid inclined material policy')
            source_values={}
            for role,obj in by_role.items():
                material=obj['material']
                source_values[role]={
                    'contact_friction':float(material['contact_friction']),
                    'rolling_friction':float(material['rolling_friction']),
                }
                material['contact_friction']=max(
                    float(material_policy['contact_friction_minimum']),
                    float(material['contact_friction']))
                material['rolling_friction']=float(material_policy['rolling_friction'])
            parameters['inclined_material_policy']={
                **copy.deepcopy(material_policy),
                'source_values_by_role':source_values,
            }
        frame=inclined_frame(support)
        if 'independent_lane_lead_in_pair_radii' in initial:
            lead_ratio=float(initial['independent_lane_lead_in_pair_radii'])
            if lead_ratio!=3.:raise ValueError('unsupported inclined independent-lane lead')
            lead=lead_ratio*(radius['Q']+radius['R'])
            xy['R'][0]=xy['Q'][0]+lead
            parameters.update(R_lead_in_Q_R_radii=lead_ratio,R_lead_from_Q_m=lead)
    rotation=(np.eye(2) if motion_axis=='world_x' else np.array([[0.,-1.],[1.,0.]]))
    xy={role:rotation@value for role,value in xy.items()}
    velocities={role:np.concatenate([rotation@value[:2],value[2:]]) for role,value in velocities.items()}
    if support.get('exact_static_binding') is not None:
        xy,centering_offset=center_layout_envelope(
            xy,extents,float(rules['scene']['layout']['edge_clearance_in_object_radii']))
        parameters['exact_mesh_layout_centering_offset_xy_m']=centering_offset.tolist()
    approach=rotation@np.array([1.,0.])
    center=[float(np.mean(bounds[a])) for a in ('x','y')]
    for role,obj in by_role.items():
        state=copy.deepcopy(obj['initial_state'])
        state.update(position_m=[center[0]+xy[role][0],center[1]+xy[role][1],support['surface_center_z_m']-relative_bounds[role][0][2]],
                     contact_point_m=[center[0]+xy[role][0],center[1]+xy[role][1],support['surface_center_z_m']],
                     linear_velocity_m_s=velocities[role].tolist(),
                     angular_velocity_rad_s=copy.deepcopy(initial['angular_velocity_rad_s']))
        if inclined:
            position,contact=supported_sphere_state(frame,state['position_m'][:2],radius[role])
            state.update(position_m=position.tolist(),contact_point_m=contact.tolist())
        obj['initial_state']=state
        obj['expected_motion']={'motion_family':'three_object_'+template_id,'contact_mode':'supported',
                                'must_remain_finite':True,'initial_motion_mode':initial['motion_mode'] if role in template['initially_moving_roles'] else 'rest'}
    contract={'schema_version':'physweep_three_object_motion_contract_v1','template':copy.deepcopy(template),
        'roles':{r:by_role[r]['object_id'] for r in ROLES},'thresholds':copy.deepcopy(rules['motion']['thresholds']),
        'rules_sha256':rules['rules_sha256'],'approach_direction_world':[float(approach[0]),float(approach[1]),0.0],
        'initial_parameters':parameters,
        'camera_rules':copy.deepcopy(rules['scene']['camera'])}
    if motion_axis!='world_x':contract['layout_axis']=motion_axis
    if layout_variant!='axis_aligned':contract['layout_variant']=layout_variant
    contract['contract_sha256']=sha256_json(contract)
    scene['three_object']=contract
    dimensions=scene['semantic_sampling']['five_dimensions']
    dimensions['motion']={'family':'three_object_'+template_id,'subtype':template_id,
                          'direction':'positive_world_x_reference' if motion_axis=='world_x' else 'positive_world_y_reference'}
    if layout_variant!='axis_aligned':dimensions['motion']['layout_variant']=layout_variant
    dimensions['camera_observation']={'observation_intent':'three_object_events','structure_context':'horizontal_surface'}
    if inclined:dimensions['camera_observation']['structure_context']='inclined_surface'
    return scene


def validate_motion_contract(metadata: dict) -> dict:
    contract=metadata.get('three_object')
    if not isinstance(contract,dict) or contract.get('schema_version')!='physweep_three_object_motion_contract_v1':
        raise ValueError('three-object metadata requires an explicit motion contract')
    payload={k:v for k,v in contract.items() if k!='contract_sha256'}
    if sha256_json(payload)!=contract.get('contract_sha256'):raise ValueError('three-object motion contract hash mismatch')
    objects=metadata['simulation']['objects']
    if [o['object_id'] for o in objects]!=list(OBJECT_IDS) or set(contract['roles'])!=set(ROLES) or set(contract['roles'].values())!=set(OBJECT_IDS):
        raise ValueError('three-object motion identities are invalid')
    template=contract['template'];template_id=template['id']
    if template_id not in {'chain_transfer','successive_hits','converging_hits','pair_control'}:raise ValueError('three-object template not yet implemented')
    layout_variant=contract.get('layout_variant','axis_aligned')
    if layout_variant not in template.get('layout_variants',['axis_aligned']):
        raise ValueError('motion contract layout variant is not admitted by its template')
    if contract['initial_parameters'].get('layout_variant',layout_variant)!=layout_variant:
        raise ValueError('motion contract layout variant fields disagree')
    by_id={o['object_id']:o for o in objects}
    validate_role_geometry(objects,contract['roles'],template)
    if metadata['simulation']['support'].get('support_shape')=='inclined_ramp':
        from tools.core.inclined_support import inclined_frame,validate_sphere_on_ramp
        if template_id!='pair_control' or template.get('support_scope')!='inclined_cross_slope_pair_v1' or any(o['geometry']['type']!='sphere' for o in objects):
            raise ValueError('unsupported inclined template or geometry')
        frame=inclined_frame(metadata['simulation']['support'])
        for obj in objects:validate_sphere_on_ramp(frame,obj['initial_state']['position_m'],obj['geometry']['size_m'][0]/2,0.)
        parameters=contract['initial_parameters']
        material_policy=parameters.get('inclined_material_policy')
        if material_policy is not None:
            expected_keys={'schema_version','contact_friction_minimum','rolling_friction','application_stage','source_values_by_role'}
            if (set(material_policy)!=expected_keys
                    or material_policy['schema_version']!='physweep_three_object_inclined_material_policy_v1'
                    or material_policy['application_stage']!='initial_state_construction_before_physics_and_sweeps'
                    or set(material_policy['source_values_by_role'])!=set(ROLES)):
                raise ValueError('inclined material policy evidence is invalid')
            minimum=float(material_policy['contact_friction_minimum'])
            rolling=float(material_policy['rolling_friction'])
            if not 0<minimum<=1 or not 0<rolling<1:
                raise ValueError('inclined material policy values are invalid')
            sweep=metadata.get('sweep',{})
            swept_contact_object=(sweep.get('target_object_id')
                if sweep.get('kind')=='sweep' and sweep.get('axis')=='contact_friction' else None)
            for role in ROLES:
                obj=by_id[contract['roles'][role]]
                source=material_policy['source_values_by_role'][role]
                if not isinstance(source,dict) or set(source)!={'contact_friction','rolling_friction'}:
                    raise ValueError('inclined source material evidence is invalid')
                expected_contact=max(minimum,float(source['contact_friction']))
                contact_matches=(obj['object_id']==swept_contact_object
                    or abs(float(obj['material']['contact_friction'])-expected_contact)<=1e-12)
                if (not contact_matches
                        or abs(float(obj['material']['rolling_friction'])-rolling)>1e-12):
                    raise ValueError('inclined material values contradict declared construction policy')
        if parameters.get('R_lead_in_Q_R_radii')!=template.get('independent_lane_lead_in_pair_radii'):
            raise ValueError('inclined independent-lane lead policy is missing or inconsistent')
        if 'R_lead_in_Q_R_radii' in parameters:
            q=by_id[contract['roles']['Q']];r=by_id[contract['roles']['R']]
            expected=3.*(q['geometry']['size_m'][0]+r['geometry']['size_m'][0])/2
            if parameters['R_lead_in_Q_R_radii']!=3. or abs(parameters['R_lead_from_Q_m']-expected)>1e-12 or abs(r['initial_state']['position_m'][0]-q['initial_state']['position_m'][0]-expected)>1e-12:
                raise ValueError('inclined independent-lane lead contradicts initial positions')
    states=[by_id[contract['roles'][r]]['initial_state'] for r in ROLES]
    positions=np.asarray([s['position_m'] for s in states],dtype=float)
    direction=np.asarray(contract['approach_direction_world'],dtype=float)
    if (direction.shape!=(3,) or not any(np.allclose(direction,value,rtol=0,atol=1e-12)
            for value in ([1,0,0],[0,1,0]))):
        raise ValueError('chain constructor requires a declared positive world X or Y direction')
    axis=contract.get('layout_axis','world_x')
    if axis not in {'world_x','world_y'} or (axis=='world_x')!=bool(np.allclose(direction,[1,0,0],rtol=0,atol=1e-12)):
        raise ValueError('layout axis contradicts the approach direction')
    lateral=np.array([-direction[1],direction[0],0.])
    longitudinal=positions@direction
    transverse=positions@lateral
    if not np.isfinite(positions).all():raise ValueError('nonfinite initial layout')
    if template_id=='chain_transfer' and layout_variant.startswith('angled_chain_'):
        sign=1 if layout_variant.endswith('_left') else -1
        if not longitudinal[0]<longitudinal[1]<longitudinal[2] or not sign*(transverse[1]-transverse[0])>0 or not sign*(transverse[2]-transverse[1])>0:
            raise ValueError('angled chain contradicts its declared turn side')
    elif template_id in {'chain_transfer','converging_hits'}:
        if not longitudinal[0]<longitudinal[1]<longitudinal[2] or np.ptp(transverse)>1e-9:
            raise ValueError('collinear template contradicts initial role layout')
    elif template_id=='successive_hits':
        sign=-1 if layout_variant=='offset_successive_right' else 1
        if not longitudinal[0]<longitudinal[1] or not longitudinal[2]>longitudinal[0] or not sign*(transverse[1]-transverse[0])>0 or not sign*(transverse[0]-transverse[2])>0:
            raise ValueError('offset first-contact template contradicts initial layout')
    elif layout_variant.startswith('crossing_control_'):
        sign=1 if layout_variant.endswith('_left') else -1
        if not longitudinal[2]<longitudinal[0]<longitudinal[1] or abs(transverse[0]-transverse[1])>1e-9 or not sign*(transverse[2]-transverse[0])>0:
            raise ValueError('crossing control contradicts its initial lane layout')
    elif not longitudinal[0]<longitudinal[1] or abs(transverse[0]-transverse[1])>1e-9 or transverse[2]<=transverse[1]:
        raise ValueError('parallel control contradicts initial lane layout')
    t=contract['thresholds']
    for i,state in enumerate(states):
        velocity=np.asarray(state['linear_velocity_m_s'],dtype=float)
        angular=np.asarray(state['angular_velocity_rad_s'],dtype=float)
        if not np.isfinite(velocity).all() or not np.isfinite(angular).all():raise ValueError('nonfinite initial motion')
        role=ROLES[i]
        if role in template['initially_moving_roles']:
            if role=='R' and layout_variant.startswith('crossing_control_'):
                side=1 if layout_variant.endswith('_left') else -1
                valid=(-side*float(velocity@lateral)>=t['moving_linear_speed_m_s']
                       and abs(float(velocity@direction))<=1e-9
                       and abs(float(velocity[2]))<=1e-9)
            else:
                sign=-1 if role=='R' and template_id=='converging_hits' else 1
                valid=sign*float(velocity@direction)>=t['moving_linear_speed_m_s'] and np.linalg.norm(velocity-(velocity@direction)*direction)<=1e-9
        else:valid=np.linalg.norm(velocity)<=t['stationary_linear_speed_m_s']
        if not valid or np.linalg.norm(angular)>t['stationary_angular_speed_rad_s']:
            raise ValueError('initial motion contradicts declared role or supported sliding mode')
    return contract
