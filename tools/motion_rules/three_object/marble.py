"""Initial-only marble arrangement with weak, explicitly ordered contact semantics."""
import numpy as np
from tools.core.hashing import sha256_json
from tools.core.marble_fixture_layout import validate_marble_fixture_construction

SCHEMA='physweep_marble_run_three_object_scene_v1'
IDS=('object_a','object_b','object_c');ROLES=('P','Q','R')

def initial_states(radius,limits,parameters):
    if parameters not in limits['parameter_cases']:raise ValueError('marble initial parameters outside frozen finite domain')
    if not np.isfinite(radius) or radius<=0:raise ValueError('invalid marble radius')
    origin=np.asarray(limits['reference_position_m']);direction=np.asarray(limits['longitudinal_direction'])
    if origin.shape!=(3,) or direction.shape!=(3,) or not np.isfinite(origin).all() or not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-1)>1e-10:
        raise ValueError('invalid marble initial reference frame')
    direction=direction/np.linalg.norm(direction)
    p=parameters;distances=[p['P_shift_m'],p['P_shift_m']+2*radius+p['PQ_gap_m'],p['P_shift_m']+4*radius+p['PQ_gap_m']+p['QR_gap_m']]
    if max(distances)>limits['maximum_initial_longitudinal_offset_m']:raise ValueError('marble layout exceeds initial channel domain')
    states={role:{'position_m':(origin+d*direction).tolist(),'orientation_quaternion_wxyz':[1.,0.,0.,0.],
        'linear_velocity_m_s':(p['P_speed_m_s']*direction if role=='P' else np.zeros(3)).tolist(),'angular_velocity_rad_s':[0.,0.,0.]}
        for role,d in zip(ROLES,distances)}
    points=np.asarray([states[r]['position_m'] for r in ROLES])
    if min(np.linalg.norm(points[i]-points[j])-2*radius for i in range(3) for j in range(i))<limits['minimum_initial_surface_gap_m']:
        raise ValueError('marble initial balls overlap or lack separation')
    return states

def validate_marble_contract(metadata):
    if metadata.get('schema_version')!=SCHEMA:raise ValueError('wrong three-object marble schema')
    c=metadata.get('three_object',{})
    if c.get('schema_version') not in {'physweep_three_object_marble_motion_contract_v1','physweep_three_object_marble_motion_contract_v2'}:raise ValueError('missing marble motion contract')
    if sha256_json({k:v for k,v in c.items() if k!='contract_sha256'})!=c.get('contract_sha256'):raise ValueError('marble motion contract hash mismatch')
    objects=metadata['simulation']['objects'];roles=c['roles'];t=c['template']
    if [o['object_id'] for o in objects]!=list(IDS) or set(roles)!=set(ROLES) or set(roles.values())!=set(IDS):raise ValueError('invalid marble identities or roles')
    if (t['id']!='ordered_contacts' or t['initially_moving_roles']!=['P'] or t['required_pairs']!=[['P','Q'],['Q','R']]
        or t['allowed_pairs']!=[['P','Q'],['Q','R'],['P','R']] or t['forbidden_pairs']!=[] or t['first_contact_order']!=[['P','Q'],['Q','R']]):
        raise ValueError('marble ordered-contact relation changed')
    s=metadata['semantics'];physics=metadata['physics']
    if s['dynamic_object_count']!=3 or s['scene_family']!='marble_run' or s['motion_profile']!='ordered_contacts':raise ValueError('marble semantics differ from declared scope')
    if sha256_json(physics['fixture'])!=c['fixture_sha256']:raise ValueError('marble fixture binding changed')
    if c['schema_version']=='physweep_three_object_marble_motion_contract_v2':
        validate_marble_fixture_construction(physics['fixture'],c['fixture_construction'])
        if metadata['source_binding']['fixture_construction']!=c['fixture_construction']:
            raise ValueError('marble fixture construction differs from provenance')
    elif 'fixture_construction' in c:raise ValueError('marble v1 cannot silently apply a source fixture transformation')
    radius=objects[0]['collision_proxy']['radius_m'];states=initial_states(radius,c['initial_limits'],c['initial_parameters'])
    for role,oid in roles.items():
        obj=next(o for o in objects if o['object_id']==oid)
        if obj['collision_proxy']!={'type':'sphere','radius_m':radius} or obj['geometry']!={'type':'sphere','size_m':[2*radius]*3}:raise ValueError('marble collision and visual geometry disagree')
        if obj['initial_state']!=states[role]:raise ValueError('marble initial state contradicts construction')
        m=obj['material'];keys=('mass_kg','contact_friction','contact_restitution','rolling_friction','spinning_friction','linear_damping','angular_damping')
        if any(type(m[k]) not in (int,float) or not np.isfinite(m[k]) or m[k]<0 for k in keys) or m['mass_kg']<=0:raise ValueError('marble seven material values must be finite and valid')
        visual=obj['visual_profile']
        if visual['recipe_id'] not in c['appearance_palette'] or obj['visual']['color_rgba']!=c['appearance_palette'][visual['recipe_id']]:raise ValueError('marble appearance differs from declared recipe')
    if set(o['visual_profile']['recipe_id'] for o in objects)!=set(c['appearance_palette']):raise ValueError('three distinct marble appearance recipes required')
    return c
