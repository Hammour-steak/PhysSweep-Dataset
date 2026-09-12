"""Declared initial pinfield pair encounter with a separately moving third ball."""
import numpy as np
from tools.core.hashing import sha256_json

SCHEMA='physweep_passive_pinball_three_object_scene_v1'
IDS=('object_a','object_b','object_c')
ROLES=('P','Q','R')


def initial_states(radius,fixture_source,frame,limits,parameters):
    if parameters not in limits['parameter_cases']:
        raise ValueError('pinball initial parameters outside frozen finite domain')
    if not np.isfinite(radius) or radius<=0:raise ValueError('invalid pinball radius')
    top=np.asarray(fixture_source['top_center_m']);right=np.asarray(frame['right']);down=np.asarray(frame['down']);normal=np.asarray(frame['normal'])
    basis=np.stack([right,down,normal])
    if not np.allclose(basis@basis.T,np.eye(3),rtol=0,atol=1e-10):raise ValueError('invalid pinball fixture basis')
    side=parameters['side'];qx=parameters['Q_x_abs_m']
    xs=[side*(qx+2*radius+parameters['gap_m']),side*qx,-side*limits['R_x_abs_m']]
    velocities=[-side*parameters['speed_m_s']*right,np.zeros(3),limits['R_initial_down_speed_m_s']*down]
    height=fixture_source['board_thickness_m']/2+radius+limits['normal_surface_gap_m']
    states={}
    for role,x,velocity in zip(ROLES,xs,velocities):
        if abs(x)+radius+limits['edge_margin_m']>fixture_source['board_width_m']/2:
            raise ValueError('pinball initial state exceeds board width')
        states[role]={'position_m':(top+x*right+limits['initial_down_m']*down+height*normal).tolist(),
            'orientation_quaternion_wxyz':[1.,0.,0.,0.],'linear_velocity_m_s':velocity.tolist(),'angular_velocity_rad_s':[0.,0.,0.]}
    positions=np.asarray([states[r]['position_m'] for r in ROLES])
    if min(np.linalg.norm(positions[i]-positions[j])-2*radius for i in range(3) for j in range(i))<limits['minimum_initial_surface_gap_m']:
        raise ValueError('pinball initial balls overlap or lack separation')
    return states


def validate_pinball_contract(metadata):
    if metadata.get('schema_version')!=SCHEMA:raise ValueError('wrong three-object pinball schema')
    c=metadata.get('three_object',{})
    if c.get('schema_version')!='physweep_three_object_pinball_motion_contract_v1':raise ValueError('missing pinball motion contract')
    if sha256_json({k:v for k,v in c.items() if k!='contract_sha256'})!=c.get('contract_sha256'):raise ValueError('pinball motion contract hash mismatch')
    objects=metadata['simulation']['objects'];roles=c['roles'];t=c['template']
    if [o['object_id'] for o in objects]!=list(IDS) or set(roles)!=set(ROLES) or set(roles.values())!=set(IDS):raise ValueError('invalid pinball identities or roles')
    if (t['id']!='pair_control' or t['initially_moving_roles']!=['P','R'] or t['required_pairs']!=[['P','Q']]
        or t['allowed_pairs']!=[['P','Q']] or t['forbidden_pairs']!=[['P','R'],['Q','R']] or t['first_contact_order']!=[]):
        raise ValueError('pinball pair-control relation changed')
    semantics=metadata['semantics'];physics=metadata['physics']
    if semantics['dynamic_object_count']!=3 or semantics['scene_family']!='passive_pinball' or semantics['motion_profile']!='pair_control':raise ValueError('pinball semantics differ from declared scope')
    if sha256_json({'fixture':physics['fixture'],'fixture_source':physics['fixture_source']})!=c['fixture_sha256']:raise ValueError('pinball source fixture changed')
    radius=objects[0]['collision_proxy']['radius_m']
    states=initial_states(radius,physics['fixture_source'],physics['fixture']['frame'],c['initial_limits'],c['initial_parameters'])
    for role,oid in roles.items():
        obj=next(o for o in objects if o['object_id']==oid)
        if obj['collision_proxy']!={'type':'sphere','radius_m':radius} or obj['geometry']!={'type':'sphere','size_m':[2*radius]*3}:raise ValueError('pinball collision and visual geometry disagree')
        if obj['initial_state']!=states[role]:raise ValueError('pinball initial state contradicts construction')
        m=obj['material']
        keys=('mass_kg','contact_friction','contact_restitution','rolling_friction','spinning_friction','linear_damping','angular_damping')
        if any(type(m[k]) not in (int,float) or not np.isfinite(m[k]) or m[k]<0 for k in keys) or m['mass_kg']<=0:raise ValueError('pinball seven material values must be finite and valid')
        visual=obj['visual_profile']
        if visual['recipe_id'] not in c['appearance_palette'] or obj['visual']['color_rgba']!=c['appearance_palette'][visual['recipe_id']]:raise ValueError('pinball appearance differs from declared recipe')
    if set(o['visual_profile']['recipe_id'] for o in objects)!=set(c['appearance_palette']):raise ValueError('three distinct pinball appearance recipes required')
    return c
