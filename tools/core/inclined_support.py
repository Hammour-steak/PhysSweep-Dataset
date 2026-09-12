"""Resolve the declared shallow Y-ramp's actual collision top plane."""
from __future__ import annotations
import numpy as np
from tools.core.rigid_geometry import quaternion_wxyz_from_euler_degrees, quaternion_matrix_wxyz


def inclined_frame(support: dict) -> dict:
    if support.get('support_shape')!='inclined_ramp':raise ValueError('inclined scope requires a ramp')
    primary=[c for c in support['colliders'] if c.get('role')=='primary_support' and c.get('collision_enabled',True)]
    if len(primary)!=1 or primary[0]['primitive']!='box':raise ValueError('inclined scope requires one analytic primary box')
    box=primary[0];angles=np.asarray(box['rotation_euler_degrees'],dtype=float)
    size=np.asarray(box['size_m'],dtype=float);origin=np.asarray(box['position_m'],dtype=float)
    if angles.shape!=(3,) or size.shape!=(3,) or origin.shape!=(3,) or not np.isfinite([angles,size,origin]).all() or (size<=0).any():raise ValueError('invalid ramp collision dimensions')
    if not 8<=angles[0]<=12 or not np.allclose(angles[1:],0,rtol=0,atol=1e-9) or not np.allclose(origin[:2],0,rtol=0,atol=1e-9):
        raise ValueError('inclined scope requires a centered shallow Y ramp')
    rotation=np.asarray(quaternion_matrix_wxyz(quaternion_wxyz_from_euler_degrees(angles)))
    normal=rotation[:,2];top=origin+normal*size[2]/2;offset=float(normal@top)
    frame=support['surface_frame']
    for key,expected in (('normal',normal),('tangent_uphill',rotation[:,1]),('tangent_cross',rotation[:,0])):
        if not np.allclose(frame[key],expected,rtol=0,atol=5e-8):raise ValueError('ramp frame contradicts actual collider orientation')
    if abs(float(frame['slope_angle_degrees'])-angles[0])>1e-6 or abs(support['surface_center_z_m']-offset/normal[2])>2e-6:
        raise ValueError('ramp declared surface height contradicts actual collision plane')
    visual=support.get('visual_geometry',{})
    if visual.get('primitive')!='solid_wedge' or visual.get('slope_axis')!='y':raise ValueError('inclined scope requires a declared wedge visual')
    width,length=visual['size_xy_m'];low=visual['base_z_m'];high=visual['high_top_z_m']
    if not np.isfinite([width,length,low,high]).all() or min(width,length)<=0 or high<=low:raise ValueError('invalid ramp visual geometry')
    visual_heights=np.array([low,high]);ys=np.array([-length/2,length/2])
    physical_heights=(offset-normal[1]*ys)/normal[2]
    if not np.allclose(visual_heights,physical_heights,rtol=0,atol=2e-6) or abs(width-size[0])>2e-6:
        raise ValueError('ramp visual top plane disagrees with collision plane')
    return {'normal':normal,'rotation':rotation,'origin':origin,'size':size,'plane_offset':offset}


def supported_sphere_state(frame: dict, xy, radius: float):
    normal=frame['normal'];x,y=xy
    center=np.array([x,y,(frame['plane_offset']+radius-normal[0]*x-normal[1]*y)/normal[2]])
    return center,center-radius*normal


def validate_sphere_on_ramp(frame: dict, position, radius: float, margin_factor: float):
    position=np.asarray(position,dtype=float);normal=frame['normal']
    if abs(float(normal@position-frame['plane_offset'])-radius)>2e-6:
        raise ValueError('three-object initial position does not rest on ramp collision plane')
    contact=position-radius*normal;local=frame['rotation'].T@(contact-frame['origin'])
    if np.any(np.abs(local[:2])+margin_factor*radius>frame['size'][:2]/2):
        raise ValueError('three-object initial contact exceeds finite ramp surface')
