"""First-pilot sphere/flat-host compatibility, separate from motion outcomes."""
from __future__ import annotations
import numpy as np
from tools.core.rigid_geometry import quaternion_wxyz_from_euler_degrees,quaternion_matrix_wxyz


def validate_host(host: dict, rules: dict, *, allow_exact_mesh: bool=False) -> None:
    support=host['simulation']['support']
    inclined=rules.get('support_scope')=='inclined_cross_slope_pair_v1'
    if inclined:
        from tools.core.inclined_support import inclined_frame
        inclined_frame(support)
    normal=np.asarray(support['surface_frame']['normal'],dtype=float)
    roles={c.get('role') for c in support['colliders'] if c.get('collision_enabled',True)}
    exact=support.get('exact_static_binding') if allow_exact_mesh else None
    invalid_shape=support.get('support_shape') not in rules['support_shapes']
    invalid_normal=normal.shape!=(3,) or (not inclined and not np.allclose(normal,[0,0,1],rtol=0,atol=1e-12))
    invalid_slope=abs(float(support['surface_frame']['slope_angle_degrees']))>rules['maximum_abs_slope_degrees']
    invalid_camera=support.get('camera_envelope') is not None
    if exact is None:
        invalid_roles=not set(rules['required_collider_roles'])<=roles<=set(rules['allowed_collider_roles'])
        if invalid_shape or invalid_normal or invalid_slope or invalid_roles or invalid_camera:
            raise ValueError('host outside first three-object pilot support scope')
    else:
        roles.add('primary_support')
        if invalid_shape:raise ValueError('host support shape is outside the three-object scope')
        if invalid_normal:raise ValueError('host surface normal is outside the three-object scope')
        if invalid_slope:raise ValueError('host surface slope is outside the three-object scope')
        if not set(rules['required_collider_roles'])<=roles<=set(rules['allowed_collider_roles']):
            raise ValueError('host collider roles are outside the three-object scope')
        if invalid_camera:raise ValueError('host camera envelope is outside the three-object scope')
    if host['appearance']['scene_visual']['visual_type'] not in rules['allowed_visual_types']:
        raise ValueError('host outside first three-object pilot visual scope')
    if exact is not None:
        visual=host['appearance'].get('support_visual',{})
        if (support.get('collision_authority')!='exact_static_proxy'
                or exact.get('asset_id')!=support.get('asset_id')
                or visual.get('visual_type')!='mesh_support'
                or visual.get('asset_id')!=exact.get('asset_id')
                or any(c.get('collision_enabled',True) for c in support['colliders']
                       if c.get('role') in {'primary_support','support_structure'})):
            raise ValueError('exact-mesh host collision, visual or identity binding is inconsistent')


def validate_initial_layout(scene: dict, rules: dict, separation_m: float, *, root=None):
    inclined=rules.get('support_scope')=='inclined_cross_slope_pair_v1'
    if inclined:
        from tools.core.inclined_support import inclined_frame,validate_sphere_on_ramp
        frame=inclined_frame(scene['simulation']['support'])
        if any(o['geometry']['type']!='sphere' for o in scene['simulation']['objects']):raise ValueError('inclined pilot requires spheres')
    if any(o['geometry']['type']!='sphere' for o in scene['simulation']['objects']):
        return validate_mixed_initial_layout(scene,rules,separation_m)
    objects=scene['simulation']['objects'];support=scene['simulation']['support']
    bounds=support['safe_surface_bounds']
    for obj in objects:
        size=np.asarray(obj['geometry']['size_m'],dtype=float)
        if obj['geometry']['type']!='sphere' or not np.isfinite(size).all() or (size<=0).any() or not np.allclose(size,size[0],rtol=0,atol=1e-9):
            raise ValueError('first three-object pilot requires sphere geometry')
        pos=np.asarray(obj['initial_state']['position_m'],dtype=float)
        margin=float(size[0]/2*rules['layout']['edge_clearance_in_object_radii'])
        if not np.isfinite(pos).all() or any(not bounds[axis][0]+margin<=pos[i]<=bounds[axis][1]-margin for i,axis in enumerate(('x','y'))):
            raise ValueError('three-object initial layout exceeds support bounds')
        if inclined:
            validate_sphere_on_ramp(frame,pos,float(size[0]/2),rules['layout']['edge_clearance_in_object_radii'])
        elif abs(pos[2]-support['surface_center_z_m']-size[0]/2)>1e-6:
            raise ValueError('three-object initial position does not rest on support')
        for collider in [*support['colliders'],*scene['environment_binding']['colliders']]:
            if not collider.get('collision_enabled',True):continue
            if collider['primitive']!='box':raise ValueError('first pilot requires analytic static box clearance')
            rotation=np.asarray(quaternion_matrix_wxyz(quaternion_wxyz_from_euler_degrees(collider['rotation_euler_degrees'])))
            local=rotation.T@(pos-np.asarray(collider['position_m']))
            outside=np.abs(local)-np.asarray(collider['size_m'])/2
            signed_distance=float(np.linalg.norm(np.maximum(outside,0))+min(float(np.max(outside)),0))
            if signed_distance-size[0]/2 < -scene['qa']['limits']['maximum_initial_penetration_m']:
                raise ValueError('three-object initial sphere intersects static collision geometry')
    for i,obj in enumerate(objects):
        for other in objects[i+1:]:
            gap=np.linalg.norm(np.asarray(obj['initial_state']['position_m'])-other['initial_state']['position_m'])-(obj['geometry']['size_m'][0]+other['geometry']['size_m'][0])/2
            if gap<separation_m:raise ValueError('three-object initial overlap or insufficient separation')
    exact=support.get('exact_static_binding')
    if exact is not None:
        if root is None:raise ValueError('exact-mesh initial placement requires the resource root')
        from tools.scene_rules.exact_mesh_support import validate_exact_mesh_sphere_placement
        return validate_exact_mesh_sphere_placement(root=root,support=support,objects=objects)
    return None


def validate_mixed_initial_layout(scene: dict, rules: dict, separation_m: float) -> None:
    """Conservative initial envelopes; actual contact evidence comes from physics."""
    from tools.core.primitive_support import initial_bounds
    objects=scene['simulation']['objects'];support=scene['simulation']['support'];bounds=support['safe_surface_bounds']
    centers=[];extents=[];tolerance=scene['qa']['limits']['maximum_initial_penetration_m']
    for obj in objects:
        relative_low,relative_high=initial_bounds(obj);half=(relative_high-relative_low)/2
        pos=np.asarray(obj['initial_state']['position_m'],dtype=float)
        if pos.shape!=(3,) or not np.isfinite(pos).all():raise ValueError('invalid primitive initial position')
        envelope_center=pos+(relative_low+relative_high)/2
        if any(not bounds[axis][0]+half[i]*rules['layout']['edge_clearance_in_object_radii']<=envelope_center[i]<=bounds[axis][1]-half[i]*rules['layout']['edge_clearance_in_object_radii'] for i,axis in enumerate(('x','y'))):
            raise ValueError('three-object initial layout exceeds support bounds')
        if abs(pos[2]+relative_low[2]-support['surface_center_z_m'])>1e-6:
            raise ValueError('three-object initial position does not rest on support')
        for collider in [*support['colliders'],*scene['environment_binding']['colliders']]:
            if not collider.get('collision_enabled',True):continue
            if collider['primitive']!='box':raise ValueError('mixed pilot requires analytic static box clearance')
            rotation=np.asarray(quaternion_matrix_wxyz(quaternion_wxyz_from_euler_degrees(collider['rotation_euler_degrees'])))
            delta=envelope_center-np.asarray(collider['position_m']);other=np.asarray(collider['size_m'])/2
            axes=[*np.eye(3),*rotation.T,*[np.cross(a,b) for a in np.eye(3) for b in rotation.T]]
            overlaps=[]
            for axis in axes:
                length=np.linalg.norm(axis)
                if length<1e-10:continue
                axis=axis/length
                overlaps.append(float(np.abs(axis)@half+np.abs(axis@rotation)@other-abs(axis@delta)))
            if min(overlaps)>tolerance:raise ValueError('initial primitive envelope intersects static collision geometry')
        centers.append(envelope_center);extents.append(half)
    for i in range(3):
        for j in range(i+1,3):
            if objects[i]['geometry']['type']==objects[j]['geometry']['type']=='sphere':
                gap=float(np.linalg.norm(centers[i]-centers[j])-extents[i][0]-extents[j][0])
            else:gap=float(np.linalg.norm(np.maximum(np.abs(centers[i]-centers[j])-extents[i]-extents[j],0)))
            if gap<separation_m:raise ValueError('initial primitive envelopes overlap or lack separation')
