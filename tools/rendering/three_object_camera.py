"""Base-only camera admission for declared generic three-object scenes."""
from __future__ import annotations

from contextlib import ExitStack
import json
import math
import numpy as np

from tools.motion_rules.three_object.motion import validate_motion_contract
from tools.dataset_contract.trajectory_contract import object_trajectory_view
from tools.rendering.camera_solver import (
    _trajectory_aabb_corners, camera_occlusion_colliders, project_points,
    segments_intersect_box,
)


class ThreeObjectCameraAdmissionError(ValueError):
    """All finite camera candidates failed base observation requirements."""


def sphere_trajectory_view(metadata, arrays):
    result=object_trajectory_view(metadata, arrays)
    for obj in metadata['simulation']['objects']:
        if obj['collision_proxy']['type']!='sphere':raise ValueError('specialized camera requires spherical proxies')
        oid=obj['object_id'];radius=obj['collision_proxy']['radius_m']
        result[oid+'__aabb_min_m']=np.asarray(result[oid+'__position_m'])-radius
        result[oid+'__aabb_max_m']=np.asarray(result[oid+'__position_m'])+radius
    return result


def blocked_by_sphere(camera: np.ndarray, points: np.ndarray, center: np.ndarray,
                      radius: float) -> np.ndarray:
    """Intersect open sight segments with an actual spherical proxy."""
    rays=points-camera
    relative=camera-center
    a=np.sum(rays*rays,axis=1)
    b=2*(rays@relative)
    c=float(relative@relative-radius*radius)
    discriminant=b*b-4*a*c
    hit=np.zeros(len(points),dtype=bool)
    valid=discriminant>=0
    roots=(-b[valid]-np.sqrt(discriminant[valid]))/(2*a[valid])
    hit[valid]=(roots>0)&(roots<1-1e-6)
    if c<0:hit[:]=True
    return hit


def sphere_front_samples(camera: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    toward=camera-center
    distance=float(np.linalg.norm(toward))
    if distance<=radius:raise ValueError('camera lies inside a dynamic object')
    toward/=distance
    right=np.cross(toward,[0.,0.,1.])
    if np.linalg.norm(right)<1e-8:right=np.array([1.,0.,0.])
    right/=np.linalg.norm(right)
    up=np.cross(toward,right)
    # Equal projected-disc area samples; this is a proxy estimate, not a mask area.
    indices=np.arange(64,dtype=float)+.5
    radial=np.sqrt(indices/64)*.95
    theta=indices*(math.pi*(3-math.sqrt(5)))
    tangent=radial[:,None]*(np.cos(theta)[:,None]*right+np.sin(theta)[:,None]*up)
    return center+radius*(tangent+np.sqrt(1-radial*radial)[:,None]*toward)


def projected_frames_inside(projected: np.ndarray, camera: dict, margin: float) -> np.ndarray:
    """Return whether every physical-envelope point is inside each frame bound."""
    projected=np.asarray(projected,dtype=float)
    if projected.ndim!=3 or projected.shape[2]!=3 or not np.isfinite(projected).all():
        raise ValueError('projected envelopes must be a finite frame x point x 3 array')
    inside=np.all((projected[:,:,:2]>=margin)&(projected[:,:,:2]<=1-margin),axis=(1,2))
    inside&=np.all((projected[:,:,2]>camera['clip_start_m'])&(projected[:,:,2]<camera['clip_end_m']),axis=1)
    return inside


def projected_frame_overflow(projected: np.ndarray) -> np.ndarray:
    """Maximum normalized image-boundary overflow for every frame."""
    projected=np.asarray(projected,dtype=float)
    if projected.ndim!=3 or projected.shape[2]!=3 or not np.isfinite(projected).all():
        raise ValueError('projected envelopes must be a finite frame x point x 3 array')
    uv=projected[:,:,:2]
    return np.maximum.reduce((
        np.maximum(0.,-np.min(uv[:,:,0],axis=1)),
        np.maximum(0.,np.max(uv[:,:,0],axis=1)-1.),
        np.maximum(0.,-np.min(uv[:,:,1],axis=1)),
        np.maximum(0.,np.max(uv[:,:,1],axis=1)-1.),
    ))


def required_event_frame_indices(metadata: dict, trajectory: dict, contract: dict) -> list[int]:
    """Locate required collision keyframes from the admitted physical trajectories."""
    objects={obj['object_id']:obj for obj in metadata['simulation']['objects']}
    frame_count=len(np.asarray(trajectory[next(iter(objects))+'__position_m']))
    padding=int(contract['camera_rules']['event_frame_padding'])
    selected={0}
    encoded=trajectory.get('three_object_event_evidence_json')
    if encoded is not None:
        evidence=json.loads(str(np.asarray(encoded).item()))
        hz=evidence.get('simulation_hz');events=evidence.get('events')
        if (evidence.get('schema_version')!='physweep_three_object_contact_events_v1'
            or type(hz) is not int or hz<=0 or not isinstance(events,list)):
            raise ValueError('invalid three-object event evidence for camera keyframes')
        fps=float(metadata['simulation']['time']['output_fps'])
        first_by_pair={}
        for event in events:
            pair=tuple(event.get('object_ids',()))
            step=event.get('start_substep')
            if (len(pair)!=2 or pair!=tuple(sorted(pair)) or not set(pair)<=set(objects)
                or type(step) is not int or step<0):
                raise ValueError('invalid contact event for camera keyframes')
            first_by_pair.setdefault(pair,step)
        for first_role,second_role in contract.get('template',{}).get('required_pairs',[]):
            pair=tuple(sorted((contract['roles'][first_role],contract['roles'][second_role])))
            if pair not in first_by_pair:
                raise ValueError('required contact event is missing from camera evidence')
            frame_position=first_by_pair[pair]*fps/hz
            before=int(math.floor(frame_position));after=int(math.ceil(frame_position))
            selected.update(range(max(0,before-padding),min(frame_count,after+padding+1)))
        return sorted(selected)
    for first_role,second_role in contract.get('template',{}).get('required_pairs',[]):
        first_id=contract['roles'][first_role];second_id=contract['roles'][second_role]
        first_lower=np.asarray(trajectory[first_id+'__aabb_min_m'],dtype=float)
        first_upper=np.asarray(trajectory[first_id+'__aabb_max_m'],dtype=float)
        second_lower=np.asarray(trajectory[second_id+'__aabb_min_m'],dtype=float)
        second_upper=np.asarray(trajectory[second_id+'__aabb_max_m'],dtype=float)
        if any(values.shape!=(frame_count,3) or not np.isfinite(values).all()
               for values in (first_lower,first_upper,second_lower,second_upper)):
            raise ValueError('camera event fallback requires finite frame-aligned object bounds')
        gap=np.maximum(np.maximum(first_lower-second_upper,second_lower-first_upper),0.)
        event=int(np.argmin(np.linalg.norm(gap,axis=1)))
        selected.update(range(max(0,event-padding),min(frame_count,event+padding+1)))
    return sorted(selected)


def object_framing_passes(summary: dict, rules: dict) -> bool:
    """Apply the declared two-sided size and bounded-overflow policy."""
    return bool(
        summary['minimum_extent_fraction_of_short_side']>=rules['minimum_object_extent_fraction_of_short_side']
        and summary['maximum_extent_fraction_of_short_side']<=rules['maximum_object_extent_fraction_of_short_side']
        and summary['safe_frame_violation_fraction']<=rules['maximum_safe_frame_violation_fraction']
        and summary['out_of_frame_fraction']<=rules['maximum_out_of_frame_fraction']
        and summary['maximum_frame_overflow_fraction']<=rules['maximum_frame_overflow_fraction']
        and summary['required_event_frames_inside']
        and summary['minimum_unoccluded_proxy_sample_fraction']>=1-rules['maximum_occluded_fraction']
    )


def audit_camera(metadata: dict, trajectory: dict, camera: dict, *, root=None) -> dict:
    contract=validate_motion_contract(metadata)
    compound=any(obj.get('collision_profile',{}).get('type')=='compound' for obj in metadata['simulation']['objects'])
    exact=metadata['simulation']['support'].get('exact_static_binding')
    if compound or exact is not None:
        from tools.rendering.three_object_primitives import pybullet_compound_ray_world
        from tools.rendering.static_fixture_rays import exact_static_support_rays
        if exact is not None and root is None:raise ValueError('exact-mesh camera audit requires the resource root')
        with ExitStack() as stack:
            compound_occlusion=(stack.enter_context(pybullet_compound_ray_world(metadata['simulation']['objects']))
                                if compound else None)
            static_occlusion=(stack.enter_context(exact_static_support_rays(root,exact))
                              if exact is not None else None)
            return audit_three_objects(metadata, trajectory, camera, contract=contract,
                resolution=metadata['render_request']['resolution'], blockers=camera_occlusion_colliders(metadata),
                static_occlusion=static_occlusion,compound_occlusion=compound_occlusion)
    return audit_three_objects(metadata, trajectory, camera, contract=contract,
        resolution=metadata['render_request']['resolution'], blockers=camera_occlusion_colliders(metadata))


def audit_three_objects(metadata, trajectory, camera, *, contract, resolution, blockers, static_occlusion=None, compound_occlusion=None):
    rules=contract['camera_rules']
    bounded_scale_policy='minimum_object_extent_fraction_of_short_side' in rules
    objects=metadata['simulation']['objects']
    ids=[o['object_id'] for o in objects]
    radii=np.array([o['geometry']['size_m'][0]/2 for o in objects])
    mixed=any(o['geometry']['type']!='sphere' or o.get('collision_profile',{}).get('type')=='compound' for o in objects)
    if mixed:
        from tools.rendering.three_object_primitives import blocked_by_primitive,compound_surface_samples,surface_samples
        quaternions={o['object_id']:np.asarray(trajectory[o['object_id']+'__quaternion_wxyz']) for o in objects if o['geometry']['type']!='sphere'}
    positions=np.stack([trajectory[i+'__position_m'] for i in ids],axis=1)
    if positions.ndim!=3 or positions.shape[1:]!=(3,3) or not np.isfinite(positions).all():
        raise ValueError('three-object camera requires finite object trajectories')
    width,height=resolution
    short_side=float(min(width,height))
    position=np.asarray(camera['position_m'],dtype=float)
    target=np.asarray(camera['target_m'],dtype=float)
    event_frames=required_event_frame_indices(metadata,trajectory,contract) if bounded_scale_policy else [0]
    summaries={}
    for index,(object_id,radius) in enumerate(zip(ids,radii)):
        lower=np.asarray(trajectory[object_id+'__aabb_min_m'])
        upper=np.asarray(trajectory[object_id+'__aabb_max_m'])
        corners=_trajectory_aabb_corners(lower,upper)
        projected=project_points(corners.reshape(-1,3),position,target,camera['focal_length_mm'],
                                 camera['sensor_width_mm'],width/height).reshape(-1,8,3)
        margin=rules['frame_margin_fraction']
        safe_inside=projected_frames_inside(projected,camera,margin)
        image_inside=projected_frames_inside(projected,camera,0.)
        event_inside=projected_frames_inside(projected,camera,rules.get('event_frame_margin_fraction',margin))
        overflow=projected_frame_overflow(projected)
        short_extents=[];long_extents=[];characteristic_extents=[];visible=[]
        for frame in range(len(positions)):
            shape=objects[index]['geometry']['type']
            profile_type=objects[index].get('collision_profile',{}).get('type',shape)
            if shape=='sphere' and profile_type=='sphere':
                samples=sphere_front_samples(position,positions[frame,index],float(radius));weights=np.ones(len(samples))
            elif profile_type=='compound':
                samples,weights=compound_surface_samples(objects[index]['collision_profile']['colliders'],positions[frame,index],quaternions[object_id][frame],position)
            else:
                samples,weights=surface_samples(shape,objects[index]['geometry']['size_m'],positions[frame,index],quaternions[object_id][frame],position)
            uv=project_points(samples,position,target,camera['focal_length_mm'],camera['sensor_width_mm'],width/height)
            if profile_type=='compound':
                # A rotating world's AABB contains empty corners and can report a
                # margin violation even when the whole compound is safely framed.
                # Use the same exact child-union surface as the compound extent
                # and visibility audit instead of those nonphysical corners.
                surface=uv[None,:,:]
                safe_inside[frame]=projected_frames_inside(surface,camera,margin)[0]
                image_inside[frame]=projected_frames_inside(surface,camera,0.)[0]
                event_inside[frame]=projected_frames_inside(surface,camera,rules.get('event_frame_margin_fraction',margin))[0]
                overflow[frame]=projected_frame_overflow(surface)[0]
            span_x=float(np.ptp(uv[:,0])*width);span_y=float(np.ptp(uv[:,1])*height)
            short_extents.append(min(span_x,span_y));long_extents.append(max(span_x,span_y))
            characteristic_extents.append(math.sqrt(span_x*span_y))
            blocked=np.zeros(len(samples),dtype=bool)
            for collider in blockers:blocked|=segments_intersect_box(position,samples,collider)
            if static_occlusion is not None:blocked|=static_occlusion(position,samples)
            for other in range(3):
                if other==index:continue
                other_shape=objects[other]['geometry']['type']
                other_profile=objects[other].get('collision_profile',{}).get('type',other_shape)
                if other_shape=='sphere' and other_profile=='sphere':blocked|=blocked_by_sphere(position,samples,positions[frame,other],float(radii[other]))
                elif other_profile=='compound':
                    if compound_occlusion is None:raise ValueError('compound camera audit requires actual PyBullet ray evidence')
                    blocked|=compound_occlusion(ids[other],position,samples,positions[frame,other],quaternions[ids[other]][frame])
                else:blocked|=blocked_by_primitive(position,samples,positions[frame,other],quaternions[ids[other]][frame],other_shape,objects[other]['geometry']['size_m'])
            visible.append(float(np.mean(~blocked)) if shape=='sphere' and profile_type=='sphere' else float(np.sum(weights[~blocked])/np.sum(weights)))
        minimum_extent=min(short_extents);maximum_extent=max(long_extents)
        safe_violation=float(np.mean(~safe_inside));image_violation=float(np.mean(~image_inside))
        maximum_overflow=float(max(overflow));events_visible=bool(np.all(event_inside[event_frames]))
        if not bounded_scale_policy:
            summaries[object_id]={
                'minimum_extent_px':minimum_extent,
                'out_of_frame_fraction':safe_violation,
                'minimum_unoccluded_proxy_sample_fraction':min(visible),
                'per_frame_unoccluded_proxy_sample_fraction':visible,
                'passed':bool(np.all(safe_inside) and minimum_extent>=rules['minimum_object_extent_px']
                              and min(visible)>=1-rules['maximum_occluded_fraction']),
            }
            continue
        summary={
            'minimum_extent_px':minimum_extent,
            'median_characteristic_extent_px':float(np.median(characteristic_extents)),
            'maximum_extent_px':maximum_extent,
            'minimum_extent_fraction_of_short_side':minimum_extent/short_side,
            'median_characteristic_extent_fraction_of_short_side':float(np.median(characteristic_extents))/short_side,
            'maximum_extent_fraction_of_short_side':maximum_extent/short_side,
            'safe_frame_violation_fraction':safe_violation,
            'out_of_frame_fraction':image_violation,
            'maximum_frame_overflow_fraction':maximum_overflow,
            'required_event_frames_inside':events_visible,
            'minimum_unoccluded_proxy_sample_fraction':min(visible),
            'per_frame_safe_inside':safe_inside.tolist(),
            'per_frame_image_inside':image_inside.tolist(),
            'per_frame_unoccluded_proxy_sample_fraction':visible,
        }
        summary['passed']=object_framing_passes(summary,rules)
        summaries[object_id]=summary
    exact_mesh=metadata.get('simulation',{}).get('support',{}).get('exact_static_binding') is not None
    return {'schema_version':'physweep_three_object_camera_audit_v2' if bounded_scale_policy else 'physweep_three_object_camera_audit_v1','objects':summaries,
            'frame_count':len(positions),**({'required_event_frame_indices':event_frames} if bounded_scale_policy else {}),
            'passed':all(r['passed'] for r in summaries.values()),
            'occlusion_method':('sphere discs plus projected-area-weighted primitive or coaxial-compound surfaces against actual PyBullet compound children, analytic primitive children, static-box proxies and exact static-support mesh rays' if mixed and exact_mesh else
                                'sphere discs plus projected-area-weighted primitive or coaxial-compound surfaces against actual PyBullet compound children, analytic primitive children and static-box proxies' if mixed else
                                '64 front-sphere samples against dynamic spheres, static-box proxies and exact static-support mesh rays' if exact_mesh else
                                '64 front-sphere samples against sphere and static-box proxies'),
            'manual_keyframe_review_required':True}


def solve_three_object_camera(metadata: dict, trajectory: dict, *, root=None) -> dict:
    contract=validate_motion_contract(metadata)
    compound=any(obj.get('collision_profile',{}).get('type')=='compound' for obj in metadata['simulation']['objects'])
    exact=metadata['simulation']['support'].get('exact_static_binding')
    direction=np.asarray(contract['approach_direction_world'],dtype=float)
    direction_rotation=(None if np.allclose(direction,[1.,0.,0.],rtol=0,atol=1e-12) else
                        np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]))
    if compound or exact is not None:
        from tools.rendering.three_object_primitives import pybullet_compound_ray_world
        from tools.rendering.static_fixture_rays import exact_static_support_rays
        if exact is not None and root is None:raise ValueError('exact-mesh camera solve requires the resource root')
        with ExitStack() as stack:
            compound_occlusion=(stack.enter_context(pybullet_compound_ray_world(metadata['simulation']['objects']))
                                if compound else None)
            static_occlusion=(stack.enter_context(exact_static_support_rays(root,exact))
                              if exact is not None else None)
            def compound_audit(current_metadata,current_trajectory,camera):
                return audit_three_objects(current_metadata,current_trajectory,camera,contract=contract,
                    resolution=current_metadata['render_request']['resolution'],blockers=camera_occlusion_colliders(current_metadata),
                    static_occlusion=static_occlusion,compound_occlusion=compound_occlusion)
            return solve_camera_candidates(metadata, trajectory, contract=contract, resolution=metadata['render_request']['resolution'],
                inclined=metadata['simulation']['support'].get('support_shape')=='inclined_ramp', audit_fn=compound_audit,
                direction_rotation=direction_rotation)
    return solve_camera_candidates(metadata, trajectory, contract=contract, resolution=metadata['render_request']['resolution'],
        inclined=metadata['simulation']['support'].get('support_shape')=='inclined_ramp', audit_fn=audit_camera,
        direction_rotation=direction_rotation)


def camera_selection_score(audit: dict, rules: dict) -> float:
    """Prefer a consistent physical-object scale among already admissible views."""
    if not audit['passed']:raise ValueError('only admissible cameras have a selection score')
    target=float(rules['target_object_extent_fraction_of_short_side'])
    values=np.asarray([
        record['median_characteristic_extent_fraction_of_short_side']
        for record in audit['objects'].values()
    ],dtype=float)
    extent_error=float(np.mean((np.log(values/target))**2))
    scale_spread=float(np.var(np.log(values)))
    safe_frame_cost=float(sum(
        record['safe_frame_violation_fraction'] for record in audit['objects'].values()
    ))
    return extent_error+.25*scale_spread+.1*safe_frame_cost


def solve_camera_candidates(metadata, trajectory, *, contract, resolution, inclined, audit_fn, direction_rotation=None, framing_points=None):
    if metadata.get('sweep',{}).get('kind')=='sweep':
        raise ValueError('a sweep must reuse its base camera')
    rules=contract['camera_rules']
    requested=metadata['camera_request']['requested_view_family']
    families=rules['view_families']
    bounded_scale_policy='candidate_distance_factors' in rules
    if requested not in {f['id'] for f in families}:raise ValueError('unknown three-object view family')
    families=sorted(families,key=lambda family:family['id']!=requested)
    ids=[o['object_id'] for o in metadata['simulation']['objects']]
    lower_by_object=[np.asarray(trajectory[i+'__aabb_min_m']) for i in ids]
    upper_by_object=[np.asarray(trajectory[i+'__aabb_max_m']) for i in ids]
    lower=np.concatenate(lower_by_object);upper=np.concatenate(upper_by_object)
    bounds=np.array([lower.min(axis=0),upper.max(axis=0)])
    target=bounds.mean(axis=0)
    corners=_trajectory_aabb_corners(bounds[:1],bounds[1:])[0]
    if bounded_scale_policy:
        event_frames=required_event_frame_indices(metadata,trajectory,contract)
        focus_lower=np.concatenate([values[event_frames] for values in lower_by_object])
        focus_upper=np.concatenate([values[event_frames] for values in upper_by_object])
        focus_bounds=np.array([focus_lower.min(axis=0),focus_upper.max(axis=0)])
        target=focus_bounds.mean(axis=0)
        corners=_trajectory_aabb_corners(focus_bounds[:1],focus_bounds[1:])[0]
        if inclined:
            corners=_trajectory_aabb_corners(focus_lower,focus_upper).reshape(-1,3)
    elif inclined:
        # Fit every actual per-frame object envelope. Empty corners of the
        # global world-axis box can unnecessarily shrink small ramp objects.
        corners=_trajectory_aabb_corners(lower,upper).reshape(-1,3)
    if framing_points is not None:
        static_points=np.asarray(framing_points,dtype=float)
        if static_points.ndim!=2 or static_points.shape[1]!=3 or not len(static_points) or not np.isfinite(static_points).all():
            raise ValueError('camera fixture framing points must be finite N x 3')
        corners=np.concatenate([_trajectory_aabb_corners(lower,upper).reshape(-1,3),static_points])
        target=(corners.min(axis=0)+corners.max(axis=0))/2
    width,height=resolution
    factors=rules['candidate_distance_factors'] if bounded_scale_policy else (1.02,1.2,1.45,1.75)
    attempts=[]
    for family in families:
        admitted=[]
        azimuth=math.radians(family['azimuth_degrees']);elevation=math.radians(family['elevation_degrees'])
        outward=np.array([math.cos(azimuth)*math.cos(elevation),math.sin(azimuth)*math.cos(elevation),math.sin(elevation)])
        if direction_rotation is not None:outward=np.asarray(direction_rotation)@outward
        if inclined:
            from tools.core.inclined_support import inclined_frame
            normal=inclined_frame(metadata['simulation']['support'])['normal']
            forward=np.asarray(contract['approach_direction_world'])
            lateral=np.cross(normal,forward)
            outward=outward[0]*forward+outward[1]*lateral+outward[2]*normal
        right=np.cross(-outward,[0.,0.,1.]);right/=np.linalg.norm(right)
        up=np.cross(right,-outward)
        relative=corners-target
        for lens in rules['focal_length_mm']:
            half_horizontal=36/(2*lens)*(1-2*rules['frame_margin_fraction'])
            half_vertical=half_horizontal/(width/height)
            minimum=max(float(np.max(relative@outward+np.abs(relative@right)/half_horizontal)),
                        float(np.max(relative@outward+np.abs(relative@up)/half_vertical)))
            for factor in factors:
                if len(attempts)>=rules['maximum_candidate_count']:break
                camera={'solver_version':'three_object_base_camera_v2' if bounded_scale_policy else 'three_object_base_camera_v1','profile':family['id'],
                        'observation_intent':'three_object_events','structure_context':'horizontal_surface',
                        'position_m':(target+outward*max(minimum,.2)*factor).tolist(),'target_m':target.tolist(),
                        'focal_length_mm':lens,'sensor_width_mm':36.,'clip_start_m':.05,'clip_end_m':100.}
                if inclined:camera.update(solver_version='three_object_base_camera_surface_v3' if bounded_scale_policy else 'three_object_base_camera_surface_v2',structure_context='inclined_surface')
                audit=audit_fn(metadata,trajectory,camera)
                score=camera_selection_score(audit,rules) if bounded_scale_policy and audit['passed'] else None
                attempts.append({'family':family['id'],'focal_length_mm':lens,'distance_factor':factor,
                                 'passed':audit['passed'],'selection_score':score,
                                 'objects':{i:{k:v for k,v in a.items() if not k.startswith('per_frame')} for i,a in audit['objects'].items()}})
                if audit['passed']:
                    if not bounded_scale_policy:
                        camera['diagnostics']={'requested_view_family':requested,'selected_view_family':family['id'],
                            'fallback_reason':None if family['id']==requested else 'requested family had no admissible candidate',
                            'selected_azimuth_degrees':family['azimuth_degrees'],'selected_elevation_degrees':family['elevation_degrees'],
                            'motion_target_m':target.tolist(),'evaluated_candidates':len(attempts),'candidates':attempts,'admission':audit}
                        return camera
                    admitted.append((score,len(attempts)-1,camera,audit))
        if admitted:
            score,selected_index,camera,audit=min(admitted,key=lambda item:(item[0],item[1]))
            camera['diagnostics']={'requested_view_family':requested,'selected_view_family':family['id'],
                'fallback_reason':None if family['id']==requested else 'requested family had no admissible candidate',
                'selected_azimuth_degrees':family['azimuth_degrees'],'selected_elevation_degrees':family['elevation_degrees'],
                'motion_target_m':target.tolist(),'evaluated_candidates':len(attempts),
                'selected_candidate_index':selected_index,'selection_score':score,
                'candidates':attempts,'admission':audit}
            return camera
    raise ThreeObjectCameraAdmissionError(f'no admissible three-object base camera: {attempts}')
