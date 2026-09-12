"""Three-marble camera admission against the actual concave track and catch fixture."""
from contextlib import contextmanager
from itertools import product
from pathlib import Path
import numpy as np
from tools.core.hashing import sha256_file
from tools.core.rigid_geometry import quaternion_matrix_wxyz
from tools.motion_rules.three_object.marble import validate_marble_contract
from tools.physics.specialized_sphere_simulation import _marble_fixture
from tools.rendering.static_fixture_rays import static_fixture_rays
from tools.rendering.three_object_camera import audit_three_objects,solve_camera_candidates,sphere_trajectory_view
from tools.rendering.camera_solver import project_points


def fixture_framing_points(root,metadata,arrays):
    rules=metadata['three_object']['camera_rules'];fixture=metadata['physics']['fixture']
    required=set(rules['mandatory_fixture_components']);found=set();points=[]
    for component in fixture['mesh_components']:
        if component['id'] not in required:continue
        binding=component['collision'];path=Path(root)/binding['path']
        if sha256_file(path)!=binding['sha256']:raise ValueError('marble camera collision mesh hash changed')
        vertices=np.asarray([[float(v) for v in line.split()[1:4]] for line in path.read_text().splitlines() if line.split()[:1]==['v']])
        if vertices.ndim!=2 or vertices.shape[1]!=3 or not len(vertices) or not np.isfinite(vertices).all():raise ValueError('invalid marble camera mesh vertices')
        x,y,z,w=component['base_orientation_quaternion_xyzw'];rotation=np.asarray(quaternion_matrix_wxyz([w,x,y,z]))
        points.append((vertices*np.asarray(component['mesh_scale']))@rotation.T+component['base_position_m']);found.add(component['id'])
    for c in fixture['analytic_colliders']:
        if c['id'] not in required:continue
        x,y,z,w=c.get('orientation_quaternion_xyzw',[0.,0.,0.,1.]);rotation=np.asarray(quaternion_matrix_wxyz([w,x,y,z]))
        corners=np.asarray(list(product((-1.,1.),repeat=3)))*np.asarray(c['half_extents_m'])
        points.append(corners@rotation.T+c['position_m']);found.add(c['id'])
    if found!=required:raise ValueError('missing mandatory marble fixture components')
    # Include the actual floor support under any ball near its surface. The full
    # floor remains a physical/ray collider; unused distant boundary is not fitted.
    floor=next(c for c in fixture['analytic_colliders'] if c['id']=='safety_floor')
    if floor.get('orientation_quaternion_xyzw',[0.,0.,0.,1.])!=[0.,0.,0.,1.]:raise ValueError('marble safety floor must remain horizontal')
    top=floor['position_m'][2]+floor['half_extents_m'][2]
    positions=np.asarray(arrays['position_m'])
    for index,obj in enumerate(metadata['simulation']['objects']):
        radius=obj['collision_proxy']['radius_m'];near=positions[:,index][np.abs(positions[:,index,2]-(top+radius))<=2*radius]
        if len(near):
            support=np.repeat(near,4,axis=0);support[:,:2]+=np.tile(np.asarray(list(product((-radius,radius),repeat=2))),(len(near),1));support[:,2]=top;points.append(support)
    return np.concatenate(points)


@contextmanager
def marble_fixture_rays(root,metadata,arrays):
    validate_marble_contract(metadata)
    scene={'adapter_payload':{'fixture':metadata['physics']['fixture']}}
    with static_fixture_rays(lambda pb:_marble_fixture(pb,scene,root)[0]) as (blocked,_):
        yield blocked,fixture_framing_points(root,metadata,arrays)


def audit_marble_camera(metadata,arrays,camera,*,static_occlusion,fixture_points):
    if not callable(static_occlusion):raise ValueError('marble camera requires actual fixture ray evidence')
    contract=validate_marble_contract(metadata);rules=contract['camera_rules']
    if rules['reference']!='world_XYZ_front_positive_Y_up_positive_Z' or rules['whole_fixture_required'] is not False:raise ValueError('unsupported marble camera framing contract')
    points=np.asarray(fixture_points)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():raise ValueError('missing marble fixture bounds')
    audit=audit_three_objects(metadata,sphere_trajectory_view(metadata,arrays),camera,contract=contract,
        resolution=metadata['render']['resolution'],blockers=[],static_occlusion=static_occlusion)
    width,height=metadata['render']['resolution'];margin=rules['frame_margin_fraction']
    uv=project_points(points,np.asarray(camera['position_m']),np.asarray(camera['target_m']),camera['focal_length_mm'],camera['sensor_width_mm'],width/height)
    visible=bool(np.all((uv[:,:2]>=margin)&(uv[:,:2]<=1-margin)) and np.all((uv[:,2]>camera['clip_start_m'])&(uv[:,2]<camera['clip_end_m'])))
    audit['fixture_framing']={'passed':visible,'method':'all transformed concave mesh vertices, catch box corners and visited safety-floor support patches','point_count':len(points),'mandatory_components':rules['mandatory_fixture_components']}
    audit['passed']=bool(audit['passed'] and visible)
    audit['occlusion_method']='64 front-sphere samples against dynamic spheres, actual concave track meshes and all static boxes; translucent catch wall treated conservatively as opaque'
    audit['fixture_sha256']=contract['fixture_sha256']
    return audit


def solve_marble_camera(root,metadata,arrays):
    contract=validate_marble_contract(metadata)
    with marble_fixture_rays(root,metadata,arrays) as (blocked,points):
        camera=solve_camera_candidates(metadata,sphere_trajectory_view(metadata,arrays),contract=contract,
            resolution=metadata['render']['resolution'],inclined=False,framing_points=points,
            audit_fn=lambda m,_,c:audit_marble_camera(m,arrays,c,static_occlusion=blocked,fixture_points=points))
    camera.update(solver_version='three_object_marble_camera_v1',structure_context='marble_run_tracks_catch_and_visited_floor')
    camera['diagnostics']['view_coordinate_frame']='world XYZ; front is positive Y, world Z up'
    return camera
