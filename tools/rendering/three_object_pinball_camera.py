"""Whole-pinfield camera admission using actual box and cylinder ray geometry."""
import numpy as np
import copy,json
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from tools.core.hashing import sha256_file,sha256_json
from tools.core.rigid_geometry import quaternion_matrix_wxyz
from tools.motion_rules.three_object.pinball import validate_pinball_contract
from tools.physics.specialized_sphere_simulation import _pinball_fixture
from tools.rendering.static_fixture_rays import static_fixture_rays
from tools.rendering.three_object_camera import audit_three_objects,solve_camera_candidates,sphere_trajectory_view as trajectory_view
from tools.rendering.camera_solver import project_points


def camera_contract(metadata):
    contract = copy.deepcopy(validate_pinball_contract(metadata))
    path = Path(__file__).resolve().parents[2] / 'configs/three_object_pinball_camera.json'
    config = json.loads(path.read_text(encoding='utf-8'))
    if (config['schema_version'] != 'physweep_pinball_camera_v1'
            or config['source_camera_rules_sha256'] != sha256_json(contract['camera_rules'])):
        raise ValueError('pinball camera configuration differs from frozen source rules')
    if config['reference'] != 'world_XYZ_front_positive_Y_up_positive_Z':
        raise ValueError('unsupported pinball camera reference frame')
    families, lenses = config['view_families'], config['focal_length_mm']
    if [f['id'] for f in families] != [f['id'] for f in contract['camera_rules']['view_families']]:
        raise ValueError('pinball camera changes requested family identities')
    if any(set(f) != {'id', 'azimuth_degrees', 'elevation_degrees'}
           or not np.isfinite([f['azimuth_degrees'], f['elevation_degrees']]).all() for f in families):
        raise ValueError('invalid pinball camera angles')
    if not lenses or not np.isfinite(lenses).all() or min(lenses) <= 0:
        raise ValueError('invalid pinball camera lenses')
    contract['camera_rules'].update(reference=config['reference'],
                                  view_families=copy.deepcopy(families), focal_length_mm=list(lenses))
    return contract, {'path': str(path), 'sha256': sha256_file(path),
                      'source_camera_rules_sha256': config['source_camera_rules_sha256'],
                      'effective_camera_rules': copy.deepcopy(contract['camera_rules'])}


def fixture_framing_points(metadata):
    """Convex envelopes containing each true collider, without rotated-box AABB voids."""
    points=[]
    for c in metadata['physics']['fixture']['colliders']:
        if c['shape']=='box':
            local=np.asarray(list(product((-1.,1.),repeat=3)))*np.asarray(c['half_extents_m'])
        elif c['shape']=='cylinder':
            # A regular circumscribed 32-gon contains the whole circular section;
            # its prism therefore bounds every point of the actual cylinder.
            sides=32;angles=np.arange(sides)*2*np.pi/sides
            radius=float(c['radius_m'])/np.cos(np.pi/sides)
            local=np.asarray([[radius*np.cos(a),radius*np.sin(a),z] for z in (-c['length_m']/2,c['length_m']/2) for a in angles])
        else:raise ValueError('unsupported pinball camera fixture collider')
        x,y,z,w=c['orientation_quaternion_xyzw'];rotation=np.asarray(quaternion_matrix_wxyz([w,x,y,z]))
        points.append(local@rotation.T+np.asarray(c['position_m']))
    if not points:raise ValueError('empty pinball camera fixture')
    return np.concatenate(points)


@contextmanager
def pinball_fixture_rays(root,metadata):
    validate_pinball_contract(metadata)
    scene={'adapter_payload':{'fixture':metadata['physics']['fixture']}}
    with static_fixture_rays(lambda pb:_pinball_fixture(pb,scene,root)[0]) as (blocked,_):
        yield blocked,fixture_framing_points(metadata)


def audit_pinball_camera(metadata,arrays,camera,*,static_occlusion,fixture_points):
    if not callable(static_occlusion):raise ValueError('pinball camera requires actual fixture ray evidence')
    contract,binding=camera_contract(metadata);rules=contract['camera_rules']
    if rules.get('whole_fixture_required') is not True:raise ValueError('pinball camera requires the complete fixture')
    points=np.asarray(fixture_points)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
        raise ValueError('missing complete pinball fixture bounds')
    audit=audit_three_objects(metadata,trajectory_view(metadata,arrays),camera,contract=contract,
        resolution=metadata['render']['resolution'],blockers=[],static_occlusion=static_occlusion)
    w,h=metadata['render']['resolution'];margin=rules['frame_margin_fraction']
    uv=project_points(points,np.asarray(camera['position_m']),np.asarray(camera['target_m']),camera['focal_length_mm'],camera['sensor_width_mm'],w/h)
    visible=bool(np.all((uv[:,:2]>=margin)&(uv[:,:2]<=1-margin)) and np.all((uv[:,2]>camera['clip_start_m'])&(uv[:,2]<camera['clip_end_m'])))
    audit['fixture_framing']={'passed':visible,'method':'exact oriented box vertices and circumscribed 32-sided cylinder prisms inside frame margin and clipping range','point_count':len(points)}
    audit['passed']=bool(audit['passed'] and visible)
    audit['occlusion_method']='64 front-sphere samples against dynamic spheres and actual static pinboard boxes/cylinders; conservative convex envelopes used only for framing'
    audit['fixture_sha256']=contract['fixture_sha256']
    audit['camera_frame_binding']=binding
    return audit


def solve_pinball_camera(root,metadata,arrays):
    contract,binding=camera_contract(metadata)
    with pinball_fixture_rays(root,metadata) as (blocked,points):
        camera=solve_camera_candidates(metadata,trajectory_view(metadata,arrays),contract=contract,
            resolution=metadata['render']['resolution'],inclined=False,framing_points=points,
            audit_fn=lambda m,_,c:audit_pinball_camera(m,arrays,c,static_occlusion=blocked,fixture_points=points))
    camera.update(solver_version='three_object_pinball_camera_v1',structure_context='passive_pinball_full_fixture')
    camera['camera_frame_binding']=binding
    camera['diagnostics']['view_coordinate_frame']='world XYZ; front is positive Y, world Z up'
    return camera
