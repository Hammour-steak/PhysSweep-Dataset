"""Billiards camera adapter over common projection and three-object visibility."""
import copy
import numpy as np
from contextlib import contextmanager
from tools.assets.static_support_proxy import create_pybullet_static_support
from tools.motion_rules.three_object.billiards import validate_billiards_contract
from tools.rendering.three_object_camera import audit_three_objects, solve_camera_candidates, sphere_trajectory_view as trajectory_view
from tools.rendering.static_fixture_rays import exact_static_support_rays

@contextmanager
def static_support_rays(root,binding):
    """Compatibility boundary for billiards mesh construction and ray tests."""
    with exact_static_support_rays(
        root,binding,create_support=create_pybullet_static_support
    ) as blocked:
        yield blocked


def audit_billiards_camera(metadata,arrays,camera,*,static_occlusion):
    if not callable(static_occlusion):raise ValueError('billiards camera requires actual static-mesh ray evidence')
    contract=validate_billiards_contract(metadata)
    audit=audit_three_objects(metadata,trajectory_view(metadata,arrays),camera,contract=contract,
        resolution=metadata['render']['resolution'],blockers=[],static_occlusion=static_occlusion)
    audit['occlusion_method']='64 front-sphere samples against actual dynamic spheres and hashed transformed static triangle mesh rays'
    audit['static_support_binding']=copy.deepcopy(metadata['physics']['static_support_binding'])
    return audit


def solve_billiards_camera(root,metadata,arrays):
    contract=validate_billiards_contract(metadata)
    heading=contract['initial_parameters']['heading']
    with static_support_rays(root,metadata['physics']['static_support_binding']) as blocked:
        def audit_fn(m,_,camera):return audit_billiards_camera(m,arrays,camera,static_occlusion=blocked)
        camera=solve_camera_candidates(metadata,trajectory_view(metadata,arrays),contract=contract,
            resolution=metadata['render']['resolution'],inclined=False,audit_fn=audit_fn,
            direction_rotation=np.diag([heading,heading,1.]))
    camera.update(solver_version='three_object_billiards_camera_v1',structure_context='billiards_table')
    camera['diagnostics']['view_coordinate_frame']='initial P approach along local +X, world +Z up'
    return camera
