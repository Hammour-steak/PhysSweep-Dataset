"""Initial three-sphere billiards geometry, independent of execution and outcomes."""
from __future__ import annotations
import copy
import math
import numpy as np
from tools.core.hashing import sha256_json

SCHEMA = 'physweep_billiards_three_object_scene_v1'
IDS = ('object_a', 'object_b', 'object_c')
ROLES = ('P', 'Q', 'R')


def initial_states(radius: float, bed: dict, limits: dict, parameters: dict) -> dict:
    for key in ('speed_m_s', 'PQ_surface_gap_m', 'QR_surface_gap_m', 'lane_y_m'):
        value = parameters[key]
        if isinstance(value, bool) or not math.isfinite(value) or not limits[key][0] <= value <= limits[key][1]:
            raise ValueError(f'billiards initial parameter outside bound: {key}')
    heading = parameters['heading']
    if type(heading) is not int or heading not in limits['headings']:
        raise ValueError('unsupported billiards heading')
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('invalid billiards radius')
    center = bed['center_xy_m']; size = bed['size_xy_m']; z = float(bed['z_m'])
    if len(center) != 2 or len(size) != 2 or not all(math.isfinite(v) for v in [*center, *size, z]) or min(size) <= 0:
        raise ValueError('invalid billiards safe bed')
    xq = limits['Q_reference_x_m']
    xs = {'P': xq - 2 * radius - parameters['PQ_surface_gap_m'], 'Q': xq,
          'R': xq + 2 * radius + parameters['QR_surface_gap_m']}
    states = {}
    for role in ROLES:
        position = [center[0] + heading * xs[role], center[1] + parameters['lane_y_m'], z + radius]
        if any(abs(position[i] - center[i]) + radius + limits['support_edge_margin_m'] > size[i] / 2 for i in range(2)):
            raise ValueError('three-sphere initial layout exceeds billiards bed')
        states[role] = {'position_m': position, 'orientation_quaternion_wxyz': [1., 0., 0., 0.],
                        'linear_velocity_m_s': [heading * parameters['speed_m_s'] if role == 'P' else 0., 0., 0.],
                        'angular_velocity_rad_s': copy.deepcopy(limits['angular_velocity_rad_s'])}
    return states


def validate_billiards_contract(metadata: dict) -> dict:
    if metadata.get('schema_version') != SCHEMA:
        raise ValueError('wrong three-object billiards schema')
    contract = metadata.get('three_object', {})
    if contract.get('schema_version') != 'physweep_three_object_billiards_motion_contract_v1':
        raise ValueError('missing explicit three-object billiards contract')
    if sha256_json({k: v for k, v in contract.items() if k != 'contract_sha256'}) != contract.get('contract_sha256'):
        raise ValueError('billiards motion contract hash mismatch')
    objects = metadata['simulation']['objects']; roles = contract['roles']
    if [o['object_id'] for o in objects] != list(IDS) or set(roles) != set(ROLES) or set(roles.values()) != set(IDS):
        raise ValueError('invalid three-object billiards identity or role order')
    template = contract['template']
    if template['id'] != 'chain_transfer' or template['initially_moving_roles'] != ['P']:
        raise ValueError('unimplemented billiards three-object motion')
    if (template['required_pairs'] != [['P','Q'],['Q','R']] or template['allowed_pairs'] != [['P','Q'],['Q','R']]
            or template['first_contact_order'] != [['P','Q'],['Q','R']]):
        raise ValueError('billiards chain relation was changed')
    if metadata['semantics']['dynamic_object_count'] != 3 or metadata['semantics']['scene_family'] != 'billiards':
        raise ValueError('billiards semantics differ from execution scope')
    radius = float(objects[0]['collision_proxy']['radius_m'])
    bed = metadata['physics']['static_support_binding']['target_support_frame']['safe_surface']
    expected = initial_states(radius, bed, contract['initial_limits'], contract['initial_parameters'])
    by_id = {o['object_id']: o for o in objects}
    for role, oid in roles.items():
        obj = by_id[oid]
        if obj['collision_proxy'] != {'type':'sphere', 'radius_m':radius} or obj['geometry'] != {'type':'sphere', 'size_m':[2 * radius] * 3}:
            raise ValueError('billiards sphere geometry changed or mismatched')
        if obj['initial_state'] != expected[role]:
            raise ValueError('billiards initial state contradicts frozen construction')
        if any(not np.isfinite(float(v)) for v in obj['material'].values()):
            raise ValueError('billiards material must be finite')
    slots = [o['visual_profile']['material_slot'] for o in objects]
    if sorted(slots) != ['cue_ball', 'object_ball_1', 'object_ball_2']:
        raise ValueError('three billiards materials need explicit complete slot assignment')
    return contract
