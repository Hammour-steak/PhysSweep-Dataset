"""Group input invariants; response trajectories deliberately are not compared."""
from __future__ import annotations
import copy
from tools.core.sweep_values import SWEEP_AXES,SWEEP_DERIVED_LEVELS,sweep_target_indices
from tools.core.integration_configuration import generic_integration_configuration


def validate_group_inputs(parent: dict, members: list[dict], target_indices: list[int]) -> dict:
    if sweep_target_indices(3,target_indices)!=(0,) or len(members)!=13:
        raise ValueError('three-object group must have one explicit target and 13 members')
    expected={(axis,level) for axis in SWEEP_AXES for level in SWEEP_DERIVED_LEVELS}
    specialized=parent.get('schema_version') in {'physweep_billiards_three_object_scene_v1','physweep_passive_pinball_three_object_scene_v1','physweep_marble_run_three_object_scene_v1'}
    if specialized:
        time_source=parent['physics'] if parent['schema_version']=='physweep_billiards_three_object_scene_v1' else parent['simulation']['time']
        integration={key:time_source[key] for key in ('duration_s','output_fps','simulation_hz','frame_count')}
        if integration['output_fps']<=0 or integration['simulation_hz']<=0 or integration['simulation_hz']%integration['output_fps']:
            raise ValueError('billiards group time is not frame aligned')
    else:integration=generic_integration_configuration(parent)
    seen=set();base_count=0;ids=set()
    for member in members:
        if member['scene_id'] in ids:raise ValueError('duplicate group member')
        ids.add(member['scene_id']);sweep=member['sweep'];simulation=copy.deepcopy(member['simulation'])
        if sweep['kind']=='base':base_count+=1
        elif sweep['kind']=='sweep':
            if (sweep['target_object_id'],sweep['target_object_index'])!=('object_a',0):raise ValueError('unexpected three-object sweep target')
            key=(sweep['axis'],sweep['level_index'])
            if key not in expected or key in seen:raise ValueError('invalid or duplicate intervention level')
            seen.add(key);axis=sweep['axis'];value=simulation['objects'][0]['material'][axis]
            if value!=sweep['value'] or value==parent['simulation']['objects'][0]['material'][axis]:raise ValueError('intervention value not applied or equals base')
            simulation['objects'][0]['material'][axis]=parent['simulation']['objects'][0]['material'][axis]
        else:raise ValueError('unknown sweep kind')
        if specialized:
            if member['physics']!=parent['physics'] or member['render']!=parent['render']:
                raise ValueError('billiards group fixture, render or execution changed')
            expected_materials=[{'object_id':obj['object_id'],'object_index':i,'material':{
                key:obj['material'][key] for key in ('mass_kg','contact_friction','contact_restitution')}}
                for i,obj in enumerate(member['simulation']['objects'])]
            if sweep.get('resolved_object_physics')!=expected_materials:
                raise ValueError('billiards group material binding disagrees with object inputs')
            member_integration=integration
        else:member_integration=generic_integration_configuration(member)
        if simulation!=parent['simulation'] or member_integration!=integration:
            raise ValueError('nonintervention physics input or effective integration changed')
        for key in ('appearance','environment_binding','render_request','camera_request','three_object','semantic_sampling','source_binding','visual_resource_binding','coverage'):
            if member.get(key)!=parent.get(key):raise ValueError(f'group invariant changed: {key}')
        if member['object_identity']['text']!=parent['object_identity']['text']:raise ValueError('group caption binding changed')
    if base_count!=1 or seen!=expected:raise ValueError('incomplete three-object group')
    return {'passed':True,'base_count':1,'sweep_count':12,'target_object_indices':[0],'integration':integration}
