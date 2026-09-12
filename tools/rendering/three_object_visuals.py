"""Small three-object visual contracts shared by binding and rendering."""
import math


def event_inspection_frames(evidence: dict, time: dict) -> list[int]:
    count=int(time['frame_count']);fps=float(time['output_fps'])
    hz=float(evidence['simulation_hz'])
    frames={1,(count+1)//2,count}
    # First contact per edge defines the two declared chain events. Later
    # recontacts remain in the audit, without an unbounded inspection pass.
    seen=set()
    for event in evidence['events']:
        pair=tuple(event['object_ids'])
        if pair in seen:continue
        seen.add(pair)
        time_s=event['start_substep']/hz
        before=math.floor(time_s*fps)+1
        after=math.ceil(time_s*fps)+1
        frames.update(max(1,min(count,f)) for f in (before-1,before,after,after+1))
    return sorted(frames)


def frozen_lighting_report(render: dict) -> dict:
    if render.get('three_object_lighting_policy')!='frozen_shared_preset_v1':
        raise ValueError('three-object render requires a frozen lighting policy')
    if render.get('use_motion_blur') is not False:
        raise ValueError('three-object initial frame must not sample future motion')
    exposure=float(render['color_management']['exposure'])
    if not math.isfinite(exposure):raise ValueError('nonfinite frozen exposure')
    return {'policy':'frozen_shared_preset_v1','result_exposure_ev':exposure,
            'world_strength_scale':1.0,'fill_light_scale':1.0,'exposure_delta_ev':0.0,
            'rendered_frame_exposure':{'policy':'disabled_for_frozen_group','final_exposure_ev':exposure}}
