"""Validate compact dynamic contact intervals against a declared execution span."""
import numpy as np


def validate_contact_event_evidence(evidence, ids, simulation_hz, steps, distance_tolerance_m):
    if (evidence.get('schema_version')!='physweep_three_object_contact_events_v1' or evidence.get('object_ids')!=ids
        or evidence.get('simulation_hz')!=simulation_hz or evidence.get('observed_substeps')!=steps+1
        or evidence.get('distance_tolerance_m')!=distance_tolerance_m):
        raise ValueError('incomplete or mismatched event observation')
    events=evidence.get('events')
    if not isinstance(events,list):raise ValueError('missing event intervals')
    previous={}
    for e in events:
        pair=tuple(e['object_ids']);start=e['start_substep'];end=e['end_substep']
        if (len(pair)!=2 or pair!=tuple(sorted(set(pair))) or not set(pair)<=set(ids)
            or type(start) is not int or type(end) is not int or not 0<=start<=end<=steps
            or start<=previous.get(pair,-2)+1):raise ValueError('invalid, overlapping, or unmerged event interval')
        previous[pair]=end
        if not np.isfinite(e['minimum_contact_distance_m']) or e['minimum_contact_distance_m']>evidence['distance_tolerance_m']:
            raise ValueError('event distance contradicts contact criterion')
        for key in ('linear_velocity_before_m_s','linear_velocity_after_start_m_s'):
            if set(e[key])!=set(pair) or any(np.asarray(e[key][o]).shape!=(3,) or not np.isfinite(e[key][o]).all() for o in pair):
                raise ValueError('event velocity evidence incomplete')
    if events!=sorted(events,key=lambda e:(e['start_substep'],e['object_ids'])):raise ValueError('events out of order')
    return events
