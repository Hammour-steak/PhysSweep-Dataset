"""Compact substep contact intervals, independent of trajectory point counts."""
from __future__ import annotations
import copy
import math


class ContactEventCollector:
    def __init__(self, object_ids: list[str], simulation_hz: int, distance_tolerance_m: float):
        if len(object_ids)!=3 or len(set(object_ids))!=3 or simulation_hz<=0:
            raise ValueError('contact collector requires three distinct objects and positive frequency')
        self.ids=tuple(object_ids);self.hz=simulation_hz;self.tolerance=distance_tolerance_m
        self.last_step=-1;self.active={};self.events=[];self.previous={}

    def observe(self, step: int, contacts: dict[tuple[str,str],float], velocities: dict[str,list[float]]) -> None:
        if step!=self.last_step+1 or set(velocities)!=set(self.ids):raise ValueError('missing or unordered contact observation')
        if any(len(v)!=3 or not all(math.isfinite(x) for x in v) for v in velocities.values()):raise ValueError('invalid event velocity')
        present={}
        for pair,distance in contacts.items():
            if len(pair)!=2 or pair[0]==pair[1] or not set(pair)<=set(self.ids) or not math.isfinite(distance):
                raise ValueError('invalid dynamic contact pair')
            key=tuple(sorted(pair))
            if distance<=self.tolerance:present[key]=min(present.get(key,distance),distance)
        for pair in list(self.active):
            if pair not in present:self.events.append(self.active.pop(pair))
        for pair,distance in present.items():
            if pair not in self.active:
                before=self.previous if self.previous else velocities
                self.active[pair]={'object_ids':list(pair),'start_substep':step,'end_substep':step,
                    'minimum_contact_distance_m':distance,
                    'linear_velocity_before_m_s':{o:list(before[o]) for o in pair},
                    'linear_velocity_after_start_m_s':{o:list(velocities[o]) for o in pair}}
            event=self.active[pair];event['end_substep']=step
            event['minimum_contact_distance_m']=min(event['minimum_contact_distance_m'],distance)
        self.last_step=step;self.previous=copy.deepcopy(velocities)

    def evidence(self) -> dict:
        events=sorted([*self.events,*self.active.values()],key=lambda e:(e['start_substep'],e['object_ids']))
        return {'schema_version':'physweep_three_object_contact_events_v1','object_ids':list(self.ids),
            'simulation_hz':self.hz,'observed_substeps':self.last_step+1,'distance_tolerance_m':self.tolerance,
            'events':copy.deepcopy(events)}
