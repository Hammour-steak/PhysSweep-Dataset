"""Three-body numerical checks and explicit base-only interaction semantics."""
from __future__ import annotations
import json
import numpy as np

from tools.core.rigid_geometry import declared_collision_descriptors, quaternion_matrix_wxyz
from tools.core.integration_configuration import generic_integration_configuration
from tools.core.rigid_dynamics import expected_object_inertia
from tools.motion_rules.three_object.motion import validate_motion_contract

OBJECT_CHECKS=('finite_state','initial_state','runtime_material','runtime_support','runtime_inertia','runtime_proxy','bounded_penetration')
HARD_GROUP_CHECKS=('time_axis','event_evidence','bounded_group_energy')
SEMANTIC_CHECKS=('required_contacts','allowed_contacts','event_order','event_window','template_motion_response')


def expected_check_ids(objects: list[dict]) -> dict[str,str]:
    return {**{f"{o['object_id']}__{name}":'integrity' for o in objects for name in OBJECT_CHECKS},
            **{name:'integrity' for name in HARD_GROUP_CHECKS},**{name:'base_semantics' for name in SEMANTIC_CHECKS}}


def audit_hard_results(objects: list[dict], audit: dict, is_sweep: bool) -> list[bool]:
    expected=expected_check_ids(objects);records=audit.get('checks')
    if not isinstance(records,list):raise ValueError('three-object audit checks are missing')
    by_id={r['id']:r for r in records}
    if len(by_id)!=len(records) or set(by_id)!=set(expected):raise ValueError('three-object audit check list differs from the required contract')
    if any(by_id[k].get('category')!=v for k,v in expected.items()):raise ValueError('three-object audit category mismatch')
    return [bool(by_id[k].get('passed')) for k,v in expected.items() if v=='integrity' or not is_sweep]


def validate_event_evidence(evidence: dict, metadata: dict) -> list[dict]:
    objects=metadata['simulation']['objects'];ids=[o['object_id'] for o in objects]
    integration=generic_integration_configuration(metadata)
    steps=(metadata['simulation']['time']['frame_count']-1)*integration['steps_per_frame']
    from tools.core.contact_event_contract import validate_contact_event_evidence
    return validate_contact_event_evidence(evidence, ids, integration['simulation_hz'], steps,
                                          metadata['three_object']['thresholds']['contact_distance_tolerance_m'])


def interaction_checks(contract: dict, events: list, hz: int, initial_velocity_by_id: dict) -> list:
    """The same declared graph/order/response policy over family-specific evidence."""
    checks=[]
    def check(name,passed,value,expected):
        checks.append({'id':name,'category':'base_semantics','passed':bool(passed),'value':value,'expected':expected})
    observed={tuple(e['object_ids']) for e in events};roles=contract['roles'];template=contract['template'];thresholds=contract['thresholds']
    def pair(p):return tuple(sorted(roles[r] for r in p))
    required={pair(p) for p in template['required_pairs']};allowed={pair(p) for p in template['allowed_pairs']}
    first={p:next(e for e in events if tuple(e['object_ids'])==p) for p in observed}
    check('required_contacts',required<=observed,[list(p) for p in sorted(observed)],[list(p) for p in sorted(required)])
    check('allowed_contacts',observed<=allowed,[list(p) for p in sorted(observed)],[list(p) for p in sorted(allowed)])
    ordered=[pair(p) for p in template['first_contact_order']]
    starts=[first[p]['start_substep']/hz for p in ordered if p in first]
    order_ok=len(starts)==len(ordered) and all(b-a>=thresholds['minimum_event_gap_s'] for a,b in zip(starts,starts[1:]))
    check('event_order',order_ok,starts,thresholds['minimum_event_gap_s'])
    window_ok=required<=observed and all(thresholds['first_event_min_s']<=first[p]['start_substep']/hz<=thresholds['last_event_max_s'] for p in required)
    check('event_window',window_ok,starts,[thresholds['first_event_min_s'],thresholds['last_event_max_s']])
    pq=first.get(pair(['P','Q']));qr=first.get(pair(['Q','R']));q_id=roles['Q']
    if template['id']=='chain_transfer':
        response=False
        if pq and qr and order_ok:
            before=np.linalg.norm(pq['linear_velocity_before_m_s'][q_id]);after=np.linalg.norm(pq['linear_velocity_after_start_m_s'][q_id])
            response=before<=thresholds['stationary_linear_speed_m_s'] and after-before>=thresholds['transfer_speed_gain_m_s']
        interpretation='Q initially rests, gains speed at P-Q before Q-R'
    else:
        role_speeds={role:float(np.linalg.norm(initial_velocity_by_id[roles[role]]))
                     for role in template['initially_moving_roles']}
        response=all(speed>=thresholds['moving_linear_speed_m_s'] for speed in role_speeds.values())
        interpretation='declared initial role motion; contact graph, order and window checked separately'
    check('template_motion_response',response,None,interpretation)
    return checks


def audit_three_object_motion(metadata: dict, trajectory: dict) -> dict:
    contract=validate_motion_contract(metadata);objects=metadata['simulation']['objects']
    classes=expected_check_ids(objects);checks=[];metrics={};energy=[]
    def check(name,passed,value,expected):
        checks.append({'id':name,'category':classes[name],'passed':bool(passed),'value':value,'expected':expected})
    time=np.asarray(trajectory['time_s']);count=metadata['simulation']['time']['frame_count']
    expected_time=np.arange(count)/metadata['simulation']['time']['output_fps']
    check('time_axis',time.shape==expected_time.shape and np.allclose(time,expected_time,rtol=0,atol=1e-9),len(time),count)
    tolerance=float(metadata['qa']['limits'].get('parameter_match_absolute_tolerance',1e-6))
    support=metadata['simulation']['support']['dynamics']
    gravity=np.asarray(metadata['simulation']['world']['gravity_m_s2'])
    for obj in objects:
        oid=obj['object_id'];prefix=oid+'__';initial=obj['initial_state'];material=obj['material']
        pos=np.asarray(trajectory[prefix+'position_m']);vel=np.asarray(trajectory[prefix+'linear_velocity_m_s'])
        angular=np.asarray(trajectory[prefix+'angular_velocity_rad_s']);q=np.asarray(trajectory[prefix+'quaternion_wxyz'])
        finite=all(a.shape==(count,n) and np.isfinite(a).all() for a,n in ((pos,3),(vel,3),(angular,3),(q,4)))
        check(prefix+'finite_state',finite,None,'finite F x channel arrays')
        if not finite:raise ValueError('invalid three-object trajectory state arrays')
        orientation_error=min(np.linalg.norm(q[0]-initial['orientation_quaternion_wxyz']),np.linalg.norm(q[0]+initial['orientation_quaternion_wxyz']))
        initial_ok=all(np.allclose(a[0],initial[k],rtol=0,atol=tolerance) for a,k in ((pos,'position_m'),(vel,'linear_velocity_m_s'),(angular,'angular_velocity_rad_s')))
        initial_ok=initial_ok and orientation_error<=tolerance and np.max(np.abs(np.linalg.norm(q,axis=1)-1))<=tolerance
        check(prefix+'initial_state',initial_ok,float(orientation_error),tolerance)
        actual=np.asarray(trajectory[prefix+'runtime_dynamics']);expected=np.asarray([material[k] for k in ('mass_kg','contact_friction','contact_restitution','rolling_friction','spinning_friction')])
        check(prefix+'runtime_material',actual.shape==expected.shape and np.allclose(actual,expected,rtol=0,atol=tolerance),actual.tolist(),expected.tolist())
        actual_support=np.asarray(trajectory[prefix+'runtime_support_dynamics']);expected_support=np.asarray([support['lateral_friction'],support['restitution']])
        check(prefix+'runtime_support',actual_support.ndim==2 and actual_support.shape[1]==2 and len(actual_support)>0 and np.allclose(actual_support,expected_support,rtol=0,atol=tolerance),actual_support.tolist(),expected_support.tolist())
        inertia=np.asarray(trajectory[prefix+'runtime_inertia_diagonal_kg_m2'])
        expected_inertia=expected_object_inertia(obj)
        check(prefix+'runtime_inertia',inertia.shape==(3,) and np.isfinite(inertia).all() and (inertia>0).all() and np.allclose(inertia,expected_inertia,rtol=1e-6,atol=1e-10),inertia.tolist(),expected_inertia.tolist())
        expected_proxy=declared_collision_descriptors(obj);proxy_ok=True
        for key,value in expected_proxy.items():
            arr=np.asarray(trajectory[prefix+'runtime_proxy_'+key]);target=np.asarray(value)
            if key=='quaternions_xyzw':ok=arr.shape==target.shape and np.all(np.minimum(np.linalg.norm(arr-target,axis=1),np.linalg.norm(arr+target,axis=1))<=tolerance)
            else:ok=arr.shape==target.shape and np.allclose(arr,target,rtol=0,atol=tolerance)
            proxy_ok=proxy_ok and ok
        check(prefix+'runtime_proxy',proxy_ok,None,'declared collider descriptors')
        distances=np.asarray(trajectory[prefix+'minimum_contact_distance_m']);penetration=max(0.0,-float(np.min(distances)))
        limit=metadata['qa']['limits']['maximum_trajectory_penetration_m']
        check(prefix+'bounded_penetration',np.isfinite(distances).all() and penetration<=limit,penetration,limit)
        omega_body=np.asarray([np.asarray(quaternion_matrix_wxyz(qi)).T@wi for qi,wi in zip(q,angular)])
        energy.append(0.5*material['mass_kg']*np.sum(vel*vel,axis=1)+0.5*np.sum(omega_body*omega_body*inertia,axis=1)-material['mass_kg']*(pos@gravity))
        metrics[oid]={'maximum_speed_m_s':float(np.linalg.norm(vel,axis=1).max()),'maximum_penetration_m':penetration}
    total=np.sum(energy,axis=0);gain=float(total.max()-total[0]);limits=metadata['qa']['limits']
    scale=max(float(total[0]-total.min()),sum(o['material']['mass_kg']*np.linalg.norm(gravity)*min(o['geometry']['size_m']) for o in objects),1e-9)
    allowance=limits.get('maximum_unforced_energy_gain_fraction',0.05)*scale+limits.get('maximum_unforced_energy_gain_j_per_kg',0.01)*sum(o['material']['mass_kg'] for o in objects)
    check('bounded_group_energy',gain<=allowance,gain,allowance)
    evidence=json.loads(str(np.asarray(trajectory['three_object_event_evidence_json']).item()))
    events=validate_event_evidence(evidence,metadata)
    check('event_evidence',True,len(events),'complete consecutive substep coverage')
    checks.extend(interaction_checks(contract, events, evidence['simulation_hz'], {
        o['object_id']:trajectory[o['object_id']+'__linear_velocity_m_s'][0] for o in objects}))
    is_sweep=metadata.get('sweep',{}).get('kind')=='sweep'
    for record in checks:
        if is_sweep and record['category']=='base_semantics':record['severity']='advisory'
    result={'schema_version':'physweep_three_object_trajectory_audit_v2','scene_id':metadata['scene_id'],
            'checks':checks,'metrics':metrics,'contact_events':evidence,'advisories':[r for r in checks if r.get('severity')=='advisory']}
    result['passed']=all(audit_hard_results(objects,result,is_sweep))
    return result
