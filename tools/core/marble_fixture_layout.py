"""Explicit geometric transformations of a verified passive marble fixture."""
import copy,math
from tools.core.hashing import sha256_json


def construct_marble_fixture(source_fixture,layout):
    if set(layout)!={'id','retained_track_ids','catch_translation_m'} or layout['id']!='three_segment_source_fixture_v1':
        raise ValueError('unsupported marble fixture construction')
    source=copy.deepcopy(source_fixture);meshes=source['mesh_components'];boxes=source['analytic_colliders']
    if [m['id'] for m in meshes]!=['track_1','track_2','track_3','track_4'] or layout['retained_track_ids']!=['track_1','track_2','track_3']:
        raise ValueError('three-segment marble layout requires the original four ordered source tracks')
    if [b['id'] for b in boxes]!=['catch_floor','catch_side_positive_y','catch_side_negative_y','catch_end','safety_floor']:
        raise ValueError('marble catch or safety-floor source components changed')
    translation=layout['catch_translation_m']
    expected=[round(float(a)-float(b),12) for a,b in zip(meshes[2]['base_position_m'],meshes[3]['base_position_m'])]
    if len(translation)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in translation) or translation!=expected:
        raise ValueError('marble catch translation must follow the retained terminal track displacement')
    source['mesh_components']=meshes[:3]
    for box in boxes:
        if box['id']!='safety_floor':box['position_m']=[float(a)+float(b) for a,b in zip(box['position_m'],translation)]
    return source


def validate_marble_fixture_construction(fixture,binding):
    if set(binding)!={'source_fixture','source_fixture_sha256','layout','result_fixture_sha256'}:
        raise ValueError('incomplete marble fixture construction evidence')
    original=binding['source_fixture']
    if sha256_json(original)!=binding['source_fixture_sha256']:
        raise ValueError('original marble fixture construction hash changed')
    expected=construct_marble_fixture(original,binding['layout'])
    if fixture!=expected or sha256_json(fixture)!=binding['result_fixture_sha256']:
        raise ValueError('marble fixture differs from declared source transformation')
