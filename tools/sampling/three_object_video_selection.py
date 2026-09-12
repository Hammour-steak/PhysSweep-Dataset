"""Choose a bounded video trial from base inputs without consuming sweep results."""
from collections import Counter

TEMPLATES=('chain_transfer','converging_hits','successive_hits','pair_control')


def coverage_features(metadata):
    cell=metadata['coverage']['cell'];template=cell['template'];appearance=metadata['appearance']
    values={('template_role',template,cell['target_role']),('template_host',template,cell['host_class']),
            ('template_camera',template,cell['requested_camera_family']),
            ('environment',appearance['scene_visual']['environment_category']),
            ('support_material',appearance['materials']['support_surface']['record']['asset_id']),
            ('scale_pattern',*(cell['scale_by_role'][r] for r in ('P','Q','R')))}
    values.update(('object_asset',o['visual_profile']['id']) for o in metadata['simulation']['objects'])
    return values


def build_video_plan(base_metadata, *, geometry_scope='sphere_flat_v1'):
    if geometry_scope=='sphere_cross_slope_pair_v1':return inclined_video_plan(base_metadata)
    if geometry_scope=='mixed_flat_R_v1':return mixed_video_plan(base_metadata)
    if geometry_scope!='sphere_flat_v1':raise ValueError('unknown video coverage scope')
    by_id={m['scene_id']:m for m in base_metadata}
    if len(by_id)!=len(base_metadata):raise ValueError('duplicate base scene in video candidate pool')
    if any(m.get('sweep',{}).get('kind')=='sweep' for m in base_metadata):raise ValueError('video selection accepts base inputs only')
    weights={'template_role':10,'template_host':6,'template_camera':5,'environment':4,'support_material':2,'scale_pattern':2,'object_asset':1}
    covered=set();primary=set();slots=[]
    for repeat in range(2):
        for template in TEMPLATES:
            candidates=[m for m in base_metadata if m['coverage']['cell']['template']==template and m['scene_id'] not in primary]
            ranked=sorted(candidates,key=lambda m:(-sum(weights[f[0]] for f in coverage_features(m)-covered),m['scene_id']))
            order=[m['scene_id'] for m in ranked[:4]]
            slots.append({'slot_id':f'{template}_{repeat}','template':template,'candidate_order':order})
            if order:primary.add(order[0]);covered.update(coverage_features(by_id[order[0]]))
    return {'schema_version':'physweep_three_object_video_selection_v1','requested_groups':8,'slots':slots,
            'maximum_camera_attempts_per_slot':4,'weights':weights,
            'policy':'two video groups per supported template; deterministic base-only coverage ranking; camera rejection may use the frozen next candidate in the same template',
            'scope':'bounded video experiment, not formal production quotas','uses_sweep_outcomes':False,
            'planned_primary_features':[list(f) for f in sorted(covered)]}


def final_video_coverage(metadata):
    counts=Counter(feature for m in metadata for feature in coverage_features(m))
    return [{'feature':list(feature),'groups':count} for feature,count in sorted(counts.items())]


def mixed_video_plan(base_metadata):
    if len({m['scene_id'] for m in base_metadata})!=len(base_metadata) or any(m.get('sweep',{}).get('kind')=='sweep' for m in base_metadata):raise ValueError('mixed video selection requires distinct base inputs')
    covered=set();slots=[];weights={'template_role':10,'template_host':6,'template_camera':5,'environment':4,'support_material':2,'scale_pattern':2,'object_asset':1}
    for template in ('chain_transfer','pair_control'):
        for shape in ('cuboid','cylinder'):
            eligible=[m for m in base_metadata if m['coverage']['cell']['template']==template and m['coverage']['cell']['shape_by_role']=={'P':'sphere','Q':'sphere','R':shape}]
            ranked=sorted(eligible,key=lambda m:(-sum(weights[f[0]] for f in coverage_features(m)-covered),m['scene_id']))[:4]
            slots.append({'slot_id':template+'_'+shape,'template':template,'R_shape':shape,'candidate_order':[m['scene_id'] for m in ranked]})
            if ranked:covered.update(coverage_features(ranked[0]))
    return {'schema_version':'physweep_three_object_video_selection_v1','requested_groups':4,'slots':slots,'maximum_camera_attempts_per_slot':4,
            'weights':weights,'policy':'one complete group per template and R shape; base-only frozen candidate order; no cross-shape reallocation',
            'uses_sweep_outcomes':False,'scope':'mixed flat R extension video trial; not production quota'}


def inclined_video_plan(base_metadata):
    if len({m['scene_id'] for m in base_metadata})!=len(base_metadata) or any(m.get('sweep',{}).get('kind')=='sweep' for m in base_metadata):raise ValueError('inclined video selection requires distinct base inputs')
    covered=set();primary=set();slots=[]
    weights={'template_role':10,'template_host':6,'template_camera':5,'environment':4,'support_material':2,'scale_pattern':2,'object_asset':1}
    for repeat in range(2):
        for host in ('ground_feature','raised_feature'):
            eligible=[m for m in base_metadata if m['coverage']['cell']['template']=='pair_control' and m['coverage']['cell']['host_class']==host and m['scene_id'] not in primary]
            ranked=sorted(eligible,key=lambda m:(-sum(weights[f[0]] for f in coverage_features(m)-covered),m['scene_id']))[:4]
            slots.append({'slot_id':f'pair_control_{host}_{repeat}','template':'pair_control','host_class':host,'candidate_order':[m['scene_id'] for m in ranked]})
            if ranked:covered.update(coverage_features(ranked[0]));primary.add(ranked[0]['scene_id'])
    return {'schema_version':'physweep_three_object_video_selection_v1','requested_groups':4,'slots':slots,'maximum_camera_attempts_per_slot':4,
      'weights':weights,'policy':'two complete pair-control groups per ramp host class; base-only frozen order; no cross-host reallocation',
      'uses_sweep_outcomes':False,'scope':'shallow inclined sphere pilot, not production quota'}
