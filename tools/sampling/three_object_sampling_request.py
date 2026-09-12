"""Resolve and hash the three-object pilot rules without implicit overrides."""
from __future__ import annotations
import copy
import itertools
import math
from pathlib import Path

from tools.core.hashing import sha256_file, sha256_json
from tools.core.json_io import read_json
from tools.core.sweep_values import counterfactual_mass_values
from tools.sampling.released_object_sources import declared_within

ROOT = Path(__file__).resolve().parents[2]
OBJECT_IDS = ('object_a','object_b','object_c')
ROLES = ('P','Q','R')


def candidate_seed(master_seed: int, cell_id: str, candidate_index: int, purpose: str) -> int:
    if type(master_seed) is not int or type(candidate_index) is not int or candidate_index < 0:
        raise ValueError('seed and candidate index must be integers; index must be nonnegative')
    if not cell_id or purpose not in {'physics','appearance','camera'}:
        raise ValueError('invalid seed namespace')
    return int(sha256_json([master_seed,cell_id,candidate_index,purpose])[:16],16)


def validate_rules(matrix: dict, motion: dict, scene: dict, sweep: dict, dataset: dict) -> None:
    coverage_mode=matrix.get('coverage_mode')
    if coverage_mode not in {None,'all_medium_visual_assets_v1','primitive_asset_proxy_v1','compound_asset_proxy_v1','exact_mesh_host_v1','motion_diversity_v1','multi_mesh_pair_roles_v1'}:
        raise ValueError('unsupported visual asset coverage mode or geometry scope')
    if coverage_mode in {'all_medium_visual_assets_v1','primitive_asset_proxy_v1','compound_asset_proxy_v1'} and matrix.get('geometry_scope')!='mixed_flat_R_v1':
        raise ValueError('object visual coverage requires mixed flat geometry scope')
    if coverage_mode=='exact_mesh_host_v1' and matrix.get('geometry_scope')!='sphere_flat_v1':
        raise ValueError('exact mesh hosts require the sphere flat geometry scope')
    if coverage_mode=='motion_diversity_v1' and (matrix.get('geometry_scope')!='motion_diversity_v1' or matrix.get('candidate_budget')!=16):
        raise ValueError('motion diversity trial requires its explicit scope and sixteen candidates')
    if coverage_mode=='multi_mesh_pair_roles_v1':
        required_assets=matrix.get('required_asset_ids')
        required_generic=matrix.get('required_generic_visual_asset_ids')
        if (matrix.get('geometry_scope')!=coverage_mode or matrix.get('candidate_budget')!=42
                or not isinstance(required_assets,list) or len(required_assets)!=10 or len(set(required_assets))!=10
                or not isinstance(required_generic,list) or len(required_generic)!=84 or len(set(required_generic))!=84
                or any(not isinstance(value,str) or not value for value in [*required_assets,*required_generic])):
            raise ValueError('multi-mesh role trial requires ten nonball assets, all 84 generic visuals and 42 candidates')
    source=matrix.get('source',{})
    if coverage_mode in {'motion_diversity_v1','multi_mesh_pair_roles_v1'}:
        if (source.get('families')!=['generic','asset']
                or source.get('generation_metadata_schemas')!={'generic':'physweep_pybullet_rigid_metadata_v1','asset':'physweep_asset_proxy_scene_v3'}
                or source.get('asset_eligibility',{}).get('asset_proxy_policy')!='centered_primitive_or_upright_axisymmetric_compound'):
            raise ValueError('mixed visual trial requires released generic and eligible asset sources')
    if coverage_mode in {'primitive_asset_proxy_v1','compound_asset_proxy_v1'}:
        schemas=source.get('generation_metadata_schemas')
        eligibility=source.get('asset_eligibility')
        required=matrix.get('required_asset_ids')
        if source.get('families')!=['generic','asset'] or schemas!={'generic':'physweep_pybullet_rigid_metadata_v1','asset':'physweep_asset_proxy_scene_v3'}:
            raise ValueError('primitive asset coverage requires explicit generic and asset source schemas')
        if (not isinstance(eligibility,dict) or eligibility.get('body_model')!='rigid_body'
                or eligibility.get('required_pose_profile')!='support_normal'
                or eligibility.get('asset_proxy_policy')!='centered_primitive_or_upright_axisymmetric_compound'
                or eligibility.get('asset_proxy_maximum_aabb_center_offset_m')!=.001
                or eligibility.get('asset_scale_bin_maximum_extent_m')!={'small':.18,'medium':.23,'large':None}):
            raise ValueError('primitive asset normalization contract differs from released 2obj behavior')
        required_count=8 if coverage_mode=='primitive_asset_proxy_v1' else 3
        if (not isinstance(required,list) or len(required)!=required_count or len(set(required))!=required_count
                or any(not isinstance(asset,str) or not asset for asset in required)
                or matrix.get('candidate_budget')!=2*len(required)):
            if coverage_mode=='primitive_asset_proxy_v1':
                raise ValueError('primitive asset coverage requires eight explicit assets and two cells each')
            raise ValueError('compound asset coverage requires three explicit assets and two cells each')
    elif coverage_mode=='exact_mesh_host_v1':
        schemas=source.get('generation_metadata_schemas');eligibility=source.get('asset_eligibility')
        required=matrix.get('required_support_asset_ids');assignments=matrix.get('mesh_host_assignments')
        excluded=matrix.get('excluded_support_asset_ids',[])
        timing=matrix.get('exact_mesh_timing_policy')
        overrides=matrix.get('effective_safe_surface_overrides')
        if source.get('families')!=['generic','asset'] or schemas!={'generic':'physweep_pybullet_rigid_metadata_v1','asset':'physweep_asset_proxy_scene_v3'}:
            raise ValueError('exact mesh hosts require explicit generic and asset source schemas')
        if (not isinstance(eligibility,dict) or eligibility.get('body_model')!='rigid_body'
                or eligibility.get('required_pose_profile')!='support_normal'
                or eligibility.get('asset_proxy_policy')!='centered_primitive_or_upright_axisymmetric_compound'):
            raise ValueError('exact mesh host asset normalization contract is incomplete')
        if (not isinstance(required,list) or len(required)!=len(set(required))
                or not isinstance(excluded,list) or len(excluded)!=len(set(excluded))
                or set(required)&set(excluded) or len(required)+len(excluded)!=19
                or matrix.get('candidate_budget')!=len(required) or not isinstance(assignments,list)
                or len(assignments)!=len(required) or {row.get('asset_id') for row in assignments}!=set(required)):
            raise ValueError('exact mesh host coverage requires one explicit cell or exclusion for each of 19 released hosts')
        if timing is None:
            if len(required)!=19 or excluded:
                raise ValueError('legacy exact mesh host coverage cannot exclude hosts without a timing policy')
        elif (set(timing)!={'schema_version','nominal_first_contact_time_s','extra_capacity_reserve_fraction','maximum_initial_speed_m_s_by_template','chain_transfer_minimum_driver_to_receiver_mass_ratio'}
                or timing.get('schema_version')!='physweep_exact_mesh_timing_policy_v1'
                or timing.get('nominal_first_contact_time_s')!=.21
                or timing.get('extra_capacity_reserve_fraction')!=.1
                or timing.get('maximum_initial_speed_m_s_by_template')!={'chain_transfer':1.1,'pair_control':.4}
                or timing.get('chain_transfer_minimum_driver_to_receiver_mass_ratio')!=.8
                or excluded!=['support_rattan_coffee_c434f22e']):
            raise ValueError('exact mesh timing repair policy or explicit exclusion differs from the bounded scope')
        for row in assignments:
            if (set(row)!={'asset_id','template','motion_axis','scale'}
                    or row['template'] not in {'chain_transfer','pair_control'}
                    or row['motion_axis'] not in {'world_x','world_y'}
                    or row['scale'] not in {'small','medium'}):
                raise ValueError('exact mesh host assignment is invalid')
        if (not isinstance(overrides,dict) or not set(overrides)<=set(required)
                or any(set(value)!={'center_xy_m','size_xy_m','z_m'} for value in overrides.values())):
            raise ValueError('exact mesh host effective surface overrides are invalid')
    elif ('required_asset_ids' in matrix and coverage_mode!='multi_mesh_pair_roles_v1') or 'required_support_asset_ids' in matrix:
        raise ValueError('asset target list requires primitive asset coverage mode')
    if matrix.get('schema_version') != 'physweep_three_object_sampling_matrix_v1':
        raise ValueError('unsupported three-object matrix')
    if motion.get('schema_version') != 'physweep_three_object_motion_rules_v1' or scene.get('schema_version') != 'physweep_three_object_scene_rules_v2':
        raise ValueError('unsupported three-object rules')
    if matrix.get('object_ids') != list(OBJECT_IDS) or matrix.get('sweep_target_indices') != [0]:
        raise ValueError('three-object identity and target must be explicit')
    if type(dataset.get('object_count')) is not int or dataset['object_count'] != 3 or dataset.get('release_root') != 'outputs/three_object':
        raise ValueError('three-object output configuration mismatch')
    if type(sweep.get('required_dynamic_objects')) is not int or sweep['required_dynamic_objects'] != 3:
        raise ValueError('three-object sweep count mismatch')
    counterfactual_mass_values(1.0, sweep['mass_intervention_multipliers'])
    if sweep['mass_intervention_multipliers'] != [0.25,0.5,1.0,2.0,4.0]:
        raise ValueError('pilot mass intervention differs from approved design')
    template_ids = [r['id'] for r in motion['templates']]
    if len(template_ids) != len(set(template_ids)) or template_ids != matrix['template_ids']:
        raise ValueError('duplicate or inconsistent template references')
    if set(template_ids) != {'chain_transfer','successive_hits','converging_hits','pair_control'}:
        raise ValueError('pilot requires four explicit templates')
    all_pairs={frozenset(p) for p in itertools.combinations(ROLES,2)}
    for template in motion['templates']:
        if template['roles'] != list(ROLES) or set(template['shape_by_role']) != set(ROLES):
            raise ValueError('invalid template roles')
        scope=matrix.get('geometry_scope','sphere_flat_v1')
        if scope not in {'sphere_flat_v1','mixed_flat_R_v1','sphere_cross_slope_pair_v1','motion_diversity_v1','multi_mesh_pair_roles_v1'}:raise ValueError('unknown geometry scope')
        for role,shapes in template['shape_by_role'].items():
            mixed_R=scope=='mixed_flat_R_v1' and role=='R' and template['id'] in {'chain_transfer','pair_control'}
            moving_R=scope=='motion_diversity_v1' and role=='R' and template['id']=='pair_control'
            multi_pair=scope=='multi_mesh_pair_roles_v1' and template['id']=='pair_control'
            allowed={'sphere','cuboid','cylinder'} if mixed_R or moving_R or multi_pair else {'sphere'}
            if not shapes or len(set(shapes))!=len(shapes) or not set(shapes)<=allowed:
                raise ValueError('shape role outside implemented pilot compatibility')
        expected_dynamic=['R'] if scope=='motion_diversity_v1' and template['id']=='pair_control' else None
        if template.get('dynamic_primitive_roles')!=expected_dynamic:
            raise ValueError('dynamic primitive role admission differs from the explicit scope')
        expected_mixed=list(ROLES) if scope=='multi_mesh_pair_roles_v1' and template['id']=='pair_control' else None
        if template.get('mixed_primitive_roles')!=expected_mixed:
            raise ValueError('mixed primitive roles differ from the explicit scope')
        expected_variants={
            'chain_transfer':['axis_aligned','angled_chain_left','angled_chain_right'],
            'successive_hits':['axis_aligned','offset_successive_left','offset_successive_right'],
            'converging_hits':['axis_aligned'],
            'pair_control':['axis_aligned','crossing_control_left','crossing_control_right'],
        }[template['id']] if scope=='motion_diversity_v1' else ['axis_aligned']
        if template.get('layout_variants',['axis_aligned'])!=expected_variants:
            raise ValueError('layout variants differ from the explicit motion scope')
        if not set(template['initially_moving_roles']) <= set(ROLES) or not set(template['sweep_target_roles']) <= set(ROLES):
            raise ValueError('unknown motion role')
        pairs={key:[frozenset(p) for p in template[key]] for key in ('required_pairs','forbidden_pairs','allowed_pairs','first_contact_order')}
        if any(any(p not in all_pairs for p in values) or len(values)!=len(set(values)) for values in pairs.values()):
            raise ValueError('invalid or duplicate event pair')
        required,allowed,forbidden=(set(pairs[k]) for k in ('required_pairs','allowed_pairs','forbidden_pairs'))
        if not required <= allowed or allowed & forbidden or allowed | forbidden != all_pairs:
            raise ValueError('unclassified or conflicting contact pairs')
        if not set(pairs['first_contact_order']) <= required:
            raise ValueError('ordered events must be required')
        relation=template['interaction']
        if relation=='connected_three' and len(required)<2 or relation=='pair_plus_independent' and len(allowed)!=1:
            raise ValueError('contact graph contradicts template')
        if relation not in {'connected_three','pair_plus_independent'}:
            raise ValueError('unsupported pilot interaction')
        if template['caption_template'] != ('three_object_initial_state_v2' if scope in {'mixed_flat_R_v1','motion_diversity_v1','multi_mesh_pair_roles_v1'} else 'three_object_initial_state_v1'):
            raise ValueError('unknown caption template')
    thresholds=motion['thresholds']
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in thresholds.values()):
        raise ValueError('thresholds must be positive and finite')
    if not thresholds['stationary_linear_speed_m_s'] < thresholds['moving_linear_speed_m_s']:
        raise ValueError('stationary and moving thresholds overlap')
    if not thresholds['first_event_min_s'] < thresholds['last_event_max_s'] < 4.0:
        raise ValueError('events lack an observation window')
    inclined=matrix.get('geometry_scope')=='sphere_cross_slope_pair_v1'
    lead=motion['initial_state'].get('independent_lane_lead_in_pair_radii')
    if lead is not None and (not inclined or type(lead) not in (int,float) or lead!=3.):raise ValueError('unsupported independent-lane lead policy')
    for template in motion['templates']:
        if template.get('independent_lane_lead_in_pair_radii')!=(lead if template['id']=='pair_control' else None):raise ValueError('template and constructor lead policy disagree')
    if inclined:
        if scene.get('support_scope')!='inclined_cross_slope_pair_v1' or scene.get('minimum_abs_slope_degrees')!=8. or scene['maximum_abs_slope_degrees']!=12. or scene['support_shapes']!=['inclined_ramp']:
            raise ValueError('inclined rules differ from implemented shallow ramp scope')
        for template in motion['templates']:
            if template.get('support_scope')!=('inclined_cross_slope_pair_v1' if template['id']=='pair_control' else 'flat_only'):
                raise ValueError('inclined template compatibility must be explicit')
        material_policy=motion['initial_state'].get('inclined_material_policy')
        if (material_policy is not None and (not isinstance(material_policy,dict)
                or set(material_policy)!={'schema_version','contact_friction_minimum','rolling_friction','application_stage'}
                or material_policy.get('schema_version')!='physweep_three_object_inclined_material_policy_v1'
                or material_policy.get('contact_friction_minimum')!=.32
                or material_policy.get('rolling_friction')!=.018
                or material_policy.get('application_stage')!='initial_state_construction_before_physics_and_sweeps')):
            raise ValueError('inclined material policy differs from the camera-compatible physical scope')
    elif 'inclined_material_policy' in motion['initial_state']:
        raise ValueError('flat rules contain an inclined material policy')
    primitive_cross_speed=motion['initial_state'].get('crossing_primitive_speed_m_s')
    if (matrix.get('geometry_scope')=='motion_diversity_v1')!=(primitive_cross_speed==1.1):
        raise ValueError('primitive crossing speed must be explicit only in the motion diversity scope')
    if scene['family']!='generic' or (not inclined and scene['maximum_abs_slope_degrees']!=0) or scene['masks'] is not False:
        raise ValueError('pilot scene scope mismatch')
    camera=scene['camera'];views=camera['view_families']
    if not views or len({v['id'] for v in views})!=len(views):
        raise ValueError('camera families must be distinct and nonempty')
    for view in views:
        if not 0 < view['elevation_degrees'] < 90 or not math.isfinite(view['azimuth_degrees']):
            raise ValueError('invalid camera family angles')
    extent_limits=[camera.get(key) for key in (
        'minimum_object_extent_fraction_of_short_side',
        'target_object_extent_fraction_of_short_side',
        'maximum_object_extent_fraction_of_short_side')]
    if (any(type(value) not in (int,float) or not math.isfinite(value) for value in extent_limits)
            or not 0 < extent_limits[0] < extent_limits[1] < extent_limits[2] < .5
            or 'minimum_object_extent_px' in camera):
        raise ValueError('camera object screen-fraction range is invalid')
    safe_violation=camera.get('maximum_safe_frame_violation_fraction')
    image_violation=camera.get('maximum_out_of_frame_fraction')
    overflow=camera.get('maximum_frame_overflow_fraction')
    margin=camera.get('frame_margin_fraction')
    event_margin=camera.get('event_frame_margin_fraction')
    if (any(type(value) not in (int,float) or not math.isfinite(value)
            for value in (safe_violation,image_violation,overflow,margin,event_margin))
            or not 0 <= image_violation < safe_violation < 1
            or not 0 < overflow < margin < .5
            or not 0 <= event_margin <= margin):
        raise ValueError('camera temporal framing limits are invalid')
    distance_factors=camera.get('candidate_distance_factors')
    if (not isinstance(distance_factors,list) or len(distance_factors)<3
            or any(type(value) not in (int,float) or not math.isfinite(value) or value<=0 for value in distance_factors)
            or distance_factors!=sorted(set(distance_factors))
            or not distance_factors[0]<1<distance_factors[-1]
            or type(camera.get('event_frame_padding')) is not int or camera['event_frame_padding']<0
            or camera.get('maximum_candidate_count')!=len(views)*len(camera['focal_length_mm'])*len(distance_factors)):
        raise ValueError('camera candidate or event-frame policy is invalid')
    if camera.get('audit_policy')!='bounded_object_screen_fraction_with_limited_non_event_overflow_and_geometric_occlusion':
        raise ValueError('camera audit policy is invalid')
    if camera['policy']!='base_only_fixed_group' or scene['sweep_visibility_policy']!='diagnostic_only':
        raise ValueError('camera policy contradicts counterfactual contract')
    for value in [matrix['candidate_budget'],matrix['maximum_source_attempts_per_candidate'],*matrix['source_budgets'].values()]:
        if type(value) is not int or value<=0: raise ValueError('budgets must be positive integers')


def load_pilot_rules(root: Path = ROOT, *, matrix_path: Path | None = None) -> dict:
    root=root.resolve()
    matrix_path=declared_within(root/'configs',matrix_path) if matrix_path is not None else root/'configs/three_object_sampling_matrix.json'
    matrix=read_json(matrix_path)
    paths={'matrix':matrix_path}
    for name,key in [('motion','motion_rules'),('scene','scene_rules'),('sweep','sweep_config'),('dataset','dataset_config')]:
        paths[name]=declared_within(root/'configs',Path(matrix[key]))
    values={key:read_json(path) for key,path in paths.items()}
    validate_rules(**values)
    binding=values['sweep']['base_config']
    base=declared_within(root/'configs',Path(binding['path']))
    if sha256_file(base)!=binding['sha256']: raise ValueError('shared sweep config hash mismatch')
    paths['shared_sweep']=base
    for name in matrix['shared_resources']:
        paths[name]=declared_within(root,Path(name))
    bindings={key:{'path':p.relative_to(root).as_posix(),'sha256':sha256_file(p)} for key,p in paths.items()}
    media=read_json(root/'configs/production_video.json')
    if (media['duration_s'],media['output_fps'],media['frame_count'],media['resolution'])!=(4.0,24,97,[1280,720]):
        raise ValueError('shared media contract differs from three-object pilot design')
    return {**copy.deepcopy(values),'media':media,'bindings':bindings,'rules_sha256':sha256_json(bindings)}
