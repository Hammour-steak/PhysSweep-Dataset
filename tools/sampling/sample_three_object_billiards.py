"""Construct declared three-sphere billiards initial scenes from verified 1obj sources."""
from __future__ import annotations
import copy
from pathlib import Path
from tools.assets.billiards_template import refresh_billiards_fixture
from tools.core.hashing import sha256_file, sha256_json
from tools.core.json_io import read_json
from tools.core.paths import safe_scene_id
from tools.dataset_contract.object_identity_contract import attach_object_identity
from tools.motion_rules.three_object.billiards import IDS, ROLES, SCHEMA, initial_states, validate_billiards_contract


def load_billiards_rules(code_root: Path) -> dict:
    path = code_root / 'configs/three_object_billiards_rules.json'
    config = read_json(path); bindings = {'billiards': {'path': str(path), 'sha256': sha256_file(path)}}
    documents = {}
    for key in ('motion_rules', 'scene_rules'):
        binding = config[key]; source = code_root / binding['path']
        if sha256_file(source) != binding['sha256']:
            raise ValueError('shared billiards rule reference changed')
        documents[key] = read_json(source); bindings[key] = {'path': str(source), 'sha256': binding['sha256']}
    if config['schema_version'] != 'physweep_three_object_billiards_rules_v1' or config['template_ids'] != ['chain_transfer']:
        raise ValueError('unimplemented billiards pilot rules')
    return {'config': config, **documents, 'bindings': bindings, 'rules_sha256': sha256_json(bindings)}


def build_three_object_billiards_scene(*, data_root: Path, host_source: dict, object_sources: list[dict],
        rules: dict, scene_id: str, role_order: list[str], material_slots: list[str], parameters: dict,
        seeds: dict, requested_view: str) -> dict:
    config = rules['config']
    if len(object_sources) != 3 or len({r['source']['scene_id'] for r in object_sources}) != 3:
        raise ValueError('three distinct verified billiards sources required')
    for row in [host_source, *object_sources]:
        if row['metadata']['schema_version'] != config['source_schema'] or row['metadata']['semantics']['dynamic_object_count'] != 1:
            raise ValueError('billiards candidate requires single-object generation sources')
        if row['metadata'].get('sweep', {}).get('kind') == 'sweep':
            raise ValueError('sweep cannot be a billiards source')
    if sorted(role_order) != sorted(ROLES) or sorted(material_slots) != sorted(config['visual_material_slots']):
        raise ValueError('complete explicit billiards role and material assignments required')
    scene = refresh_billiards_fixture(data_root, host_source['metadata'])
    prior_binding = copy.deepcopy(host_source['metadata']['physics']['static_support_binding'])
    for key in ('sweep', 'admission', 'object_identity', 'simulation_record', 'outputs', 'implementation'):
        scene.pop(key, None)
    fixture = scene['physics']['static_support_binding']; radius = float(object_sources[0]['metadata']['physics']['ball_radius_m'])
    states = initial_states(radius, fixture['target_support_frame']['safe_surface'], config['initial_limits'], parameters)
    visual = fixture['visual']; objects = []
    for oid, role, slot, source in zip(IDS, role_order, material_slots, object_sources):
        original = source['metadata']; physics = original['physics']
        if physics['ball_radius_m'] != radius:
            raise ValueError('initial billiards scope requires equal source radii')
        binding = physics['backend_config']; path = data_root / binding['path']
        if sha256_file(path) != binding['sha256']:
            raise ValueError('billiards source backend changed')
        extras = read_json(path)['billiards_rules']['ball_dynamics']
        material = copy.deepcopy(physics['runtime_material'])
        material.update({key: float(extras[key]) for key in ('rolling_friction', 'spinning_friction', 'linear_damping', 'angular_damping')})
        if material['mass_kg'] != physics['ball_mass_kg']:
            raise ValueError('source billiards mass fields disagree')
        objects.append({'object_id': oid, 'semantic_type': 'billiard_ball', 'body_model': 'rigid_body',
            'geometry': {'type':'sphere', 'size_m':[2 * radius] * 3},
            'collision_proxy': {'type':'sphere', 'radius_m':radius}, 'material':material,
            'initial_state': copy.deepcopy(states[role]),
            'visual_profile': {'asset_id':fixture['asset_id'], 'path':visual['path'], 'sha256':visual['sha256'], 'material_slot':slot}})
    scene.update(schema_version=SCHEMA, scene_id=safe_scene_id(scene_id), dataset_id='physweep_three_object_billiards_pilot_v1',
                 dataset_stage='three_object_base_candidate', seed=seeds['physics'])
    for key in ('initial_states','ball_mass_kg','ball_radius_m','runtime_material','trajectory_path','audit_path','simulation_record_path','two_object_quality'):
        scene['physics'].pop(key, None)
    scene['physics']['profile'] = 'chain_transfer'
    media = config['media']
    scene['physics'].update({key: media[key] for key in ('duration_s','output_fps','frame_count')})
    scene['simulation'] = {'objects': objects}
    scene['semantics'] = {'scene_family':'billiards', 'dynamic_object_count':3, 'motion_profile':'chain_transfer',
                          'description':'Three separated spheres start in a line on the billiards bed; only the end sphere moves initially.'}
    scene['semantic_rules'] = copy.deepcopy(rules['bindings']['billiards'])
    scene['render'].update(engine=media['render_engine'], resolution=copy.deepcopy(media['resolution']), samples=media['samples'])
    for key in ('video_path','inspection_frame_dir'):
        scene['render'].pop(key,None)
    scene.pop('camera',None)
    if requested_view not in {r['id'] for r in rules['scene_rules']['camera']['view_families']}:
        raise ValueError('unknown requested billiards view')
    scene['camera_request'] = {'policy':'three_object_base_events', 'requested_view_family':requested_view}
    template = copy.deepcopy(next(t for t in rules['motion_rules']['templates'] if t['id']=='chain_transfer'))
    contract = {'schema_version':'physweep_three_object_billiards_motion_contract_v1', 'template':template,
        'roles':dict(zip(role_order, IDS)), 'thresholds':copy.deepcopy(rules['motion_rules']['thresholds']),
        'rules_sha256':rules['rules_sha256'], 'initial_limits':copy.deepcopy(config['initial_limits']),
        'initial_parameters':copy.deepcopy(parameters), 'camera_rules':copy.deepcopy(rules['scene_rules']['camera'])}
    scene['three_object'] = {**contract, 'contract_sha256':sha256_json(contract)}
    scene['source_binding'] = {'objects':[copy.deepcopy(r['source']) for r in object_sources], 'host':copy.deepcopy(host_source['source']),
        'rules':copy.deepcopy(rules['bindings']), 'seeds':copy.deepcopy(seeds),
        'fixture_provenance_refresh': {'before':prior_binding, 'after':copy.deepcopy(fixture)},
        'appearance_assignment': {'policy':'permuted_three_verified_fixture_material_slots_v1', 'object_slots':dict(zip(IDS,material_slots))}}
    validate_billiards_contract(scene)
    attach_object_identity(scene)
    return scene
