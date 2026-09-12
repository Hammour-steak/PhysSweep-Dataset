"""Build first-pilot candidates using the shared reviewed object compiler."""
from __future__ import annotations
import copy
from pathlib import Path

from tools.core.paths import safe_scene_id
from tools.dataset_contract.object_identity_contract import attach_object_identity
from tools.motion_rules.three_object.motion import apply_motion,validate_motion_contract
from tools.sampling.object_collection import compile_object_collection_scene
from tools.sampling.three_object_sampling_request import candidate_seed, OBJECT_IDS
from tools.scene_rules.three_object import validate_host,validate_initial_layout


def build_three_object_scene(*, host_source: dict, object_sources: list[dict], rules: dict,
                            scene_id: str, candidate_index: int=0, role_order: list[str]|None=None,
                            speed_m_s: float=1.2, spacing_ratio: float=2.0,
                            template_id: str='chain_transfer',offset_ratio: float=0.6,
                            requested_view_family: str='front_oblique',sampling_cell_id: str|None=None,
                            motion_axis: str='world_x',layout_variant: str='axis_aligned',
                            root: Path|None=None) -> dict:
    scene_id=safe_scene_id(scene_id)
    if len(object_sources)!=3:raise ValueError('three reviewed object sources required')
    source_ids=[r['source']['scene_id'] for r in object_sources]
    if len(set(source_ids))!=3:raise ValueError('three distinct source scenes required')
    host=host_source['metadata'];validate_host(
        host,rules['scene'],
        allow_exact_mesh=rules['matrix'].get('coverage_mode')=='exact_mesh_host_v1')
    scene=compile_object_collection_scene(host,[r['metadata'] for r in object_sources],[{'object_id':o} for o in OBJECT_IDS])
    cell_id=sampling_cell_id or template_id+':'+','.join(role_order or ['P','Q','R'])
    seeds={purpose:candidate_seed(rules['matrix']['master_seed'],cell_id,candidate_index,purpose) for purpose in rules['matrix']['seed_domains']}
    scene.update(scene_id=scene_id,dataset_id='physweep_three_object_pilot_v1',dataset_stage='three_object_base_candidate',
                 seed=seeds['physics'],sample_index=candidate_index)
    scene['qa']={'status':'sampled_pending_simulation','limits':copy.deepcopy(host['qa']['limits'])}
    media=rules['media']
    scene['render_request']={'resolution':copy.deepcopy(media['resolution']),'samples':media['samples'],
                             'engine':media['render_engine'],'fps':media['output_fps']}
    scene['simulation']['time'].update(duration_s=media['duration_s'],output_fps=media['output_fps'],frame_count=media['frame_count'])
    if requested_view_family not in {f['id'] for f in rules['scene']['camera']['view_families']}:raise ValueError('unknown requested view family')
    scene['camera_request']={'policy':'three_object_base_events','requested_view_family':requested_view_family}
    scene['source_binding']={'objects':[copy.deepcopy(r['source']) for r in object_sources],
                             'host':copy.deepcopy(host_source['source']),'rules':copy.deepcopy(rules['bindings']),'seeds':seeds}
    if 'environment_source' in host_source:
        scene['source_binding']['host_environment']=copy.deepcopy(host_source['environment_source'])
    apply_motion(scene,rules,role_order=role_order or ['P','Q','R'],speed_m_s=speed_m_s,spacing_ratio=spacing_ratio,
                 template_id=template_id,offset_ratio=offset_ratio,motion_axis=motion_axis,
                 layout_variant=layout_variant)
    placement=validate_initial_layout(scene,rules['scene'],rules['motion']['thresholds']['initial_separation_m'],root=root)
    if placement is not None:scene['simulation']['support']['initial_placement_validation']=placement
    validate_motion_contract(scene)
    attach_object_identity(scene)
    return scene
