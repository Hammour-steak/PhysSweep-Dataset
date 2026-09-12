"""Sample production candidates from the admitted source catalog and motion rules."""

from __future__ import annotations

import copy
import random
from pathlib import Path

from tools.assets.three_object_resources import resource_snapshot
from tools.core.hashing import sha256_json
from tools.sampling.derive_physics_sweep import load_sweep_config
from tools.sampling.released_object_sources import verified_generation_records
from tools.sampling.sample_three_object_billiards import load_billiards_rules, build_three_object_billiards_scene
from tools.sampling.sample_three_object_pinball import load_pinball_rules, build_three_object_pinball_scene
from tools.sampling.sample_three_object_marble import load_marble_rules, build_three_object_marble_scene, localize_marble_source_rows
from tools.sampling.three_object_coverage import build_plan, candidates
from tools.sampling.three_object_sampling_request import candidate_seed, load_pilot_rules
from tools.sampling.three_object_sources import load_sources

CODE_ROOT = Path(__file__).resolve().parents[2]
GENERIC_RULES = {
    'sphere': 'three_object_sampling_matrix.json',
    'mixed': 'three_object_d5a_sampling_matrix.json',
    'motion': 'three_object_d6a_motion_matrix.json',
    'multi_mesh': 'three_object_d6e_multi_mesh_pair_sampling_matrix.json',
    'inclined': 'three_object_d5br2_sampling_matrix.json',
    'exact_support': 'three_object_d5k_sampling_matrix.json',
}
SPECIAL_SCHEMAS = {
    'billiards': 'physweep_billiards_scene_v4',
    'passive_pinball': 'physweep_passive_pinball_scene_v1',
    'marble_run': 'physweep_marble_run_scene_v1',
}


def allocate_counts(count: int, weights: dict[str, float], *, minimum: int = 0) -> dict[str, int]:
    if count < 1 or not weights or any(value <= 0 for value in weights.values()):
        raise ValueError('count and family weights must be positive')
    if minimum * len(weights) > count:
        raise ValueError('count is too small to cover the requested families')
    remaining = count - minimum * len(weights)
    total = sum(weights.values())
    exact = {key: remaining * value / total for key, value in weights.items()}
    result = {key: minimum + int(value) for key, value in exact.items()}
    for key in sorted(weights, key=lambda key: (-(exact[key] % 1), key))[:count-sum(result.values())]:
        result[key] += 1
    return result


class CandidateFactory:
    def __init__(self, root: Path, source_root: Path, source_manifest: Path,
                 released_manifest: Path, work_id: str, seed: int):
        self.root = root
        self.source_root = source_root
        self.source_manifest = source_manifest
        self.released_manifest = released_manifest
        self.work_id = work_id
        self.seed = seed
        self.generic_cache = {}
        self.special_cache = {}

    def generic_context(self, mode: str):
        if mode not in self.generic_cache:
            rules = load_pilot_rules(CODE_ROOT, matrix_path=Path(GENERIC_RULES[mode]))
            rules = copy.deepcopy(rules)
            rules['matrix']['master_seed'] = self.seed
            pool = load_sources(root=self.root, source_root=self.source_root,
                                source_manifest=self.source_manifest, released_manifest=self.released_manifest,
                                rules=rules)
            plan = build_plan(pool, rules)
            available = [cell for cell in plan['cells'] if cell['capacity_condition']]
            if not available:
                raise ValueError(f'no eligible source capacity for {mode}')
            self.generic_cache[mode] = (rules, pool, plan, available)
        return self.generic_cache[mode]

    def generic(self, mode: str, ordinal: int, attempt: int) -> dict:
        rules, pool, prototype, cells = self.generic_context(mode)
        cell = copy.deepcopy(cells[ordinal % len(cells)])
        cell['cell_id'] = f'{mode}_{ordinal:06d}_{attempt:03d}'
        plan = {**prototype, 'candidate_count': 1, 'cells': [cell]}
        sweep = load_sweep_config(CODE_ROOT / 'configs/three_object_physics_sweep.json')
        outcome = next(candidates(pool, rules, plan, self.root, self.work_id, sweep))
        if outcome['scene'] is None:
            raise ValueError(f'candidate construction failed: {outcome["attempts"][-3:]}')
        scene = outcome['scene']
        scene['coverage'].update(family='generic', generation_mode=mode)
        return scene

    def special_context(self, family: str):
        if family not in self.special_cache:
            rules = (load_billiards_rules(CODE_ROOT) if family == 'billiards' else
                     load_pinball_rules(CODE_ROOT, self.root) if family == 'passive_pinball' else
                     load_marble_rules(CODE_ROOT, self.root))
            contract = {'released_base_manifest_schema_version': 'physweep_base_release_view_v15',
                        'generation_manifest_schema_version': 'physweep_release_metadata_manifest_v2', 'sample_kind': 'base'}
            rows = verified_generation_records(root=self.root, released_base_manifest_path=self.released_manifest,
                                               source_root=self.source_root, source_manifest_path=self.source_manifest,
                                               source_contract=contract, family_schemas={SPECIAL_SCHEMAS[family]: family})
            if family == 'marble_run':
                rows = localize_marble_source_rows(rows, self.source_root, self.root,
                                                  self.root/'datasets'/self.work_id/'inputs/collision')
            if len(rows) < 3:
                raise ValueError(f'{family} needs three distinct source objects')
            self.special_cache[family] = (rules, rows)
        return self.special_cache[family]

    def special(self, family: str, ordinal: int, attempt: int) -> dict:
        rules, rows = self.special_context(family)
        config = rules['config']
        cell = f'{family}_{ordinal:06d}_{attempt:03d}'
        seeds = {purpose: candidate_seed(self.seed, cell, ordinal, purpose)
                 for purpose in ('physics', 'appearance', 'camera')}
        rng = random.Random(seeds['physics'])
        objects = rng.sample(rows, 3)
        host = rows[rng.randrange(len(rows))]
        roles = ['P', 'Q', 'R']
        rng.shuffle(roles)
        args = dict(data_root=self.root, host_source=host, object_sources=objects, rules=rules,
                    scene_id=self.work_id+'_'+cell, role_order=roles, seeds=seeds)
        if family == 'billiards':
            limits = config['initial_limits']
            parameters = {key: rng.uniform(*limits[key]) for key in ('speed_m_s', 'PQ_surface_gap_m', 'QR_surface_gap_m', 'lane_y_m')}
            parameters['heading'] = rng.choice(limits['headings'])
            slots = list(config['visual_material_slots']); rng.shuffle(slots)
            views = rules['scene_rules']['camera']['view_families']
            scene = build_three_object_billiards_scene(**args, material_slots=slots, parameters=parameters,
                                                       requested_view=views[ordinal % len(views)]['id'])
        else:
            palette = list(config['appearance_palette']); rng.shuffle(palette)
            views = config['camera_rules']['view_families']
            backgrounds = config['background_contract']['profile_ids']
            cases = config['initial_limits']['parameter_cases']
            builder = build_three_object_pinball_scene if family == 'passive_pinball' else build_three_object_marble_scene
            scene = builder(**args, palette_order=palette, parameters=cases[(ordinal+attempt) % len(cases)],
                            requested_view=views[ordinal % len(views)]['id'],
                            background_profile=backgrounds[ordinal % len(backgrounds)])
        scene['visual_resource_binding'] = resource_snapshot(self.root, scene)
        scene['coverage'] = {'family': family, 'cell': {'cell_id': cell}, 'seed_binding': sha256_json(seeds)}
        return scene

    def build(self, mode: str, ordinal: int, attempt: int) -> dict:
        return self.generic(mode, ordinal, attempt) if mode in GENERIC_RULES else self.special(mode, ordinal, attempt)
