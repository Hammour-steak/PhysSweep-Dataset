"""Explicit family quotas and deterministic physical variation for 2obj bases."""
from __future__ import annotations

import copy
import math
from itertools import product
from typing import Any

from tools.core.hashing import sha256_json
from tools.motion_rules.two_object.specialized import SCENE_FAMILIES, family_index

SCHEMA = "physweep_two_object_sampling_request_v1"


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def profile_quotas(family: dict, count: int) -> dict[str, int]:
    ids = sorted(str(profile['id']) for profile in family['profiles'])
    if _positive_integer(count, 'family count') < len(ids):
        raise ValueError('family quota must cover every declared profile')
    quotient, remainder = divmod(count, len(ids))
    return {profile_id: quotient + (index < remainder) for index, profile_id in enumerate(ids)}


def validate_sampling_request(document: dict, rules: dict) -> None:
    if set(document) != {'schema_version', 'family_base_counts', 'generic_coverage', 'specialized_variation'} or document.get('schema_version') != SCHEMA:
        raise ValueError('unsupported two-object sampling request')
    counts = document['family_base_counts']
    if set(counts) != {'generic', *SCENE_FAMILIES}:
        raise ValueError('sampling request must declare all four family quotas')
    for family, count in counts.items():
        _positive_integer(count, family)
    generic = document['generic_coverage']
    if set(generic) != {'replicates_per_cell', 'maximum_object_source_reuse', 'maximum_host_source_reuse'}:
        raise ValueError('generic coverage settings are incomplete')
    for name, value in generic.items():
        _positive_integer(value, name)
    variations = document['specialized_variation']
    if set(variations) != set(SCENE_FAMILIES):
        raise ValueError('specialized variation must declare all families')
    families = family_index(rules)
    for name, variation in variations.items():
        if set(variation) != {'separation_scales', 'speed_scales'}:
            raise ValueError(f'{name}: unsupported physical variation')
        for axis, values in variation.items():
            if not isinstance(values, list) or not values or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0 for value in values
            ) or len(set(values)) != len(values):
                raise ValueError(f'{name}: {axis} must contain unique finite positive values')
        capacity = len(variation['separation_scales']) * len(variation['speed_scales'])
        if max(profile_quotas(families[name], counts[name]).values()) > capacity:
            raise ValueError(f'{name}: quota exceeds distinct physical initial-state capacity')


def effective_matrix(matrix: dict, request: dict | None) -> dict:
    result = copy.deepcopy(matrix)
    if request is not None:
        coverage = result['coverage_plan']
        if coverage.get('replicate_order') != 'complete_cells_before_repeats':
            raise ValueError('production quotas require complete-cells-before-repeats coverage')
        settings = request['generic_coverage']
        coverage['replicates_per_cell'] = settings['replicates_per_cell']
        for key in ('maximum_object_source_reuse', 'maximum_host_source_reuse'):
            coverage['selection_policy'][key] = settings[key]
    return result


def varied_profiles(family: dict, count: int, variation: dict, seed: int):
    """Return balanced profiles with declared spacing/speed changes, never retries."""
    quotas = profile_quotas(family, count)
    position_key, velocity_key = {
        'billiards': ('position_xy_m', 'linear_velocity_xy_m_s'),
        'passive_pinball': ('local_position_m', 'local_velocity_m_s'),
        'marble_run': ('initial_track_offset_m', 'track_velocity_m_s'),
    }[family['id']]
    grid = list(product(variation['separation_scales'], variation['speed_scales']))
    for profile in sorted(family['profiles'], key=lambda item: item['id']):
        ranked = sorted(grid, key=lambda values: (
            values != (1.0, 1.0),
            sha256_json([seed, family['id'], profile['id'], *values]),
        ))
        for index, (separation, speed) in enumerate(ranked[:quotas[profile['id']]]):
            resolved = copy.deepcopy(profile)
            scalar = family['id'] == 'marble_run'
            positions = [obj[position_key] for obj in profile['objects']]
            positions = [[value] for value in positions] if scalar else positions
            center = [(a+b)/2 for a,b in zip(*positions)]
            for obj, position in zip(resolved['objects'], positions):
                adjusted = [round(mid+(value-mid)*separation, 12) for value,mid in zip(position,center)]
                obj[position_key] = adjusted[0] if scalar else adjusted
                velocity = obj[velocity_key]
                obj[velocity_key] = round(velocity*speed, 12) if scalar else [round(value*speed, 12) for value in velocity]
            state_sha = sha256_json(resolved['objects'])
            identity = {
                'schema_version': 'physweep_specialized_initial_state_variation_v1',
                'profile_id': profile['id'], 'variant_index': index,
                'separation_scale': float(separation), 'speed_scale': float(speed),
                'initial_profile_sha256': state_sha,
            }
            yield resolved, identity
