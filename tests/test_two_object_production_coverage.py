from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from tools.cli.plan_two_object_coverage import selection_sha256, verify_input_bindings
from tools.core.hashing import sha256_file
from tools.sampling.sample_two_object_base import _validated_intents
from tools.sampling.sample_two_object_coverage import (
    coverage_cells, coverage_summary, select_coverage_sources,
    source_capacity_bounds, source_selection_audit,
)

ROOT = Path(__file__).resolve().parents[1]


class ProductionCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matrix = json.loads((ROOT / 'configs/two_object_sampling_matrix.json').read_text())
        cls.matrix['coverage_plan']['replicates_per_cell'] = 3
        cls.cells, cls.full_count = coverage_cells(cls.matrix)

    def test_40k_prefix_finishes_two_layers_before_third(self):
        summary = coverage_summary(self.cells[:3068], self.full_count, self.matrix)
        self.assertEqual(summary['logical_cell_count'], 1317)
        self.assertEqual(summary['selected_logical_cell_count'], 1317)
        self.assertEqual(summary['replicate_counts'], {0: 1317, 1: 1317, 2: 434})
        self.assertEqual(summary['logical_cell_multiplicity'], {2: 883, 3: 434})
        self.assertTrue(summary['complete_logical_coverage'])
        self.assertFalse(summary['complete_cartesian_product'])
        self.assertGreaterEqual(summary['axis_counts']['interaction_class']['interacting'] / 3068, .8)

    def test_every_boundary_preserves_coverage_before_repeating(self):
        fields = ('motion_id', 'shape_pair_id', 'scale_pair_id', 'scene_class')
        universe = {tuple(cell[key] for key in fields) for cell in self.cells}
        self.assertEqual(len(universe), 1317)
        counts = Counter()
        for index, cell in enumerate(self.cells):
            key = tuple(cell[field] for field in fields)
            self.assertEqual(cell['replicate_index'], index // 1317)
            self.assertEqual(counts[key], index // 1317)
            counts[key] += 1
        self.assertEqual(set(counts.values()), {3})

    def test_first_layer_and_cameras_match_single_replicate(self):
        matrix = copy.deepcopy(self.matrix)
        matrix['coverage_plan']['replicates_per_cell'] = 1
        single, count = coverage_cells(matrix)
        self.assertEqual(count, 1317)
        self.assertEqual(single, self.cells[:1317])

    def test_reuse_limits_are_explicit_positive_integers(self):
        matrix = copy.deepcopy(self.matrix)
        policy = matrix['coverage_plan']['selection_policy']
        for key in ('maximum_object_source_reuse', 'maximum_host_source_reuse'):
            for value in (0, -1, True, 2.5, '3', None):
                with self.subTest(key=key, value=value):
                    changed = copy.deepcopy(matrix)
                    changed['coverage_plan']['selection_policy'][key] = value
                    with self.assertRaisesRegex(ValueError, 'positive integers'):
                        _validated_intents(changed)
            policy[key] = 3
        _validated_intents(matrix)
        policy['source_reuse_scope'] = 'reset_each_replicate'
        with self.assertRaisesRegex(ValueError, 'may not be weakened'):
            _validated_intents(matrix)

    @staticmethod
    def small_pool():
        objects = [
            {'metadata': {}, 'source': {'scene_id': f'object_{index}'},
             'source_family': 'generic', 'shape_family_id': 'sphere',
             'scale_bin': 'small', 'visual_profile_id': f'profile_{index % 2}'}
            for index in range(6)
        ]
        hosts = [
            {'metadata': {}, 'source': {'scene_id': f'host_{index}'},
             'scene_rule_id': 'ground_patch_flat', 'scene_class': 'ground_flat',
             'visual_profile_id': 'host_profile', 'visual_type': 'procedural_room',
             'environment_category': 'minimal'}
            for index in range(3)
        ]
        cells = [
            {'cell_id': f'cell_{index}', 'replicate_index': index // 3,
             'motion_id': 'surface_single_independent_2obj', 'scene_class': 'ground_flat',
             'camera_view_family_id': 'side_left_mid',
             'object_a_shape': 'sphere', 'object_b_shape': 'sphere',
             'object_a_scale_bin': 'small', 'object_b_scale_bin': 'small'}
            for index in range(9)
        ]
        return cells, objects, hosts

    def test_assignment_keeps_global_caps_and_pair_uniqueness_across_layers(self):
        matrix = copy.deepcopy(self.matrix)
        policy = matrix['coverage_plan']['selection_policy']
        policy['maximum_object_source_reuse'] = 3
        policy['maximum_host_source_reuse'] = 3
        cells, objects, hosts = self.small_pool()
        selected = select_coverage_sources(cells, objects, hosts, matrix)
        repeated = select_coverage_sources(cells, objects[::-1], hosts[::-1], matrix)
        self.assertEqual(selection_sha256(selected), selection_sha256(repeated))
        audit = source_selection_audit(selected, matrix)
        self.assertEqual(audit['unique_unordered_pairs'], 9)
        self.assertEqual(audit['object_reuse_histogram'], {3: 6})
        self.assertEqual(audit['host_reuse_histogram'], {3: 3})
        swapped = copy.deepcopy(selected)
        swapped[-1]['objects'] = list(reversed(swapped[0]['objects']))
        with self.assertRaisesRegex(ValueError, 'unordered source pair'):
            source_selection_audit(swapped, matrix)
        policy['maximum_host_source_reuse'] = 2
        with self.assertRaisesRegex(ValueError, 'across replicates'):
            source_selection_audit(selected, matrix)

    def test_capacity_rejects_shape_shortage_even_when_global_total_is_enough(self):
        cells, objects, hosts = self.small_pool()
        cells = cells[:4]
        for obj in objects[2:]:
            obj['shape_family_id'] = 'cuboid'
        capacity = source_capacity_bounds(cells, objects, hosts, self.matrix)
        self.assertEqual(capacity['deficits'], [
            {'kind': 'object_shape_scale', 'key': ['sphere', 'small'], 'required': 8, 'available': 4}
        ])
        with self.assertRaisesRegex(ValueError, 'insufficient declared'):
            select_coverage_sources(cells, objects, hosts, self.matrix)

    def test_planning_rejects_input_or_code_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'
            source.write_text('{}')
            binding = {'source': {'path': str(source), 'sha256': sha256_file(source)}}
            with patch('tools.cli.plan_two_object_coverage.generation_code_sha256', return_value='first'):
                verify_input_bindings(binding, 'first')
                with self.assertRaisesRegex(ValueError, 'code changed'):
                    verify_input_bindings(binding, 'older')
                source.write_text('{"changed": true}')
                with self.assertRaisesRegex(ValueError, 'input changed'):
                    verify_input_bindings(binding, 'first')


if __name__ == '__main__':
    unittest.main()
