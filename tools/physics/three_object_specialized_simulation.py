"""Explicit three-sphere specialized execution and shared numerical admission."""
from pathlib import Path
import json
import numpy as np

from tools.core.contact_event_contract import validate_contact_event_evidence
from tools.core.hashing import sha256_file
from tools.motion_rules.three_object.billiards import validate_billiards_contract
from tools.motion_rules.three_object.pinball import validate_pinball_contract
from tools.motion_rules.three_object.marble import validate_marble_contract
from tools.motion_rules.three_object.interaction import (
    expected_check_ids, audit_hard_results, interaction_checks,
)
from tools.physics.specialized_sphere_simulation import simulate_specialized_spheres, _billiards_fixture, _pinball_fixture, _marble_fixture


def integrity_configuration(adapter='billiards_three_object_v1'):
    filenames = {'billiards_three_object_v1': 'three_object_billiards_integrity.json',
                 'passive_pinball_three_object_v1': 'three_object_pinball_integrity.json',
                 'marble_run_three_object_v1': 'three_object_marble_integrity.json'}
    if adapter not in filenames: raise ValueError('unsupported specialized integrity adapter')
    path = Path(__file__).resolve().parents[2] / 'configs' / filenames[adapter]
    return json.loads(path.read_text()), {'path': str(path), 'sha256': sha256_file(path)}


def audit_three_object_billiards(scene, arrays, execution):
    """Require runtime evidence for every object; only motion is advisory in sweeps."""
    if scene['backend_binding']['adapter_id'] != 'billiards_three_object_v1' or len(scene['objects']) != 3:
        raise ValueError('three-object billiards audit requires its explicit adapter')
    contract = validate_billiards_contract(scene['source_metadata'])
    limits, binding = integrity_configuration()
    support = scene['adapter_payload']['backend']['billiards_rules']['support_dynamics']
    return _audit_three_spheres(scene, arrays, execution, contract, limits, binding,
        [0., support['lateral_friction'], support['restitution']], ['pool_table'],
        'physweep_three_object_billiards_audit_v1')


def audit_three_object_pinball(scene, arrays, execution):
    if scene['backend_binding']['adapter_id'] != 'passive_pinball_three_object_v1' or len(scene['objects']) != 3:
        raise ValueError('three-object pinball audit requires its explicit adapter')
    contract = validate_pinball_contract(scene['source_metadata'])
    limits, binding = integrity_configuration('passive_pinball_three_object_v1')
    fixture = scene['adapter_payload']['fixture']; support = fixture['material']
    return _audit_three_spheres(scene, arrays, execution, contract, limits, binding,
        [0., support['contact_friction'], support['contact_restitution']], [c['id'] for c in fixture['colliders']],
        'physweep_three_object_pinball_audit_v1', minimum_path_m=contract['minimum_path_length_per_object_m'])


def audit_three_object_marble(scene, arrays, execution):
    if scene['backend_binding']['adapter_id']!='marble_run_three_object_v1' or len(scene['objects'])!=3:
        raise ValueError('three-object marble audit requires its explicit adapter')
    contract=validate_marble_contract(scene['source_metadata'])
    limits,binding=integrity_configuration('marble_run_three_object_v1')
    fixture=scene['adapter_payload']['fixture']
    support=[(c,fixture['mesh_material']) for c in fixture['mesh_components']]+[(c,fixture['analytic_material']) for c in fixture['analytic_colliders']]
    return _audit_three_spheres(scene,arrays,execution,contract,limits,binding,
        [[0.,material['contact_friction'],material['contact_restitution']] for _,material in support],[c['id'] for c,_ in support],
        'physweep_three_object_marble_audit_v1',minimum_path_m=contract['minimum_path_length_per_object_m'])


def _audit_three_spheres(scene, arrays, execution, contract, limits, binding,
                         expected_support, expected_support_ids, audit_schema, minimum_path_m=None):
    objects = scene['objects']; count = scene['time']['frame_count']
    ids = [o['object_id'] for o in objects]; classes = expected_check_ids(objects)
    checks = []; metrics = {}; energy = np.zeros(count)
    tolerance = limits['parameter_absolute_tolerance']

    def check(name, passed, value, expected):
        checks.append({'id': name, 'category': classes[name], 'passed': bool(passed),
                       'value': value, 'expected': expected})

    def evidence_array(name, shape):
        value = np.asarray(arrays[name])
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f'missing, nonfinite or incomplete three-sphere evidence: {name}')
        return value

    pos = evidence_array('position_m', (count, 3, 3))
    vel = evidence_array('linear_velocity_m_s', (count, 3, 3))
    angular = evidence_array('angular_velocity_rad_s', (count, 3, 3))
    q = evidence_array('quaternion_wxyz', (count, 3, 4))
    material = evidence_array('runtime_material', (3, 3))
    extras = evidence_array('runtime_material_extras', (3, 4))
    inertia = evidence_array('inertia_diagonal_kg_m2', (3, 3))
    distances = evidence_array('adapter__minimum_contact_distance_m', (count, 3))
    times = evidence_array('time_s', (count,))
    expected_time = np.arange(count) / scene['time']['output_fps']
    check('time_axis', np.allclose(times, expected_time, rtol=0, atol=limits['time_absolute_tolerance_s']),
          times.tolist(), expected_time.tolist())
    proxies = execution['runtime_proxies']; supports = execution['runtime_support']
    if len(proxies) != 3 or not supports:
        raise ValueError('missing runtime proxy or support evidence')
    actual_support = np.asarray([[s['mass_kg'], s['lateral_friction'], s['restitution']] for s in supports])
    support_ok = ([s['fixture_id'] for s in supports] == expected_support_ids and
                  np.isfinite(actual_support).all() and np.allclose(actual_support, expected_support, rtol=0, atol=tolerance))
    gravity = np.asarray(scene['world']['gravity_m_s2'])
    for i, obj in enumerate(objects):
        oid = obj['object_id']; prefix = oid + '__'; initial = obj['initial_state']; m = obj['material']
        check(prefix + 'finite_state', True, None, 'finite F x channel arrays')
        iq = np.asarray(initial['orientation_quaternion_xyzw'])[[3, 0, 1, 2]]
        orientation_error = min(np.linalg.norm(q[0, i] - iq), np.linalg.norm(q[0, i] + iq))
        initial_ok = all(np.allclose(a[0, i], initial[key], rtol=0, atol=tolerance) for a, key in (
            (pos, 'position_m'), (vel, 'linear_velocity_m_s'), (angular, 'angular_velocity_rad_s')))
        initial_ok = initial_ok and orientation_error <= tolerance and np.max(np.abs(np.linalg.norm(q[:, i], axis=1) - 1)) <= tolerance
        check(prefix + 'initial_state', initial_ok, float(orientation_error), tolerance)
        expected_material = [m[k] for k in ('mass_kg', 'contact_friction', 'contact_restitution',
                                          'rolling_friction', 'spinning_friction', 'linear_damping', 'angular_damping')]
        actual_material = np.concatenate([material[i], extras[i]])
        check(prefix + 'runtime_material', np.allclose(actual_material, expected_material, rtol=0, atol=tolerance),
              actual_material.tolist(), expected_material)
        check(prefix + 'runtime_support', support_ok, actual_support.tolist(), expected_support)
        radius = obj['collision_proxy']['radius_m']; expected_inertia = np.full(3, .4 * m['mass_kg'] * radius ** 2)
        check(prefix + 'runtime_inertia', (inertia[i] > 0).all() and np.allclose(inertia[i], expected_inertia,
              rtol=limits['inertia_relative_tolerance'], atol=limits['inertia_absolute_tolerance_kg_m2']),
              inertia[i].tolist(), expected_inertia.tolist())
        proxy = proxies[i]
        proxy_ok = proxy['shape_type'] == 2
        for key, expected in [('dimensions_m', [radius] * 3), ('local_position_m', [0.] * 3)]:
            actual = np.asarray(proxy[key]); target = np.asarray(expected)
            proxy_ok = proxy_ok and actual.shape == target.shape and np.allclose(actual, target, rtol=0, atol=tolerance)
        pq = np.asarray(proxy['local_quaternion_xyzw']); identity = np.array([0., 0., 0., 1.])
        proxy_ok = proxy_ok and pq.shape == (4,) and min(np.linalg.norm(pq - identity), np.linalg.norm(pq + identity)) <= tolerance
        check(prefix + 'runtime_proxy', proxy_ok, proxy, 'one centered sphere with declared radius')
        penetration = max(0., -float(distances[:, i].min()))
        check(prefix + 'bounded_penetration', penetration <= limits['maximum_contact_penetration_m'],
              penetration, limits['maximum_contact_penetration_m'])
        # A sphere has isotropic inertia: angular energy is unchanged by orientation.
        energy += (.5 * m['mass_kg'] * np.sum(vel[:, i] ** 2, axis=1) +
                   .5 * np.sum(angular[:, i] ** 2 * inertia[i], axis=1) - m['mass_kg'] * (pos[:, i] @ gravity))
        metrics[oid] = {'maximum_speed_m_s': float(np.linalg.norm(vel[:, i], axis=1).max()), 'maximum_penetration_m': penetration}
    gain = float(energy.max() - energy[0])
    scale = max(float(energy[0] - energy.min()), sum(o['material']['mass_kg'] * np.linalg.norm(gravity) *
                2 * o['collision_proxy']['radius_m'] for o in objects), 1e-9)
    allowance = limits['maximum_unforced_energy_gain_fraction'] * scale + limits['maximum_unforced_energy_gain_j_per_kg'] * sum(o['material']['mass_kg'] for o in objects)
    check('bounded_group_energy', gain <= allowance, gain, allowance)
    evidence = execution['interaction_events']; hz = scene['time']['simulation_hz']
    steps = (count - 1) * (hz // scene['time']['output_fps'])
    events = validate_contact_event_evidence(evidence, ids, hz, steps, contract['thresholds']['contact_distance_tolerance_m'])
    check('event_evidence', True, len(events), 'complete consecutive substep coverage')
    checks.extend(interaction_checks(contract, events, hz, {oid: vel[0, i] for i, oid in enumerate(ids)}))
    if minimum_path_m is not None:
        paths = np.asarray(execution['path_lengths'])
        if paths.shape != (3,) or not np.isfinite(paths).all() or (paths < 0).any():
            raise ValueError('missing or invalid three-object substep path evidence')
        output_paths = np.linalg.norm(np.diff(pos, axis=0), axis=2).sum(axis=0)
        if (paths + tolerance < output_paths).any():
            raise ValueError('substep path evidence shorter than observed output polyline')
        response = next(c for c in checks if c['id'] == 'template_motion_response')
        response['passed'] = bool(response['passed'] and (paths >= minimum_path_m).all())
        response['path_evidence'] = {'path_m_by_object': dict(zip(ids, paths.tolist())),
                                     'minimum_object_path_m': minimum_path_m, 'sampling': 'consecutive_simulation_substeps'}
    is_sweep = scene['variant']['kind'] == 'sweep'
    for record in checks:
        if is_sweep and record['category'] == 'base_semantics': record['severity'] = 'advisory'
    audit = {'schema_version': audit_schema, 'scene_id': scene['scene_id'],
             'integrity_configuration': binding, 'checks': checks, 'metrics': metrics, 'contact_events': evidence,
             'advisories': [r for r in checks if r.get('severity') == 'advisory'],
             'solver_execution': execution['solver_execution'], 'contact_processing_execution': execution['contact_processing_execution'],
             'contact_processing_evidence': 'pybullet_changeDynamics_arguments',
             'runtime_material_evidence': 'getDynamicsInfo except damping retained from changeDynamics arguments',
             'contact_count_sampling': 'maximum_over_preceding_output_interval; frame_zero_initial_state'}
    audit['passed'] = all(audit_hard_results(objects, audit, is_sweep))
    return audit


def simulate_three_object_specialized(scene, root):
    branches = {'billiards_three_object_v1': (validate_billiards_contract, _billiards_fixture, audit_three_object_billiards),
                'passive_pinball_three_object_v1': (validate_pinball_contract, _pinball_fixture, audit_three_object_pinball),
                'marble_run_three_object_v1': (validate_marble_contract, _marble_fixture, audit_three_object_marble)}
    adapter = scene['backend_binding']['adapter_id']
    if adapter not in branches or len(scene['objects']) != 3:
        raise ValueError('unsupported specialized three-object adapter or count')
    validate, fixture, audit = branches[adapter]
    contract = validate(scene['source_metadata'])
    arrays, execution = simulate_specialized_spheres(scene, root, fixture_builder=fixture,
        event_distance_tolerance_m=contract['thresholds']['contact_distance_tolerance_m'])
    return arrays, audit(scene, arrays, execution)
