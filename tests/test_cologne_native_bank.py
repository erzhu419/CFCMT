from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from cf_h2o.traffic_signal.mechanism_world_model import MechanismDataset
from scripts.data.cologne_native_bank import (
    TARGETS, PROTOCOL, B195_PROTOCOL, build_native_group, assemble_native_bank, assert_complete_b100,
    assemble_expanded_native_bank, assert_complete_b195,
)


def fixture_group(number=0):
    group = f'cologne1:seed2027:{number}:cluster_357187_359543'
    phases = [f'phase{i}' for i in range(8)]
    original = MechanismDataset(
        feature_names=('queue',), features=np.arange(8)[:, None].astype(float),
        context_names=('lanes',), context=np.ones((8, 1)) * 8,
        priors={key: np.arange(8) + 100.0 for key in TARGETS},
        targets={key: np.arange(8) + 200.0 for key in TARGETS},
        domains=np.asarray(['cologne1'] * 8), metadata={
            'scenario': 'cologne1', 'sumocfg': 'fixed.sumocfg',
            'signal_routing_graph': {'adjacency': {'cluster_357187_359543': []}},
            'action_group_ids': [group] * 8, 'candidate_states': phases,
            'row_tls': ['cluster_357187_359543'] * 8, 'row_times': [25200 + number * 10] * 8,
            'state_snapshot_protocol': {'old': True}, 'behavior_trace_sha256': 'old',
            'counterfactual_replay_audit': {'passed': True},
            'counterfactual_safety_audit': {'protocol': 'old'},
        })
    safe = dict(raw_collision_events=0, starting_teleports=0, ending_teleports=0,
                no_teleport_passed=True)
    results, traces = [], {}
    for index, phase in enumerate(phases):
        costs = np.arange(450) / 8 + index
        traces[f'{group}:native:{index}'] = costs.tolist()
        results.append(dict(candidate_index=index, phase=phase, status='PASS',
            checks={'features_exact': True, 'full_horizon': True},
            prefix_safety=safe.copy(), safety=safe.copy(), cost=float(costs.mean()),
            sample_count=450, end_time_sec=25650 + number * 10,
            first_interval_costs=costs[:10].tolist(),
            one_step=dict(total_q=10, green_q=11, red_q=12, green_down_occ=.13, mean_speed=14),
            terminal=dict(total_q=20, green_q=21, red_q=22, green_down_occ=.23, mean_speed=24)))
    result = dict(group_id=group, time_sec=25200 + number * 10, candidate_states=phases,
                  native_branches=results, status='INVALID', reused=True,
                  restored_pp_control={'status': 'PASS', 'stored_label_exact': False})
    return original, result, traces


def test_reused_native_labels_replace_all_targets_and_preserve_original_inputs():
    original, result, traces = fixture_group()
    before = deepcopy(original)
    data = build_native_group(original, result, traces)
    assert data.targets['next_total_queue'].tolist() == [20] * 8
    assert data.targets['one_step_total_queue'].tolist() == [10] * 8
    assert data.targets['terminal_system_load'][0] == 449 / 8
    assert data.targets['interval_cost'][0] == np.mean(np.arange(450) / 8)
    np.testing.assert_array_equal(data.targets['prefix_mean_cost_450s'], data.targets['interval_cost'])
    np.testing.assert_array_equal(data.features, original.features)
    np.testing.assert_array_equal(data.context, original.context)
    for name in TARGETS:
        np.testing.assert_array_equal(data.priors[name], original.priors[name])
        np.testing.assert_array_equal(original.targets[name], before.targets[name])
    assert data.metadata['protocol'] == PROTOCOL
    assert data.metadata['native_group_sources'][0]['source_group_status'] == 'INVALID'
    assert data.metadata['signal_routing_graph'] == original.metadata['signal_routing_graph']
    assert data.metadata['state_restores'] == 0
    assert not any(key in data.metadata for key in
                   ('state_snapshot_protocol', 'counterfactual_replay_audit',
                    'behavior_trace_sha256', 'counterfactual_safety_audit'))


@pytest.mark.parametrize('failure', ['unsafe', 'prefix_unsafe', 'incomplete', 'trace', 'phase'])
def test_bad_branch_prevents_the_entire_group_entering_bank(failure):
    original, result, traces = fixture_group()
    branch = result['native_branches'][3]
    if failure == 'unsafe':
        branch['status'] = 'FAIL_UNSAFE'
        branch['safety']['raw_collision_events'] = 1
    elif failure == 'prefix_unsafe':
        branch['prefix_safety']['starting_teleports'] = 1
    elif failure == 'incomplete':
        result['native_branches'].pop()
    elif failure == 'trace':
        traces[f"{result['group_id']}:native:3"][-1] += 1
    elif failure == 'phase':
        branch['phase'] = 'wrong_phase'
    with pytest.raises(ValueError):
        build_native_group(original, result, traces)


def test_only_the_complete_original_b100_roster_can_fit():
    parts = [build_native_group(*fixture_group(i)) for i in range(100)]
    expected = [part.metadata['action_group_ids'][0] for part in parts]
    partial = assemble_native_bank(parts[:-1], expected, failed_groups=expected[-1:])
    assert partial.size == 792
    assert partial.metadata['native_collection_audit']['failed_groups'] == expected[-1:]
    assert not partial.metadata['fit_ready']
    repeated = assemble_native_bank([partial], expected)
    assert repeated.metadata['native_collection_audit']['failed_groups'] == expected[-1:]
    with pytest.raises(ValueError, match='complete unchanged original B100'):
        assert_complete_b100(partial, expected)
    complete = assemble_native_bank(list(reversed(parts)), expected)
    assert complete.size == 800
    assert_complete_b100(complete, expected)
    assert complete.metadata['action_group_ids'][::8] == expected
    with pytest.raises(ValueError, match='overlap'):
        assemble_native_bank(parts + parts[:1], expected)
    with pytest.raises(ValueError, match='failed group'):
        assemble_native_bank(parts, expected, failed_groups=expected[:1])
    with pytest.raises(ValueError, match='outside the frozen roster'):
        assemble_native_bank(parts, expected[:-1])


def test_seed_shards_concatenate_without_replacing_original_selection():
    parts = [build_native_group(*fixture_group(i)) for i in range(100)]
    expected = [part.metadata['action_group_ids'][0] for part in parts]
    shards = [assemble_native_bank(parts[a:b], expected[a:b])
              for a, b in ((0, 34), (34, 67), (67, 100))]
    assert all(not shard.metadata['fit_ready'] for shard in shards)
    complete = assemble_native_bank(shards, expected)
    assert_complete_b100(complete, expected)
    altered = replace(parts[0], metadata={**parts[0].metadata,
                                       'state_snapshot_protocol': {'old': True}})
    with pytest.raises(ValueError, match='old snapshot'):
        assemble_native_bank([altered], expected[:1])


def k_group(number):
    original, result, traces = fixture_group(number)
    for branch in result['native_branches']:
        branch['status'] = 'COMPLETE'
        for key in ('prefix_safety', 'safety'):
            branch[key].update(raw_collision_events=2, unique_collision_incidents=1)
    return original, result, traces


def test_k_retains_complete_colliding_labels_with_truthful_metadata_but_p_remains_strict():
    original, result, traces = k_group(100)
    original.metadata['strict_safety_monitoring'] = {'counterfactual_collision_handling': 'censor_entire_matched_action_group'}
    data = build_native_group(original, result, traces, native_protocol=B195_PROTOCOL)
    np.testing.assert_array_equal(data.features, original.features)
    assert data.targets['interval_cost'][0] == np.mean(traces[f"{result['group_id']}:native:0"])
    assert data.metadata['strict_safety_monitoring']['counterfactual_collision_handling'] == 'retain_complete_matched_action_group_and_audit_collisions'
    assert original.metadata['strict_safety_monitoring']['counterfactual_collision_handling'] == 'censor_entire_matched_action_group'
    source = data.metadata['native_group_sources'][0]
    assert source['collision_admission_gate'] is False
    assert source['branch_safety'][0]['prefix_raw_collision_events'] == 2
    assert source['branch_safety'][0]['window_raw_collision_events'] == 2
    for branch in result['native_branches']:
        branch['status'] = 'PASS'
    with pytest.raises(ValueError, match='unsafe'):
        build_native_group(original, result, traces)
    result['native_branches'][0]['status'] = 'COMPLETE'
    result['native_branches'][0]['sample_count'] = 449
    with pytest.raises(ValueError, match='trace'):
        build_native_group(original, result, traces, native_protocol=B195_PROTOCOL)


def test_k_expansion_preserves_all_p_rows_and_requires_exactly95_complete_additions():
    original_parts = [build_native_group(*fixture_group(index)) for index in range(100)]
    original_ids = [part.metadata['action_group_ids'][0] for part in original_parts]
    original = assemble_native_bank(original_parts, original_ids)
    additional = [build_native_group(*k_group(index), native_protocol=B195_PROTOCOL) for index in range(100, 195)]
    expected = original_ids + [part.metadata['action_group_ids'][0] for part in additional]
    with pytest.raises(ValueError, match='All95'):
        assemble_expanded_native_bank(original, additional[:-1], expected)
    expanded = assemble_expanded_native_bank(original, list(reversed(additional)), expected)
    assert_complete_b195(expanded, expected)
    assert expanded.size == 1560
    np.testing.assert_array_equal(expanded.features[:800], original.features)
    for target in TARGETS:
        np.testing.assert_array_equal(expanded.targets[target][:800], original.targets[target])
        np.testing.assert_array_equal(expanded.priors[target][:800], original.priors[target])
    sources = expanded.metadata['native_group_sources']
    assert sum(source['native_bank_protocol'] == PROTOCOL for source in sources) == 100
    assert sum(source['native_bank_protocol'] == B195_PROTOCOL for source in sources) == 95
    assert all(source['collision_admission_gate'] for source in sources[:100])
    assert expanded.metadata['reused_original_b100_groups'] == original_ids
    assert original.metadata['native_bank_protocol'] == PROTOCOL
