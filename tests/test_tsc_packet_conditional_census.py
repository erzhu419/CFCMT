"""Current-only census preserves the complete manifest and frozen geometry."""
from copy import deepcopy
import gzip
import json
import subprocess
import sys

import pytest

from experiments.tsc_packet_conditional_census_20261110 import census
from experiments.tsc_packet_conditional_domain_20261110.current import GeometryView
from experiments.tsc_packet_conditional_domain_20261110.budget import symbolic_budget
from experiments.tsc_packet_conditional_law_20261110.selection import select_stochastic
from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import PacketObserver
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    corridor, lane, laws, packet, parallel_receivers, road_graph,
)


def current_frame(static, rows, time_sec=0):
    packets = {row['packet_id']: deepcopy(row) for row in rows}
    return dict(time_s=time_sec, packets=packets,
        native_lanes=PacketObserver(static)._reconstruct(packets),
        native_pending_ids=[], executors={}, unresolved=[], checks=dict(synthetic=True))


def lateral_view(count=4):
    static = road_graph([lane('a'+str(i), 200, edge='a', index=i) for i in range(5)], [])
    rows = [packet('focal', 'a2', 40, 5), packet('right', 'a1', 44, 5),
            packet('left', 'a3', 45, 5), packet('outer', 'a4', 52, 5),
            packet('extra', 'a2', 50, 5)][:count]
    return GeometryView(static, current_frame(static, rows), laws(static), {}, 23)


@pytest.mark.parametrize('count', [4, 5])
def test_frame_memo_preserves_all_focal_sets_bounds_reasons_and_cap(count):
    view = lateral_view(count)
    before = deepcopy(view.state)
    expected = {focal: select_stochastic(view, focal, max_stochastic=4)
                for focal in view.state['packets']}
    memo = census.FrameSelector(view)
    for focal, original in expected.items():
        actual = memo.select(focal, max_stochastic=4)
        assert actual == original
        assert symbolic_budget(view, actual) == symbolic_budget(view, original)
    focal = memo.select('focal', max_stochastic=4)
    assert len(focal['required_participants']) == count
    assert focal['status'] == ('SELECTED_GEOMETRIC_ENVELOPE' if count == 4
                               else 'OBSTRUCTION_STOCHASTIC_SET_EXCEEDS_CAP')
    assert view.state == before


def test_frame_memo_keeps_an_entire_frame_intent_obstruction_instead_of_pruning():
    static = road_graph([lane('a0', 200, edge='a', index=0),
                         lane('a1', 200, edge='a', index=1), lane('b', 200)], [])
    history = laws(static)
    history['intent_probs']['b'] = {'END': 0.}
    view = GeometryView(static, current_frame(static, [packet('focal', 'a0', 40, 5),
        packet('near', 'a1', 44, 5), packet('bad_far_background', 'b', 150, 0)]), history, {}, 23)
    original = select_stochastic(view, 'focal', max_stochastic=4)
    actual = census.FrameSelector(view).select('focal', max_stochastic=4)
    assert actual == original
    assert actual['status'] == 'OBSTRUCTION_GEOMETRIC_INTENT_DOMAIN'
    assert actual['bounds'] == {}
    assert symbolic_budget(view, actual)['physical_resolver_calls_upper_bound'] is None


def test_frame_envelopes_are_computed_once_and_reused_for_every_focal(monkeypatch):
    view = lateral_view(4)
    calls = []
    original = census._envelope
    def counted(current, row):
        calls.append(row['packet_id'])
        return original(current, row)
    monkeypatch.setattr(census, '_envelope', counted)
    memo = census.FrameSelector(view)
    for focal in view.state['packets']:
        memo.select(focal, max_stochastic=4)
    assert calls == list(view.state['packets'])


def test_current_head_selectors_match_frozen_law_for_entry_start_and_receiving_lanes():
    from experiments.tsc_packet_joint_law_definition_20261110.law import _eligibility

    static, history = corridor(incoming_length=20, internal_lengths=(10,), outgoing_length=50)
    rows = [packet('far_stopped', 'in', 5, 0), packet('near', 'in', 19, 5),
            packet('internal', 'I0', 7, 1), packet('out', 'out', 20, 3)]
    current = current_frame(static, rows)
    for signal in ('g', 'G', 'r'):
        view = GeometryView(static, current, history, {'J': signal}, 23)
        for row in view.state['packets'].values():
            assert census.current_eligibility(view, row) == _eligibility(view, row)
        assert census.current_eligibility(view, view.state['packets']['far_stopped'])['start']
        assert census.current_eligibility(view, view.state['packets']['near'])['yield_entry'] == (signal == 'g')
    static, history = parallel_receivers()
    view = GeometryView(static, current_frame(static, [packet('receiver', 'I1', 7, 10)]),
                        history, {'J': 'GG'}, 23)
    actual = census.current_eligibility(view, view.state['packets']['receiver'])
    assert actual == _eligibility(view, view.state['packets']['receiver'])
    assert actual['lane_change']


def test_production_import_has_no_physical_or_likelihood_dependencies():
    source = '''
import sys
from experiments.tsc_packet_conditional_census_20261110 import census
for name in (
    'experiments.tsc_packet_discrete_braking_20261110.motion',
    'experiments.tsc_packet_joint_law_definition_20261110.law',
    'experiments.tsc_packet_joint_law_definition_20261110.observation',
    'experiments.tsc_packet_conditional_cache_20261110.cache',
    'experiments.tsc_target_packet_training_20261109.training_data',
):
    assert name not in sys.modules, name
'''
    subprocess.run([sys.executable, '-c', source], check=True)


def reference(seed, time_sec, packet_id, y=0):
    return dict(seed=seed, time_sec=time_sec, packet_id=packet_id, y=y)


def test_grouping_shares_current_cases_and_preserves_head_ordinals_without_labels():
    seed = census.SEEDS[0]
    rows = {head: [] for head in census.HEADS}
    rows['motion'] = [reference(seed, 0, 'ego', 8.5), reference(seed, 1, 'other', 7.)]
    rows['lane_change'] = [reference(seed, 0, 'ego', 2)]
    grouped = census.group_references(rows, seed)
    assert grouped == {0: {'ego': [('motion', 0), ('lane_change', 0)]},
                       1: {'other': [('motion', 1)]}}
    changed = deepcopy(rows)
    for values in changed.values():
        for row in values:
            row['y'] = -999
    assert census.group_references(changed, seed) == grouped


def test_original_four_seed_four_head_budget_is_required():
    manifest = {str(seed): {head: [reference(seed, 0, 'synthetic', 0)]*10000
                           for head in census.HEADS} for seed in census.SEEDS}
    census.validate_manifest(manifest)
    incomplete = deepcopy(manifest)
    incomplete[str(census.SEEDS[0])]['motion'].pop()
    with pytest.raises(ValueError, match='10000'):
        census.validate_manifest(incomplete)
    incomplete = dict(manifest)
    incomplete.pop(str(census.SEEDS[-1]))
    with pytest.raises(ValueError, match='four original training seeds'):
        census.validate_manifest(incomplete)


def _registered_case(view):
    selection = census.FrameSelector(view).select('focal', max_stochastic=4)
    budget = symbolic_budget(view, selection)
    return dict(status=census.REGISTERED if budget['status'] == census.REGISTERED else selection['status'],
        required_actor_count=len(selection['required_participants']),
        physical_calls_upper=budget['physical_resolver_calls_upper_bound'],
        leaf_visits_upper=budget['lateral_leaf_branches_upper_bound'])


def test_partial_cost_never_becomes_full_cost_when_a_case_or_head_is_undefined():
    valid, outside = _registered_case(lateral_view(4)), _registered_case(lateral_view(5))
    first = census.empty_statistics()
    census.add_reference(first, valid, True)
    census.add_reference(first, valid, True)  # The same current law serves two original heads.
    complete = census.finalize_statistics(first)
    assert complete['reference_records'] == 2 and complete['full_target_defined']
    assert complete['full_full_forward_upper_bound'] == 2*valid['physical_calls_upper']
    second = census.empty_statistics()
    census.add_reference(second, outside, True)
    census.add_reference(second, valid, False)
    partial = census.merge_statistics([complete, census.finalize_statistics(second)])
    assert partial['reference_records'] == 4 and partial['geometry_in_domain'] == 3
    assert partial['eligible_records'] == 3 and partial['defined_records'] == 2
    assert partial['defined_subset_full_forward_upper_bound'] == 2*valid['physical_calls_upper']
    assert partial['full_full_forward_upper_bound'] is None and partial['full_leaf_upper_bound'] is None
    assert not partial['full_target_defined']
    assert partial['disposition_counts'] == {'DEFINED_TARGET_REFERENCE': 2,
        'OBSTRUCTION_STOCHASTIC_SET_EXCEEDS_CAP': 1, 'OUTSIDE_HEAD_SELECTOR': 1}


def test_seed_streams_only_current_inputs_and_outputs_all_head_linkage_without_y(tmp_path, monkeypatch):
    seed = census.SEEDS[0]
    static = road_graph([lane(road+str(i), 200, edge=road, index=i)
                         for road in ('a', 'b') for i in range(5)], [])
    rows = []
    for road, count in [('a', 4), ('b', 5)]:
        for suffix, lane_index, position in [('focal', 2, 40), ('right', 1, 44),
                                           ('left', 3, 45), ('outer', 4, 52), ('extra', 2, 50)][:count]:
            rows.append(packet(road+'_'+suffix, road+str(lane_index), position, 5))
    frame = current_frame(static, rows)
    cache = tmp_path/'cache'
    training = cache/'training'/f'seed{seed}'
    training.mkdir(parents=True)
    with gzip.open(training/'frames_server_only.jsonl.gz', 'wt') as stream:
        stream.write(json.dumps(frame)+'\n')
        stream.write('NOT_A_DECODED_NEXT_OBSERVATION\n')
    with gzip.open(training/'execution_server_only.jsonl.gz', 'wt') as stream:
        stream.write(json.dumps(dict(time_sec=0, applied_states={}))+'\n')
        stream.write('NOT_A_DECODED_FUTURE_ACTION\n')
    reservoir = {head: [] for head in census.HEADS}
    reservoir['motion'] = [reference(seed, 0, 'a_focal', 8.), reference(seed, 0, 'b_focal', 9.)]
    reservoir['lane_change'] = [reference(seed, 0, 'a_focal', 0), reference(seed, 0, 'b_focal', 2)]
    reservoir['start'] = [reference(seed, 0, 'a_focal', 1)]
    output = tmp_path/'first'
    output.mkdir()
    parsed = []
    original_loads = census.json.loads
    def capture(text):
        value = original_loads(text)
        parsed.append((value.get('time_s'), value.get('time_sec')))
        return value
    with monkeypatch.context() as scoped:
        scoped.setattr(census.json, 'loads', capture)
        result = census._seed((seed, cache, static, laws(static), reservoir, output))
    assert parsed == [(0, None), (None, 0)]
    assert result['current_frames_decoded'] == result['interval_actions_decoded'] == 1
    assert result['native_next_pairs_formed'] == 0 and result['unique_current_cases'] == 2
    stats = result['reference_statistics']
    assert stats['reference_records'] == 5 and stats['defined_records'] == 2
    assert stats['defined_subset_full_forward_upper_bound'] == 32
    assert result['shared_defined_subset_full_forward_upper_bound'] == 16
    assert result['full_shared_target_full_forward_upper_bound'] is None
    with gzip.open(output/f'seed{seed}_references_server_only.jsonl.gz', 'rt') as stream:
        details = [json.loads(line) for line in stream]
    assert len(details) == 5 and all('y' not in row for row in details)
    assert len({(row['head'], row['manifest_ordinal']) for row in details}) == 5
    linked = [row for row in details if row['packet_id'] == 'a_focal']
    assert len(linked) == 3 and len({row['case_id'] for row in linked}) == 1
    for values in reservoir.values():
        for row in values:
            row['y'] = -999
    second = tmp_path/'second'
    second.mkdir()
    repeat = census._seed((seed, cache, static, laws(static), reservoir, second))
    assert repeat['reference_statistics'] == stats
    assert repeat['heads'] == result['heads']
    assert repeat['shared_defined_subset_full_forward_upper_bound'] == 16
    with gzip.open(second/f'seed{seed}_references_server_only.jsonl.gz', 'rt') as stream:
        assert [json.loads(line) for line in stream] == details
