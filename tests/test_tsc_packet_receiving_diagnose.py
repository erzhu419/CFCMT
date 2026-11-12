"""Synthetic paired-admission equivalence and permission-summary semantics."""
from copy import deepcopy

import pytest

from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import PacketObserver
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    corridor, competing_approaches, executor_seed, packet)
from experiments.tsc_target_packet_training_20261109 import conditional_features as old
from experiments.tsc_packet_receiving_path_20261110 import conditional_features as new
from experiments.tsc_packet_receiving_path_20261110.diagnose import (
    paired_admissions, update_permission, _empty_permission, receiving_only_focus)


def frame(static, packets, seeds):
    rows = {p['packet_id']: deepcopy(p) for p in packets}
    return dict(time_s=0, packets=rows, native_lanes=PacketObserver(static)._reconstruct(rows),
        native_pending_ids=[], executors={tls: dict(snapshot=deepcopy(row['snapshot']))
        for tls, row in seeds.items()}, unresolved=[], checks=dict(synthetic=True))


def test_fast_same_kernel_comparison_matches_full_helpers_and_restores_state():
    static, laws = corridor(incoming_length=20, internal_lengths=(10,), outgoing_length=50)
    seeds = {'J': executor_seed(('g',))}
    cases = [(static, laws, seeds,
        [packet('ego', 'in', 19, 2), packet('receiver', 'out', 3, 0, trail=['I0', 'out'])])]
    short, short_laws = corridor(incoming_length=20, internal_lengths=(2, 2), outgoing_length=30)
    cases.append((short, short_laws, {'J': executor_seed(('g',))},
        [packet('ego', 'in', 15, 1), packet('long', 'out', 1, 0, length=8, trail=['in', 'I0', 'I1', 'out'])]))
    conflict, conflict_laws = competing_approaches()
    cases.append((conflict, conflict_laws, {'J': executor_seed(('Gg',))},
        [packet('ego', 'b', 19, 2), packet('priority', 'a', 19, 2)]))
    for static, laws, seeds, packets in cases:
        observed = frame(static, packets, seeds)
        requests = {'J': seeds['J']['snapshot']['current_state']}
        baseline = old.prepare_feature_kernel(static, observed, seeds, laws, requests, 31)
        full_new = new.prepare_feature_kernel(static, observed, seeds, laws, requests, 31)
        before = deepcopy(baseline.state)
        noise = {key: row['intent_rng'].getstate() for key, row in baseline.noise._packets.items()}
        for intent in laws['intent_probs'][static['lanes'][packets[0]['lane_id']]['edge_id']]:
            pair = paired_admissions(baseline, 'ego', intent)
            old_full = old.candidate_features(baseline, 'ego', intent)
            new_full = new.candidate_features(full_new, 'ego', intent)
            assert pair['baseline']['yield_risk'] == old_full['yield_risk']
            assert pair['baseline']['upper'] == pytest.approx(old_full['upper'])
            assert pair['candidate']['yield_risk'] == new_full['yield_risk']
            assert pair['candidate']['admissible'] == new_full['admissible']
            assert pair['candidate']['upper'] == pytest.approx(new_full['upper'])
            for key, value in new_full['admission_detail'].items():
                assert pair['candidate'][key] == pytest.approx(value)
        assert baseline.state == before
        assert noise == {key: row['intent_rng'].getstate() for key, row in baseline.noise._packets.items()}


def test_permission_mass_and_any_expansion_remain_distinct_for_both_labels():
    stats = _empty_permission()
    def pair(old_risk, new_risk):
        return [dict(baseline={'yield_risk': a}, candidate={'yield_risk': b})
                for a, b in zip(old_risk, new_risk)]
    update_permission(stats, 0, [.2, .8], pair([False, True], [True, True]))
    update_permission(stats, 0, [.2, .8], pair([True, False], [False, False]))
    update_permission(stats, 1, [.2, .8], pair([False, False], [False, True]))
    assert stats['0']['rows'] == 2
    assert stats['0']['baseline_permission_mass_sum'] == pytest.approx(1.)
    assert stats['0']['candidate_permission_mass_sum'] == pytest.approx(1.)
    assert stats['0']['any_expansion_count'] == 0 and stats['0']['any_contraction_count'] == 1
    assert stats['1']['any_expansion_count'] == 1
    assert stats['1']['candidate_permission_mass_sum'] == pytest.approx(.8)


def test_receiving_only_focus_excludes_other_connection_and_conflict_overlap():
    row = dict(matches_actual_intent=True, same_physical_connection=True,
               reasons=['current_receiving_space_blocked'])
    assert receiving_only_focus({'candidates': [row]}) is row
    assert receiving_only_focus({'candidates': [dict(row, same_physical_connection=False)]}) is None
    assert receiving_only_focus({'candidates': [dict(row, reasons=row['reasons']+
        ['frozen_other_intents_conflict_reservation'])]}) is None
