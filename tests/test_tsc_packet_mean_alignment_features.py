"""Current-only theta0 feature/mask and complete frozen-reference tests."""
from copy import deepcopy
import json
from math import isfinite, log, pi

import numpy as np
import pytest

from experiments.tsc_packet_mean_alignment_20261110 import conditional_features as features
from experiments.tsc_packet_mean_alignment_20261110 import audit
from experiments.tsc_packet_entity_footprint_20261110.geometry import build_entity_geometry
from experiments.tsc_target_packet_kernel_20261109.topology import RoadTopology
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import packet, parallel_receivers
from test_tsc_packet_training_data import fixture, persist


def feature_kernel(packets, seed=23):
    static, seeds, frames, actions, protocol = fixture()
    static, laws = parallel_receivers()
    frames[0]['packets'] = {row['packet_id']: deepcopy(row) for row in packets}
    from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import PacketObserver
    frames[0]['native_lanes'] = PacketObserver(static)._reconstruct(frames[0]['packets'])
    table = build_entity_geometry(RoadTopology(static), 2.)
    return features.prepare_feature_kernel(static, frames[0], seeds, laws, {}, seed, conflict_geometry=table)


def test_full38_context_uses_canonical_model_clock_without_next_frame_and_restores_state():
    kernel = feature_kernel([packet('ego', 'a0', 10, 2), packet('rear', 'a1', 4, 3)])
    old = deepcopy(kernel.state)
    old_connections = deepcopy(kernel._connections)
    before_rng = {key: (value['intent_rng'].getstate(), value['decision_rng'].getstate())
                  for key, value in kernel.noise._packets.items()}
    out = features.candidate_features(kernel, 'ego', 'out')
    end = features.candidate_features(kernel, 'ego', 'END')
    repeated = features.candidate_features(kernel, 'ego', 'out')
    assert out == repeated
    assert len(kernel.model.features(out['context'])) == 38
    assert kernel.state == old and kernel._connections == old_connections
    assert before_rng == {key: (value['intent_rng'].getstate(), value['decision_rng'].getstate())
                          for key, value in kernel.noise._packets.items()}
    assert out['mean0_mps'] <= out['physical_ceiling_mps']
    side = out['lane_detail']['sides']['left']
    assert side['neighbour_states']
    rear = side['neighbour_states'][0]
    assert rear['resolved'] and rear['position_m'] > 4
    assert end['lane_detail']['clock'] == 'accepted longitudinal t+1; other lateral actions stay'


def test_same_query_neighbour_motion_does_not_reuse_another_focal_intent_memo():
    one = feature_kernel([packet('ego', 'a0', 17, 4), packet('rear', 'a1', 13, 4)])
    two = feature_kernel([packet('ego', 'a0', 17, 4), packet('rear', 'a1', 13, 4)])
    a = features.candidate_features(one, 'ego', 'out')
    b = features.candidate_features(one, 'ego', 'END')
    b_first = features.candidate_features(two, 'ego', 'END')
    a_last = features.candidate_features(two, 'ego', 'out')
    assert a == a_last and b == b_first


def test_zero_objective_keeps_gaussian_support_and_counts_excluded_discrete_positive():
    motion = dict(head='motion', prior=[.2, .8], risk=[True, True], y=25.,
                  mean0_mps=[4., 6.], physical_ceiling_mps=[10., 10.])
    logp = audit.zero_log_probability('motion', motion)
    assert isfinite(logp)
    stat = audit.new_head_statistics()
    audit.add_statistics(stat, motion, logp, True)
    finished = audit.finish_statistics(stat)
    assert finished['full_objective_status'] == 'FINITE'
    assert finished['observed_above_max_mean0_count'] == finished['observed_above_max_physical_ceiling_count'] == 1
    lane = dict(head='lane_change', prior=[.2, .8], risk=[False, False], y=0,
                mask=[[False, True, False], [False, True, False]])
    logp = audit.zero_log_probability('lane_change', lane)
    assert not isfinite(logp)
    stat = audit.new_head_statistics()
    audit.add_statistics(stat, lane, logp, False)
    finished = audit.finish_statistics(stat)
    assert finished['full_mean_nll'] is None
    assert finished['full_unsupported_labels'] == finished['excluded_positive_labels'] == 1
    assert finished['retained_unsupported_labels'] == 0
    # Stay rows excluded by risk remain probability one in the full reservoir.
    lane['y'] = 1
    assert audit.zero_log_probability('lane_change', lane) == pytest.approx(0.)


def test_zero_mixture_respects_prior_weight_and_label_specific_mask():
    lane = dict(prior=[.25, .75], risk=[True, True], y=0,
                mask=[[True, True, False], [False, True, True]])
    assert audit.zero_log_probability('lane_change', lane) == pytest.approx(log(.25/2))
    start = dict(prior=[.25, .75], risk=[True, False], y=1)
    assert audit.zero_log_probability('start', start) == pytest.approx(log(.25/2))
    start['y'] = 0
    assert audit.zero_log_probability('start', start) == pytest.approx(log(.25/2+.75))
    motion = dict(prior=[1.], risk=[True], y=3., mean0_mps=[3.])
    assert audit.zero_log_probability('motion', motion) == pytest.approx(-.5*log(2*pi))


def test_small_complete_reservoir_rebuild_records_every_excluded_label(tmp_path, monkeypatch):
    static, seeds, frames, actions, protocol = fixture()
    directory = tmp_path/'seed1'
    persist(directory, frames, actions)
    (directory/'executor_seed_server_only.json').write_text(json.dumps(seeds))
    output = tmp_path/'output'
    output.mkdir()
    laws = dict(intent_probs={'a': {'out': .5, 'END': .5}, 'out': {'END': 1.}},
                due_rate_per_sec=0., entry_probs={}, profiles=[])
    references = {head: [dict(seed=1, time_sec=1., packet_id='p', y=1 if head != 'lane_change' else 0)]
                  for head in audit.HEADS}
    sampling = {head: dict(superset_seen=1, sampled=1, retained=0, allzero_excluded=0,
                          allzero_excluded_positive_labels=0) for head in audit.HEADS}
    actual = features.candidate_features
    def no_discrete_support(*args):
        result = actual(*args)
        result.update(start_risk=False, yield_risk=False,
                      lane_mask=dict(left=False, stay=True, right=False))
        return result
    monkeypatch.setattr(audit, 'candidate_features', no_discrete_support)
    task = (1, directory, static, laws, references, sampling, 2., None,
            references['lane_change'], output)
    result = audit._seed(task)
    assert result['heads']['motion']['retained'] == 1
    for head in ('start', 'yield_entry', 'lane_change'):
        assert result['heads'][head]['reservoir_records'] == 1
        assert result['heads'][head]['excluded_positive_labels'] == 1
        assert result['heads'][head]['full_unsupported_labels'] == 1
    with np.load(output/'seed1_full_features_server_only.npz') as arrays:
        assert arrays['lane_change_X'].shape == (1, 2, 38)
        assert not arrays['lane_change_retained'][0]
        assert arrays['motion_mean0_mps'].shape == (1, 2)
        assert 'motion_upper' not in arrays
    summary = audit.summarize([result])
    assert summary['zero_objective']['full']['sum_four_head_mean_nll'] is None
    assert summary['zero_objective']['full']['unsupported_label_count'] == 3
    assert summary['original_lane9'][0]['reference'] == references['lane_change'][0]


def test_saved_shared_prior_dictionary_and_r0_component_reproduce():
    reference = dict(seed=1, time_sec=7., packet_id='ego', y=1)
    key = audit.reference_key(reference)
    prior = {'out': .8, 'END': .2}
    saved = {key: dict(reference=reference, prior=prior, candidates=[
        dict(intent='out', connection_key='c', mean0_mps=6., yield_risk=True),
        dict(intent='END', connection_key=None, mean0_mps=1., yield_risk=False)])}
    candidate = [dict(connection_key='c', mean0_mps=6., yield_risk=True),
                 dict(connection_key=None, mean0_mps=1., yield_risk=False)]
    audit.verify_old_yield(saved, reference, list(prior), prior, candidate)
    assert not saved


def incompatible_kernel():
    from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
        lane, connection, road_graph, laws, executor_seed,
    )
    from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import PacketObserver
    static = road_graph([
        lane('a0', 20, edge='a', index=0), lane('a1', 20, edge='a', index=1),
        lane('Ia', 8, internal=True), lane('Ib', 8, internal=True),
        lane('out_a', 100), lane('out_b', 100)], [
        connection('a1', 'out_a', 'Ia', tls='J', link=0),
        connection('a1', 'out_b', 'Ib', tls='J', link=1),
        connection('Ia', 'out_a'), connection('Ib', 'out_b')],
        foes=((0, 1), (1, 0)), yields=((1, 0),))
    target_laws = laws(static, intents={'a': {'out_a': .5, 'out_b': .5},
                                      'out_a': {'END': 1.}, 'out_b': {'END': 1.}})
    seeds = {'J': executor_seed(('gg',))}
    packets = {row['packet_id']: row for row in [packet('ego', 'a0', 15, 1), packet('neighbour', 'a1', 12, 2)]}
    frame = dict(time_s=0, packets=packets, native_lanes=PacketObserver(static)._reconstruct(packets),
        native_pending_ids=[], executors={tls: dict(snapshot=deepcopy(row['snapshot'])) for tls, row in seeds.items()},
        unresolved=[], checks=dict(synthetic=True))
    table = build_entity_geometry(RoadTopology(static), 2.)
    kernel = features.prepare_feature_kernel(static, frame, seeds, target_laws, {}, 23, conflict_geometry=table)
    kernel.state['intents']['neighbour'] = dict(road_id='a', choice='out_b')
    kernel._connections['neighbour'] = kernel._connection(kernel.state['packets']['neighbour'])
    features.prepare_receiving_context(kernel)
    return kernel


def test_incompatible_focal_borrowed_connection_does_not_become_a_neighbour_foe():
    kernel = incompatible_kernel()
    candidate = features.candidate_features(kernel, 'ego', 'out_a')
    assert candidate['connection_key'] == 'conn_0'
    assert candidate['mean0_mps'] > 0
    side = candidate['lane_detail']['sides']['left']
    assert side['resolved_neighbour_ids'] == ['neighbour']
    assert side['neighbour_states'][0]['resolved']



def test_full_forward_incompatible_borrowed_connection_keeps_only_actual_path_reservations():
    from experiments.tsc_target_packet_kernel_20261109.local_model import FrozenLocalModel
    from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import coefficients
    kernel = incompatible_kernel()
    kernel.state['intents']['ego'] = dict(road_id='a', choice='out_a')
    kernel.model = FrozenLocalModel(coefficients(start=True, yielding=True, lateral='stay'))
    assert kernel.step()['ledger_passed']
    resolved = kernel._last_receiving_resolution
    assert resolved['priority_order'].index('ego') < resolved['priority_order'].index('neighbour')
    ego = resolved['processed']['ego']
    assert not ego['committed'] and not ego['accepted']
    assert ego['packet']['lane_id'] == 'a0' and ego['connection']['from_lane'] == 'a1'
    assert kernel.state['packets']['ego']['lane_id'] == 'a0'
    assert kernel.state['packets']['neighbour']['position_m'] > 12
