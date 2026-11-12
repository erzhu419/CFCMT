"""Synthetic streaming/law/latent-feature tests; no native start or real fit."""
from copy import deepcopy
import gzip
import json

import numpy as np
import pytest

from experiments.tsc_network_dense_measurement_v2_20261109.packet_observer import PacketObserver
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    executor_seed, packet, parallel_receivers,
)
from experiments.tsc_target_packet_training_20261109 import training_data as data
from experiments.tsc_target_packet_training_20261109.conditional_features import (
    candidate_features, prepare_feature_kernel,
)


def fixture():
    static, unused = parallel_receivers()
    seeds = {'J': executor_seed(('gg',))}
    packets = [
        [packet('p', 'a0', 15, 0)],
        [packet('p', 'a0', 16, 1), packet('q', 'a1', 5, 2)],
        [packet('p', 'I0', 1, 5, trail=['a0', 'I0']), packet('q', 'a1', 6, 1)],
        [packet('p', 'out0', 4, 11, trail=['I0', 'out0'])],
        [packet('p', 'out0', 5, 1), packet('r', 'a1', 5, 3)],
    ]
    frames = []
    for t, group in enumerate(packets):
        rows = {p['packet_id']: p for p in group}
        snapshot = deepcopy(seeds['J']['snapshot'])
        snapshot['green_elapsed_sec'] += t
        events = []
        if t == 2:
            events.append(dict(kind='lane_transition', vehicle_id='p', observed_lane_path=['a0', 'I0']))
        if t == 3:
            events += [dict(kind='lane_transition', vehicle_id='p', observed_lane_path=['I0', 'out0']),
                       dict(kind='terminal_exit', vehicle_id='q', last_lane_id='a1')]
        frames.append(dict(time_s=t, packets=rows, native_lanes=PacketObserver(static)._reconstruct(rows),
            native_pending_ids=[], native_active_ids=list(rows), departed_ids=['q'] if t == 1 else ['r'] if t == 4 else [],
            arrived_ids=['q'] if t == 3 else [], events=events, unresolved=[], checks={'synthetic': True},
            executors={'J': {'snapshot': snapshot}}))
    actions = [dict(time_sec=t, requests={'J': 'gg'} if t == 0 else {}, applied_states={'J': 'gg'},
                    due_ids=['forbidden_future_XML_label'], halted_int=999999) for t in range(4)]
    protocol = dict(cohorts={'training_seeds': [1]},
        proposed_budget={'seconds_per_single_trajectory': 4},
        model={'training': {'reservoir_cap_per_head_seed': 10000, 'reservoir_seed': 76116999}})
    return static, seeds, frames, actions, protocol


def persist(directory, frames, actions):
    directory.mkdir()
    for name, rows in (('frames_server_only.jsonl.gz', frames), ('execution_server_only.jsonl.gz', actions)):
        with gzip.open(directory/name, 'wt') as stream:
            for row in rows:
                stream.write(json.dumps(row)+'\n')


def test_history_END_on_nonsink_censoring_and_due_are_observed_not_XML(tmp_path):
    static, seeds, frames, actions, protocol = fixture()
    directory = tmp_path/'train'; persist(directory, frames, actions)
    laws, reservoirs, sampling, history = data.estimate_laws_and_sample({1: directory}, protocol, static)
    assert history['history_law_estimations'] == 4
    assert history['observed_road_exit_counts']['a'] == {'out': 1, 'END': 1}
    assert history['observed_road_exit_counts']['out'] == {}  # Censored p is not non-END or END.
    assert history['right_censored_ordinary_visits'] == 2
    assert history['inferred_due_count'] == 2 and laws['due_rate_per_sec'] == .5
    assert laws['entry_probs'] == {'a1': 1.}
    assert [p['entry_speed_mps'] for p in laws['profiles']] == [2., 3.]
    assert laws['intent_probs']['a'] == {'out': .5, 'END': .5}
    assert laws['intent_probs']['out'] == {'END': 1.}
    assert sampling['1']['yield_entry']['superset_seen'] == 1
    assert reservoirs[1]['yield_entry'][0]['y'] == 1


def test_reservoir_indices_do_not_depend_on_future_speed_label(tmp_path):
    static, seeds, frames, actions, protocol = fixture()
    protocol['model']['training']['reservoir_cap_per_head_seed'] = 1
    one, two = tmp_path/'one', tmp_path/'two'
    persist(one, frames, actions)
    changed = deepcopy(frames)
    changed[-1]['packets']['p']['speed_mps'] = 19.  # A next-speed label with no later current frame.
    persist(two, changed, actions)
    a = data.estimate_laws_and_sample({1: one}, protocol, static)
    b = data.estimate_laws_and_sample({1: two}, protocol, static)
    strip = lambda result: {h: [(r['time_sec'], r['packet_id']) for r in result[1][1][h]] for h in data.HEADS}
    assert strip(a) == strip(b)
    assert all(a[2]['1'][head]['sampled'] <= 1 for head in data.HEADS)


def test_candidate_hypotheses_restore_intents_connections_and_noise(tmp_path):
    static, seeds, frames, actions, protocol = fixture()
    directory = tmp_path/'train'; persist(directory, frames, actions)
    laws = data.estimate_laws_and_sample({1: directory}, protocol, static)[0]
    kernel = prepare_feature_kernel(static, frames[1], seeds, laws, {}, 23)
    before, connections = deepcopy(kernel.state), deepcopy(kernel._connections)
    noise = {key: row['intent_rng'].getstate() for key, row in kernel.noise._packets.items()}
    move = candidate_features(kernel, 'p', 'out')
    end = candidate_features(kernel, 'p', 'END')
    assert move['yield_risk'] and not end['yield_risk']
    assert move['start_risk'] and end['start_risk']
    assert kernel.state == before and kernel._connections == connections
    assert noise == {key: row['intent_rng'].getstate() for key, row in kernel.noise._packets.items()}
    assert candidate_features(kernel, 'p', 'out') == move


def test_subthreshold_red_light_creep_cannot_label_a_positive_start():
    static, seeds, frames, actions, protocol = fixture()
    seeds = {'J': executor_seed(('gg',), current='rr', mode='all_red', remaining=10.)}
    frame = deepcopy(frames[0])
    frame['packets']['p']['position_m'] = 19.99
    frame['executors']['J']['snapshot'] = seeds['J']['snapshot']
    laws = dict(intent_probs={'a': {'out': .5, 'END': .5}, 'out': {'END': 1.}},
                due_rate_per_sec=0., entry_probs={}, profiles=[])
    kernel = prepare_feature_kernel(static, frame, seeds, laws, {}, 23)
    candidate = candidate_features(kernel, 'p', 'out')
    assert 0 < candidate['upper'] < .1
    assert candidate['motion_risk'] and not candidate['start_risk']


def test_build_scaled_latent_arrays_keeps_END_prior_and_ignores_future_payloads(tmp_path):
    static, seeds, frames, actions, protocol = fixture()
    one, two = tmp_path/'one', tmp_path/'two'
    persist(one, frames, actions)
    changed, changed_actions = deepcopy(frames), deepcopy(actions)
    for frame in changed:
        frame['future_cost'] = -999
        for p in frame['packets'].values():
            p.update(route=['future_target'], next_link='oracle', route_ends_here=True)
    for action in changed_actions:
        action['due_ids'] = ['different_sealed_future']
        action['halted_int'] = -500
    persist(two, changed, changed_actions)
    a = data.build_training_data({1: one}, protocol, static, seeds)
    b = data.build_training_data({1: two}, protocol, static, seeds)
    for head in data.HEADS:
        assert a['heads'][head]['X'].shape[2] == 38
        for key in a['heads'][head]:
            np.testing.assert_array_equal(a['heads'][head][key], b['heads'][head][key])
        np.testing.assert_allclose(a['heads'][head]['prior'].sum(axis=1), 1.)
        assert np.all(a['heads'][head]['X'][:, :, 0][a['heads'][head]['prior'] > 0] == 1.)
    yielding = a['heads']['yield_entry']
    assert yielding['prior'].shape == (1, 2)
    np.testing.assert_array_equal(yielding['risk'][0], [True, False])
    np.testing.assert_array_equal(yielding['prior'][0], [.5, .5])
    # The speed feature is divided by15 once, not an empirical or second scale.
    reference = a['record_references']['yield_entry'][0]
    assert reference['time_sec'] == 1 and reference['packet_id'] == 'p'
    assert yielding['X'][0, 0, 1] == pytest.approx(1/15)


def test_allzero_current_risk_exclusions_are_counted_without_label_rescue(tmp_path, monkeypatch):
    static, seeds, frames, actions, protocol = fixture()
    directory = tmp_path/'train'; persist(directory, frames, actions)
    actual_helper = data.candidate_features

    def no_decision_support(*args):
        candidate = actual_helper(*args)
        candidate.update(start_risk=False, yield_risk=False, lane_mask={'left': False, 'stay': True, 'right': False})
        return candidate

    monkeypatch.setattr(data, 'candidate_features', no_decision_support)
    result = data.build_training_data({1: directory}, protocol, static, seeds)
    assert result['heads']['motion']['X'].shape[0] > 0
    assert result['heads']['start']['X'].shape[0] == result['heads']['yield_entry']['X'].shape[0] == 0
    assert result['sampling']['1']['yield_entry']['allzero_excluded_positive_labels'] == 1
    assert result['sampling']['1']['start']['allzero_excluded_positive_labels'] == 1
    assert result['sampling']['1']['yield_entry']['sampled'] == 1


def test_calibration_seed_cannot_enter_training_preparation(tmp_path):
    static, seeds, frames, actions, protocol = fixture()
    with pytest.raises(ValueError, match='Only all frozen training seeds'):
        data.estimate_laws_and_sample({11: tmp_path/'not_read'}, protocol, static)
