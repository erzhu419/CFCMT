"""Shared dependencies, saved-risk comparability and causal query isolation."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.tsc_packet_shared_population_20261110 import query
from experiments.tsc_packet_discrete_braking_20261110.conditional_features import candidate_features
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import packet
from test_tsc_packet_discrete_braking import context


def test_frozen_beta_reaches_background_dependencies_and_changes_focal_entry_permission(monkeypatch):
    packets = {key: packet(key, 'in', 2., 5.) for key in ('focal', 'background')}
    connection = dict(key='movement')
    initial_intents = {key: dict(road_id='in', choice='END') for key in packets}
    kernel = SimpleNamespace(state=dict(packets=packets, intents=deepcopy(initial_intents)),
        topology=SimpleNamespace(lanes={'in': dict(internal=False, length_m=10., edge_id='in')}),
        model=SimpleNamespace(features=lambda row: row))
    kernel._connections = {key: (None, True) for key in packets}
    kernel._connection = lambda row: (connection, True) if kernel.state['intents'][row['packet_id']]['choice'] == 'out' else (None, True)
    kernel._signal = lambda unused: 'g'
    contexts_seen = []
    def canonical(row, current, unused_connection, unused_compatible):
        assert current is packets
        contexts_seen.append(kernel.state['intents']['focal']['choice'])
        feature = np.zeros(38)
        feature[0] = float(row['packet_id'] == 'background')
        return feature
    kernel._context = canonical
    kernel._mean_alignment_context = dict(packets=packets, connections=dict(kernel._connections),
        permit_all={key: {'start': True, 'yield': True} for key in packets},
        zero_raw_cache={}, zero_forecast_cache={})
    monkeypatch.setattr(query, 'physical_proposal', lambda *unused: 10.)
    class DependencyResolver:
        def __init__(self, ctx, unused, decisions, permits, *, focal_id):
            self.context, self.decisions = ctx, decisions
            self.results, self.forecast_cache = {}, {}
        def view(self, unused):
            return packets
        def resolve(self, key):
            if key in self.results:
                return self.results[key]
            proposal = max(0., 10.+min(self.decisions[key]['motion_residual_mps'], 0.))
            travel = min(proposal, self.resolve('background')['travel_m']) if key == 'focal' else proposal
            self.results[key] = dict(travel_m=travel, physical_proposal_mps=proposal,
                same_step_path_bound_m=20., spatial_limit_m=20., spatial_summary={'active_conflict': False})
            return self.results[key]
    monkeypatch.setattr(query, 'MotionResolver', DependencyResolver)
    beta = np.zeros(38); beta[0] = -3.
    result = query.query_pair(kernel, 'focal', 'out', beta)
    assert result['baseline']['mean_mps'] == 10.
    assert result['candidate']['mean_mps'] == 7.
    assert result['baseline']['yield_risk'] and not result['candidate']['yield_risk']
    assert result['candidate']['physical_ceiling_mps'] == result['candidate']['residual_physical_proposal_mps'] == 10.
    assert result['candidate']['diagnostics']['focal_clipped_residual_mps'] == 0.
    assert result['candidate']['diagnostics']['negative_background_scores'] == 1
    assert result['comparison']['changed_background_accepted_responses'] == 1
    assert contexts_seen and set(contexts_seen) == {'out'}
    assert kernel.state['intents'] == initial_intents
    assert kernel._connections['focal'] == (None, True)


@pytest.mark.parametrize('lane,position,speed,intent', [
    ('in', 19., 5., 'out'), ('in', 19., 5., 'END'),
    ('in', 19., 10., 'out'), ('I0', 5., 5., 'out'), ('out', 48., 5., 'END'),
])
def test_general_zero_arm_reproduces_saved_operator_on_crossing_internal_and_end(lane, position, speed, intent):
    kernel = context([packet('ego', lane, position, speed)], signal='g')
    previous = candidate_features(kernel, 'ego', intent)
    pair = query.query_pair(kernel, 'ego', intent, np.zeros(38))
    assert pair['baseline'] == pair['candidate']
    baseline = pair['baseline']
    assert baseline['mean_mps'] == pytest.approx(previous['mean0_mps'], abs=1e-12)
    assert baseline['physical_ceiling_mps'] == pytest.approx(previous['physical_ceiling_mps'], abs=1e-12)
    assert baseline['start_risk'] == previous['start_risk']
    assert baseline['yield_risk'] == previous['yield_risk']
    assert baseline['admissible'] == previous['admissible']
    assert baseline['connection_key'] == previous['connection_key']


def test_nonzero_intent_query_order_preserves_state_background_priors_rng_and_caches():
    packets = [packet('ego', 'in', 19., 5.), packet('leader', 'out', 8., 6.)]
    first, second = context(packets, signal='g'), context(packets, signal='g')
    state, connections = deepcopy(first.state), deepcopy(first._connections)
    frozen = deepcopy(first._mean_alignment_context)
    rng = {key: (row['intent_rng'].getstate(), row['decision_rng'].getstate())
           for key, row in first.noise._packets.items()}
    beta = np.zeros(38); beta[0] = -2.
    forward = {goal: query.query_pair(first, 'ego', goal, beta) for goal in ('out', 'END')}
    reverse = {goal: query.query_pair(second, 'ego', goal, beta) for goal in ('END', 'out')}
    assert forward == reverse
    assert first.state == state and first._connections == connections
    assert first._mean_alignment_context == frozen
    assert rng == {key: (row['intent_rng'].getstate(), row['decision_rng'].getstate())
                   for key, row in first.noise._packets.items()}
