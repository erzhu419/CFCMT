"""Focal residual clips and query isolation, without optimizer or native data."""
from copy import deepcopy

import pytest

from experiments.tsc_packet_mean_alignment_20261110.mean import (
    conditional_motion_mean, make_conditional_resolver)
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import packet
from test_tsc_packet_same_step_conditional import context


def query(kernel, intent, residual):
    pid = 'ego'
    prior = deepcopy(kernel.state['intents'].get(pid))
    old_connection = kernel._connections[pid]
    try:
        focal = kernel.state['packets'][pid]
        road = kernel.topology.lanes[focal['lane_id']]['edge_id']
        kernel.state['intents'][pid] = dict(road_id=road, choice=intent)
        connection, compatible = kernel._connection(focal)
        kernel._connections[pid] = connection, compatible
        return conditional_motion_mean(kernel._same_step_receiving_context, kernel, pid,
                                       connection, compatible, residual_mps=residual)
    finally:
        if prior is None:
            kernel.state['intents'].pop(pid, None)
        else:
            kernel.state['intents'][pid] = prior
        kernel._connections[pid] = old_connection


def test_original_residual_clip_is_retained_without_using_projected_mu0_as_upper():
    kernel = context([packet('ego', 'in', 5, 7)])
    negative = query(kernel, 'out', -100.)
    nominal = query(kernel, 'out', 0.)
    positive = query(kernel, 'out', 100.)
    assert negative['clipped_residual_mps'] == -4.
    assert positive['clipped_residual_mps'] == 3.
    assert negative['mean_mps'] == pytest.approx(6.)
    assert nominal['mean_mps'] == positive['mean_mps'] == pytest.approx(10.)
    assert negative['physical_ceiling_mps'] == nominal['physical_ceiling_mps'] == 10.


def test_nonzero_and_candidate_query_order_leave_state_rng_and_original_caches_unchanged():
    packets = [packet('ego', 'in', 19, 5), packet('leader', 'out', 5, 6)]
    first, second = context(packets), context(packets)
    old_state = deepcopy(first.state)
    old_cache = deepcopy(first._conditional_zero_speeds)
    old_decisions = deepcopy(first._same_step_receiving_context['zero_decisions'])
    rng = {key: (row['intent_rng'].getstate(), row['decision_rng'].getstate())
           for key, row in first.noise._packets.items()}
    conditions = [('out', -2.), ('END', -1.), ('out', 0.), ('out', 1.)]
    answers = {(intent, residual): query(first, intent, residual) for intent, residual in conditions}
    reversed_answers = {(intent, residual): query(second, intent, residual)
                        for intent, residual in reversed(conditions)}
    assert answers == reversed_answers
    assert first.state == old_state and first._conditional_zero_speeds == old_cache
    assert first._same_step_receiving_context['zero_decisions'] == old_decisions
    assert rng == {key: (row['intent_rng'].getstate(), row['decision_rng'].getstate())
                   for key, row in first.noise._packets.items()}


def test_creator_does_not_share_accepted_motion_memo_between_residual_queries():
    kernel = context([packet('ego', 'I0', 5, 5), packet('leader', 'out', 5, 6)])
    current = kernel._same_step_receiving_context
    connection, compatible = kernel._connections['ego']
    first = make_conditional_resolver(current, kernel, 'ego', connection, compatible, -2.)
    second = make_conditional_resolver(current, kernel, 'ego', connection, compatible, 0.)
    first.resolve('ego')
    assert first.results and not second.results and not second.processed
    assert first.working['ego'] is not second.working['ego']
