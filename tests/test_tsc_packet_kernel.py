"""Synthetic full-kernel acceptance; no native simulator, raw data or fitting."""
from copy import deepcopy

import pytest

from experiments.tsc_target_packet_kernel_20261109.kernel import PacketKernel
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import (
    coefficients, competing_approaches, corridor, executor_seed, initial_observation,
    laws, packet, parallel_receivers, turn_lane_two_changes,
)


def make_kernel(static, target_laws, packets=(), *, pending=(), seeds=None, coeffs=None):
    seeds = seeds or {}
    initial = initial_observation(static, packets, pending, seeds)
    return PacketKernel(deepcopy(static), initial, deepcopy(seeds), deepcopy(target_laws),
                        coefficients=coeffs or coefficients(), random_seed=0)


def transition_for(row, key):
    return next(event for event in row['events']
                if event['kind'] == 'lane_transition' and event['vehicle_id'] == key)


def test_one_second_can_cross_multiple_internal_lanes_without_losing_travel_or_mass():
    static, target_laws = corridor(incoming_length=10, internal_lengths=(2, 3))
    seeds = {'J': executor_seed(('G',), clearance=('I0', 'I1'))}
    kernel = make_kernel(static, target_laws, [packet('p', 'in', 9, 12)], seeds=seeds)
    row = kernel.step(initial_requests={'J': 'G'})
    p = kernel.state['packets']['p']
    assert p['lane_id'] == 'out' and p['position_m'] == pytest.approx(9.)
    assert p['speed_mps'] == pytest.approx(15.)
    assert transition_for(row, 'p')['observed_lane_path'] == ['in', 'I0', 'I1', 'out']
    assert row['ledger_passed'] and not row['exited_ids']
    assert sum(sensor['N'] for sensor in kernel.state['lane_sensors'].values()) == 1


def test_packet_already_internal_continues_when_entry_signal_is_red():
    static, target_laws = corridor(incoming_length=10, internal_lengths=(2, 3))
    seeds = {'J': executor_seed(('G',), current='r', mode='all_red', remaining=5,
                                 clearance=('I0', 'I1'))}
    kernel = make_kernel(static, target_laws,
        [packet('inside', 'I0', 1, 5, trail=['in', 'I0'])], seeds=seeds)
    row = kernel.step()
    assert kernel.state['packets']['inside']['lane_id'] == 'out'
    assert kernel.state['packets']['inside']['position_m'] == pytest.approx(4.)
    assert kernel.state['executors']['J']['mode'] == 'all_red'
    assert row['ledger_passed']


def test_protected_entry_precedes_yielding_foe_and_existing_internal_packet_blocks_g():
    static, target_laws = competing_approaches()
    seeds = {'J': executor_seed(('Gg',), clearance=('Ia', 'Ib'))}
    priority = make_kernel(static, target_laws,
        [packet('protected', 'a', 19, 10), packet('yielding', 'b', 19, 10)], seeds=seeds)
    priority.step()
    assert priority.state['packets']['protected']['lane_id'] in ('Ia', 'out_a')
    assert priority.state['packets']['yielding']['lane_id'] == 'b'
    assert priority.state['packets']['yielding']['position_m'] <= 20
    unopposed = make_kernel(static, target_laws, [packet('yielding', 'b', 19, 10)], seeds=seeds)
    unopposed.step()
    assert unopposed.state['packets']['yielding']['lane_id'] in ('Ib', 'out_b')
    occupied = make_kernel(static, target_laws,
        [packet('inside', 'Ia', 5, 0), packet('yielding', 'b', 19, 10)],
        seeds=seeds, coeffs=coefficients(start=0))
    occupied.step()
    assert occupied.state['packets']['inside']['lane_id'] == 'Ia'
    assert occupied.state['packets']['yielding']['lane_id'] == 'b'


def test_receiving_long_vehicle_tail_blocks_upstream_and_occupancy_can_exceed_one():
    static, target_laws = corridor(incoming_length=20, internal_lengths=(5,), outgoing_length=30)
    seeds = {'J': executor_seed(('G',), clearance=('I0',))}
    kernel = make_kernel(static, target_laws,
        [packet('long', 'out', 3, 0, length=12, trail=['in', 'I0', 'out']),
         packet('following', 'in', 14, 2)], seeds=seeds, coeffs=coefficients(start=0))
    assert kernel.state['lane_sensors']['I0']['O'] == pytest.approx(1.8)
    assert kernel.state['lane_sensors']['I0']['N'] == 0
    row = kernel.step()
    assert kernel.state['packets']['following']['lane_id'] == 'in'
    assert kernel.state['packets']['following']['position_m'] <= 14+1e-10
    assert kernel.state['packets']['long']['position_m'] == 3
    assert kernel.state['lane_sensors']['I0']['O'] == pytest.approx(1.8)
    assert sum(sensor['N'] for sensor in kernel.state['lane_sensors'].values()) == 2
    assert row['ledger_passed']


def test_connection_then_lane_change_separates_front_path_from_remapped_body_occupancy():
    static, target_laws = parallel_receivers()
    seeds = {'J': executor_seed(('GG',), clearance=('I0', 'I1'))}
    kernel = make_kernel(static, target_laws, [packet('p', 'I1', 7, 1)],
                         seeds=seeds, coeffs=coefficients(lateral='right'))
    row = kernel.step()
    p = kernel.state['packets']['p']
    assert p['lane_id'] == 'out0' and p['position_m'] == pytest.approx(3.)
    event = transition_for(row, 'p')
    assert event['transition_type'] == 'connection_then_lane_change'
    assert event['observed_lane_path'] == ['I1', 'out1', 'out0']
    assert p['body_trail'] == ['I0', 'out0']
    assert kernel.state['lane_sensors']['I0']['O'] == pytest.approx(.25)
    assert kernel.state['lane_sensors']['I1']['O'] == 0
    assert kernel.state['lane_sensors']['I0']['N'] == kernel.state['lane_sensors']['I1']['N'] == 0
    assert row['ledger_passed']


def test_blocked_pending_is_retained_and_END_exits_a_nonsink_without_next_signal():
    static, target_laws = competing_approaches(with_foes=False)
    target_laws['intent_probs']['a'] = {'END': 1.}
    target_laws['intent_probs']['b'] = {'END': 1.}
    target_laws['entry_probs'] = {'a': 1.}
    seeds = {'J': executor_seed(('GG',), clearance=('Ia', 'Ib'))}
    waiting = make_kernel(static, target_laws, [packet('block', 'a', 5, 0)], pending=['waiting'],
                          seeds=seeds, coeffs=coefficients(start=0))
    for _ in range(3):
        row = waiting.step()
        assert row['inserted_ids'] == [] and row['ledger_passed']
        assert {p['packet_id'] for p in waiting.state['pending']} == {'waiting'}
        assert waiting.state['pending'][0]['entry_lane_id'] == 'a'
    red = {'J': executor_seed(('GG',), current='rr', mode='all_red', remaining=10,
                               clearance=('Ia', 'Ib'))}
    ending = make_kernel(static, target_laws, [packet('end', 'b', 19, 5)], seeds=red)
    assert static['successors']['b']  # It is not a topological sink.
    row = ending.step()
    assert row['exited_ids'] == ['end'] and ending.state['packets'] == {}
    assert row['ledger_passed']


def test_PP_runs_every_ten_seconds_and_clearance_polls_predicted_front_count_not_body_only():
    static, target_laws = corridor(incoming_length=20, internal_lengths=(10,))
    red = {'J': executor_seed(('G',), current='r', mode='all_red', remaining=1, clearance=('I0',))}
    kernel = make_kernel(static, target_laws,
        [packet('queue', 'in', 5, 0), packet('inside', 'I0', 5, 0)],
        seeds=red, coeffs=coefficients(start=0))
    rows = [kernel.step() for _ in range(11)]
    assert [index for index, row in enumerate(rows) if row['requests']] == [0, 10]
    assert all(row['ledger_passed'] for row in rows)
    assert kernel.state['executors']['J']['mode'] == 'all_red'
    assert kernel.state['executors']['J']['remaining_sec'] == 1
    assert kernel.state['lane_sensors']['I0']['N'] == 1
    short, short_laws = corridor(incoming_length=20, internal_lengths=(5,), outgoing_length=30)
    body_only = make_kernel(short, short_laws,
        [packet('long', 'out', 3, 0, length=12, trail=['in', 'I0', 'out'])],
        seeds=red, coeffs=coefficients(start=0))
    assert body_only.state['lane_sensors']['I0']['O'] > 1
    body_only.step()
    assert body_only.state['lane_sensors']['I0']['N'] == 0
    assert body_only.state['executors']['J']['mode'] == 'green'


def test_450_steps_preserve_generated_due_pending_terminal_ledgers_and_poststep_cost():
    static, target_laws = competing_approaches(with_foes=False)
    target_laws.update(due_rate_per_sec=.2, entry_probs={'a': 1.})
    target_laws['intent_probs'].update(a={'END': 1.}, b={'END': 1.})
    seeds = {'J': executor_seed(('GG',), clearance=('Ia', 'Ib'))}
    kernel = make_kernel(static, target_laws,
        [packet('block', 'a', 5, 0), packet('end', 'b', 19, 5)], pending=['initial_waiting'],
        seeds=seeds, coeffs=coefficients(start=0))
    result = kernel.rollout(450)
    rows = result['steps']
    due = [key for row in rows for key in row['due_ids']]
    assert due and len(set(due)) == len(due)
    assert all(row['ledger_passed'] for row in rows)
    assert [key for row in rows for key in row['exited_ids']] == ['end']
    assert all(not row['inserted_ids'] for row in rows)
    assert set(kernel.state['packets']) == {'block'}
    assert {p['packet_id'] for p in kernel.state['pending']} == {'initial_waiting', *due}
    assert kernel.state['time_s'] == 450
    assert sum(row['halted_total'] for row in rows) == rows[-1]['I_accum'] == 450
    assert rows[-1]['I_accum']/(450*len(static['controlled_lane_ids'])) == .5
    assert sum(bool(row['requests']) for row in rows) == 45
    assert result['completed_steps'] == 450 and result['ledger_passed']
    assert result['I_accum'] == 450 and result['C450'] == .5
    assert len(result['halted_total']) == len(result['active_count']) == len(result['pending_count']) == 451


def test_450_seconds_have_continuing_insertions_motion_and_exits_with_closed_identity_cost():
    static, target_laws = corridor(incoming_length=30, internal_lengths=(5, 5), outgoing_length=40)
    target_laws.update(due_rate_per_sec=.15, entry_probs={'in': 1.})
    seeds = {'J': executor_seed(('G',), clearance=('I0', 'I1'))}
    kernel = make_kernel(static, target_laws, [packet('initial', 'in', 10, 8)], seeds=seeds)
    result = kernel.rollout(450)
    rows = result['steps']
    due = [key for row in rows for key in row['due_ids']]
    inserted = [key for row in rows for key in row['inserted_ids']]
    exited = [key for row in rows for key in row['exited_ids']]
    assert len(inserted) > 10 and len(exited) > 10
    assert any(row['inserted_ids'] for row in rows[300:])
    assert any(row['exited_ids'] for row in rows[300:])
    assert any(event['kind'] == 'lane_transition' for row in rows[300:] for event in row['events'])
    assert len(set(due)) == len(due) and len(set(inserted)) == len(inserted) and len(set(exited)) == len(exited)
    active = set(kernel.state['packets'])
    pending = {row['packet_id'] for row in kernel.state['pending']}
    assert not active & pending and not (active | pending) & set(exited)
    assert {'initial', *due} == active | pending | set(exited)
    assert all(row['ledger_passed'] for row in rows) and result['ledger_passed']
    assert result['completed_steps'] == kernel.state['time_s'] == 450
    assert result['I_accum'] == sum(row['halted_total'] for row in rows) > 0
    assert result['C450'] == pytest.approx(result['I_accum']/(450*len(static['controlled_lane_ids'])))


def test_predicted_queue_PP_switches_two_phases_through_yellow_allred_and_green():
    static, target_laws = competing_approaches(with_foes=False)
    seeds = {'J': executor_seed(('Gr', 'rG'), min_green=2, clearance=('Ia', 'Ib'))}
    kernel = make_kernel(static, target_laws,
                         [packet('a_car', 'a', 10, 2), packet('b_queue', 'b', 19, 0)], seeds=seeds)
    rows = [kernel.step() for _ in range(15)]
    assert [index for index, row in enumerate(rows) if row['requests']] == [0, 10]
    assert rows[0]['requests'] == {'J': 'rG'}
    assert rows[10]['requests'] == {'J': 'Gr'}
    assert [rows[index]['applied_signals']['J'] for index in range(4)] == ['Yr', 'Yr', 'rr', 'rG']
    assert rows[2]['executors']['J']['current_state'] == 'rG'  # Poststep state is not applied-step state.
    assert [rows[index]['applied_signals']['J'] for index in range(10, 14)] == ['rY', 'rY', 'rr', 'Gr']
    assert all(row['ledger_passed'] for row in rows)


def test_incompatible_turn_stays_persistent_across_two_required_lane_changes():
    static, target_laws = turn_lane_two_changes()
    seeds = {'J': executor_seed(('G',), clearance=('I0',))}
    kernel = make_kernel(static, target_laws, [packet('turner', 'a0', 17, 2)],
                         seeds=seeds, coeffs=coefficients(lateral='left'))
    first = kernel.step()
    assert kernel.state['packets']['turner']['lane_id'] == 'a1'
    assert transition_for(first, 'turner')['observed_lane_path'] == ['a0', 'a1']
    original_intent = deepcopy(kernel.state['intents']['turner'])
    assert original_intent == {'road_id': 'a', 'choice': 'out'}
    second = kernel.step()
    assert kernel.state['packets']['turner']['lane_id'] == 'a2'
    assert transition_for(second, 'turner')['observed_lane_path'] == ['a1', 'a2']
    assert kernel.state['intents']['turner'] == original_intent
    for _ in range(4):
        row = kernel.step()
        assert row['ledger_passed']
        if kernel.state['packets']['turner']['lane_id'] == 'out':
            break
    assert kernel.state['packets']['turner']['lane_id'] == 'out'
    assert kernel.state['packets']['turner']['body_path_complete']
