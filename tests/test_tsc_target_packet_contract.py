"""Zero-native, zero-real-data fixtures for the proposed recursive contract."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.tsc_target_packet_protocol_20261109.packet_contract import (
    PARAMETER_FIELDS, EXECUTOR_FIELDS, initialize_observation,
    persistent_intent, assign_pending_entries, partition_insertions,
    terminate_at_road_end, verify_identity_step, bounded_speed_proposal,
)
from experiments.tsc_target_packet_protocol_20261109.calibration_gates import evaluate_calibration


def test_initialization_does_not_copy_sealed_route_or_future_state():
    class Future:
        def __deepcopy__(self, memo):
            raise AssertionError('Future data were read')

    packet = dict(packet_id='car', lane_id='approach', position_m=2., speed_mps=0.,
                  acceleration_mps2=0., lateral_position_m=0., lateral_speed_mps=0.,
                  allowed_speed_mps=10., is_stopped=False, type_id='car',
                  parameters=dict.fromkeys(PARAMETER_FIELDS, 1.),
                  body_trail=['internal', 'approach'], body_path_complete=True,
                  route=Future(), next_TLS=Future(), route_ends_here=Future())
    frame = dict(time_s=600, packets={'car': packet}, native_lanes={'approach': {'N': 1}},
                 native_pending_ids=['waiting'], unresolved=[], checks={'measured': True},
                 executors={'tls': {'snapshot': dict.fromkeys(EXECUTOR_FIELDS, 0)}},
                 future_PP=Future(), sealed_due=Future())
    state = initialize_observation(frame, 600)
    assert set(state['packets']['car']) < set(packet)
    assert state['pending'] == [{'packet_id': 'waiting', 'entry_lane_id': None}]
    assert state['packets']['car']['body_trail'] == ['internal', 'approach']
    state['packets']['car']['body_trail'].append('new')
    assert packet['body_trail'] == ['internal', 'approach']
    with pytest.raises(ValueError, match='anchor'):
        initialize_observation(frame, 599)


def test_end_intent_persists_across_seconds_and_changes_on_new_road():
    inventory = {}
    law = {'straight': .8, 'END': .2}
    assert persistent_intent(inventory, 'car', 'ordinary_with_successors', law, .9) == 'END'
    assert persistent_intent(inventory, 'car', 'ordinary_with_successors', law, .1) == 'END'
    assert not terminate_at_road_end({'position_m': 99}, 'END', 100)
    assert terminate_at_road_end({'position_m': 100}, 'END', 100)
    assert not terminate_at_road_end({'position_m': 100}, 'straight', 100)
    assert persistent_intent(inventory, 'car', 'next_road', law, .1) == 'straight'


def test_residual_cannot_bypass_speed_acceleration_or_receiving_bounds():
    packet = dict(speed_mps=0., acceleration_mps2=0., allowed_speed_mps=10.,
                  parameters=dict(accel_mps2=2., decel_mps2=3., tau_s=1.))
    assert bounded_speed_proposal(packet, 100.) == 2.  # profile limit, not signed last acceleration
    assert bounded_speed_proposal(packet, -100.) == 0.
    packet.update(speed_mps=9., allowed_speed_mps=10.)
    assert bounded_speed_proposal(packet, 100.) == 10.
    assert bounded_speed_proposal(packet, 100., admissible_travel_m=.5) == .5
    assert bounded_speed_proposal(packet, 100., net_gap_m=0., leader_speed_mps=0.) == 0.


def test_blocked_boundary_preserves_assignment_and_identity_mass():
    pending = [{'packet_id': 'wait', 'entry_lane_id': None}]
    assigned = assign_pending_entries(pending, {'entryA': .5, 'entryB': .5}, {'wait': .8})
    _, blocked = partition_insertions(assigned, [])
    reassigned = assign_pending_entries(blocked, {'entryA': .5, 'entryB': .5}, {'wait': .1})
    assert reassigned == [{'packet_id': 'wait', 'entry_lane_id': 'entryB'}]
    assert verify_identity_step(['car'], ['wait'], [], [], [], ['car'], ['wait'])
    inserted, waiting = partition_insertions(reassigned, ['wait'])
    assert inserted[0]['packet_id'] == 'wait' and waiting == []
    assert verify_identity_step(['car'], ['wait'], [], ['wait'], ['car'], ['wait'], [])
    with pytest.raises(ValueError, match='does not close'):
        verify_identity_step(['car'], ['wait'], [], [], [], ['car'], [])
    with pytest.raises(ValueError, match='once'):
        partition_insertions(reassigned, ['wait', 'wait'])


def _protocol_and_windows():
    protocol = json.loads((Path(__file__).parents[1]/'experiments/tsc_target_packet_protocol_20261109/protocol.json').read_text())
    windows = []
    for seed in protocol['cohorts']['calibration_seeds']:
        for anchor in protocol['calibration']['anchors_sec']:
            actual = dict(halted_total=[10]+[5]*450, active_count=[10]*451,
                          terminal_count=10, nonsink_terminal_count=3)
            predicted = dict(halted_total=[0]+[5]*450, active_count=[10]*451,
                             terminal_count=10, nonsink_terminal_count=3)
            windows.append(dict(seed=seed, anchor_sec=anchor, actual=actual,
                                predicted_scenarios=[deepcopy(predicted) for _ in range(8)]))
    return protocol, windows


def test_numerical_gate_uses_poststeps_and_every_window():
    protocol, windows = _protocol_and_windows()
    result = evaluate_calibration(windows, protocol)
    assert result['status'] == 'PASS_NUMERICAL_CALIBRATION'
    assert all(row['actual_C450'] == pytest.approx(5/144) for row in result['windows'])
    assert all(row['absolute_cost_error'] == 0 for row in result['windows'])
    with pytest.raises(ValueError, match='retained'):
        evaluate_calibration(windows[:-1], protocol)


def test_opposite_cost_errors_do_not_cancel_within_seed():
    protocol, windows = _protocol_and_windows()
    for row in windows:
        row['actual']['halted_total'] = [0]+[20]*450
        row['actual']['active_count'] = [40]*451
        for scenario in row['predicted_scenarios']:
            scenario['halted_total'] = [0]+[20]*450
            scenario['active_count'] = [40]*451
    for index, error in ((0, 9), (1, -9)):
        for scenario in windows[index]['predicted_scenarios']:
            scenario['halted_total'] = [0]+[20+error]*450
    result = evaluate_calibration(windows, protocol)
    assert result['status'] == 'HOLD_CALIBRATION'
    assert result['checks']['every_window_cost_error']
    assert not result['checks']['every_seed_cost_MAE']


@pytest.mark.parametrize('field,value', [('active_count', 12), ('terminal_count', 20), ('nonsink_terminal_count', 0)])
def test_good_cost_does_not_hide_wrong_population_or_terminal_prediction(field, value):
    protocol, windows = _protocol_and_windows()
    for scenario in windows[0]['predicted_scenarios']:
        scenario[field] = [value]*451 if field == 'active_count' else value
    result = evaluate_calibration(windows, protocol)
    assert result['status'] == 'HOLD_CALIBRATION'
    assert result['checks']['every_seed_cost_MAE']
