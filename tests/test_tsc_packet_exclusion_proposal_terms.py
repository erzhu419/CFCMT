"""Decomposition agrees with the unchanged current-only physical formula."""
import json

import pytest

from experiments.tsc_packet_exclusion_mechanisms_20261110.proposal_terms import proposal_terms
from experiments.tsc_packet_same_step_receiving_20261110.motion import ZERO, physical_proposal
from experiments.tsc_target_packet_kernel_20261109.synthetic_fixtures import packet
from test_tsc_packet_same_step_conditional import context


def readout(kernel, key='ego'):
    p = kernel.state['packets'][key]
    connection, compatible = kernel._connections[key]
    result = proposal_terms(kernel, p, connection, compatible, kernel.state['packets'])
    expected = physical_proposal(kernel, p, ZERO, kernel.state['packets'], connection, compatible)
    assert result['physical_proposal_mps'] == expected
    json.dumps(result, allow_nan=False)
    assert not result['current_body_cap_applied']
    return result


def test_free_proposal_acceleration_and_finite_path_ties_are_reported():
    kernel = context([packet('ego', 'out', 43, 4)])
    result = readout(kernel)
    assert result['physical_proposal_mps'] == 7.
    assert result['binding_terms'] == ['current_speed_plus_acceleration', 'finite_path']
    assert result['leader']['packet_id'] is None
    assert result['signal_stop']['branch'] == 'no_connection'


def test_leader_safe_term_includes_actual_leader_parameters_and_gap():
    kernel = context([packet('ego', 'in', 5, 8), packet('leader', 'in', 12, 0)])
    result = readout(kernel)
    assert result['physical_proposal_mps'] == 0.
    assert result['binding_terms'] == ['leader_safe']
    assert result['leader']['packet_id'] == 'leader'
    assert result['leader']['net_gap_m'] == 0.
    assert result['leader']['parameters'] == kernel.state['packets']['leader']['parameters']


@pytest.mark.parametrize('signal,speed,branch,active', [
    ('r', 5., 'red', True), ('y', 5., 'yellow_cannot_stop', False),
    ('Y', 1., 'yellow_can_stop', True),
])
def test_red_and_both_yellow_branches_reproduce_the_original_stop_rule(signal, speed, branch, active):
    kernel = context([packet('ego', 'in', 19, speed)])
    kernel.signals['J'] = signal
    result = readout(kernel)
    assert result['signal_stop']['branch'] == branch
    assert result['signal_stop']['active'] == active
    if active:
        assert 'signal_stop' in result['binding_terms']
        assert result['signal_stop']['binding_terms'] == ['stationary_stop_safe']


def test_incompatible_connection_uses_only_current_lane_and_the_existing_stop_term():
    kernel = context([packet('ego', 'in', 19, 5)])
    connection, unused = kernel._connections['ego']
    kernel._connections['ego'] = connection, False
    result = readout(kernel)
    assert result['path_lane_ids'] == ['in']
    assert result['signal_stop']['branch'] == 'incompatible_connection'
    assert result['finite_path_remaining_m'] == 1.
