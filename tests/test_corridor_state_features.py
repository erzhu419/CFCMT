"""Observable current service, phase queue units and explicit unknown rows."""
from copy import deepcopy

import pytest

from experiments.tsc_corridor_state_stratification_20261109.state_features import recover_state_features

FULL = (1 << 10)-1
SUFFIX = FULL ^ 15


def example(queue=2):
    return dict(t=100, tls='signal', states=['A', 'B', 'C'], pp=0, gp=2,
                base=[18, 10*queue, 30],
                protected={'north': [SUFFIX, FULL, 0], 'east': [0, 0, SUFFIX]})


@pytest.mark.parametrize('queue,queue_bin', [(0, 'zero'), (2, 'positive')])
def test_current_previous_request_and_aggregate_queue_give_all_four_states(queue, queue_bin):
    row = example(queue)
    saved = deepcopy(row)
    features = recover_state_features(row, 'B')
    assert features['identified'] and features['current_request_index'] == 1
    assert features['current_request'] == 'B'
    assert features['current_phase_aggregate_queue'] == queue
    assert features['by_edge']['north']['stratum'] == ['protected', queue_bin]
    assert features['by_edge']['east']['stratum'] == ['not_protected', queue_bin]
    assert features['by_edge']['north']['queue_positive'] == bool(queue)
    assert row == saved


def test_current_left_phase_can_be_identified_without_any_protected_straight_edge():
    row = example()
    row['protected']['north'][1] = 0
    features = recover_state_features(row, 'B')
    assert features['identified'] and features['current_phase_aggregate_queue'] == 2
    assert all(not edge['monitored_straight_protected'] for edge in features['by_edge'].values())


def test_locked_target_state_is_not_substituted_for_current_phase_or_queue():
    row = dict(states=['B'], pp=0, base=[20], protected={'north': [FULL], 'east': [0]})
    features = recover_state_features(row, 'B')
    assert not features['unlocked'] and not features['identified']
    assert features['reason'] == 'locked'
    assert features['current_request'] is features['current_phase_aggregate_queue'] is None
    assert all(edge['stratum'] == ['unknown', 'unknown'] for edge in features['by_edge'].values())


@pytest.mark.parametrize('previous,bad_base', [(None, False), ('B', True)])
def test_unidentifiable_rows_are_retained_as_unknown(previous, bad_base):
    row = example()
    if bad_base:
        row['base'][1] = 19
    features = recover_state_features(row, previous)
    assert features['unlocked'] and not features['identified']
    assert features['reason'] != 'identified'
    assert features['current_phase_aggregate_queue'] is None
    assert set(features['by_edge']) == set(row['protected'])
    assert all(edge['monitored_straight_protected'] is None for edge in features['by_edge'].values())
