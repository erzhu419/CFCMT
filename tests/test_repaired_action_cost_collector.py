"""The PP discovery is an arm, not an extra simulation or a selected outcome."""
import sys
from types import SimpleNamespace

from experiments.tsc_repaired_action_cost_pilot_20261109 import collect_pilot as collector


def setup_helpers(monkeypatch):
    monkeypatch.setitem(sys.modules, 'cologne_action_branch_checks', SimpleNamespace(
        compare_physical=lambda a, b, tolerance: {'passed': a == b}))
    monkeypatch.setitem(sys.modules, 'recover_tsc_joint_response_trial', SimpleNamespace(
        revised_audit=lambda *args: ({'prefix_no_collision': False, 'branch_no_collision': False,
                                     'execution_conforms_to_clearance_rule': True}, {})))


def test_pp_discovery_is_one_of_eight_and_resolved_state_is_fixed(tmp_path, monkeypatch):
    setup_helpers(monkeypatch)
    calls, closes = [], []
    api = SimpleNamespace(simulation=SimpleNamespace(getTime=lambda: 1070), close=lambda: closes.append(1))

    def branch(api, v3, case, candidate, output):
        if candidate is None:
            assert 'resolved_checkpoint_sec' not in case
        else:
            assert case['resolved_checkpoint_sec'] == 620
        calls.append(candidate)
        index = 5 if candidate is None else candidate
        return dict(candidate_index=index, reference_index=5, candidate_states=list('abcdefgh'),
            resolved_checkpoint_sec=620, start={'pending_ids': [], 'vehicles': ['same']},
            checks={'complete': True}, observation={'frames': [{'offset_sec': 0}]},
            preaction={'movement_ids': ['m']}, network_preaction={'state': 'same'},
            labels=[[index, 0]], cost=index/100, integrals={'450': index*36}, segments={},
            controlled_lane_count=8, prefix_safety={'raw_collision_events': 0},
            safety={'raw_collision_events': 1}, future={'error': None})

    monkeypatch.setattr(collector, 'collect_branch', branch)
    result = collector.collect_case(api, None, {'case_id': 'fixed', 'checkpoint_sec': 600,
        'tls_id': 'signal'}, tmp_path/'group', {}, [])
    assert result['status'] == 'COMPLETE'
    assert calls == [None, 0, 1, 2, 3, 4, 6, 7]
    assert len(closes) == result['new_branches_attempted'] == 8
    assert [row['candidate_index'] for row in result['action_table']] == list(range(8))
    assert result['action_table'][5]['cost_minus_pp'] == 0
    assert all(row['safety']['raw_collision_events'] == 1 for row in result['branches'])


def test_first_failed_measurement_is_kept_without_replacement_or_zero_labels(tmp_path, monkeypatch):
    setup_helpers(monkeypatch)
    calls, closes = [], []
    api = SimpleNamespace(close=lambda: closes.append(1))

    def failed(api, v3, case, candidate, output):
        calls.append(candidate)
        raise ValueError('No eligible epoch in the frozen window')

    monkeypatch.setattr(collector, 'collect_branch', failed)
    result = collector.collect_case(api, None, {'case_id': 'fixed'}, tmp_path/'group', {}, [])
    assert result['status'] == 'HOLD_INCOMPLETE_MEASUREMENT'
    assert calls == [None] and closes == [1]
    assert result['new_branches_attempted'] == 1
    assert 'action_table' not in result and 'labels_10s' not in result['branches'][0]
    assert (tmp_path/'group'/'first_pp'/'result.json').exists()
