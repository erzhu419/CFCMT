from copy import deepcopy
import json
import sys
from types import SimpleNamespace

import pytest

from scripts.cluster import collect_tsc_joint_response_measurement as runner


def audit_fixture():
    nominal = ['Gr'] * 3 + ['yr'] * 2 + ['rG'] * 5
    t0 = dict(offset_sec=0, time_sec=600, applied_states={'tls': 'rr'},
        vehicles={'v': {'lane_id': 'in_0'}},
        focal_lane_measurements={
            'in_0': dict(vehicles=1, halted=1, occupancy_raw=.1, mean_speed_mps=0.),
            'out_0': dict(vehicles=0, halted=0, occupancy_raw=0., mean_speed_mps=11.11)})
    observation = dict(frames=[t0] + [dict(offset_sec=i + 1, time_sec=601 + i,
        applied_states={'tls': state}) for i, state in enumerate(nominal)],
        catalog={'m': dict(tls_id='tls', incoming_lanes=['in_0'], outgoing_lanes=['out_0'], internal_lanes=[])},
        lane_geometry={lane: dict(length_m=600.) for lane in ['in_0', 'out_0']})
    static = dict(lane_lengths={'in_0': 600., 'out_0': 600.},
        conflicts_by_tls={'tls': dict(full_context_complete=True)})
    branch = dict(candidate_index=3, reference_index=3, candidate_states=list('ABCDEFGH'),
        resolved_checkpoint_sec=600, start=dict(time_sec=600, pending_ids=[]),
        preaction=dict(nominal_states={'tls': nominal}, nominal_request_accepted={'tls': True},
            movement_ids=['m'], feature_names=['count'], features=[[1.]]),
        network_preaction={'tls_ids': ['tls']}, labels=[[1., 1.]], cost=1., costs=[1.] * 450,
        checks={'full_cost_horizon': True}, measurement={'unresolved_count': 0},
        unobserved_first_receiving_count=0,
        prefix_safety=dict(raw_collision_events=0, no_teleport_passed=True),
        safety=dict(raw_collision_events=0, no_teleport_passed=True))
    return branch, observation, static


def test_signal_audit_compares_ten_completed_steps_without_using_t0_signal():
    branch, observation, static = audit_fixture()
    checks, details = runner.audit_branch(branch, observation, static, 'tls', observation['frames'][0])
    assert all(checks.values())
    assert details['actual_states'] == branch['preaction']['nominal_states']['tls']
    assert details['nominal_actual_mismatch_step_indices'] == []

    observation['frames'][4]['applied_states']['tls'] = 'Gr'
    checks, details = runner.audit_branch(branch, observation, static, 'tls', observation['frames'][0])
    assert not checks['nominal_actual_focal_signals_equal']
    assert details['nominal_actual_mismatch_step_indices'] == [3]

    branch, observation, static = audit_fixture()
    observation['frames'].pop()
    checks, _ = runner.audit_branch(branch, observation, static, 'tls', observation['frames'][0])
    assert not checks['nominal_actual_focal_signals_equal']


@pytest.mark.parametrize('failure,check', [
    ('missing_direct_t0', 'direct_t0_complete'),
    ('missing_direct_field', 'direct_t0_complete'),
    ('count_mismatch', 'direct_counts_match_positions'),
    ('prefix_collision', 'prefix_no_collision'),
    ('branch_collision', 'branch_no_collision'),
])
def test_observation_or_collision_failures_are_rejected(failure, check):
    branch, observation, static = audit_fixture()
    direct = observation['frames'][0]['focal_lane_measurements']
    if failure == 'missing_direct_t0':
        observation['frames'][0].pop('focal_lane_measurements')
    elif failure == 'missing_direct_field':
        direct['in_0'].pop('mean_speed_mps')
    elif failure == 'count_mismatch':
        direct['in_0']['vehicles'] = 2
    elif failure == 'prefix_collision':
        branch['prefix_safety']['raw_collision_events'] = 1
    else:
        branch['safety']['raw_collision_events'] = 1
    checks, _ = runner.audit_branch(branch, observation, static, 'tls', observation['frames'][0])
    assert not checks[check]
    assert not all(checks.values())


def test_first_pp_collision_preserves_invalid_case_and_stops_before_other_actions(monkeypatch, tmp_path):
    branch, observation, static = audit_fixture()
    branch['safety']['raw_collision_events'] = 1
    calls, closed, updates = [], [], []

    def collect(api, v3, case, index, output, *args):
        calls.append(index)
        runner.write_json(output / 'observations_server_only.json', observation)
        return deepcopy(branch)

    monkeypatch.setattr(runner.native, 'collect_branch', collect)
    monkeypatch.setitem(sys.modules, 'cologne_action_branch_checks', SimpleNamespace(
        compare_physical=lambda a, b, tolerance: {'passed': a == b}))
    case = dict(case_id='fixed_source', city='hangzhou', seed=1, tls_id='tls')
    directory = tmp_path / 'case'
    result = runner.collect_case(SimpleNamespace(close=lambda: closed.append(True)), None,
        case, directory, static, lambda case, summary: updates.append(summary))
    assert calls == [None] and len(closed) == len(updates) == 1
    assert result['status'] == 'INVALID' and result['new_branches_attempted'] == 1
    assert result['branches'][0]['checks']['branch_no_collision'] is False
    saved = json.loads((directory / 'result.json').read_text())
    group = json.loads((directory / 'group_data_server_only.json').read_text())
    assert saved == result
    assert not group['complete'] and len(group['X']) == len(group['branches']) == 1
    assert not (directory / 'joint_interface_server_only.json').exists()
    assert not list(directory.glob('action_*'))
