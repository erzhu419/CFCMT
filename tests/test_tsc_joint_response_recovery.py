from copy import deepcopy
import json
import shutil
import sys
from types import SimpleNamespace

import pytest

from scripts.cluster import recover_tsc_joint_response_trial as recovery


def read(path):
    return json.loads(path.read_text())


def fixture_branch(index):
    return dict(candidate_index=index, reference_index=3, candidate_states=list('ABCDEFGH'),
        start=dict(time_sec=620, pending_ids=[]), resolved_checkpoint_sec=620,
        preaction=dict(feature_names=['count'], movement_ids=['m'], features=[[float(index)]],
                       nominal_states={'tls': ['rG'] * 10}, nominal_request_accepted={'tls': True}),
        labels=[[index, index + 1]], cost=float(index), costs=[float(index)] * 450,
        checks={'full_cost_horizon': True}, network_preaction={'tls_ids': ['tls']},
        measurement={'unresolved_count': 0}, unobserved_first_receiving_count=0,
        prefix_safety=dict(raw_collision_events=0, no_teleport_passed=True),
        safety=dict(raw_collision_events=0, no_teleport_passed=True))


def observation(index):
    return dict(candidate=index, frames=[dict(time_sec=620, offset_sec=0, vehicles={})],
                preaction=fixture_branch(index)['preaction'], catalog={}, lane_geometry={})


def install_mocks(monkeypatch, calls, invalid_index=None):
    def collect(api, v3, case, index, output, *unused):
        calls.append(index)
        index = 3 if index is None else index
        recovery.write_json(output / 'observations_server_only.json', observation(index))
        return fixture_branch(index)
    monkeypatch.setattr(recovery.measured.native, 'collect_branch', collect)
    monkeypatch.setattr(recovery.measured, 'audit_branch', lambda *args: (
        dict(direct_t0_complete=True, nominal_actual_focal_signals_equal=False),
        dict(nominal_actual_mismatch_step_indices=[6])))
    monkeypatch.setattr(recovery, 'audit_execution', lambda obs, *a, **kw:
        dict(matches=obs['candidate'] != invalid_index, nominal_mismatch_steps=[6]))
    monkeypatch.setitem(sys.modules, 'cologne_action_branch_checks', SimpleNamespace(
        compare_physical=lambda a, b, tolerance: {'passed': a == b}))
    monkeypatch.setattr(recovery.measured, 'build_joint_response_interface', lambda clean, *a, **kw:
        dict(protocol='fixture', lane_nodes=[], candidate_views=clean['branches'],
             reference_index=clean['reference_index'], movement_nodes=[{'movement_id': 'm'}]))


def old_partial(directory, case):
    (directory / 'first_pp').mkdir(parents=True)
    branch = fixture_branch(3)
    summary = dict(candidate_index=3, status='HOLD_EXECUTION',
        checks=dict(full_cost_horizon=True, same_native_state=True, same_network_preaction=True,
                    nominal_actual_focal_signals_equal=False),
        prefix_safety=branch['prefix_safety'], safety=branch['safety'])
    recovery.write_json(directory / 'physical_start_server_only.json', branch['start'])
    recovery.write_json(directory / 'first_pp/observations_server_only.json', observation(3))
    recovery.write_json(directory / 'first_pp/result.json', summary)
    recovery.write_json(directory / 'result.json', dict(case_id=case['case_id'], city=case['city'], seed=case['seed'],
        status='HOLD_EXECUTION', resolved_checkpoint_sec=620, new_branches_attempted=1, branches=[summary]))
    recovery.write_json(directory / 'group_data_server_only.json', dict(case=case, case_id=case['case_id'],
        city=case['city'], seed=case['seed'], complete=False, reference_index=3,
        candidate_states=branch['candidate_states'], feature_names=['count'], movement_ids=['m'],
        X=[branch['preaction']['features']], Y=[branch['labels']],
        branches=[{k: branch[k] for k in ['candidate_index', 'preaction', 'labels', 'cost', 'costs']}]))


def case_and_static():
    return dict(case_id='held_case', city='jinan', seed=1, tls_id='tls', checkpoint_sec=600,
                checkpoint_search_sec=60), dict(lane_speed_limits={}, conflicts_by_tls={'tls': {}})


def test_reuses_old_pp_runs_only_seven_missing_actions_and_sorts_rows(monkeypatch, tmp_path):
    case, static = case_and_static(); old = tmp_path / 'old'; output = tmp_path / 'recovered'
    old_partial(old, case); shutil.copytree(old, output)
    old_observation = (old / 'first_pp/observations_server_only.json').read_text()
    calls, updates = [], []; install_mocks(monkeypatch, calls)
    result, added = recovery.recover_case(SimpleNamespace(close=lambda: None), None, case, output, static,
        [], lambda c, r: updates.append(r), 7)
    assert calls == [0, 1, 2, 4, 5, 6, 7] and added == len(updates) == 7
    assert result['status'] == 'COMPLETE' and result['new_branches_attempted'] == 8
    assert result['reused_old_attempts'] == 1 and result['recovery_new_branches_attempted'] == 7
    assert result['branches'][0]['reused_previous_status'] == 'HOLD_EXECUTION'
    assert 'nominal_actual_focal_signals_equal' not in result['branches'][0]['checks']
    assert all(all(row['checks'].values()) for row in result['branches'])
    group = read(output / 'group_data_server_only.json')
    assert group['complete'] and [b['candidate_index'] for b in group['branches']] == list(range(8))
    assert group['X'] == [[[float(i)]] for i in range(8)]
    assert group['Y'] == [[[i, i + 1]] for i in range(8)]
    assert (output / 'joint_interface_server_only.json').exists()
    assert read(old / 'result.json')['status'] == 'HOLD_EXECUTION'
    assert (old / 'first_pp/observations_server_only.json').read_text() == old_observation
    assert (output / 'first_pp/observations_server_only.json').read_text() == old_observation


def test_first_new_invalid_action_stops_without_retry_or_zero_padding(monkeypatch, tmp_path):
    case, static = case_and_static(); output = tmp_path / 'case'; old_partial(output, case)
    calls = []; install_mocks(monkeypatch, calls, invalid_index=0)
    result, added = recovery.recover_case(SimpleNamespace(close=lambda: None), None, case, output,
        static, [], lambda *a: None, 7)
    assert calls == [0] and added == 1 and result['status'] == 'INVALID'
    assert result['new_branches_attempted'] == 2
    data = read(output / 'group_data_server_only.json')
    assert not data['complete'] and len(data['X']) == len(data['Y']) == 2
    assert not (output / 'action_1').exists()
    assert not (output / 'joint_interface_server_only.json').exists()


def test_invalid_old_pp_stops_before_any_new_simulation(monkeypatch, tmp_path):
    case, static = case_and_static(); output = tmp_path / 'case'; old_partial(output, case)
    calls = []; install_mocks(monkeypatch, calls, invalid_index=3)
    result, added = recovery.recover_case(SimpleNamespace(close=lambda: None), None, case, output,
        static, [], lambda *a: None, 7)
    assert not calls and added == 0 and result['status'] == 'INVALID'
    assert not result['branches'][0]['checks']['execution_conforms_to_clearance_rule']


def test_unstarted_group_runs_one_pp_and_seven_other_actions(monkeypatch, tmp_path):
    case, static = case_and_static(); calls = []; install_mocks(monkeypatch, calls)
    result, added = recovery.recover_case(SimpleNamespace(close=lambda: None), None, case, tmp_path / 'new',
        static, [], lambda *a: None, 8)
    assert calls == [None, 0, 1, 2, 4, 5, 6, 7] and added == 8
    assert result['status'] == 'COMPLETE' and result['reused_old_attempts'] == 0


def test_completed_group_is_kept_without_reaudit_or_simulation(monkeypatch, tmp_path):
    output = tmp_path / 'complete'; output.mkdir()
    saved = dict(status='COMPLETE', branches=[{'legacy': 'preserved'}])
    recovery.write_json(output / 'result.json', saved)
    monkeypatch.setattr(recovery.measured.native, 'collect_branch', lambda *a: pytest.fail('unexpected simulation'))
    result, added = recovery.recover_case(None, None, {}, output, {}, [], lambda *a: None, 0)
    assert result == saved and added == 0 and read(output / 'result.json') == saved


def test_no_remaining_budget_prevents_a_new_action(monkeypatch, tmp_path):
    case, static = case_and_static(); output = tmp_path / 'case'; old_partial(output, case)
    calls = []; install_mocks(monkeypatch, calls)
    with pytest.raises(ValueError, match='224-branch budget'):
        recovery.recover_case(SimpleNamespace(close=lambda: None), None, case, output,
            static, [], lambda *a: None, 0)
    assert not calls
