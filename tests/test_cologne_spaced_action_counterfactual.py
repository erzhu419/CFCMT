from copy import deepcopy
from dataclasses import replace
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_no_stay_fraction_rigid import prepare_case
from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts/data'))
    return importlib.import_module('run_cologne_spaced_action_counterfactual')


@pytest.mark.parametrize('arm,accepted_times', [
    ('model_once_then_pp', [25280, 25850]), ('pp_only', [25280]),
])
def test_x_prefix_cooldown_and_branch_action_boundaries(tmp_path, replay, arm, accepted_times):
    originator, executor, contrast, _ = prepare_case(
        tmp_path, (0., -1., 2.), 2, threshold=.20790450432178334)
    selected = originator.select(contrast, reference_index=0)
    agrees = replace(selected, selected_index=0, learned_differs=False, eligible=False,
                     priority=0., predicted_score=selected.reference_score)
    states = contrast.metadata['candidate_states']
    cooldowns, accepted = {}, []
    for time_sec in range(25280, 26300, 10):
        # Intervening PP agreements still advance the real coordinator cooldown.
        native = selected if time_sec == 25280 or time_sec >= 25850 else agrees
        decision = replay.apply_branch_policy(arm, time_sec, native, 25850)
        proposal = v3.ContrastProposal(
            prior_candidate=SimpleNamespace(state=states[0]),
            selected_candidate=SimpleNamespace(state=states[decision.selected_index]),
            learned_differs=decision.learned_differs, eligible=decision.eligible,
            priority=decision.priority, rejection=decision.rejection,
            originator_diagnostics=decision.to_dict())
        audit = v3.ContrastGuardAudit()
        action = v3._coordinate_contrast_proposals(
            {'A': proposal}, cooldowns=cooldowns, cooldown_intervals=44,
            max_simultaneous_overrides=None, audit=audit,
            current_green_states={'A': executor.current_green_state})['A']
        if audit.accepted_overrides:
            accepted.append(time_sec)
        if time_sec >= 25860 or arm == 'pp_only' and time_sec >= 25850:
            assert action.state == states[0]
            assert not decision.eligible and not decision.learned_differs
    assert accepted == accepted_times


@pytest.fixture
def x_reference():
    root = Path(__file__).parents[1] / 'cf_h2o/results/cluster'
    protocol = json.loads((root / 'tsc_v154y_cologne_spaced_action_counterfactual_20260910/protocol_v1.json').read_text())
    reference = json.loads((root / 'tsc_v154x_cologne_spaced_no_stay_20260910/evaluation/seed_41242/rigid_thresholded/result.json').read_text())
    return protocol, reference


def test_all_seven_actual_windows_include_collision_and_ignore_raw_model_proposal_count(replay, x_reference):
    protocol, reference = x_reference
    assert reference['run_valid'] and reference['status'] == 'FAIL'
    assert reference['originator_diagnostics']['phase_pressure_override_count'] == 76
    trace, windows = replay.validate_reference(protocol, 41242, reference)
    assert len(trace) == 8 and len(windows) == 7
    assert [row['time_sec'] for row in windows] == [25280., 25850., 26400., 26860., 27310., 27780., 28270.]
    # Keeping seven rows while substituting the unfinished final window must fail.
    changed = deepcopy(protocol)
    last = next(row for row in changed['windows'] if row['seed'] == 41242 and row['time_sec'] == 28270.)
    last.update(time_sec=28730., window_end_sec=29180., selected_trace=trace[-1])
    with pytest.raises(ValueError, match='seven complete windows'):
        replay.validate_reference(changed, 41242, reference)


def test_full_actual_trace_keeps_prefix_and_removes_only_pp_target(replay, x_reference):
    _, reference = x_reference
    trace = reference['metrics']['accepted_intervention_trace']
    before = [row for row in trace if row['time_sec'] < 25850.]
    model = [row for row in trace if row['time_sec'] < 25860.]
    assert replay.reference_trace_exact('pp_only', before, trace, 25850.)
    assert replay.reference_trace_exact('model_once_then_pp', model, trace, 25850.)
    assert not replay.reference_trace_exact('pp_only', model, trace, 25850.)
    modified = deepcopy(model)
    modified[0]['target_originator']['predicted_score'] += .01
    assert not replay.reference_trace_exact('model_once_then_pp', modified, trace, 25850.)


def test_collision_diagnostic_does_not_discard_complete_cost_window(replay):
    checks = {'full_horizon': True, 'native_reference_trace_exact': True,
              'active_population_accounting': True}
    clean = {'raw_collision_events': 0, 'no_teleport_passed': True}
    collision = {'raw_collision_events': 13, 'no_teleport_passed': True}
    assert replay.branch_status(checks, clean, collision) == {'status': 'PASS', 'safety_status': 'FAIL'}
    assert replay.branch_status(checks, collision, clean) == {'status': 'PASS', 'safety_status': 'FAIL'}
    assert replay.branch_status({**checks, 'full_horizon': False}, clean, collision) == {
        'status': 'INVALID', 'safety_status': 'FAIL'}


@pytest.fixture
def b_reference():
    root = Path(__file__).parents[1] / 'cf_h2o/results/cluster'
    protocol = json.loads((root / 'tsc_v155c_cologne_raw_target_action_counterfactual_20260910/protocol_v1.json').read_text())
    reference = json.loads((root / 'tsc_v155b_cologne_raw_target_closed_loop_20260910/evaluation/seed_52282/rigid_thresholded/result.json').read_text())
    return protocol, reference


def test_raw_c_accepts_all_complete_b_windows(replay, b_reference):
    protocol, reference = b_reference
    trace, windows = replay.validate_reference(protocol, 52282, reference)
    assert reference['status'] == 'COMPLETE'
    assert protocol['frozen_threshold']['threshold'] == 0.
    assert protocol['score_units'] == protocol['frozen_threshold']['score_units'] == 'raw_native_halted_per_lane_cost_difference'
    assert len(trace) == 8
    assert [row['time_sec'] for row in windows] == [25270., 25760., 26220., 26780., 27230., 27730., 28190.]


@pytest.mark.parametrize('field,value', [('threshold', .20790450432178334), ('model_path', 'old_normalized_model.pkl')])
def test_raw_c_rejects_threshold_or_model_binding_from_other_fit(replay, b_reference, field, value):
    protocol, reference = deepcopy(b_reference)
    protocol['frozen_threshold'][field] = value
    with pytest.raises(ValueError, match='frozen accepted trajectory'):
        replay.validate_reference(protocol, 52282, reference)
