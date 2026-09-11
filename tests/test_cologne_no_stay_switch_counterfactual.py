from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_no_stay_fraction_rigid import prepare_case, coordinate
from cf_h2o.eval import traffic_signal_resco_cfcmt_v3 as v3


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts/data'))
    return importlib.import_module('run_cologne_no_stay_switch_counterfactual')


@pytest.mark.parametrize('arm,expected_indices', [
    ('model_once_then_pp', (1, 1, 0, 0)),
    ('pp_only', (1, 0, 0, 0)),
])
def test_native_no_stay_prefix_and_branch_boundaries(
        tmp_path, replay, arm, expected_indices):
    originator, executor, contrast, _ = prepare_case(
        tmp_path, (0., -1., 2.), 2, threshold=.20790450432178334)
    original = originator.select(contrast, reference_index=0)
    states = contrast.metadata['candidate_states']
    original_record = original.to_dict()
    for time_sec, expected_index in zip((27350., 27360., 27370., 27800.), expected_indices):
        decision = replay.apply_branch_policy(arm, time_sec, original)
        proposal = v3.ContrastProposal(
            prior_candidate=SimpleNamespace(state=states[0]),
            selected_candidate=SimpleNamespace(state=states[decision.selected_index]),
            learned_differs=decision.learned_differs, eligible=decision.eligible,
            priority=decision.priority, rejection=decision.rejection,
            originator_diagnostics=decision.to_dict())
        actual, audit = coordinate(proposal, executor)
        assert actual.state == states[expected_index]
        assert audit.accepted_switch_overrides == int(expected_index != 0)
        assert audit.accepted_stay_overrides == 0
        if expected_index == 0:
            assert decision.priority == 0. and decision.rejection is None
            assert decision.predicted_score == decision.reference_score
            assert not decision.eligible and not decision.learned_differs
        assert original.to_dict() == original_record


@pytest.fixture
def u_reference():
    path = Path(__file__).parents[1] / 'cf_h2o/results/cluster/tsc_v154u_cologne_recalibrated_no_stay_20260910/evaluation/seed_63792/rigid_thresholded/result.json'
    return json.loads(path.read_text())


def test_complete_real_u_trace_binds_selected_high_score_action(replay, u_reference):
    trace = u_reference['metrics']['accepted_intervention_trace']
    target = next(row for row in trace if row['time_sec'] == 27360.)
    assert replay.validate_reference(u_reference, target, u_reference['frozen_threshold']) == trace
    truncated = deepcopy(u_reference)
    truncated['metrics']['accepted_intervention_trace'].pop()
    with pytest.raises(ValueError, match='complete V154U trace'):
        replay.validate_reference(truncated, target, u_reference['frozen_threshold'])


@pytest.mark.parametrize('arm,cutoff', [('model_once_then_pp', 27370.), ('pp_only', 27360.)])
def test_full_prefix_trace_comparison_detects_same_count_content_change_and_cutoff(
        replay, u_reference, arm, cutoff):
    trace = u_reference['metrics']['accepted_intervention_trace']
    expected = [row for row in trace if row['time_sec'] < cutoff]
    assert replay.reference_trace_exact(arm, expected, trace)
    altered = deepcopy(expected)
    altered[0]['target_originator']['predicted_score'] += .01
    assert not replay.reference_trace_exact(arm, altered, trace)
    assert not replay.reference_trace_exact(arm, expected[:-1], trace)
    if arm == 'model_once_then_pp':
        assert expected[-1]['time_sec'] == 27360.
    else:
        extra_target = next(row for row in trace if row['time_sec'] == 27360.)
        assert not replay.reference_trace_exact(arm, expected + [extra_target], trace)
