from copy import deepcopy
import importlib
from pathlib import Path

import pytest

from test_cologne_action_branch_checks import snapshot


@pytest.fixture
def replay(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts/data'))
    return importlib.import_module('run_cologne_no_stay_continuation_replay')


def native_branch():
    physical = snapshot()
    physical['time_sec'] = 27360.
    return {
        'capture': {'physical': physical, 'pending_ids': ['pending_a', 'pending_b'],
                    'decision': {'time_sec': 27360., 'requested_phase': 'rrG', 'forced_pp': False}},
        'decisions': [{'time_sec': 27350., 'originator_decision': {'predicted_score': -.4}},
                      {'time_sec': 27360., 'originator_decision': {'predicted_score': -.8}}],
        'samples': [{'time_sec': 27361.+i, 'cost': i/8} for i in range(10)],
    }


def test_reused_prefix_and_first_action_match_before_continuation_diverges(replay):
    old = native_branch()
    new = deepcopy(old)
    new['decisions'].append({'time_sec': 27370., 'requested_phase': 'Grr'})
    new['samples'].append({'time_sec': 27371., 'cost': 100.})
    checks, comparisons = replay.compare_reused_prefix(new, old)
    assert all(checks.values())
    assert comparisons['physical_start']['max_position_or_speed_difference'] == 0.
    assert comparisons['first_interval']['left_count'] == 10


def test_reuse_matching_rejects_changed_native_state_decision_or_first_action(replay):
    old = native_branch()
    new = deepcopy(old)
    new['capture']['physical']['executors']['tls']['remaining_sec'] += 1e-12
    new['capture']['pending_ids'][0] = 'different_pending_vehicle'
    new['decisions'][0]['originator_decision']['predicted_score'] += .01
    new['capture']['decision']['requested_phase'] = 'Grr'
    new['samples'][9]['cost'] += .125
    checks, _ = replay.compare_reused_prefix(new, old)
    assert checks['target_captured']
    for name in ('physical_start_exact', 'pending_start_exact', 'prefix_decisions_exact',
                 'target_decision_exact', 'first_ten_costs_exact'):
        assert not checks[name]
    shortened = deepcopy(old)
    shortened['samples'] = shortened['samples'][:9]
    assert not replay.compare_reused_prefix(shortened, old)[0]['first_ten_costs_exact']


def test_repeated_trace_requires_all_u_interventions_until_end_exclusive(replay):
    trace = [{'time_sec': t, 'target_originator': {'predicted_score': -.8}}
             for t in (27350., 27360., 27370., 27800., 27810.)]
    assert replay.repeated_trace_exact(trace[:-1], trace)
    assert not replay.repeated_trace_exact(trace[:2], trace)
    assert not replay.repeated_trace_exact(trace, trace)
    changed = deepcopy(trace[:-1])
    changed[2]['target_originator']['predicted_score'] += .01
    assert not replay.repeated_trace_exact(changed, trace)


def test_comparison_separates_first_action_from_continuation(replay):
    runs = {name: {'mean_halted_per_lane': value} for name, value in zip(replay.ARMS, (1., 2., 3.))}
    comparison = replay.comparison_metrics(runs)
    assert comparison['one_action_ranking'] == 'beneficial'
    assert comparison['one_action_minus_pp'] == -1.
    assert comparison['repeated_minus_one_action'] == 2.
    assert comparison['repeated_minus_pp'] == 1.
    assert comparison['repeated_relative_to_one_action_percent'] == 200.
