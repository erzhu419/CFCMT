from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts/data'))
    return importlib.import_module('run_cologne_threshold_closed_loop')


@pytest.fixture
def frozen_case():
    root = Path(__file__).parents[1] / 'cf_h2o/results/cluster'
    protocol = json.loads((root / 'tsc_v155b_cologne_raw_target_closed_loop_20260910/protocol_v1.json').read_text())
    fit = json.loads((root / 'tsc_v155a_cologne_raw_target_refit_20260910/fit_v1/result.json').read_text())
    return protocol, protocol['frozen_threshold'], fit


def test_completed_a_fit_with_raw_threshold_zero_is_admitted(runner, frozen_case):
    protocol, frozen, fit = frozen_case
    runner.validate_raw_target_fit(protocol, frozen, fit)
    assert fit['status'] == 'COMPLETE'
    assert protocol['spacing_contract'] == runner.SPACING_CONTRACT


@pytest.mark.parametrize('mismatch', ['units', 'model', 'budget', 'network'])
def test_raw_fit_rejects_incompatible_units_model_budget_or_network(runner, frozen_case, mismatch):
    protocol, frozen, fit = deepcopy(frozen_case)
    if mismatch == 'units':
        fit['fit_diagnostics']['full']['score_units'] = 'group_normalized_advantage'
    elif mismatch == 'model':
        fit['model_path'] = fit['old_model_path']
    elif mismatch == 'budget':
        fit['fit_diagnostics']['full']['rows'] = 792
    else:
        fit['sumocfg'] = 'different_network.sumocfg'
    with pytest.raises(ValueError, match='raw-target B100 fit'):
        runner.validate_raw_target_fit(protocol, frozen, fit)


def test_b_completion_separates_incidents_without_changing_earlier_protocols(runner):
    assert runner.completion_status(runner.RAW_TARGET_PROTOCOL, True, False) == 'COMPLETE'
    assert runner.completion_status(runner.RAW_TARGET_PROTOCOL, True, True) == 'COMPLETE'
    assert runner.completion_status(runner.RAW_TARGET_PROTOCOL, False, False) == 'INVALID'
    assert runner.completion_status(runner.SPACED_PROTOCOL, True, False) == 'FAIL'
    assert runner.completion_status(runner.SPACED_PROTOCOL, True, True) == 'PASS'


def test_b_cli_completion_uses_valid_run_while_old_collision_exit_is_preserved(runner, tmp_path, monkeypatch):
    protocol_path = tmp_path / 'protocol.json'
    protocol_path.write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['runner', '--protocol', str(protocol_path),
                                   '--seed', '52282', '--output', str(tmp_path / 'output')])
    result = {'protocol': runner.RAW_TARGET_PROTOCOL, 'run_valid': True,
              'status': 'COMPLETE', 'zero_incident_status': 'FAIL'}
    monkeypatch.setattr(runner, 'run', lambda *_: result)
    runner.main()
    result.update(run_valid=False, status='INVALID')
    with pytest.raises(SystemExit) as invalid:
        runner.main()
    assert invalid.value.code == 2
    result.update(protocol=runner.SPACED_PROTOCOL, run_valid=True, status='FAIL')
    with pytest.raises(SystemExit) as previous:
        runner.main()
    assert previous.value.code == 2
